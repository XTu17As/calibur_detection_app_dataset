import os
import json
import random
import math
import shutil
import time
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.ops as tvops

# COCO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# KAGGLE/NOTEBOOK
from IPython.display import FileLink, display

# CONFIG 
# KAGGLE PATHS
KAGGLE_INPUT_DIR = "/kaggle/input/skripsi-splitthenaug-384-threads-frontal-v5-merged/Skripsi_SplitThenAug_384_Threads_frontal_v5_merged"
ANNOTATION_JSON = os.path.join(KAGGLE_INPUT_DIR, "annotations_all.json")
IMG_DIR = os.path.join(KAGGLE_INPUT_DIR, "images")
OUTPUT_DIR = "/kaggle/working/outputs_tinyvit5m_fcos4_fixed_trio_100_8_frontal"

BATCH_SIZE = 8
NUM_WORKERS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 100
IMAGE_SIZE = 384
VAL_IMAGE_SIZE = 384
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PRINT_INTERVAL = 20
TRAIN_VAL_SPLIT = 0.8
SEED = 69
MIN_SCORE_THRES = 0.1

STRIDES = [8, 16, 32, 64, 128]
scale_ranges_default = [(0, 32), (32, 64), (64, 128), (128, 256), (256, 1e8)]
NMS_IOU_THRESH = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Dataset (COCO style) 
class CocoDataset(Dataset):
    def __init__(self, ann_file, img_dir, img_ids, img_size=IMAGE_SIZE, is_train=True):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.ids = img_ids
        self.img_size = img_size
        self.is_train = is_train

        cat_ids = sorted(self.coco.getCatIds())
        self.catid2idx = {cid: i for i, cid in enumerate(cat_ids)}
        self.num_classes = len(cat_ids)

        if is_train:
            self.transforms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ColorJitter(0.1, 0.1, 0.05, 0.02),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img_info = self.coco.loadImgs(self.ids[idx])[0]
        path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(path).convert('RGB')
        w0,h0 = img.size
        img_t = self.transforms(img)
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels = [], []
        for a in anns:
            x,y,w,h = a['bbox']
            sx,sy = self.img_size/w0, self.img_size/h0
            x1,y1 = max(0,x*sx), max(0,y*sy)
            x2,y2 = min(self.img_size-1,(x+w)*sx), min(self.img_size-1,(y+h)*sy)
            if x2<=x1 or y2<=y1: continue
            boxes.append([x1,y1,x2,y2])
            labels.append(self.catid2idx[a['category_id']])
        if len(boxes)==0:
            boxes = torch.zeros((0,4),dtype=torch.float32)
            labels = torch.zeros((0,),dtype=torch.long)
        else:
            boxes = torch.tensor(boxes,dtype=torch.float32)
            labels = torch.tensor(labels,dtype=torch.long)
        return img_t, {'boxes':boxes,'labels':labels,'image_id':torch.tensor([img_info['id']])}

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return imgs, targets

# TinyViT-5M Backbone (Official Spec) 
class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, rd_ratio=0.25):
        super().__init__()
        rd_channels = int(in_channels * rd_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, rd_channels, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, in_channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_channels = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.GELU())
        
        layers.append(nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1, groups=hidden_channels, bias=False))
        layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.GELU())
        
        layers.append(SqueezeExcite(hidden_channels))
        
        layers.append(nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep = torch.rand(shape, device=x.device) >= self.drop_prob
        return x / (1.0 - self.drop_prob) * keep

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size*window_size, C)
    return x

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / (window_size * window_size)))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
    return x

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=64, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.BatchNorm2d(embed_dim)
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = self.norm(x)
        x = x.flatten(2).transpose(1,2)
        return x, Hp, Wp

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=True)
        self.proj = nn.Linear(dim, dim)
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        coords = coords.flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1,2,0).contiguous()
        self.register_buffer('relative_index', ((relative_coords[:,:,0] + window_size - 1) * (2*window_size-1) + (relative_coords[:,:,1] + window_size - 1)).flatten())
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2*window_size-1)*(2*window_size-1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        ws = self.window_size
        x4 = x.view(B, H, W, C)
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x4 = F.pad(x4, (0,0, 0, pad_w, 0, pad_h))
        Hp, Wp = x4.shape[1], x4.shape[2]
        x_windows = window_partition(x4, ws)
        qkv = self.qkv(x_windows).reshape(x_windows.size(0), x_windows.size(1), 3, self.num_heads, C//self.num_heads)
        q, k, v = qkv.unbind(2)
        q = q.permute(0,2,1,3); k = k.permute(0,2,1,3); v = v.permute(0,2,1,3)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        bias = self.relative_position_bias_table[self.relative_index].view(ws*ws, ws*ws, -1).permute(2,0,1)
        attn = attn + bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1,2).reshape(x_windows.size(0), x_windows.size(1), C)
        out = self.proj(out)
        x = window_reverse(out, ws, Hp, Wp)
        x = x[:, :H, :W, :].reshape(B, H*W, C)
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.H, self.W = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(), nn.Linear(int(dim*mlp_ratio), dim))
        
    def forward(self, x, H, W):
        B, N, C = x.shape 
        shortcut = x
        x = self.norm1(x)
        
        x4 = x.view(B, H, W, C)
        if self.shift_size > 0:
            x4 = torch.roll(x4, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
        
        x = x4.view(B, H*W, C)
        x = self.attn(x, H, W)
        x = x.view(B, H, W, C)
        
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1,2))
        x = x.view(B, H*W, C)
        
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim * 4)
        self.reduction = nn.Linear(in_dim * 4, out_dim)
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        x4 = x.view(B, H, W, C)
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x4 = F.pad(x4, (0,0,0, pad_w, 0, pad_h))
        Hp, Wp = x4.shape[1], x4.shape[2]
        x0 = x4[:, 0::2, 0::2, :]
        x1 = x4[:, 1::2, 0::2, :]
        x2 = x4[:, 0::2, 1::2, :]
        x3 = x4[:, 1::2, 1::2, :]
        x_cat = torch.cat([x0, x1, x2, x3], dim=-1).view(B, -1, 4*C)
        x_cat = self.norm(x_cat)
        x_out = self.reduction(x_cat)
        return x_out, Hp//2, Wp//2

class TinyViT5MBackbone(nn.Module):
    def __init__(self,
                 embed_dims=[64, 128, 160, 320], 
                 depths=[2, 2, 6, 2],         
                 num_heads=[2, 4, 5, 10],      
                 window_size=7,
                 drop_path_rate=0.1,
                 expand_ratio=4,
                 input_size=IMAGE_SIZE):
        super().__init__()
        assert len(embed_dims) == 4 and len(depths) == 4 and len(num_heads) == 4
        
        self.patch_embed = PatchEmbed(in_ch=3, embed_dim=embed_dims[0], patch_size=4)
        
        patch_res_h = input_size // 4
        patch_res_w = input_size // 4
        
        self.stage_resolutions = []
        self.stage_resolutions.append((patch_res_h, patch_res_w))
        
        patch_res_h, patch_res_w = patch_res_h // 2, patch_res_w // 2
        self.stage_resolutions.append((patch_res_h, patch_res_w))
        
        patch_res_h, patch_res_w = patch_res_h // 2, patch_res_w // 2
        self.stage_resolutions.append((patch_res_h, patch_res_w))
        
        patch_res_h, patch_res_w = patch_res_h // 2, patch_res_w // 2
        self.stage_resolutions.append((patch_res_h, patch_res_w))
        
        self.stages = nn.ModuleList()
        self.mergers = nn.ModuleList()
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        cur = 0

        stage1_blocks = []
        for j in range(depths[0]):
            in_c = embed_dims[0]
            out_c = embed_dims[0]
            s = 1 
            stage1_blocks.append(MBConv(in_c, out_c, stride=s, expand_ratio=expand_ratio))
        self.stages.append(nn.ModuleList(stage1_blocks))
        cur += depths[0]
        self.mergers.append(PatchMerging(embed_dims[0], embed_dims[1]))

        for i in range(1, 4):
            blocks = []
            for j in range(depths[i]):
                shift = (j % 2) * (window_size // 2)
                
                blocks.append(SwinBlock(
                    embed_dims[i], 
                    input_resolution=self.stage_resolutions[i],
                    num_heads=num_heads[i],
                    window_size=window_size, 
                    shift_size=shift, 
                    drop_path=dp_rates[cur + j]
                ))
                
            self.stages.append(nn.ModuleList(blocks))
            cur += depths[i]
            if i < 3:
                self.mergers.append(PatchMerging(embed_dims[i], embed_dims[i+1]))
                
    def forward(self, x):
        B = x.shape[0] 
        x, H, W = self.patch_embed(x) 
        
        C = x.shape[2] 
        x_conv = x.transpose(1, 2).contiguous().view(B, C, H, W)
        for blk in self.stages[0]:
            x_conv = blk(x_conv) 
        
        x, H, W = self.mergers[0](x_conv.flatten(2).transpose(1,2), H, W) 
        
        outs = []
        for i, stage in enumerate(self.stages[1:], start=1):
            for blk in stage:
                x = blk(x, H, W)
                
            C = x.shape[2] 
            fmap = x.transpose(1,2).contiguous().view(B, C, H, W)
            outs.append(fmap)

            if i < len(self.mergers): 
                x, H, W = self.mergers[i](x, H, W)

        c3, c4, c5 = outs[0], outs[1], outs[2]
        return [c3, c4, c5]

# FPN + FCOS Head 
class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.output_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3,1,1) for _ in in_channels])
        self.extra_p6 = nn.Conv2d(out_channels, out_channels, 3,2,1)
        self.extra_p7 = nn.Conv2d(out_channels, out_channels, 3,2,1)
        
    def forward(self, inputs):
        c3, c4, c5 = inputs
        p5 = self.lateral_convs[2](c5)
        p4 = self.lateral_convs[1](c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lateral_convs[0](c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p3 = self.output_convs[0](p3)
        p4 = self.output_convs[1](p4)
        p5 = self.output_convs[2](p5)
        p6 = self.extra_p6(p5)
        p7 = self.extra_p7(p6)
        return [p3, p4, p5, p6, p7]

class FCOSHead(nn.Module):
    def __init__(self, num_classes, in_channels=128):
        super().__init__()
        cls_layers = []
        reg_layers = []
        
        num_convs = 4 
        for _ in range(num_convs):
            cls_layers += [nn.Conv2d(in_channels, in_channels, 3,1,1, bias=True), nn.GroupNorm(32,in_channels), nn.ReLU()]
            reg_layers += [nn.Conv2d(in_channels, in_channels, 3,1,1, bias=True), nn.GroupNorm(32,in_channels), nn.ReLU()]
        self.cls_tower = nn.Sequential(*cls_layers)
        self.reg_tower = nn.Sequential(*reg_layers)
        
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3,1,1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3,1,1)
        self.centerness = nn.Conv2d(in_channels, 1, 3,1,1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        pi = 0.01
        bias_value = -math.log((1 - pi) / pi)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        
    def forward(self, features):
        cls_outs, reg_outs, cent_outs = [], [], []
        for f in features:
            cls_t = self.cls_tower(f)
            reg_t = self.reg_tower(f)
            cls_outs.append(self.cls_logits(cls_t))
            reg_outs.append(F.relu(self.bbox_pred(reg_t)))
            cent_outs.append(self.centerness(reg_t))
        return cls_outs, reg_outs, cent_outs

class TinyViT_FPN_FCOS(nn.Module):
    def __init__(self, num_classes, input_size=IMAGE_SIZE):
        super().__init__()
        self.backbone = TinyViT5MBackbone(
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_size=7,
            drop_path_rate=0.1,
            input_size=input_size
        )
        self.fpn = FPN(in_channels=[128, 160, 320], out_channels=128)
        self.head = FCOSHead(num_classes, in_channels=128)
        
    def forward(self, x):
        feats = self.backbone(x)
        fpn_feats = self.fpn(feats)
        cls, reg, cen = self.head(fpn_feats)
        return cls, reg, cen, fpn_feats

# Helpers / losses / IoU 
def compute_locations(feature, stride, device=None):
    if isinstance(feature, torch.Tensor):
        _,_,h,w = feature.shape
    else:
        _,_,h,w = feature
    device = device if device is not None else feature.device if isinstance(feature, torch.Tensor) else torch.device(DEVICE)
    shifts_x = (torch.arange(0, w, device=device) + 0.5) * stride
    shifts_y = (torch.arange(0, h, device=device) + 0.5) * stride
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
    shift_x = shift_x.reshape(-1); shift_y = shift_y.reshape(-1)
    return torch.stack((shift_x, shift_y), dim=1)

def decode_boxes(locations, ltrb):
    x = locations[:,0]; y = locations[:,1]
    l,t,r,b = ltrb[:,0], ltrb[:,1], ltrb[:,2], ltrb[:,3]
    return torch.stack([x - l, y - t, x + r, y + b], dim=1)

class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.35, reduction='sum'):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha; self.reduction = reduction
        
    def forward(self, logits, targets_onehot):
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets_onehot, reduction='none')
        p_t = prob * targets_onehot + (1 - prob) * (1 - targets_onehot)
        alpha_t = self.alpha * targets_onehot + (1 - self.alpha) * (1 - targets_onehot)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * ce
        return loss.sum() if self.reduction=='sum' else loss.mean()

def box_iou_tensor(boxes1, boxes2):
    if boxes1.numel()==0 or boxes2.numel()==0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    area1 = (boxes1[:,2]-boxes1[:,0]).clamp(0) * (boxes1[:,3]-boxes1[:,1]).clamp(0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clamp(0) * (boxes2[:,3]-boxes2[:,1]).clamp(0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(0)
    inter = wh[:,:,0] * wh[:,:,1]
    union = area1[:,None] + area2[None,:] - inter
    return inter / (union + 1e-6)

def mean_iou_for_image(pred_boxes, gt_boxes):
    if gt_boxes.shape[0]==0 or pred_boxes.shape[0]==0: return 0.0
    pred_boxes_t = torch.tensor(pred_boxes) if not isinstance(pred_boxes, torch.Tensor) else pred_boxes
    gt_boxes_t = torch.tensor(gt_boxes) if not isinstance(gt_boxes, torch.Tensor) else gt_boxes
    ious = box_iou_tensor(pred_boxes_t, gt_boxes_t)
    best_per_gt, _ = ious.max(0)
    return float(best_per_gt.mean().item())

# Data loaders (STRATIFIED SPLIT) 
def prepare_loaders(ann_file, img_dir, split=TRAIN_VAL_SPLIT):
    coco = COCO(ann_file)
    
    cat_ids = sorted(coco.getCatIds())
    cat_to_imgs = defaultdict(list)
    for cat_id in cat_ids:
        img_ids_for_cat = coco.getImgIds(catIds=[cat_id])
        img_ids_for_cat = sorted(list(set(img_ids_for_cat)))
        random.shuffle(img_ids_for_cat)
        cat_to_imgs[cat_id] = img_ids_for_cat

    all_img_ids = set(coco.imgs.keys())
    train_ids_set = set()
    val_ids_set = set()
    
    for cat_id in cat_ids:
        imgs_for_this_cat = cat_to_imgs[cat_id]
        
        unassigned_imgs = []
        for img_id in imgs_for_this_cat:
            if img_id not in train_ids_set and img_id not in val_ids_set:
                unassigned_imgs.append(img_id)
        
        n_train_for_cat = int(len(unassigned_imgs) * split)
        
        new_train_ids = unassigned_imgs[:n_train_for_cat]
        new_val_ids = unassigned_imgs[n_train_for_cat:]
        
        train_ids_set.update(new_train_ids)
        val_ids_set.update(new_val_ids)

    assigned_ids = train_ids_set.union(val_ids_set)
    unassigned_leftovers = list(all_img_ids - assigned_ids)
    random.shuffle(unassigned_leftovers)
    
    n_train_leftover = int(len(unassigned_leftovers) * split)
    train_ids_set.update(unassigned_leftovers[:n_train_leftover])
    val_ids_set.update(unassigned_leftovers[n_train_leftover:])
    
    train_ids = sorted(list(train_ids_set))
    val_ids = sorted(list(val_ids_set))

    print(f"Total images: {len(all_img_ids)}. Using {len(train_ids)} for training and {len(val_ids)} for validation (stratified {split*100:.0f}/{(1-split)*100:.0f} split).")

    train_ds = CocoDataset(ann_file, img_dir, train_ids, is_train=True, img_size=IMAGE_SIZE)
    val_ds = CocoDataset(ann_file, img_dir, val_ids, is_train=False, img_size=VAL_IMAGE_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    
    cat_ids = sorted(coco.getCatIds())
    idx2cat = {i: cid for i, cid in enumerate(cat_ids)}
    return train_loader, val_loader, coco, idx2cat

# Evaluation (Using Hard NMS)
def evaluate(model, val_loader, coco_gt, device, idx2cat):
    model.eval()
    results = []
    img_ids_seen = []
    mean_ious = []
    with torch.no_grad():
        for imgs, targets in tqdm(val_loader, desc='Val'):
            imgs = imgs.to(device)
            if imgs.shape[2] != VAL_IMAGE_SIZE or imgs.shape[3] != VAL_IMAGE_SIZE:
                imgs = F.interpolate(imgs, size=(VAL_IMAGE_SIZE, VAL_IMAGE_SIZE), mode='bilinear', align_corners=False)

            cls_outs, reg_outs, cen_outs, feats = model(imgs)
            locations = [compute_locations(f, s, device=device).to(device) for f, s in zip(feats, STRIDES)]
            for i in range(imgs.shape[0]):
                all_boxes, all_scores, all_labels = [], [], []
                for lvl, (cls_map, reg_map, cen_map, loc) in enumerate(zip(cls_outs, reg_outs, cen_outs, locations)):
                    cls_per = cls_map[i].permute(1,2,0).reshape(-1, cls_map.shape[1]).to(device)
                    reg_per = reg_map[i].permute(1,2,0).reshape(-1,4).to(device)
                    cen_per = torch.sigmoid(cen_map[i].permute(1,2,0).reshape(-1)).to(device)
                    scores_per = torch.sigmoid(cls_per)
                    max_scores, labels = torch.max(scores_per, 1)
                    final_scores_t = max_scores * cen_per

                    keep_mask_t = final_scores_t > MIN_SCORE_THRES 

                    if keep_mask_t.sum().item() == 0:
                        continue
                    loc_sel = loc[keep_mask_t]
                    reg_sel = reg_per[keep_mask_t]
                    labels_sel = labels[keep_mask_t].cpu().numpy()
                    scores_sel = final_scores_t[keep_mask_t].cpu().numpy()
                    boxes_sel = decode_boxes(loc_sel, reg_sel * STRIDES[lvl]).cpu().numpy()
                    all_boxes.append(boxes_sel); all_scores.append(scores_sel); all_labels.append(labels_sel)

                if len(all_boxes)==0:
                    pred_boxes = np.zeros((0,4)); pred_scores = np.zeros((0,)); pred_labels = np.zeros((0,), dtype=int)
                else:
                    pred_boxes = np.vstack(all_boxes)
                    pred_scores = np.concatenate(all_scores)
                    pred_labels = np.concatenate(all_labels)
                    if pred_boxes.shape[0] > 0:
                        tb = torch.tensor(pred_boxes, device=device, dtype=torch.float32)
                        ts = torch.tensor(pred_scores, device=device, dtype=torch.float32)
                        keep_idx = tvops.nms(tb, ts, iou_threshold=NMS_IOU_THRESH).cpu().numpy()
                        pred_boxes = pred_boxes[keep_idx]
                        pred_scores = pred_scores[keep_idx]
                        pred_labels = pred_labels[keep_idx]

                img_id = int(targets[i]['image_id'].item())
                img_info = coco_gt.loadImgs(img_id)[0]
                sx, sy = img_info['width'] / VAL_IMAGE_SIZE, img_info['height'] / VAL_IMAGE_SIZE
                coco_formatted = []
                for j in range(len(pred_scores)):
                    x1,y1,x2,y2 = pred_boxes[j]
                    w = (x2 - x1) * sx
                    h = (y2 - y1) * sy
                    if w <= 0 or h <= 0: continue
                    original_cat_id = int(idx2cat[int(pred_labels[j])])
                    coco_formatted.append({
                        "image_id": img_id,
                        "category_id": original_cat_id,
                        "bbox": [float(x1 * sx), float(y1 * sy), float(w), float(h)],
                        "score": float(pred_scores[j])
                    })
                results.extend(coco_formatted)
                img_ids_seen.append(img_id)
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                if gt_boxes.shape[0] > 0 and pred_boxes.shape[0] > 0:
                    mean_ious.append(mean_iou_for_image(pred_boxes, gt_boxes))
                else:
                    mean_ious.append(0.0)
    resfile = os.path.join(OUTPUT_DIR, 'detections_val.json')
    with open(resfile, 'w') as f:
        json.dump(results, f)
    if len(results) == 0:
        print("Validation: No detections found.")
        return None, None, float(np.mean(mean_ious) if len(mean_ious)>0 else 0.0)

    coco_dt = coco_gt.loadRes(resfile)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = img_ids_seen
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()

    # Extract recall metrics (mAR)
    recall_metrics = {
        'mAR_1': float(coco_eval.stats[6]),
        'mAR_10': float(coco_eval.stats[7]),
        'mAR_100': float(coco_eval.stats[8]),
        'mAR_small': float(coco_eval.stats[9]),
        'mAR_medium': float(coco_eval.stats[10]),
        'mAR_large': float(coco_eval.stats[11])
    }

    print(f"[DEBUG] mAR values -> AR@1={recall_metrics['mAR_1']:.4f}, AR@10={recall_metrics['mAR_10']:.4f}, "
          f"AR@100={recall_metrics['mAR_100']:.4f}, AR_small={recall_metrics['mAR_small']:.4f}, "
          f"AR_medium={recall_metrics['mAR_medium']:.4f}, AR_large={recall_metrics['mAR_large']:.4f}")

    return coco_eval.stats, recall_metrics, float(np.mean(mean_ious) if len(mean_ious)>0 else 0.0)

# TRAIN 
def train():
    device = torch.device(DEVICE)
    coco_temp = COCO(ANNOTATION_JSON)
    num_classes = len(coco_temp.getCatIds())
    del coco_temp
    
    model = TinyViT_FPN_FCOS(num_classes, input_size=IMAGE_SIZE).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")
    print("WARNING: Model is initialized randomly (no pretraining).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))
    
    train_loader, val_loader, coco, idx2cat = prepare_loaders(ANNOTATION_JSON, IMG_DIR)
    scale_ranges = scale_ranges_default
    best_map = 0.0
    focal_loss = SigmoidFocalLoss().to(device)
    
    log_data = []

    for epoch in range(1, EPOCHS+1):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{EPOCHS}")
        for i, (imgs, targets) in pbar:
            imgs = imgs.to(device)
            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                t['labels'] = t['labels'].to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                cls_outs, reg_outs, cen_outs, feats = model(imgs)
                total_cls_loss = torch.tensor(0.0, device=device)
                total_reg_loss = torch.tensor(0.0, device=device)
                total_cent_loss = torch.tensor(0.0, device=device)
                num_positives = 0
                locs_per_level = [compute_locations(f, s, device=device).to(device) for f, s in zip(feats, STRIDES)]
                
                for b in range(imgs.shape[0]):
                    gt_boxes = targets[b]['boxes']
                    gt_labels = targets[b]['labels']
                    if gt_boxes.numel() == 0:
                        for lvl in range(len(cls_outs)):
                            cls_lvl = cls_outs[lvl][b].permute(1,2,0).reshape(-1, cls_outs[lvl].shape[1]).to(device)
                            total_cls_loss = total_cls_loss + focal_loss(cls_lvl, torch.zeros_like(cls_lvl))
                        continue
                    for lvl in range(len(cls_outs)):
                        low, high = scale_ranges[lvl] if lvl < len(scale_ranges) else (0, 1e8)
                        cls_lvl = cls_outs[lvl][b].permute(1,2,0).reshape(-1, cls_outs[lvl].shape[1]).to(device)
                        reg_lvl = reg_outs[lvl][b].permute(1,2,0).reshape(-1,4).to(device)
                        cen_lvl = cen_outs[lvl][b].permute(1,2,0).reshape(-1).to(device)
                        locs = locs_per_level[lvl]
                        L = locs.shape[0]
                        C = cls_lvl.shape[1]
                        targets_onehot = torch.zeros((L, C), device=device, dtype=cls_lvl.dtype)
                        x1 = gt_boxes[:, 0].unsqueeze(0); y1 = gt_boxes[:, 1].unsqueeze(0)
                        x2 = gt_boxes[:, 2].unsqueeze(0); y2 = gt_boxes[:, 3].unsqueeze(0)
                        xs = locs[:, 0].unsqueeze(1); ys = locs[:, 1].unsqueeze(1)
                        inside = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
                        if inside.numel() == 0:
                            total_cls_loss = total_cls_loss + focal_loss(cls_lvl, targets_onehot)
                            continue
                        areas = ((x2 - x1) * (y2 - y1)).clamp(min=0)
                        areas_rep = areas.repeat(L, 1)
                        areas_rep[~inside] = float('inf')
                        best_gt_idx = torch.argmin(areas_rep, dim=1)
                        min_assigned_area = areas_rep.gather(1, best_gt_idx.unsqueeze(1)).squeeze(1)
                        has_gt = min_assigned_area < float('inf')
                        if has_gt.any():
                            pos_locs = torch.nonzero(has_gt).squeeze(1)
                            assigned_gt_idx = best_gt_idx[pos_locs]
                            assigned_gt_boxes = gt_boxes[assigned_gt_idx]
                            assigned_labels = gt_labels[assigned_gt_idx].to(device)
                            box_w = (assigned_gt_boxes[:,2] - assigned_gt_boxes[:,0]).clamp(min=1e-6)
                            box_h = (assigned_gt_boxes[:,3] - assigned_gt_boxes[:,1]).clamp(min=1e-6)
                            box_size = torch.sqrt(box_w * box_h)
                            lvl_mask = (box_size >= low) & (box_size < high)
                            if lvl_mask.any():
                                valid_pos_loc_idx = pos_locs[lvl_mask]
                                valid_assigned_idx = assigned_gt_idx[lvl_mask]
                                valid_assigned_labels = assigned_labels[lvl_mask].long()
                                targets_onehot[valid_pos_loc_idx, valid_assigned_labels] = 1.0
                                num_pos_here = int(valid_pos_loc_idx.numel())
                                num_positives += num_pos_here
                                locs_pos = locs[valid_pos_loc_idx]
                                reg_preds_pos = reg_lvl[valid_pos_loc_idx]
                                pred_boxes_pos = decode_boxes(locs_pos, reg_preds_pos * STRIDES[lvl])
                                chosen_gt_valid = gt_boxes[valid_assigned_idx].to(device)
                                iou_loss = tvops.generalized_box_iou_loss(pred_boxes_pos, chosen_gt_valid, reduction='sum')
                                total_reg_loss = total_reg_loss + iou_loss
                                l = locs_pos[:,0] - chosen_gt_valid[:,0]
                                t_ = locs_pos[:,1] - chosen_gt_valid[:,1]
                                r = chosen_gt_valid[:,2] - locs_pos[:,0]
                                b_ = chosen_gt_valid[:,3] - locs_pos[:,1]
                                reg_target_for_cent = torch.stack([l, t_, r, b_], dim=1)
                                l_e, t_e, r_e, b_e = reg_target_for_cent[:,0], reg_target_for_cent[:,1], reg_target_for_cent[:,2], reg_target_for_cent[:,3]
                                cent_targets = torch.sqrt(
                                    (torch.min(l_e, r_e) / torch.max(l_e, r_e).clamp(min=1e-6)) *
                                    (torch.min(t_e, b_e) / torch.max(t_e, b_e).clamp(min=1.e-6))
                                ).clamp(0.0, 1.0)
                                total_cent_loss = total_cent_loss + F.binary_cross_entropy_with_logits(cen_lvl[valid_pos_loc_idx], cent_targets, reduction='sum')
                        total_cls_loss = total_cls_loss + focal_loss(cls_lvl, targets_onehot)
                num_positives = max(1, num_positives)
                
                loss = (1.0 * total_cls_loss + 1.0 * total_reg_loss + 1.0 * total_cent_loss) / num_positives
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().item())
            if (i+1) % PRINT_INTERVAL == 0 or (i+1) == len(train_loader):
                pbar.set_postfix({'avg_loss': f"{running_loss/(i+1):.4f}"} )
        
        current_avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} finished, avg_loss={current_avg_loss:.4f}")
        stats, recall_metrics, mean_iou = evaluate(model, val_loader, coco, device, idx2cat)
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch} took {epoch_duration:.2f} seconds.")
        
        epoch_log = {
            'epoch': epoch,
            'avg_train_loss': current_avg_loss,
            'mAP_0.5:0.95': None,
            'mAP_0.5': None,
            'mAR_100': None,
            'mAR_1': None,
            'mAR_10': None,
            'mAR_small': None,
            'mAR_medium': None,
            'mAR_large': None,
            'mean_iou': mean_iou,
            'epoch_duration_seconds': epoch_duration
        }

        if stats is not None:
            map05095 = stats[0]
            map05 = stats[1] 
            print(f"Validation mAP 0.5:0.95 = {map05095:.4f}, mAP 0.5 = {map05:.4f}, mean IoU = {mean_iou:.4f}")
            
            epoch_log['mAP_0.5:0.95'] = map05095
            epoch_log['mAP_0.5'] = map05
            
            # Update epoch_log with recall metrics
            epoch_log.update(recall_metrics)

            if map05095 > best_map:
                torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()},
                           os.path.join(OUTPUT_DIR, 'best_model.pth'))
                best_map = map05095; print("Saved best model.")
        else:
            print(f"Validation mean IoU = {mean_iou:.4f} (no COCO stats)")
        
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()},
                   os.path.join(OUTPUT_DIR, f'ckpt_epoch_{epoch}.pth'))
        
        log_data.append(epoch_log)

    return log_data


if __name__ == "__main__":
    script_start_time = time.time()
    log_data = []
    
    print("Device:", DEVICE)
    print(f"Starting training for {OUTPUT_DIR}")
    print(f"Reading data from {KAGGLE_INPUT_DIR}")
    print("Model will be initialized randomly (no pretraining).")
    
    try:
        log_data = train()
        print("Training finished successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        script_end_time = time.time()
        total_script_duration = script_end_time - script_start_time
        hours = int(total_script_duration // 3600)
        minutes = int((total_script_duration % 3600) // 60)
        seconds = int(total_script_duration % 60)
        print(f"Total script execution time: {hours}h {minutes}m {seconds}s ({total_script_duration:.2f} seconds)")
        
        log_file_path = os.path.join(OUTPUT_DIR, 'training_log.json')
        try:
            final_log_data = {
                'epochs': log_data,
                'total_script_duration_seconds': total_script_duration
            }
            with open(log_file_path, 'w') as f:
                json.dump(final_log_data, f, indent=4)
            print(f"Final log with total time saved to {log_file_path}")
        except Exception as e:
            print(f"Warning: Could not save final log file: {e}")
        
        print(f"Attempting to zip output directory: {OUTPUT_DIR}")
        if not os.path.exists(OUTPUT_DIR):
            print(f"Output directory {OUTPUT_DIR} does not exist. Nothing to zip.")
        else:
            try:
                zip_output_filename = "/kaggle/working/training_outputs"
                root_dir = os.path.dirname(OUTPUT_DIR)
                base_dir = os.path.basename(OUTPUT_DIR)
                shutil.make_archive(base_name=zip_output_filename, format='zip', root_dir=root_dir, base_dir=base_dir)
                zip_file_path = zip_output_filename + ".zip"
                if os.path.exists(zip_file_path):
                    print(f"Output successfully zipped to {zip_file_path}")
                    print("Generating download link...")
                    display(FileLink(zip_file_path))
                    print("--- Please click the link above to download your results. ---")
                else:
                    print(f"Failed to create zip file at {zip_file_path}.")
            except Exception as e:
                print(f"An error occurred during zipping: {e}")
