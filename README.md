<div align="center">

  <h1> CALIBUR : DATA SIDE </h1>
  
  <p>
    <strong>The Neural Backbone & Training Grounds for the Calibur Detection System</strong>
  </p>

  <p>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    </a>
    <a href="https://pytorch.org/">
      <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
    </a>
    <a href="#">
      <img src="https://img.shields.io/badge/Data-Augmented-blueviolet?style=for-the-badge" alt="Data" />
    </a>
    <a href="#">
      <img src="https://img.shields.io/badge/Research-Thesis-FFD11B?style=for-the-badge" alt="Thesis" />
    </a>
  </p>

  <p>
    <i>"Data is the fuel. Code is the engine. Intelligence is the destination."</i>
  </p>

  <br />

  <p align="center">
    <a href="#-the-vault">The Vault</a> •
    <a href="#-the-architect">The Code</a> •
    <a href="#-initiation-protocol">Initiation</a> •
    <a href="#-mission-brief">Advisory</a>
  </p>
</div>

<hr />

## 📡 Mission Brief

This repository serves as the **classified archive** for the deep learning models powering the Calibur project. It houses the raw intelligence (datasets) and the instruction sets (training algorithms) required to forge the object detection system used in the associated thesis research.

Here, we store the baseline truths and the experimental variables used to push the boundaries of computer vision.

---

## The Vault (Datasets)

We do not simply store files; we curate intelligence. The repository is divided into specific sectors:

### 1. The Prime Directive (Main Dataset)
> **Filename:** `Skripsi_Aug_384_from1080_alt4.rar`

This is the **Gold Standard**.
* **Status:** Production Ready.
* **Intel:** Contains the primary dataset used to train the Master Model.
* **Processing:** Subjected to rigorous pre-processing and comprehensive data augmentation pipelines to ensure maximum generalization in the field.

### 2. The Experimental Variant (Frontal)
> **Filename:** `Skripsi_SplitThenAug_384_Threads_frontal_v5.rar`

The **Challenger**.
* **Status:** Experimental / Hypothesis Testing.
* **Intel:** A specialized subset focused on restricted viewing angles (Frontal-View).
* **Objective:** Isolated and processed to test specific hypotheses regarding model performance under unidirectional imaging conditions. Used as the control variable against the Master Model.

---

## The Architect (Training Code)

> **Directory:** `training_code.py` / `training_code/`

This is where the magic happens. This directory contains the **Life Cycle Algorithms** for the AI model:

* **Data Ingestion:** Scripts to load and normalize raw inputs.
* **Neural Architecture:** The blueprint of the model itself.
* **Training Loop:** The gym where the model learns (Forward pass, Backward pass, Validation).
* **Hyperparameters:** The tuning knobs for reproducibility.

---

## 🚀 Initiation Protocol

To deploy this intelligence locally, follow the standard operating procedure:

### Step 1: Extract Intelligence
Download the `.rar` artifact corresponding to your mission objective (Main vs. Experimental) and extract it to your local sector.

### Step 2: Equip Dependecies
Ensure your environment is primed. Install all Python libraries listed within the training code imports.

### Step 3: Commence Training
Execute the protocol:
1.  Open `training_code`.
2.  Point the `data_dir` variable to your extracted dataset.
3.  Run the script to begin the neural optimization process.

---

## ⚠️ Classified Advisory

**Authorized Personnel Only.**
All data and algorithms contained herein are strictly for **academic research** and the development of the thesis object detection system. Deployment outside of this context may require recalibration of the training parameters.

---

<div align="center">
  <p>Forged for Science. Optimized for Vision.</p>
</div>
