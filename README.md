<div align="center">

# C A L I B U R - DATA SIDE

### **The Neural Backbone & Training Grounds of the Calibur Detection System**

> *Data is the fuel. Code is the engine. Intelligence is the destination.*

---

<p>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  </a>
  <img src="https://img.shields.io/badge/Data-Augmented-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Research-Thesis-FFD11B?style=for-the-badge" />
</p>


</div>

---

## Mission Brief

This repository is the **research core** of the Calibur project.

It contains the datasets, augmentation strategies, and training pipelines used to develop and evaluate the object detection models presented in the associated academic thesis.

This is not a deployment repository.
This is where **ground truth is forged**, hypotheses are tested, and models earn their performance claims.

The contents are structured to ensure:

* Reproducibility of experiments
* Clear separation between baseline and experimental data
* Transparent model training workflows

---

## The Vault (Datasets)

Data is treated as intelligence — curated, versioned, and purpose-driven.

### 1. Prime Directive (Main Dataset)

**Filename:**

```
Skripsi_Aug_384_from1080_alt4.rar
```

**Role:** Primary training dataset

**Status:** Production-grade (research baseline)

**Description:**

* Serves as the **gold standard dataset** for the thesis
* Derived from high-resolution source imagery
* Processed through a full augmentation pipeline
* Designed to maximize generalization across varied conditions

This dataset is used to train the **Master Model**, against which all experimental variants are compared.

---

### 2. Experimental Variant (Frontal)

**Filename:**

```
Skripsi_SplitThenAug_384_Threads_frontal_v5.rar
```

**Role:** Controlled experimental dataset

**Status:** Experimental / hypothesis testing

**Description:**

* Focused exclusively on frontal-view imagery
* Isolated prior to augmentation to preserve directional bias
* Used to evaluate performance under constrained viewing conditions

This dataset functions as a **controlled variable**, enabling direct comparison against the Master Model trained on the Prime Directive dataset.

---

## The Architect (Training Code)

**Location:**

```
training_code.py
# or
training_code/
```

This module defines the **full training lifecycle** of the detection model.

### Responsibilities

* Dataset loading and normalization
* Model architecture definition
* Training and validation loops
* Loss computation and optimization
* Hyperparameter configuration for reproducibility

The code is intentionally explicit rather than abstracted, prioritizing **clarity and traceability** over framework-level automation.

---

## Initiation Protocol

Follow this procedure to reproduce experiments locally.

### Step 1 — Extract Dataset

Select the dataset aligned with your objective:

* Prime Directive → baseline replication
* Experimental Variant → hypothesis testing

Extract the `.rar` file into a local directory.

---

### Step 2 — Prepare Environment

Ensure a compatible Python environment is available.

Install required dependencies as indicated by the training script imports (e.g., PyTorch, NumPy, OpenCV, etc.).

Virtual environments are strongly recommended.

---

### Step 3 — Commence Training

1. Open the training script or directory
2. Update the `data_dir` variable to point to the extracted dataset
3. Execute the training process

The model will begin iterative optimization according to the defined configuration.

---

## Advisory Notice

**Academic Use Only**

All datasets and training logic contained in this repository are intended solely for:

* Academic research
* Thesis experimentation
* Reproducible evaluation

Any use outside this scope may require architectural changes, dataset rebalancing, or retraining.

---

<div align="center">

*Forged for research. Optimized for vision.*

</div>
