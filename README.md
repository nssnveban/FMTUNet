# FMTUNet
# Synergistic Mamba-Transformer Fusion for High-Fidelity Semantic Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Task: Semantic Segmentation](https://img.shields.io/badge/Task-Semantic%20Segmentation-blue.svg)](https://github.com/topics/semantic-segmentation)

Official PyTorch implementation of **FMTUNet**, as described in the paper:  
**"Synergistic Mamba-Transformer Fusion for High-Fidelity Semantic Segmentation of Multi-modal Remote Sensing Imagery"**, submitted to *Computers & Geosciences*.

---

## 🚀 Introduction

**FMTUNet** is a novel multi-modal fusion network designed to address the heterogeneity and geometric complexity of high-resolution remote sensing imagery (RGB + DSM). 

Moving beyond standard CNN-Transformer hybrids, our approach reconstructs the interaction paradigm by synergizing:
1.  **Synergistic Mamba Fusion (SMF):** Exploits the linear-complexity selective scan of Mamba for dynamic, long-range shallow feature alignment.
2.  **Global Modeling Module (GMM):** Utilizes **Hybrid Deformable Attention (HDA)** to capture irregular geometries and fine-grained boundaries, offering superior adaptivity compared to rigid fixed-grid attention.

**Key Performance:**
*   **ISPRS Vaihingen:** 84.31% mIoU (SOTA)
*   **ISPRS Potsdam:** 86.32% mIoU (SOTA)

> **Note:** This model prioritizes **segmentation accuracy and boundary fidelity** for complex geospatial scenes. As a result, it employs sophisticated attention mechanisms that require sufficient GPU memory for training.

---

## 🖼️ Architecture

![Architecture](figures/framework.png)
*(Please place your Figure 1 here as `figures/framework.png`)*

---

## 💻 System Requirements

Due to the integration of **Mamba (State Space Models)** and **Deformable Attention**, specific hardware and software environments are required:

*   **OS:** Linux (Ubuntu 20.04/22.04 recommended). *Windows support for `mamba-ssm` is experimental.*
*   **GPU:** NVIDIA GPU with **compute capability ≥ 8.0** (Ampere architecture or newer) is recommended.
*   **VRAM:** 
    *   **Training:** ≥ 24GB (e.g., RTX 3090, 4090, A100) is strongly recommended due to the high-resolution input and complex attention maps.
    *   **Inference:** ≥ 12GB.
*   **CUDA:** 11.8 or higher.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[YOUR_USERNAME]/FMTUNet.git
    cd FMTUNet
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    conda create -n fmtunet python=3.10
    conda activate fmtunet
    ```

3.  **Install dependencies:**
    *First, ensure you have PyTorch installed compatible with your CUDA version.*
    ```bash
    # Example for CUDA 11.8
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
    
    # Install other requirements
    pip install -r requirements.txt
    
    # Install Mamba-SSM (This step might take a while)
    pip install causal-conv1d>=1.2.0
    pip install mamba-ssm
    ```

---

## 📂 Data Preparation

We use the **ISPRS 2D Semantic Labeling Challenge** datasets (Vaihingen & Potsdam). Due to license restrictions, we cannot distribute the data directly.

1.  Request the data from the [ISPRS Official Website](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx).
2.  Organize the data as follows:

```text
datasets/
├── vaihingen/
│   ├── top/           # IRRG images
│   ├── dsm/           # DSM data
│   └── gts/           # Ground Truth
└── potsdam/
    ├── top/           # RGB/IRRG images
    ├── dsm/           # DSM data
    └── gts/           # Ground Truth
