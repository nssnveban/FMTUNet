# FMTUNet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Task: Land Cover Classification](https://img.shields.io/badge/Task-Land%20Cover%20Classification-green.svg)](https://github.com/topics/remote-sensing)

---

## 📖 Introduction

**FMTUNet** is designed to address the "accuracy bottleneck" in automated earth observation. Unlike lightweight models that sacrifice detail for speed, FMTUNet prioritizes **segmentation precision and boundary integrity**, making it ideal for high-quality offline mapping tasks.

It reconstructs the multi-modal interaction paradigm by synergizing:
1.  **Synergistic Mamba Fusion (SMF):** Exploits the linear-complexity selective scan of Mamba for dynamic, long-range feature alignment between Optical and DSM data.
2.  **Global Modeling Module (GMM):** Utilizes **Hybrid Deformable Attention (HDA)**. While computationally intensive, this mechanism is critical for capturing the irregular geometries (e.g., distorted building footprints) that standard fixed-grid attention misses.

**Performance Highlights:**
*   **ISPRS Vaihingen:** 92.43% OA | 84.31% mIoU 
*   **ISPRS Potsdam:** 91.41% OA | 86.32% mIoU 

---

## 💻 System Requirements

This model is optimized for **accuracy-critical applications** in geosciences. Due to the sophisticated attention mechanisms and high-resolution inputs, it requires a robust hardware environment.

*   **OS:** Linux (Ubuntu 20.04/22.04 recommended). *Windows support for `mamba-ssm` is experimental.*
*   **GPU:** NVIDIA GPU with **Compute Capability ≥ 8.0** (Ampere architecture or newer).
*   **VRAM:** 
    *   **Training:** ≥ 24GB (e.g., RTX 3090/4090, A100/A800) is strongly recommended.
    *   **Inference:** ≥ 12GB.
*   **CUDA:** 11.8 (Strictly required for the provided installation steps).

---

## 🛠️ Installation

To ensure reproducibility, please follow these steps strictly to configure the Mamba environment.

### 1. Clone the repository
```bash
git clone https://github.com/nssnveban/FMTUNet.git
cd FMTUNet
```

### 2. Create Environment
```bash
conda create -n fmtunet python=3.10
conda activate fmtunet
```

### 3. Install PyTorch (CUDA 11.8)
Note: We use PyTorch 2.1.1 which is compatible with Mamba 2.2.2.
```bash
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install Mamba-SSM & Dependencies
This step requires nvcc (CUDA compiler) to be available in your path.
```bash
# Install core libraries
pip install -r requirements.txt

# Install Mamba components
pip install causal-conv1d==1.4.0
pip install mamba-ssm==2.2.2
```
---

## ⚡ Usage

### Training & Validation

The training script integrates both the training loop and periodic validation on the test set. To start the training process, simply run:

```bash
python train.py
```

Note: The training parameters (e.g., base_lr, epochs) are defined directly within train.py. Please modify the script to adjust these hyperparameters according to your hardware capabilities.

### Customizing the Optimization Metric
By default, the script saves the model checkpoint that achieves the highest Overall Accuracy (OA) during validation.

If you prefer to optimize for Mean Intersection over Union (mIoU), you can modify the train function in train.py (around line 197):
```python
# Save model if current accuracy is better than best
if metrics['accuracy'] > acc_best: # <--- Change this key and the follows 
    torch.save(net.state_dict(), './results_posd/posd_epoch{}_{}'.format(e, metrics['accuracy']))
    acc_best = metrics['accuracy']
```

---

## 📂 Data Preparation

We use the **ISPRS 2D Semantic Labeling Challenge** datasets (Vaihingen & Potsdam). Due to license restrictions, we cannot distribute the data directly.

1.  Request the data from the [ISPRS Official Website](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx), or can be downloaded [here](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets)
2.  Organize the data as follows:

```text
datasets/
├── vaihingen/
│   ├── top/                             # IRRG images
│   ├── dsm/                             # DSM data
│   ├── gts_for_participants/            # Ground Truth
│   └── gts_eroded_for_participants/     
└── potsdam/
    ├── 1_DSM_normalisation/             # DSM data
    ├── 4_Ortho_RGBIR/                   # RGB/IRRG images
    ├── 5_Labels_for_participants/       # Ground Truth
    └── 5_Labels_for_participants_no_Boundary/
```
## 🤝 Acknowledgement

We thank the authors of [ViT](https://github.com/google-research/vision_transformer), [Mamba](https://github.com/state-spaces/mamba), and [FTransUNet](https://github.com/sstary/SSRS) for their open-source contributions. We also acknowledge the ISPRS WG II/4 for providing the Vaihingen and Potsdam benchmark datasets.
