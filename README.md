# FMTUNet
# FMTUNet: A High-Fidelity Synergistic Mamba-Transformer Framework for Remote Sensing Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Task: Land Cover Classification](https://img.shields.io/badge/Task-Land%20Cover%20Classification-green.svg)](https://github.com/topics/remote-sensing)

Official PyTorch implementation of **FMTUNet**, as described in the paper submitted to *Computers & Geosciences*:  
**"FMTUNet: A High-Fidelity Synergistic Mamba-Transformer Framework for Complex Remote Sensing Semantic Segmentation"**.

---

## 📖 Introduction

**FMTUNet** is designed to address the "accuracy bottleneck" in automated earth observation. Unlike lightweight models that sacrifice detail for speed, FMTUNet prioritizes **segmentation precision and boundary integrity**, making it ideal for high-quality offline mapping tasks.

It reconstructs the multi-modal interaction paradigm by synergizing:
1.  **Synergistic Mamba Fusion (SMF):** Exploits the linear-complexity selective scan of Mamba for dynamic, long-range feature alignment between Optical and DSM data.
2.  **Global Modeling Module (GMM):** Utilizes **Hybrid Deformable Attention (HDA)**. While computationally intensive, this mechanism is critical for capturing the irregular geometries (e.g., distorted building footprints) that standard fixed-grid attention misses.

**Performance (mIoU):**
*   **ISPRS Vaihingen:** 84.31% (SOTA)
*   **ISPRS Potsdam:** 86.32% (SOTA)

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
git clone https://github.com/[YOUR_USERNAME]/FMTUNet.git
cd FMTUNet

