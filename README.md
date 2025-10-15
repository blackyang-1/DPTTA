# DP-TTA: Test-time Adaptation for Transient Electromagnetic Signal Denoising via Dictionary-driven Prior Regularization

> **Official PyTorch Implementation**  
> IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2025  
> DOI: [10.1109/TGRS.2025.3620531](https://doi.org/10.1109/TGRS.2025.3620531)  
> Authors: Meng Yang, Kecheng Chen, Wei Luo, Xianjie Chen, Yong Jia, Mingyue Wang, Fanqiang Lin  

---

## 🧠 Overview

**DP-TTA** (Dictionary-driven Prior Regularization Test-time Adaptation) is a novel framework for **Transient Electromagnetic (TEM) signal denoising**.  
It mitigates the **domain shift problem** — where noise characteristics vary across geological environments — by integrating **dictionary-driven priors** into **test-time adaptation (TTA)**.  

The core idea is that **TEM signals possess domain-invariant physical properties** (exponential decay and smoothness).  
DP-TTA exploits these as priors to guide dynamic parameter updates during inference, enabling better denoising performance in unseen environments.

---

## ✨ Highlights

- 🔹 **Dictionary-driven Prior Regularization** — explicitly encodes physical characteristics (exponential decay, smoothness).
- 🔹 **Test-time Self-supervised Adaptation** — updates model parameters on unseen test data.
- 🔹 **Three Self-supervised Losses:**
  - `L_denoising`: output consistency  
  - `L_sparse`: sparse code consistency  
  - `L_one-order`: smoothness consistency  
- 🔹 **Unified TTA Objective:**

  \[
  L_{TTA} = \beta_1 (L_{sparse} + L_{one-order}) + \beta_2 L_{denoising}
  \]

- 🔹 Outperforms most TEM denoising methods in both simulation and real-world experiments.

---

## 📁 Repository Structure
.
├── dptta.py # Main script: DP-TTA test-time adaptation
├── dptta_utils.py # Self-supervised losses, metrics, and helper functions
├── para_cfg.py # Hyperparameter settings
├── run_dptta_all.py # Batch runner for multiple TTA parameter configurations
├── test.py # Evaluate trained models without TTA
├── train.py # Source-domain pretraining for DTEMDNet
│
├── dictionary_learning/
│ ├── Dic_TEM_signal_project.ipynb # Learn dictionary atoms and sparse codes
│ └── tem_signal_example/
│ └── clean_signal.mat # Example TEM clean signal
│
├── lib/
│ ├── basic_layers.py # Core neural blocks
│ └── utils.py # 1D→2D transforms, normalization, metrics
│
└── model/
├── DTEMDNet.py # Main denoising model (CNN + dictionary regression)
├── DnCNN.py # Baseline CNN model
├── Resnet6.py # Baseline ResNet6 model
└── Resnet9.py # Baseline ResNet9 model


> ⚠️ Before running, **update the placeholder paths** in `train.py`, `test.py`, and `dptta.py` (e.g., `"your prepared dictionary data path"` and `"pretrained model path"`).

---

## ⚙️ Setup

### Environment
```bash
conda env create -f environment.yml
conda activate dptta
cuda 13.0




