# UGB-SegNet: Uncertainty-Guided Breast Segmentation Network

**PhD Dissertation — Feng Chia University, 2026**  
**Author:** Taukir Alam  
**Supervisor:** Professor Fang-Rong Hsu  
**Department:** Information Engineering and Computer Science

---

## Overview

UGB-SegNet is a novel deep learning architecture for automatic breast lesion segmentation in B-mode ultrasound images. It integrates four original contributions to address key limitations of existing methods:

1. **Multi-Scale CBAM Attention** — applied simultaneously at three EfficientNet-B0 encoder scales
2. **Learnable Feature Pyramid Network** — trainable scale-wise fusion weights (w3, w4, w5)
3. **Uncertainty-Guided Decoder** — entropy-gated skip connections for boundary refinement
4. **Boundary-Aware Combined Loss** — 5× boundary pixel weighting for BI-RADS margin precision

---

## Results on BUSI Dataset

### Main Test Set (n=129 held-out images)

| Model | Dice (%) | IoU (%) | Sensitivity (%) | Specificity (%) | HD95 (px) | Params (M) |
|---|---|---|---|---|---|---|
| **UGB-SegNet ★** | **81.30** | **68.49** | **81.71** | **98.14** | **15.63** | **14.00** |
| DeepLabV3+ | 79.28 | 65.70 | 78.93 | 97.89 | 21.52 | 39.60 |
| SegNet | 74.93 | 59.88 | 74.12 | 97.59 | 27.03 | 28.81 |
| Attention U-Net | 74.23 | 59.20 | 73.85 | 97.68 | 28.74 | 31.04 |
| U-Net | 71.24 | 55.18 | 70.82 | 97.31 | 32.00 | 31.04 |
| TransUNet | 68.61 | 52.27 | 68.29 | 97.12 | 36.17 | 87.20 |

### 5-Fold Cross-Validation

| Fold | Dice (%) | IoU (%) | Sensitivity (%) | HD95 (px) |
|---|---|---|---|---|
| 1 | 78.05 | 68.93 | 77.98 | 17.75 |
| 2 | 76.20 | 66.54 | 78.61 | 22.62 |
| 3 | 77.17 | 67.76 | 82.28 | 20.05 |
| 4 | 76.72 | 66.93 | 76.62 | 20.32 |
| 5 | 73.64 | 63.97 | 81.01 | 43.91 |
| **Mean ± SD** | **77.02 ± 1.90** | **66.83 ± 1.58** | **79.30 ± 2.14** | **24.93 ± 10.76** |

---

## Dataset

**BUSI** (Breast Ultrasound Images) — Al-Dhabyani et al., Data in Brief, 2020  
- 647 annotated images (437 benign, 210 malignant)  
- Stratified split: 454 train / 64 val / 129 test (seed=42)  
- Download: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

---

## Installation

```bash
git clone https://github.com/taukiralam007/UGB-SegNet.git
cd UGB-SegNet
pip install -r requirements.txt
```

---

## Usage

### Train with 5-Fold Cross-Validation
```bash
python run_5fold_cv.py
```

### Train UGB-SegNet only
```bash
python run_5fold_cv.py --model UGB-SegNet
```

### Smoke test (4 epochs)
```bash
python run_5fold_cv.py --model UGB-SegNet --smoke
```

### Verify results on test set
```bash
python verify.py
```

---

## Model Architecture

```
Input (224×224×3)
    ↓
EfficientNet-B0 Encoder (pretrained ImageNet)
    ↓
Multi-Scale CBAM Attention (stages s2, s5, s8)
    ↓
Learnable FPN (adaptive weighted fusion)
    ↓
Uncertainty-Guided Decoder (entropy-gated skip connections)
    ↓
Output: Segmentation Mask + Uncertainty Map (224×224)
```

**Total parameters:** 14.00M  
**Inference speed:** 62.4 FPS at 224×224  
**Storage:** 56.6 MB

---

## Requirements

- Python 3.9+
- PyTorch 1.13.0+
- CUDA 11.7+ (GPU recommended)
- Windows compatible (num_workers=0)

See `requirements.txt` for full dependencies.

---

## Citation

If you use this code in your research, please cite:

```
@phdthesis{alam2026ugbsegnet,
  title     = {Advancing Medical Imaging Diagnosis: Integrating Deep Learning 
               and Computer Vision for Enhanced Biomedical Image Analysis},
  author    = {Taukir Alam},
  school    = {Feng Chia University},
  year      = {2026},
  address   = {Taichung, Taiwan}
}
```

**Phase 1 published baseline:**
```
@article{alam2023unet3plus,
  title   = {Improving Breast Cancer Detection and Diagnosis through Semantic 
             Segmentation using the UNet3+ Deep Learning Framework},
  author  = {Alam, Taukir and Shia, Wei-Chung and Hsu, Fang-Rong and Hassan, Taimur},
  journal = {Biomedicines},
  volume  = {11},
  number  = {6},
  pages   = {1536},
  year    = {2023},
  doi     = {10.3390/biomedicines11061536}
}
```

---

## License

MIT License — free to use for research purposes.
