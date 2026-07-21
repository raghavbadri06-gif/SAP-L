# SAP-L: Spectral Abnormality Progression Learning for Heel Spur Severity Assessment

## Overview

SAP-L is a **progression-aware representation learning framework** for automated heel spur severity assessment from lateral foot radiographs. Unlike conventional deep learning approaches that primarily optimize classification accuracy, SAP-L explicitly models the ordinal progression of disease severity by introducing frequency-domain constraints into the learned latent representations.

SAP-L transforms latent features into the spectral domain using the **Discrete Cosine Transform (DCT)**, decomposes them into low-, mid-, and high-frequency components, and introduces **Spectral Progression Regularization (SPR)** to enforce monotonic ordering of class-wise high-frequency spectral energy. This encourages progression-aware latent representations while remaining compatible with existing convolutional neural network backbones.

### Performance

The framework is evaluated using **GhostNet**, **EfficientNet-B0**, and **MobileNetV3** on the publicly available Heel Bone Dataset. **EfficientNet-B0 achieved the best overall classification performance**, while SAP-L consistently enhanced latent-space organization across all evaluated backbones through improved representation quality and class separability.

## Dataset

The experiments were performed on the publicly available **Heel Bone Dataset**.

**Link: [Kaggle Dataset](https://www.kaggle.com/datasets/osamahtaher/heel-dataset)**

### Classes
- **Normal** - Healthy heel bone
- **Heel Spur** - Mild abnormality
- **Severe Heel Spur** - Advanced condition (referred to as "Sever" in the manuscript)

## Architecture

### SAP-L Module
SAP-L is a lightweight representation learning module that can be integrated into standard CNN backbones.

### Pipeline

```
Input Radiograph
        |
        v
 CNN Backbone
(GhostNet / EfficientNet-B0 / MobileNetV3)
        |
        v
 Latent Feature Representation
        |
        v
 DCT-based Spectral Decomposition
        |
        v
 Low / Mid / High Frequency Components
        |
        v
 Spectral Progression Regularization (SPR)
        |
        v
 Progression-aware Latent Representation
        |
        v
 Classification
```

## Key Features

- Spectral Abnormality Progression Learning (SAP-L)
- DCT-based latent feature decomposition
- Spectral Progression Regularization (SPR)
- Progression-aware representation learning
- Compatible with multiple CNN backbones
- Multi-seed experimental evaluation
- Bootstrap statistical validation
- Latent-space visualization (UMAP & t-SNE)
- Grad-CAM explainability
- Representation quality analysis

## Backbone Architectures

The framework supports:

| Backbone | Performance |
|----------|-------------|
| GhostNet | Supported |
| EfficientNet-B0 | Best Classification Performance |
| MobileNetV3 | Supported |

## Representation Analysis

SAP-L improves latent-space organization using:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Inter-class Centroid Distance
- KNN Consistency
- Transition Overlap
- Kruskal-Wallis Separability

## Implementation Environment

### Hardware

| Component | Specification |
|-----------|---------------|
| Workstation | Dell G15 |
| Processor | Intel Core Ultra 9 (13th Gen) |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU (12 GB) |
| RAM | 16 GB DDR5 |
| Storage | 1 TB NVMe SSD |
| Operating System | Windows 11 Pro |

### Software

- Python 3.10
- PyTorch
- timm
- NumPy
- SciPy
- scikit-learn
- OpenCV
- Matplotlib
- UMAP
- Grad-CAM utilities

### High-Performance Computing (HPC)

Large-scale multi-seed experiments were conducted on a High-Performance Computing (HPC) environment for efficient training and reproducibility. The repository also supports execution on local CUDA-enabled NVIDIA GPUs.

## Repository Structure

```
SAP-L/
├── configs/
│   └── ...
├── datasets/
│   └── ...
├── models/
│   └── ...
├── sapl/
│   ├── dct.py
│   ├── spectral_decomposition.py
│   ├── spr_loss.py
│   └── spectral_utils.py
├── training/
│   └── ...
├── evaluation/
│   └── ...
├── visualization/
│   └── ...
├── statistics/
│   └── ...
├── requirements.txt
└── README.md
```

## Citation

If you use this repository in your research, please cite the associated publication when published.

```
@article{sapl2026,
  title={SAP-L: Spectral Abnormality Progression Learning for Heel Spur Severity Assessment},
  author={},
  journal={},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Heel Bone Dataset contributors for making the data publicly available
- The open-source community for the tools and libraries used in this project
