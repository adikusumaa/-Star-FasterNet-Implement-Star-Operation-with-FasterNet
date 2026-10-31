# Star-FasterNet

**Analysis of FasterNet and MobileNetV3 Deep Learning Models Based on Facial Images for Driver Drowsiness Detection on ARM Mobile Platforms**

*Partial Convolution · Star Operation · Mobile ARM Benchmark · 5-Fold Cross Validation · Statistical Validation*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://mediapipe.dev/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-blue.svg)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.10+-purple.svg)](https://scipy.org/)
[![PyTorch Lite](https://img.shields.io/badge/PyTorch_Lite-Android-green.svg)](https://pytorch.org/mobile/home/)
[![ARM](https://img.shields.io/badge/ARM-Realme_5_Pro-lightgrey.svg)](https://www.arm.com/)

---

## Executive Summary

**Star-FasterNet** is a research repository that compares the performance of **MobileNetV3**, **FasterNet**, and three hybrid **Star-FasterNet** variants (Case I, II, III) for **driver drowsiness detection** using facial images on **ARM mobile platforms**.

This research combines **Partial Convolution (PConv)** from FasterNet with **Star Operation** to map features into a high-dimensional non-linear space implicitly without physically widening the channel dimension. Evaluation is performed on the **YAWDD** and **UTA-RLDD** datasets using 5-Fold Cross Validation, accuracy, precision, recall, F1-score, latency, CPU usage, memory usage, params, GFLOPs, efficiency score, and non-parametric statistical tests (Friedman, Iman-Davenport, Nemenyi, Kruskal-Wallis, Dunn).

**Key results:**

- **Star-FasterNet Case I** achieves the highest accuracy: **93.37%** on YAWDD and **96.60%** on UTA-RLDD.
- **Star-FasterNet Case III** achieves the best computational efficiency: latency **15.09 ms**, CPU usage **44.23%**, memory usage **7.87 MB**, params **1.35 M**, GFLOPs **0.141 G**, efficiency score **0.9950** on YAWDD and **0.9897** on UTA-RLDD.
- All Star-FasterNet variants are statistically not significantly different from MobileNetV3 in accuracy, but are superior in latency and resource usage.

---

## Key Highlights & Interface Preview

*(Placeholder for Android benchmark application screenshots and result visualizations. Images will be added after repository publication.)*

| Android Benchmark App | Confusion Matrix Visualization |
| :---: | :---: |
| `img/android_benchmark.png` | `img/confusion_matrix.png` |

---

## Research Impact

| Metric | Baseline (MobileNetV3) | Star-FasterNet Case III | Improvement |
| :--- | :--- | :--- | :--- |
| ARM latency | 48.33 ms | 15.09 ms | 3.2× faster |
| CPU usage | 47.22% | 44.23% | 6.3% lower |
| Memory usage | 19.83 MB | 7.87 MB | 2.5× more efficient |
| Params | 4.20 M | 1.35 M | 3.1× smaller |
| GFLOPs | 0.234 G | 0.142 G | 1.6× lighter |
| Efficiency score (YAWDD) | 0.7729 | 0.9950 | +28.7% |
| Efficiency score (UTA-RLDD) | 0.7729 | 0.9897 | +28.0% |

---

## Key Features

### 1. Model Architectures
- **MobileNetV3** – depthwise separable convolution baseline with Squeeze-and-Excitation.
- **FasterNet** – partial convolution + pointwise convolution.
- **Star-FasterNet Case I** – PConv + full two-branch Star Operation (W₁ᵀX+B₁) ★ (W₂ᵀX+B₂).
- **Star-FasterNet Case II** – PConv + one-branch Star Operation (W₁ᵀX+B₁) ★ X.
- **Star-FasterNet Case III** – PConv + identity Star Operation X ★ X without additional parameters.

### 2. Training Pipeline
- Subject-wise 5-Fold Cross Validation (70% train, 20% validation, 10% test).
- Augmentation: resize 224×224, random horizontal flip, random rotation, color jitter, normalization, random erasing.
- Hyperparameters: 100 epochs, batch size 32, AdamW, CosineAnnealingLR, label smoothing 0.1, CrossEntropyLoss.

### 3. Evaluation and Benchmarking
- Confusion matrix, accuracy, precision, recall, and F1-score per fold.
- Model conversion to PyTorch Lite (`.ptl`) using TorchScript tracing and `optimize_for_mobile`.
- Benchmark on Android ARM device (Realme 5 Pro, Snapdragon 712 AIE) with 10 warm-up iterations + 50 benchmark iterations, repeated 5 times.
- Measurement of latency, CPU usage, memory usage, and estimated FPS.

### 4. Statistical Tests
- **Friedman** and **Iman-Davenport** for accuracy (non-parametric, non-independent).
- **Kruskal-Wallis** for latency, CPU usage, and memory usage (non-parametric, independent).
- **Post-Hoc Nemenyi** for accuracy and **Dunn Test** (Bonferroni) for computational metrics.
- Normality tests: **Kolmogorov-Smirnov** and **Shapiro-Wilk** as prerequisites.

### 5. Efficiency Score
Uses the Multi-Criteria Decision Making (MCDM) framework with Simple Additive Weighting (SAW):
- Accuracy weight 0.5; latency 0.166667; CPU usage 0.166667; memory usage 0.166667.

---

## Main Results Tables

### Average 5-Fold Accuracy (%)

| Model | YAWDD Acc | YAWDD F1 | UTA-RLDD Acc | UTA-RLDD F1 |
| :--- | ---: | ---: | ---: | ---: |
| MobileNetV3 | 93.098 | 92.990 | 96.334 | 96.332 |
| FasterNet | 91.084 | 90.951 | 90.492 | 90.401 |
| Star-FasterNet Case I | **93.376** | **93.286** | **96.605** | **96.601** |
| Star-FasterNet Case II | 92.866 | 92.758 | 94.868 | 94.816 |
| Star-FasterNet Case III | 92.446 | 92.325 | 94.610 | 94.594 |

### Mobile ARM Computational Efficiency

| Model | Params (M) | GFLOPs | Latency (ms) | CPU (%) | Memory (MB) | Eff. YAWDD | Eff. UTA-RLDD |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MobileNetV3 | 4.204594 | 0.233561 | 48.33 | 47.22 | 19.83 | 0.7729 | 0.7729 |
| FasterNet | 2.629622 | 0.339372 | 43.87 | 49.52 | 13.15 | 0.7937 | 0.7744 |
| Star-FasterNet Case I | 2.201382 | 0.272011 | 27.31 | 47.78 | 11.61 | 0.8593 | 0.8593 |
| Star-FasterNet Case II | 1.777382 | 0.206782 | 20.81 | 46.55 | 9.70 | 0.9117 | 0.9054 |
| Star-FasterNet Case III | **1.353382** | **0.141553** | **15.09** | **44.23** | **7.87** | **0.9950** | **0.9897** |

### Statistical Tests

| Parameter | Test | Value | Threshold | Conclusion |
| :--- | :--- | ---: | ---: | :--- |
| YAWDD accuracy | Friedman | 12.32 | 9.488 | Reject H0 |
| UTA-RLDD accuracy | Friedman | 13.24 | 9.488 | Reject H0 |
| YAWDD accuracy | Iman-Davenport | 6.416 | 3.007 | Reject H0 |
| UTA-RLDD accuracy | Iman-Davenport | 7.834 | 3.007 | Reject H0 |
| Latency | Kruskal-Wallis | 1192.10 | p < 0.05 | Reject H0 |
| CPU usage | Kruskal-Wallis | 97.8702 | p < 0.05 | Reject H0 |
| Memory usage | Kruskal-Wallis | 23.1571 | p < 0.05 | Reject H0 |

---

## Repository Structure

    star-fasternet/
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── LICENSE
    ├── configs/
    │   ├── mobilenetv3.yaml
    │   ├── fasternet.yaml
    │   ├── star_fasternet_case1.yaml
    │   ├── star_fasternet_case2.yaml
    │   └── star_fasternet_case3.yaml
    ├── models/
    │   ├── __init__.py
    │   ├── mobilenetv3.py
    │   ├── fasternet.py
    │   └── star_fasternet.py
    ├── scripts/
    │   ├── preprocess.py
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── convert_ptl.py
    │   └── statistical_test.py
    ├── data/
    │   ├── README_data.md
    │   └── splits/
    │       ├── fold_1.csv
    │       ├── fold_2.csv
    │       ├── fold_3.csv
    │       ├── fold_4.csv
    │       └── fold_5.csv
    ├── checkpoints/
    │   └── .gitkeep
    ├── results/
    │   ├── tables/
    │   ├── figures/
    │   └── logs/
    ├── android/
    │   └── README_android.md
    ├── docs/
    │   ├── images/
    │   └── paper/
    └── tests/
        └── README_tests.md

---

## Tech Stack

| Category | Technology |
| :--- | :--- |
| Deep Learning | PyTorch, torchvision |
| Computer Vision | OpenCV, MediaPipe Face Mesh |
| Evaluation | scikit-learn, scipy, scikit-posthocs |
| Complexity | thop (params & GFLOPs) |
| Data Processing | pandas, numpy, openpyxl |
| Visualization | matplotlib, seaborn |
| Mobile Deployment | PyTorch Lite (`.ptl`), Android Studio (Kotlin) |
| Benchmark | Android `System.nanoTime`, `SystemClock`, `/proc/stat` |
| Configuration | PyYAML |

---

## Dataset

### YAWDD (Yawning Detection Dataset)
- 17,134 driver facial images (43.7% drowsiness, 56.3% non-drowsiness).
- 107 subjects, variations in lighting, glasses, and camera angles.
- Source: Kaggle / IEEE DataPort.

### UTA-RLDD (University of Texas at Arlington Real-Life Drowsiness Dataset)
- 14,738 images (50.89% active, 49.11% fatigue).
- 60 subjects, 224×224 resolution.
- Source: UTA Vision-Learning-Mining Lab.

> The dataset is not included in this repository. Please download from the official source and place it in the `data/` folder.

---

## Installation and Setup

### Prerequisites
- Python 3.10+
- Conda (recommended)
- Android Studio (for mobile benchmark)
- Android ARM device (Realme 5 Pro or equivalent)

### 1. Clone Repository

    git clone https://github.com/adikusumaa/star-fasternet.git
    cd star-fasternet

### 2. Create Environment

    conda create -n starfasternet python=3.10 -y
    conda activate starfasternet
    pip install -r requirements.txt

### 3. Prepare Dataset

Place the YAWDD and UTA-RLDD datasets that have been cropped using MediaPipe Face Mesh into the `data/` folder with the structure shown in the Repository Structure section.

### 4. Train Models

Example of training Star-FasterNet Case III on YAWDD fold 1:

    python scripts/train.py --config configs/star_fasternet_case3.yaml --data-dir data/yawdd --fold 1 --epochs 100 --batch-size 32

The best weights will be saved in `checkpoints/`.

### 5. Evaluate Models

    python scripts/evaluate.py --config configs/star_fasternet_case3.yaml --checkpoint checkpoints/star_fasternet_case3_yawdd_fold1.pth --data-dir data/yawdd --fold 1

### 6. Convert to PyTorch Lite

    python scripts/convert_ptl.py --config configs/star_fasternet_case3.yaml --checkpoint checkpoints/star_fasternet_case3_yawdd_fold1.pth --output checkpoints/star_fasternet_case3.ptl

### 7. Mobile ARM Benchmark

Use the Android application provided in the `android/` folder (or create your own with Android Studio) to load the `.ptl` file and run the benchmark. Protocol: 10 warm-up iterations, 50 benchmark iterations, repeated 5 times.

### 8. Statistical Tests

    python scripts/statistical_test.py --results results/tables/ --output results/tables/statistical_results.csv

---

## Usage Examples

### Train all models on 5 folds

    for model in mobilenetv3 fasternet star_fasternet_case1 star_fasternet_case2 star_fasternet_case3; do
        for fold in 1 2 3 4 5; do
            python scripts/train.py --config configs/${model}.yaml --data-dir data/yawdd --fold $fold
        done
    done

### Evaluate all models

    for model in mobilenetv3 fasternet star_fasternet_case1 star_fasternet_case2 star_fasternet_case3; do
        for fold in 1 2 3 4 5; do
            python scripts/evaluate.py --config configs/${model}.yaml --checkpoint checkpoints/${model}_yawdd_fold${fold}.pth --data-dir data/yawdd --fold $fold
        done
    done

### Convert all models to PyTorch Lite

    for model in mobilenetv3 fasternet star_fasternet_case1 star_fasternet_case2 star_fasternet_case3; do
        python scripts/convert_ptl.py --config configs/${model}.yaml --checkpoint checkpoints/${model}_yawdd_fold1.pth --output checkpoints/${model}.ptl
    done

---

## Testing

Run all unit tests:

    pytest tests/ -v

Run a specific module:

    pytest tests/test_models.py -v -s

Test coverage includes model architectures, preprocessing pipeline, PyTorch Lite conversion, metric calculations, and statistical tests.

---

## Security and Compliance

| Aspect | Implementation |
| :--- | :--- |
| Data Privacy | Dataset is not included; users must comply with each dataset license |
| Model Security | Model weights are stored locally; no external API is called during inference |
| Query Safety | No SQL queries; only model inference |
| Benchmark Integrity | Warm-up protocol and 5× repetition to reduce noise |
| Reproducibility | Fixed seed (42) for training and evaluation |

---

## Roadmap

| Phase | Description | Status |
| :--- | :--- | :--- |
| 1 | Implement MobileNetV3, FasterNet, Star-FasterNet Case I–III architectures | Completed |
| 2 | Preprocessing & augmentation pipeline | Completed |
| 3 | 5-Fold training & evaluation on YAWDD and UTA-RLDD | Completed |
| 4 | PyTorch Lite conversion & mobile ARM benchmark | Completed |
| 5 | Statistical tests & efficiency score | Completed |
| 6 | Repository publication & documentation | Completed |
| 7 | Additional dataset variations and drowsiness scenarios | Planned |

---

## Future Enhancements

- Add more diverse subject variations, lighting conditions, and camera viewpoints.
- Explore other convolution types such as GSConv for lighter architectures.
- Integrate the model into a real-time mobile application with auditory alerts.
- Add support for other datasets (NTHU-DDD, DPD).
- Publicly release model weights and benchmark results.

---

## Citation

If this repository is helpful, please cite:

    @article{chen2023run,
      title={Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks},
      author={Chen, Jierun and Kao, Shiu-hong and He, Hao and Zhuo, Weipeng and Wen, Song and Lee, Chul-Ho and Chan, S-H Gary},
      journal={arXiv preprint arXiv:2303.03667},
      year={2023}
    }

    @inproceedings{ma2024rewrite,
      title={Rewrite the Stars},
      author={Ma, Xu and Dai, Xiyang and Bai, Yueru and Wang, Yizhou and Fu, Yun},
      booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2024}
    }

    @misc{nuhgraha2026starfasternet,
      title={Analysis of FasterNet and MobileNetV3 Deep Learning Models Based on Facial Images for Driver Drowsiness Detection on ARM Mobile Platforms},
      author={Nur Adiyanto Kusuma Nuhgraha},
      year={2026},
      note={Undergraduate Thesis, Universitas Brawijaya}
    }

---

## Acknowledgement

This repository is built with reference to:

- FasterNet: https://github.com/JierunChen/FasterNet
- StarNet / Rewrite the Stars
- MobileNetV3
- PyTorch, torchvision, timm
- MediaPipe Face Mesh
- scikit-learn, scipy, scikit-posthocs
- thop

---

## Contributing

1. Fork this repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'feat: add AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## License

Distributed under the MIT License. Developed by **Nur Adiyanto Kusuma Nuhgraha**  
Computer Engineering Study Program, Faculty of Computer Science, Universitas Brawijaya  
Email: adikusuma@student.ub.ac.id