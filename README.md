# 🎨 WikiArt Painting Classifier
### Deep Learning Project — NOVA IMS 2025/2026

A deep learning image classification system built on a subset of the **WikiArt** dataset, implemented in **Keras**. The project explores and compares multiple architectures — from custom CNNs to fine-tuned pretrained models — with a rigorous evaluation framework.

> **Deadline:** 24 April 2026, 17:00 · **Submission:** Moodle (GROUP_X.rar)

---

## 📁 Repository Structure

```
dl-project/
│
├── data/
│   ├── raw/                        # Original wikiart images (gitignored)
│   ├── processed/                  # Preprocessed & resized images
│   └── splits/                     # train.csv, val.csv, test.csv
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis (Miguel)
│   ├── 02_baseline_cnn.ipynb       # Baseline CNN (Goncalo)
│   ├── 03_custom_cnn.ipynb         # Deeper custom CNN (Goncalo)
│   ├── 04_transfer_learning.ipynb  # Pretrained backbone experiments (Anastasiia)
│   └── 05_evaluation.ipynb         # Cross-model evaluation & error analysis (Leonor)
│
├── src/
│   ├── data_loader.py              # Dataset pipeline & augmentation (Miguel + Anastasiia)
│   ├── models/
│   │   ├── baseline.py             # Shallow baseline CNN (Goncalo)
│   │   ├── custom_cnn.py           # Deep custom CNN (Goncalo)
│   │   └── transfer.py             # Transfer learning model (Anastasiia)
│   ├── train.py                    # Shared training loop
│   └── evaluate.py                 # Metrics, plots, confusion matrix (Leonor)
│
├── results/
│   ├── figures/                    # All plots and visualisations (Leonor's output)
│   └── logs/                       # Training history CSVs / TensorBoard logs
│
├── report/
│   └── report.pdf                  # Final 7-page report (Henrique)
│
├── requirements.txt
├── README.md
└── .gitignore
```

> **Note:** The `data/raw/` and `data/processed/` folders are gitignored. Raw images should be obtained from the course-provided link or shared drive. Only the split CSVs in `data/splits/` are versioned.

---

## 👥 Team & Responsibilities

| Member | Role | Primary Responsibilities |
|--------|------|--------------------------|
| **Miguel** | Project Lead & Data Specialist | Dataset preprocessing, EDA, train/val/test splits, repo management, final submission packaging |
| **Goncalo** | Baseline & CNN Architect | Baseline CNN, deep custom CNN, hyperparameter tuning, architectural ablations |
| **Anastasiia** | Transfer Learning Specialist | Pretrained backbones (EfficientNetV2 / ResNet50), fine-tuning, data augmentation pipeline |
| **Leonor** | Evaluation & Analysis Lead | Metrics (accuracy, F1-macro), confusion matrices, error analysis, all result figures |
| **Henrique** | Report Writer | Full 7-page report, problem formalisation, methodology narrative, submission formatting |

---

## 🗓️ Project Timeline

| Week | Dates | Milestones | Owner(s) |
|------|-------|------------|----------|
| **Week 1** | Mar 17–23 | Repo setup, dataset download & EDA, baseline model running end-to-end | Miguel, Goncalo |
| **Week 2** | Mar 24–30 | Custom CNN tuned, transfer learning experiments started, error analysis begun | Goncalo, Anastasiia, Leonor |
| **Week 3** | Mar 31–Apr 6 | All models benchmarked, full results ready, report v1 drafted | Anastasiia, Leonor, Henrique |
| **Week 4** | Apr 7–13 | Report v2 with all sections integrated, final hyperparameter refinements | All |
| **Week 5** | Apr 14–20 | Final review, report polish, submission prep | Miguel, Henrique |
| **Deadline** | **Apr 24, 17:00** | Upload `GROUP_X.rar` to Moodle | Miguel |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-org/dl-project.git
cd dl-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the data
Place the raw WikiArt images in `data/raw/`, then run the preprocessing script:
```bash
python src/data_loader.py --input data/raw/ --output data/processed/
```
This will generate the split CSVs in `data/splits/`.

### 4. Train a model
```bash
# Baseline CNN
python src/train.py --model baseline --epochs 30

# Custom CNN
python src/train.py --model custom_cnn --epochs 50

# Transfer learning (EfficientNetV2)
python src/train.py --model transfer --backbone efficientnetv2 --epochs 20
```

### 5. Evaluate
```bash
python src/evaluate.py --model transfer --checkpoint results/logs/transfer_best.h5
```

---

## 🏗️ Models Overview

### Baseline CNN (Goncalo)
A shallow convolutional network used as a performance floor. Establishes the training pipeline and confirms data is loading correctly before more complex models are introduced.

### Custom CNN (Goncalo)
A deeper architecture with multiple Conv blocks, Batch Normalisation, and Dropout, designed and tuned specifically for the WikiArt classification task.

### Transfer Learning (Anastasiia)
Fine-tuned pretrained backbone (EfficientNetV2 or ResNet50) with a custom classification head. Experiments cover both feature-extraction mode and full fine-tuning, combined with an augmentation pipeline (random crop, horizontal flip, colour jitter).

---

## 📊 Evaluation Framework (Leonor)

All models are evaluated on the held-out test set using:

- **Accuracy** — overall and per-class
- **F1-macro** — accounts for class imbalance
- **Confusion matrix** — full and normalised
- **Learning curves** — training vs. validation loss/accuracy
- **Error analysis** — qualitative review of misclassified samples

All figures are saved to `results/figures/` with descriptive names (e.g. `confusion_matrix_efficientnet_normalised.png`) for direct use in the report.

---

## 🤝 Git Workflow

1. Each member works on a dedicated branch: `feature/<topic>` (e.g. `feature/transfer-learning`)
2. Open a Pull Request into `main` when a milestone is complete
3. **Miguel** reviews and merges all PRs
4. Commit messages follow the format: `[Goncalo] Add custom CNN with BN and dropout`

---

## 📋 Submission Checklist

- [ ] Source code (all notebooks + `src/` scripts)
- [ ] Link to dataset (if different from the provided wikiart subset)
- [ ] Report PDF — max 7 pages, Arial 12pt (`report/report.pdf`)
- [ ] All files packaged as `GROUP_X.rar`
- [ ] Uploaded to Moodle by **24 April 2026, 17:00**

---

## ⚙️ Requirements

Key dependencies (see `requirements.txt` for pinned versions):

- `tensorflow >= 2.10`
- `keras`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `Pillow`

---

## 📄 License

Academic project — NOVA IMS Deep Learning course, 2025/2026.