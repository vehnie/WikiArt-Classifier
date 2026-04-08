# 🎨 WikiArt Painting Classifier
### Deep Learning Project — NOVA IMS 2025/2026

A deep learning image classification system built on a subset of the **WikiArt** dataset, implemented in **Keras**. The project explores and compares multiple architectures — from custom CNNs and fine-tuned pretrained models to a Vision Transformer (ViT) — with a rigorous evaluation framework and Grad-CAM explainability analysis.

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
│   ├── 05_vit.ipynb                # Vision Transformer fine-tuning (Miguel)
│   ├── 06_gradcam.ipynb            # Grad-CAM explainability analysis (Miguel)
│   └── 07_evaluation.ipynb         # Cross-model evaluation & error analysis (Leonor)
│
├── src/
│   ├── data_loader.py              # Dataset pipeline & augmentation (Miguel + Anastasiia)
│   ├── models/
│   │   ├── baseline.py             # Shallow baseline CNN (Goncalo)
│   │   ├── custom_cnn.py           # Deep custom CNN (Goncalo)
│   │   ├── transfer.py             # Transfer learning model (Anastasiia)
│   │   └── vit.py                  # Vision Transformer fine-tuning (Miguel)
│   ├── train.py                    # Shared training loop
│   ├── evaluate.py                 # Metrics, plots, confusion matrix (Leonor)
│   └── gradcam.py                  # Grad-CAM heatmap generation (Miguel) 
│
├── results/
│   ├── figures/                    # All plots and visualisations (Leonor)
│   ├── gradcam/                    # Grad-CAM heatmaps per class (Miguel)
│   ├── logs/                       # Training history CSVs
│   └── models/                     # Saved model checkpoints (.keras)
│
├── app/
│   ├── server.py                   # Flask backend for prediction API (Miguel)
│   └── static/
│       └── index.html              # Web frontend with drag-and-drop upload
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
| **Miguel** | Project Lead, Data & SOTA | Dataset preprocessing, EDA, train/val/test splits, repo management, **ViT fine-tuning**, **Grad-CAM explainability**, final submission packaging |
| **Goncalo** | Baseline & CNN Architect | Baseline CNN, deep custom CNN, hyperparameter tuning, architectural ablations |
| **Anastasiia** | Transfer Learning Specialist | Pretrained backbones (EfficientNetV2 / ResNet50), fine-tuning, data augmentation pipeline |
| **Leonor** | Evaluation & Analysis Lead | Metrics (accuracy, F1-macro), confusion matrices, error analysis, all result figures |
| **Henrique** | Report Writer | Full 7-page report, problem formalisation, methodology narrative, submission formatting |

---

## 🗓️ Project Timeline

| Week | Dates | Milestones | Owner(s) |
|------|-------|------------|----------|
| **Week 1** | Mar 17–23 | Repo setup, dataset download & EDA, baseline model running end-to-end | Miguel, Goncalo |
| **Week 2** | Mar 24–30 | Custom CNN tuned, transfer learning experiments started, error analysis begun, ViT setup | Goncalo, Anastasiia, Leonor, Miguel |
| **Week 3** | Mar 31–Apr 6 | All models benchmarked (incl. ViT), Grad-CAM heatmaps generated, report v1 drafted | Miguel, Anastasiia, Leonor, Henrique |
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

# Transfer learning (ResNet50)
python src/train.py --model transfer --epochs 20 --lr 1e-5 --dropout 0.5

# Vision Transformer (ViT-B/16 fine-tuned)
python src/train.py --model vit --epochs 20
```

### 5. Evaluate
```bash
python src/evaluate.py --model vit --checkpoint results/models/vit.keras
```

### 6. Generate Grad-CAM heatmaps
```bash
python src/gradcam.py --model vit --checkpoint results/models/vit.keras --output results/gradcam/
```

### 7. Launch the web app
```bash
python app/server.py
```
Open http://127.0.0.1:5000 — drag & drop paintings to predict the artist. Use the dropdown to switch between all trained models (Baseline CNN, Custom CNN, Transfer/ResNet50, ViT).

---

## 🏗️ Models Overview

### Baseline CNN (Goncalo)
A shallow convolutional network used as a performance floor. Establishes the training pipeline and confirms data is loading correctly before more complex models are introduced.

### Custom CNN (Goncalo)
A deeper architecture with multiple Conv blocks, Batch Normalisation, and Dropout, designed and tuned specifically for the WikiArt classification task.

### Transfer Learning (Anastasiia)
Fine-tuned pretrained ResNet50 backbone with a custom classification head. Five experiments varying fine-tuning depth (5–10 layers), learning rate (1e-3, 1e-5), and dropout (0.4–0.6). Uses ImageNet preprocessing, EarlyStopping, ReduceLROnPlateau, and the shared augmentation pipeline.

### Vision Transformer — ViT-B/16 (Miguel)
Fine-tuned pretrained Vision Transformer (ViT-B/16) with a custom classification head. The image is split into 16×16 patches, each treated as a token fed into multi-head self-attention blocks — allowing the model to capture long-range compositional relationships across a painting that CNNs inherently miss. Loaded via `keras-hub` from ImageNet-21K pretrained weights. Training uses AdamW with weight decay, cosine LR schedule with linear warmup, label smoothing, and an intermediate Dense(256, GELU) layer before the classifier.

---

## 📊 Evaluation Framework (Leonor)

All models are evaluated on the held-out test set using:

- **Accuracy** — overall and per-class
- **F1-macro** — accounts for class imbalance
- **Confusion matrix** — full and normalised
- **Learning curves** — training vs. validation loss/accuracy
- **Error analysis** — qualitative review of misclassified samples

All figures are saved to `results/figures/` with descriptive names (e.g. `vit_confusion_matrix_normalised.png`) for direct use in the report.

---

## 🔍 Explainability — Grad-CAM (Miguel)

Applied to the best-performing model across all four architectures. Grad-CAM computes the gradient of the predicted class score with respect to the final convolutional (or attention) layer's feature maps, producing a heatmap that highlights which regions of a painting were decisive for the classification.

This is applied to:
- **Correct predictions** — confirming the model attends to semantically meaningful features (brushwork, composition, colour palette)
- **Misclassified samples** — diagnosing failure modes and potential dataset biases
- **Cross-class comparisons** — visualising what distinguishes, e.g., Impressionism from Post-Impressionism in the model's representation

Heatmaps are saved to `results/gradcam/` with naming convention `gradcam_<class>_<correct|wrong>_<id>.png`.

---

## 🖥️ Web App — Interactive Classifier (Miguel)

A Flask-based web application for interactive artist prediction. Upload one or more paintings and get instant top-5 predictions with confidence scores.

**Features:**
- Drag-and-drop multi-image upload (JPG, PNG, BMP, TIFF)
- **Model selector** — switch between all trained models at runtime (Baseline CNN, Custom CNN, Transfer/ResNet50, ViT)
- Top-5 predictions per image with confidence bars
- Dark-themed responsive UI

**Run locally:**
```bash
python app/server.py                              # default: ViT model, port 5000
python app/server.py --model results/models/custom_cnn.keras --port 8080
```

**Architecture:** `app/server.py` (Flask backend with `/predict` and `/models/switch` endpoints) + `app/static/index.html` (vanilla HTML/CSS/JS frontend, no build step).

---

## 🤝 Git Workflow

1. Each member works on a dedicated branch: `feature/<topic>` (e.g. `feature/transfer-learning`, `feature/vit`, `feature/gradcam`)
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

Key dependencies (see `requirements.txt` for full list):

- `tensorflow >= 2.10`
- `keras`
- `keras-hub`             — ViT pretrained weights (ImageNet-21K)
- `keras-tuner`           — Hyperparameter search (Custom CNN)
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `Pillow`

---

## 📄 License

Academic project — NOVA IMS Deep Learning course, 2025/2026.