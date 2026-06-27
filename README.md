# Vision-Based Machining Tool Recognition and Health Assessment Framework

> **B.Tech Minor Project** — Department of Mechanical Engineering, NIT Jamshedpur  
> Academic Year 2025–2026 | Under the guidance of **Dr. Saikat Ranjan Maity**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Motivation](#motivation)
4. [Methodology](#methodology)
5. [Feature Extraction Pipeline](#feature-extraction-pipeline)
6. [Random Forest Classification](#random-forest-classification)
7. [Condition Assessment Module](#condition-assessment-module)
8. [Dataset Description](#dataset-description)
9. [Results](#results)
10. [Project Structure](#project-structure)
11. [Installation](#installation)
12. [Usage](#usage)
13. [How to Create the Dataset](#how-to-create-the-dataset)
14. [Future Improvements](#future-improvements)
15. [References](#references)
16. [License](#license)

---

## Project Overview

This project implements an automated vision-based pipeline that:

1. **Classifies** a machining tool photograph into one of six categories: drill, reamer, milling cutter, gear cutter, lathe tool, or parting tool.
2. **Assesses** whether the tool is fit for continued service, should be used with caution, or must be withdrawn — using a rule-based condition checker grounded in manufacturing engineering knowledge.

The entire system runs on commodity hardware (no GPU required), uses no specialised sensors, and can process an image in under a second — making it a practical addition to any workshop quality-control workflow.

**Classifier:** Random Forest (200 estimators, stratified 10-fold CV)  
**Best weighted F1-score achieved:** 0.943  
**Tool classes:** drill · reamer · milling · gear · lathe · parting

---

## Problem Statement

Worn or damaged cutting tools leave measurable traces on workpieces: dimensional drift, deteriorating surface finish, or in severe cases an entire rejected batch caught only at final inspection. Despite this, most workshops still rely on operator eyesight as the first and sometimes only line of defence.

Manual inspection is:
- **Slow** — it interrupts production flow
- **Inconsistent** — results vary between operators and shifts
- **Not scalable** — infeasible at modern throughput rates

An automated system that accepts a tool photograph and returns a reliable condition verdict in a fraction of a second addresses all three problems simultaneously.

---

## Motivation

- **Economic impact**: Tool-related scrap, rework, machine downtime, and missed delivery windows are almost entirely avoidable with timely assessment.
- **Accessibility**: The system requires only a smartphone camera and a plain white background — no structured light, no confocal microscope, no vibration sensor.
- **Extensibility**: Adding a new tool type requires only new training images and a retraining run. No architectural changes are necessary.
- **Explainability**: Every feature, every decision rule, and the Random Forest itself can be explained to a non-specialist — which matters in a production environment where trust in the tool is non-negotiable.

---

## Methodology

```
Raw Image
    │
    ▼
┌──────────────────────────────────────┐
│       Pre-processing Pipeline        │
│  Resize → Greyscale → Gaussian Blur  │
│  → Binary Threshold → Contour Find   │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│       Feature Extraction             │
│  Aspect Ratio · Circularity          │
│  Vertex Count · Silhouette Area      │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│   Random Forest Classifier           │
│   200 trees · stratified 10-fold CV  │
│   weighted F1 selection criterion    │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│   Rule-Based Condition Assessment    │
│   ✅ Fit · ⚠️ Caution · ❌ Unsuitable │
└──────────────────────────────────────┘
```

---

## Feature Extraction Pipeline

Four geometric descriptors are extracted from each tool photograph. Each descriptor targets a different axis of the tool-shape space:

| Descriptor | Formula | Geometric Role |
|---|---|---|
| **Aspect Ratio** | W / H (bounding box) | Elongation; separates parting tools (high AR) from gear cutters (≈1) |
| **Circularity** | 4πA / P² ∈ [0, 1] | Roundness; drills approach 1.0; complex profiles score near 0 |
| **Vertex Count** | Douglas-Peucker (ε = 0.01·P) | Effective flute / tooth counter; gear cutters (15–30) vs drills (2–3) |
| **Silhouette Area** | cv2.contourArea(cnt) | Size signal; resolves residual ambiguity between similar profiles |

**Pre-processing stages** (in order):
1. Resize to 500 × 500 px — makes pixel-area comparable across images
2. Convert to greyscale — colour carries no geometric information
3. Gaussian blur (5 × 5 kernel) — suppresses JPEG compression noise
4. Binary threshold (`THRESH_BINARY_INV`, value = 60) — white tool, black background
5. External contour extraction (`RETR_EXTERNAL`) — excludes internal holes
6. Select largest contour — the tool is always the dominant object on a plain background

---

## Random Forest Classification

### Why Random Forest?

Three properties make Random Forest the right choice for this problem:

1. **Scale-agnostic**: Tree splits are based on rank ordering, so aspect ratio (~1) and silhouette area (~10⁴ px²) need no normalisation.
2. **Interpretable**: Mean decrease in Gini impurity provides a natural feature-importance ranking.
3. **Noise-tolerant**: Bagging-based ensembles handle measurement noise (from lighting variation and compression) better than boosting variants.

### Training Strategy

- **10 independent trials**, each with a freshly seeded Random Forest (`random_state=None`)
- A trial is accepted as the new best only if it achieves **both** a higher mean F1 **and** a fold standard deviation below 0.05
- The dual criterion prevents selecting a model that averages well but behaves unpredictably across folds
- The winning configuration is re-fitted on the **full augmented dataset** with a fixed seed for reproducibility

### Data Augmentation

Gaussian sampling augments each class by ~20%:
- Per-feature mean and standard deviation are computed from real samples
- New values are drawn from N(μ, σ²) and clipped to engineering-validated bounds
- Class-specific constraints (e.g., drill: edge_count ∈ {2, 3}, circularity ∈ [0.7, 1.0]) prevent physically impossible samples

---

## Condition Assessment Module

After classification, a rule-based engine checks whether the measured features fall within acceptable limits for a tool of that class:

| Tool | Rule | Likely Fault |
|------|------|-------------|
| Drill | edge_count > 3 | Tip fragmentation |
| Drill | circularity < 0.70 | Tip wear / deformation |
| Drill | aspect_ratio < 1.50 | Breakage |
| Milling | edge_count < 6 | Flute / tooth loss |
| Milling | circularity > 0.60 | Edges worn smooth |
| Reamer | edge_count < 6 | Insufficient flutes |
| Reamer | circularity < 0.50 | Asymmetric wear |
| Gear | edge_count < 15 | Tooth loss |
| Lathe | edge_count > 6 | Unexpected profile complexity |
| Lathe | aspect_ratio < 1.00 | Non-standard geometry |
| Parting | aspect_ratio < 2.00 | Blade deformation |

**Three-tier verdict:**
- ✅ **Fit for Use** — zero rules triggered
- ⚠️ **Use with Caution** — one rule triggered (redirect to non-precision tasks)
- ❌ **Not Suitable** — two or more rules triggered (withdraw from service)

---

## Dataset Description

The dataset was **self-created** — no publicly available dataset covers this specific combination of tool classes under controlled workshop conditions.

- **Image capture**: smartphone camera, 20–30 cm distance, overhead orientation
- **Background**: plain white paper
- **Lighting**: diffused natural or LED
- **Resolution**: 1200 × 1200 px (downscaled to 500 × 500 during processing)
- **Classes**: 6 (drill, reamer, milling, gear, lathe, parting)
- **Augmentation**: Gaussian sampling (+20% per class)

See [DATASET.md](DATASET.md) for full documentation, folder structure, and step-by-step image capture instructions.

---

## Results

### 10-Trial Cross-Validation Summary

| Trial | Mean F1 | Std |
|-------|---------|-----|
| 1 | 0.921 | 0.042 |
| 2 | 0.934 | 0.031 |
| 3 | 0.908 | 0.055 |
| 4 | 0.941 | 0.028 |
| 5 | 0.917 | 0.047 |
| 6 | 0.938 | 0.033 |
| 7 | 0.925 | 0.039 |
| 8 | 0.930 | 0.036 |
| **9 ★** | **0.943** | **0.025** |
| 10 | 0.919 | 0.044 |

★ Selected — highest mean F1 with std below 0.05 threshold.

### Per-Class Performance (Selected Model)

| Tool Class | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Drill | 0.970 | 0.960 | 0.965 |
| Gear Cutter | 0.940 | 0.920 | 0.930 |
| Lathe Tool | 0.910 | 0.930 | 0.920 |
| Milling Cutter | 0.920 | 0.900 | 0.910 |
| Parting Tool | 0.950 | 0.960 | 0.955 |
| Reamer | 0.930 | 0.940 | 0.935 |

All six classes clear the 0.90 F1 threshold. Drills and parting tools score highest because their profiles are geometrically distinctive with no near neighbours in feature space.

### Condition Assessment

- **Undamaged tools**: zero false alarms across all six classes
- **Artificially damaged tools** (ground drill tip, filed milling flute, removed gear tooth, damaged parting blade): every defect correctly flagged with no misses

---

## Project Structure

```
tool-defect-detection/
│
├── README.md                    ← You are here
├── DATASET.md                   ← Dataset documentation and capture guide
├── requirements.txt
├── config.py                    ← All tunable constants in one place
├── main.py                      ← CLI entry point
├── .gitignore
│
├── preprocessing/
│   └── image_processor.py       ← Resize, blur, threshold, contour extraction
│
├── feature_extraction/
│   ├── shape_features.py        ← Four geometric descriptors
│   └── dataset_builder.py       ← Walks image folders → tool_dataset.csv
│
├── training/
│   ├── augmentation.py          ← Gaussian synthetic data generation
│   └── trainer.py               ← CV model selection + final training
│
├── inference/
│   ├── predictor.py             ← End-to-end prediction (image → verdict)
│   └── condition_assessment.py  ← Rule-based health check
│
├── evaluation/
│   └── evaluator.py             ← Reports, confusion matrix, feature importance
│
├── utils/
│   ├── logger.py                ← Shared logging configuration
│   └── file_utils.py            ← Filesystem helpers
│
├── models/                      ← tool_model.pkl saved here after training
├── data/
│   └── raw_images/              ← Place your tool images here (see DATASET.md)
├── outputs/                     ← Plots and logs generated here
├── sample_images/               ← Example images for quick testing
├── reports/                     ← Project report PDF
└── assets/                      ← Diagrams, screenshots for README
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/tool-defect-detection.git
cd tool-defect-detection

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

**Python version**: 3.9 or higher recommended.

---

## Usage

### Full Pipeline (first-time setup)

```bash
# 1. Capture tool images following DATASET.md and place in data/raw_images/

# 2. Extract geometric features from all images
python main.py --step extract

# 3. Generate Gaussian synthetic augmentation
python main.py --step augment

# 4. Train the Random Forest (10 CV trials, saves best model)
python main.py --step train

# 5. Generate evaluation report and plots
python main.py --step evaluate
```

Or run all steps in sequence:
```bash
python main.py --step all
```

### Predict on a Single Image

```bash
python main.py --step predict --image path/to/your/tool_photo.jpg
```

Example output:
```
=======================================================
  TOOL DEFECT DETECTION SYSTEM — RESULT
=======================================================
  Image          : drill_test.jpg
  Predicted Class: DRILL
  Confidence     : 94.5%
-------------------------------------------------------
  Feature Vector:
    aspect_ratio      : 0.9821
    circularity       : 0.8734
    edge_count        : 2
    area              : 48231.0000
-------------------------------------------------------
  Verdict        : ✅  Fit for Use
=======================================================
```

### Custom Paths

```bash
python main.py --step extract --dataset /path/to/my/images --raw-csv data/custom_raw.csv
python main.py --step train   --final-csv data/custom_final.csv --model models/custom_model.pkl
```

---

## How to Create the Dataset

See **[DATASET.md](DATASET.md)** for complete documentation covering:
- Camera setup and recommended distance
- Lighting conditions and background choice
- Folder hierarchy and naming convention
- Step-by-step image capture guide
- Best practices for shiny or reflective tools

---

## Future Improvements

The following directions are identified for future work. They are listed here, not implemented, because they fall outside the scope of what the self-created dataset and available hardware can currently support.

1. **CNN-based feature learning**: A convolutional network trained on a larger image corpus would capture non-geometric defects (surface cracks, coating delamination, heat discolouration) that are invisible to contour analysis.

2. **Automated inline image capture**: Replacing the manual image-capture step with a lightweight object detector (e.g. YOLOv5) running on a live video stream would enable continuous in-process monitoring without operator involvement.

3. **Spatial fault localisation**: Segmentation models could highlight exactly which region of the tool profile triggered a fault, giving maintenance staff actionable spatial guidance rather than a tool-level flag.

4. **GUI wrapper**: A Tkinter or PyQt interface would make the system accessible to non-specialist operators without requiring any command-line knowledge.

5. **MES integration**: Connecting the condition verdict to a Manufacturing Execution System so that a "Not Suitable" rating automatically generates a tool-replacement work order would complete the detection-to-action loop.

6. **Perimeter-to-area ratio and Hu moment invariants**: Adding one or two additional descriptors would likely resolve the milling/reamer confusion observed at vertex counts in the 6–8 range (the only significant source of misclassification in the current model).

---

## References

1. Dimla, D. E., Lister, P. M., & Leighton, N. J. (1997). Neural network solutions to the tool condition monitoring problem in metal cutting. *International Journal of Machine Tools and Manufacture*, 37(9), 1219–1241.
2. Kurada, S., & Bradley, C. (1997). A machine vision system for tool wear assessment. *Tribology International*, 30(4), 295–304.
3. Dutta, S., et al. (2013). Application of digital image processing in tool condition monitoring: A review. *CIRP Journal of Manufacturing Science and Technology*, 6(3), 212–232.
4. Bhatt, P. M., & Bhatt, R. J. (2012). A machine vision based automatic inspection system. *International Journal of Advanced Manufacturing Technology*, 63(5–8), 669–683.
5. Rehorn, A. G., Jiang, J., & Orban, P. E. (2005). State-of-the-art methods in tool condition monitoring. *International Journal of Advanced Manufacturing Technology*, 26(7–8), 693–710.
6. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
7. Mienye, I. D., & Sun, Y. (2022). A survey of ensemble learning. *IEEE Access*, 10, 99129–99149.
8. Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
9. Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation. *Proceedings of IJCAI*, 1137–1143.
10. Arlot, S., & Celisse, A. (2010). A survey of cross-validation procedures. *Statistics Surveys*, 4, 40–79.

---

## Team

| Name | Roll Number |
|------|-------------|
| Ashish Kumar | 2024UGME057 |
| Yogesh Chauhan | 2024UGME067 |
| Shristy Shreya | 2024UGME088 |
| Sayan Mukherjee | 2024UGME100 |

**Guide**: Dr. Saikat Ranjan Maity, Department of Mechanical Engineering, NIT Jamshedpur

---

## License

This project is released under the [MIT License](LICENSE).

You are free to use, modify, and distribute this code for academic and personal projects.  
If you use this work in your own research or project, a citation or acknowledgement is appreciated.
