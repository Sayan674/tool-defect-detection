# Vision-Based Machining Tool Recognition and Health Assessment Framework

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)
![Status](https://img.shields.io/badge/Status-Proof--of--Concept-f59e0b?style=flat-square)

> **B.Tech Minor Project** — Department of Mechanical Engineering, NIT Jamshedpur, 2025–2026  
> Under the guidance of **Dr. Saikat Ranjan Maity**

---

## Overview

Industrial machining depends critically on the condition of cutting tools. A worn drill tip, a milling cutter with a missing flute, or a gear cutter that has shed a tooth each leave traces on the workpiece — dimensional drift, deteriorating surface finish, or in the worst case an entire rejected batch caught only at final inspection. Yet most workshops still rely on operator eyesight as the primary screening method, an approach that is slow, inconsistent between shifts, and not scalable to modern production rates.

This project builds an end-to-end machine learning pipeline that accepts a tool photograph, extracts four geometric shape descriptors using classical computer vision, classifies the tool using a Random Forest ensemble, and then passes the classification and measured features to a rule-based condition assessor that returns an actionable maintenance verdict.

### Scope and Design Philosophy

Industrial machining involves hundreds of distinct cutting tool families, each with unique geometries and failure characteristics. Rather than attempting to solve the full problem at once — which would require a large, professionally annotated dataset and substantially more engineering effort — this project intentionally focuses on **six representative cutting tool categories** as a proof-of-concept.

The objective was to first validate the complete end-to-end pipeline on a well-understood, representative subset before expanding to broader tool families. Limiting scope improves reliability, simplifies validation, and creates a stronger engineering foundation for future extension. The framework is explicitly designed to be modular: classification, feature extraction, augmentation, and condition assessment are all independent components that can be extended without restructuring the codebase.

**Supported tool categories:**

| Label | Tool Type | Key Geometric Property |
|---|---|---|
| `drill` | Twist Drill | Near-circular profile, vertex count 2–3 |
| `reamer` | Reamer | Symmetrical flutes, vertex count 6–10 |
| `milling` | Milling Cutter | Variable flute count, vertex count 6–15 |
| `gear` | Gear Cutter | High tooth density, vertex count 15–30 |
| `lathe` | Lathe Tool / Insert | Simple angular profile, vertex count 3–6 |
| `parting` | Parting Tool / Blade | High aspect ratio, vertex count 2–4 |

---

## Architecture

The system is composed of five decoupled stages, each implemented as a standalone module:

```
Raw Tool Image
      │
      ▼
┌─────────────────────────────────────┐
│  preprocessing/image_processor.py  │
│  Resize → Greyscale → Gaussian Blur │
│  → Binary Threshold → Contour Find  │
└─────────────────────────────────────┘
      │  largest external contour
      ▼
┌─────────────────────────────────────┐
│  feature_extraction/shape_features  │
│  Aspect Ratio  ·  Circularity       │
│  Vertex Count  ·  Silhouette Area   │
└─────────────────────────────────────┘
      │  4-element feature vector
      ▼
┌─────────────────────────────────────┐
│  training/trainer.py                │
│  Random Forest  ·  200 estimators   │
│  10-trial CV selection  ·  joblib   │
└─────────────────────────────────────┘
      │  predicted class + feature vector
      ▼
┌─────────────────────────────────────┐
│  inference/condition_assessment.py  │
│  Rule-based geometry checks         │
│  per supported tool category        │
└─────────────────────────────────────┘
      │
      ▼
  ✅ Fit for Use
  ⚠️  Use with Caution
  ❌  Not Suitable
```

---

## End-to-End Workflow

This section describes the complete journey from raw images to a maintenance verdict for a first-time user.

**Step 1 — Collect tool images.**  
Capture photographs of one or more of the six supported cutting tool categories using a smartphone camera or DSLR. Each image should show a single tool on a plain white background under even lighting. See [DATASET.md](DATASET.md) for detailed capture guidelines.

**Step 2 — Organise into class folders.**  
Place images into per-class sub-folders inside `data/raw_images/`. Each folder name must exactly match one of the supported class labels:

```
data/raw_images/
├── drill/
├── milling/
├── gear/
└── ...
```

**Step 3 — Extract features.**  
Run the feature extraction step. The pipeline processes every image through the four-stage CV pre-processing pipeline, computes the four geometric descriptors, and writes the results to `data/tool_dataset.csv`:

```bash
python main.py --step extract
```

**Step 4 — Augment the dataset.**  
Apply Gaussian-based synthetic augmentation to add approximately 20% additional samples per class, producing `data/final_dataset.csv`:

```bash
python main.py --step augment
```

**Step 5 — Train the classifier.**  
Run ten independent cross-validation trials to identify the most stable Random Forest configuration. The winning model is re-fitted on the full augmented dataset and saved to `models/tool_model.pkl`:

```bash
python main.py --step train
```

> **Note:** Prediction requires a trained model. Always run `--step train` before `--step predict`.

**Step 6 — Evaluate the model.**  
Generate the full evaluation report for your dataset. The evaluation module automatically analyses the trained model and writes reports and visualisations to `outputs/`:

```bash
python main.py --step evaluate
```

**Step 7 — Predict on a new image.**  
Pass any previously unseen tool photograph to the pipeline:

```bash
python main.py --step predict --image path/to/tool_photo.jpg
```

During prediction, the system automatically:

1. Loads the saved model from `models/tool_model.pkl`
2. Pre-processes the image (resize → greyscale → Gaussian blur → binary threshold → contour extraction)
3. Extracts the four geometric features from the largest external contour
4. Queries the Random Forest to identify the supported tool category
5. Passes the predicted class and extracted features to the maintenance assessment module
6. Returns the detected tool type together with a three-tier maintenance verdict: **Fit for Use**, **Use with Caution**, or **Not Suitable**

**Shortcut — run the full pipeline in one command:**

```bash
python main.py --step all
```

This executes extract → augment → train → evaluate in sequence.

---

## Two Distinct Subsystems

Understanding the distinction between classification and condition assessment is important for understanding this project's scope.

### 1. Classification (data-driven)

The Random Forest classifier is trained dynamically on whatever dataset the user provides. It learns from the images you capture and adapts to your data. If you provide images for only two of the six supported classes, the model trains on those two classes.

### 2. Maintenance Assessment (rule-driven)

The condition assessment module is **not** a learned model. It is a deterministic rule engine that checks whether a tool's measured geometric features fall within acceptable limits for its identified class.

These rules encode manufacturing engineering knowledge about specific failure modes:

- A worn drill tip loses circularity and may gain spurious contour vertices
- A milling cutter with broken flutes shows a reduced vertex count
- A parting blade that has been deformed loses its characteristically high aspect ratio

This module currently has rules defined specifically for the six supported tool categories. **Supporting additional tool families requires defining their expected geometric ranges and failure characteristics** — the engineering knowledge, not just the training images. This is an intentional design constraint, not a limitation of the code architecture.

All rule thresholds live in `config.py` under `CONDITION_RULES`, allowing a domain expert to adjust them without touching the logic in `inference/condition_assessment.py`.

---

## Feature Extraction Pipeline

Four geometric shape descriptors are computed from each tool silhouette. Each descriptor targets a different axis of the tool-shape space and together they span it without redundancy.

| Descriptor | Formula | Role |
|---|---|---|
| **Aspect Ratio** | W / H (bounding box) | Elongation — separates parting tools (high) from gear cutters (≈1) |
| **Circularity** | 4πA / P² ∈ [0, 1] | Roundness — drills approach 1.0; complex profiles score near 0 |
| **Vertex Count** | Douglas-Peucker approximation (ε = 0.01 · P) | Effective flute/tooth counter |
| **Silhouette Area** | `cv2.contourArea(contour)` | Size signal — resolves residual ambiguity between similar profiles |

**Pre-processing stages (in order):**

1. **Resize to 500 × 500 px** — makes pixel-area comparable across images taken at different distances
2. **Convert to greyscale** — colour carries no geometric information for this task
3. **Gaussian blur (5 × 5 kernel)** — suppresses JPEG compression noise without blurring genuine profile corners
4. **Binary threshold** (`THRESH_BINARY_INV`, value = 60) — produces a white tool silhouette on a black background
5. **External contour extraction** (`RETR_EXTERNAL`) — excludes internal holes (e.g. reamer bore)
6. **Select largest contour** — on a plain background the tool is always the dominant object

### Why handcrafted features rather than CNN features?

CNNs learn their own feature representations from large labelled datasets. This project uses a self-captured dataset that is, by necessity, small. Handcrafted geometric features are appropriate here for two reasons:

1. They encode *a priori* knowledge about what actually distinguishes these tool classes — the discriminating geometry is known in advance, so there is no need to learn it from data.
2. They are fully interpretable: every feature can be visualised, its value inspected, and its effect on the classification decision traced through the feature-importance ranking produced by the Random Forest.

CNN-based feature extraction is a natural next step once a larger dataset exists and is listed explicitly in [Future Work](#future-work).

---

## Classifier Design

### Why Random Forest?

Three properties make Random Forest the appropriate choice here:

1. **Scale-agnostic.** Tree splits are based on rank ordering within a feature, so `aspect_ratio` (order ~1) and `silhouette_area` (order ~10⁴ px²) require no normalisation before training.
2. **Interpretable.** The mean decrease in Gini impurity is a byproduct of training, producing a natural feature-importance ranking that confirms the classifier is reasoning about geometry sensibly — not latching onto spurious correlates.
3. **Noise-tolerant.** Bagging-based ensembles handle measurement noise better than boosting variants. Lighting variation and JPEG compression both affect extracted feature values; a classifier that degrades gracefully under this noise is essential.

### Two-Phase Training Strategy

**Phase 1 — Model selection:** Ten independent trials are run, each with a freshly seeded Random Forest (`random_state=None`, ensuring genuine diversity between trials). Each trial is evaluated using stratified cross-validation and scored on a weighted basis across all folds. A trial is accepted as the new best candidate only when it simultaneously achieves a higher cross-validation score *and* demonstrates stable, consistent performance across folds. The dual criterion matters: a model that averages well but oscillates across folds cannot be trusted in deployment.

**Phase 2 — Final training:** The winning configuration is re-fitted on the complete augmented dataset with a fixed random seed (`random_state=42`) for full reproducibility, then serialised to `models/tool_model.pkl` with `joblib`.

### Data Augmentation

Because the dataset is self-captured and necessarily small, Gaussian sampling is used to generate approximately 20% additional samples per class. For each class, per-feature mean (μ) and standard deviation (σ) are computed from the real samples. New values are drawn from N(μ, σ²) and clipped to engineering-validated bounds that prevent physically impossible feature combinations (e.g. a drill with 15 vertices, or a parting tool with aspect ratio < 2).

This approach is simpler and more transparent than SMOTE or GAN-based synthesis, and every generated sample can be inspected without any specialist knowledge of generative modelling.

---

## Dataset

The dataset was self-captured — no publicly available image dataset covers this specific combination of tool classes under controlled workshop conditions. See [DATASET.md](DATASET.md) for the full documentation including camera setup, lighting requirements, folder structure, naming convention, and step-by-step image capture instructions.

### Bringing Your Own Dataset

You can substitute your own images. Your dataset must contain images organised into per-class sub-folders, where each folder name matches one of the six supported labels (`drill`, `reamer`, `milling`, `gear`, `lathe`, `parting`).

Valid dataset configurations:

| Configuration | Supported |
|---|---|
| All 6 classes | ✅ |
| Any subset of the 6 classes (e.g. drill + milling only) | ✅ |
| Tool categories outside the supported 6 | ⚠️ Classification will train, but condition assessment rules do not exist for unsupported classes |

The classifier adapts to whatever classes are present in your dataset folder. The condition assessment module will skip the health check and return "Fit for Use" for any class that does not have defined rules in `config.py`.

---

## Evaluation

The project includes a complete, standalone evaluation pipeline (`evaluation/evaluator.py`) that can be run independently of training at any time. After training, run:

```bash
python main.py --step evaluate
```

The evaluation module automatically analyses the trained model against the user's dataset and generates the following reports and visualisations, all saved to `outputs/`:

- **Classification Report** — a per-class breakdown of how well the model distinguishes each supported tool category, saved to the console
- **Confusion Matrix** — a normalised matrix showing where the model agrees and disagrees with the ground truth, saved as `outputs/confusion_matrix.png`
- **Feature Importance Plot** — a bar chart showing the relative contribution of each geometric descriptor to the classifier's decisions, saved as `outputs/feature_importance.png`
- **Training Logs** — timestamped logs for every pipeline run, written to `outputs/logs/`

Evaluation results are **not published here** because the framework evaluates user-provided datasets. Model behaviour depends directly on the dataset used — image quality, lighting consistency, class balance, and the number of images per class all affect the outcome. Run the evaluation step after training to obtain the results for your specific dataset.

---

## Sample Output

### CLI Prediction Output

```
=======================================================
  TOOL DEFECT DETECTION SYSTEM — RESULT
=======================================================
  Image          : drill_test_01.jpg
  Predicted Class: DRILL
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

### Condition Fault Example — Worn Drill

```
=======================================================
  TOOL DEFECT DETECTION SYSTEM — RESULT
=======================================================
  Image          : drill_worn_tip.jpg
  Predicted Class: DRILL
-------------------------------------------------------
  Feature Vector:
    aspect_ratio      : 1.0312
    circularity       : 0.5801
    edge_count        : 5
    area              : 46887.0000
-------------------------------------------------------
  Verdict        : ❌  Not Suitable
  Fault Details  :
    • Extra contour vertices detected (edge_count=5 > 3)
      — likely tip fragmentation or surface damage
    • Low circularity (0.580 < 0.70)
      — tip deformation or asymmetric wear
=======================================================
```

### Pipeline Log (verbose mode)

```
[2026-06-01 14:03:11] INFO     main — === STEP: Feature Extraction ===
[2026-06-01 14:03:11] INFO     feature_extraction.dataset_builder — Building raw dataset from: data/raw_images
[2026-06-01 14:03:11] INFO     feature_extraction.dataset_builder — Found 6 class folders: ['drill', 'gear', 'lathe', 'milling', 'parting', 'reamer']
[2026-06-01 14:03:12] INFO     feature_extraction.dataset_builder — Raw dataset saved → data/tool_dataset.csv
[2026-06-01 14:03:12] INFO     main — === STEP: Data Augmentation ===
[2026-06-01 14:03:12] INFO     training.augmentation — Augmentation complete. Final dataset saved → data/final_dataset.csv
[2026-06-01 14:03:12] INFO     main — === STEP: Training ===
[2026-06-01 14:03:13] INFO     training.trainer — Trial  1/10 — completed
[2026-06-01 14:03:14] INFO     training.trainer — Trial  2/10 — completed
[2026-06-01 14:03:15] INFO     training.trainer — Trial  3/10 — completed  ★ New best model selected
...
[2026-06-01 14:03:18] INFO     training.trainer — Selecting best candidate from completed trials ...
[2026-06-01 14:03:18] INFO     training.trainer — Re-training on full dataset ...
[2026-06-01 14:03:18] INFO     training.trainer — Model saved → models/tool_model.pkl
```

### Evaluation Outputs

After running `python main.py --step evaluate`, the following files are generated in `outputs/`:

| File | Description |
|---|---|
| `confusion_matrix.png` | Normalised confusion matrix visualisation |
| `feature_importance.png` | Bar chart of relative feature contributions |
| `cv_summary.csv` | Per-trial training summary |
| `logs/` | Timestamped log files for every pipeline run |

---

## Project Structure

```
tool-defect-detection/
│
├── README.md                         ← This file
├── DATASET.md                        ← Dataset documentation and capture guide
├── requirements.txt
├── config.py                         ← All tunable constants in one place
├── main.py                           ← Unified CLI entry point
├── LICENSE
│
├── preprocessing/
│   └── image_processor.py            ← Resize, blur, threshold, contour extraction
│
├── feature_extraction/
│   ├── shape_features.py             ← Four geometric descriptor functions
│   └── dataset_builder.py            ← Walks image folders → tool_dataset.csv
│
├── training/
│   ├── augmentation.py               ← Gaussian synthetic sample generation
│   └── trainer.py                    ← Two-phase CV model selection + final training
│
├── inference/
│   ├── predictor.py                  ← Predictor class: image path → PredictionResult
│   └── condition_assessment.py       ← Rule-based three-tier health verdict
│
├── evaluation/
│   └── evaluator.py                  ← Classification report, confusion matrix, feature importance
│
├── utils/
│   ├── logger.py                     ← Shared logging configuration
│   └── file_utils.py                 ← Filesystem helpers with meaningful error messages
│
├── models/                           ← tool_model.pkl saved here after training
├── data/
│   └── raw_images/                   ← Place your tool images here (see DATASET.md)
├── outputs/                          ← Plots and logs generated here
└── sample_images/                    ← Example images for quick testing
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/tool-defect-detection.git
cd tool-defect-detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

**Python version:** 3.9 or higher.

---

## Usage

### Full Pipeline (first-time setup)

```bash
# 1. Capture tool images following DATASET.md and place them in:
#    data/raw_images/<class_name>/
#    e.g. data/raw_images/drill/, data/raw_images/milling/

# 2. Extract geometric features from all images → data/tool_dataset.csv
python main.py --step extract

# 3. Generate Gaussian synthetic augmentation → data/final_dataset.csv
python main.py --step augment

# 4. Train the Random Forest (10 CV trials, saves best model)
python main.py --step train

# 5. Evaluate the model (report + plots saved to outputs/)
python main.py --step evaluate
```

Run all steps in one command:

```bash
python main.py --step all
```

### Predict on a Single Image

> **Note:** A trained model must exist at `models/tool_model.pkl` before running prediction. Always run `--step train` first.

```bash
python main.py --step predict --image path/to/tool_photo.jpg
```

When this command is executed, the following steps happen automatically in sequence:

1. **Load trained model** — the serialised Random Forest is loaded from `models/tool_model.pkl`
2. **Read the image** — the tool photograph is read from the path provided
3. **Preprocess the image** — the image is resized to 500 × 500 px, converted to greyscale, smoothed with a Gaussian blur, and segmented into a binary mask using `THRESH_BINARY_INV`
4. **Extract contour** — external contours are detected using `RETR_EXTERNAL`; the largest contour is selected as the tool outline
5. **Compute geometric features** — aspect ratio, circularity, vertex count (Douglas-Peucker), and silhouette area are computed from the contour
6. **Predict tool category** — the four-element feature vector is passed to the Random Forest, which returns the predicted supported tool category
7. **Apply maintenance assessment** — the predicted class and extracted features are passed to the rule-based condition assessment module, which checks the measured geometry against class-specific thresholds
8. **Display verdict** — the detected tool type and three-tier maintenance verdict (**Fit for Use**, **Use with Caution**, or **Not Suitable**) are printed to the terminal, along with any triggered fault descriptions

### Use Custom Paths

```bash
# Custom dataset location
python main.py --step extract --dataset /path/to/my/images

# Custom output paths
python main.py --step train --final-csv data/my_dataset.csv --model models/my_model.pkl
```

### CLI Reference

```
usage: main.py --step {extract,augment,train,evaluate,predict,all} [options]

Options:
  --step       Pipeline step to execute (required)
  --dataset    Root directory of per-class image folders
  --raw-csv    Output path for raw feature CSV
  --final-csv  Output path for augmented dataset CSV
  --model      Output path for trained model (.pkl)
  --image      Image path for --step predict
```

---

## Configuration

All tunable constants are collected in `config.py`. Adjust these to adapt the pipeline to different image capture setups or datasets — no changes to source code are required.

| Parameter | Default | Description |
|---|---|---|
| `IMAGE_SIZE` | `(500, 500)` | Resize target for all images |
| `BLUR_KERNEL` | `(5, 5)` | Gaussian blur kernel size |
| `THRESHOLD_VALUE` | `60` | Binary segmentation threshold |
| `DP_EPSILON_RATIO` | `0.01` | Douglas-Peucker tolerance as fraction of perimeter |
| `RF_N_ESTIMATORS` | `200` | Number of trees in the Random Forest |
| `CV_N_SPLITS` | `10` | Number of folds for stratified CV |
| `CV_N_TRIALS` | `10` | Number of independent training trials |
| `CV_MAX_STD_F1` | `0.05` | Stability threshold for cross-validation model acceptance |
| `AUGMENTATION_RATIO` | `0.20` | Fraction of synthetic samples per class |

Condition assessment thresholds are defined per class under `CONDITION_RULES` in `config.py`.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | ≥ 4.8.0 | Image loading, preprocessing, contour extraction |
| `numpy` | ≥ 1.24.0 | Numerical operations |
| `scikit-learn` | ≥ 1.3.0 | Random Forest, cross-validation, evaluation metrics |
| `joblib` | ≥ 1.3.0 | Model serialisation |
| `pandas` | ≥ 2.0.0 | Feature CSV management |
| `matplotlib` | ≥ 3.7.0 | Confusion matrix and feature importance plots |
| `seaborn` | ≥ 0.13.0 | Plot styling |

No GPU is required. All computation runs on CPU.

---

## Known Limitations

- **Fixed tool categories.** Classification and condition assessment are currently defined for six tool classes. Introducing new categories requires both new training images and new condition-assessment rules encoding the expected failure characteristics of those tools.
- **Plain background requirement.** The binary segmentation step (`THRESH_BINARY_INV`, threshold = 60) assumes the tool is darker than the background. Images captured on textured, patterned, or near-black backgrounds will produce unreliable contours. See [DATASET.md](DATASET.md) for image capture guidelines.
- **Single-tool images.** The pipeline selects the largest contour in the image as the tool outline. Images containing multiple tools or significant background objects will produce incorrect feature vectors.
- **No surface-texture defect detection.** Contour-based features describe the silhouette shape. Defects that do not change the outer profile — surface cracks, coating delamination, heat discolouration — are invisible to the current feature set.

---

## Future Work

The following directions represent natural extensions to this framework:

- **Additional tool categories.** Define geometric augmentation bounds and condition assessment rules for additional cutting tool families (boring bars, form tools, thread mills, broaches) and retrain.
- **CNN-based feature learning.** Replace handcrafted geometric descriptors with features learned by a convolutional network. This would capture non-geometric defects (surface cracks, coating wear) currently invisible to contour analysis. The primary prerequisite is a substantially larger labelled dataset.
- **Vision Transformer (ViT) backbone.** ViT-based encoders have shown strong results on industrial inspection tasks and represent a natural progression once data volume allows.
- **Explainable AI (SHAP / LIME).** Add post-hoc explainability on top of the Random Forest to produce per-prediction feature attribution reports useful in audit contexts.
- **Real-time inline inspection.** Replace the manual image-capture step with a lightweight object detector (e.g. YOLOv5) running on a live camera feed to enable continuous in-process monitoring without operator involvement.
- **Segmentation-based fault localisation.** Rather than flagging a fault at the tool level, instance segmentation could highlight the specific region of the tool profile where the anomaly was detected.
- **Web or edge deployment.** Wrap the inference pipeline in a REST API (FastAPI) or deploy to an edge device (Raspberry Pi, NVIDIA Jetson) for integration into existing workshop infrastructure.
- **MES integration.** Connect condition verdicts to a Manufacturing Execution System so that a "Not Suitable" rating automatically generates a tool-replacement work order.
- **Larger benchmark datasets.** Partner with industrial collaborators to build a publicly available, annotated image dataset covering a wider range of tool types and wear conditions — currently the most significant bottleneck to expanding the system's scope.

---

## References

1. Dimla, D. E., Lister, P. M., & Leighton, N. J. (1997). Neural network solutions to the tool condition monitoring problem. *Int. J. Machine Tools and Manufacture*, 37(9), 1219–1241.
2. Kurada, S., & Bradley, C. (1997). A machine vision system for tool wear assessment. *Tribology International*, 30(4), 295–304.
3. Dutta, S., et al. (2013). Application of digital image processing in tool condition monitoring. *CIRP J. Manufacturing Science and Technology*, 6(3), 212–232.
4. Bhatt, P. M., & Bhatt, R. J. (2012). A machine vision based automatic inspection system. *Int. J. Advanced Manufacturing Technology*, 63(5–8), 669–683.
5. Rehorn, A. G., Jiang, J., & Orban, P. E. (2005). State-of-the-art methods in tool condition monitoring. *Int. J. Advanced Manufacturing Technology*, 26(7–8), 693–710.
6. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
7. Mienye, I. D., & Sun, Y. (2022). A survey of ensemble learning. *IEEE Access*, 10, 99129–99149.
8. Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321–357.
9. Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation. *Proc. IJCAI*, 1137–1143.
10. Arlot, S., & Celisse, A. (2010). A survey of cross-validation procedures. *Statistics Surveys*, 4, 40–79.

---

## License

Released under the [MIT License](LICENSE). Free to use, modify, and distribute for academic and personal projects. A citation or acknowledgement is appreciated if this work contributes to your own research.
