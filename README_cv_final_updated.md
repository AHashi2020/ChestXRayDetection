# CV Final Project: Chest X-Ray Classification Under Domain Shift

## Overview

This project studies chest X-ray classification under **domain shift**, where a model trained on one dataset can lose performance when evaluated on images from a different hospital system or imaging pipeline.

The project is intentionally designed as a careful, student-scale experimental pipeline rather than a brand-new model architecture. The main workflow is:

- train on **NIH ChestXray14**
- validate internally on NIH
- evaluate externally on **VinDr-CXR**
- compare **binary**, **14-label multi-label**, and **grouped-label** settings
- test **entropy-gated test-time adaptation (TTA)** as a lightweight robustness method

The main goal is to measure how much performance changes under external shift and whether cautious deployment-time adaptation helps.

---

## Main Research Questions

This project asks:

1. Can a chest X-ray model trained on NIH weak labels generalize to an external dataset?
2. How much does performance drop when moving from NIH to VinDr-CXR?
3. Does entropy-gated test-time adaptation improve robustness under domain shift?
4. Does using a **smaller grouped label system** work better than the original 14-label setup for external evaluation and TTA?

---

## Datasets

### 1. NIH ChestXray14
Used for training and internal validation.

- Image format: PNG
- Labels: report-derived weak labels
- Split strategy: **patient-level split** to reduce leakage

### 2. VinDr-CXR test set
Used for external evaluation.

Why it matters:
- external to NIH
- different clinical source
- DICOM rather than PNG
- better test of true cross-dataset robustness

Because only a local subset may be downloaded, the VinDr scripts filter evaluation to images that are both present and readable.

---

## Prediction Setups

## 1. Binary setup
Binary task:
- `No Finding` = normal
- any other finding = abnormal

This is the simplest setup and serves as the main baseline for source-vs-external generalization.

## 2. Original 14-label multi-label setup
The original NIH multi-label pipeline predicts these labels:

- Atelectasis
- Cardiomegaly
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- Edema
- Emphysema
- Fibrosis
- Pleural_Thickening
- Hernia

If an image is labeled `No Finding`, all disease columns are 0.

## 3. New grouped 5-label setup
A new grouped-label system was added to reduce sparsity, improve external overlap, and make TTA less noisy.

The grouped labels are:

- `Airspace_Opacity`
  - Atelectasis
  - Infiltration
  - Consolidation
  - Pneumonia
  - Edema

- `Cardiomediastinal_Abnormality`
  - Cardiomegaly

- `Pleural_Abnormality`
  - Effusion
  - Pneumothorax
  - Pleural_Thickening

- `Focal_Lesion`
  - Mass
  - Nodule

- `Chronic_Parenchymal_Change`
  - Emphysema
  - Fibrosis

`Hernia` is dropped from the grouped setup because it is rare and does not map cleanly to the external VinDr evaluation.

This grouped system is meant to:
- reduce rare-label instability
- improve class frequency
- create cleaner external overlap with VinDr
- make TTA more stable than the original 14-label setup

---

## Project Structure

## Binary NIH pipeline

### `prepareNIH.py`
Builds the binary NIH train/validation split.

What it does:
- reads `NIH/Data_Entry_2017.csv`
- finds image files
- creates binary labels
- splits by patient ID
- saves:
  - `train.csv`
  - `val.csv`

### `trainNIH.py`
Trains the binary normal-vs-abnormal model.

Model setup:
- pretrained DenseNet-121
- final classifier replaced with 1 output
- head training followed by fine-tuning

Outputs:
- `training_outputs/checkpoint_latest.pth`
- `training_outputs/best_model.pth`
- `training_outputs/final_model.pth`
- `training_outputs/history.csv`
- `training_outputs/epoch_models/`

### `test_vindr_binary.py`
Runs external binary evaluation on VinDr-CXR.

What it does:
- loads `training_outputs/best_model.pth`
- reads VinDr DICOM images
- maps VinDr labels into:
  - `0 = normal`
  - `1 = abnormal`
- saves:
  - `vindr_binary_predictions.csv`

---

## Original 14-label NIH multi-label pipeline

### `prepareNIH_labels.py`
Builds the NIH 14-label train/validation split.

What it does:
- reads NIH metadata
- parses finding strings into 14 disease columns
- splits by patient ID
- saves:
  - `train_labels.csv`
  - `val_labels.csv`

### `trainNIH_labels.py`
Trains the 14-label NIH classifier.

Model setup:
- pretrained DenseNet-121
- final classifier replaced with 14 outputs
- loss: `BCEWithLogitsLoss`
- class imbalance handled with `pos_weight`
- validation thresholds saved per label

Outputs:
- `training_outputs_labels/checkpoint_latest.pth`
- `training_outputs_labels/best_model.pth`
- `training_outputs_labels/final_model.pth`
- `training_outputs_labels/history.csv`
- `training_outputs_labels/epoch_models/`
- `training_outputs_labels/best_thresholds.json`

### `test_vindr_labels.py`
Runs external multi-label evaluation on VinDr-CXR.

What it does:
- loads `training_outputs_labels/best_model.pth`
- loads per-label thresholds
- evaluates only labels with reasonable NIH ↔ VinDr overlap
- saves:
  - `vindr_labels_predictions.csv`
  - `vindr_labels_metrics.csv`

Important note:
This is **overlap-based evaluation**, not perfect one-to-one label transfer.

Examples:
- NIH `Effusion` → VinDr `Pleural effusion`
- NIH `Fibrosis` → VinDr `Pulmonary fibrosis`
- NIH `Mass` + `Nodule` → VinDr `Nodule/Mass`
- `Hernia` is excluded

---

## New grouped 5-label pipeline

### `prepareNIH_less_labels.py`
Builds the grouped NIH train/validation split.

What it does:
- reads NIH metadata
- first parses original NIH findings
- then maps them into the 5 grouped labels
- splits by patient ID
- saves:
  - `train_less_labels.csv`
  - `val_less_labels.csv`

### `trainNIH_less_labels.py`
Trains the grouped 5-label classifier.

Model setup:
- pretrained DenseNet-121
- final classifier replaced with 5 outputs
- loss: `BCEWithLogitsLoss`
- class imbalance handled with `pos_weight`
- per-label thresholds selected on NIH validation

Outputs:
- `training_outputs_less_labels/checkpoint_latest.pth`
- `training_outputs_less_labels/best_model.pth`
- `training_outputs_less_labels/final_model.pth`
- `training_outputs_less_labels/history.csv`
- `training_outputs_less_labels/epoch_models/`
- `training_outputs_less_labels/best_thresholds.json`

### `test_vindr_less_labels.py`
Runs grouped external evaluation on VinDr-CXR.

Grouped VinDr mapping:
- `Airspace_Opacity` ← Atelectasis / Infiltration / Consolidation / Pneumonia / Edema
- `Cardiomediastinal_Abnormality` ← Cardiomegaly / Enlarged cardiac silhouette
- `Pleural_Abnormality` ← Pleural effusion / Effusion / Pneumothorax / Pleural thickening
- `Focal_Lesion` ← Nodule/Mass / Nodule / Mass
- `Chronic_Parenchymal_Change` ← Emphysema / Pulmonary fibrosis / Fibrosis

Outputs:
- `vindr_less_labels_predictions.csv`
- `vindr_less_labels_metrics.csv`

---

## Entropy-Gated Test-Time Adaptation

This project adds **entropy-gated test-time adaptation (TTA)** as a lightweight deployment-time robustness method.

### Motivation
A model trained on NIH may become less reliable on VinDr because of domain shift. Instead of retraining from scratch or using a heavy domain adaptation pipeline, the project tests whether cautious adaptation at inference time helps.

### Core idea
At test time:

1. run the source model on external VinDr batches
2. compute prediction entropy
3. accept only low-entropy samples for adaptation
4. ignore high-entropy samples
5. update only **BatchNorm parameters**
6. compare TTA predictions against the source-only baseline

### Why this is cautious
- only confident samples are used
- only BN parameters are updated
- adaptation is done episodically in the current setup to reduce drift

### `tta_utils.py`
Shared helper functions for TTA.

Includes:
- binary entropy computation
- multi-label entropy computation
- threshold estimation from NIH validation
- BatchNorm-only TTA setup
- binary and multi-label TTA step helpers
- checkpoint/image utilities

---

## Binary TTA

### `test_vindr_binary_tta.py`
Runs binary VinDr evaluation with and without TTA.

What it does:
- loads `training_outputs/best_model.pth`
- computes or reuses entropy threshold from NIH validation
- reuses saved baseline predictions when available
- runs binary TTA on VinDr
- saves:
  - `vindr_binary_predictions_baseline.csv`
  - `vindr_binary_predictions_tta.csv`
  - `vindr_binary_tta_log.csv`
  - `vindr_binary_comparison.csv`
  - `vindr_binary_metrics_comparison.csv`
  - `vindr_binary_entropy_threshold.json`

### Binary TTA result
Current binary TTA result is **mixed**:
- recall improved
- F1 improved
- precision decreased
- ROC-AUC and average precision decreased

So binary TTA changed the error tradeoff, but it was not a clean overall win.

---

## 14-label multi-label TTA

### `test_vindr_labels_tta.py`
Runs multi-label VinDr evaluation with and without TTA.

What it does:
- loads `training_outputs_labels/best_model.pth`
- loads `training_outputs_labels/best_thresholds.json`
- computes or reuses entropy threshold from NIH validation
- reuses baseline predictions when available
- evaluates on the overlap-based NIH ↔ VinDr label mapping
- saves:
  - `vindr_labels_predictions_baseline.csv`
  - `vindr_labels_predictions_tta.csv`
  - `vindr_labels_tta_log.csv`
  - `vindr_labels_metrics_baseline_tta.csv`
  - `vindr_labels_metrics_tta.csv`
  - `vindr_labels_metrics_comparison.csv`
  - `vindr_labels_comparison.csv`
  - `vindr_labels_entropy_threshold.json`

### Multi-label TTA result
The 14-label multi-label TTA implementation was improved to:
- reuse saved threshold and baseline predictions
- perform real updates on most batches
- use overlap-aware grouped entropy for adaptation

Even after those fixes, overall multi-label TTA still did **not** improve performance relative to baseline. This is an important negative result and suggests that lightweight BN-only entropy-gated adaptation is less effective in the harder multi-label external setting.

---

## Grouped-label TTA

### `test_vindr_less_labels_tta.py`
Runs grouped 5-label VinDr evaluation with and without TTA.

What it does:
- loads `training_outputs_less_labels/best_model.pth`
- loads grouped thresholds
- computes or reuses NIH validation entropy threshold
- reuses saved grouped baseline predictions
- runs grouped-label TTA on VinDr
- saves:
  - `vindr_less_labels_predictions_baseline.csv`
  - `vindr_less_labels_predictions_tta.csv`
  - `vindr_less_labels_tta_log.csv`
  - `vindr_less_labels_metrics_baseline_tta.csv`
  - `vindr_less_labels_metrics_tta.csv`
  - `vindr_less_labels_metrics_comparison.csv`
  - `vindr_less_labels_comparison.csv`
  - `vindr_less_labels_entropy_threshold.json`

This grouped-label pipeline is the next experiment designed to test whether a simpler, more stable label space performs better externally and under TTA than the full 14-label formulation.

---

## Why External Evaluation Matters

Internal NIH validation alone does not fully test robustness.

A model can perform well on NIH and still degrade when moved to a different dataset. VinDr-CXR matters because it is:
- external to NIH
- DICOM-based
- from a different clinical source

That makes it a much stronger test of generalization.

---

## Current Findings

### Binary baseline
Works reasonably well on external VinDr and provides the cleanest source-only baseline.

### Binary TTA
Mixed result:
- higher recall
- higher F1
- lower precision
- lower ranking metrics

### 14-label multi-label baseline
Useful for overlap-based external evaluation, but sparse labels and imperfect label matching make the task harder.

### 14-label multi-label TTA
Implemented correctly, but still not beneficial overall.

### Grouped 5-label pipeline
Added as the next-stage experiment to reduce sparsity, improve external overlap, and test whether grouped labels help both baseline performance and TTA.

---

## How to Run

## 1. Binary pipeline
```bash
python prepareNIH.py
python trainNIH.py
python test_vindr_binary.py
python test_vindr_binary_tta.py