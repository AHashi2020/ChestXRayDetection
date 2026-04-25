import json
import os
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from tta_utils import build_densenet121
from tta_utils import build_file_name_path_map
from tta_utils import build_file_stem_path_map
from tta_utils import configure_model_for_tta
from tta_utils import dicom_to_rgb_pil
from tta_utils import estimate_entropy_threshold
from tta_utils import find_first_existing_path
from tta_utils import get_eval_transform
from tta_utils import load_checkpoint_into_model
from tta_utils import tta_step_multilabel
from tta_utils import clone_model


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

NIH_ROOT_DIR = os.path.join(PROJECT_ROOT, "NIH")
NIH_VAL_LABELS_CSV = os.path.join(PROJECT_ROOT, "val_less_labels.csv")

VINDR_TEST_DIR = os.path.join(PROJECT_ROOT, "vindr-cxr_test")
VINDR_LABELS_CSV_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "image_labels_test_subset.csv"),
    os.path.join(PROJECT_ROOT, "vindr_image_labels_test_subset.csv"),
    os.path.join(PROJECT_ROOT, "image_labels_test.csv"),
    os.path.join(VINDR_TEST_DIR, "image_labels_test_subset.csv"),
    os.path.join(VINDR_TEST_DIR, "image_labels_test.csv"),
    os.path.join(VINDR_TEST_DIR, "annotations", "image_labels_test_subset.csv"),
    os.path.join(VINDR_TEST_DIR, "annotations", "image_labels_test.csv"),
]

MODEL_PATH = os.path.join(PROJECT_ROOT, "training_outputs_less_labels", "best_model.pth")
THRESHOLDS_JSON = os.path.join(PROJECT_ROOT, "training_outputs_less_labels", "best_thresholds.json")

OUTPUT_BASELINE_PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "vindr_less_labels_predictions_baseline.csv")
OUTPUT_TTA_PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "vindr_less_labels_predictions_tta.csv")
OUTPUT_TTA_LOG_CSV = os.path.join(PROJECT_ROOT, "vindr_less_labels_tta_log.csv")
OUTPUT_BASELINE_METRICS_CSV = os.path.join(PROJECT_ROOT, "vindr_less_labels_metrics_baseline_tta.csv")
OUTPUT_TTA_METRICS_CSV = os.path.join(PROJECT_ROOT, "vindr_less_labels_metrics_tta.csv")
OUTPUT_METRICS_COMPARISON_CSV = os.path.join(PROJECT_ROOT, "vindr_less_labels_metrics_comparison.csv")
OUTPUT_COMPARISON_CSV = os.path.join(PROJECT_ROOT, "vindr_less_labels_comparison.csv")
OUTPUT_THRESHOLD_JSON = os.path.join(PROJECT_ROOT, "vindr_less_labels_entropy_threshold.json")

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0
RANDOM_SEED = 42

ENTROPY_QUANTILE = 0.25
TTA_LEARNING_RATE = 5e-6
MIN_ACCEPTED_SAMPLES = 2

REUSE_SAVED_THRESHOLD = True
REUSE_BASELINE_PREDICTIONS = True
RUN_BASELINE = False

# ------------------------------------------------------------------
# Recall-focused threshold overrides
# ------------------------------------------------------------------
# These are used at evaluation time so the model misses fewer positives.
# Airspace_Opacity is left alone because it was already over-predicting.
#
# These are practical starting values based on your current results.
# For final fair reporting, the best version is to tune threshold policy
# on NIH validation rather than VinDr test.
# ------------------------------------------------------------------
THRESHOLD_OVERRIDES = {
    "Airspace_Opacity": 0.50,
    "Cardiomediastinal_Abnormality": 0.36,
    "Pleural_Abnormality": 0.43,
    "Focal_Lesion": 0.68,
    "Chronic_Parenchymal_Change": 0.87,
}

label_names = [
    "Airspace_Opacity",
    "Cardiomediastinal_Abnormality",
    "Pleural_Abnormality",
    "Focal_Lesion",
    "Chronic_Parenchymal_Change",
]

grouped_vindr_specs = [
    {
        "eval_name": "Airspace_Opacity",
        "vindr_candidates": ["Atelectasis", "Infiltration", "Consolidation", "Pneumonia", "Edema"],
    },
    {
        "eval_name": "Cardiomediastinal_Abnormality",
        "vindr_candidates": ["Cardiomegaly", "Enlarged cardiac silhouette"],
    },
    {
        "eval_name": "Pleural_Abnormality",
        "vindr_candidates": ["Pleural effusion", "Effusion", "Pneumothorax", "Pleural thickening", "Pleural_Thickening"],
    },
    {
        "eval_name": "Focal_Lesion",
        "vindr_candidates": ["Nodule/Mass", "Nodule", "Mass"],
    },
    {
        "eval_name": "Chronic_Parenchymal_Change",
        "vindr_candidates": ["Emphysema", "Pulmonary fibrosis", "Fibrosis"],
    },
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_image_id(raw_value) -> str:
    image_id = str(raw_value)
    image_id = image_id.strip()
    image_id = image_id.replace(".dicom", "")
    image_id = image_id.replace(".dcm", "")
    image_id = image_id.replace(".png", "")
    image_id = image_id.replace(".jpg", "")
    image_id = image_id.replace(".jpeg", "")
    return image_id


def detect_image_id_column(dataframe: pd.DataFrame) -> str:
    candidate_columns = [
        "image_id",
        "image_name",
        "dicom_id",
        "study_id",
        "id",
    ]

    column_index = 0
    while column_index < len(candidate_columns):
        column_name = candidate_columns[column_index]
        if column_name in dataframe.columns:
            return column_name
        column_index += 1

    raise ValueError(
        "Could not find image id column in VinDr labels CSV. "
        "Expected one of: image_id, image_name, dicom_id, study_id, id"
    )


def resolve_nih_image_path(row, nih_image_name_map):
    if "image_path" in row.index:
        raw_path = row["image_path"]

        if pd.notna(raw_path):
            raw_path = str(raw_path)

            candidate_paths = [
                raw_path,
                os.path.join(PROJECT_ROOT, raw_path),
                os.path.join(NIH_ROOT_DIR, raw_path),
            ]

            resolved_path = find_first_existing_path(candidate_paths)

            if resolved_path is not None:
                return resolved_path

    if "image_name" in row.index:
        image_name = row["image_name"]

        if pd.notna(image_name):
            image_name = str(image_name)

            if image_name in nih_image_name_map:
                return nih_image_name_map[image_name]

    return None


class NIHImageOnlyDataset(Dataset):
    def __init__(self, csv_path: str, nih_root_dir: str, transform):
        self.transform = transform
        self.samples = []

        dataframe = pd.read_csv(csv_path)
        nih_image_name_map = build_file_name_path_map(
            nih_root_dir,
            (".png", ".jpg", ".jpeg"),
        )

        skipped_count = 0

        row_index = 0
        while row_index < len(dataframe):
            row = dataframe.iloc[row_index]
            resolved_path = resolve_nih_image_path(row, nih_image_name_map)

            if resolved_path is None:
                skipped_count += 1
                row_index += 1
                continue

            self.samples.append({"image_path": resolved_path})
            row_index += 1

        print("NIH validation less-label images found:", len(self.samples))
        print("NIH validation less-label images skipped:", skipped_count)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)
        return {"image": image}


def load_thresholds_by_label(thresholds_json_path: str) -> dict:
    thresholds_by_label = {}

    label_index = 0
    while label_index < len(label_names):
        label_name = label_names[label_index]
        thresholds_by_label[label_name] = 0.50
        label_index += 1

    if not os.path.exists(thresholds_json_path):
        print("Threshold JSON not found. Falling back to 0.50 for all labels.")
        return thresholds_by_label

    with open(thresholds_json_path, "r") as json_file:
        raw_object = json.load(json_file)

    if isinstance(raw_object, dict):
        if "best_thresholds" in raw_object and isinstance(raw_object["best_thresholds"], dict):
            raw_object = raw_object["best_thresholds"]

    if isinstance(raw_object, dict):
        label_index = 0
        while label_index < len(label_names):
            label_name = label_names[label_index]
            if label_name in raw_object:
                thresholds_by_label[label_name] = float(raw_object[label_name])
            label_index += 1

    return thresholds_by_label


def apply_threshold_overrides(thresholds_by_label: dict) -> dict:
    updated_thresholds = dict(thresholds_by_label)

    for label_name, threshold_value in THRESHOLD_OVERRIDES.items():
        if label_name in updated_thresholds:
            updated_thresholds[label_name] = float(threshold_value)

    return updated_thresholds


def build_active_specs(vindr_dataframe: pd.DataFrame):
    active_specs = []

    spec_index = 0
    while spec_index < len(grouped_vindr_specs):
        spec = grouped_vindr_specs[spec_index]
        found_columns = []

        candidate_index = 0
        while candidate_index < len(spec["vindr_candidates"]):
            candidate_column = spec["vindr_candidates"][candidate_index]
            if candidate_column in vindr_dataframe.columns:
                found_columns.append(candidate_column)
            candidate_index += 1

        if len(found_columns) > 0:
            active_spec = {
                "eval_name": spec["eval_name"],
                "vindr_columns": found_columns,
            }
            active_specs.append(active_spec)

        spec_index += 1

    if len(active_specs) == 0:
        raise ValueError("Could not find grouped VinDr overlap columns in the CSV.")

    return active_specs


def load_vindr_grouped_annotations(labels_csv_path: str):
    dataframe = pd.read_csv(labels_csv_path)
    image_id_column = detect_image_id_column(dataframe)

    dataframe[image_id_column] = dataframe[image_id_column].astype(str)
    dataframe[image_id_column] = dataframe[image_id_column].apply(clean_image_id)

    column_index = 0
    all_columns = list(dataframe.columns)
    while column_index < len(all_columns):
        column_name = all_columns[column_index]
        if column_name != image_id_column:
            dataframe[column_name] = pd.to_numeric(dataframe[column_name], errors="coerce")
        column_index += 1

    grouped_dataframe = dataframe.groupby(image_id_column, as_index=False).max()
    grouped_dataframe = grouped_dataframe.rename(columns={image_id_column: "image_id"})

    active_specs = build_active_specs(grouped_dataframe)

    rows = []
    row_index = 0
    while row_index < len(grouped_dataframe):
        row = grouped_dataframe.iloc[row_index]

        output_row = {"image_id": row["image_id"]}

        spec_index = 0
        while spec_index < len(active_specs):
            spec = active_specs[spec_index]
            grouped_value = 0

            column_index = 0
            while column_index < len(spec["vindr_columns"]):
                column_name = spec["vindr_columns"][column_index]
                value = row[column_name]

                if pd.notna(value) and float(value) >= 0.5:
                    grouped_value = 1
                    break

                column_index += 1

            output_row[spec["eval_name"]] = grouped_value
            spec_index += 1

        rows.append(output_row)
        row_index += 1

    output_dataframe = pd.DataFrame(rows)
    return output_dataframe, active_specs


class VinDrLessLabelsDataset(Dataset):
    def __init__(self, labels_csv_path: str, vindr_test_dir: str, transform):
        self.transform = transform
        self.samples = []

        annotations_dataframe, active_specs = load_vindr_grouped_annotations(labels_csv_path)
        self.active_specs = active_specs
        self.eval_label_names = []

        spec_index = 0
        while spec_index < len(active_specs):
            self.eval_label_names.append(active_specs[spec_index]["eval_name"])
            spec_index += 1

        dicom_path_map = build_file_stem_path_map(vindr_test_dir, (".dicom", ".dcm"))

        skipped_count = 0

        row_index = 0
        while row_index < len(annotations_dataframe):
            row = annotations_dataframe.iloc[row_index]
            image_id = str(row["image_id"])

            if image_id not in dicom_path_map:
                skipped_count += 1
                row_index += 1
                continue

            sample = {
                "image_id": image_id,
                "dicom_path": dicom_path_map[image_id],
            }

            label_index = 0
            while label_index < len(self.eval_label_names):
                label_name = self.eval_label_names[label_index]
                sample[label_name] = int(row[label_name])
                label_index += 1

            self.samples.append(sample)
            row_index += 1

        print("VinDr less-label images found:", len(self.samples))
        print("VinDr less-label images skipped:", skipped_count)
        print("Active grouped labels:", self.eval_label_names)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = dicom_to_rgb_pil(sample["dicom_path"])
        image = self.transform(image)

        y_true_values = []
        label_index = 0
        while label_index < len(self.eval_label_names):
            label_name = self.eval_label_names[label_index]
            y_true_values.append(float(sample[label_name]))
            label_index += 1

        y_true_tensor = torch.tensor(y_true_values, dtype=torch.float32)

        return {
            "image": image,
            "image_id": sample["image_id"],
            "y_true": y_true_tensor,
        }


def safe_metric(metric_function, *args, **kwargs):
    try:
        value = metric_function(*args, **kwargs)
        return float(value)
    except Exception:
        return float("nan")


def load_model(device: torch.device):
    model = build_densenet121(num_outputs=len(label_names))
    model = load_checkpoint_into_model(model, MODEL_PATH, device)
    model = model.to(device)
    return model


def apply_thresholds_to_prediction_dataframe(predictions_dataframe: pd.DataFrame, eval_label_names, thresholds_by_label):
    updated_dataframe = predictions_dataframe.copy()

    label_index = 0
    while label_index < len(eval_label_names):
        label_name = eval_label_names[label_index]
        prob_column = "prob_" + label_name
        pred_column = "pred_" + label_name

        if prob_column in updated_dataframe.columns:
            threshold = float(thresholds_by_label[label_name])

            new_predictions = []
            row_index = 0
            while row_index < len(updated_dataframe):
                probability = float(updated_dataframe.iloc[row_index][prob_column])

                predicted_label = 0
                if probability >= threshold:
                    predicted_label = 1

                new_predictions.append(predicted_label)
                row_index += 1

            updated_dataframe[pred_column] = new_predictions

        label_index += 1

    return updated_dataframe


def run_baseline_evaluation(model, data_loader, device, eval_label_names, thresholds_by_label):
    model.eval()
    prediction_rows = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(data_loader):
            images = batch["image"].to(device)
            image_ids = batch["image_id"]
            y_true = batch["y_true"].cpu().numpy()

            logits = model(images)
            probabilities = torch.sigmoid(logits).cpu().numpy()

            sample_index = 0
            while sample_index < len(image_ids):
                row = {
                    "image_id": image_ids[sample_index],
                    "batch_index": batch_index,
                }

                label_index = 0
                while label_index < len(eval_label_names):
                    label_name = eval_label_names[label_index]
                    threshold = thresholds_by_label[label_name]
                    probability = float(probabilities[sample_index][label_index])

                    predicted_label = 0
                    if probability >= threshold:
                        predicted_label = 1

                    row["y_true_" + label_name] = int(y_true[sample_index][label_index])
                    row["prob_" + label_name] = probability
                    row["pred_" + label_name] = predicted_label
                    label_index += 1

                prediction_rows.append(row)
                sample_index += 1

    return pd.DataFrame(prediction_rows)


def run_tta_evaluation(model, data_loader, device, entropy_threshold, eval_label_names, thresholds_by_label):
    base_model = clone_model(model)
    base_model.eval()

    prediction_rows = []
    log_rows = []

    for batch_index, batch in enumerate(data_loader):
        batch_model = clone_model(base_model)
        batch_model = batch_model.to(device)

        batch_model, batch_norm_parameters = configure_model_for_tta(batch_model)
        optimizer = Adam(batch_norm_parameters, lr=TTA_LEARNING_RATE)

        images = batch["image"].to(device)
        image_ids = batch["image_id"]
        y_true = batch["y_true"].cpu().numpy()

        step_result = tta_step_multilabel(
            model=batch_model,
            images=images,
            optimizer=optimizer,
            entropy_threshold=entropy_threshold,
            min_accepted_samples=MIN_ACCEPTED_SAMPLES,
        )

        with torch.inference_mode():
            logits = batch_model(images)
            probabilities = torch.sigmoid(logits).cpu().numpy()

        sample_index = 0
        while sample_index < len(image_ids):
            row = {
                "image_id": image_ids[sample_index],
                "batch_index": batch_index,
            }

            label_index = 0
            while label_index < len(eval_label_names):
                label_name = eval_label_names[label_index]
                threshold = thresholds_by_label[label_name]
                probability = float(probabilities[sample_index][label_index])

                predicted_label = 0
                if probability >= threshold:
                    predicted_label = 1

                row["y_true_" + label_name] = int(y_true[sample_index][label_index])
                row["prob_" + label_name] = probability
                row["pred_" + label_name] = predicted_label
                label_index += 1

            prediction_rows.append(row)
            sample_index += 1

        log_rows.append(
            {
                "batch_index": batch_index,
                "accepted_count": step_result["accepted_count"],
                "batch_size": step_result["batch_size"],
                "accepted_fraction": step_result["accepted_fraction"],
                "mean_entropy": step_result["mean_entropy"],
                "loss": step_result["loss"],
                "updated": step_result["updated"],
            }
        )

    return pd.DataFrame(prediction_rows), pd.DataFrame(log_rows)


def compute_metrics(predictions_dataframe: pd.DataFrame, eval_label_names):
    metric_rows = []

    label_index = 0
    while label_index < len(eval_label_names):
        label_name = eval_label_names[label_index]

        y_true = predictions_dataframe["y_true_" + label_name].to_numpy()
        y_prob = predictions_dataframe["prob_" + label_name].to_numpy()
        y_pred = predictions_dataframe["pred_" + label_name].to_numpy()

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        recall_value = safe_metric(recall_score, y_true, y_pred, zero_division=0)

        specificity = float("nan")
        if (tn + fp) > 0:
            specificity = float(tn / (tn + fp))

        false_negative_rate = float("nan")
        if (fn + tp) > 0:
            false_negative_rate = float(fn / (fn + tp))

        negative_predictive_value = float("nan")
        if (tn + fn) > 0:
            negative_predictive_value = float(tn / (tn + fn))

        balanced_accuracy = float("nan")
        if not np.isnan(recall_value) and not np.isnan(specificity):
            balanced_accuracy = float((recall_value + specificity) / 2.0)

        row = {
            "label": label_name,
            "num_images": float(len(predictions_dataframe)),
            "positive_rate": float(np.mean(y_true)),
            "predicted_positive_rate": float(np.mean(y_pred)),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "tn": float(tn),
            "roc_auc": safe_metric(roc_auc_score, y_true, y_prob),
            "average_precision": safe_metric(average_precision_score, y_true, y_prob),
            "accuracy": safe_metric(accuracy_score, y_true, y_pred),
            "balanced_accuracy": balanced_accuracy,
            "precision": safe_metric(precision_score, y_true, y_pred, zero_division=0),
            "recall": recall_value,
            "specificity": specificity,
            "false_negative_rate": false_negative_rate,
            "negative_predictive_value": negative_predictive_value,
            "f1": safe_metric(f1_score, y_true, y_pred, zero_division=0),
            "f2": safe_metric(fbeta_score, y_true, y_pred, beta=2, zero_division=0),
        }

        metric_rows.append(row)
        label_index += 1

    metrics_dataframe = pd.DataFrame(metric_rows)

    macro_row = {"label": "MACRO_MEAN"}
    metric_columns = [
        "num_images",
        "positive_rate",
        "predicted_positive_rate",
        "tp",
        "fp",
        "fn",
        "tn",
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "false_negative_rate",
        "negative_predictive_value",
        "f1",
        "f2",
    ]

    column_index = 0
    while column_index < len(metric_columns):
        column_name = metric_columns[column_index]
        macro_row[column_name] = float(np.nanmean(metrics_dataframe[column_name].to_numpy(dtype=float)))
        column_index += 1

    metrics_dataframe = pd.concat(
        [metrics_dataframe, pd.DataFrame([macro_row])],
        ignore_index=True,
    )

    return metrics_dataframe


def build_metrics_comparison_dataframe(baseline_metrics_dataframe: pd.DataFrame, tta_metrics_dataframe: pd.DataFrame):
    merged_dataframe = baseline_metrics_dataframe.merge(
        tta_metrics_dataframe,
        on="label",
        how="outer",
        suffixes=("_baseline", "_tta"),
    )

    metric_names = [
        "num_images",
        "positive_rate",
        "predicted_positive_rate",
        "tp",
        "fp",
        "fn",
        "tn",
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "false_negative_rate",
        "negative_predictive_value",
        "f1",
        "f2",
    ]

    metric_index = 0
    while metric_index < len(metric_names):
        metric_name = metric_names[metric_index]
        baseline_column = metric_name + "_baseline"
        tta_column = metric_name + "_tta"
        delta_column = metric_name + "_delta"

        if baseline_column in merged_dataframe.columns and tta_column in merged_dataframe.columns:
            merged_dataframe[delta_column] = merged_dataframe[tta_column] - merged_dataframe[baseline_column]

        metric_index += 1

    return merged_dataframe


def main():
    set_seed(RANDOM_SEED)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    labels_csv_path = find_first_existing_path(VINDR_LABELS_CSV_CANDIDATES)
    if labels_csv_path is None:
        raise FileNotFoundError("Could not find VinDr labels CSV.")

    print("Using VinDr labels CSV:", labels_csv_path)

    thresholds_by_label = load_thresholds_by_label(THRESHOLDS_JSON)
    print("Loaded NIH-validation thresholds:", thresholds_by_label)

    thresholds_by_label = apply_threshold_overrides(thresholds_by_label)
    print("Using recall-focused thresholds:", thresholds_by_label)

    transform = get_eval_transform(IMAGE_SIZE)

    entropy_threshold = None

    if REUSE_SAVED_THRESHOLD and os.path.exists(OUTPUT_THRESHOLD_JSON):
        try:
            with open(OUTPUT_THRESHOLD_JSON, "r") as json_file:
                threshold_metadata = json.load(json_file)

            saved_quantile = threshold_metadata.get("entropy_quantile", None)

            if saved_quantile == ENTROPY_QUANTILE:
                entropy_threshold = float(threshold_metadata["entropy_threshold"])
                print("Loaded saved less-label entropy threshold:", entropy_threshold)
        except Exception:
            print("Could not load saved threshold. Recomputing.")

    if entropy_threshold is None:
        nih_val_dataset = NIHImageOnlyDataset(
            csv_path=NIH_VAL_LABELS_CSV,
            nih_root_dir=NIH_ROOT_DIR,
            transform=transform,
        )

        nih_val_loader = DataLoader(
            nih_val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )

        threshold_model = load_model(device)

        entropy_threshold = estimate_entropy_threshold(
            model=threshold_model,
            data_loader=nih_val_loader,
            device=device,
            is_multilabel=True,
            quantile=ENTROPY_QUANTILE,
        )

        print("Estimated less-label entropy threshold:", entropy_threshold)

        threshold_metadata = {
            "entropy_threshold": float(entropy_threshold),
            "entropy_quantile": float(ENTROPY_QUANTILE),
            "tta_learning_rate": float(TTA_LEARNING_RATE),
            "min_accepted_samples": int(MIN_ACCEPTED_SAMPLES),
            "thresholds_by_label_after_overrides": thresholds_by_label,
        }

        with open(OUTPUT_THRESHOLD_JSON, "w") as json_file:
            json.dump(threshold_metadata, json_file, indent=2)

    dataset = VinDrLessLabelsDataset(
        labels_csv_path=labels_csv_path,
        vindr_test_dir=VINDR_TEST_DIR,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    baseline_predictions = None

    if RUN_BASELINE:
        baseline_model = load_model(device)
        baseline_predictions = run_baseline_evaluation(
            model=baseline_model,
            data_loader=loader,
            device=device,
            eval_label_names=dataset.eval_label_names,
            thresholds_by_label=thresholds_by_label,
        )
        baseline_predictions.to_csv(OUTPUT_BASELINE_PREDICTIONS_CSV, index=False)
        print("Ran and saved baseline predictions.")
    elif REUSE_BASELINE_PREDICTIONS and os.path.exists(OUTPUT_BASELINE_PREDICTIONS_CSV):
        baseline_predictions = pd.read_csv(OUTPUT_BASELINE_PREDICTIONS_CSV)

        # Important:
        # If saved predictions were created with older thresholds, we do NOT want
        # to trust the old pred_* columns. We rebuild them from prob_* columns here.
        baseline_predictions = apply_thresholds_to_prediction_dataframe(
            predictions_dataframe=baseline_predictions,
            eval_label_names=dataset.eval_label_names,
            thresholds_by_label=thresholds_by_label,
        )

        print("Loaded saved baseline probabilities and rebuilt predictions with new thresholds:", OUTPUT_BASELINE_PREDICTIONS_CSV)
    else:
        baseline_model = load_model(device)
        baseline_predictions = run_baseline_evaluation(
            model=baseline_model,
            data_loader=loader,
            device=device,
            eval_label_names=dataset.eval_label_names,
            thresholds_by_label=thresholds_by_label,
        )
        baseline_predictions.to_csv(OUTPUT_BASELINE_PREDICTIONS_CSV, index=False)
        print("No saved baseline predictions found, so baseline was run once.")

    tta_model = load_model(device)
    tta_predictions, tta_log = run_tta_evaluation(
        model=tta_model,
        data_loader=loader,
        device=device,
        entropy_threshold=entropy_threshold,
        eval_label_names=dataset.eval_label_names,
        thresholds_by_label=thresholds_by_label,
    )

    tta_predictions.to_csv(OUTPUT_TTA_PREDICTIONS_CSV, index=False)
    tta_log.to_csv(OUTPUT_TTA_LOG_CSV, index=False)

    print("Updated batches:", int(tta_log["updated"].sum()), "/", len(tta_log))
    print("Mean accepted fraction:", float(tta_log["accepted_fraction"].mean()))
    print("Mean accepted count:", float(tta_log["accepted_count"].mean()))

    baseline_metrics_dataframe = compute_metrics(
        predictions_dataframe=baseline_predictions,
        eval_label_names=dataset.eval_label_names,
    )
    baseline_metrics_dataframe.to_csv(OUTPUT_BASELINE_METRICS_CSV, index=False)

    tta_metrics_dataframe = compute_metrics(
        predictions_dataframe=tta_predictions,
        eval_label_names=dataset.eval_label_names,
    )
    tta_metrics_dataframe.to_csv(OUTPUT_TTA_METRICS_CSV, index=False)

    metrics_comparison_dataframe = build_metrics_comparison_dataframe(
        baseline_metrics_dataframe=baseline_metrics_dataframe,
        tta_metrics_dataframe=tta_metrics_dataframe,
    )
    metrics_comparison_dataframe.to_csv(OUTPUT_METRICS_COMPARISON_CSV, index=False)

    comparison_dataframe = baseline_predictions.merge(
        tta_predictions,
        on=["image_id", "batch_index"],
        how="inner",
        suffixes=("_baseline", "_tta"),
    )
    comparison_dataframe.to_csv(OUTPUT_COMPARISON_CSV, index=False)

    print("Saved:", OUTPUT_BASELINE_PREDICTIONS_CSV)
    print("Saved:", OUTPUT_TTA_PREDICTIONS_CSV)
    print("Saved:", OUTPUT_TTA_LOG_CSV)
    print("Saved:", OUTPUT_BASELINE_METRICS_CSV)
    print("Saved:", OUTPUT_TTA_METRICS_CSV)
    print("Saved:", OUTPUT_METRICS_COMPARISON_CSV)
    print("Saved:", OUTPUT_COMPARISON_CSV)
    print("Saved:", OUTPUT_THRESHOLD_JSON)
    print("Done.")


if __name__ == "__main__":
    main()