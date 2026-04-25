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
from sklearn.metrics import f1_score
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
from tta_utils import tta_step_binary
from tta_utils import clone_model


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

NIH_ROOT_DIR = os.path.join(PROJECT_ROOT, "NIH")
NIH_VAL_CSV = os.path.join(PROJECT_ROOT, "val.csv")

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

BINARY_MODEL_PATH = os.path.join(PROJECT_ROOT, "training_outputs", "best_model.pth")

OUTPUT_BASELINE_PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "vindr_binary_predictions_baseline.csv")
OUTPUT_TTA_PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "vindr_binary_predictions_tta.csv")
OUTPUT_TTA_LOG_CSV = os.path.join(PROJECT_ROOT, "vindr_binary_tta_log.csv")
OUTPUT_COMPARISON_CSV = os.path.join(PROJECT_ROOT, "vindr_binary_comparison.csv")
OUTPUT_METRICS_CSV = os.path.join(PROJECT_ROOT, "vindr_binary_metrics_comparison.csv")
OUTPUT_THRESHOLD_JSON = os.path.join(PROJECT_ROOT, "vindr_binary_entropy_threshold.json")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
RANDOM_SEED = 42

ENTROPY_QUANTILE = 0.25
TTA_LEARNING_RATE = 1e-5
MIN_ACCEPTED_SAMPLES = 8

BINARY_DECISION_THRESHOLD = 0.50

REUSE_SAVED_THRESHOLD = True
REUSE_BASELINE_PREDICTIONS = True
RUN_BASELINE = False


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

    for column_name in candidate_columns:
        if column_name in dataframe.columns:
            return column_name

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
    """
    Used only for entropy threshold estimation on NIH validation images.
    """

    def __init__(self, csv_path: str, nih_root_dir: str, transform):
        self.transform = transform
        self.samples = []

        dataframe = pd.read_csv(csv_path)
        nih_image_name_map = build_file_name_path_map(
            nih_root_dir,
            (".png", ".jpg", ".jpeg"),
        )

        skipped_count = 0

        for _, row in dataframe.iterrows():
            resolved_path = resolve_nih_image_path(row, nih_image_name_map)

            if resolved_path is None:
                skipped_count += 1
                continue

            self.samples.append({"image_path": resolved_path})

        print("NIH validation images found:", len(self.samples))
        print("NIH validation images skipped:", skipped_count)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)

        output = {"image": image}
        return output


def load_vindr_binary_annotations(labels_csv_path: str) -> pd.DataFrame:
    """
    Expected format:
    one row per image, or multiple rows that can be grouped by image id.
    Numeric label columns should be 0/1.
    """
    dataframe = pd.read_csv(labels_csv_path)
    image_id_column = detect_image_id_column(dataframe)

    dataframe[image_id_column] = dataframe[image_id_column].astype(str)
    dataframe[image_id_column] = dataframe[image_id_column].apply(clean_image_id)

    for column_name in dataframe.columns:
        if column_name == image_id_column:
            continue

        dataframe[column_name] = pd.to_numeric(dataframe[column_name], errors="coerce")

    grouped_dataframe = dataframe.groupby(image_id_column, as_index=False).max()
    grouped_dataframe = grouped_dataframe.rename(columns={image_id_column: "image_id"})

    no_finding_column = None
    no_finding_candidates = [
        "No finding",
        "No Finding",
        "no_finding",
        "No_finding",
    ]

    for column_name in no_finding_candidates:
        if column_name in grouped_dataframe.columns:
            no_finding_column = column_name
            break

    binary_rows = []
    numeric_columns = []

    for column_name in grouped_dataframe.columns:
        if column_name == "image_id":
            continue

        if pd.api.types.is_numeric_dtype(grouped_dataframe[column_name]):
            numeric_columns.append(column_name)

    if len(numeric_columns) == 0:
        raise ValueError(
            "No numeric label columns found in VinDr labels CSV. "
            "This script expects an image-level wide CSV."
        )

    for _, row in grouped_dataframe.iterrows():
        abnormal_positive = 0

        for column_name in numeric_columns:
            if column_name == no_finding_column:
                continue

            value = row[column_name]

            if pd.isna(value):
                continue

            if float(value) >= 0.5:
                abnormal_positive = 1
                break

        binary_label = int(abnormal_positive)

        binary_rows.append(
            {
                "image_id": row["image_id"],
                "label": binary_label,
            }
        )

    output_dataframe = pd.DataFrame(binary_rows)
    return output_dataframe


class VinDrBinaryDataset(Dataset):
    def __init__(self, labels_csv_path: str, vindr_test_dir: str, transform):
        self.transform = transform
        self.samples = []

        annotations_dataframe = load_vindr_binary_annotations(labels_csv_path)
        dicom_path_map = build_file_stem_path_map(
            vindr_test_dir,
            (".dicom", ".dcm"),
        )

        skipped_count = 0

        for _, row in annotations_dataframe.iterrows():
            image_id = str(row["image_id"])

            if image_id not in dicom_path_map:
                skipped_count += 1
                continue

            dicom_path = dicom_path_map[image_id]

            self.samples.append(
                {
                    "image_id": image_id,
                    "label": int(row["label"]),
                    "dicom_path": dicom_path,
                }
            )

        print("VinDr binary images found:", len(self.samples))
        print("VinDr binary images skipped:", skipped_count)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = dicom_to_rgb_pil(sample["dicom_path"])
        image = self.transform(image)

        output = {
            "image": image,
            "label": sample["label"],
            "image_id": sample["image_id"],
        }
        return output


def safe_metric(metric_function, *args, **kwargs):
    try:
        value = metric_function(*args, **kwargs)
        return float(value)
    except Exception:
        return float("nan")


def compute_binary_metrics(predictions_dataframe: pd.DataFrame) -> dict:
    y_true = predictions_dataframe["y_true"].to_numpy()
    y_prob = predictions_dataframe["prob_abnormal"].to_numpy()
    y_pred = predictions_dataframe["y_pred"].to_numpy()

    metrics = {}
    metrics["num_images"] = float(len(predictions_dataframe))
    metrics["positive_rate"] = float(np.mean(y_true))
    metrics["roc_auc"] = safe_metric(roc_auc_score, y_true, y_prob)
    metrics["average_precision"] = safe_metric(average_precision_score, y_true, y_prob)
    metrics["accuracy"] = safe_metric(accuracy_score, y_true, y_pred)
    metrics["precision"] = safe_metric(precision_score, y_true, y_pred, zero_division=0)
    metrics["recall"] = safe_metric(recall_score, y_true, y_pred, zero_division=0)
    metrics["f1"] = safe_metric(f1_score, y_true, y_pred, zero_division=0)

    return metrics


def metrics_comparison_dataframe(baseline_metrics: dict, tta_metrics: dict) -> pd.DataFrame:
    rows = []

    metric_names = []
    for metric_name in baseline_metrics.keys():
        metric_names.append(metric_name)

    for metric_name in tta_metrics.keys():
        if metric_name not in metric_names:
            metric_names.append(metric_name)

    for metric_name in metric_names:
        baseline_value = baseline_metrics.get(metric_name, float("nan"))
        tta_value = tta_metrics.get(metric_name, float("nan"))

        delta_value = float("nan")
        if not pd.isna(baseline_value) and not pd.isna(tta_value):
            delta_value = float(tta_value - baseline_value)

        rows.append(
            {
                "metric": metric_name,
                "baseline": baseline_value,
                "tta": tta_value,
                "delta_tta_minus_baseline": delta_value,
            }
        )

    comparison_dataframe = pd.DataFrame(rows)
    return comparison_dataframe


def load_binary_model(device: torch.device):
    model = build_densenet121(num_outputs=1)
    model = load_checkpoint_into_model(model, BINARY_MODEL_PATH, device)
    model = model.to(device)
    return model


def run_baseline_evaluation(model, data_loader, device):
    model.eval()

    prediction_rows = []

    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            image_ids = batch["image_id"]

            logits = model(images)
            probabilities = torch.sigmoid(logits).view(-1).cpu().numpy()

            for sample_index in range(len(image_ids)):
                probability = float(probabilities[sample_index])

                predicted_label = 0
                if probability >= BINARY_DECISION_THRESHOLD:
                    predicted_label = 1

                prediction_rows.append(
                    {
                        "image_id": image_ids[sample_index],
                        "y_true": int(labels[sample_index]),
                        "prob_abnormal": probability,
                        "y_pred": predicted_label,
                        "batch_index": batch_index,
                    }
                )

    predictions_dataframe = pd.DataFrame(prediction_rows)
    return predictions_dataframe


def run_tta_evaluation(model, data_loader, device, entropy_threshold: float):
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
        labels = batch["label"].cpu().numpy()
        image_ids = batch["image_id"]

        step_result = tta_step_binary(
            model=batch_model,
            images=images,
            optimizer=optimizer,
            entropy_threshold=entropy_threshold,
            min_accepted_samples=MIN_ACCEPTED_SAMPLES,
        )

        with torch.no_grad():
            logits = batch_model(images)
            probabilities = torch.sigmoid(logits).view(-1).cpu().numpy()

        for sample_index in range(len(image_ids)):
            probability = float(probabilities[sample_index])

            predicted_label = 0
            if probability >= BINARY_DECISION_THRESHOLD:
                predicted_label = 1

            prediction_rows.append(
                {
                    "image_id": image_ids[sample_index],
                    "y_true": int(labels[sample_index]),
                    "prob_abnormal": probability,
                    "y_pred": predicted_label,
                    "batch_index": batch_index,
                }
            )

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

    predictions_dataframe = pd.DataFrame(prediction_rows)
    log_dataframe = pd.DataFrame(log_rows)

    return predictions_dataframe, log_dataframe

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
        raise FileNotFoundError(
            "Could not find VinDr image-level labels CSV. "
            "Edit VINDR_LABELS_CSV_CANDIDATES near the top of the script."
        )

    print("Using VinDr labels CSV:", labels_csv_path)

    transform = get_eval_transform(IMAGE_SIZE)

    # ---------------------------------------------
    # Reuse saved entropy threshold if available
    # ---------------------------------------------
    entropy_threshold = None

    if REUSE_SAVED_THRESHOLD and os.path.exists(OUTPUT_THRESHOLD_JSON):
        try:
            with open(OUTPUT_THRESHOLD_JSON, "r") as json_file:
                threshold_metadata = json.load(json_file)

            saved_quantile = threshold_metadata.get("entropy_quantile", None)

            if saved_quantile == ENTROPY_QUANTILE:
                entropy_threshold = float(threshold_metadata["entropy_threshold"])
                print("Loaded saved binary entropy threshold:", entropy_threshold)
            else:
                print("Saved entropy threshold exists but quantile changed. Recomputing.")
        except Exception:
            print("Could not load saved entropy threshold. Recomputing.")

    if entropy_threshold is None:
        nih_val_dataset = NIHImageOnlyDataset(
            csv_path=NIH_VAL_CSV,
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

        threshold_model = load_binary_model(device)

        entropy_threshold = estimate_entropy_threshold(
            model=threshold_model,
            data_loader=nih_val_loader,
            device=device,
            is_multilabel=False,
            quantile=ENTROPY_QUANTILE,
        )

        print("Estimated binary entropy threshold:", entropy_threshold)

        threshold_metadata = {
            "entropy_threshold": float(entropy_threshold),
            "entropy_quantile": float(ENTROPY_QUANTILE),
            "tta_learning_rate": float(TTA_LEARNING_RATE),
            "min_accepted_samples": int(MIN_ACCEPTED_SAMPLES),
            "decision_threshold": float(BINARY_DECISION_THRESHOLD),
        }

        with open(OUTPUT_THRESHOLD_JSON, "w") as json_file:
            json.dump(threshold_metadata, json_file, indent=2)

    # ---------------------------------------------
    # Build VinDr dataset / loader
    # ---------------------------------------------
    vindr_dataset = VinDrBinaryDataset(
        labels_csv_path=labels_csv_path,
        vindr_test_dir=VINDR_TEST_DIR,
        transform=transform,
    )

    vindr_loader = DataLoader(
        vindr_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # ---------------------------------------------
    # Reuse baseline predictions if available
    # ---------------------------------------------
    baseline_predictions = None

    if RUN_BASELINE:
        baseline_model = load_binary_model(device)
        baseline_predictions = run_baseline_evaluation(
            model=baseline_model,
            data_loader=vindr_loader,
            device=device,
        )
        baseline_predictions.to_csv(OUTPUT_BASELINE_PREDICTIONS_CSV, index=False)
        print("Ran and saved baseline predictions.")
    elif REUSE_BASELINE_PREDICTIONS and os.path.exists(OUTPUT_BASELINE_PREDICTIONS_CSV):
        baseline_predictions = pd.read_csv(OUTPUT_BASELINE_PREDICTIONS_CSV)
        print("Loaded saved baseline predictions:", OUTPUT_BASELINE_PREDICTIONS_CSV)
    else:
        baseline_model = load_binary_model(device)
        baseline_predictions = run_baseline_evaluation(
            model=baseline_model,
            data_loader=vindr_loader,
            device=device,
        )
        baseline_predictions.to_csv(OUTPUT_BASELINE_PREDICTIONS_CSV, index=False)
        print("No saved baseline predictions found, so baseline was run once.")

    # ---------------------------------------------
    # Run TTA
    # ---------------------------------------------
    tta_model = load_binary_model(device)
    tta_predictions, tta_log = run_tta_evaluation(
        model=tta_model,
        data_loader=vindr_loader,
        device=device,
        entropy_threshold=entropy_threshold,
    )

    tta_predictions.to_csv(OUTPUT_TTA_PREDICTIONS_CSV, index=False)
    tta_log.to_csv(OUTPUT_TTA_LOG_CSV, index=False)

    print("Updated batches:", int(tta_log["updated"].sum()), "/", len(tta_log))
    print("Mean accepted fraction:", float(tta_log["accepted_fraction"].mean()))
    print("Mean accepted count:", float(tta_log["accepted_count"].mean()))

    # ---------------------------------------------
    # Metrics
    # ---------------------------------------------
    baseline_metrics = compute_binary_metrics(baseline_predictions)
    tta_metrics = compute_binary_metrics(tta_predictions)

    comparison_metrics_dataframe = metrics_comparison_dataframe(
        baseline_metrics=baseline_metrics,
        tta_metrics=tta_metrics,
    )
    comparison_metrics_dataframe.to_csv(OUTPUT_METRICS_CSV, index=False)

    comparison_dataframe = baseline_predictions.merge(
        tta_predictions,
        on=["image_id", "y_true", "batch_index"],
        how="inner",
        suffixes=("_baseline", "_tta"),
    )
    comparison_dataframe.to_csv(OUTPUT_COMPARISON_CSV, index=False)

    print()
    print("Saved:")
    print(" -", OUTPUT_BASELINE_PREDICTIONS_CSV)
    print(" -", OUTPUT_TTA_PREDICTIONS_CSV)
    print(" -", OUTPUT_TTA_LOG_CSV)
    print(" -", OUTPUT_COMPARISON_CSV)
    print(" -", OUTPUT_METRICS_CSV)
    print(" -", OUTPUT_THRESHOLD_JSON)
    print()
    print("Baseline metrics:")
    print(baseline_metrics)
    print()
    print("TTA metrics:")
    print(tta_metrics)

if __name__ == "__main__":
    main()