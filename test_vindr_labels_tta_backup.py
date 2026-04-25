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

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


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

REUSE_SAVED_THRESHOLD = True
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

NIH_ROOT_DIR = os.path.join(PROJECT_ROOT, "NIH")
NIH_VAL_LABELS_CSV = os.path.join(PROJECT_ROOT, "val_labels.csv")

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

MULTILABEL_MODEL_PATH = os.path.join(PROJECT_ROOT, "training_outputs_labels", "best_model.pth")
MULTILABEL_THRESHOLDS_JSON = os.path.join(PROJECT_ROOT, "training_outputs_labels", "best_thresholds.json")

OUTPUT_BASELINE_PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "vindr_labels_predictions_baseline.csv")
OUTPUT_TTA_PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "vindr_labels_predictions_tta.csv")
OUTPUT_TTA_LOG_CSV = os.path.join(PROJECT_ROOT, "vindr_labels_tta_log.csv")
OUTPUT_BASELINE_METRICS_CSV = os.path.join(PROJECT_ROOT, "vindr_labels_metrics_baseline_tta.csv")
OUTPUT_TTA_METRICS_CSV = os.path.join(PROJECT_ROOT, "vindr_labels_metrics_tta.csv")
OUTPUT_METRICS_COMPARISON_CSV = os.path.join(PROJECT_ROOT, "vindr_labels_metrics_comparison.csv")
OUTPUT_COMPARISON_CSV = os.path.join(PROJECT_ROOT, "vindr_labels_comparison.csv")
OUTPUT_THRESHOLD_JSON = os.path.join(PROJECT_ROOT, "vindr_labels_entropy_threshold.json")

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0
RANDOM_SEED = 42

ENTROPY_QUANTILE = 0.35
TTA_LEARNING_RATE = 1e-5
MIN_ACCEPTED_SAMPLES = 3

REUSE_SAVED_THRESHOLD = True
REUSE_BASELINE_PREDICTIONS = True
RUN_BASELINE = False

NIH_LABEL_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
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

        print("NIH validation label images found:", len(self.samples))
        print("NIH validation label images skipped:", skipped_count)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)

        output = {"image": image}
        return output


def load_thresholds_by_label(thresholds_json_path: str) -> dict:
    thresholds_by_label = {}

    for label_name in NIH_LABEL_NAMES:
        thresholds_by_label[label_name] = 0.50

    if not os.path.exists(thresholds_json_path):
        print("Threshold JSON not found. Falling back to 0.50 for all labels.")
        return thresholds_by_label

    with open(thresholds_json_path, "r") as json_file:
        raw_object = json.load(json_file)

    if isinstance(raw_object, dict):
        if "best_thresholds" in raw_object and isinstance(raw_object["best_thresholds"], dict):
            raw_object = raw_object["best_thresholds"]

    if isinstance(raw_object, dict):
        for label_name in NIH_LABEL_NAMES:
            if label_name in raw_object:
                thresholds_by_label[label_name] = float(raw_object[label_name])

    return thresholds_by_label


def build_nih_label_to_index() -> dict:
    label_to_index = {}

    for label_index in range(len(NIH_LABEL_NAMES)):
        label_name = NIH_LABEL_NAMES[label_index]
        label_to_index[label_name] = label_index

    return label_to_index


def build_overlap_specs():
    """
    Only use cleaner NIH <-> VinDr overlaps.
    """
    overlap_specs = []

    overlap_specs.append(
        {
            "eval_name": "Atelectasis",
            "prediction_labels": ["Atelectasis"],
            "vindr_candidates": ["Atelectasis"],
        }
    )

    overlap_specs.append(
        {
            "eval_name": "Cardiomegaly",
            "prediction_labels": ["Cardiomegaly"],
            "vindr_candidates": ["Cardiomegaly", "Enlarged cardiac silhouette"],
        }
    )

    overlap_specs.append(
        {
            "eval_name": "Pleural_Effusion",
            "prediction_labels": ["Effusion"],
            "vindr_candidates": ["Pleural effusion", "Effusion"],
        }
    )

    overlap_specs.append(
        {
            "eval_name": "Consolidation",
            "prediction_labels": ["Consolidation"],
            "vindr_candidates": ["Consolidation"],
        }
    )

    overlap_specs.append(
        {
            "eval_name": "Edema",
            "prediction_labels": ["Edema"],
            "vindr_candidates": ["Edema"],
        }
    )

    overlap_specs.append(
        {
            "eval_name": "Emphysema",
            "prediction_labels": ["Emphysema"],
            "vindr_candidates": ["Emphysema"],
        }
    )

    overlap_specs.append(
        {
            "eval_name": "Fibrosis",
            "prediction_labels": ["Fibrosis"],
            "vindr_candidates": ["Pulmonary fibrosis", "Fibrosis"],
        }
    )

    overlap_specs.append(
        {
            "eval_name": "Pleural_Thickening",
            "prediction_labels": ["Pleural_Thickening"],
            "vindr_candidates": ["Pleural thickening", "Pleural_Thickening"],
        }
    )

    overlap_specs.append(
        {
            "eval_name": "Pneumothorax",
            "prediction_labels": ["Pneumothorax"],
            "vindr_candidates": ["Pneumothorax"],
        }
    )

    overlap_specs.append(
        {
            "eval_name": "Nodule_Mass",
            "prediction_labels": ["Mass", "Nodule"],
            "vindr_candidates": ["Nodule/Mass", "Nodule", "Mass"],
        }
    )

    if "Infiltration" in NIH_LABEL_NAMES:
        overlap_specs.append(
            {
                "eval_name": "Infiltration",
                "prediction_labels": ["Infiltration"],
                "vindr_candidates": ["Infiltration"],
            }
        )

    return overlap_specs


def load_vindr_wide_annotations(labels_csv_path: str) -> pd.DataFrame:
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
    return grouped_dataframe


def build_active_overlap_specs(vindr_dataframe: pd.DataFrame):
    base_specs = build_overlap_specs()
    active_specs = []

    for spec in base_specs:
        found_columns = []

        for candidate_column in spec["vindr_candidates"]:
            if candidate_column in vindr_dataframe.columns:
                found_columns.append(candidate_column)

        if len(found_columns) == 0:
            continue

        active_spec = {
            "eval_name": spec["eval_name"],
            "prediction_labels": spec["prediction_labels"],
            "vindr_columns": found_columns,
        }
        active_specs.append(active_spec)

    if len(active_specs) == 0:
        raise ValueError(
            "Could not find any overlapping evaluation columns in the VinDr labels CSV."
        )

    return active_specs


def compute_vindr_multilabel_ground_truth(row, active_specs):
    ground_truth_values = []

    for spec in active_specs:
        positive_value = 0

        for column_name in spec["vindr_columns"]:
            value = row[column_name]

            if pd.isna(value):
                continue

            if float(value) >= 0.5:
                positive_value = 1
                break

        ground_truth_values.append(int(positive_value))

    return ground_truth_values


class VinDrMultiLabelDataset(Dataset):
    def __init__(self, labels_csv_path: str, vindr_test_dir: str, transform):
        self.transform = transform
        self.samples = []

        annotations_dataframe = load_vindr_wide_annotations(labels_csv_path)
        self.active_specs = build_active_overlap_specs(annotations_dataframe)
        self.eval_label_names = []

        for spec in self.active_specs:
            self.eval_label_names.append(spec["eval_name"])

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

            ground_truth_values = compute_vindr_multilabel_ground_truth(row, self.active_specs)

            self.samples.append(
                {
                    "image_id": image_id,
                    "dicom_path": dicom_path_map[image_id],
                    "y_true": ground_truth_values,
                }
            )

        print("VinDr multi-label images found:", len(self.samples))
        print("VinDr multi-label images skipped:", skipped_count)
        print("Active overlap labels:", self.eval_label_names)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = dicom_to_rgb_pil(sample["dicom_path"])
        image = self.transform(image)

        y_true_tensor = torch.tensor(sample["y_true"], dtype=torch.float32)

        output = {
            "image": image,
            "image_id": sample["image_id"],
            "y_true": y_true_tensor,
        }
        return output


def safe_metric(metric_function, *args, **kwargs):
    try:
        value = metric_function(*args, **kwargs)
        return float(value)
    except Exception:
        return float("nan")


def load_multilabel_model(device: torch.device):
    model = build_densenet121(num_outputs=len(NIH_LABEL_NAMES))
    model = load_checkpoint_into_model(model, MULTILABEL_MODEL_PATH, device)
    model = model.to(device)
    return model


def compute_eval_outputs_from_probabilities(probabilities_tensor, active_specs, thresholds_by_label, nih_label_to_index):
    batch_size = int(probabilities_tensor.shape[0])
    number_of_eval_labels = len(active_specs)

    eval_probabilities = torch.zeros((batch_size, number_of_eval_labels), dtype=torch.float32)
    eval_predictions = torch.zeros((batch_size, number_of_eval_labels), dtype=torch.int64)

    for spec_index in range(number_of_eval_labels):
        spec = active_specs[spec_index]
        prediction_labels = spec["prediction_labels"]

        if len(prediction_labels) == 1:
            label_name = prediction_labels[0]
            label_index = nih_label_to_index[label_name]
            threshold = float(thresholds_by_label[label_name])

            label_probabilities = probabilities_tensor[:, label_index]
            label_predictions = (label_probabilities >= threshold).long()

            eval_probabilities[:, spec_index] = label_probabilities
            eval_predictions[:, spec_index] = label_predictions

        elif len(prediction_labels) == 2:
            first_label = prediction_labels[0]
            second_label = prediction_labels[1]

            first_index = nih_label_to_index[first_label]
            second_index = nih_label_to_index[second_label]

            first_threshold = float(thresholds_by_label[first_label])
            second_threshold = float(thresholds_by_label[second_label])

            first_probabilities = probabilities_tensor[:, first_index]
            second_probabilities = probabilities_tensor[:, second_index]

            combined_probabilities = torch.maximum(first_probabilities, second_probabilities)

            first_predictions = (first_probabilities >= first_threshold).long()
            second_predictions = (second_probabilities >= second_threshold).long()

            combined_predictions = torch.zeros_like(first_predictions)
            combined_predictions[first_predictions == 1] = 1
            combined_predictions[second_predictions == 1] = 1

            eval_probabilities[:, spec_index] = combined_probabilities
            eval_predictions[:, spec_index] = combined_predictions

        else:
            raise ValueError("Unexpected number of prediction labels in overlap spec.")

    return eval_probabilities.cpu().numpy(), eval_predictions.cpu().numpy()


def run_multilabel_baseline_evaluation(model, data_loader, device, thresholds_by_label, active_specs, eval_label_names):
    model.eval()

    nih_label_to_index = build_nih_label_to_index()
    prediction_rows = []

    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            images = batch["image"].to(device)
            image_ids = batch["image_id"]
            y_true = batch["y_true"].cpu().numpy()

            logits = model(images)
            probabilities = torch.sigmoid(logits).cpu()

            eval_probabilities, eval_predictions = compute_eval_outputs_from_probabilities(
                probabilities_tensor=probabilities,
                active_specs=active_specs,
                thresholds_by_label=thresholds_by_label,
                nih_label_to_index=nih_label_to_index,
            )

            for sample_index in range(len(image_ids)):
                row = {
                    "image_id": image_ids[sample_index],
                    "batch_index": batch_index,
                }

                for label_index in range(len(eval_label_names)):
                    label_name = eval_label_names[label_index]

                    row["y_true_" + label_name] = int(y_true[sample_index][label_index])
                    row["prob_" + label_name] = float(eval_probabilities[sample_index][label_index])
                    row["pred_" + label_name] = int(eval_predictions[sample_index][label_index])

                prediction_rows.append(row)

    predictions_dataframe = pd.DataFrame(prediction_rows)
    return predictions_dataframe


def run_multilabel_tta_evaluation(model, data_loader, device, entropy_threshold, thresholds_by_label, active_specs, eval_label_names):
    base_model = clone_model(model)
    base_model.eval()

    nih_label_to_index = build_nih_label_to_index()

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

        with torch.no_grad():
            logits = batch_model(images)
            probabilities = torch.sigmoid(logits).cpu()

        eval_probabilities, eval_predictions = compute_eval_outputs_from_probabilities(
            probabilities_tensor=probabilities,
            active_specs=active_specs,
            thresholds_by_label=thresholds_by_label,
            nih_label_to_index=nih_label_to_index,
        )

        for sample_index in range(len(image_ids)):
            row = {
                "image_id": image_ids[sample_index],
                "batch_index": batch_index,
            }

            for label_index in range(len(eval_label_names)):
                label_name = eval_label_names[label_index]

                row["y_true_" + label_name] = int(y_true[sample_index][label_index])
                row["prob_" + label_name] = float(eval_probabilities[sample_index][label_index])
                row["pred_" + label_name] = int(eval_predictions[sample_index][label_index])

            prediction_rows.append(row)

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

def compute_multilabel_metrics(predictions_dataframe: pd.DataFrame, eval_label_names):
    metric_rows = []

    for label_name in eval_label_names:
        y_true = predictions_dataframe["y_true_" + label_name].to_numpy()
        y_prob = predictions_dataframe["prob_" + label_name].to_numpy()
        y_pred = predictions_dataframe["pred_" + label_name].to_numpy()

        row = {
            "label": label_name,
            "num_images": float(len(predictions_dataframe)),
            "positive_rate": float(np.mean(y_true)),
            "roc_auc": safe_metric(roc_auc_score, y_true, y_prob),
            "average_precision": safe_metric(average_precision_score, y_true, y_prob),
            "accuracy": safe_metric(accuracy_score, y_true, y_pred),
            "precision": safe_metric(precision_score, y_true, y_pred, zero_division=0),
            "recall": safe_metric(recall_score, y_true, y_pred, zero_division=0),
            "f1": safe_metric(f1_score, y_true, y_pred, zero_division=0),
        }

        metric_rows.append(row)

    metrics_dataframe = pd.DataFrame(metric_rows)

    macro_row = {"label": "MACRO_MEAN"}

    metric_columns = [
        "num_images",
        "positive_rate",
        "roc_auc",
        "average_precision",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    for column_name in metric_columns:
        column_values = metrics_dataframe[column_name].to_numpy(dtype=float)
        macro_row[column_name] = float(np.nanmean(column_values))

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
        "roc_auc",
        "average_precision",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    for metric_name in metric_names:
        baseline_column = metric_name + "_baseline"
        tta_column = metric_name + "_tta"
        delta_column = metric_name + "_delta"

        if baseline_column in merged_dataframe.columns and tta_column in merged_dataframe.columns:
            merged_dataframe[delta_column] = merged_dataframe[tta_column] - merged_dataframe[baseline_column]

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
        raise FileNotFoundError(
            "Could not find VinDr image-level labels CSV. "
            "Edit VINDR_LABELS_CSV_CANDIDATES near the top of the script."
        )

    print("Using VinDr labels CSV:", labels_csv_path)

    thresholds_by_label = load_thresholds_by_label(MULTILABEL_THRESHOLDS_JSON)
    print("Loaded multilabel thresholds:", thresholds_by_label)

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
                print("Loaded saved multi-label entropy threshold:", entropy_threshold)
            else:
                print("Saved entropy threshold exists but quantile changed. Recomputing.")
        except Exception:
            print("Could not load saved entropy threshold. Recomputing.")

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

        threshold_model = load_multilabel_model(device)

        entropy_threshold = estimate_entropy_threshold(
            model=threshold_model,
            data_loader=nih_val_loader,
            device=device,
            is_multilabel=True,
            quantile=ENTROPY_QUANTILE,
        )

        print("Estimated multi-label entropy threshold:", entropy_threshold)

        threshold_metadata = {
            "entropy_threshold": float(entropy_threshold),
            "entropy_quantile": float(ENTROPY_QUANTILE),
            "tta_learning_rate": float(TTA_LEARNING_RATE),
            "min_accepted_samples": int(MIN_ACCEPTED_SAMPLES),
            "thresholds_by_label": thresholds_by_label,
        }

        with open(OUTPUT_THRESHOLD_JSON, "w") as json_file:
            json.dump(threshold_metadata, json_file, indent=2)

    # ---------------------------------------------
    # Build VinDr dataset / loader
    # ---------------------------------------------
    vindr_dataset = VinDrMultiLabelDataset(
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
        baseline_model = load_multilabel_model(device)
        baseline_predictions = run_multilabel_baseline_evaluation(
            model=baseline_model,
            data_loader=vindr_loader,
            device=device,
            thresholds_by_label=thresholds_by_label,
            active_specs=vindr_dataset.active_specs,
            eval_label_names=vindr_dataset.eval_label_names,
        )
        baseline_predictions.to_csv(OUTPUT_BASELINE_PREDICTIONS_CSV, index=False)
        print("Ran and saved baseline predictions.")
    elif REUSE_BASELINE_PREDICTIONS and os.path.exists(OUTPUT_BASELINE_PREDICTIONS_CSV):
        baseline_predictions = pd.read_csv(OUTPUT_BASELINE_PREDICTIONS_CSV)
        print("Loaded saved baseline predictions:", OUTPUT_BASELINE_PREDICTIONS_CSV)
    else:
        baseline_model = load_multilabel_model(device)
        baseline_predictions = run_multilabel_baseline_evaluation(
            model=baseline_model,
            data_loader=vindr_loader,
            device=device,
            thresholds_by_label=thresholds_by_label,
            active_specs=vindr_dataset.active_specs,
            eval_label_names=vindr_dataset.eval_label_names,
        )
        baseline_predictions.to_csv(OUTPUT_BASELINE_PREDICTIONS_CSV, index=False)
        print("No saved baseline predictions found, so baseline was run once.")

    # ---------------------------------------------
    # Run TTA
    # ---------------------------------------------
    tta_model = load_multilabel_model(device)
    tta_predictions, tta_log = run_multilabel_tta_evaluation(
        model=tta_model,
        data_loader=vindr_loader,
        device=device,
        entropy_threshold=entropy_threshold,
        thresholds_by_label=thresholds_by_label,
        active_specs=vindr_dataset.active_specs,
        eval_label_names=vindr_dataset.eval_label_names,
    )

    tta_predictions.to_csv(OUTPUT_TTA_PREDICTIONS_CSV, index=False)
    tta_log.to_csv(OUTPUT_TTA_LOG_CSV, index=False)

    print("Updated batches:", int(tta_log["updated"].sum()), "/", len(tta_log))
    print("Mean accepted fraction:", float(tta_log["accepted_fraction"].mean()))
    print("Mean accepted count:", float(tta_log["accepted_count"].mean()))

    # ---------------------------------------------
    # Metrics
    # ---------------------------------------------
    baseline_metrics_dataframe = compute_multilabel_metrics(
        predictions_dataframe=baseline_predictions,
        eval_label_names=vindr_dataset.eval_label_names,
    )
    baseline_metrics_dataframe.to_csv(OUTPUT_BASELINE_METRICS_CSV, index=False)

    tta_metrics_dataframe = compute_multilabel_metrics(
        predictions_dataframe=tta_predictions,
        eval_label_names=vindr_dataset.eval_label_names,
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

    print()
    print("Saved:")
    print(" -", OUTPUT_BASELINE_PREDICTIONS_CSV)
    print(" -", OUTPUT_TTA_PREDICTIONS_CSV)
    print(" -", OUTPUT_TTA_LOG_CSV)
    print(" -", OUTPUT_BASELINE_METRICS_CSV)
    print(" -", OUTPUT_TTA_METRICS_CSV)
    print(" -", OUTPUT_METRICS_COMPARISON_CSV)
    print(" -", OUTPUT_COMPARISON_CSV)
    print(" -", OUTPUT_THRESHOLD_JSON)
    print()
    print("Done.")

if __name__ == "__main__":
    main()