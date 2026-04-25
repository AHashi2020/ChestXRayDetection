import json
import os
import random

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from tta_utils import build_densenet121
from tta_utils import build_file_stem_path_map
from tta_utils import dicom_to_rgb_pil
from tta_utils import find_first_existing_path
from tta_utils import get_eval_transform
from tta_utils import load_checkpoint_into_model


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

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

OUTPUT_PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "vindr_less_labels_predictions.csv")
OUTPUT_METRICS_CSV = os.path.join(PROJECT_ROOT, "vindr_less_labels_metrics.csv")

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0
RANDOM_SEED = 42

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


def compute_metrics(predictions_dataframe: pd.DataFrame, eval_label_names):
    metric_rows = []

    label_index = 0
    while label_index < len(eval_label_names):
        label_name = eval_label_names[label_index]

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
        label_index += 1

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
    print("Loaded less-label thresholds:", thresholds_by_label)

    transform = get_eval_transform(IMAGE_SIZE)

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

    model = load_model(device)
    model.eval()

    prediction_rows = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
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
                while label_index < len(dataset.eval_label_names):
                    label_name = dataset.eval_label_names[label_index]
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

    predictions_dataframe = pd.DataFrame(prediction_rows)
    predictions_dataframe.to_csv(OUTPUT_PREDICTIONS_CSV, index=False)

    metrics_dataframe = compute_metrics(
        predictions_dataframe=predictions_dataframe,
        eval_label_names=dataset.eval_label_names,
    )
    metrics_dataframe.to_csv(OUTPUT_METRICS_CSV, index=False)

    print("Saved:", OUTPUT_PREDICTIONS_CSV)
    print("Saved:", OUTPUT_METRICS_CSV)
    print("Done.")


if __name__ == "__main__":
    main()