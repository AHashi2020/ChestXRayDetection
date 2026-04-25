import os
import numpy as np
import pandas as pd
from PIL import Image
import json


import pydicom

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import densenet121
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


# --------------------------------------------------
# Config
# --------------------------------------------------
vindr_root = "/Users/Annmarie/Desktop/cv_final_code/vindr-cxr_test"
image_dir = os.path.join(vindr_root, "test")
labels_csv = os.path.join(vindr_root, "annotations", "image_labels_test_subset.csv")
thresholds_json_path = "training_outputs_labels/best_thresholds.json"

model_path = "training_outputs_labels/best_model.pth"
predictions_output_csv = "vindr_labels_predictions.csv"
metrics_output_csv = "vindr_labels_metrics.csv"

batch_size = 16
num_workers = 0
threshold = 0.5

nih_label_names = [
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


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def strip_extension(file_name):
    lower_name = file_name.lower()

    if lower_name.endswith(".dicom"):
        return file_name[:-6]
    if lower_name.endswith(".dcm"):
        return file_name[:-4]

    return os.path.splitext(file_name)[0]

def load_thresholds():
    if not os.path.exists(thresholds_json_path):
        print("No thresholds json found. Using 0.5 for all labels.", flush=True)
        return None

    with open(thresholds_json_path, "r") as file:
        thresholds_dict = json.load(file)

    print("Loaded thresholds from", thresholds_json_path, flush=True)
    return thresholds_dict


def get_downloaded_image_ids(folder_path):
    image_ids = set()

    for file_name in os.listdir(folder_path):
        lower_name = file_name.lower()

        if lower_name.endswith(".dicom") or lower_name.endswith(".dcm"):
            image_id = strip_extension(file_name)
            image_ids.add(image_id)

    return image_ids


def can_read_dicom(dicom_path):
    try:
        dicom = pydicom.dcmread(dicom_path)

        if "PixelData" not in dicom:
            return False

        _ = dicom.pixel_array
        return True
    except Exception:
        return False


def filter_to_valid_dicoms(dataframe):
    kept_rows = []

    row_index = 0
    while row_index < len(dataframe):
        row = dataframe.iloc[row_index]
        image_id = str(row["image_id"])

        try:
            dicom_path = find_dicom_path(image_id)
        except Exception:
            row_index += 1
            continue

        if can_read_dicom(dicom_path):
            kept_rows.append(row)

        row_index += 1

    if len(kept_rows) == 0:
        return pd.DataFrame(columns=dataframe.columns)

    filtered_dataframe = pd.DataFrame(kept_rows)
    filtered_dataframe = filtered_dataframe.reset_index(drop=True)
    return filtered_dataframe


def canonicalize_name(text):
    text = str(text).strip().lower()

    cleaned_chars = []
    index = 0
    while index < len(text):
        current_char = text[index]
        if current_char.isalnum():
            cleaned_chars.append(current_char)
        index += 1

    return "".join(cleaned_chars)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_labels):
    model = densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_labels)
    return model


def find_image_id_column(columns):
    column_index = 0
    while column_index < len(columns):
        current_column = columns[column_index]
        normalized = canonicalize_name(current_column)

        if normalized == "imageid":
            return current_column

        column_index += 1

    raise ValueError("Could not find image_id column in image_labels_test_subset.csv")


def find_dicom_path(image_id):
    candidate_paths = [
        os.path.join(image_dir, str(image_id)),
        os.path.join(image_dir, str(image_id) + ".dicom"),
        os.path.join(image_dir, str(image_id) + ".dcm"),
    ]

    candidate_index = 0
    while candidate_index < len(candidate_paths):
        current_path = candidate_paths[candidate_index]
        if os.path.exists(current_path):
            return current_path
        candidate_index += 1

    raise FileNotFoundError("Could not find DICOM for image_id: " + str(image_id))


def dicom_to_pil(dicom_path):
    dicom = pydicom.dcmread(dicom_path)

    if "PixelData" not in dicom:
        raise ValueError("DICOM has no PixelData: " + dicom_path)

    pixel_array = dicom.pixel_array

    if hasattr(dicom, "PhotometricInterpretation"):
        if str(dicom.PhotometricInterpretation) == "MONOCHROME1":
            pixel_array = np.max(pixel_array) - pixel_array

    pixel_array = pixel_array.astype(np.float32)

    min_value = float(pixel_array.min())
    max_value = float(pixel_array.max())

    if max_value > min_value:
        pixel_array = (pixel_array - min_value) / (max_value - min_value)
    else:
        pixel_array = np.zeros_like(pixel_array, dtype=np.float32)

    pixel_array = pixel_array * 255.0
    pixel_array = pixel_array.clip(0, 255).astype(np.uint8)

    image = Image.fromarray(pixel_array).convert("RGB")
    return image


def get_column_lookup(dataframe):
    lookup = {}
    column_index = 0

    while column_index < len(dataframe.columns):
        current_column = dataframe.columns[column_index]
        lookup[canonicalize_name(current_column)] = current_column
        column_index += 1

    return lookup


def get_required_column(column_lookup, candidate_names):
    candidate_index = 0
    while candidate_index < len(candidate_names):
        candidate_name = candidate_names[candidate_index]

        if candidate_name in column_lookup:
            return column_lookup[candidate_name]

        candidate_index += 1

    raise ValueError("Could not find matching VinDr column for: " + str(candidate_names))


def get_model_index(label_names, target_name):
    index = 0
    while index < len(label_names):
        if label_names[index] == target_name:
            return index
        index += 1

    raise ValueError("Could not find model label: " + target_name)


def build_overlap_specs(column_lookup):
    overlap_specs = []

    overlap_specs.append({
        "eval_name": "Atelectasis",
        "vindr_column": get_required_column(column_lookup, ["atelectasis"]),
        "model_indices": [get_model_index(nih_label_names, "Atelectasis")],
    })

    overlap_specs.append({
        "eval_name": "Cardiomegaly",
        "vindr_column": get_required_column(column_lookup, ["cardiomegaly"]),
        "model_indices": [get_model_index(nih_label_names, "Cardiomegaly")],
    })

    overlap_specs.append({
        "eval_name": "Pleural_Effusion",
        "vindr_column": get_required_column(column_lookup, ["pleuraleffusion", "effusion"]),
        "model_indices": [get_model_index(nih_label_names, "Effusion")],
    })

    overlap_specs.append({
        "eval_name": "Infiltration",
        "vindr_column": get_required_column(column_lookup, ["infiltration"]),
        "model_indices": [get_model_index(nih_label_names, "Infiltration")],
    })

    overlap_specs.append({
        "eval_name": "Pneumonia",
        "vindr_column": get_required_column(column_lookup, ["pneumonia"]),
        "model_indices": [get_model_index(nih_label_names, "Pneumonia")],
    })

    overlap_specs.append({
        "eval_name": "Pneumothorax",
        "vindr_column": get_required_column(column_lookup, ["pneumothorax"]),
        "model_indices": [get_model_index(nih_label_names, "Pneumothorax")],
    })

    overlap_specs.append({
        "eval_name": "Consolidation",
        "vindr_column": get_required_column(column_lookup, ["consolidation"]),
        "model_indices": [get_model_index(nih_label_names, "Consolidation")],
    })

    overlap_specs.append({
        "eval_name": "Edema",
        "vindr_column": get_required_column(column_lookup, ["edema"]),
        "model_indices": [get_model_index(nih_label_names, "Edema")],
    })

    overlap_specs.append({
        "eval_name": "Emphysema",
        "vindr_column": get_required_column(column_lookup, ["emphysema"]),
        "model_indices": [get_model_index(nih_label_names, "Emphysema")],
    })

    overlap_specs.append({
        "eval_name": "Pulmonary_Fibrosis",
        "vindr_column": get_required_column(column_lookup, ["pulmonaryfibrosis", "fibrosis"]),
        "model_indices": [get_model_index(nih_label_names, "Fibrosis")],
    })

    overlap_specs.append({
        "eval_name": "Pleural_Thickening",
        "vindr_column": get_required_column(column_lookup, ["pleuralthickening"]),
        "model_indices": [get_model_index(nih_label_names, "Pleural_Thickening")],
    })

    overlap_specs.append({
        "eval_name": "Nodule_Mass",
        "vindr_column": get_required_column(column_lookup, ["nodulemass"]),
        "model_indices": [
            get_model_index(nih_label_names, "Mass"),
            get_model_index(nih_label_names, "Nodule"),
        ],
    })

    return overlap_specs


def load_vindr_overlap_dataframe(csv_path):
    dataframe = pd.read_csv(csv_path)
    image_id_column = find_image_id_column(list(dataframe.columns))
    column_lookup = get_column_lookup(dataframe)
    overlap_specs = build_overlap_specs(column_lookup)

    result_dataframe = pd.DataFrame()
    result_dataframe["image_id"] = dataframe[image_id_column]

    spec_index = 0
    while spec_index < len(overlap_specs):
        current_spec = overlap_specs[spec_index]
        result_dataframe[current_spec["eval_name"]] = dataframe[current_spec["vindr_column"]].astype(float)
        spec_index += 1

    return result_dataframe, overlap_specs


def compute_label_metrics(true_values, pred_probs, pred_labels):
    metrics = {}

    unique_values = set()
    index = 0
    while index < len(true_values):
        unique_values.add(float(true_values[index]))
        index += 1

    if len(unique_values) < 2:
        metrics["auroc"] = None
    else:
        metrics["auroc"] = float(roc_auc_score(true_values, pred_probs))

    metrics["accuracy"] = float(accuracy_score(true_values, pred_labels))
    metrics["precision"] = float(precision_score(true_values, pred_labels, zero_division=0))
    metrics["recall"] = float(recall_score(true_values, pred_labels, zero_division=0))
    metrics["f1"] = float(f1_score(true_values, pred_labels, zero_division=0))
    metrics["positives"] = int(np.sum(true_values == 1))
    metrics["negatives"] = int(np.sum(true_values == 0))
    metrics["predicted_positives"] = int(np.sum(pred_labels == 1))
    metrics["predicted_negatives"] = int(np.sum(pred_labels == 0))

    return metrics


# --------------------------------------------------
# Dataset
# --------------------------------------------------
class VinDrLabelsDataset(Dataset):
    def __init__(self, dataframe, overlap_specs, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.overlap_specs = overlap_specs
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_id = row["image_id"]
        dicom_path = find_dicom_path(image_id)
        image = dicom_to_pil(dicom_path)

        if self.transform is not None:
            image = self.transform(image)

        label_values = []
        spec_index = 0
        while spec_index < len(self.overlap_specs):
            eval_name = self.overlap_specs[spec_index]["eval_name"]
            label_values.append(float(row[eval_name]))
            spec_index += 1

        label_tensor = torch.tensor(label_values, dtype=torch.float32)

        return image, label_tensor, str(image_id)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Loading VinDr overlap labels...", flush=True)
    dataframe, overlap_specs = load_vindr_overlap_dataframe(labels_csv)
    print("Loaded", len(dataframe), "VinDr test rows.", flush=True)

    downloaded_image_ids = get_downloaded_image_ids(image_dir)
    print("Downloaded image files found:", len(downloaded_image_ids), flush=True)

    dataframe = dataframe[dataframe["image_id"].astype(str).isin(downloaded_image_ids)].copy()
    dataframe = dataframe.reset_index(drop=True)
    print("Rows after filtering to downloaded images:", len(dataframe), flush=True)

    print("Checking DICOM readability...", flush=True)
    dataframe = filter_to_valid_dicoms(dataframe)
    print("Rows after filtering to readable DICOMs:", len(dataframe), flush=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = VinDrLabelsDataset(dataframe, overlap_specs, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = get_device()
    print("Using device:", device, flush=True)

    model = build_model(len(nih_label_names)).to(device)

    payload = torch.load(model_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    all_image_ids = []
    all_true = []
    all_pred_probs = []

    with torch.no_grad():
        for images, labels, image_ids in dataloader:
            images = images.to(device)

            logits = model(images)
            probabilities = torch.sigmoid(logits).cpu()

            batch_size_local = len(labels)
            batch_index = 0
            while batch_index < batch_size_local:
                row_predictions = []

                spec_index = 0
                while spec_index < len(overlap_specs):
                    model_indices = overlap_specs[spec_index]["model_indices"]

                    best_prob = 0.0
                    index_in_group = 0
                    while index_in_group < len(model_indices):
                        current_model_index = model_indices[index_in_group]
                        current_prob = float(probabilities[batch_index][current_model_index].item())

                        if current_prob > best_prob:
                            best_prob = current_prob

                        index_in_group += 1

                    row_predictions.append(best_prob)
                    spec_index += 1

                all_image_ids.append(str(image_ids[batch_index]))
                all_true.append(labels[batch_index].numpy())
                all_pred_probs.append(np.array(row_predictions, dtype=np.float32))

                batch_index += 1

    true_array = np.stack(all_true, axis=0)
    pred_prob_array = np.stack(all_pred_probs, axis=0)
    thresholds_dict = load_thresholds()
    pred_label_array = np.zeros_like(pred_prob_array, dtype=np.int32)
    metrics_rows = []
    valid_aurocs = []
    valid_accuracies = []
    valid_precisions = []
    valid_recalls = []
    valid_f1s = []

    # First: apply threshold for each label
    spec_index = 0
    while spec_index < len(overlap_specs):
        eval_name = overlap_specs[spec_index]["eval_name"]

        threshold_name = eval_name

        if eval_name == "Pleural_Effusion":
            threshold_name = "Effusion"
        elif eval_name == "Pulmonary_Fibrosis":
            threshold_name = "Fibrosis"
        elif eval_name == "Nodule_Mass":
            threshold_name = None

        if thresholds_dict is None:
            current_threshold = 0.5
        else:
            if threshold_name is None:
                mass_threshold = float(thresholds_dict.get("Mass", 0.5))
                nodule_threshold = float(thresholds_dict.get("Nodule", 0.5))
                current_threshold = min(mass_threshold, nodule_threshold)
            else:
                current_threshold = float(thresholds_dict.get(threshold_name, 0.5))

        row_index = 0
        while row_index < pred_prob_array.shape[0]:
            if pred_prob_array[row_index, spec_index] >= current_threshold:
                pred_label_array[row_index, spec_index] = 1
            else:
                pred_label_array[row_index, spec_index] = 0
            row_index += 1

        print(eval_name, "- threshold used:", current_threshold, flush=True)
        spec_index += 1


    # Second: compute metrics for each label
    spec_index = 0
    while spec_index < len(overlap_specs):
        eval_name = overlap_specs[spec_index]["eval_name"]

        current_true = true_array[:, spec_index]
        current_prob = pred_prob_array[:, spec_index]
        current_pred = pred_label_array[:, spec_index]

        metrics = compute_label_metrics(current_true, current_prob, current_pred)

        if metrics["auroc"] is not None:
            valid_aurocs.append(metrics["auroc"])

        valid_accuracies.append(metrics["accuracy"])
        valid_precisions.append(metrics["precision"])
        valid_recalls.append(metrics["recall"])
        valid_f1s.append(metrics["f1"])

        print(eval_name, flush=True)
        print("  AUROC:", metrics["auroc"], flush=True)
        print("  Accuracy:", round(metrics["accuracy"], 4), flush=True)
        print("  Precision:", round(metrics["precision"], 4), flush=True)
        print("  Recall:", round(metrics["recall"], 4), flush=True)
        print("  F1:", round(metrics["f1"], 4), flush=True)
        print("  Positives:", metrics["positives"], flush=True)
        print("  Negatives:", metrics["negatives"], flush=True)

        row = {
            "label": eval_name,
            "auroc": metrics["auroc"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "positives": metrics["positives"],
            "negatives": metrics["negatives"],
            "predicted_positives": metrics["predicted_positives"],
            "predicted_negatives": metrics["predicted_negatives"],
        }
        metrics_rows.append(row)

        spec_index += 1


    # Third: compute macro averages after lists are filled
    macro_auroc = None
    if len(valid_aurocs) > 0:
        macro_auroc = float(sum(valid_aurocs) / len(valid_aurocs))

    macro_accuracy = float(sum(valid_accuracies) / len(valid_accuracies))
    macro_precision = float(sum(valid_precisions) / len(valid_precisions))
    macro_recall = float(sum(valid_recalls) / len(valid_recalls))
    macro_f1 = float(sum(valid_f1s) / len(valid_f1s))

    print("", flush=True)
    print("Overall summary", flush=True)
    print("Macro AUROC:", macro_auroc, flush=True)
    print("Macro Accuracy:", round(macro_accuracy, 4), flush=True)
    print("Macro Precision:", round(macro_precision, 4), flush=True)
    print("Macro Recall:", round(macro_recall, 4), flush=True)
    print("Macro F1:", round(macro_f1, 4), flush=True)

    metrics_dataframe = pd.DataFrame(metrics_rows)
    summary_row = {
        "label": "MACRO_AVERAGE",
        "auroc": macro_auroc,
        "accuracy": macro_accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "positives": None,
        "negatives": None,
        "predicted_positives": None,
        "predicted_negatives": None,
    }
    metrics_dataframe = pd.concat(
        [metrics_dataframe, pd.DataFrame([summary_row])],
        ignore_index=True,
    )
    metrics_dataframe.to_csv(metrics_output_csv, index=False)
    print("Saved metrics to", metrics_output_csv, flush=True)

    predictions_dataframe = pd.DataFrame()
    predictions_dataframe["image_id"] = all_image_ids

    spec_index = 0
    while spec_index < len(overlap_specs):
        eval_name = overlap_specs[spec_index]["eval_name"]
        predictions_dataframe["true_" + eval_name] = true_array[:, spec_index]
        predictions_dataframe["pred_prob_" + eval_name] = pred_prob_array[:, spec_index]
        predictions_dataframe["pred_label_" + eval_name] = pred_label_array[:, spec_index]
        spec_index += 1

    predictions_dataframe.to_csv(predictions_output_csv, index=False)
    print("Saved predictions to", predictions_output_csv, flush=True)


if __name__ == "__main__":
    main()