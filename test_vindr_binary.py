import os
import numpy as np
import pandas as pd
from PIL import Image

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import densenet121
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


# --------------------------------------------------
# Config
# --------------------------------------------------
vindr_root = "/Users/Annmarie/Desktop/cv_final_code/vindr-cxr_test"
image_dir = os.path.join(vindr_root, "test")
labels_csv = os.path.join(vindr_root, "annotations", "image_labels_test_subset.csv")

model_path = "training_outputs/best_model.pth"
output_csv = "vindr_binary_predictions.csv"

batch_size = 16
num_workers = 0


# --------------------------------------------------
# Helpers
# --------------------------------------------------
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
def strip_extension(file_name):
    lower_name = file_name.lower()

    if lower_name.endswith(".dicom"):
        return file_name[:-6]
    if lower_name.endswith(".dcm"):
        return file_name[:-4]

    return os.path.splitext(file_name)[0]


def get_downloaded_image_ids(folder_path):
    image_ids = set()

    for file_name in os.listdir(folder_path):
        lower_name = file_name.lower()

        if lower_name.endswith(".dicom") or lower_name.endswith(".dcm"):
            image_id = strip_extension(file_name)
            image_ids.add(image_id)

    return image_ids

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


def build_model():
    model = densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    return model


def find_image_id_column(columns):
    column_index = 0
    while column_index < len(columns):
        current_column = columns[column_index]
        normalized = canonicalize_name(current_column)
        if normalized == "imageid":
            return current_column
        column_index += 1

    raise ValueError("Could not find image_id column in image_labels_test.csv")


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

def load_binary_vindr_dataframe(csv_path):
    dataframe = pd.read_csv(csv_path)

    image_id_column = find_image_id_column(list(dataframe.columns))

    label_columns = []
    column_index = 0
    while column_index < len(dataframe.columns):
        current_column = dataframe.columns[column_index]
        if current_column != image_id_column:
            label_columns.append(current_column)
        column_index += 1

    binary_labels = []
    row_index = 0
    while row_index < len(dataframe):
        row = dataframe.iloc[row_index]

        abnormal_sum = 0.0

        label_index = 0
        while label_index < len(label_columns):
            current_column = label_columns[label_index]
            normalized_name = canonicalize_name(current_column)

            if normalized_name != "nofinding":
                current_value = float(row[current_column])
                abnormal_sum += current_value

            label_index += 1

        if abnormal_sum > 0.0:
            binary_labels.append(1.0)
        else:
            binary_labels.append(0.0)

        row_index += 1

    result_dataframe = pd.DataFrame()
    result_dataframe["image_id"] = dataframe[image_id_column]
    result_dataframe["label"] = binary_labels

    return result_dataframe


# --------------------------------------------------
# Dataset
# --------------------------------------------------
class VinDrBinaryDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
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

        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        return image, label, str(image_id)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Loading VinDr labels...", flush=True)
    dataframe = load_binary_vindr_dataframe(labels_csv)
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

    dataset = VinDrBinaryDataset(dataframe, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = get_device()
    print("Using device:", device, flush=True)

    model = build_model().to(device)

    payload = torch.load(model_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    all_image_ids = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, image_ids in dataloader:
            images = images.to(device)

            logits = model(images).squeeze(1)
            probabilities = torch.sigmoid(logits).cpu()

            batch_index = 0
            while batch_index < len(labels):
                all_image_ids.append(str(image_ids[batch_index]))
                all_labels.append(float(labels[batch_index].item()))
                all_probs.append(float(probabilities[batch_index].item()))
                batch_index += 1

    predictions = []
    index = 0
    while index < len(all_probs):
        if all_probs[index] >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
        index += 1

    auroc = roc_auc_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, predictions)
    matrix = confusion_matrix(all_labels, predictions)

    print("VinDr binary AUROC:", round(float(auroc), 4), flush=True)
    print("VinDr binary accuracy:", round(float(accuracy), 4), flush=True)
    print("Confusion matrix:", flush=True)
    print(matrix, flush=True)

    output_dataframe = pd.DataFrame()
    output_dataframe["image_id"] = all_image_ids
    output_dataframe["true_label"] = all_labels
    output_dataframe["pred_prob_abnormal"] = all_probs
    output_dataframe["pred_label"] = predictions
    output_dataframe.to_csv(output_csv, index=False)

    print("Saved predictions to", output_csv, flush=True)


if __name__ == "__main__":
    main()