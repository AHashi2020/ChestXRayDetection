import json
import os
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
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
from tta_utils import clone_model
from tta_utils import configure_model_for_tta
from tta_utils import dicom_to_rgb_pil
from tta_utils import find_first_existing_path
from tta_utils import get_eval_transform
from tta_utils import load_checkpoint_into_model


# --------------------------------------------------
# Config
# --------------------------------------------------
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

MULTILABEL_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "training_outputs_labels",
    "best_model.pth",
)
MULTILABEL_THRESHOLDS_JSON = os.path.join(
    PROJECT_ROOT,
    "training_outputs_labels",
    "best_thresholds.json",
)

# Existing files from your old script.
EXISTING_BASELINE_PREDICTIONS_CSV = os.path.join(
    PROJECT_ROOT,
    "vindr_labels_predictions_baseline.csv",
)
EXISTING_OLD_TTA_PREDICTIONS_CSV = os.path.join(
    PROJECT_ROOT,
    "vindr_labels_predictions_tta.csv",
)
EXISTING_OLD_TTA_LOG_CSV = os.path.join(
    PROJECT_ROOT,
    "vindr_labels_tta_log.csv",
)
EXISTING_THRESHOLD_JSON = os.path.join(
    PROJECT_ROOT,
    "vindr_labels_entropy_threshold.json",
)

# New outputs for this compare script.
OUTPUT_BASELINE_PREDICTIONS_CSV = os.path.join(
    PROJECT_ROOT,
    "vindr_labels_predictions_baseline_compare.csv",
)
OUTPUT_OLD_TTA_PREDICTIONS_CSV = os.path.join(
    PROJECT_ROOT,
    "vindr_labels_predictions_old_tta_compare.csv",
)
OUTPUT_NEW_TTA_PREDICTIONS_CSV = os.path.join(
    PROJECT_ROOT,
    "vindr_labels_predictions_tta2.csv",
)

OUTPUT_OLD_TTA_LOG_CSV = os.path.join(
    PROJECT_ROOT,
    "vindr_labels_old_tta_compare_log.csv",
)
OUTPUT_NEW_TTA_LOG_CSV = os.path.join(
    PROJECT_ROOT,
    "vindr_labels_tta2_log.csv",
)

OUTPUT_BASELINE_METRICS_CSV = os.path.join(
    PROJECT_ROOT,
    "vindr_labels_metrics_baseline_compare.csv",
)
OUTPUT_OLD_TTA_METRICS_CSV = os.path.join(
    PROJECT_ROOT,
    "vindr_labels_metrics_old_tta_compare.csv",
)
OUTPUT_NEW_TTA_METRICS_CSV = os.path.join(
    PROJECT_ROOT,
    "vindr_labels_metrics_tta2.csv",
)

OUTPUT_COMPARISON_CSV = os.path.join(
    PROJECT_ROOT,
    "vindr_labels_metrics_baseline_old_tta_tta2_comparison.csv",
)
OUTPUT_THRESHOLD_JSON = os.path.join(
    PROJECT_ROOT,
    "vindr_labels_entropy_threshold_tta2.json",
)

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0
RANDOM_SEED = 42

# Entropy gate
ENTROPY_QUANTILE = 0.15
MIN_ACCEPTED_SAMPLES = 2
MAX_ADAPT_SAMPLES = 2

# Old BN-only TTA learning rate
OLD_TTA_LEARNING_RATE = 1e-6

# New head + BN TTA learning rate
# Start smaller because more parameters are being updated.
NEW_TTA_LEARNING_RATE = 5e-7

# Reuse options
REUSE_SAVED_THRESHOLD = True
REUSE_EXISTING_BASELINE = True
REUSE_EXISTING_OLD_TTA = True

RUN_BASELINE_IF_NOT_FOUND = True
RUN_OLD_TTA_IF_NOT_FOUND = True
RUN_NEW_TTA = True

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

EPSILON = 1e-6


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Small helpers
# --------------------------------------------------
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


def safe_metric(metric_function, *args, **kwargs):
    try:
        value = metric_function(*args, **kwargs)
        return float(value)
    except Exception:
        return float("nan")


def count_parameter_list_numel(parameter_list) -> int:
    total = 0

    for parameter in parameter_list:
        total += parameter.numel()

    return total


# --------------------------------------------------
# Datasets
# --------------------------------------------------
class NIHImageOnlyDataset(Dataset):
    """
    Used only to estimate the entropy threshold on NIH validation images.
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

            ground_truth_values = compute_vindr_multilabel_ground_truth(
                row,
                self.active_specs,
            )

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


# --------------------------------------------------
# Label helpers
# --------------------------------------------------
def load_thresholds_by_label(thresholds_json_path: str) -> dict:
    thresholds_by_label = {}

    for label_name in NIH_LABEL_NAMES:
        thresholds_by_label[label_name] = 0.50

    if not os.path.exists(thresholds_json_path):
        print("Threshold JSON not found. Using 0.50 for all labels.")
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
    Only evaluate cleaner NIH <-> VinDr overlaps.
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


# --------------------------------------------------
# Model helpers
# --------------------------------------------------
def load_multilabel_model(device: torch.device):
    model = build_densenet121(num_outputs=len(NIH_LABEL_NAMES))
    model = load_checkpoint_into_model(model, MULTILABEL_MODEL_PATH, device)
    model = model.to(device)
    return model


def compute_eval_outputs_from_probabilities(
    probabilities_tensor,
    active_specs,
    thresholds_by_label,
    nih_label_to_index,
):
    batch_size = int(probabilities_tensor.shape[0])
    number_of_eval_labels = len(active_specs)

    eval_probabilities = torch.zeros(
        (batch_size, number_of_eval_labels),
        dtype=torch.float32,
    )
    eval_predictions = torch.zeros(
        (batch_size, number_of_eval_labels),
        dtype=torch.int64,
    )

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

            combined_probabilities = torch.maximum(
                first_probabilities,
                second_probabilities,
            )

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


# --------------------------------------------------
# Entropy helpers
# --------------------------------------------------
def build_tta_index_groups(active_specs, nih_label_to_index):
    """
    Build the exact output groups used for overlap-based VinDr evaluation.
    Example:
    - Pleural_Effusion -> [Effusion]
    - Nodule_Mass -> [Mass, Nodule]
    """
    tta_index_groups = []
    tta_group_names = []

    for spec in active_specs:
        group_indices = []

        for label_name in spec["prediction_labels"]:
            group_indices.append(nih_label_to_index[label_name])

        tta_index_groups.append(group_indices)
        tta_group_names.append(spec["eval_name"])

    return tta_index_groups, tta_group_names


def overlap_entropy_from_logits(logits: torch.Tensor, tta_index_groups):
    """
    Compute entropy only on the overlap groups actually used in VinDr evaluation.
    For grouped labels like Nodule_Mass, use max probability across the group.
    Returns shape [batch_size].
    """
    probabilities = torch.sigmoid(logits)

    overlap_probabilities = []

    for group_indices in tta_index_groups:
        if len(group_indices) == 1:
            group_probability = probabilities[:, group_indices[0]]
        else:
            group_probability = torch.amax(probabilities[:, group_indices], dim=1)

        overlap_probabilities.append(group_probability)

    overlap_probabilities = torch.stack(overlap_probabilities, dim=1)

    entropy = -(
        overlap_probabilities * torch.log(overlap_probabilities + EPSILON)
        + (1.0 - overlap_probabilities) * torch.log(1.0 - overlap_probabilities + EPSILON)
    )

    sample_entropy = entropy.mean(dim=1)
    return sample_entropy


def estimate_overlap_entropy_threshold(
    model,
    data_loader,
    device,
    tta_index_groups,
    quantile: float,
):
    """
    Estimate the entropy gate on NIH validation images.
    This keeps the gate independent from VinDr labels.
    """
    model.eval()
    all_entropies = []

    with torch.inference_mode():
        for batch in data_loader:
            images = batch["image"].to(device)
            logits = model(images)
            batch_entropy = overlap_entropy_from_logits(logits, tta_index_groups)
            all_entropies.append(batch_entropy.cpu())

    all_entropies = torch.cat(all_entropies, dim=0)
    threshold = torch.quantile(all_entropies, quantile).item()
    return float(threshold)


@torch.enable_grad()
def tta_step_multilabel_overlap(
    model,
    images,
    optimizer,
    entropy_threshold,
    tta_index_groups,
    min_accepted_samples=2,
    max_adapt_samples=4,
):
    """
    One TTA step using overlap-label entropy only.
    We adapt only on accepted low-entropy samples.
    """
    logits = model(images)
    sample_entropy = overlap_entropy_from_logits(logits, tta_index_groups)

    accepted_indices = torch.nonzero(
        sample_entropy <= entropy_threshold,
        as_tuple=False,
    ).view(-1)

    accepted_count = int(accepted_indices.numel())
    batch_size = int(sample_entropy.shape[0])

    selected_count = accepted_count

    # If too many samples are accepted, keep only the lowest-entropy ones.
    if accepted_count > max_adapt_samples:
        accepted_entropy = sample_entropy[accepted_indices]
        _, keep_order = torch.topk(
            accepted_entropy,
            k=max_adapt_samples,
            largest=False,
        )
        accepted_indices = accepted_indices[keep_order]
        selected_count = int(accepted_indices.numel())

    result = {
        "accepted_count": accepted_count,
        "selected_count": selected_count,
        "batch_size": batch_size,
        "accepted_fraction": accepted_count / max(batch_size, 1),
        "mean_entropy": float(sample_entropy.mean().item()),
        "loss": None,
        "updated": False,
    }

    if selected_count < min_accepted_samples:
        return result

    # Entropy minimization loss on accepted samples only.
    loss = sample_entropy[accepted_indices].mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    result["loss"] = float(loss.item())
    result["updated"] = True
    return result


# --------------------------------------------------
# New TTA method: classifier head + BN
# --------------------------------------------------
def configure_model_for_head_and_bn_tta(model: nn.Module):
    """
    New TTA method:
    - update BatchNorm affine parameters
    - update classifier head
    - freeze everything else

    This is stronger than BN-only, but much safer than full-model TTA.
    """
    model.train()

    # Freeze every parameter first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    tta_parameters = []

    # Unfreeze classifier head.
    if hasattr(model, "classifier"):
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
            tta_parameters.append(parameter)
    else:
        raise ValueError("Expected DenseNet model to have .classifier")

    # Unfreeze BN affine parameters.
    # Also disable running stats so BN uses current batch stats.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

            if module.weight is not None:
                module.weight.requires_grad = True
                tta_parameters.append(module.weight)

            if module.bias is not None:
                module.bias.requires_grad = True
                tta_parameters.append(module.bias)

    if len(tta_parameters) == 0:
        raise ValueError("No trainable parameters selected for head + BN TTA.")

    return model, tta_parameters


def configure_model_for_old_bn_tta(model: nn.Module):
    """
    Wrapper around your existing BN-only TTA setup from tta_utils.
    """
    model, tta_parameters = configure_model_for_tta(model)

    if len(tta_parameters) == 0:
        raise ValueError("No trainable parameters returned for old BN-only TTA.")

    return model, tta_parameters


# --------------------------------------------------
# Evaluation functions
# --------------------------------------------------
def run_multilabel_baseline_evaluation(
    model,
    data_loader,
    device,
    thresholds_by_label,
    active_specs,
    eval_label_names,
):
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


def run_multilabel_tta_evaluation(
    model,
    data_loader,
    device,
    entropy_threshold,
    thresholds_by_label,
    active_specs,
    eval_label_names,
    tta_index_groups,
    configure_tta_function,
    tta_learning_rate,
    tta_mode_name,
):
    """
    Generic TTA runner so we can run:
    - old BN-only TTA
    - new head + BN TTA

    Both use the same entropy gate and same overlap evaluation.
    """
    base_model = clone_model(model)
    base_model.eval()

    nih_label_to_index = build_nih_label_to_index()

    prediction_rows = []
    log_rows = []

    for batch_index, batch in enumerate(data_loader):
        # Episodic TTA:
        # every batch starts from the original source model again.
        batch_model = clone_model(base_model)
        batch_model = batch_model.to(device)

        batch_model, tta_parameters = configure_tta_function(batch_model)
        optimizer = Adam(tta_parameters, lr=tta_learning_rate)

        trainable_parameter_count = count_parameter_list_numel(tta_parameters)

        images = batch["image"].to(device)
        image_ids = batch["image_id"]
        y_true = batch["y_true"].cpu().numpy()

        step_result = tta_step_multilabel_overlap(
            model=batch_model,
            images=images,
            optimizer=optimizer,
            entropy_threshold=entropy_threshold,
            tta_index_groups=tta_index_groups,
            min_accepted_samples=MIN_ACCEPTED_SAMPLES,
            max_adapt_samples=MAX_ADAPT_SAMPLES,
        )

        with torch.inference_mode():
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
                "tta_mode": tta_mode_name,
                "batch_index": batch_index,
                "trainable_parameter_count": trainable_parameter_count,
                "accepted_count": step_result["accepted_count"],
                "selected_count": step_result["selected_count"],
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


def add_metric_suffix(metrics_dataframe: pd.DataFrame, suffix: str):
    rename_map = {}

    for column_name in metrics_dataframe.columns:
        if column_name == "label":
            continue

        rename_map[column_name] = column_name + "_" + suffix

    renamed_dataframe = metrics_dataframe.rename(columns=rename_map)
    return renamed_dataframe


def build_three_way_metrics_comparison(
    baseline_metrics_dataframe: pd.DataFrame,
    old_tta_metrics_dataframe: pd.DataFrame,
    new_tta_metrics_dataframe: pd.DataFrame,
):
    baseline_renamed = add_metric_suffix(baseline_metrics_dataframe, "baseline")
    old_tta_renamed = add_metric_suffix(old_tta_metrics_dataframe, "old_tta")
    new_tta_renamed = add_metric_suffix(new_tta_metrics_dataframe, "tta2")

    comparison_dataframe = baseline_renamed.merge(
        old_tta_renamed,
        on="label",
        how="outer",
    )

    comparison_dataframe = comparison_dataframe.merge(
        new_tta_renamed,
        on="label",
        how="outer",
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
        old_tta_column = metric_name + "_old_tta"
        new_tta_column = metric_name + "_tta2"

        if baseline_column in comparison_dataframe.columns and old_tta_column in comparison_dataframe.columns:
            comparison_dataframe[metric_name + "_delta_old_minus_baseline"] = (
                comparison_dataframe[old_tta_column] - comparison_dataframe[baseline_column]
            )

        if baseline_column in comparison_dataframe.columns and new_tta_column in comparison_dataframe.columns:
            comparison_dataframe[metric_name + "_delta_tta2_minus_baseline"] = (
                comparison_dataframe[new_tta_column] - comparison_dataframe[baseline_column]
            )

        if old_tta_column in comparison_dataframe.columns and new_tta_column in comparison_dataframe.columns:
            comparison_dataframe[metric_name + "_delta_tta2_minus_old_tta"] = (
                comparison_dataframe[new_tta_column] - comparison_dataframe[old_tta_column]
            )

    return comparison_dataframe


def print_macro_summary(metrics_dataframe: pd.DataFrame, title: str):
    macro_dataframe = metrics_dataframe[metrics_dataframe["label"] == "MACRO_MEAN"]

    if len(macro_dataframe) == 0:
        print(title, ": MACRO_MEAN row not found")
        return

    macro_row = macro_dataframe.iloc[0]

    print()
    print(title)
    print("  Macro ROC-AUC:", macro_row["roc_auc"])
    print("  Macro AP     :", macro_row["average_precision"])
    print("  Macro Prec   :", macro_row["precision"])
    print("  Macro Recall :", macro_row["recall"])
    print("  Macro F1     :", macro_row["f1"])


# --------------------------------------------------
# Main
# --------------------------------------------------
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
    print("Loaded thresholds:", thresholds_by_label)

    transform = get_eval_transform(IMAGE_SIZE)

    # Build VinDr dataset first.
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

    nih_label_to_index = build_nih_label_to_index()
    tta_index_groups, tta_group_names = build_tta_index_groups(
        vindr_dataset.active_specs,
        nih_label_to_index,
    )

    print("TTA overlap groups:", tta_group_names)

    # ---------------------------------------------
    # Load or estimate entropy threshold
    # ---------------------------------------------
    entropy_threshold = None

    if REUSE_SAVED_THRESHOLD:
        threshold_candidates = [
            OUTPUT_THRESHOLD_JSON,
            EXISTING_THRESHOLD_JSON,
        ]

        for threshold_path in threshold_candidates:
            if not os.path.exists(threshold_path):
                continue

            try:
                with open(threshold_path, "r") as json_file:
                    threshold_metadata = json.load(json_file)

                saved_quantile = threshold_metadata.get("entropy_quantile", None)
                saved_groups = threshold_metadata.get("tta_group_names", None)

                if saved_quantile == ENTROPY_QUANTILE and saved_groups == tta_group_names:
                    entropy_threshold = float(threshold_metadata["entropy_threshold"])
                    print("Loaded saved entropy threshold from:", threshold_path)
                    print("Entropy threshold:", entropy_threshold)
                    break
            except Exception:
                pass

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

        entropy_threshold = estimate_overlap_entropy_threshold(
            model=threshold_model,
            data_loader=nih_val_loader,
            device=device,
            tta_index_groups=tta_index_groups,
            quantile=ENTROPY_QUANTILE,
        )

        print("Estimated entropy threshold:", entropy_threshold)

    threshold_metadata = {
        "entropy_threshold": float(entropy_threshold),
        "entropy_quantile": float(ENTROPY_QUANTILE),
        "min_accepted_samples": int(MIN_ACCEPTED_SAMPLES),
        "max_adapt_samples": int(MAX_ADAPT_SAMPLES),
        "old_tta_learning_rate": float(OLD_TTA_LEARNING_RATE),
        "new_tta_learning_rate": float(NEW_TTA_LEARNING_RATE),
        "tta_group_names": tta_group_names,
        "thresholds_by_label": thresholds_by_label,
    }

    with open(OUTPUT_THRESHOLD_JSON, "w") as json_file:
        json.dump(threshold_metadata, json_file, indent=2)

    # ---------------------------------------------
    # Baseline
    # ---------------------------------------------
    baseline_predictions = None

    if REUSE_EXISTING_BASELINE and os.path.exists(EXISTING_BASELINE_PREDICTIONS_CSV):
        baseline_predictions = pd.read_csv(EXISTING_BASELINE_PREDICTIONS_CSV)
        print("Loaded existing baseline predictions:", EXISTING_BASELINE_PREDICTIONS_CSV)
    elif RUN_BASELINE_IF_NOT_FOUND:
        baseline_model = load_multilabel_model(device)

        baseline_predictions = run_multilabel_baseline_evaluation(
            model=baseline_model,
            data_loader=vindr_loader,
            device=device,
            thresholds_by_label=thresholds_by_label,
            active_specs=vindr_dataset.active_specs,
            eval_label_names=vindr_dataset.eval_label_names,
        )
        print("Ran baseline evaluation.")
    else:
        raise ValueError("Baseline predictions are required but were not loaded or run.")

    baseline_predictions.to_csv(OUTPUT_BASELINE_PREDICTIONS_CSV, index=False)

    # ---------------------------------------------
    # Old BN-only TTA
    # ---------------------------------------------
    old_tta_predictions = None
    old_tta_log = None

    if REUSE_EXISTING_OLD_TTA and os.path.exists(EXISTING_OLD_TTA_PREDICTIONS_CSV):
        old_tta_predictions = pd.read_csv(EXISTING_OLD_TTA_PREDICTIONS_CSV)
        print("Loaded existing old TTA predictions:", EXISTING_OLD_TTA_PREDICTIONS_CSV)

        if os.path.exists(EXISTING_OLD_TTA_LOG_CSV):
            old_tta_log = pd.read_csv(EXISTING_OLD_TTA_LOG_CSV)
            print("Loaded existing old TTA log:", EXISTING_OLD_TTA_LOG_CSV)
    elif RUN_OLD_TTA_IF_NOT_FOUND:
        old_tta_model = load_multilabel_model(device)

        old_tta_predictions, old_tta_log = run_multilabel_tta_evaluation(
            model=old_tta_model,
            data_loader=vindr_loader,
            device=device,
            entropy_threshold=entropy_threshold,
            thresholds_by_label=thresholds_by_label,
            active_specs=vindr_dataset.active_specs,
            eval_label_names=vindr_dataset.eval_label_names,
            tta_index_groups=tta_index_groups,
            configure_tta_function=configure_model_for_old_bn_tta,
            tta_learning_rate=OLD_TTA_LEARNING_RATE,
            tta_mode_name="old_bn_only_tta",
        )
        print("Ran old BN-only TTA.")
    else:
        raise ValueError("Old TTA predictions are required but were not loaded or run.")

    old_tta_predictions.to_csv(OUTPUT_OLD_TTA_PREDICTIONS_CSV, index=False)

    if old_tta_log is not None:
        old_tta_log.to_csv(OUTPUT_OLD_TTA_LOG_CSV, index=False)

    # ---------------------------------------------
    # New head + BN TTA
    # ---------------------------------------------
    if not RUN_NEW_TTA:
        raise ValueError("RUN_NEW_TTA is False. This script is meant to run the new TTA.")

    new_tta_model = load_multilabel_model(device)

    new_tta_predictions, new_tta_log = run_multilabel_tta_evaluation(
        model=new_tta_model,
        data_loader=vindr_loader,
        device=device,
        entropy_threshold=entropy_threshold,
        thresholds_by_label=thresholds_by_label,
        active_specs=vindr_dataset.active_specs,
        eval_label_names=vindr_dataset.eval_label_names,
        tta_index_groups=tta_index_groups,
        configure_tta_function=configure_model_for_head_and_bn_tta,
        tta_learning_rate=NEW_TTA_LEARNING_RATE,
        tta_mode_name="head_plus_bn_tta",
    )

    new_tta_predictions.to_csv(OUTPUT_NEW_TTA_PREDICTIONS_CSV, index=False)
    new_tta_log.to_csv(OUTPUT_NEW_TTA_LOG_CSV, index=False)

    print("Ran new head + BN TTA.")

    # ---------------------------------------------
    # Metrics
    # ---------------------------------------------
    baseline_metrics_dataframe = compute_multilabel_metrics(
        predictions_dataframe=baseline_predictions,
        eval_label_names=vindr_dataset.eval_label_names,
    )
    baseline_metrics_dataframe.to_csv(OUTPUT_BASELINE_METRICS_CSV, index=False)

    old_tta_metrics_dataframe = compute_multilabel_metrics(
        predictions_dataframe=old_tta_predictions,
        eval_label_names=vindr_dataset.eval_label_names,
    )
    old_tta_metrics_dataframe.to_csv(OUTPUT_OLD_TTA_METRICS_CSV, index=False)

    new_tta_metrics_dataframe = compute_multilabel_metrics(
        predictions_dataframe=new_tta_predictions,
        eval_label_names=vindr_dataset.eval_label_names,
    )
    new_tta_metrics_dataframe.to_csv(OUTPUT_NEW_TTA_METRICS_CSV, index=False)

    comparison_dataframe = build_three_way_metrics_comparison(
        baseline_metrics_dataframe=baseline_metrics_dataframe,
        old_tta_metrics_dataframe=old_tta_metrics_dataframe,
        new_tta_metrics_dataframe=new_tta_metrics_dataframe,
    )
    comparison_dataframe.to_csv(OUTPUT_COMPARISON_CSV, index=False)

    # ---------------------------------------------
    # Console summary
    # ---------------------------------------------
    print_macro_summary(baseline_metrics_dataframe, "Baseline")
    print_macro_summary(old_tta_metrics_dataframe, "Old BN-only TTA")
    print_macro_summary(new_tta_metrics_dataframe, "New Head + BN TTA")

    if old_tta_log is not None and "updated" in old_tta_log.columns:
        print()
        print("Old BN-only TTA updated batches:", int(old_tta_log["updated"].sum()), "/", len(old_tta_log))

    if "updated" in new_tta_log.columns:
        print("New head + BN TTA updated batches:", int(new_tta_log["updated"].sum()), "/", len(new_tta_log))

    if old_tta_log is not None and "accepted_fraction" in old_tta_log.columns:
        print("Old BN-only mean accepted fraction:", float(old_tta_log["accepted_fraction"].mean()))

    if "accepted_fraction" in new_tta_log.columns:
        print("New head + BN mean accepted fraction:", float(new_tta_log["accepted_fraction"].mean()))

    if old_tta_log is not None and "trainable_parameter_count" in old_tta_log.columns:
        print("Old BN-only trainable params per batch:",
              int(old_tta_log["trainable_parameter_count"].iloc[0]))

    if "trainable_parameter_count" in new_tta_log.columns:
        print("New head + BN trainable params per batch:",
              int(new_tta_log["trainable_parameter_count"].iloc[0]))

    print()
    print("Saved:")
    print(" -", OUTPUT_BASELINE_PREDICTIONS_CSV)
    print(" -", OUTPUT_OLD_TTA_PREDICTIONS_CSV)
    print(" -", OUTPUT_NEW_TTA_PREDICTIONS_CSV)
    print(" -", OUTPUT_OLD_TTA_LOG_CSV)
    print(" -", OUTPUT_NEW_TTA_LOG_CSV)
    print(" -", OUTPUT_BASELINE_METRICS_CSV)
    print(" -", OUTPUT_OLD_TTA_METRICS_CSV)
    print(" -", OUTPUT_NEW_TTA_METRICS_CSV)
    print(" -", OUTPUT_COMPARISON_CSV)
    print(" -", OUTPUT_THRESHOLD_JSON)
    print()
    print("Done.")


if __name__ == "__main__":
    main()