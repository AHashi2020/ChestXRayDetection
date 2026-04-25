import copy
import os

import numpy as np
import pydicom
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms


EPSILON = 1e-6
DEFAULT_IMAGE_SIZE = 224


def get_eval_transform(image_size: int = DEFAULT_IMAGE_SIZE):
    """
    Standard ImageNet normalization for DenseNet-style evaluation.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transform


def build_densenet121(num_outputs: int) -> nn.Module:
    """
    Build a DenseNet-121 with a custom classifier size.
    """
    try:
        model = models.densenet121(weights=None)
    except TypeError:
        model = models.densenet121(pretrained=False)

    input_features = model.classifier.in_features
    model.classifier = nn.Linear(input_features, num_outputs)
    return model


def extract_state_dict(checkpoint_object):
    """
    Support common checkpoint formats:
    - raw state_dict
    - {'model_state_dict': ...}
    - {'state_dict': ...}
    """
    if isinstance(checkpoint_object, dict):
        if "model_state_dict" in checkpoint_object:
            return checkpoint_object["model_state_dict"]

        if "state_dict" in checkpoint_object:
            return checkpoint_object["state_dict"]

    return checkpoint_object


def strip_module_prefix(state_dict: dict) -> dict:
    """
    Remove 'module.' if model was saved under DataParallel.
    """
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]

        cleaned_state_dict[new_key] = value

    return cleaned_state_dict


def load_checkpoint_into_model(model: nn.Module, checkpoint_path: str, device: torch.device):
    """
    Load a checkpoint into a model.
    """
    checkpoint_object = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint_object)
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    return model


def clone_model(model: nn.Module) -> nn.Module:
    """
    Deep copy of a model.
    """
    return copy.deepcopy(model)


def find_first_existing_path(candidate_paths):
    """
    Return the first existing path from a list of candidates.
    """
    for candidate_path in candidate_paths:
        if candidate_path is None:
            continue

        if os.path.exists(candidate_path):
            return candidate_path

    return None


def build_file_name_path_map(root_dir: str, allowed_extensions):
    """
    Map exact file name -> full path.
    Example: '00000001_000.png' -> '/.../NIH/images_001/00000001_000.png'
    """
    mapping = {}

    for current_root, _, file_names in os.walk(root_dir):
        for file_name in file_names:
            lowered_name = file_name.lower()

            matched_extension = False

            for extension in allowed_extensions:
                if lowered_name.endswith(extension):
                    matched_extension = True
                    break

            if not matched_extension:
                continue

            if file_name not in mapping:
                full_path = os.path.join(current_root, file_name)
                mapping[file_name] = full_path

    return mapping


def build_file_stem_path_map(root_dir: str, allowed_extensions):
    """
    Map file stem -> full path.
    Example: 'img_12345' -> '/.../vindr-cxr_test/img_12345.dicom'
    """
    mapping = {}

    for current_root, _, file_names in os.walk(root_dir):
        for file_name in file_names:
            lowered_name = file_name.lower()

            matched_extension = False

            for extension in allowed_extensions:
                if lowered_name.endswith(extension):
                    matched_extension = True
                    break

            if not matched_extension:
                continue

            stem_name, _ = os.path.splitext(file_name)

            if stem_name not in mapping:
                full_path = os.path.join(current_root, file_name)
                mapping[stem_name] = full_path

    return mapping


def dicom_to_rgb_pil(dicom_path: str) -> Image.Image:
    """
    Read DICOM, apply common rescale/inversion handling, normalize to uint8,
    then convert to RGB PIL image.
    """
    dicom_object = pydicom.dcmread(dicom_path)
    pixel_array = dicom_object.pixel_array.astype(np.float32)

    if hasattr(dicom_object, "RescaleSlope"):
        pixel_array = pixel_array * float(dicom_object.RescaleSlope)

    if hasattr(dicom_object, "RescaleIntercept"):
        pixel_array = pixel_array + float(dicom_object.RescaleIntercept)

    photometric_interpretation = ""
    if hasattr(dicom_object, "PhotometricInterpretation"):
        photometric_interpretation = str(dicom_object.PhotometricInterpretation)

    if photometric_interpretation == "MONOCHROME1":
        pixel_array = np.max(pixel_array) - pixel_array

    minimum_value = float(np.min(pixel_array))
    maximum_value = float(np.max(pixel_array))

    if maximum_value > minimum_value:
        pixel_array = (pixel_array - minimum_value) / (maximum_value - minimum_value)
    else:
        pixel_array = np.zeros_like(pixel_array, dtype=np.float32)

    pixel_array = pixel_array * 255.0
    pixel_array = np.clip(pixel_array, 0.0, 255.0)
    pixel_array = pixel_array.astype(np.uint8)

    pil_image = Image.fromarray(pixel_array)
    pil_image = pil_image.convert("RGB")
    return pil_image


def binary_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Binary entropy per sample.
    logits shape can be [B] or [B, 1].
    Returns shape [B].
    """
    probabilities = torch.sigmoid(logits).view(-1)

    entropy = -(
        probabilities * torch.log(probabilities + EPSILON)
        + (1.0 - probabilities) * torch.log(1.0 - probabilities + EPSILON)
    )

    return entropy


def multilabel_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Multi-label entropy per sample.
    logits shape [B, C].
    Returns shape [B].
    """
    probabilities = torch.sigmoid(logits)

    per_class_entropy = -(
        probabilities * torch.log(probabilities + EPSILON)
        + (1.0 - probabilities) * torch.log(1.0 - probabilities + EPSILON)
    )

    sample_entropy = per_class_entropy.mean(dim=1)
    return sample_entropy


def estimate_entropy_threshold(
    model,
    data_loader,
    device,
    is_multilabel: bool,
    quantile: float = 0.50,
) -> float:
    """
    Estimate entropy threshold using NIH validation images only.
    """
    model.eval()
    all_entropies = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            logits = model(images)

            if is_multilabel:
                batch_entropy = multilabel_entropy_from_logits(logits)
            else:
                batch_entropy = binary_entropy_from_logits(logits)

            all_entropies.append(batch_entropy.cpu())

    all_entropies = torch.cat(all_entropies, dim=0)
    threshold = torch.quantile(all_entropies, quantile).item()
    return float(threshold)


def configure_model_for_tta(model: nn.Module):
    """
    TTA setup:
    - enable train mode
    - freeze everything
    - unfreeze BatchNorm affine params only
    - force BN layers to use batch stats
    """
    model.train()

    for parameter in model.parameters():
        parameter.requires_grad = False

    batch_norm_parameters = []

    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.train()

            if module.weight is not None:
                module.weight.requires_grad = True
                batch_norm_parameters.append(module.weight)

            if module.bias is not None:
                module.bias.requires_grad = True
                batch_norm_parameters.append(module.bias)

            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

    if len(batch_norm_parameters) == 0:
        raise RuntimeError("No BatchNorm parameters found for TTA.")

    return model, batch_norm_parameters


@torch.enable_grad()
def tta_step_binary(
    model: nn.Module,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    entropy_threshold: float,
    min_accepted_samples: int = 4,
):
    """
    One binary TTA update step.
    Only low-entropy samples are allowed to update the model.
    """
    logits = model(images)
    sample_entropy = binary_entropy_from_logits(logits)

    accepted_mask = sample_entropy <= entropy_threshold
    accepted_count = int(accepted_mask.sum().item())
    batch_size = int(sample_entropy.shape[0])

    result = {
        "accepted_count": accepted_count,
        "batch_size": batch_size,
        "accepted_fraction": accepted_count / max(batch_size, 1),
        "mean_entropy": float(sample_entropy.mean().item()),
        "loss": None,
        "updated": False,
    }

    if accepted_count < min_accepted_samples:
        return result

    loss = sample_entropy[accepted_mask].mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    result["loss"] = float(loss.item())
    result["updated"] = True
    return result


@torch.enable_grad()
def tta_step_multilabel(
    model: nn.Module,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    entropy_threshold: float,
    min_accepted_samples: int = 4,
):
    """
    One multi-label TTA update step.
    Only low-entropy samples are allowed to update the model.
    """
    logits = model(images)
    sample_entropy = multilabel_entropy_from_logits(logits)

    accepted_mask = sample_entropy <= entropy_threshold
    accepted_count = int(accepted_mask.sum().item())
    batch_size = int(sample_entropy.shape[0])

    result = {
        "accepted_count": accepted_count,
        "batch_size": batch_size,
        "accepted_fraction": accepted_count / max(batch_size, 1),
        "mean_entropy": float(sample_entropy.mean().item()),
        "loss": None,
        "updated": False,
    }

    if accepted_count < min_accepted_samples:
        return result

    loss = sample_entropy[accepted_mask].mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    result["loss"] = float(loss.item())
    result["updated"] = True
    return result