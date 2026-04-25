import os
import time
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.metrics import roc_auc_score


# --------------------------------------------------
# Config
# --------------------------------------------------
train_csv = "train_labels.csv"
val_csv = "val_labels.csv"

label_names = [
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

batch_size = 16

# First normal training run
head_epochs = 3
finetune_epochs = 5
head_lr = 1e-3
finetune_lr = 1e-4

# Later continued training run from best_model.pth
continue_from_best_model = True
continued_total_finetune_epochs = 10
continued_finetune_lr = 5e-5

train_print_every = 100
eval_print_every = 100
num_workers = 0

resume_from_checkpoint = True
save_epoch_models = True

save_dir = "training_outputs_labels"
latest_checkpoint_path = os.path.join(save_dir, "checkpoint_latest.pth")
best_model_path = os.path.join(save_dir, "best_model.pth")
final_model_path = os.path.join(save_dir, "final_model.pth")
history_csv_path = os.path.join(save_dir, "history.csv")
epoch_model_dir = os.path.join(save_dir, "epoch_models")


# --------------------------------------------------
# Dataset
# --------------------------------------------------
class ChestXrayLabelsDataset(Dataset):
    def __init__(self, csv_path, label_names, transform=None):
        print("Loading CSV:", csv_path, flush=True)
        self.dataframe = pd.read_csv(csv_path)
        self.label_names = label_names
        self.transform = transform
        print("Loaded", len(self.dataframe), "rows from", csv_path, flush=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image = Image.open(row["image_path"]).convert("RGB")

        label_values = []
        label_index = 0
        while label_index < len(self.label_names):
            current_label_name = self.label_names[label_index]
            current_value = float(row[current_label_name])
            label_values.append(current_value)
            label_index += 1

        label_tensor = torch.tensor(label_values, dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        return image, label_tensor


# --------------------------------------------------
# Setup helpers
# --------------------------------------------------
def ensure_save_dirs():
    os.makedirs(save_dir, exist_ok=True)

    if save_epoch_models:
        os.makedirs(epoch_model_dir, exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_labels):
    print("Building DenseNet-121...", flush=True)
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_labels)

    print("Model built. Final classifier replaced with", num_labels, "outputs.", flush=True)
    return model

import json
import numpy as np
from sklearn.metrics import f1_score


thresholds_json_path = os.path.join(save_dir, "best_thresholds.json")


def find_best_threshold_for_label(true_values, pred_probs):
    best_threshold = 0.5
    best_f1 = -1.0

    threshold = 0.05
    while threshold <= 0.95:
        pred_labels = []
        index = 0
        while index < len(pred_probs):
            if pred_probs[index] >= threshold:
                pred_labels.append(1)
            else:
                pred_labels.append(0)
            index += 1

        current_f1 = float(f1_score(true_values, pred_labels, zero_division=0))

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(round(threshold, 4))

        threshold += 0.01

    return best_threshold, best_f1


def save_best_thresholds(model, dataloader, device, label_names):
    print("Finding best per-label thresholds on NIH validation...", flush=True)
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            logits = model(images)
            probabilities = torch.sigmoid(logits).cpu()

            all_labels.append(labels.cpu())
            all_probs.append(probabilities)

    all_labels_tensor = torch.cat(all_labels, dim=0)
    all_probs_tensor = torch.cat(all_probs, dim=0)

    all_labels_numpy = all_labels_tensor.numpy()
    all_probs_numpy = all_probs_tensor.numpy()

    thresholds_dict = {}

    label_index = 0
    while label_index < len(label_names):
        current_label_name = label_names[label_index]
        current_true = all_labels_numpy[:, label_index]
        current_prob = all_probs_numpy[:, label_index]

        unique_values = set()
        row_index = 0
        while row_index < len(current_true):
            unique_values.add(float(current_true[row_index]))
            row_index += 1

        if len(unique_values) < 2:
            thresholds_dict[current_label_name] = 0.5
            print(current_label_name, "- only one class in val, using threshold 0.5", flush=True)
        else:
            best_threshold, best_f1 = find_best_threshold_for_label(current_true, current_prob)
            thresholds_dict[current_label_name] = best_threshold
            print(
                current_label_name,
                "- best threshold:",
                best_threshold,
                "- best val F1:",
                round(best_f1, 4),
                flush=True,
            )

        label_index += 1

    with open(thresholds_json_path, "w") as file:
        json.dump(thresholds_dict, file, indent=2)

    print("Saved thresholds to", thresholds_json_path, flush=True)


def freeze_backbone(model):
    print("Freezing backbone...", flush=True)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    print("Backbone frozen.", flush=True)


def unfreeze_backbone(model):
    print("Unfreezing backbone...", flush=True)
    for parameter in model.features.parameters():
        parameter.requires_grad = True
    print("Backbone unfrozen.", flush=True)


def get_finetune_lr():
    if continue_from_best_model:
        return continued_finetune_lr
    return finetune_lr


def get_total_finetune_epochs():
    if continue_from_best_model:
        return continued_total_finetune_epochs
    return finetune_epochs


def create_optimizer_for_phase(model, phase_name):
    if phase_name == "head":
        freeze_backbone(model)
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=head_lr)
        print("Head optimizer ready.", flush=True)
        return optimizer

    unfreeze_backbone(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=get_finetune_lr())
    print("Finetune optimizer ready.", flush=True)
    return optimizer


def compute_pos_weight(train_csv_path, label_names):
    print("Computing pos_weight from training CSV...", flush=True)
    dataframe = pd.read_csv(train_csv_path)

    pos_weights = []
    label_index = 0

    while label_index < len(label_names):
        label_name = label_names[label_index]

        positive_count = float(dataframe[label_name].sum())
        total_count = float(len(dataframe))
        negative_count = total_count - positive_count

        if positive_count <= 0:
            current_pos_weight = 1.0
        else:
            current_pos_weight = negative_count / positive_count

        pos_weights.append(current_pos_weight)

        print(
            label_name,
            "- positives:",
            int(positive_count),
            "- negatives:",
            int(negative_count),
            "- pos_weight:",
            round(current_pos_weight, 4),
            flush=True,
        )

        label_index += 1

    return torch.tensor(pos_weights, dtype=torch.float32)


# --------------------------------------------------
# Saving / loading
# --------------------------------------------------
def save_history_csv(history_rows):
    if len(history_rows) == 0:
        return

    history_dataframe = pd.DataFrame(history_rows)
    history_dataframe.to_csv(history_csv_path, index=False)
    print("Saved history to", history_csv_path, flush=True)


def save_latest_checkpoint(
    model,
    optimizer,
    phase_name,
    next_epoch,
    best_val_macro_auroc,
    history_rows,
):
    checkpoint = {
        "phase_name": phase_name,
        "next_epoch": next_epoch,
        "best_val_macro_auroc": best_val_macro_auroc,
        "model_state_dict": model.state_dict(),
        "history_rows": history_rows,
        "label_names": label_names,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    else:
        checkpoint["optimizer_state_dict"] = None

    torch.save(checkpoint, latest_checkpoint_path)
    print("Saved latest checkpoint to", latest_checkpoint_path, flush=True)


def save_best_model(model, phase_name, epoch_number, val_macro_auroc, per_label_aurocs):
    payload = {
        "phase_name": phase_name,
        "epoch_number": epoch_number,
        "val_macro_auroc": val_macro_auroc,
        "per_label_aurocs": per_label_aurocs,
        "label_names": label_names,
        "model_state_dict": model.state_dict(),
    }

    torch.save(payload, best_model_path)
    print("Saved best model to", best_model_path, flush=True)


def save_final_model(model, best_val_macro_auroc, history_rows):
    payload = {
        "best_val_macro_auroc": best_val_macro_auroc,
        "history_rows": history_rows,
        "label_names": label_names,
        "model_state_dict": model.state_dict(),
    }

    torch.save(payload, final_model_path)
    print("Saved final model to", final_model_path, flush=True)


def save_epoch_model(model, phase_name, epoch_number, val_macro_auroc):
    if not save_epoch_models:
        return

    safe_auroc = str(round(val_macro_auroc, 4)).replace(".", "_")
    filename = phase_name + "_epoch_" + str(epoch_number) + "_macro_auroc_" + safe_auroc + ".pth"
    filepath = os.path.join(epoch_model_dir, filename)

    payload = {
        "phase_name": phase_name,
        "epoch_number": epoch_number,
        "val_macro_auroc": val_macro_auroc,
        "label_names": label_names,
        "model_state_dict": model.state_dict(),
    }

    torch.save(payload, filepath)
    print("Saved epoch model to", filepath, flush=True)


def load_history_rows_from_csv():
    if os.path.exists(history_csv_path):
        history_dataframe = pd.read_csv(history_csv_path)
        return history_dataframe.to_dict("records")
    return []


def load_best_model_for_more_training(model, device):
    if not os.path.exists(best_model_path):
        print("No best_model.pth found. Falling back to latest checkpoint / fresh start.", flush=True)
        return None

    print("Loading best model for continued training...", flush=True)
    payload = torch.load(best_model_path, map_location=device)

    model.load_state_dict(payload["model_state_dict"])
    print("Loaded weights from best model.", flush=True)

    history_rows = load_history_rows_from_csv()

    return {
        "phase_name": "finetune",
        "next_epoch": int(payload.get("epoch_number", 0)),
        "best_val_macro_auroc": float(payload.get("val_macro_auroc", -1.0)),
        "history_rows": history_rows,
        "optimizer_state_dict": None,
    }


def load_latest_checkpoint(model, device):
    if not resume_from_checkpoint:
        print("Resume disabled. Starting fresh.", flush=True)
        return {
            "phase_name": "head",
            "next_epoch": 0,
            "best_val_macro_auroc": -1.0,
            "history_rows": [],
            "optimizer_state_dict": None,
        }

    if not os.path.exists(latest_checkpoint_path):
        print("No checkpoint found. Starting fresh.", flush=True)
        return {
            "phase_name": "head",
            "next_epoch": 0,
            "best_val_macro_auroc": -1.0,
            "history_rows": [],
            "optimizer_state_dict": None,
        }

    print("Loading checkpoint from", latest_checkpoint_path, flush=True)
    checkpoint = torch.load(latest_checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded model weights from checkpoint.", flush=True)

    phase_name = checkpoint.get("phase_name", "head")
    next_epoch = checkpoint.get("next_epoch", 0)
    best_val_macro_auroc = checkpoint.get("best_val_macro_auroc", -1.0)
    history_rows = checkpoint.get("history_rows", [])
    optimizer_state_dict = checkpoint.get("optimizer_state_dict", None)

    print("Checkpoint phase:", phase_name, flush=True)
    print("Checkpoint next epoch:", next_epoch, flush=True)
    print("Checkpoint best macro AUROC:", best_val_macro_auroc, flush=True)

    return {
        "phase_name": phase_name,
        "next_epoch": next_epoch,
        "best_val_macro_auroc": best_val_macro_auroc,
        "history_rows": history_rows,
        "optimizer_state_dict": optimizer_state_dict,
    }


# --------------------------------------------------
# Train / eval
# --------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, loss_fn, device, phase_name, epoch_number):
    print("Starting", phase_name, "epoch", epoch_number, flush=True)
    model.train()

    total_loss = 0.0
    total_examples = 0
    batch_start_time = time.time()

    for batch_index, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size_local = images.size(0)
        total_loss += loss.item() * batch_size_local
        total_examples += batch_size_local

        if (batch_index + 1) % train_print_every == 0:
            elapsed = time.time() - batch_start_time
            average_loss = total_loss / total_examples

            print(
                phase_name,
                "epoch",
                epoch_number,
                "- batch",
                batch_index + 1,
                "/",
                len(dataloader),
                "- avg loss:",
                round(average_loss, 4),
                "- elapsed:",
                round(elapsed, 1),
                "sec",
                flush=True,
            )

    average_loss = total_loss / total_examples

    print(
        "Finished",
        phase_name,
        "epoch",
        epoch_number,
        "- final avg loss:",
        round(average_loss, 4),
        flush=True,
    )

    return average_loss


@torch.no_grad()
def evaluate(model, dataloader, device, label_names, phase_name):
    print("Starting evaluation for", phase_name, "...", flush=True)
    model.eval()

    all_labels = []
    all_probs = []

    for batch_index, (images, labels) in enumerate(dataloader):
        images = images.to(device)

        logits = model(images)
        probabilities = torch.sigmoid(logits).cpu()
        labels = labels.cpu()

        all_labels.append(labels)
        all_probs.append(probabilities)

        if (batch_index + 1) % eval_print_every == 0:
            print(
                phase_name,
                "eval batch",
                batch_index + 1,
                "/",
                len(dataloader),
                flush=True,
            )

    all_labels_tensor = torch.cat(all_labels, dim=0)
    all_probs_tensor = torch.cat(all_probs, dim=0)

    all_labels_numpy = all_labels_tensor.numpy()
    all_probs_numpy = all_probs_tensor.numpy()

    per_label_aurocs = {}
    valid_aurocs = []

    label_index = 0
    while label_index < len(label_names):
        current_label_name = label_names[label_index]
        current_true = all_labels_numpy[:, label_index]
        current_prob = all_probs_numpy[:, label_index]

        unique_values = set()
        row_index = 0
        while row_index < len(current_true):
            unique_values.add(float(current_true[row_index]))
            row_index += 1

        if len(unique_values) < 2:
            current_auroc = None
            print(current_label_name, "- AUROC skipped because only one class appears in validation.", flush=True)
        else:
            current_auroc = float(roc_auc_score(current_true, current_prob))
            valid_aurocs.append(current_auroc)
            print(current_label_name, "- AUROC:", round(current_auroc, 4), flush=True)

        per_label_aurocs[current_label_name] = current_auroc
        label_index += 1

    if len(valid_aurocs) == 0:
        macro_auroc = 0.0
    else:
        macro_auroc = float(sum(valid_aurocs) / len(valid_aurocs))

    predictions = (all_probs_tensor >= 0.5).float()
    mean_label_accuracy = float((predictions == all_labels_tensor).float().mean().item())

    print(
        "Finished evaluation for",
        phase_name,
        "- macro AUROC:",
        round(macro_auroc, 4),
        "- mean label accuracy:",
        round(mean_label_accuracy, 4),
        flush=True,
    )

    return macro_auroc, mean_label_accuracy, per_label_aurocs


def run_phase(
    model,
    train_loader,
    val_loader,
    loss_fn,
    device,
    phase_name,
    start_epoch,
    total_epochs,
    best_val_macro_auroc,
    history_rows,
    optimizer_state_dict,
    label_names,
):
    optimizer = create_optimizer_for_phase(model, phase_name)

    if optimizer_state_dict is not None and start_epoch > 0:
        optimizer.load_state_dict(optimizer_state_dict)
        print("Loaded optimizer state for", phase_name, "phase.", flush=True)

    epoch = start_epoch

    while epoch < total_epochs:
        epoch_number = epoch + 1

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            phase_name,
            epoch_number,
        )

        val_macro_auroc, mean_label_accuracy, per_label_aurocs = evaluate(
            model,
            val_loader,
            device,
            label_names,
            phase_name + " epoch " + str(epoch_number),
        )

        print(phase_name, "epoch:", epoch_number, flush=True)
        print("train loss:", train_loss, flush=True)
        print("val macro auroc:", val_macro_auroc, flush=True)
        print("val mean label accuracy:", mean_label_accuracy, flush=True)
        print("", flush=True)

        history_row = {
            "phase": phase_name,
            "epoch": epoch_number,
            "train_loss": train_loss,
            "val_macro_auroc": val_macro_auroc,
            "val_mean_label_accuracy": mean_label_accuracy,
        }

        label_index = 0
        while label_index < len(label_names):
            current_label_name = label_names[label_index]
            history_row["auroc_" + current_label_name] = per_label_aurocs[current_label_name]
            label_index += 1

        history_rows.append(history_row)

        save_history_csv(history_rows)
        save_latest_checkpoint(
            model=model,
            optimizer=optimizer,
            phase_name=phase_name,
            next_epoch=epoch_number,
            best_val_macro_auroc=best_val_macro_auroc,
            history_rows=history_rows,
        )

        save_epoch_model(
            model=model,
            phase_name=phase_name,
            epoch_number=epoch_number,
            val_macro_auroc=val_macro_auroc,
        )

        if val_macro_auroc > best_val_macro_auroc:
            best_val_macro_auroc = val_macro_auroc

            save_best_model(
                model=model,
                phase_name=phase_name,
                epoch_number=epoch_number,
                val_macro_auroc=val_macro_auroc,
                per_label_aurocs=per_label_aurocs,
            )

            save_latest_checkpoint(
                model=model,
                optimizer=optimizer,
                phase_name=phase_name,
                next_epoch=epoch_number,
                best_val_macro_auroc=best_val_macro_auroc,
                history_rows=history_rows,
            )

        epoch += 1

    return best_val_macro_auroc, history_rows


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    ensure_save_dirs()

    device = get_device()
    print("Using device:", device, flush=True)

    print("Creating transforms...", flush=True)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    print("Transforms ready.", flush=True)

    print("Building datasets...", flush=True)
    train_dataset = ChestXrayLabelsDataset(train_csv, label_names, transform=train_transform)
    val_dataset = ChestXrayLabelsDataset(val_csv, label_names, transform=val_transform)

    print("Train dataset size:", len(train_dataset), flush=True)
    print("Val dataset size:", len(val_dataset), flush=True)

    print("Building dataloaders...", flush=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print("Dataloaders ready.", flush=True)
    print("Train batches per epoch:", len(train_loader), flush=True)
    print("Val batches:", len(val_loader), flush=True)

    model = build_model(len(label_names)).to(device)
    print("Model moved to device.", flush=True)

    checkpoint_info = None

    if continue_from_best_model:
        checkpoint_info = load_best_model_for_more_training(model, device)

    if checkpoint_info is None:
        checkpoint_info = load_latest_checkpoint(model, device)

    current_phase = checkpoint_info["phase_name"]
    next_epoch = checkpoint_info["next_epoch"]
    best_val_macro_auroc = checkpoint_info["best_val_macro_auroc"]
    history_rows = checkpoint_info["history_rows"]
    optimizer_state_dict = checkpoint_info["optimizer_state_dict"]

    pos_weight = compute_pos_weight(train_csv, label_names).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("Loss function ready.", flush=True)

    if current_phase == "done":
        print("Checkpoint says training is already complete.", flush=True)
        print("Best macro AUROC from checkpoint:", best_val_macro_auroc, flush=True)
        return

    if current_phase == "head":
        if next_epoch < head_epochs:
            best_val_macro_auroc, history_rows = run_phase(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                phase_name="head",
                start_epoch=next_epoch,
                total_epochs=head_epochs,
                best_val_macro_auroc=best_val_macro_auroc,
                history_rows=history_rows,
                optimizer_state_dict=optimizer_state_dict,
                label_names=label_names,
            )
        else:
            print("Head phase already complete. Moving to finetune.", flush=True)

        current_phase = "finetune"
        next_epoch = 0
        optimizer_state_dict = None

    if current_phase == "finetune":
        if checkpoint_info["phase_name"] == "finetune":
            next_epoch = checkpoint_info["next_epoch"]
            optimizer_state_dict = checkpoint_info["optimizer_state_dict"]

        total_finetune_epochs = get_total_finetune_epochs()

        if next_epoch < total_finetune_epochs:
            best_val_macro_auroc, history_rows = run_phase(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                phase_name="finetune",
                start_epoch=next_epoch,
                total_epochs=total_finetune_epochs,
                best_val_macro_auroc=best_val_macro_auroc,
                history_rows=history_rows,
                optimizer_state_dict=optimizer_state_dict,
                label_names=label_names,
            )
        else:
            print("Finetune phase already complete.", flush=True)

    save_final_model(
        model=model,
        best_val_macro_auroc=best_val_macro_auroc,
        history_rows=history_rows,
    )

    save_latest_checkpoint(
        model=model,
        optimizer=None,
        phase_name="done",
        next_epoch=0,
        best_val_macro_auroc=best_val_macro_auroc,
        history_rows=history_rows,
    )

    save_history_csv(history_rows)
    save_best_thresholds(model, val_loader, device, label_names)

    print("Training complete.", flush=True)
    print("Best validation macro AUROC:", best_val_macro_auroc, flush=True)
    print("Latest checkpoint:", latest_checkpoint_path, flush=True)
    print("Best model:", best_model_path, flush=True)
    print("Final model:", final_model_path, flush=True)


if __name__ == "__main__":
    main()