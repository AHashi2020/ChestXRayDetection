import os
import time
import pandas as pd
from PIL import Image
import certifi

# Uncomment this if you need the certifi SSL fix
# os.environ["SSL_CERT_FILE"] = certifi.where()

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.metrics import roc_auc_score, accuracy_score


# --------------------------------------------------
# Config
# --------------------------------------------------
train_csv = "train.csv"
val_csv = "val.csv"

continue_from_best_model = True
continued_total_finetune_epochs = 6
continued_finetune_lr = 5e-5

batch_size = 16
head_epochs = 4
finetune_epochs = 8

head_lr = 1e-3
finetune_lr = 5e-5

train_print_every = 100
eval_print_every = 100

num_workers = 0
resume_from_checkpoint = True
save_epoch_models = True

save_dir = "training_outputs"
latest_checkpoint_path = os.path.join(save_dir, "checkpoint_latest.pth")
best_model_path = os.path.join(save_dir, "best_model.pth")
final_model_path = os.path.join(save_dir, "final_model.pth")
history_csv_path = os.path.join(save_dir, "history.csv")
epoch_model_dir = os.path.join(save_dir, "epoch_models")


# --------------------------------------------------
# Dataset
# --------------------------------------------------
class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        print("Loading CSV:", csv_path, flush=True)
        self.dataframe = pd.read_csv(csv_path)
        self.transform = transform
        print("Loaded", len(self.dataframe), "rows from", csv_path, flush=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image = Image.open(row["image_path"]).convert("RGB")
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


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


def build_model():
    print("Building DenseNet-121...", flush=True)
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)

    print("Model built. Final classifier replaced.", flush=True)
    return model


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

def create_optimizer_for_phase(model, phase_name):
    if phase_name == "head":
        freeze_backbone(model)
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=head_lr)
        print("Head optimizer ready.", flush=True)
        return optimizer

    unfreeze_backbone(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=continued_finetune_lr)
    print("Finetune optimizer ready.", flush=True)
    return optimizer

# --------------------------------------------------
# Saving / loading
# --------------------------------------------------
def save_history_csv(history_rows):
    if len(history_rows) == 0:
        return

    history_dataframe = pd.DataFrame(history_rows)
    history_dataframe.to_csv(history_csv_path, index=False)
    print("Saved history to", history_csv_path, flush=True)


def save_latest_checkpoint(model, optimizer, phase_name, next_epoch, best_val_auroc, history_rows):
    checkpoint = {
        "phase_name": phase_name,
        "next_epoch": next_epoch,
        "best_val_auroc": best_val_auroc,
        "model_state_dict": model.state_dict(),
        "history_rows": history_rows,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    else:
        checkpoint["optimizer_state_dict"] = None

    torch.save(checkpoint, latest_checkpoint_path)
    print("Saved latest checkpoint to", latest_checkpoint_path, flush=True)


def save_best_model(model, phase_name, epoch_number, val_auroc):
    best_payload = {
        "phase_name": phase_name,
        "epoch_number": epoch_number,
        "val_auroc": val_auroc,
        "model_state_dict": model.state_dict(),
    }

    torch.save(best_payload, best_model_path)
    print("Saved best model to", best_model_path, flush=True)


def save_final_model(model, best_val_auroc, history_rows):
    final_payload = {
        "best_val_auroc": best_val_auroc,
        "history_rows": history_rows,
        "model_state_dict": model.state_dict(),
    }

    torch.save(final_payload, final_model_path)
    print("Saved final model to", final_model_path, flush=True)


def save_epoch_model(model, phase_name, epoch_number, val_auroc):
    if not save_epoch_models:
        return

    safe_auroc = str(round(val_auroc, 4)).replace(".", "_")
    filename = phase_name + "_epoch_" + str(epoch_number) + "_auroc_" + safe_auroc + ".pth"
    filepath = os.path.join(epoch_model_dir, filename)

    payload = {
        "phase_name": phase_name,
        "epoch_number": epoch_number,
        "val_auroc": val_auroc,
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
    print("Loading best model for continued training...", flush=True)
    payload = torch.load(best_model_path, map_location=device)

    model.load_state_dict(payload["model_state_dict"])
    print("Loaded weights from best model.", flush=True)

    history_rows = load_history_rows_from_csv()

    return {
        "phase_name": "finetune",
        "next_epoch": int(payload.get("epoch_number", 0)),
        "best_val_auroc": float(payload.get("val_auroc", -1.0)),
        "history_rows": history_rows,
        "optimizer_state_dict": None,
    }


def load_latest_checkpoint(model, device):
    if not resume_from_checkpoint:
        print("Resume disabled. Starting fresh.", flush=True)
        return {
            "phase_name": "head",
            "next_epoch": 0,
            "best_val_auroc": -1.0,
            "history_rows": [],
            "optimizer_state_dict": None,
        }

    if not os.path.exists(latest_checkpoint_path):
        print("No checkpoint found. Starting fresh.", flush=True)
        return {
            "phase_name": "head",
            "next_epoch": 0,
            "best_val_auroc": -1.0,
            "history_rows": [],
            "optimizer_state_dict": None,
        }

    print("Loading checkpoint from", latest_checkpoint_path, flush=True)
    checkpoint = torch.load(latest_checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded model weights from checkpoint.", flush=True)

    phase_name = checkpoint.get("phase_name", "head")
    next_epoch = checkpoint.get("next_epoch", 0)
    best_val_auroc = checkpoint.get("best_val_auroc", -1.0)
    history_rows = checkpoint.get("history_rows", [])
    optimizer_state_dict = checkpoint.get("optimizer_state_dict", None)

    print("Checkpoint phase:", phase_name, flush=True)
    print("Checkpoint next epoch:", next_epoch, flush=True)
    print("Checkpoint best AUROC:", best_val_auroc, flush=True)

    return {
        "phase_name": phase_name,
        "next_epoch": next_epoch,
        "best_val_auroc": best_val_auroc,
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

        logits = model(images).squeeze(1)
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
def evaluate(model, dataloader, device, phase_name):
    print("Starting evaluation for", phase_name, "...", flush=True)
    model.eval()

    all_labels = []
    all_probs = []

    for batch_index, (images, labels) in enumerate(dataloader):
        images = images.to(device)

        logits = model(images).squeeze(1)
        probabilities = torch.sigmoid(logits).cpu()
        labels = labels.cpu()

        index = 0
        while index < len(labels):
            all_labels.append(float(labels[index].item()))
            all_probs.append(float(probabilities[index].item()))
            index += 1

        if (batch_index + 1) % eval_print_every == 0:
            print(
                phase_name,
                "eval batch",
                batch_index + 1,
                "/",
                len(dataloader),
                flush=True,
            )

    predictions = []
    index = 0
    while index < len(all_probs):
        if all_probs[index] >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
        index += 1

    accuracy = accuracy_score(all_labels, predictions)
    auroc = roc_auc_score(all_labels, all_probs)

    print(
        "Finished evaluation for",
        phase_name,
        "- accuracy:",
        round(accuracy, 4),
        "- auroc:",
        round(auroc, 4),
        flush=True,
    )

    return accuracy, auroc


def run_phase(
    model,
    train_loader,
    val_loader,
    loss_fn,
    device,
    phase_name,
    start_epoch,
    total_epochs,
    best_val_auroc,
    history_rows,
    optimizer_state_dict,
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

        val_accuracy, val_auroc = evaluate(
            model,
            val_loader,
            device,
            phase_name + " epoch " + str(epoch_number),
        )

        print(phase_name, "epoch:", epoch_number, flush=True)
        print("train loss:", train_loss, flush=True)
        print("val accuracy:", val_accuracy, flush=True)
        print("val auroc:", val_auroc, flush=True)
        print("", flush=True)

        history_rows.append(
            {
                "phase": phase_name,
                "epoch": epoch_number,
                "train_loss": train_loss,
                "val_accuracy": val_accuracy,
                "val_auroc": val_auroc,
            }
        )

        save_history_csv(history_rows)
        save_latest_checkpoint(
            model=model,
            optimizer=optimizer,
            phase_name=phase_name,
            next_epoch=epoch_number,
            best_val_auroc=best_val_auroc,
            history_rows=history_rows,
        )

        save_epoch_model(
            model=model,
            phase_name=phase_name,
            epoch_number=epoch_number,
            val_auroc=val_auroc,
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            save_best_model(
                model=model,
                phase_name=phase_name,
                epoch_number=epoch_number,
                val_auroc=val_auroc,
            )

            save_latest_checkpoint(
                model=model,
                optimizer=optimizer,
                phase_name=phase_name,
                next_epoch=epoch_number,
                best_val_auroc=best_val_auroc,
                history_rows=history_rows,
            )

        epoch += 1

    return best_val_auroc, history_rows


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
    train_dataset = ChestXrayDataset(train_csv, transform=train_transform)
    val_dataset = ChestXrayDataset(val_csv, transform=val_transform)

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

    model = build_model().to(device)
    print("Model moved to device.", flush=True)
    if continue_from_best_model:
        checkpoint_info = load_best_model_for_more_training(model, device)
    else:
        checkpoint_info = load_latest_checkpoint(model, device)

    current_phase = checkpoint_info["phase_name"]
    next_epoch = checkpoint_info["next_epoch"]
    best_val_auroc = checkpoint_info["best_val_auroc"]
    history_rows = checkpoint_info["history_rows"]
    optimizer_state_dict = checkpoint_info["optimizer_state_dict"]

    loss_fn = nn.BCEWithLogitsLoss()
    print("Loss function ready.", flush=True)

    if current_phase == "done":
        print("Checkpoint says training is already complete.", flush=True)
        print("Best AUROC from checkpoint:", best_val_auroc, flush=True)
        return

    if current_phase == "head":
        if next_epoch < head_epochs:
            best_val_auroc, history_rows = run_phase(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                phase_name="finetune",
                start_epoch=next_epoch,
                total_epochs=head_epochs,
                best_val_auroc=best_val_auroc,
                history_rows=history_rows,
                optimizer_state_dict=optimizer_state_dict,
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

        if next_epoch < finetune_epochs:
            best_val_auroc, history_rows = run_phase(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                phase_name="finetune",
                start_epoch=next_epoch,
                total_epochs=finetune_epochs,
                best_val_auroc=best_val_auroc,
                history_rows=history_rows,
                optimizer_state_dict=optimizer_state_dict,
            )
        else:
            print("Finetune phase already complete.", flush=True)

    save_final_model(
        model=model,
        best_val_auroc=best_val_auroc,
        history_rows=history_rows,
    )

    save_latest_checkpoint(
        model=model,
        optimizer=None,
        phase_name="done",
        next_epoch=0,
        best_val_auroc=best_val_auroc,
        history_rows=history_rows,
    )

    save_history_csv(history_rows)

    print("Training complete.", flush=True)
    print("Best validation AUROC:", best_val_auroc, flush=True)
    print("Latest checkpoint:", latest_checkpoint_path, flush=True)
    print("Best model:", best_model_path, flush=True)
    print("Final model:", final_model_path, flush=True)


if __name__ == "__main__":
    main()