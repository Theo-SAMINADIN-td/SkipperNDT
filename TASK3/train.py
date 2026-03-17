import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from torchvision import models


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class PipelineDataset(Dataset):
    """Dataset pour charger les images .npz avec labels automatiques"""

    def __init__(self, file_paths, labels, target_size=(224, 224)):
        self.file_paths = file_paths
        self.labels = labels
        self.target_size = target_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx], allow_pickle=True)
        image = data["data"]

        if image.dtype == np.float16:
            image = image.astype(np.float32)

        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = self._resize_image(image)
        image = self._normalize_channels(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label

    def _resize_image(self, image):
        h, w, c = image.shape
        target_h, target_w = self.target_size
        resized = zoom(image, (target_h / h, target_w / w, 1), order=1)
        return resized

    def _normalize_channels(self, image):
        normalized = np.zeros_like(image)
        for c in range(image.shape[2]):
            channel = image[:, :, c]
            mean = np.mean(channel)
            std = np.std(channel)
            normalized[:, :, c] = (channel - mean) / std if std > 0 else channel - mean
        return normalized


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

class CurrentIntensityClassifier(nn.Module):
    """
    ResNet18 pré-entraîné pour classification binaire sur 4 canaux.
    - Dropout réduit (0.1) pour stabiliser l'entraînement précoce
    - Pas de sigmoid final (BCEWithLogitsLoss)
    """

    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Adapter pour 4 canaux d'entrée
        self.resnet.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Tête de classification avec dropout léger
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, 1)
        )

    def forward(self, x):
        return self.resnet(x)

    def freeze_backbone(self):
        """Gèle tous les paramètres sauf la tête FC (phase 1)"""
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        print("  Backbone frozen — training head only")

    def unfreeze_all(self):
        """Dégèle tous les paramètres (phase 2)"""
        for param in self.resnet.parameters():
            param.requires_grad = True
        print("  All layers unfrozen — full fine-tuning")


# ──────────────────────────────────────────────
# Training & Validation
# ──────────────────────────────────────────────

def apply_threshold(outputs, threshold=0.5):
    """Applique sigmoid puis seuil pour obtenir les prédictions binaires"""
    return (torch.sigmoid(outputs) > threshold).float()


def train_epoch(model, dataloader, criterion, optimizer, device, threshold=0.5):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = apply_threshold(outputs.detach(), threshold)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / len(dataloader), accuracy_score(all_labels, all_preds)


def validate_epoch(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = apply_threshold(outputs, threshold)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return epoch_loss, accuracy, recall, f1, all_preds, all_labels


def find_best_threshold(model, dataloader, device, min_recall=0.85):
    """
    Cherche le seuil optimal sur le validation set.
    Maximise F1 sous contrainte recall >= min_recall.
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Threshold search", leave=False):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()

    best_threshold = 0.5
    best_f1 = 0.0
    print("\n  Threshold search results:")
    for threshold in np.arange(0.2, 0.71, 0.05):
        preds = (all_probs > threshold).astype(float)
        rec = recall_score(all_labels, preds)
        f1 = f1_score(all_labels, preds)
        acc = accuracy_score(all_labels, preds)
        marker = " ✓" if rec >= min_recall else ""
        print(f"    threshold={threshold:.2f} → Acc={acc:.4f} | Recall={rec:.4f} | F1={f1:.4f}{marker}")
        if f1 > best_f1 and rec >= min_recall:
            best_f1 = f1
            best_threshold = threshold

    print(f"\n  Best threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    return best_threshold


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_training_history(history, freeze_epochs=0, save_path="training_history.png"):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for ax, train_key, val_key, title, ylabel, target in [
        (axes[0, 0], "train_loss", "val_loss",   "Loss",     "Loss",     None),
        (axes[0, 1], "train_acc",  "val_acc",    "Accuracy", "Accuracy", 0.90),
    ]:
        ax.plot(history[train_key], label=f"Train {ylabel}")
        ax.plot(history[val_key],   label=f"Val {ylabel}")
        if target:
            ax.axhline(y=target, color="r", linestyle="--", label=f"Target {target:.0%}")
        if freeze_epochs > 0:
            ax.axvline(x=freeze_epochs - 1, color="orange", linestyle=":", label="Unfreeze")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    axes[1, 0].plot(history["val_recall"], label="Val Recall")
    axes[1, 0].axhline(y=0.85, color="r", linestyle="--", label="Target 85%")
    if freeze_epochs > 0:
        axes[1, 0].axvline(x=freeze_epochs - 1, color="orange", linestyle=":", label="Unfreeze")
    axes[1, 0].set_title("Recall (Validation)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history["val_f1"], label="Val F1-Score")
    if freeze_epochs > 0:
        axes[1, 1].axvline(x=freeze_epochs - 1, color="orange", linestyle=":", label="Unfreeze")
    axes[1, 1].set_title("F1-Score (Validation)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("F1-Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")
    plt.close()


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_data_with_labels(data_dir, label_column="label"):
    """
    Charge les fichiers .npz et extrait les labels à partir du CSV.

    label_column: "label" pour pipe presence (TASK1)
                  "intensity_class" pour current intensity (TASK3)
    """
    csv_path = os.path.join(data_dir, "pipe_detection_label.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Label CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=",")

    if label_column not in df.columns:
        available_cols = list(df.columns)
        print(f"  ⚠ Column '{label_column}' not found. Available: {available_cols}")
        if "label" in df.columns:
            print("  → Using 'label' column instead")
            label_column = "label"
        else:
            raise ValueError(f"No suitable label column found. Available: {available_cols}")

    file_paths, labels = [], []
    missing = 0

    for _, row in df.iterrows():
        fpath = os.path.join(data_dir, row["field_file"])
        if os.path.exists(fpath):
            file_paths.append(fpath)
            labels.append(float(row[label_column]))
        else:
            missing += 1

    print(f"  ✓ {len(file_paths)} samples loaded (column: '{label_column}') | {missing} files missing")
    return file_paths, labels


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":

    # ── Hyperparameters ──────────────────────
    DATA_DIR        = "Training_data_inspection_validation_float16"
    BATCH_SIZE      = 16
    NUM_EPOCHS      = 30
    FREEZE_EPOCHS   = 5       # Epochs to train head-only before full fine-tuning
    LR_HEAD         = 1e-3    # Higher LR during frozen phase (head only)
    LR_FULL         = 1e-4    # Lower LR during full fine-tuning
    TARGET_SIZE     = (224, 224)
    DROPOUT_RATE    = 0.1     # Reduced from 0.3 to stabilise early training
    LABEL_COLUMN    = "label" # Change to "intensity_class" when available
    # ─────────────────────────────────────────

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    print("\n1. Loading data...")
    file_paths, labels = load_data_with_labels(DATA_DIR, label_column=LABEL_COLUMN)
    print(f"  Total: {len(file_paths)} | "
          f"Insufficient: {sum(1 for l in labels if l == 0.0)} | "
          f"Sufficient: {sum(1 for l in labels if l == 1.0)}")

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    print(f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    # 2. Datasets & loaders
    print("\n2. Creating datasets...")
    train_dataset = PipelineDataset(train_files, train_labels, target_size=TARGET_SIZE)
    val_dataset   = PipelineDataset(val_files,   val_labels,   target_size=TARGET_SIZE)
    test_dataset  = PipelineDataset(test_files,  test_labels,  target_size=TARGET_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. Model & loss
    print("\n3. Creating model...")
    model = CurrentIntensityClassifier(dropout_rate=DROPOUT_RATE).to(device)

    num_insufficient = sum(1 for l in train_labels if l == 0.0)
    num_sufficient   = sum(1 for l in train_labels if l == 1.0)
    # Neutral pos_weight=1.0: threshold tuning handles class balance
    pos_weight = 1.0
    print(f"  pos_weight: {pos_weight} (neutral) | "
          f"Insufficient: {num_insufficient} | Sufficient: {num_sufficient}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight)).to(device)

    # 4. Two-phase training
    print("\n4. Training model...")
    print(f"  Phase 1: {FREEZE_EPOCHS} epochs — head only (lr={LR_HEAD})")
    print(f"  Phase 2: {NUM_EPOCHS - FREEZE_EPOCHS} epochs — full fine-tune (lr={LR_FULL})")

    model.freeze_backbone()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "val_recall", "val_f1"]}

    best_score      = 0.0   # Combined: 0.4*recall + 0.6*f1
    best_recall     = 0.0
    best_epoch      = 0
    best_model_name = ""

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Switch to full fine-tuning after FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS:
            model.unfreeze_all()
            optimizer = optim.Adam(model.parameters(), lr=LR_FULL)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_recall, val_f1, val_preds, val_true = validate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        for key, val in zip(
            ["train_loss", "val_loss", "train_acc", "val_acc", "val_recall", "val_f1"],
            [train_loss, val_loss, train_acc, val_acc, val_recall, val_f1]
        ):
            history[key].append(val)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

        # Save best model: combined score with recall >= 0.85 gate
        combined_score = 0.4 * val_recall + 0.6 * val_f1
        if combined_score > best_score and val_recall >= 0.85:
            best_score  = combined_score
            best_recall = val_recall
            best_epoch  = epoch + 1

            if best_model_name and os.path.exists(best_model_name):
                os.remove(best_model_name)

            best_model_name = f"current_intensity_classifier_epoch{epoch + 1}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "combined_score": combined_score,
                },
                best_model_name,
            )
            print(f"  ✓ Model saved (score={combined_score:.4f} | Recall={val_recall:.4f} | F1={val_f1:.4f})")

    plot_training_history(history, freeze_epochs=FREEZE_EPOCHS)

    # 5. Threshold tuning on validation set
    print("\n5. Finding optimal decision threshold on validation set...")
    if not best_model_name:
        raise RuntimeError("No model was saved — val_recall never reached 0.85 during training.")
    model.load_state_dict(torch.load(best_model_name)["model_state_dict"])
    best_threshold = find_best_threshold(model, val_loader, device)

    # 6. Final evaluation on test set
    print("\n6. Final evaluation on test set...")
    test_loss, test_acc, test_recall, test_f1, test_preds, test_true = validate_epoch(
        model, test_loader, criterion, device, threshold=best_threshold
    )

    print(f"\n{'=' * 50}")
    print("FINAL TEST RESULTS - CURRENT INTENSITY CLASSIFICATION")
    print(f"{'=' * 50}")
    print(f"Decision threshold : {best_threshold:.2f}")
    print(f"Accuracy           : {test_acc:.4f}  (Target: > 0.90)  {'✓' if test_acc > 0.90 else '✗'}")
    print(f"Recall             : {test_recall:.4f}  (Target: > 0.85)  {'✓' if test_recall > 0.85 else '✗'}")
    print(f"F1-Score           : {test_f1:.4f}")
    print(f"Best epoch         : {best_epoch}")

    cm = confusion_matrix(test_true, test_preds)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(test_true, test_preds, target_names=["Insufficient", "Sufficient"]))

    results = {
        "test_accuracy"   : float(test_acc),
        "test_recall"     : float(test_recall),
        "test_f1"         : float(test_f1),
        "best_epoch"      : best_epoch,
        "best_threshold"  : float(best_threshold),
        "confusion_matrix": cm.tolist(),
        "target_accuracy" : 0.90,
        "target_recall"   : 0.85,
        "objectives_met"  : {
            "accuracy": bool(test_acc > 0.90),
            "recall"  : bool(test_recall > 0.85),
        },
    }

    with open("test_results_task3.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n✓ Training complete! Results saved.")
    print(f"  - Model   : {best_model_name}")
    print("  - History : training_history.png")
    print("  - Results : test_results_task3.json")