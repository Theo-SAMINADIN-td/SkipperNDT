import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import glob
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


class PipelineDataset(Dataset):
    """Dataset pour charger les images .npz avec labels automatiques"""

    def __init__(self, file_paths, labels, transform=None, target_size=(224, 224)):
        self.file_paths = file_paths
        self.labels = labels
        self.target_size = target_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx], allow_pickle=True)
        image = data["data"]

        if image.dtype == np.float16:
            image = image.astype(np.float32)  # For scipy zoom

        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        image = self._resize_image(image)

        image = self._normalize_channels(image)

        image = torch.from_numpy(image).permute(2, 0, 1).float()

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label

    def _resize_image(self, image):
        """Redimensionne l'image à la taille cible"""

        h, w, c = image.shape
        target_h, target_w = self.target_size

        zoom_h = target_h / h
        zoom_w = target_w / w

        resized = zoom(image, (zoom_h, zoom_w, 1), order=1)

        return resized

    def _normalize_channels(self, image):
        """Normalise chaque canal indépendamment"""
        normalized = np.zeros_like(image)

        for c in range(image.shape[2]):
            channel = image[:, :, c]
            mean = np.mean(channel)
            std = np.std(channel)

            if std > 0:
                normalized[:, :, c] = (channel - mean) / std
            else:
                normalized[:, :, c] = channel - mean

        return normalized


class PipelinePresenceClassifier(nn.Module):
    """Pretrained ResNet pour la classification binaire 4 canaux, sans sigmoid final (car on utilise BCEWithLogitsLoss)"""

    def __init__(self):
        super(PipelinePresenceClassifier, self).__init__()
        weight = models.ResNet18_Weights.IMAGENET1K_V1
        self.resnet = models.resnet18(weights=weight)

        self.resnet.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.resnet(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
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
        all_preds.extend((outputs > 0.5).float().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / len(dataloader), accuracy_score(all_labels, all_preds)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            all_preds.extend((outputs > 0.5).float().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return epoch_loss, accuracy, recall, f1, all_preds, all_labels


def plot_training_history(history, save_path="training_history.png"):
    """Affiche l'historique d'entraînement"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    axes[0, 0].plot(history["val_loss"], label="Val Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history["train_acc"], label="Train Accuracy")
    axes[0, 1].plot(history["val_acc"], label="Val Accuracy")
    axes[0, 1].axhline(y=0.92, color="r", linestyle="--", label="Target 92%")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history["val_recall"], label="Val Recall")
    axes[1, 0].axhline(y=0.95, color="r", linestyle="--", label="Target 95%")
    axes[1, 0].set_title("Recall (Validation)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history["val_f1"], label="Val F1-Score")
    axes[1, 1].set_title("F1-Score (Validation)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("F1-Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")
    plt.close()


def load_data_with_labels(data_dir, real_data_dir):
    """
    Charge les fichiers .npz et extrait les labels à partir des noms de fichiers

    Label 0: no_pipe (absence de conduite)
    Label 1: perfect, missed, ou tout autre (présence de conduite)
    """
    csv_path = os.path.join(data_dir, "pipe_detection_label.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Label CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=";")

    file_paths, labels = [], []
    missing = 0

    for _, row in df.iterrows():
        fpath = os.path.join(data_dir, row["field_file"])
        if os.path.exists(fpath):
            file_paths.append(fpath)
            labels.append(float(row["label"]))
        else:
            missing += 1

    print(
        f"  ✓ {len(file_paths)} samples loaded from CSV | {missing} files missing on disk"
    )

    for fpath in glob.glob(os.path.join(real_data_dir, "*.npz")):
        print(f"  ✓ Adding real data sample: {fpath}")
        file_paths.append(fpath)
        labels.append(0.0 if "no_pipe" in fpath else 1.0)

    return file_paths, labels


if __name__ == "__main__":
    DATA_DIR = "Training_database_float16.zip"
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    TARGET_SIZE = (224, 224)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n1. Loading data...")
    file_paths, labels = load_data_with_labels("Training_database_float16", "real_data")
    print(f"Total samples: {len(file_paths)}")
    print(f"Class 0 (no pipe): {sum(1 for l in labels if l == 0.0)}")
    print(f"Class 1 (with pipe): {sum(1 for l in labels if l == 1.0)}")

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )

    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    print(f"\nTrain samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")

    print("\n2. Creating datasets...")
    train_dataset = PipelineDataset(train_files, train_labels, target_size=TARGET_SIZE)
    val_dataset = PipelineDataset(val_files, val_labels, target_size=TARGET_SIZE)
    test_dataset = PipelineDataset(test_files, test_labels, target_size=TARGET_SIZE)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    print("\n3. Creating model...")
    model = PipelinePresenceClassifier().to(device)

    # Loss avec pondération pour gérer le déséquilibre des classes
    num_no_pipe = sum(1 for l in train_labels if l == 0.0)
    num_with_pipe = sum(1 for l in train_labels if l == 1.0)
    weight = num_no_pipe / num_with_pipe if num_with_pipe > 0 else 1.0

    # Permet de ne pas utiliser sigmoid dans le modèle et d'avoir une meilleure stabilité numérique
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    print("\n4. Training model...")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_recall": [],
        "val_f1": [],
    }

    best_recall = 0.0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_recall, val_f1, val_preds, val_true = validate_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_recall": val_recall,
                "val_f1": val_f1,
            },
            f"pipeline_classifier_epoch{epoch + 1}_{time.time()}.pth",
        )
        print(f"Model saved (Recall: {val_recall:.4f})")

    plot_training_history(history)

    print("\n5. Final evaluation on test set...")
    model.load_state_dict(
        torch.load(f"pipeline_classifier_epoch{NUM_EPOCHS}_{time.time()}.pth")[
            "model_state_dict"
        ]
    )

    test_loss, test_acc, test_recall, test_f1, test_preds, test_true = validate_epoch(
        model, test_loader, criterion, device
    )

    print(f"\n{'=' * 50}")
    print("FINAL TEST RESULTS")
    print(f"{'=' * 50}")
    print(f"Accuracy: {test_acc:.4f} (Target: > 0.92)")
    print(f"Recall: {test_recall:.4f} (Target: > 0.95)")
    print(f"F1-Score: {test_f1:.4f}")
    print(f"Best epoch: {best_epoch}")

    cm = confusion_matrix(test_true, test_preds)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            test_true, test_preds, target_names=["No Pipe", "With Pipe"]
        )
    )

    results = {
        "test_accuracy": float(test_acc),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
        "best_epoch": best_epoch,
        "confusion_matrix": cm.tolist(),
        "target_accuracy": 0.92,
        "target_recall": 0.95,
        "objectives_met": {"accuracy": test_acc > 0.92, "recall": test_recall > 0.95},
    }

    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n✓ Training complete! Results saved.")
    print("  - Model: best_pipeline_classifier.pth")
    print("  - History plot: training_history.png")
    print("  - Results: test_results.json")
