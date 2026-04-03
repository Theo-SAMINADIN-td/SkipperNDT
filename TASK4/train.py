"""
Pipe Type Classifier - Binary Classification (Single vs Parallel)
S'entraîne uniquement sur les images avec conduites (label=1)
Fine-tuning à partir du modèle de présence (ResNet18)
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 20
WARMUP_EPOCHS = 3
LEARNING_RATE = 1e-3
LR_BACKBONE = 1e-5
TARGET_SIZE = (224, 224)
NUM_WORKERS = 2


class BaseNpzDataset(Dataset):
    def __init__(self, file_paths, target_size=(128, 128), augment=False):
        self.file_paths = file_paths
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx], allow_pickle=True)
        image = data["data"].astype(np.float32)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = self._normalize_channels(image)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        tensor = TF.resize(tensor, list(self.target_size), antialias=True)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                tensor = TF.hflip(tensor)
            if torch.rand(1).item() > 0.5:
                tensor = TF.vflip(tensor)
            angle = torch.FloatTensor(1).uniform_(-30, 30).item()
            tensor = TF.rotate(tensor, angle)

        return tensor, self._make_target(idx)

    def _make_target(self, idx):
        raise NotImplementedError

    def _normalize_channels(self, image):
        normalized = np.zeros_like(image)
        for c in range(image.shape[2]):
            ch = image[:, :, c]
            std = np.std(ch)
            normalized[:, :, c] = (
                (ch - np.mean(ch)) / std if std > 0 else ch - np.mean(ch)
            )
        return normalized


class PipelinePresenceClassifier(nn.Module):
    """
    Correspond exactement au checkpoint task1_epoch8_best.pth :
    - resnet.conv1 adapté à 4 canaux
    - resnet.fc = Linear(512, 1)  ← pas de classifier séparé
    """

    def __init__(self, num_channels=4):
        super().__init__()
        self.resnet = models.resnet18(weights=None)

        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            num_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        return self.resnet(x)


class PipeTypeDataset(BaseNpzDataset):
    """Label 0 = single pipe | Label 1 = parallel pipes"""

    def __init__(self, file_paths, labels, target_size=(128, 128), augment=False):
        super().__init__(file_paths, target_size, augment)
        self.labels = labels

    def _make_target(self, idx):
        return torch.tensor(self.labels[idx], dtype=torch.float32)


class PipeTypeClassifier(nn.Module):
    """Classifieur single vs parallel — backbone ResNet18 transféré depuis Task1"""

    def __init__(self, num_channels=4):
        super().__init__()
        self.resnet = models.resnet18(weights=None)

        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            num_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, in_features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.classifier(self.resnet(x))

    def load_backbone_from_presence_model(self, presence_model_path, device):
        if not os.path.exists(presence_model_path):
            raise FileNotFoundError(
                f"Modèle de présence introuvable : {presence_model_path}"
            )

        checkpoint = torch.load(
            presence_model_path, map_location=device, weights_only=False
        )
        presence_model = PipelinePresenceClassifier(num_channels=4)
        presence_model.load_state_dict(checkpoint["model_state_dict"])

        backbone_state = {
            k: v
            for k, v in presence_model.resnet.state_dict().items()
            if not k.startswith("fc.")
        }
        self.resnet.load_state_dict(backbone_state, strict=False)
        print("✓ Poids du backbone ResNet18 transférés (couches conv + bn, hors fc)")

        for param in self.resnet.parameters():
            param.requires_grad = False
        print(f"✓ Backbone gelé pour {WARMUP_EPOCHS} époques de warm-up")


def load_pipe_type_data(csv_path, data_dir_train, data_dir_real):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    df = pd.read_csv(csv_path, sep=";")
    df_pipes = df[df["label"] == 1].copy()

    print(f"  Samples avec conduite : {len(df_pipes)}")
    print(f"  Single pipe           : {(df_pipes['pipe_type'] == 'single').sum()}")
    print(f"  Parallel pipe         : {(df_pipes['pipe_type'] == 'parallel').sum()}")
    if "source" in df_pipes.columns:
        print(
            f"  Répartition sources   :\n{df_pipes['source'].value_counts().to_string()}"
        )

    file_paths, labels = [], []
    found_train, found_real, missing = 0, 0, 0

    for _, row in df_pipes.iterrows():
        field_file = str(row["field_file"]).strip()

        path_train = os.path.join(data_dir_train, field_file)
        path_real = os.path.join(data_dir_real, field_file)

        if os.path.exists(path_train):
            fpath = path_train
            found_train += 1
        elif os.path.exists(path_real):
            fpath = path_real
            found_real += 1
        else:
            missing += 1
            continue

        file_paths.append(fpath)
        labels.append(1.0 if row["pipe_type"] == "parallel" else 0.0)

    print(
        f"  ✓ {len(file_paths)} samples chargés "
        f"({found_train} training_db | {found_real} real_data) | {missing} manquants"
    )
    return file_paths, labels


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, all_preds, all_labels = 0.0, [], []

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_preds.extend((torch.sigmoid(outputs) > 0.5).float().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / len(dataloader), accuracy_score(all_labels, all_preds)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            outputs = model(images)
            running_loss += criterion(outputs, labels).item()
            all_preds.extend((torch.sigmoid(outputs) > 0.5).float().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return epoch_loss, accuracy, recall, f1, all_preds, all_labels


def unfreeze_backbone(model, optimizer, lr_backbone):
    for param in model.resnet.parameters():
        param.requires_grad = True
    optimizer.add_param_group({"params": model.resnet.parameters(), "lr": lr_backbone})
    print(f"✓ Backbone dégelé avec lr={lr_backbone}")


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    axes[0, 0].plot(history["val_loss"], label="Val Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history["train_acc"], label="Train Accuracy")
    axes[0, 1].plot(history["val_acc"], label="Val Accuracy")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history["val_recall"], label="Val Recall (Parallel)")
    axes[1, 0].set_title("Recall - Parallel Pipes")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history["val_f1"], label="Val F1")
    axes[1, 1].set_title("F1-Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  ✓ Historique sauvegardé : {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Train a pipeline presence classifier on magnetic images"
    )
    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Path to .npz folder",
        default="Training_database_float16/*.npz",
    )
    parser.add_argument(
        "--csv-data",
        type=str,
        required=True,
        help="Path to CSV file",
        default="Training_database_float16/pipe_detection_label.csv",
    )
    parser.add_argument(
        "--real-data",
        type=str,
        required=True,
        help="Path to real data folder",
        default="real_data",
    )
    parser.add_argument(
        "--presence-model",
        type=str,
        required=True,
        help="Path to presence model file",
        default="task1_epoch8_best.pth",
    )
    args = parser.parse_args()
    DATA_DIR_TRAIN = args.input_data
    CSV_PATH = args.csv_data
    DATA_DIR_REAL = args.real_data
    PRESENCE_MODEL_PATH = args.presence_model

    MODELS_DIR = "./MODELS"
    os.makedirs(MODELS_DIR, exist_ok=True)

    BEST_TYPE_MODEL_PATH = f"{MODELS_DIR}/best_pipe_type_classifier.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print(" GPU non détecté. To CPU.")

    print("\n1. Chargement des données (pipes uniquement)...")
    file_paths, labels = load_pipe_type_data(CSV_PATH, DATA_DIR_TRAIN, DATA_DIR_REAL)

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    print(
        f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}"
    )

    # 2. DataLoaders
    print("\n2. Création des datasets...")
    train_loader = DataLoader(
        PipeTypeDataset(train_files, train_labels, TARGET_SIZE, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        PipeTypeDataset(val_files, val_labels, TARGET_SIZE, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        PipeTypeDataset(test_files, test_labels, TARGET_SIZE, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Modèle + transfert backbone
    print("\n3. Création du modèle avec transfert de backbone ResNet18...")
    model = PipeTypeClassifier(num_channels=4).to(device)
    model.load_backbone_from_presence_model(PRESENCE_MODEL_PATH, device)
    model = model.to(device)

    num_single = train_labels.count(0.0)
    num_parallel = train_labels.count(1.0)
    pos_weight = torch.tensor(num_single / num_parallel).to(device)
    print(f"  pos_weight (parallel boost): {pos_weight:.2f}")
    print("  Augmentation activée sur le train set (hflip + vflip + rotation ±30°)")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # 4. Entraînement
    print("\n4. Entraînement...")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_recall": [],
        "val_f1": [],
    }
    best_f1, best_epoch = 0.0, 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        if epoch == WARMUP_EPOCHS:
            print("\n--- Fin du warm-up : dégel du backbone ---")
            unfreeze_backbone(model, optimizer, LR_BACKBONE)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_recall, val_f1, _, _ = validate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        for k, v in zip(
            history.keys(),
            [train_loss, val_loss, train_acc, val_acc, val_recall, val_f1],
        ):
            history[k].append(v)

        print(f"  Train → Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(
            f"  Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch + 1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": float(val_acc),
                    "val_recall": float(val_recall),
                    "val_f1": float(val_f1),
                },
                BEST_TYPE_MODEL_PATH,
            )
            print(f"  ✓ Meilleur modèle sauvegardé sur Drive (F1: {val_f1:.4f})")

    plot_training_history(history, f"{MODELS_DIR}/pipe_type_training_history.png")

    # 5. Évaluation finale
    print("\n5. Évaluation finale sur le test set...")
    model.load_state_dict(
        torch.load(BEST_TYPE_MODEL_PATH, map_location=device, weights_only=False)[
            "model_state_dict"
        ]
    )

    _, test_acc, test_recall, test_f1, test_preds, test_true = validate_epoch(
        model, test_loader, criterion, device
    )

    print(f"\n{'=' * 50}")
    print("FINAL TEST RESULTS - PIPE TYPE")
    print(f"{'=' * 50}")
    print(f"Accuracy  : {test_acc:.4f}")
    print(f"Recall    : {test_recall:.4f}")
    print(f"F1-Score  : {test_f1:.4f}")
    print(f"Best epoch: {best_epoch}")

    cm = confusion_matrix(test_true, test_preds)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(
        classification_report(
            test_true, test_preds, target_names=["Single Pipe", "Parallel Pipes"]
        )
    )

    results = {
        "test_accuracy": float(test_acc),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
        "best_epoch": best_epoch,
        "confusion_matrix": cm.tolist(),
    }
    results_path = f"{MODELS_DIR}/pipe_type_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n✓ Entraînement terminé !")
    print(f"  Modèle   : {BEST_TYPE_MODEL_PATH}")
    print(f"  Résultats: {results_path}")


if __name__ == "__main__":
    main()
