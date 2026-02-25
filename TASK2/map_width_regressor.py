"""
Map Width Regressor - Regression Task
Prédit la largeur effective de la zone d'influence magnétique (5-80m)
à partir des images magnétiques multicanales.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from common import BaseNpzDataset, build_efficientnet_v2s_backbone



# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MapWidthDataset(BaseNpzDataset):
    """
    Dataset qui retourne (image_tensor, meta_tensor, width_target).
    meta = [original_H_m, original_W_m]  — donne au modèle l'échelle physique
    que le redimensionnement à 224x224 efface, sans fuiter la cible.
    """

    def __init__(self, file_paths, widths, target_size=(224, 224)):
        super().__init__(file_paths, target_size)
        self.widths = widths

    def _make_target(self, idx):
        return torch.tensor(self.widths[idx], dtype=torch.float32)

    def __getitem__(self, idx):
        # Load raw to capture original spatial dimensions
        raw = np.load(self.file_paths[idx], allow_pickle=True)['data']
        orig_h, orig_w = raw.shape[:2]

        image, target = super().__getitem__(idx)

        # Scale metadata: [H_m, W_m] only — no target leakage
        meta = torch.tensor(
            [orig_h * 0.2, orig_w * 0.2],
            dtype=torch.float32
        )
        return image, meta, target


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class MapWidthRegressor(nn.Module):
    """
    Régression de largeur magnétique basée sur EfficientNet-V2-S.
    Fusionne les features CNN avec les métadonnées physiques (H_m, W_m, fwhm)
    pour permettre au modèle de raisonner sur l'échelle réelle.
    """

    def __init__(self, num_channels: int = 4, meta_dim: int = 2, pretrained: bool = False):
        super().__init__()
        backbone, in_features = build_efficientnet_v2s_backbone(num_channels, pretrained)
        self.features = backbone

        # Fusion: CNN features + meta
        fused = in_features + meta_dim
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fused, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x, meta):
        feats = self.features(x)           # (B, in_features)
        fused = torch.cat([feats, meta], dim=1)  # (B, in_features + meta_dim)
        return self.regressor(fused).squeeze(1)  # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data_with_widths(data_dir):
    """
    Charge les labels depuis pipe_detection_label.csv (ground-truth de simulation).
    Seuls les fichiers avec label=1 (présence de conduite) sont conservés.
    Le fichier CSV est délimité par des points-virgules.
    """
    csv_path = os.path.join(data_dir, 'pipe_detection_label.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Label CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=';')
    # Keep only samples with a pipe present
    df = df[df['label'] == 1].reset_index(drop=True)

    file_paths, widths = [], []
    missing = 0

    for _, row in df.iterrows():
        fpath = os.path.join(data_dir, row['field_file'])
        if os.path.exists(fpath):
            file_paths.append(fpath)
            widths.append(float(row['width_m']))
        else:
            missing += 1

    print(f"  ✓ {len(file_paths)} samples loaded from CSV | {missing} files missing on disk")
    return file_paths, widths


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, meta, widths in tqdm(dataloader, desc="Training"):
        images  = images.to(device)
        meta    = meta.to(device)
        widths  = widths.to(device)

        optimizer.zero_grad()
        outputs = model(images, meta)
        loss    = criterion(outputs, widths)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(widths.cpu().numpy())

    mae = mean_absolute_error(all_targets, all_preds)
    return running_loss / len(dataloader), mae


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, meta, widths in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            meta   = meta.to(device)
            widths = widths.to(device)

            outputs = model(images, meta)
            loss    = criterion(outputs, widths)

            running_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(widths.cpu().numpy())

    mae   = mean_absolute_error(all_targets, all_preds)
    rmse  = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2    = r2_score(all_targets, all_preds)

    return running_loss / len(dataloader), mae, rmse, r2, all_preds, all_targets


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history, save_path='training_history.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'],   label='Val Loss')
    axes[0].set_title('Loss (MSE)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history['train_mae'], label='Train MAE')
    axes[1].plot(history['val_mae'],   label='Val MAE')
    axes[1].axhline(y=1.0, color='r', linestyle='--', label='Target MAE < 1m')
    axes[1].set_title('MAE (metres)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (m)')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")
    plt.close()


def plot_predictions_vs_targets(preds, targets, save_path='predictions_vs_targets.png'):
    preds   = np.array(preds)
    targets = np.array(targets)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    axes[0].scatter(targets, preds, alpha=0.5, s=20)
    lims = [max(0, min(targets.min(), preds.min()) - 2),
            max(targets.max(), preds.max()) + 2]
    axes[0].plot(lims, lims, 'r--', label='Perfect prediction')
    axes[0].set_xlabel('True width (m)')
    axes[0].set_ylabel('Predicted width (m)')
    axes[0].set_title('Predictions vs Targets')
    axes[0].legend(); axes[0].grid(True)

    # Error histogram
    errors = preds - targets
    axes[1].hist(errors, bins=30, edgecolor='black')
    axes[1].axvline(x=0,    color='r', linestyle='--', label='Zero error')
    axes[1].axvline(x=+0.5, color='g', linestyle=':', label='±0.5m tolerance')
    axes[1].axvline(x=-0.5, color='g', linestyle=':')
    axes[1].set_xlabel('Prediction error (m)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Error Distribution')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Prediction plot saved to {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    DATA_DIR      = 'Training_database_float16'
    BATCH_SIZE    = 16
    NUM_EPOCHS    = 100
    LEARNING_RATE = 3e-4
    EARLY_STOP_PATIENCE = 15
    TARGET_SIZE   = (224, 224)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load & label data
    print("\n1. Loading labels from CSV...")
    file_paths, widths = load_data_with_widths(DATA_DIR)
    print(f"Total usable samples : {len(file_paths)}")
    print(f"Width range          : {min(widths):.1f}m – {max(widths):.1f}m")
    print(f"Width mean ± std     : {np.mean(widths):.1f}m ± {np.std(widths):.1f}m")

    # 2. Split
    train_files, temp_files, train_widths, temp_widths = train_test_split(
        file_paths, widths, test_size=0.3, random_state=42
    )
    val_files, test_files, val_widths, test_widths = train_test_split(
        temp_files, temp_widths, test_size=0.5, random_state=42
    )
    print(f"\nTrain : {len(train_files)} | Val : {len(val_files)} | Test : {len(test_files)}")

    # 3. Datasets & Loaders
    print("\n2. Creating datasets...")
    train_ds = MapWidthDataset(train_files, train_widths, TARGET_SIZE)
    val_ds   = MapWidthDataset(val_files,   val_widths,   TARGET_SIZE)
    test_ds  = MapWidthDataset(test_files,  test_widths,  TARGET_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. Model, loss, optimizer
    print("\n3. Creating model (EfficientNet-V2-S for regression)...")
    model     = MapWidthRegressor(num_channels=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 5. Training loop
    print("\n4. Training...")
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mae':  [], 'val_mae':  [],
        'val_rmse': [], 'val_r2': []
    }
    best_mae, best_epoch = float('inf'), 0
    early_stop_counter = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae, val_rmse, val_r2, _, _ = validate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)

        print(f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.3f}m")
        print(f"Val   Loss: {val_loss:.4f}  | Val   MAE: {val_mae:.3f}m | RMSE: {val_rmse:.3f}m | R²: {val_r2:.4f}")

        if val_mae < best_mae:
            best_mae   = val_mae
            best_epoch = epoch + 1
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae':  float(val_mae),
                'val_rmse': float(val_rmse),
                'val_r2':   float(val_r2),
            }, 'best_map_width_regressor.pth')
            print(f"✓ Best model saved (MAE: {val_mae:.3f}m)")
        else:
            early_stop_counter += 1
            print(f"  No improvement ({early_stop_counter}/{EARLY_STOP_PATIENCE})")
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    plot_training_history(history)

    # 6. Final evaluation
    print("\n5. Final evaluation on test set...")
    model.load_state_dict(torch.load('best_map_width_regressor.pth', weights_only=False)['model_state_dict'])
    test_loss, test_mae, test_rmse, test_r2, test_preds, test_targets = validate_epoch(
        model, test_loader, criterion, device
    )

    within_tolerance = np.mean(np.abs(np.array(test_preds) - np.array(test_targets)) <= 0.5) * 100

    print(f"\n{'='*50}")
    print("FINAL TEST RESULTS")
    print(f"{'='*50}")
    print(f"MAE  : {test_mae:.3f}m  (Target: < 1m)")
    print(f"RMSE : {test_rmse:.3f}m")
    print(f"R²   : {test_r2:.4f}")
    print(f"Within ±0.5m tolerance: {within_tolerance:.1f}%")
    print(f"Best epoch: {best_epoch}")

    plot_predictions_vs_targets(test_preds, test_targets)

    results = {
        'test_mae':   float(test_mae),
        'test_rmse':  float(test_rmse),
        'test_r2':    float(test_r2),
        'within_0_5m_tolerance_pct': float(within_tolerance),
        'best_epoch': best_epoch,
        'target_mae': 1.0,
        'objectives_met': {
            'mae_lt_1m': test_mae < 1.0,
        }
    }
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\n✓ Training complete! Results saved.")
    print("  - Model  : best_map_width_regressor.pth")
    print("  - History: training_history.png")
    print("  - Scatter: predictions_vs_targets.png")
    print("  - Results: test_results.json")


if __name__ == "__main__":
    main()
