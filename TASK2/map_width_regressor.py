"""
MAP WIDTH REGRESSOR - TÂCHE 2: REGRESSION
Prédire la largeur effective (en mètres) de la zone
d'influence magnétique d'un tuyau enfoui

Comment ça marche :
  1. On charge l'IMAGE depuis les fichiers .npz  (données brutes du drone)
  2. On charge le LABEL depuis le fichier .csv   (largeur réelle en mètres)
  3. Le CNN apprend à prédire la largeur en regardant l'image
  4. On mesure l'erreur avec MAE (objectif: MAE < 1.0m)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
# CHEMINS VERS LES DONNÉES
# ══════════════════════════════════════════════════════════════════
DATA_DIR = '/Users/macdekhail/Desktop/SkipperNDT/TASK2/data/Training_database_float16'
CSV_PATH = '/Users/macdekhail/Desktop/SkipperNDT/TASK2/data/Training_database_float16/pipe_detection_label.csv'

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

CONFIG = {
    'BATCH_SIZE'   : 16,
    'LEARNING_RATE': 1e-4,
    'NUM_EPOCHS'   : 30,
    'WEIGHT_DECAY' : 1e-4,
    'PATIENCE'     : 15,
    'TARGET_SIZE'  : 224,
    'NUM_CHANNELS' : 4,
    'WIDTH_MIN'    : 5.0,
    'WIDTH_MAX'    : 80.0,
    'DEVICE'       : 'cuda' if torch.cuda.is_available() else 'cpu',
    'SEED'         : 42,
}

print(f"Device utilisé: {CONFIG['DEVICE']}")

# ══════════════════════════════════════════════════════════════════
# ARCHITECTURE CNN
# ══════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Bloc convolutif: Conv → BatchNorm → ReLU → MaxPool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class MapWidthRegressor(nn.Module):
    """
    CNN de régression.
    Entrée : (Batch, 4, 224, 224)
    Sortie : (Batch, 1)  <- largeur en mètres, pas de Sigmoid !
    """
    def __init__(self, num_channels=4):
        super().__init__()

        self.block1 = ConvBlock(num_channels, 64)
        self.block2 = ConvBlock(64,  128)
        self.block3 = ConvBlock(128, 256)
        self.block4 = ConvBlock(256, 512)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.adaptive_pool(x)
        x = self.fc_head(x)
        return x


# ══════════════════════════════════════════════════════════════════
# DATASET - relie chaque NPZ (image) a son label (width_m du CSV)
# ══════════════════════════════════════════════════════════════════

class MapWidthDataset(Dataset):
    """
    Pour chaque fichier NPZ :
      - L'IMAGE  vient du NPZ  -> npz['data']  shape (H, W, 4)
      - Le LABEL vient du CSV  -> colonne width_m

    Le lien entre les deux = le NOM DU FICHIER.
    """

    def __init__(self, npz_files, csv_path, target_size=224, normalize=True):
        self.target_size = target_size
        self.normalize   = normalize
        self.data        = []
        self.labels      = []

        # Charger le CSV et construire un dictionnaire nom -> largeur
        df = pd.read_csv(csv_path, sep=';')

        # Garder seulement les fichiers avec tuyau (label=1)
        df_pipe = df[df['label'] == 1]

        # { 'sample_00000.npz': 62.4, ... }
        width_lookup = dict(zip(df_pipe['field_file'], df_pipe['width_m']))

        print(f"CSV charge -> {len(width_lookup)} fichiers avec tuyau")
        print(f"Chargement des images NPZ...")

        ok = 0
        skip = 0

        for i, npz_path in enumerate(npz_files):
            try:
                nom = Path(npz_path).name

                if nom not in width_lookup:
                    skip += 1
                    continue

                width_label = float(width_lookup[nom])

                npz_data = np.load(npz_path, allow_pickle=True)
                mag_data = npz_data['data']

                if mag_data.shape[2] != 4:
                    skip += 1
                    continue

                self.data.append(mag_data)
                self.labels.append(width_label)
                ok += 1

                if ok % 200 == 0:
                    print(f"   {ok} images chargees...")

            except Exception as e:
                skip += 1
                continue

        print(f"OK: {ok} echantillons valides")
        print(f"Ignores: {skip} fichiers (no_pipe ou absent du CSV)\n")

        if len(self.data) == 0:
            raise ValueError("0 echantillon valide ! Verifiez DATA_DIR et CSV_PATH.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mag_data    = self.data[idx].astype(np.float32)
        width_label = self.labels[idx]

        mag_data[np.isnan(mag_data)] = 0
        mag_data[np.isinf(mag_data)] = 0

        if mag_data.shape[0] != self.target_size or mag_data.shape[1] != self.target_size:
            mag_data = self._resize(mag_data, self.target_size)

        if self.normalize:
            for c in range(4):
                ch  = mag_data[:, :, c]
                std = np.std(ch)
                if std > 0:
                    mag_data[:, :, c] = (ch - np.mean(ch)) / std

        mag_data = np.transpose(mag_data, (2, 0, 1))

        return torch.from_numpy(mag_data).float(), torch.tensor(width_label, dtype=torch.float32)

    @staticmethod
    def _resize(image, size):
        from scipy.ndimage import zoom
        h, w, _ = image.shape
        return zoom(image, [size/h, size/w, 1.0], order=0)


# ══════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════

class RegressionTrainer:

    def __init__(self, model, config, device):
        self.model      = model.to(device)
        self.config     = config
        self.device     = device
        self.loss_train = nn.MSELoss()
        self.loss_eval  = nn.L1Loss()
        self.optimizer  = optim.Adam(
            model.parameters(),
            lr=config['LEARNING_RATE'],
            weight_decay=config['WEIGHT_DECAY']
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=7
        )
        self.history = {
            'train_loss': [], 'val_mae': [], 'val_mse': [],
            'best_epoch': 0,  'best_val_mae': float('inf')
        }

    def _train_epoch(self, loader):
        self.model.train()
        total = 0.0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)
            out    = self.model(images)
            loss   = self.loss_train(out, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        return total / len(loader)

    def _validate(self, loader):
        self.model.eval()
        mae_t, mse_t = 0.0, 0.0
        preds, targets = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                out    = torch.clamp(self.model(images), self.config['WIDTH_MIN'], self.config['WIDTH_MAX'])
                mae_t += self.loss_eval(out, labels).item()
                mse_t += self.loss_train(out, labels).item()
                preds.extend(out.cpu().numpy().squeeze())
                targets.extend(labels.cpu().numpy().squeeze())
        n = len(loader)
        return mae_t/n, mse_t/n, np.array(preds), np.array(targets)

    def train(self, train_loader, val_loader, save_path):
        print("Demarrage entrainement...")
        print("=" * 70)
        best_mae     = float('inf')
        patience_cpt = 0

        for epoch in range(self.config['NUM_EPOCHS']):
            train_loss            = self._train_epoch(train_loader)
            val_mae, val_mse, _, _ = self._validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_mae'].append(val_mae)
            self.history['val_mse'].append(val_mse)
            self.scheduler.step(val_mae)

            print(f"Epoch {epoch+1:3d}/{self.config['NUM_EPOCHS']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val MAE: {val_mae:.4f}m")

            if val_mae < best_mae:
                best_mae     = val_mae
                patience_cpt = 0
                self.history['best_val_mae'] = best_mae
                self.history['best_epoch']   = epoch + 1
                torch.save(self.model.state_dict(), save_path)
                print(f"   Meilleur modele sauvegarde (MAE: {val_mae:.4f}m)")
            else:
                patience_cpt += 1
                if patience_cpt >= self.config['PATIENCE']:
                    print(f"\nEarly stopping a l'epoch {epoch+1}")
                    break

        print("=" * 70)
        print(f"Entrainement termine - Meilleur MAE: {best_mae:.4f}m\n")


# ══════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════

def split_dataset(files):
    np.random.seed(CONFIG['SEED'])
    np.random.shuffle(files)
    n       = len(files)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    return files[:n_train], files[n_train:n_train+n_val], files[n_train+n_val:]


def plot_history(history, out_dir):
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['train_loss'], linewidth=2, label='Train MSE')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history['val_mae'], linewidth=2, color='orange', label='Val MAE')
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Objectif 1.0m')
    ax2.axvline(x=history['best_epoch']-1, color='g', linestyle='--', label='Meilleur modele')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MAE (metres)')
    ax2.set_title('Validation MAE', fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'training_history.png', dpi=100)
    plt.close()
    print("training_history.png sauvegarde")


def plot_scatter(targets, preds, out_dir):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets, preds, alpha=0.5, s=30)
    v = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax.plot(v, v, 'r--', lw=2, label='Prediction parfaite')
    ax.set_xlabel('Reel (m)'); ax.set_ylabel('Predit (m)')
    ax.set_title('Predictions vs Realite', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'scatter_predicted_vs_real.png', dpi=100)
    plt.close()
    print("scatter_predicted_vs_real.png sauvegarde")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*70)
    print("   MAP WIDTH REGRESSOR - TACHE 2")
    print("="*70 + "\n")

    out_dir = Path(__file__).parent / 'outputs'
    out_dir.mkdir(exist_ok=True)

    # 1. Trouver tous les fichiers NPZ
    npz_files = [str(p) for p in Path(DATA_DIR).glob('*.npz')]
    if not npz_files:
        print(f"Aucun NPZ dans {DATA_DIR}")
        return
    print(f"{len(npz_files)} fichiers NPZ trouves\n")

    # 2. Split 70/15/15
    train_files, val_files, test_files = split_dataset(npz_files)
    print(f"Split: Train={len(train_files)} | Val={len(val_files)} | Test={len(test_files)}\n")

    # 3. Creer les datasets (NPZ + CSV)
    train_ds = MapWidthDataset(train_files, CSV_PATH)
    val_ds   = MapWidthDataset(val_files,   CSV_PATH)
    test_ds  = MapWidthDataset(test_files,  CSV_PATH)

    print(f"Labels train - Min: {min(train_ds.labels):.1f}m | Max: {max(train_ds.labels):.1f}m | "
          f"Mean: {np.mean(train_ds.labels):.1f}m\n")

    # 4. DataLoaders
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=0)

    # 5. Modele
    model    = MapWidthRegressor(num_channels=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modele cree - {n_params:,} parametres\n")

    # 6. Entrainement
    save_path = out_dir / 'best_map_width_regressor.pth'
    trainer   = RegressionTrainer(model, CONFIG, CONFIG['DEVICE'])
    trainer.train(train_loader, val_loader, str(save_path))

    # 7. Evaluation finale sur le test set
    model.load_state_dict(torch.load(save_path, map_location=CONFIG['DEVICE']))
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            out = torch.clamp(model(images.to(CONFIG['DEVICE'])),
                              CONFIG['WIDTH_MIN'], CONFIG['WIDTH_MAX'])
            preds.extend(out.cpu().numpy().squeeze())
            targets.extend(labels.numpy().squeeze())

    preds   = np.array(preds)
    targets = np.array(targets)

    mae  = mean_absolute_error(targets, preds)
    mse  = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    r2   = r2_score(targets, preds)

    print("\n" + "="*70)
    print("RESULTATS FINAUX")
    print("="*70)
    print(f"MAE  : {mae:.4f}m  {'OBJECTIF ATTEINT' if mae < 1.0 else 'OBJECTIF NON ATTEINT'}")
    print(f"RMSE : {rmse:.4f}m")
    print(f"R2   : {r2:.4f}")
    print("="*70 + "\n")

    # 8. Visualisations
    plot_history(trainer.history, out_dir)
    plot_scatter(targets, preds, out_dir)

    # 9. Sauvegarder JSON
    results = {
        'timestamp'   : datetime.now().isoformat(),
        'test_metrics': {'mae': float(mae), 'mse': float(mse), 'rmse': float(rmse), 'r2': float(r2)},
        'dataset'     : {'train': len(train_ds), 'val': len(val_ds), 'test': len(test_ds)},
        'best_epoch'  : trainer.history['best_epoch'],
    }
    with open(out_dir / 'regression_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Resultats sauvegardes dans outputs/")
    print("Tache 2 terminee !\n")


if __name__ == '__main__':
    main()