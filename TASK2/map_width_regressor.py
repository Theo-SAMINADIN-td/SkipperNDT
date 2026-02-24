"""

     MAP WIDTH REGRESSOR - TÂCHE 2: REGRESSION                 
  Prédire la largeur effective (en mètres) de la zone           
  d'influence magnétique d'un tuyau enfoui                      


Architecture:
  - Entrée: (Batch, 4, 224, 224) [Bx, By, Bz, Norm]
  - Backbone convolutif (4 blocs) → 512 features
  - Tête de régression: 512 → 256 → 128 → 1
  - Sortie: valeur continue (5-80 mètres)

Loss: MSELoss (entraînement), MAE (évaluation)
Métrique clé: MAE < 1.0m
"""

import os
import json
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Visualization & metrics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# 
# CONFIGURATION DES HYPERPARAMÈTRES
# 
CONFIG = {
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 1e-4,
    'NUM_EPOCHS': 100,
    'WEIGHT_DECAY': 1e-4,
    'PATIENCE': 15,  # Early stopping
    'TARGET_SIZE': 224,
    'NUM_CHANNELS': 4,
    'WIDTH_MIN': 5.0,
    'WIDTH_MAX': 80.0,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'SEED': 42,
}

print(f"Device utilisé: {CONFIG['DEVICE']}")


# 
# ARCHITECTURE DU RÉSEAU DE NEURONES
# 

class ConvBlock(nn.Module):
    """
    Bloc convolutif standard:
    Conv2d → BatchNorm → ReLU → MaxPool
    
    Paramètres:
    - in_channels: nombre de canaux en entrée
    - out_channels: nombre de canaux en sortie
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
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
    Modèle de régression pour prédire la largeur de la zone d'influence magnétique.
    
    Architecture:
    - 4 blocs convolutifs progressifs (4→64→128→256→512 canaux)
    - AdaptiveAvgPool pour normaliser les dimensions spatiales
    - Tête de régression: FC 512→256→128→1 (sortie linéaire)
    """
    def __init__(self, num_channels=4, num_output=1):
        super(MapWidthRegressor, self).__init__()
        
        #  BACKBONE CONVOLUTIF 
        # Bloc 1: 4 canaux → 64 canaux (224×224 → 112×112)
        self.block1 = ConvBlock(num_channels, 64)
        
        # Bloc 2: 64 canaux → 128 canaux (112×112 → 56×56)
        self.block2 = ConvBlock(64, 128)
        
        # Bloc 3: 128 canaux → 256 canaux (56×56 → 28×28)
        self.block3 = ConvBlock(128, 256)
        
        # Bloc 4: 256 canaux → 512 canaux (28×28 → 14×14)
        self.block4 = ConvBlock(256, 512)
        
        #  POOL & FLATTEN 
        # AdaptiveAvgPool réduit les features spatiales à (512, 1, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        #  TÊTE DE RÉGRESSION 
        # Prendre les 512 features et prédire 1 valeur de largeur
        self.fc_head = nn.Sequential(
            nn.Flatten(),  # (B, 512, 1, 1) → (B, 512)
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_output)  # Sortie linéaire (pas d'activation)
        )
    
    def forward(self, x):
        """
        Forward pass du réseau.
        Entrée: (B, 4, 224, 224)
        Sortie: (B, 1) - largeur en mètres
        """
        # Backbone convolutif
        x = self.block1(x)  # (B, 64, 112, 112)
        x = self.block2(x)  # (B, 128, 56, 56)
        x = self.block3(x)  # (B, 256, 28, 28)
        x = self.block4(x)  # (B, 512, 14, 14)
        
        # Pool et flatten
        x = self.adaptive_pool(x)  # (B, 512, 1, 1)
        
        # Tête de régression
        x = self.fc_head(x)  # (B, 1)
        
        return x


# 
# DATASET POUR LES FICHIERS .NPZ
# 

class MapWidthDataset(Dataset):
    """
    Dataset pour charger des fichiers .npz avec annotations de largeur (label).
    
    Structure attendue du .npz:
    - 'data': (H, W, 4) - les 4 canaux magnétiques
    - 'width' ou 'label': valeur float - la largeur en mètres
    
    Preprocessing:
    1. Remplacer NaN et Inf par 0
    2. Redimensionner à 224×224
    3. Normaliser (mean=0, std=1) par canal
    4. Transposer (H,W,C) → (C,H,W)
    """
    
    def __init__(self, npz_files, target_size=224, normalize=True):
        """
        Paramètres:
        - npz_files: liste des chemins vers les fichiers .npz
        - target_size: dimensions cibles (224×224)
        - normalize: appliquer la normalisation
        """
        self.npz_files = npz_files
        self.target_size = target_size
        self.normalize = normalize
        
        # Précharger les données pour éviter I/O répétés
        self.data = []
        self.labels = []
        
        print(f" Chargement de {len(npz_files)} fichiers .npz...")
        for i, npz_path in enumerate(npz_files):
            try:
                npz_data = np.load(npz_path, allow_pickle=True)
                
                # Extraire la signature magnétique
                if 'data' in npz_data.files:
                    mag_data = npz_data['data']
                else:
                    # Chercher la première clé qui contient une array 3D
                    for key in npz_data.files:
                        candidate = npz_data[key]
                        if isinstance(candidate, np.ndarray) and len(candidate.shape) == 3:
                            mag_data = candidate
                            break
                
                # Extraire le label (largeur)
                width_label = None
                for key in ['width', 'label', 'map_width', 'target']:
                    if key in npz_data.files:
                        width_label = float(npz_data[key])
                        break
                
                # Valider le label
                if width_label is None or np.isnan(width_label) or width_label <= 0:
                    print(f"    {Path(npz_path).name}: label invalide ({width_label}), ignoré")
                    continue
                
                # Valider les dimensions magnétiques
                if mag_data.shape[2] != 4:
                    print(f"    {Path(npz_path).name}: 4 canaux attendus, {mag_data.shape[2]} trouvés")
                    continue
                
                self.data.append(mag_data)
                self.labels.append(width_label)
                
                if (i + 1) % 50 == 0:
                    print(f"   {i + 1} fichiers chargés")
                
            except Exception as e:
                print(f"   Erreur chargement {Path(npz_path).name}: {e}")
                continue
        
        print(f" {len(self.data)} échantillons valides chargés\n")
        
        if len(self.data) == 0:
            raise ValueError("Aucun échantillon valide trouvé!")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Retourne un exemple (image_tenseur, label)"""
        mag_data = self.data[idx].astype(np.float32)
        width_label = self.labels[idx]
        
        #  NETTOYAGE 
        # Remplacer NaN et Inf par 0
        mag_data[np.isnan(mag_data)] = 0
        mag_data[np.isinf(mag_data)] = 0
        
        #  REDIMENSIONNEMENT 
        # Redimensionner via zoom simple (nearest neighbor)
        if mag_data.shape[0] != self.target_size or mag_data.shape[1] != self.target_size:
            mag_data = self._resize_image(mag_data, (self.target_size, self.target_size))
        
        #  NORMALISATION PAR CANAL 
        # Normaliser chaque canal indépendamment: (x - mean) / std
        if self.normalize:
            for c in range(mag_data.shape[2]):
                channel = mag_data[:, :, c]
                mean = np.mean(channel)
                std = np.std(channel)
                if std > 0:
                    mag_data[:, :, c] = (channel - mean) / std
        
        #  TRANSPOSITION 
        # (H, W, 4) → (4, H, W) pour PyTorch
        mag_data = np.transpose(mag_data, (2, 0, 1))
        
        # Convertir en tenseur PyTorch
        image_tensor = torch.from_numpy(mag_data).float()
        label_tensor = torch.tensor(width_label, dtype=torch.float32)
        
        return image_tensor, label_tensor
    
    @staticmethod
    def _resize_image(image, target_shape):
        """Redimensionner une image 3D (H, W, C) en (target_H, target_W, C)"""
        from scipy.ndimage import zoom
        
        h, w, c = image.shape
        target_h, target_w = target_shape
        
        zoom_factors = [target_h / h, target_w / w, 1.0]
        resized = zoom(image, zoom_factors, order=0)  # nearest neighbor
        
        return resized


# 
# PIPELINE D'ENTRAÎNEMENT
# 

class RegressionTrainer:
    """
    Encapsule la logique d'entraînement et d'évaluation pour la régression.
    """
    
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss & Optimizer
        self.criterion_train = nn.MSELoss()  # MSE pour l'entraînement
        self.criterion_eval = nn.L1Loss()    # MAE pour l'évaluation
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['LEARNING_RATE'],
            weight_decay=config['WEIGHT_DECAY']
        )
        
        # Scheduler: réduit le LR si MAE de validation n'améliore pas
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=7,
            verbose=True
        )
        
        # Historique
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_mse': [],
            'best_epoch': 0,
            'best_val_mae': float('inf'),
        }
    
    def train_epoch(self, train_loader):
        """Entraîner une époque."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)  # (B,) → (B, 1)
            
            # Forward pass
            outputs = self.model(images)  # (B, 1)
            loss = self.criterion_train(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Valider le modèle."""
        self.model.eval()
        total_mae = 0.0
        total_mse = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                outputs = self.model(images)
                
                # Clamp optionnel pour garder les valeurs réalistes
                outputs = torch.clamp(outputs, self.config['WIDTH_MIN'], self.config['WIDTH_MAX'])
                
                mae = self.criterion_eval(outputs, labels).item()
                mse = self.criterion_train(outputs, labels).item()
                
                total_mae += mae
                total_mse += mse
                
                predictions.extend(outputs.cpu().numpy().squeeze())
                targets.extend(labels.cpu().numpy().squeeze())
        
        avg_mae = total_mae / len(val_loader)
        avg_mse = total_mse / len(val_loader)
        
        return avg_mae, avg_mse, np.array(predictions), np.array(targets)
    
    def train(self, train_loader, val_loader, model_save_path):
        """
        Boucle d'entraînement complète avec early stopping.
        Sauvegarde le meilleur modèle basé sur MAE de validation.
        """
        print(" Démarrage de l'entraînement...")
        print("=" * 70)
        
        best_val_mae = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['NUM_EPOCHS']):
            # Entraînement
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_mae, val_mse, _, _ = self.validate(val_loader)
            
            # Historique
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_mse)
            self.history['val_mae'].append(val_mae)
            self.history['val_mse'].append(val_mse)
            
            # Scheduler
            self.scheduler.step(val_mae)
            
            # Affichage
            print(f"Epoch {epoch+1:3d}/{self.config['NUM_EPOCHS']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val MAE: {val_mae:.4f}m | "
                  f"Val MSE: {val_mse:.4f}")
            
            # Sauvegarde du meilleur modèle
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                self.history['best_val_mae'] = best_val_mae
                self.history['best_epoch'] = epoch + 1
                
                torch.save(self.model.state_dict(), model_save_path)
                print(f"   Meilleur modèle sauvegardé (MAE: {val_mae:.4f}m)")
            else:
                patience_counter += 1
                if patience_counter >= self.config['PATIENCE']:
                    print(f"\n⏹  Early stopping après {epoch+1} epochs")
                    break
        
        print("=" * 70)
        print(f" Entraînement terminé!")
        print(f"  Meilleur MAE de validation: {best_val_mae:.4f}m (Epoch {self.history['best_epoch']})\n")


# 
# FONCTIONS UTILITAIRES
# 

def plot_training_history(history, save_path):
    """Générer le graphe du historique d'entraînement."""
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['train_loss'], label='Train MSE', linewidth=2)
    ax1.plot(history['val_loss'], label='Val MSE', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('MSE Loss', fontsize=11)
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # MAE
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history['val_mae'], label='Validation MAE', linewidth=2, color='orange')
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Target MAE (1.0m)', linewidth=2)
    ax2.axvline(x=history['best_epoch']-1, color='g', linestyle='--', label='Best Model', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('MAE (mètres)', fontsize=11)
    ax2.set_title('Validation MAE', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f" Graphique sauvegardé: {save_path}")
    plt.close()


def split_dataset(npz_files, train_ratio=0.7, val_ratio=0.15):
    """Diviser le dataset: 70% train / 15% val / 15% test"""
    np.random.seed(CONFIG['SEED'])
    np.random.shuffle(npz_files)
    
    n = len(npz_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_files = npz_files[:n_train]
    val_files = npz_files[n_train:n_train+n_val]
    test_files = npz_files[n_train+n_val:]
    
    return train_files, val_files, test_files


# 
# POINT D'ENTRÉE PRINCIPAL
# 

def main():
    """Pipeline complet: chargement → split → entraînement → évaluation"""
    
    print("\n" + "="*70)
    print("   MAP WIDTH REGRESSOR - TÂCHE 2")
    print("="*70 + "\n")
    
    # Dossier de travail
    task_dir = Path(__file__).parent
    data_dir = task_dir / 'data'
    output_dir = task_dir / 'outputs'
    
    # Créer les répertoires
    output_dir.mkdir(exist_ok=True)
    
    # Chercher les fichiers .npz
    if not data_dir.exists():
        print(f"  Le dossier 'data' n'existe pas: {data_dir}")
        print("   Créez le dossier et placez vos fichiers .npz dedans")
        return
    
    npz_files = list(data_dir.glob('**/*.npz'))
    if len(npz_files) == 0:
        print(f" Aucun fichier .npz trouvé dans {data_dir}")
        return
    
    print(f" Trouvé {len(npz_files)} fichiers .npz\n")
    
    #  SPLIT DATASET 
    train_files, val_files, test_files = split_dataset(npz_files)
    print(f" Dataset split:")
    print(f"   Train: {len(train_files)} ({100*len(train_files)/len(npz_files):.1f}%)")
    print(f"   Val:   {len(val_files)} ({100*len(val_files)/len(npz_files):.1f}%)")
    print(f"   Test:  {len(test_files)} ({100*len(test_files)/len(npz_files):.1f}%)\n")
    
    #  CRÉER LES DATASETS 
    train_dataset = MapWidthDataset(train_files, target_size=224)
    val_dataset = MapWidthDataset(val_files, target_size=224)
    test_dataset = MapWidthDataset(test_files, target_size=224)
    
    # Statistiques des labels
    train_labels = np.array(train_dataset.labels)
    print(f" Statistiques des labels (Train):")
    print(f"   Min: {train_labels.min():.2f}m | Max: {train_labels.max():.2f}m")
    print(f"   Mean: {train_labels.mean():.2f}m | Std: {train_labels.std():.2f}m\n")
    
    #  DATALOADERS 
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=0
    )
    
    #  INITIALISER LE MODÈLE 
    model = MapWidthRegressor(num_channels=4, num_output=1)
    print(f" Modèle créé:")
    print(f"   Paramètres: {sum(p.numel() for p in model.parameters()):,}\n")
    
    #  ENTRAÎNER 
    model_save_path = output_dir / 'best_map_width_regressor.pth'
    trainer = RegressionTrainer(model, CONFIG, CONFIG['DEVICE'])
    trainer.train(train_loader, val_loader, str(model_save_path))
    
    #  CHARGER LE MEILLEUR MODÈLE 
    model.load_state_dict(torch.load(model_save_path, map_location=CONFIG['DEVICE']))
    
    #  ÉVALUATION SUR TEST SET 
    print(" Évaluation sur le test set...")
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(CONFIG['DEVICE'])
            outputs = model(images)
            outputs = torch.clamp(outputs, CONFIG['WIDTH_MIN'], CONFIG['WIDTH_MAX'])
            
            test_predictions.extend(outputs.cpu().numpy().squeeze())
            test_targets.extend(labels.numpy().squeeze())
    
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    
    # Calcul des métriques
    test_mae = mean_absolute_error(test_targets, test_predictions)
    test_mse = mean_squared_error(test_targets, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(test_targets, test_predictions)
    
    print("\n" + "="*70)
    print(" RÉSULTATS TEST")
    print("="*70)
    print(f"MAE:  {test_mae:.4f}m {' OBJECTIF ATTEINT' if test_mae < 1.0 else ' OBJECTIF NON ATTEINT'}")
    print(f"RMSE: {test_rmse:.4f}m")
    print(f"MSE:  {test_mse:.4f}")
    print(f"R²:   {test_r2:.4f}")
    print("="*70 + "\n")
    
    #  GÉNÉRER LES VISUALISATIONS 
    print(" Génération des visualisations...")
    
    # 1. Historique d'entraînement
    plot_training_history(
        trainer.history,
        output_dir / 'training_history_regression.png'
    )
    
    # 2. Scatter plot: prédit vs réel
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(test_targets, test_predictions, alpha=0.6, s=50)
    
    # Ligne y=x (prédictions parfaites)
    min_val = min(test_targets.min(), test_predictions.min())
    max_val = max(test_targets.max(), test_predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Réel (mètres)', fontsize=12)
    ax.set_ylabel('Prédit (mètres)', fontsize=12)
    ax.set_title('Prédictions vs Réalité (Test Set)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_predicted_vs_real.png', dpi=100, bbox_inches='tight')
    print(f" Scatter plot sauvegardé")
    plt.close()
    
    # 3. Distribution des erreurs
    errors = test_targets - test_predictions
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)
    
    # Histogramme
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(0, color='r', linestyle='--', linewidth=2, label='Erreur = 0')
    ax1.axvline(np.mean(errors), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.3f}')
    ax1.set_xlabel('Erreur = Réel - Prédit (mètres)', fontsize=11)
    ax1.set_ylabel('Fréquence', fontsize=11)
    ax1.set_title('Distribution des Erreurs', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Box plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.boxplot([errors], labels=['Erreurs'])
    ax2.axhline(0, color='r', linestyle='--', linewidth=2)
    ax2.set_ylabel('Erreur (mètres)', fontsize=11)
    ax2.set_title('Box Plot des Erreurs', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=100, bbox_inches='tight')
    print(f" Distribution des erreurs sauvegardée")
    plt.close()
    
    #  SAUVEGARDER LES RÉSULTATS JSON 
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'training': {
            'best_epoch': trainer.history['best_epoch'],
            'best_val_mae': float(trainer.history['best_val_mae']),
            'final_train_loss': float(trainer.history['train_loss'][-1]),
            'final_val_loss': float(trainer.history['val_loss'][-1]),
        },
        'test_metrics': {
            'mae': float(test_mae),
            'mse': float(test_mse),
            'rmse': float(test_rmse),
            'r2_score': float(test_r2),
        },
        'dataset_stats': {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'label_min': float(np.array(train_dataset.labels).min()),
            'label_max': float(np.array(train_dataset.labels).max()),
            'label_mean': float(np.array(train_dataset.labels).mean()),
            'label_std': float(np.array(train_dataset.labels).std()),
        },
    }
    
    results_path = output_dir / 'regression_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f" Résultats sauvegardés: {results_path}")
    print(f"\n Entraînement terminé avec succès!\n")


if __name__ == '__main__':
    main()
