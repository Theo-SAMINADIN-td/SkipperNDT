"""
Pipeline Presence Detector - Binary Classification
Détecte la présence ou l'absence de conduites dans des images magnétiques multicanales
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from common import BaseNpzDataset, build_efficientnet_v2s_backbone


class PipelineDataset(BaseNpzDataset):
    """Dataset pour charger les images .npz avec labels automatiques."""

    def __init__(self, file_paths, labels, target_size=(224, 224)):
        super().__init__(file_paths, target_size)
        self.labels = labels

    def _make_target(self, idx):
        return torch.tensor(self.labels[idx], dtype=torch.float32)


class PipelinePresenceClassifier(nn.Module):
    """
    Classifieur binaire de présence de conduites.
    Architecture EfficientNet-V2-S avec adaptation 4 canaux.
    """

    def __init__(self, num_channels=4, pretrained=False):
        super().__init__()
        backbone, in_features = build_efficientnet_v2s_backbone(num_channels, pretrained)
        self.features = backbone

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_data_with_labels(data_dir):
    """
    Charge les fichiers .npz et extrait les labels à partir des noms de fichiers
    
    Label 0: no_pipe (absence de conduite)
    Label 1: perfect, missed, ou tout autre (présence de conduite)
    """
    all_files = glob.glob(os.path.join(data_dir, '*.npz'))
    
    file_paths = []
    labels = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        # Extraire le label du nom de fichier
        if 'no_pipe' in filename:
            label = 0  # Absence de conduite
        else:
            label = 1  # Présence de conduite
        
        file_paths.append(file_path)
        labels.append(label)
    
    return file_paths, labels


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entraîne le modèle pour une époque"""
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Prédictions
        predictions = (outputs > 0.5).float()
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return epoch_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    """Valide le modèle"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            predictions = (outputs > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(outputs.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    
    return epoch_loss, accuracy, recall, f1, all_predictions, all_labels


def plot_training_history(history, save_path='training_history.png'):
    """Affiche l'historique d'entraînement"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy')
    axes[0, 1].axhline(y=0.92, color='r', linestyle='--', label='Target 92%')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Recall
    axes[1, 0].plot(history['val_recall'], label='Val Recall')
    axes[1, 0].axhline(y=0.95, color='r', linestyle='--', label='Target 95%')
    axes[1, 0].set_title('Recall (Validation)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1-Score
    axes[1, 1].plot(history['val_f1'], label='Val F1-Score')
    axes[1, 1].set_title('F1-Score (Validation)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")
    plt.close()


def main():
    # Configuration
    DATA_DIR = r'Training_database_float16'  # Chemin vers les données
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    TARGET_SIZE = (224, 224)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Charger les données
    print("\n1. Loading data...")
    file_paths, labels = load_data_with_labels(DATA_DIR)
    print(f"Total samples: {len(file_paths)}")
    print(f"Class 0 (no pipe): {labels.count(0)}")
    print(f"Class 1 (with pipe): {labels.count(1)}")
    
    # Split train/validation/test
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"\nTrain samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")
    
    # Créer les datasets
    print("\n2. Creating datasets...")
    train_dataset = PipelineDataset(train_files, train_labels, target_size=TARGET_SIZE)
    val_dataset = PipelineDataset(val_files, val_labels, target_size=TARGET_SIZE)
    test_dataset = PipelineDataset(test_files, test_labels, target_size=TARGET_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Créer le modèle
    print("\n3. Creating model...")
    model = PipelinePresenceClassifier(num_channels=4).to(device)
    
    # Loss avec pondération pour gérer le déséquilibre des classes
    num_no_pipe = train_labels.count(0)
    num_with_pipe = train_labels.count(1)
    weight = num_no_pipe / num_with_pipe if num_with_pipe > 0 else 1.0
    
    # Utiliser BCEWithLogitsLoss avec pondération pour privilégier le recall
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Entraînement
    print("\n4. Training model...")
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_recall': [], 'val_f1': []
    }
    
    best_recall = 0.0
    best_epoch = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Entraînement
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc, val_recall, val_f1, val_preds, val_true = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Mettre à jour le scheduler
        scheduler.step(val_loss)
        
        # Sauvegarder l'historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        # Afficher les résultats
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")
        
        # Sauvegarder le meilleur modèle (basé sur le recall)
        if val_recall > best_recall:
            best_recall = val_recall
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_recall': val_recall,
                'val_f1': val_f1,
            }, 'best_pipeline_classifier.pth')
            print(f"✓ Best model saved (Recall: {val_recall:.4f})")
    
    # Sauvegarder l'historique
    plot_training_history(history)
    
    # Évaluation finale sur le test set
    print("\n5. Final evaluation on test set...")
    model.load_state_dict(torch.load('best_pipeline_classifier.pth')['model_state_dict'])
    
    test_loss, test_acc, test_recall, test_f1, test_preds, test_true = validate_epoch(
        model, test_loader, criterion, device
    )
    
    print(f"\n{'='*50}")
    print("FINAL TEST RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {test_acc:.4f} (Target: > 0.92)")
    print(f"Recall: {test_recall:.4f} (Target: > 0.95)")
    print(f"F1-Score: {test_f1:.4f}")
    print(f"Best epoch: {best_epoch}")
    
    # Matrice de confusion
    cm = confusion_matrix(test_true, test_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Rapport de classification
    print("\nClassification Report:")
    print(classification_report(test_true, test_preds, target_names=['No Pipe', 'With Pipe']))
    
    # Sauvegarder les résultats
    results = {
        'test_accuracy': float(test_acc),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'best_epoch': best_epoch,
        'confusion_matrix': cm.tolist(),
        'target_accuracy': 0.92,
        'target_recall': 0.95,
        'objectives_met': {
            'accuracy': test_acc > 0.92,
            'recall': test_recall > 0.95
        }
    }
    
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n✓ Training complete! Results saved.")
    print("  - Model: best_pipeline_classifier.pth")
    print("  - History plot: training_history.png")
    print("  - Results: test_results.json")


if __name__ == "__main__":
    main()
