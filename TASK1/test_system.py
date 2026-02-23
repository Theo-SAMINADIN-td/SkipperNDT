"""
Script de test rapide pour vérifier que tout fonctionne
avant de lancer l'entraînement complet
"""

import torch
import numpy as np
from pipeline_presence_detector import (
    PipelineDataset, 
    PipelinePresenceClassifier,
    load_data_with_labels
)
from torch.utils.data import DataLoader


def test_data_loading():
    """Test du chargement des données"""
    print("1. Test du chargement des données...")
    
    try:
        data_dir = '/home/tsaminadin/Documents/HETIC/SkipperNDT/Training_database_float16'
        file_paths, labels = load_data_with_labels(data_dir)
        
        print(f"   ✓ {len(file_paths)} fichiers chargés")
        print(f"   ✓ Classe 0: {labels.count(0)}")
        print(f"   ✓ Classe 1: {labels.count(1)}")
        
        return file_paths, labels
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        return None, None


def test_dataset_creation(file_paths, labels):
    """Test de la création du dataset"""
    print("\n2. Test de la création du dataset...")
    
    try:
        # Prendre seulement 10 échantillons pour le test
        test_files = file_paths[:10]
        test_labels = labels[:10]
        
        dataset = PipelineDataset(test_files, test_labels, target_size=(224, 224))
        
        print(f"   ✓ Dataset créé avec {len(dataset)} échantillons")
        
        # Tester le chargement d'un échantillon
        image, label = dataset[0]
        print(f"   ✓ Image shape: {image.shape}")
        print(f"   ✓ Image dtype: {image.dtype}")
        print(f"   ✓ Label: {label.item()}")
        
        # Vérifier qu'il n'y a pas de NaN
        has_nan = torch.isnan(image).any()
        if has_nan:
            print(f"   ⚠️  Warning: NaN detected in image")
        else:
            print(f"   ✓ Pas de NaN dans les données")
        
        return dataset
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_dataloader(dataset):
    """Test du DataLoader"""
    print("\n3. Test du DataLoader...")
    
    try:
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        
        # Charger un batch
        images, labels = next(iter(dataloader))
        
        print(f"   ✓ Batch images shape: {images.shape}")
        print(f"   ✓ Batch labels shape: {labels.shape}")
        print(f"   ✓ Images range: [{images.min():.2f}, {images.max():.2f}]")
        
        return True
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test de la création du modèle"""
    print("\n4. Test de la création du modèle...")
    
    try:
        model = PipelinePresenceClassifier(num_channels=4)
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   ✓ Modèle créé")
        print(f"   ✓ Paramètres totaux: {total_params:,}")
        print(f"   ✓ Paramètres entraînables: {trainable_params:,}")
        
        return model
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        return None


def test_forward_pass(model):
    """Test du forward pass"""
    print("\n5. Test du forward pass...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Créer un batch de test
        dummy_input = torch.randn(2, 4, 224, 224).to(device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"   ✓ Output values: {output.flatten().tolist()}")
        
        # Vérifier que les sorties sont des probabilités valides
        if torch.all((output >= 0) & (output <= 1)):
            print(f"   ✓ Sorties dans [0, 1] (probabilités valides)")
        else:
            print(f"   ⚠️  Warning: Sorties hors de [0, 1]")
        
        return True
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(model, dataset):
    """Test d'une étape d'entraînement"""
    print("\n6. Test d'une étape d'entraînement...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()
        
        # Créer un mini dataloader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        
        # Définir l'optimiseur et la loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()
        
        # Faire une étape d'entraînement
        images, labels = next(iter(dataloader))
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"   ✓ Forward pass réussi")
        print(f"   ✓ Loss calculée: {loss.item():.4f}")
        print(f"   ✓ Backward pass réussi")
        print(f"   ✓ Paramètres mis à jour")
        
        return True
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║         TEST RAPIDE DU SYSTÈME                             ║")
    print("╚════════════════════════════════════════════════════════════╝\n")
    
    # Vérifier CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA disponible: {torch.cuda.get_device_name(0)}")
    else:
        print(f"⚠️  CUDA non disponible, utilisation du CPU")
    
    print()
    
    # Tests
    file_paths, labels = test_data_loading()
    if file_paths is None:
        print("\n✗ Test échoué lors du chargement des données")
        return
    
    dataset = test_dataset_creation(file_paths, labels)
    if dataset is None:
        print("\n✗ Test échoué lors de la création du dataset")
        return
    
    if not test_dataloader(dataset):
        print("\n✗ Test échoué lors du test du DataLoader")
        return
    
    model = test_model_creation()
    if model is None:
        print("\n✗ Test échoué lors de la création du modèle")
        return
    
    if not test_forward_pass(model):
        print("\n✗ Test échoué lors du forward pass")
        return
    
    if not test_training_step(model, dataset):
        print("\n✗ Test échoué lors de l'étape d'entraînement")
        return
    
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║  ✓ TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS !               ║")
    print("║                                                            ║")
    print("║  Vous pouvez maintenant lancer l'entraînement complet:    ║")
    print("║  python pipeline_presence_detector.py                     ║")
    print("╚════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
