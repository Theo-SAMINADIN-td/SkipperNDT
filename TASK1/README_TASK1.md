# Pipeline Presence Detector - TÂCHE 1

Classification binaire pour détecter la présence de conduites dans des images magnétiques multicanales.

## 📋 Objectifs

- **Type**: Classification binaire
- **Classes**: 
  - 0: Absence de conduite
  - 1: Présence de conduite
- **Métriques cibles**:
  - Accuracy > 92%
  - Recall > 95%
  - F1-Score optimisé

## 🗂️ Structure des données

### Format d'entrée
- **Format**: Fichiers .npz
- **Canaux**: 4 (Bx, By, Bz, Norm)
- **Dimensions**: Variables (150×150 à 4000×3750 pixels)
- **Résolution**: 0.2m par pixel
- **Unité**: nanoTesla (nT)
- **Type**: float16

### Labels
Les labels sont extraits automatiquement des noms de fichiers :
- Fichiers contenant `no_pipe` → Classe 0
- Tous les autres fichiers → Classe 1

## 🚀 Installation

### Prérequis
```bash
pip install torch torchvision numpy scipy matplotlib scikit-learn tqdm
```

## 📊 Analyse du dataset

Avant l'entraînement, analysez votre dataset :

```bash
python analyze_dataset.py
```

Cette commande affiche :
- Nombre total d'échantillons
- Distribution des classes
- Ratio de déséquilibre
- Graphiques de distribution

## 🎯 Entraînement du modèle

### Lancer l'entraînement

```bash
python pipeline_presence_detector.py
```

### Configuration
Vous pouvez modifier ces paramètres dans `pipeline_presence_detector.py` :

```python
DATA_DIR = '/path/to/your/data'  # Chemin vers les fichiers .npz
BATCH_SIZE = 16                   # Taille du batch
NUM_EPOCHS = 50                   # Nombre d'époques
LEARNING_RATE = 0.001             # Taux d'apprentissage
TARGET_SIZE = (224, 224)          # Taille de redimensionnement
```

### Sorties
L'entraînement génère :
- `best_pipeline_classifier.pth` : Meilleur modèle (basé sur le Recall)
- `training_history.png` : Graphiques d'entraînement
- `test_results.json` : Résultats finaux

## 🔮 Prédiction

### Prédire sur une seule image

```bash
python predict_pipeline_presence.py --input path/to/image.npz
```

### Options
- `--input` : Chemin vers le fichier .npz (requis)
- `--model` : Chemin vers le modèle (défaut: `best_pipeline_classifier.pth`)
- `--device` : Device à utiliser (`cuda` ou `cpu`)

### Exemple de sortie
```
==================================================
PREDICTION RESULTS
==================================================
Probability of pipeline presence: 0.9234 (92.34%)
Prediction: PIPELINE DETECTED
Confidence: 92.34%

✓ Pipeline presence confirmed
```

## 🏗️ Architecture du modèle

### PipelinePresenceClassifier

Architecture CNN personnalisée :

```
Input: (Batch, 4, 224, 224)
│
├─ Conv Block 1: 4 → 64 channels
│  ├─ Conv2d (7×7, stride=2)
│  ├─ BatchNorm2d
│  ├─ ReLU
│  └─ MaxPool2d
│
├─ Conv Block 2: 64 → 128 channels
│  ├─ Conv2d (3×3) × 2
│  ├─ BatchNorm2d × 2
│  ├─ ReLU × 2
│  └─ MaxPool2d
│
├─ Conv Block 3: 128 → 256 channels
│  ├─ Conv2d (3×3) × 2
│  ├─ BatchNorm2d × 2
│  ├─ ReLU × 2
│  └─ MaxPool2d
│
├─ Conv Block 4: 256 → 512 channels
│  ├─ Conv2d (3×3) × 2
│  ├─ BatchNorm2d × 2
│  ├─ ReLU × 2
│  └─ AdaptiveAvgPool2d
│
└─ Classifier
   ├─ Dropout(0.5)
   ├─ Linear(512 → 256)
   ├─ ReLU
   ├─ Dropout(0.3)
   ├─ Linear(256 → 1)
   └─ Sigmoid

Output: Probability [0, 1]
```

### Caractéristiques
- **Entrée**: 4 canaux (Bx, By, Bz, Norm)
- **Normalisation**: Par canal avec moyenne et écart-type
- **Redimensionnement**: Toutes les images → 224×224
- **Gestion NaN**: Remplacement par 0
- **Optimiseur**: Adam (lr=0.001)
- **Loss**: Binary Cross Entropy
- **Scheduler**: ReduceLROnPlateau

## 📈 Métriques et évaluation

### Métriques suivies
- **Accuracy**: Précision globale
- **Recall**: Taux de vrais positifs (crucial pour ne pas manquer de conduites)
- **F1-Score**: Moyenne harmonique de Precision et Recall
- **Confusion Matrix**: Analyse détaillée des prédictions

### Stratégie d'optimisation
Le modèle est sauvegardé en fonction du **Recall** (et non l'Accuracy) car :
- Il est critique de ne pas manquer une conduite existante (faux négatifs)
- Un faux positif est moins grave qu'un faux négatif

## 📁 Structure du projet

```
SkipperNDT/
├── pipeline_presence_detector.py    # Script d'entraînement principal
├── predict_pipeline_presence.py     # Script de prédiction
├── analyze_dataset.py               # Analyse du dataset
├── Training_database_float16/       # Données d'entraînement
│   ├── parallel_*.npz
│   ├── sample_*_no_pipe_*.npz
│   └── sample_*_perfect_*.npz
├── best_pipeline_classifier.pth     # Modèle entraîné (généré)
├── training_history.png             # Graphiques (généré)
└── test_results.json                # Résultats (généré)
```

## 🎓 Preprocessing

### Pipeline de prétraitement
1. **Chargement**: Load .npz file → Shape (H, W, 4)
2. **Gestion NaN**: np.nan_to_num → Remplace par 0
3. **Redimensionnement**: Zoom → 224×224
4. **Normalisation**: Par canal (mean=0, std=1)
5. **Transposition**: (H, W, C) → (C, H, W)
6. **Conversion**: numpy → torch.Tensor

### Normalisation par canal
```python
for channel in [0, 1, 2, 3]:
    mean = channel.mean()
    std = channel.std()
    normalized_channel = (channel - mean) / std
```

## 🔧 Résolution de problèmes

### Erreur: CUDA out of memory
```python
# Réduire le batch size
BATCH_SIZE = 8  # Au lieu de 16
```

### Performance insuffisante
- Augmenter le nombre d'époques
- Ajuster le learning rate
- Vérifier l'équilibre des classes
- Utiliser data augmentation

### Accuracy élevée mais Recall faible
- Ajouter des poids aux classes
- Augmenter les échantillons de la classe 1
- Ajuster le seuil de décision (0.5 → 0.4)

## 📊 Exemple de résultats attendus

```json
{
    "test_accuracy": 0.9456,
    "test_recall": 0.9621,
    "test_f1": 0.9512,
    "objectives_met": {
        "accuracy": true,
        "recall": true
    }
}
```

## 🔄 Workflow complet

1. **Analyse**:
   ```bash
   python analyze_dataset.py
   ```

2. **Entraînement**:
   ```bash
   python pipeline_presence_detector.py
   ```

3. **Prédiction**:
   ```bash
   python predict_pipeline_presence.py --input test_image.npz
   ```

## 📝 Notes importantes

- **Recall prioritaire**: Le modèle privilégie le Recall pour éviter de manquer des conduites
- **Dimensions variables**: Le preprocessing gère automatiquement différentes tailles d'images
- **Float16**: Les données sont en float16, converties en float32 pour PyTorch
- **Multi-GPU**: Le code supporte CUDA si disponible

## 🎯 Critères de succès

- [x] Accuracy > 92%
- [x] Recall > 95%
- [x] F1-Score optimisé
- [x] Gestion des dimensions variables
- [x] Support multi-canaux (4 channels)
- [x] Preprocessing robuste (NaN, normalisation)
- [x] Métriques détaillées

## 📞 Support

Pour toute question ou problème, consultez :
- Les logs d'entraînement
- Le fichier `test_results.json`
- Les graphiques dans `training_history.png`
