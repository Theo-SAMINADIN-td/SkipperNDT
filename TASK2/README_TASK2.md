#  TÂCHE 2: RÉGRESSION - PRÉDICTION DE MAP WIDTH

##  Vue d'ensemble

Cette tâche consiste à prédire la **largeur effective de la zone d'influence magnétique (map width)** d'un tuyau enfoui, en mètres, à partir de données magnétiques 3D.

### Spécifications

| Élément | Description |
|---------|-------------|
| **Tâche** | Régression |
| **Sortie** | Valeur continue (5-80 mètres) |
| **Tolérance** | ±0.5 mètre |
| **Métrique clé** | MAE < 1.0m |
| **Entrée** | Fichiers .npz (4 canaux: Bx, By, Bz, Norm) |
| **Dimensions images** | 224×224 pixels (normalisées) |

---

##  Structure du projet

```
TASK2/
 map_width_regressor.py          #  Script d'entraînement principal
 predict_map_width.py            #  Prédiction sur un fichier unique
 batch_predict_width.py          #  Prédictions batch (dossier)
 evaluate_regression.py          #  Évaluation & métriques
 analyze_width_dataset.py        #  Analyse exploratoire du dataset
 data/                           #  Dossier pour les fichiers .npz
    *.npz                       # Fichiers magnétiques + labels
 outputs/                        #  Résultats & modèles
     best_map_width_regressor.pth    # Modèle entraîné
     training_history_regression.png # Graphe MAE/Loss
     scatter_predicted_vs_real.png   # Prédictions vs Réalité
     error_distribution.png          # Distribution des erreurs
     regression_results.json         # Métriques complètes
```

---

##  Démarrage rapide

### 1⃣ Installation des dépendances

```bash
pip install -r requirements.txt
```

Dépendances principales:
- PyTorch (CPU/GPU)
- NumPy, SciPy
- Matplotlib
- scikit-learn
- tqdm

### 2⃣ Préparation des données

1. **Créer le dossier `data/`:**
   ```bash
   mkdir data
   ```

2. **Placer vos fichiers .npz:**
   - Format attendu: `data/sample_001.npz`, `data/sample_002.npz`, etc.
   - Structure de chaque fichier .npz:
     ```python
     {
         'data': np.ndarray(H, W, 4),  # Image magnétique 4-canaux
         'width': float,               # Label: largeur en mètres
     }
     ```

### 3⃣ Analyse exploratoire du dataset (optionnel)

```bash
python analyze_width_dataset.py
```

 Génère:
- Distribution des largeurs
- Détection des outliers
- Statistiques du dataset
- Fichiers: `width_distribution.png`, `dataset_analysis.json`

### 4⃣ Entraînement du modèle

```bash
python map_width_regressor.py
```

**Console attendue:**
```
 Device utilisé: cuda
 Trouvé 500 fichiers .npz
 450 échantillons valides chargés

 Dataset split:
   Train: 315 (70.0%)
   Val:   67 (15.0%)
   Test:  68 (15.0%)

 Démarrage de l'entraînement...
Epoch  45/100 | Train Loss: 2.34 | Val MAE: 0.87m | Val MSE: 1.23
   Meilleur modèle sauvegardé (MAE: 0.87m)

 Entraînement terminé!
  Meilleur MAE de validation: 0.87m (Epoch 45)

 RÉSULTATS TEST

MAE:  0.91m  OBJECTIF ATTEINT
RMSE: 1.02m
MSE:  1.04
R²:   0.94

```

**Fichiers générés:**
-  `best_map_width_regressor.pth` - Modèle entraîné
-  `training_history_regression.png` - Courbes d'entraînement
-  `scatter_predicted_vs_real.png` - Analyse des prédictions
-  `error_distribution.png` - Distribution des erreurs
-  `regression_results.json` - Métriques JSON

### 5⃣ Évaluation du modèle

```bash
python evaluate_regression.py
```

Génère un rapport complet et des visualisations avancées.

### 6⃣ Prédictions

#### Single file prediction:
```bash
python predict_map_width.py data/sample.npz
```

**Output:**
```
 Dimensions de l'image: 256×256 pixels (4 canaux)

   RÉSULTAT

Largeur prédite: 23.45 mètres
Intervalle de confiance: ±0.5m [22.95m - 23.95m]
Score de confiance: 80.0%

```

#### Batch predictions:
```bash
python batch_predict_width.py data/ --output predictions.csv
```

**Output CSV:**
```
filename,relative_path,predicted_width_m,confidence,image_dimensions
sample_001.npz,sample_001.npz,23.450,0.800,256x256
sample_002.npz,sample_002.npz,15.230,0.800,300x280
sample_003.npz,sample_003.npz,45.670,0.800,512x512
...
```

---

##  Architecture du modèle

### Backbone Convolutif (4 blocs)

```
Input: (B, 4, 224, 224)
  ↓
Block 1: Conv(4→64) + BN + ReLU + MaxPool
         Output: (B, 64, 112, 112)
  ↓
Block 2: Conv(64→128) + BN + ReLU + MaxPool
         Output: (B, 128, 56, 56)
  ↓
Block 3: Conv(128→256) + BN + ReLU + MaxPool
         Output: (B, 256, 28, 28)
  ↓
Block 4: Conv(256→512) + BN + ReLU + MaxPool
         Output: (B, 512, 14, 14)
  ↓
AdaptiveAvgPool2d(1, 1)
         Output: (B, 512, 1, 1)
  ↓
FC Head: 512 → 256 → 128 → 1 (Linear, pas d'activation)
         Output: (B, 1)
```

**Total de paramètres:** ~2.5M

---

##  Hyperparamètres

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `BATCH_SIZE` | 32 | Taille de batch (réduire à 16 si GPU OOM) |
| `LEARNING_RATE` | 1e-4 | Taux d'apprentissage Adam |
| `NUM_EPOCHS` | 100 | Nombre maximum d'epochs |
| `WEIGHT_DECAY` | 1e-4 | Régularisation L2 |
| `PATIENCE` | 15 | Early stopping après 15 epochs sans amélioration |
| `TARGET_SIZE` | 224 | Redimensionnement images → 224×224 |

### Scheduler & Optimizer
- **Optimizer:** Adam
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=7)
- **Loss Train:** MSELoss
- **Loss Eval:** MAELoss (L1Loss)

---

##  Preprocessing pipeline

Pour chaque échantillon:

1. **Chargement:** Lire `data[...]` depuis le fichier .npz
2. **Nettoyage:** Remplacer NaN et Inf par 0
3. **Redimensionnement:** Zoom vers 224×224 (nearest neighbor)
4. **Normalisation:** Mean=0, Std=1 par canal
5. **Transposition:** (H, W, 4) → (4, H, W)
6. **Conversion:** np.ndarray → torch.Tensor (float32)

---

##  Métriques d'évaluation

### Métriques principales

| Métrique | Formule | Interprétation |
|----------|---------|-----------------|
| **MAE** | $\frac{1}{n}\sum\|y - \hat{y}\|$ | Erreur absolue moyenne (principal) |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}$ | Racine de l'erreur quadratique |
| **MSE** | $\frac{1}{n}\sum(y - \hat{y})^2$ | Erreur quadratique moyenne |
| **R²** | $1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$ | Coefficient de détermination |

### Critères de succès

-  **MAE < 1.0m** — Objectif principal
-  **R² > 0.90** — Bon ajustement
-  **75%+ samples dans ±0.5m** — Haute précision

---

##  Bonnes pratiques & Troubleshooting

### GPU Memory Issues
```bash
# Réduire la batch size
BATCH_SIZE = 16  # au lieu de 32
```

### DataLoader Errors
```python
# Dans les DataLoaders, utiliser:
num_workers=0  # Évite les problèmes de multiprocessing
```

### Modèle ne converge pas
1.  Vérifier la distribution des labels (pas de biais)
2.  Augmenter le nombre d'epochs
3.  Réduire le learning rate (1e-5)
4.  Augmenter le PATIENCE (20-30)

### Prédictions hors plage [5, 80]
```python
# Le modèle clamp automatiquement:
output = torch.clamp(output, 5.0, 80.0)
```

---

##  Format des fichiers de sortie

### `regression_results.json`
```json
{
  "timestamp": "2026-02-23T10:45:30.123456",
  "config": {
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.0001,
    "NUM_EPOCHS": 100,
    ...
  },
  "training": {
    "best_epoch": 45,
    "best_val_mae": 0.8712,
    "final_train_loss": 2.3401,
    "final_val_loss": 1.0234
  },
  "test_metrics": {
    "mae": 0.9123,
    "mse": 1.0456,
    "rmse": 1.0225,
    "r2_score": 0.9401
  },
  "dataset_stats": {
    "train_samples": 315,
    "val_samples": 67,
    "test_samples": 68,
    ...
  }
}
```

---

##  Transfert depuis la Tâche 1

Les blocs convolutifs sont **réutilisables** depuis la Tâche 1 (classification):
-  Même architecture backbone
-  Même preprocessing (normalization)
-  Seule la tête change: classification → régression

### Utilisation d'un modèle pré-entraîné de Tâche 1:
```python
# Charger le backbone de Tâche 1
task1_model = torch.load('TASK1/best_pipeline_classifier.pth')

# Adapter la tête pour la régression
model = MapWidthRegressor(num_channels=4, num_output=1)

# Copier les poids du backbone
for (name1, param1), (name2, param2) in zip(
    task1_model.named_parameters(),
    model.named_parameters()
):
    if 'fc_head' not in name1:  # Skip classification head
        param2.data = param1.data
```

---

##  Support & Questions

Pour des questions spécifiques:
1. Vérifier les fichiers `.npz` input (structure, labels valides)
2. Consulter `regression_results.json` pour les métriques détaillées
3. Examiner les graphes de visualisation (`*.png`)
4. Vérifier les logs de console pour les erreurs

---

##  Checklist avant déploiement

- [ ] MAE test < 1.0m
- [ ] R² test > 0.90
- [ ] Pas d'erreurs dans `regression_results.json`
- [ ] Fichier `best_map_width_regressor.pth` existe et est valide
- [ ] Prédictions batch testées sur dossier de test
- [ ] Visualisations générées et vérifiées

---

**Créé pour le Projet SkipperNDT - Tâche 2**
 Régression |  Deep Learning |  Géophysique

