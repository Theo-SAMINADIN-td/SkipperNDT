# Guide d'Installation - Pipeline Presence Detector

## Installation rapide

### Option 1: Installation automatique (recommandé)

```bash
# Aller dans le dossier du projet
cd /home/tsaminadin/Documents/HETIC/SkipperNDT

# Installer les dépendances
pip install -r requirements.txt

# Vérifier l'installation
python test_system.py
```

### Option 2: Installation manuelle

```bash
pip install torch torchvision numpy scipy matplotlib scikit-learn tqdm
```

## Installation avec environnement virtuel (recommandé pour la production)

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Vérifier l'installation
python test_system.py
```

## Installation de PyTorch avec GPU (optionnel mais recommandé)

Si vous avez un GPU NVIDIA:

```bash
# Pour CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Pour CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Vérifier CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Vérification de l'installation

```bash
# Test complet du système
python test_system.py

# Analyse du dataset
python analyze_dataset.py
```

## Commandes utiles

### Entraînement

```bash
# Entraînement complet
python pipeline_presence_detector.py

# Avec GPU spécifique
CUDA_VISIBLE_DEVICES=0 python pipeline_presence_detector.py
```

### Prédiction

```bash
# Prédiction sur un fichier
python predict_pipeline_presence.py --input fichier.npz

# Prédictions en batch
python batch_predict.py --input_dir Training_database_float16/

# Visualisation des prédictions
python visualize_predictions.py --samples 9
```

## Résolution des problèmes

### Erreur: "No module named 'torch'"
```bash
pip install torch torchvision
```

### Erreur: "CUDA out of memory"
Modifier `BATCH_SIZE` dans `pipeline_presence_detector.py`:
```python
BATCH_SIZE = 8  # Au lieu de 16
```

### Erreur: "RuntimeError: DataLoader worker"
Modifier `num_workers` dans le code:
```python
num_workers=0  # Au lieu de 4
```

### Erreur: "FileNotFoundError" pour les données
Vérifier le chemin dans `pipeline_presence_detector.py`:
```python
DATA_DIR = '/votre/chemin/vers/Training_database_float16'
```

## Configuration matérielle recommandée

### Minimum
- CPU: 4 cores
- RAM: 8 GB
- Stockage: 10 GB
- Temps d'entraînement: ~4-6 heures

### Recommandé
- CPU: 8+ cores
- RAM: 16+ GB
- GPU: NVIDIA avec 6+ GB VRAM
- Stockage: 20 GB
- Temps d'entraînement: ~30-60 minutes

## Structure des fichiers après installation

```
SkipperNDT/
├── Training_database_float16/       # Données (2833 fichiers)
├── pipeline_presence_detector.py    # Script principal
├── predict_pipeline_presence.py     # Prédiction
├── batch_predict.py                 # Batch prédiction
├── visualize_predictions.py         # Visualisation
├── analyze_dataset.py               # Analyse
├── test_system.py                   # Tests
├── requirements.txt                 # Dépendances
├── README_TASK1.md                  # Documentation
├── INSTALLATION.md                  # Ce fichier
└── quick_start.sh                   # Script automatique
```

## Prochaines étapes

1. **Installer**: `pip install -r requirements.txt`
2. **Tester**: `python test_system.py`
3. **Analyser**: `python analyze_dataset.py`
4. **Entraîner**: `python pipeline_presence_detector.py`
5. **Prédire**: `python predict_pipeline_presence.py --input fichier.npz`

## Support

Pour toute question, consultez:
- `README_TASK1.md` pour la documentation complète
- Les commentaires dans les fichiers Python
- Les messages d'erreur détaillés
