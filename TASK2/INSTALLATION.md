#  INSTALLATION - TÂCHE 2

Guide complet d'installation et de configuration pour la Tâche 2.

---

##  Système requis

| Composant | Requis | Recommandé |
|-----------|--------|-------------|
| OS | Windows/Linux/MacOS | Windows 10+ / Ubuntu 20.04+ |
| Python | 3.9+ | 3.10, 3.11 |
| RAM | 8 GB | 16+ GB |
| GPU | Optionnel | NVIDIA (CUDA 11.8+) |
| Disque | 5 GB | 10+ GB |

---

##  Étape 1: Installer Python

### Windows
```bash
# Télécharger Python 3.11 depuis https://www.python.org/downloads/
# Installer en cochant "Add Python to PATH"

# Vérifier l'installation:
python --version
pip --version
```

### Linux / WSL
```bash
sudo apt-get update
sudo apt-get install python3.11 python3-pip

python3.11 --version
pip --version
```

---

##  Étape 2: Créer l'environnement de travail

### 2.1 Créer un dossier de travail
```bash
cd "C:\Users\HP 840\Documents\MD4 HETIC\SKP\Skipper_NDT\SkipperNDT\TASK2"
```

### 2.2 Créer un environnement virtuel Python (RECOMMANDÉ)

#### Windows
```bash
python -m venv venv
venv\Scripts\activate

# Vérifier:
(venv) C:\...> where python
```

#### Linux / WSL
```bash
python3.11 -m venv venv
source venv/bin/activate

# Vérifier:
(venv) $ which python
```

---

##  Étape 3: Installer les dépendances

### 3.1 Mettre à jour pip (important!)
```bash
python -m pip install --upgrade pip setuptools wheel
```

### 3.2 Installer les packages
```bash
pip install -r requirements.txt
```

**Durée estimée:** 5-15 minutes (dépend de la connexion et du GPU)

### 3.3 Vérifier l'installation
```bash
python -c "import torch; print(torch.__version__)"
python -c "import numpy; print(numpy.__version__)"
python -c "import matplotlib; print(matplotlib.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

---

##  Étape 4: Configuration GPU (optionnel mais recommandé)

### 4.1 Vérifier CUDA (Windows)
```bash
# Si NVIDIA GPU disponible:
nvidia-smi

# Output attendu:
# NVIDIA-SMI 535.00    Driver Version: 535.00    CUDA Version: 12.2
```

### 4.2 Installer CUDA PyTorch (si GPU disponible)
```bash
# CPU (par défaut, déjà installé):
# PyTorch fonctionne avec CPU, performance réduite

# GPU NVIDIA (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU NVIDIA (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Vérifier:
python -c "import torch; print('GPU disponible:', torch.cuda.is_available())"
```

---

##  Étape 5: Préparer les données

### 5.1 Créer la structure de dossier
```bash
# Dans le dossier TASK2/:
mkdir data
mkdir outputs
```

### 5.2 Placer les fichiers .npz
```
TASK2/data/
 sample_001.npz
 sample_002.npz
 ...
 sample_500.npz
```

**Vérification:**
```bash
ls data/*.npz | wc -l  # Doit afficher > 0
```

### 5.3 Valider le format des fichiers
```bash
# Exécuter le test (Python):
python -c "
import numpy as np
data = np.load('data/sample_001.npz', allow_pickle=True)
print('Clés:', list(data.files))
print('Shape magnétique:', data['data'].shape)  # Doit être (H, W, 4)
print('Label width:', data['width'])  # Doit être un float
"
```

---

##  Étape 6: Test d'installation

### 6.1 Test simple
```bash
python -c "
import torch
import numpy as np
from map_width_regressor import MapWidthRegressor, CONFIG

# Créer le modèle
model = MapWidthRegressor()
print(' Modèle créé avec succès')

# Test forward pass
x = torch.randn(1, 4, 224, 224)
y = model(x)
print(f' Forward pass OK: input {x.shape} → output {y.shape}')

# Vérifier device
print(f' Device: {CONFIG[\"DEVICE\"]}')
"
```

### 6.2 Test complet (analyse dataset)
```bash
python analyze_width_dataset.py
```

**Expected output:**
```
 Analyse de 500 fichiers...
 450 fichiers valides, 50 invalides

 ANALYSE DU DATASET

 STATISTIQUES PRINCIPALES:
   Échantillons valides: 450
   Min:   5.123m
   Max:   78.456m
   Mean:  35.234m
   ...
```

---

##  Étape 7: Lancer l'entraînement

```bash
# Entraînement complet:
python map_width_regressor.py

# Output attendu:
#  Démarrage de l'entraînement...
# Epoch  1/100 | Train Loss: 125.23 | Val MAE: 15.34m | Val MSE: 234.12
# Epoch  2/100 | Train Loss: 95.12  | Val MAE: 12.45m | Val MSE: 155.23
# ...
# Epoch 45/100 | Train Loss: 2.34   | Val MAE: 0.87m  | Val MSE: 1.23
#    Meilleur modèle sauvegardé (MAE: 0.87m)
```

---

##  Dépannage courant

###  "ModuleNotFoundError: No module named 'torch'"
```bash
# Solution: Réinstaller PyTorch
pip install --upgrade torch
# OU vérifier l'environnement virtuel (source venv/bin/activate)
```

###  "CUDA out of memory"
```python
# Dans map_width_regressor.py, modifier:
CONFIG['BATCH_SIZE'] = 16  # au lieu de 32
```

###  "No such file or directory: 'data/sample_001.npz'"
```bash
# Solution: Vérifier que les fichiers sont bien dans data/
ls -la data/  # Afficher les fichiers

# Créer le dossier s'il n'existe pas:
mkdir -p data
```

###  "JSON parse error in evaluate_regression.py"
```bash
# Vérifier que l'entraînement s'est bien terminé:
ls -la outputs/  # Doit contenir best_map_width_regressor.pth
```

###  Erreur d'import scipy.ndimage
```bash
pip install --upgrade scipy
```

---

##  Vérifier les résultats

Après l'entraînement, vérifier les fichiers générés:

```bash
# Windows:
dir outputs\

# Linux:
ls -la outputs/
```

**Fichiers attendus:**
-  `best_map_width_regressor.pth` (50-100 MB)
-  `training_history_regression.png`
-  `scatter_predicted_vs_real.png`
-  `error_distribution.png`
-  `regression_results.json`

---

##  Sauvegarder l'environnement

Pour reproduire l'environnement sur une autre machine:

```bash
# Créer un backup des dépendances:
pip freeze > requirements_complete.txt

# Sur une autre machine, utiliser:
pip install -r requirements_complete.txt
```

---

##  Ressources utiles

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Gallery](https://matplotlib.org/gallery.html)
- [scikit-learn Docs](https://scikit-learn.org/stable/documentation.html)

---

##  Checklist d'installation

- [ ] Python 3.9+ installé
- [ ] Environnement virtuel créé et activé
- [ ] `pip install -r requirements.txt` réussi
- [ ] CUDA/PyTorch configuré (si GPU)
- [ ] Dossier `data/` créé avec fichiers .npz
- [ ] `analyze_width_dataset.py` s'exécute sans erreur
- [ ] Test forward pass réussi
- [ ] Premiers epochs d'entraînement produisent des résultats

---

**Installation complétée!** 

Procédez à l'étape: `python map_width_regressor.py`

