# SkipperNDT : Détection Automatisée de Conduites Enfouies

Ce projet implémente une série de modèles d'apprentissage profond et de méthodes géométriques pour classifier l'état (présence, intensité du courant, type géométrique) de canalisations souterraines à partir d'images magnétiques multicanaux (données `.npz`).

L'architecture s'appuie principalement sur PyTorch et la bibliothèque de vision `torchvision` (spécifiquement le modèle **ResNet18** ré-imaginé pour accepter des tenseurs 4-canaux).

## 🛠️ 1. Installation de l'environnement

### Création de l'environnement virtuel et installation
```bash
# 1. Cloner ou naviguer dans le répertoire du projet
cd /chemin/vers/SkipperNDT

# 2. Création d'un environnement virtuel propre (Optionnel mais recommandé)
python3 -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# Sous Windows : venv\Scripts\activate

# 3. Installation des dépendances requises
pip install -r requirements.txt
```

> [!NOTE] 
> La configuration matérielle repose fortement sur l'entraînement de réseaux convolutifs. Si vous possédez une carte graphique compatible CUDA, assurez-vous que `torch` a bien détecté celle-ci pour un apprentissage accéléré. Sinon, les scripts se replieront automatiquement sur le CPU.

## 📂 2. Préparation des Données

Afin de pouvoir exécuter les réseaux de neurones, la base de données doit respecter la structure attendue :
- Les données synthétiques (float16) dans un dossier nommé `Training_database_float16/`
- Les données réelles dans un dossier nommé `real_data/`
- Le dictionnaire des labels (`pipe_detection_label.csv`)

## 🚀 3. Reproduction des Expérimentations (Train & Inférence)

Le projet est logiquement découpé en tâches d'exploitations autonomes. Les scripts incluent différents arguments de ligne de commande (ClI) pour lier les bons dossiers.

### Tâche 1 : Détection de présence de la canalisation
Le but est d'entraîner le modèle sur la présence basique du pipeline sur l'image magnétique.

**1. Lancement de l'entraînement :**
```bash
cd TASK1
python train.py \
    --input-data ../Training_database_float16 \
    --csv-data ../Training_database_float16/pipe_detection_label.csv
cd ..
```
**Arguments du Train :**
- `--input-data` : Chemin vers le dossier contenant les fichiers `.npz`.
- `--csv-data` : Chemin vers le fichier CSV définissant les labels (Ground Truth).

*Génère les poids qui permettront de faire du "transfer learning" pour les autres tâches (ex: `task1_epoch8_best.pth`).*

**2. Lancement de l'inférence :**
L'inférence parcourt un lot d'images pour prédire la présence de pipelines, comparer aux labels si structurés, et produire un résumé.
```bash
cd TASK1
python inference.py \
    --input "../Donnees_A_Tester/*.npz" \
    --model ../task1_epoch8_best.pth
cd ..
```
**Arguments d'Inférence :**
- `--input` : Chemin (expression régulière acceptée) vers un lot de fichiers `.npz`.
- `--model` : Chemin vers votre fichier `.pth` pré-entraîné.
- `--device` : (Optionnel) `cpu` ou `cuda`, par défaut détecté automatiquement.

### Tâche 2 : Estimation Géométrique de la Largeur
Il s'agit du seul programme purement mathématique (non deep-learning) basé sur une heuristique de localisation des extrema :
```bash
cd TASK2
python map_width_geometric.py \
    --input-data ../Training_database_float16 \
    --csv-data ../Training_database_float16/pipe_detection_label.csv
cd ..
```
**Arguments :** (Identiques à la tâche 1)
- `--input-data` : Path vers les `.npz`.
- `--csv-data` : Path vers le CSV des métadonnées (largeur métrique à l'origine).

### Tâche 3 : Classification de l'intensité du Courant (Protection)
Vise à évaluer si le courant de protection cathodique identifié est "Suffisant" ou "Insuffisant", en forçant le réseau à maximiser la sécurité (minimiser les faux négatifs).

**1. Lancement de l'entraînement :**
```bash
cd TASK3
python train.py \
    --input-data "../Training_database_float16/*.npz" \
    --csv-data ../Training_database_float16/pipe_detection_label.csv
cd ..
```
**Arguments du Train :**
- `--input-data` : Chemin contenant les fichiers d'entrée matriciels.
- `--csv-data` : Fichier de validation avec entêtes correspondants.

**2. Lancement de l'inférence / Évaluation :**
Afin de valider les performances sur de nouvelles données, le script d'évaluation calcule les prédictions et dresse un bilan avec courbes ROC et matrices.
```bash
cd TASK3
python evaluate.py \
    --model current_intensity_classifier_epoch25.pth \
    --data ../Task3_TEST \
    --csv ../TASK3_DATATRAINING/pipe_detection_label.csv \
    --label-column label
cd ..
```
**Arguments d'Évaluation :**
- `--model` : Chemin vers le modèle PyTorch pré-entraîné (`.pth`).
- `--data` : Dossier contenant le jeu de données `.npz` à évaluer.
- `--csv` : Chemin vers le fichier contenant les étiquettes de référence (Ground Truth).
- `--label-column` : Colonne du CSV contenant la classe (par défaut `label`).
- `--device` : (Optionnel) `cpu` ou `cuda`.

### Tâche 4 : Distinction du Type de Conduite (Simple vs. Parallèles)
Cette tâche procède par **Apprentissage par Transfert** et nécessite donc un modèle pré-entraîné de la Tâche 1 pour s'initialiser correctement.

**1. Lancement de l'entraînement :**
```bash
cd TASK4
python train.py \
    --input-data "../Training_database_float16/*.npz" \
    --csv-data ../Training_database_float16/pipe_detection_label.csv \
    --real-data ../real_data \
    --presence-model ../task1_epoch8_best.pth
cd ..
```
**Arguments du Train :**
- `--input-data` : Chemin des tableaux numpy d'entraînement.
- `--csv-data` : Tableau des labels incluant l'info de type de tuyau.
- `--real-data` : Chemin vers le dossier des jeux de données réels `(str)`.
- `--presence-model` : Chemin absolu ou relatif vers le checkpoint ResNet18 `.pth` issu de la tâche 1 pour le Transfer Learning.

**2. Lancement de l'inférence :**
L'inférence (génération de déductions sur un nouveau lot sans s'entraîner) s'exécute ainsi :
```bash
cd TASK4
python inference.py \
    --model-path ./MODELS/best_pipe_type_classifier.pth \
    --input-data ../Donnees_A_Tester/ \
    --output-file ./MODELS/mes_predictions.csv
cd ..
```
**Arguments d'Inférence :**
- `--model-path` : Fichier `.pth` généré à l'issue du `train.py`.
- `--input-data` : Chemin vers un dossier d'exemples vierges (`.npz`).
- `--output-file` : Nom voulu pour l'export Excel/CSV du résultat prédictif.

## 📊 4. Évaluation

Pour chaque tâche lancée, le code de *Train* génère localement :
- Un dictionnaire `json` regroupant les métriques finales sur la base de Test (Accuracy, Recall, F1-Score, Confusion Matrix).
- Les meilleurs poids au format `.pth` à utiliser lors du déploiement opérationnel.
- Un ensemble de graphiques via matplotlib sauvegardé en `.png` démontrant l'entraînement continu.

---
**Mainteneur / Contexte :** Projet d'analyse géomagnétique et MFL - HETIC & SkipperNDT.
