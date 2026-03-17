# ============================================================
#  ÉVALUATION DU MEILLEUR MODÈLE DE CLASSIFICATION DE TYPE DE TUYAU
# ============================================================
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import torchvision.transforms.functional as TF
from torchvision import models

# ============================================================
#  CHEMINS & CONFIGURATION
# ============================================================
CSV_PATH             = "/content/drive/MyDrive/SkipperNDT/pipe_detection_label.csv"
MODELS_DIR           = "/content/drive/MyDrive/SkipperNDT/models"
BEST_TYPE_MODEL_PATH = f"{MODELS_DIR}/best_pipe_type_classifier.pth"
DATA_DIR_REAL        = "/content/real_data"
BATCH_SIZE           = 32
TARGET_SIZE          = (128, 128)
NUM_WORKERS          = 2

# ── Auto-détection du bon sous-dossier Training_database ──────────────────
def find_train_dir():
    candidates = [
        "/content/Training_database_float16/Training_database_float16",
        "/content/Training_database_float16",
    ]
    for path in candidates:
        if os.path.isdir(path):
            files = os.listdir(path)
            npz   = [f for f in files if f.endswith(".npz")]
            if npz:
                print(f"  ✓ DATA_DIR_TRAIN détecté : {path} ({len(npz)} fichiers .npz)")
                return path
    raise FileNotFoundError(
        "Aucun dossier Training_database avec .npz trouvé. "
        "Vérifiez l'extraction avec : os.listdir('/content/Training_database_float16')"
    )

DATA_DIR_TRAIN = find_train_dir()

# ============================================================
#  BASE DATASET
# ============================================================
class BaseNpzDataset(Dataset):
    def __init__(self, file_paths, target_size=(128, 128), augment=False):
        self.file_paths  = file_paths
        self.target_size = target_size
        self.augment     = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data   = np.load(self.file_paths[idx], allow_pickle=True)
        image  = data["data"].astype(np.float32)
        image  = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image  = self._normalize_channels(image)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        tensor = TF.resize(tensor, list(self.target_size), antialias=True)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                tensor = TF.hflip(tensor)
            if torch.rand(1).item() > 0.5:
                tensor = TF.vflip(tensor)
            angle  = torch.FloatTensor(1).uniform_(-30, 30).item()
            tensor = TF.rotate(tensor, angle)

        return tensor, self._make_target(idx)

    def _make_target(self, idx):
        raise NotImplementedError

    def _normalize_channels(self, image):
        normalized = np.zeros_like(image)
        for c in range(image.shape[2]):
            ch  = image[:, :, c]
            std = np.std(ch)
            normalized[:, :, c] = (ch - np.mean(ch)) / std if std > 0 else ch - np.mean(ch)
        return normalized

# ============================================================
#  DATASET PIPE TYPE
# ============================================================
class PipeTypeDataset(BaseNpzDataset):
    """Label 0 = single pipe | Label 1 = parallel pipes"""

    def __init__(self, file_paths, labels, target_size=(128, 128), augment=False):
        super().__init__(file_paths, target_size, augment)
        self.labels = labels

    def _make_target(self, idx):
        return torch.tensor(self.labels[idx], dtype=torch.float32)

# ============================================================
#  MODÈLE CLASSIFIER
# ============================================================
class PipeTypeClassifier(nn.Module):
    """Classifieur single vs parallel — backbone ResNet18 transféré depuis Task1"""

    def __init__(self, num_channels=4):
        super().__init__()
        self.resnet = models.resnet18(weights=None)

        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            num_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        in_features      = self.resnet.fc.in_features
        self.resnet.fc   = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.classifier(self.resnet(x))

# ============================================================
#  CHARGEMENT DES DONNÉES
# ============================================================
def load_pipe_type_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    df       = pd.read_csv(csv_path, sep=';')
    df_pipes = df[df['label'] == 1].copy()

    print(f"  Samples avec conduite : {len(df_pipes)}")
    print(f"  Single pipe           : {(df_pipes['pipe_type'] == 'single').sum()}")
    print(f"  Parallel pipe         : {(df_pipes['pipe_type'] == 'parallel').sum()}")
    if 'source' in df_pipes.columns:
        print(f"  Répartition sources   :\n{df_pipes['source'].value_counts().to_string()}")

    file_paths, labels = [], []
    found_train, found_real, missing = 0, 0, 0

    for _, row in df_pipes.iterrows():
        field_file = str(row['field_file']).strip()

        path_train = os.path.join(DATA_DIR_TRAIN, field_file)
        path_real  = os.path.join(DATA_DIR_REAL,  field_file)

        if os.path.exists(path_train):
            fpath = path_train
            found_train += 1
        elif os.path.exists(path_real):
            fpath = path_real
            found_real += 1
        else:
            missing += 1
            continue

        file_paths.append(fpath)
        labels.append(1.0 if row['pipe_type'] == 'parallel' else 0.0)

    print(f"  ✓ {len(file_paths)} samples chargés "
          f"({found_train} training_db | {found_real} real_data) | {missing} manquants")
    return file_paths, labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Reconstruction du test set (même seed que l'entraînement) ─────────────
file_paths, labels = load_pipe_type_data(CSV_PATH)

_, temp_files, _, temp_labels = train_test_split(
    file_paths, labels, test_size=0.3, random_state=42, stratify=labels
)
_, test_files, _, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

test_loader = DataLoader(
    PipeTypeDataset(test_files, test_labels, TARGET_SIZE, augment=False),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
)
print(f"✓ Test set reconstruit : {len(test_files)} samples")

# ── Chargement du checkpoint ───────────────────────────────────────────────
if not os.path.exists(BEST_TYPE_MODEL_PATH):
    raise FileNotFoundError(f"Modèle introuvable : {BEST_TYPE_MODEL_PATH}")

checkpoint = torch.load(BEST_TYPE_MODEL_PATH, map_location=device, weights_only=False)

model_eval = PipeTypeClassifier(num_channels=4).to(device)
model_eval.load_state_dict(checkpoint['model_state_dict'])
model_eval.eval()

print(f"✓ Modèle chargé depuis : {BEST_TYPE_MODEL_PATH}")
print(f"  Sauvegardé à l'époque : {checkpoint['epoch'] + 1}")
print(f"  Val Accuracy          : {checkpoint['val_accuracy']:.4f}")
print(f"  Val Recall            : {checkpoint['val_recall']:.4f}")
print(f"  Val F1                : {checkpoint['val_f1']:.4f}")

# ── Inférence ─────────────────────────────────────────────────────────────
all_preds, all_probs, all_labels_list = [], [], []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Évaluation test set"):
        images  = images.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True).unsqueeze(1)
        outputs = model_eval(images)
        probs   = torch.sigmoid(outputs)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend((probs > 0.5).float().cpu().numpy())
        all_labels_list.extend(labels.cpu().numpy())

all_preds  = np.array(all_preds).flatten()
all_probs  = np.array(all_probs).flatten()
all_labels = np.array(all_labels_list).flatten()

# ── Métriques ─────────────────────────────────────────────────────────────
test_acc    = accuracy_score(all_labels, all_preds)
test_recall = recall_score(all_labels, all_preds, zero_division=0)
test_f1     = f1_score(all_labels, all_preds, zero_division=0)
cm          = confusion_matrix(all_labels, all_preds)

print(f"\n{'='*50}")
print("FINAL TEST RESULTS - PIPE TYPE CLASSIFIER")
print(f"{'='*50}")
print(f"  Accuracy  : {test_acc:.4f}")
print(f"  Recall    : {test_recall:.4f}")
print(f"  F1-Score  : {test_f1:.4f}")
print(f"\nClassification Report :")
print(classification_report(all_labels, all_preds, target_names=['Single Pipe', 'Parallel Pipes']))

# ── Visualisations ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Single', 'Parallel'],
            yticklabels=['Single', 'Parallel'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

axes[1].hist(all_probs[all_labels == 0], bins=30, alpha=0.6, label='Single Pipe',    color='steelblue')
axes[1].hist(all_probs[all_labels == 1], bins=30, alpha=0.6, label='Parallel Pipes', color='tomato')
axes[1].axvline(x=0.5, color='black', linestyle='--', label='Threshold 0.5')
axes[1].set_title('Distribution des probabilités prédites')
axes[1].set_xlabel('P(Parallel)')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{MODELS_DIR}/pipe_type_eval_best_model.png")
plt.show()

# ── Sauvegarde JSON ────────────────────────────────────────────────────────
results = {
    'best_epoch'      : int(checkpoint['epoch'] + 1),
    'test_accuracy'   : float(test_acc),
    'test_recall'     : float(test_recall),
    'test_f1'         : float(test_f1),
    'confusion_matrix': cm.tolist(),
}
results_path = f"{MODELS_DIR}/pipe_type_best_model_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\n💾 Résultats sauvegardés → {results_path}")
print(f"📊 Graphiques sauvegardés → {MODELS_DIR}/pipe_type_eval_best_model.png")