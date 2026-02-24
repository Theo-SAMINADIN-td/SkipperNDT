"""
Shared preprocessing utilities reused by TASK1 and TASK2.
"""

import numpy as np
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def fill_nan_with_median(image: np.ndarray) -> np.ndarray:
    """Remplace NaN/Inf par la médiane de chaque canal (in-place, returns image)."""
    for c in range(image.shape[2]):
        ch = image[:, :, c]
        mask = ~np.isnan(ch) & ~np.isinf(ch)
        if np.any(mask):
            ch[~mask] = np.median(ch[mask])
        else:
            ch[:] = 0.0
        image[:, :, c] = ch
    return image


def resize_with_padding(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Redimensionne en préservant le ratio d'aspect.
    Le padding est rempli avec la médiane par canal.
    """
    h, w, c = image.shape
    target_h, target_w = target_size

    scale = min(target_h / h, target_w / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    resized = zoom(image, (scale, scale, 1), order=1) if scale != 1.0 else image.copy()

    padded = np.zeros((target_h, target_w, c), dtype=resized.dtype)
    for ch in range(c):
        padded[:, :, ch] = np.median(resized[:, :, ch])

    start_h = (target_h - new_h) // 2
    start_w = (target_w - new_w) // 2
    padded[start_h:start_h + new_h, start_w:start_w + new_w, :] = resized
    return padded


def normalize_channels(image: np.ndarray) -> np.ndarray:
    """Normalisation z-score par canal."""
    normalized = np.zeros_like(image)
    for c in range(image.shape[2]):
        ch = image[:, :, c]
        mean, std = np.mean(ch), np.std(ch)
        normalized[:, :, c] = (ch - mean) / std if std > 0 else ch - mean
    return normalized


def load_and_preprocess(file_path: str, target_size: tuple) -> torch.Tensor:
    """
    Charge un fichier .npz et applique le pipeline de prétraitement complet.
    Retourne un tensor (C, H, W) float32.
    """
    data = np.load(file_path, allow_pickle=True)
    image = data['data']

    if image.dtype == np.float16:
        image = image.astype(np.float32)

    image = fill_nan_with_median(image)
    image = resize_with_padding(image, target_size)
    image = normalize_channels(image)

    return torch.from_numpy(image).permute(2, 0, 1).float()


# ─────────────────────────────────────────────────────────────────────────────
# Base Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BaseNpzDataset(Dataset):
    """
    Dataset de base réutilisable pour TASK1 et TASK2.
    Les sous-classes doivent uniquement définir `_make_target(idx)`.
    """

    def __init__(self, file_paths: list, target_size: tuple = (224, 224)):
        self.file_paths = file_paths
        self.target_size = target_size

    def __len__(self):
        return len(self.file_paths)

    def _make_target(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        image = load_and_preprocess(self.file_paths[idx], self.target_size)
        target = self._make_target(idx)
        return image, target


# ─────────────────────────────────────────────────────────────────────────────
# EfficientNet-V2-S backbone factory
# ─────────────────────────────────────────────────────────────────────────────

def build_efficientnet_v2s_backbone(num_channels: int = 4, pretrained: bool = False):
    """
    Retourne le backbone EfficientNet-V2-S adapté pour num_channels canaux,
    sans tête de classification (classifier remplacé par nn.Identity).
    """
    import torch.nn as nn
    from torchvision import models

    weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    backbone = models.efficientnet_v2_s(weights=weights)

    old_conv = backbone.features[0][0]
    assert isinstance(old_conv, nn.Conv2d)
    backbone.features[0][0] = nn.Conv2d(
        num_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()

    return backbone, in_features
