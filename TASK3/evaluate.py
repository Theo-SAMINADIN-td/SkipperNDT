"""
Evaluation script for TASK3: Current Intensity Classification
Computes comprehensive metrics on test/validation set
Usage: python evaluate.py --model <model_path> --data <data_dir> --split <test|val>
"""

import glob
import torch
import numpy as np
import argparse
import pandas as pd
from scipy.ndimage import zoom
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import json
import os

from train import CurrentIntensityClassifier


def preprocess_image(image_path: str, target_size=(224, 224)) -> torch.Tensor:
    """Prétraite une image .npz pour la prédiction"""
    data = np.load(image_path, allow_pickle=True)
    image = data["data"]

    if image.dtype == np.float16:
        image = image.astype(np.float32)

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    image = _resize_image(image)
    image = _normalize_channels(image)

    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def _resize_image(image: np.ndarray) -> np.ndarray:
    """Redimensionne l'image à la taille cible"""
    h, w, c = image.shape
    target_h, target_w = 224, 224

    zoom_h = target_h / h
    zoom_w = target_w / w

    resized = zoom(image, (zoom_h, zoom_w, 1), order=1)

    return resized


def _normalize_channels(image: np.ndarray) -> np.ndarray:
    """Normalise chaque canal indépendamment"""
    normalized = np.zeros_like(image)

    for c in range(image.shape[2]):
        channel = image[:, :, c]
        mean = np.mean(channel)
        std = np.std(channel)

        if std > 0:
            normalized[:, :, c] = (channel - mean) / std
        else:
            normalized[:, :, c] = channel - mean

    return normalized


def predict_batch(model, image_paths, device="cpu"):
    """Prédictions par batch pour efficacité"""
    predictions = []
    probabilities = []

    for image_path in tqdm(image_paths, desc="Predicting", leave=False):
        image_tensor = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probability = output.item()
            prediction = 1 if probability > 0.5 else 0

        predictions.append(prediction)
        probabilities.append(probability)

    return np.array(predictions), np.array(probabilities)


def load_ground_truth_labels(csv_path: str, data_dir: str, label_column="label"):
    """Charge les labels ground-truth depuis le CSV"""
    df = pd.read_csv(csv_path, sep=",")

    # Determine which column contains filenames
    filename_col = None
    if "field_file" in df.columns:
        filename_col = "field_file"
    elif "filename" in df.columns:
        filename_col = "filename"
    elif "file" in df.columns:
        filename_col = "file"
    else:
        # Try to infer from first column
        filename_col = df.columns[0]

    if label_column not in df.columns:
        print(f"Warning: Column '{label_column}' not found. Using 'label' column instead.")
        label_column = "label"

    if label_column not in df.columns:
        print(f"ERROR: Label column '{label_column}' not found in CSV!")
        print(f"Available columns: {df.columns.tolist()}")
        return {}

    labels_dict = {}
    for _, row in df.iterrows():
        filename = row[filename_col]
        label = float(row[label_column])
        full_path = os.path.join(data_dir, filename)
        
        # Only add if file exists on disk
        if os.path.exists(full_path):
            labels_dict[full_path] = label

    return labels_dict


def evaluate(model_path: str, data_dir: str, csv_path: str, label_column="label", device="cpu"):
    """Évalue le modèle sur un ensemble de données"""

    print(f"Loading model from: {model_path}")
    print(f"Evaluating on images from: {data_dir}")

    # Charger le modèle
    model = CurrentIntensityClassifier()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Charger les images et labels
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    labels_dict = load_ground_truth_labels(csv_path, data_dir, label_column)

    # Filtrer les images avec labels disponibles
    valid_paths = []
    ground_truth = []
    for img_path in image_paths:
        if img_path in labels_dict:
            valid_paths.append(img_path)
            ground_truth.append(int(labels_dict[img_path]))

    if len(valid_paths) == 0:
        print("ERROR: No images found with labels in CSV!")
        return

    print(f"Evaluating {len(valid_paths)} images with ground-truth labels")

    # Prédictions
    predictions, probabilities = predict_batch(model, valid_paths, device)

    # Métriques
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, zero_division=0)
    recall = recall_score(ground_truth, predictions, zero_division=0)
    f1 = f1_score(ground_truth, predictions, zero_division=0)

    # ROC-AUC
    try:
        auc_score = roc_auc_score(ground_truth, probabilities)
    except Exception:
        auc_score = None

    # Matrice de confusion
    cm = confusion_matrix(ground_truth, predictions)

    # Rapport détaillé
    report = classification_report(
        ground_truth, predictions, target_names=["Insufficient", "Sufficient"]
    )

    # Affichage des résultats
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS - CURRENT INTENSITY CLASSIFICATION")
    print(f"{'='*60}")
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    if auc_score is not None:
        print(f"ROC-AUC:   {auc_score:.4f}")

    print(f"\n{'='*60}")
    print("Confusion Matrix:")
    print(f"{'='*60}")
    print("                Predicted")
    print("                Insufficient  Sufficient")
    print(f"Actual Insufficient    {cm[0,0]:4d}        {cm[0,1]:4d}")
    print(f"       Sufficient      {cm[1,0]:4d}        {cm[1,1]:4d}")

    print(f"\n{'='*60}")
    print("Detailed Classification Report:")
    print(f"{'='*60}")
    print(report)

    # Sauvegarder les résultats
    results = {
        "model_path": model_path,
        "num_samples": len(valid_paths),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc_score) if auc_score is not None else None,
        "confusion_matrix": cm.tolist(),
        "targets": {"accuracy": 0.90, "recall": 0.85},
        "targets_met": {"accuracy": accuracy > 0.90, "recall": recall > 0.85},
    }

    results_path = "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✓ Results saved to {results_path}")

    # Générer les courbes
    plot_confusion_matrix(cm, save_path="confusion_matrix.png")
    if auc_score is not None:
        plot_roc_curve(ground_truth, probabilities, save_path="roc_curve.png")

    return results


def plot_confusion_matrix(cm, save_path="confusion_matrix.png"):
    """Affiche et sauvegarde la matrice de confusion"""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        ylabel="True label",
        xlabel="Predicted label",
    )
    ax.set_xticklabels(["Insufficient", "Sufficient"])
    ax.set_yticklabels(["Insufficient", "Sufficient"])

    # Afficher les valeurs
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    plt.title("Confusion Matrix - Current Intensity Classification")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(ground_truth, probabilities, save_path="roc_curve.png"):
    """Affiche et sauvegarde la courbe ROC"""
    fpr, tpr, _ = roc_curve(ground_truth, probabilities)
    auc = roc_auc_score(ground_truth, probabilities)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set(
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve - Current Intensity Classification",
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ ROC curve saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate current intensity classification model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Path to trained model checkpoint",
        default="current_intensity_classifier_epoch25.pth",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=False,
        help="Path to data directory",
        default="Task3_TEST",
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=False,
        help="Path to label CSV",
        default="TASK3_DATATRAINING/pipe_detection_label.csv",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        required=False,
        help="Column name for labels in CSV",
        default="label",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        return

    if not os.path.exists(args.data):
        print(f"ERROR: Data directory not found: {args.data}")
        return

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        return

    evaluate(
        args.model,
        args.data,
        args.csv,
        label_column=args.label_column,
        device=args.device,
    )


if __name__ == "__main__":
    main()
