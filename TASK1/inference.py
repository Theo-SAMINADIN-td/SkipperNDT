"""
Script de prédiction pour détecter la présence de conduites
Usage: python predict_pipeline_presence.py --input <folder> --model <model_path>
"""

import glob
import torch
import numpy as np
import argparse
from scipy.ndimage import zoom
from tqdm import tqdm
from train import PipelinePresenceClassifier


def preprocess_image(image_path : str, target_size=(224, 224)) -> torch.Tensor:
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


def _resize_image(image : np.ndarray) -> np.ndarray:
    """Redimensionne l'image à la taille cible"""
    h, w, c = image.shape
    target_h, target_w = 224, 224

    zoom_h = target_h / h
    zoom_w = target_w / w

    resized = zoom(image, (zoom_h, zoom_w, 1), order=1)

    return resized


def _normalize_channels(image : np.ndarray) -> np.ndarray:
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


def predict(model_path : str, image_path : str, device="cpu") -> tuple[float, int]:
    """Prédit la présence de conduite dans une image

    Returns:
        probability: Probabilité de présence de conduite (0-1)
        prediction: 0 (absence) ou 1 (présence)
    """
    model = PipelinePresenceClassifier()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0

    return probability, prediction


def main():
    parser = argparse.ArgumentParser(
        description="Predict pipeline presence in magnetic images"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Path to .npz folder",
        default="Training_database_float16/*.npz",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Path to trained model",
        default=r"TASK1_weights/task1_epoch8_best.pth",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    print(f"Analyzing image: {args.input}")
    no_pipe_predictions = 0
    pipe_prediction = 0
    with_pipelines_groundtruth = 0
    without_pipelines_groundtruth = 0
    for file in tqdm(glob.glob(args.input), desc="Analyzing images"):
        probability, prediction = predict(args.model, file, args.device)
        if prediction == 0:
            no_pipe_predictions += 1
        elif prediction == 1:
            pipe_prediction += 1

        if "no_pipe" in file:
            without_pipelines_groundtruth += 1
        else:
            with_pipelines_groundtruth += 1

    print(f"\nTotal images analyzed: {no_pipe_predictions + pipe_prediction}")

    print("\nSummary:")
    print(
        f"{pipe_prediction} pipeline detected, expected {with_pipelines_groundtruth}."
    )
    print(
        f"{no_pipe_predictions} no pipeline detected, expected {without_pipelines_groundtruth}."
    )


if __name__ == "__main__":
    main()
