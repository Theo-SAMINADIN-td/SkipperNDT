"""
Script de prédiction pour détecter la présence de conduites
Usage: python predict_pipeline_presence.py --input <file.npz>
"""

import torch
import numpy as np
import argparse
from scipy.ndimage import zoom
from pipeline_presence_detector import PipelinePresenceClassifier


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Prétraite une image .npz pour la prédiction
    """
    # Charger l'image
    data = np.load(image_path, allow_pickle=True)
    image = data['data']  # Shape: (H, W, 4)
    
    # Convertir en float32 si nécessaire (float16 n'est pas supporté par scipy.zoom)
    if image.dtype == np.float16:
        image = image.astype(np.float32)
    
    # Gérer les NaN
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Redimensionner l'image
    h, w, c = image.shape
    target_h, target_w = target_size
    zoom_h = target_h / h
    zoom_w = target_w / w
    resized = zoom(image, (zoom_h, zoom_w, 1), order=1)
    
    # Normaliser par canal
    normalized = np.zeros_like(resized)
    for c in range(resized.shape[2]):
        channel = resized[:, :, c]
        mean = np.mean(channel)
        std = np.std(channel)
        if std > 0:
            normalized[:, :, c] = (channel - mean) / std
        else:
            normalized[:, :, c] = channel - mean
    
    # Convertir en tensor (H, W, C) -> (C, H, W)
    image_tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0)  # Ajouter batch dimension
    
    return image_tensor


def predict(model_path, image_path, device='cpu'):
    """
    Prédit la présence de conduite dans une image
    
    Returns:
        probability: Probabilité de présence de conduite (0-1)
        prediction: 0 (absence) ou 1 (présence)
    """
    # Charger le modèle
    model = PipelinePresenceClassifier(num_channels=4)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Prétraiter l'image
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Prédiction
    with torch.no_grad():
        output = model(image_tensor)
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0
    
    return probability, prediction


def main():
    parser = argparse.ArgumentParser(description='Predict pipeline presence in magnetic images')
    parser.add_argument('--input', type=str, required=True, help='Path to input .npz file')
    parser.add_argument('--model', type=str, default='best_pipeline_classifier.pth', 
                        help='Path to trained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Prédiction
    print(f"Loading model from: {args.model}")
    print(f"Analyzing image: {args.input}")
    
    probability, prediction = predict(args.model, args.input, args.device)
    
    # Afficher les résultats
    print(f"\n{'='*50}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*50}")
    print(f"Probability of pipeline presence: {probability:.4f} ({probability*100:.2f}%)")
    print(f"Prediction: {'PIPELINE DETECTED' if prediction == 1 else 'NO PIPELINE'}")
    print(f"Confidence: {max(probability, 1-probability)*100:.2f}%")
    
    if prediction == 0:
        print(f"\n⚠️  No pipeline detected in this image")
    else:
        print(f"\n✓ Pipeline presence confirmed")


if __name__ == "__main__":
    main()
