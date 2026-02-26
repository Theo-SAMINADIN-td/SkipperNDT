"""
Script de prédiction pour détecter la présence de conduites
Usage: python predict_pipeline_presence.py --input <file.npz>
"""

import glob
import torch
import numpy as np
import argparse
from scipy.ndimage import zoom
from tqdm import tqdm
from train import PipelinePresenceClassifier

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Prétraite une image .npz pour la prédiction
    """
    data = np.load(image_path, allow_pickle=True)
    image = data['data']  # Shape: (H, W, 4)

    # Convertir en float32 si nécessaire (float16 n'est pas supporté par scipy.zoom)
    if image.dtype == np.float16:
        image = image.astype(np.float32)

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)  # Remplacer NaN/Inf par des valeurs spécifiques

    # Redimensionner l'image
    image = _resize_image(image)

    # Normaliser les données (par canal)
    image = _normalize_channels(image)

    # Convertir en tensor (H, W, C) -> (C, H, W)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0)  # Ajouter batch dimension
    
    return image_tensor



def _resize_image(image):
    """Redimensionne l'image à la taille cible"""
    from scipy.ndimage import zoom


    h, w, c = image.shape
    target_h, target_w = 224, 224  # Taille cible fixe

    # Calculer les facteurs de zoom
    zoom_h = target_h / h
    zoom_w = target_w / w

    # Redimensionner chaque canal
    resized = zoom(image, (zoom_h, zoom_w, 1), order=1)

    return resized

def _normalize_channels(image):
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

def predict(model_path, image_path, device='cpu'):
    """
    Prédit la présence de conduite dans une image
    
    Returns:
        probability: Probabilité de présence de conduite (0-1)
        prediction: 0 (absence) ou 1 (présence)
    """
    # Charger le modèle
    model = PipelinePresenceClassifier()
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
    # parser = argparse.ArgumentParser(description='Predict pipeline presence in magnetic images')
    # parser.add_argument('--input', type=str, required=True, help='Path to input .npz file')
    # parser.add_argument('--model', type=str, default='best_pipeline_classifier.pth', 
    #                     help='Path to trained model')
    # parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
    #                     help='Device to use (cuda/cpu)')
    
    # args = parser.parse_args()
    
    # # Prédiction
    # print(f"Loading model from: {args.model}")
    # print(f"Analyzing image: {args.input}")
    no_pipe_predictions = 0
    pipe_prediction= 0
    for file in tqdm(glob.glob("TEST/*.npz"), desc="Analyzing images"):
        # print(f"Analyzing image: {file}")
        
        probability, prediction = predict("weights/task1_epoch8_best.pth", file, "cuda")
        if prediction == 0: 
            no_pipe_predictions += 1
        elif prediction == 1:
            pipe_prediction += 1

        #     # Afficher les résultats
        # print(f"\n{'='*50}")
        # print("PREDICTION RESULTS")
        # print(f"{'='*50}")
        # print(f"Probability of pipeline presence: {probability:.4f} ({probability*100:.2f}%)")
        # print(f"Prediction: {'PIPELINE DETECTED' if prediction == 1 else 'NO PIPELINE'}")
        # print(f"Confidence: {max(probability, 1-probability)*100:.2f}%")
        
        # if prediction == 0:
        #     print("\n⚠️  No pipeline detected in this image")
        # else:
        #     print("\n✓ Pipeline presence confirmed")

    
    print(f"\nTotal images analyzed: {no_pipe_predictions + pipe_prediction}")
    print(f"Predicted NO PIPELINE: {no_pipe_predictions} images")
    print(f"Predicted PIPELINE DETECTED: {pipe_prediction} images")


if __name__ == "__main__":
    main()
