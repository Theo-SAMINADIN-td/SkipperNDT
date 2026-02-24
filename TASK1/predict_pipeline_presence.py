"""
Script de prédiction pour détecter la présence de conduites
Usage: python predict_pipeline_presence.py --input <file.npz>
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
from pipeline_presence_detector import PipelinePresenceClassifier
from common import load_and_preprocess


def preprocess_image(image_path, target_size=(224, 224)):
    """Prétraite une image .npz pour la prédiction."""
    tensor = load_and_preprocess(image_path, target_size)
    return tensor.unsqueeze(0)  # add batch dimension


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
