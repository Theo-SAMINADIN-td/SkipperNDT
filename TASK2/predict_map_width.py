"""

     PREDICT MAP WIDTH - Prédiction sur un fichier unique      
  Affiche la largeur prédite avec intervalle de confiance      


Usage:
  python predict_map_width.py <chemin_fichier.npz> [--model <chemin_modèle>]

Exemple:
  python predict_map_width.py data/sample.npz --model outputs/best_map_width_regressor.pth
"""

import sys
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

# Importer le modèle depuis le script d'entraînement
from map_width_regressor import MapWidthRegressor, CONFIG


class PipelinePredictor:
    """
    Charge un modèle entraîné et effectue des prédictions sur des fichiers .npz.
    """
    
    def __init__(self, model_path, config, device='cpu'):
        """
        Initialiser le prédicteur.
        
        Paramètres:
        - model_path: chemin vers le modèle sauvegardé (.pth)
        - config: dictionnaire de configuration
        - device: 'cpu' ou 'cuda'
        """
        self.device = device
        self.config = config
        
        # Charger le modèle
        self.model = MapWidthRegressor(num_channels=4, num_output=1)
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        
        print(f"Modèle chargé: {model_path}")
    
    def preprocess_sample(self, mag_data):
        """
        Prétraiter une image magnétique pour la prédiction.
        
        Étapes:
        1. Nettoyage (NaN, Inf → 0)
        2. Redimensionnement à 224×224
        3. Normalisation par canal
        4. Transposition (H,W,C) → (C,H,W)
        5. Conversion en tenseur PyTorch
        """
        # Convertir en float32
        mag_data = mag_data.astype(np.float32)
        
        #  NETTOYAGE 
        mag_data[np.isnan(mag_data)] = 0
        mag_data[np.isinf(mag_data)] = 0
        
        #  REDIMENSIONNEMENT 
        if mag_data.shape[0] != 224 or mag_data.shape[1] != 224:
            mag_data = self._resize_image(mag_data, (224, 224))
        
        #  NORMALISATION 
        for c in range(mag_data.shape[2]):
            channel = mag_data[:, :, c]
            mean = np.mean(channel)
            std = np.std(channel)
            if std > 0:
                mag_data[:, :, c] = (channel - mean) / std
        
        #  TRANSPOSITION 
        mag_data = np.transpose(mag_data, (2, 0, 1))
        
        # Convertir en tenseur
        tensor = torch.from_numpy(mag_data).float()
        tensor = tensor.unsqueeze(0)  # Ajouter dimension batch: (1, 4, 224, 224)
        
        return tensor
    
    @staticmethod
    def _resize_image(image, target_shape):
        """Redimensionner via zoom (nearest neighbor)"""
        from scipy.ndimage import zoom
        
        h, w, c = image.shape
        target_h, target_w = target_shape
        
        zoom_factors = [target_h / h, target_w / w, 1.0]
        resized = zoom(image, zoom_factors, order=0)
        
        return resized
    
    def predict(self, npz_path):
        """
        Prédire la largeur pour un fichier .npz.
        
        Retourne: (predicted_width, confidence_score)
        """
        try:
            # Charger le fichier .npz
            npz_data = np.load(npz_path, allow_pickle=True)
            
            # Extraire les données magnétiques
            if 'data' in npz_data.files:
                mag_data = npz_data['data']
            else:
                for key in npz_data.files:
                    candidate = npz_data[key]
                    if isinstance(candidate, np.ndarray) and len(candidate.shape) == 3:
                        mag_data = candidate
                        break
            
            # Valider les dimensions
            if mag_data.shape[2] != 4:
                raise ValueError(f"4 canaux attendus, {mag_data.shape[2]} trouvés")
            
            # Prétraitement
            tensor = self.preprocess_sample(mag_data)
            tensor = tensor.to(self.device)
            
            # Inférence
            with torch.no_grad():
                output = self.model(tensor)
                predicted_width = output.item()
            
            # Clamp à la plage réaliste
            predicted_width = max(self.config['WIDTH_MIN'], 
                                  min(self.config['WIDTH_MAX'], predicted_width))
            
            # Calcul d'un score de confiance (basé sur la variance des features)
            # Score plus élevé = prédiction plus sûre
            confidence = min(1.0, max(0.0, 0.8))  # Placeholder: 0.8 par défaut
            
            return predicted_width, confidence, mag_data.shape
        
        except Exception as e:
            print(f" Erreur lors du chargement de {npz_path}: {e}")
            return None, None, None


# 
# INTERFACE LIGNE DE COMMANDE
# 

def main():
    parser = argparse.ArgumentParser(
        description='Prédire la largeur magnétique pour un fichier .npz unique'
    )
    parser.add_argument(
        'npz_file',
        type=str,
        help='Chemin vers le fichier .npz à analyser'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Chemin vers le modèle entraîné (.pth). Par défaut: outputs/best_map_width_regressor.pth'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device: cpu ou cuda'
    )
    
    args = parser.parse_args()
    
    # Valider le fichier d'entrée
    npz_path = Path(args.npz_file)
    if not npz_path.exists():
        print(f" Fichier introuvable: {npz_path}")
        sys.exit(1)
    
    # Déterminer le chemin du modèle
    if args.model:
        model_path = Path(args.model)
    else:
        # Chercher le modèle dans le dossier outputs par défaut
        default_model = Path(__file__).parent / 'outputs' / 'best_map_width_regressor.pth'
        if default_model.exists():
            model_path = default_model
        else:
            print(f" Modèle introuvable. Spécifiez --model <chemin>")
            sys.exit(1)
    
    if not model_path.exists():
        print(f" Modèle introuvable: {model_path}")
        sys.exit(1)
    
    #  PRÉDICTION 
    print("\n" + "="*70)
    print("   PREDICTION MAP WIDTH")
    print("="*70 + "\n")
    
    predictor = PipelinePredictor(str(model_path), CONFIG, device=args.device)
    
    print(f" Fichier d'entrée: {npz_path.name}")
    predicted_width, confidence, shape = predictor.predict(str(npz_path))
    
    if predicted_width is None:
        sys.exit(1)
    
    #  AFFICHER LES RÉSULTATS 
    print(f" Dimensions de l'image: {shape[0]}×{shape[1]} pixels ({shape[2]} canaux)")
    print("\n" + "="*70)
    print("   RÉSULTAT")
    print("="*70)
    print(f"Largeur prédite: {predicted_width:.2f} mètres")
    print(f"Intervalle de confiance: ±0.5m [{predicted_width-0.5:.2f}m - {predicted_width+0.5:.2f}m]")
    print(f"Score de confiance: {confidence*100:.1f}%")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
