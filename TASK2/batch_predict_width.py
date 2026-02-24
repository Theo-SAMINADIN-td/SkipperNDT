"""

     BATCH PREDICT WIDTH - Prédictions sur multiple fichiers   
  Export des résultats en CSV avec colonne de confiance        


Usage:
  python batch_predict_width.py <dossier_données> [--model <chemin_modèle>] [--output <fichier_csv>]

Exemple:
  python batch_predict_width.py data/ --output predictions.csv
"""

import sys
import argparse
import numpy as np
import csv
from pathlib import Path
from tqdm import tqdm

import torch

# Importer le modèle et le prédicteur
from map_width_regressor import MapWidthRegressor, CONFIG
from predict_map_width import PipelinePredictor


# 
# PRÉDICTIONS BATCH
# 

def batch_predict(input_dir, model_path, output_csv, device='cpu'):
    """
    Prédire les largeurs pour tous les fichiers .npz d'un dossier.
    
    Paramètres:
    - input_dir: dossier contenant les fichiers .npz
    - model_path: chemin du modèle entraîné
    - output_csv: chemin du fichier CSV de sortie
    - device: 'cpu' ou 'cuda'
    """
    
    # Initialiser le prédicteur
    predictor = PipelinePredictor(str(model_path), CONFIG, device=device)
    
    # Trouver tous les fichiers .npz
    input_path = Path(input_dir)
    npz_files = sorted(list(input_path.glob('**/*.npz')))
    
    if len(npz_files) == 0:
        print(f"  Aucun fichier .npz trouvé dans {input_dir}")
        return
    
    print(f" Trouvé {len(npz_files)} fichiers à analyser\n")
    
    #  PRÉDICTIONS 
    results = []
    
    for npz_file in tqdm(npz_files, desc="Prédictions en cours"):
        try:
            predicted_width, confidence, shape = predictor.predict(str(npz_file))
            
            if predicted_width is not None:
                results.append({
                    'filename': npz_file.name,
                    'relative_path': str(npz_file.relative_to(input_path)),
                    'predicted_width_m': round(predicted_width, 3),
                    'confidence': round(confidence, 3),
                    'image_dimensions': f"{shape[0]}x{shape[1]}",
                })
        
        except Exception as e:
            print(f"\n  Erreur pour {npz_file.name}: {e}")
            continue
    
    #  EXPORT CSV 
    if len(results) > 0:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n Résultats exportés: {output_path}")
        print(f"  {len(results)} fichiers traités avec succès")
        
        # Statistiques
        widths = [r['predicted_width_m'] for r in results]
        print(f"\n Statistiques des largeurs prédites:")
        print(f"   Min: {min(widths):.2f}m | Max: {max(widths):.2f}m")
        print(f"   Mean: {np.mean(widths):.2f}m | Std: {np.std(widths):.2f}m\n")
    else:
        print(" Aucune prédiction réussie")


# 
# INTERFACE CLI
# 

def main():
    parser = argparse.ArgumentParser(
        description='Prédictions batch sur un dossier de fichiers .npz'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Dossier contenant les fichiers .npz'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Chemin vers le modèle entraîné (.pth)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Fichier CSV de sortie (défaut: predictions.csv)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device: cpu ou cuda'
    )
    
    args = parser.parse_args()
    
    # Valider le dossier d'entrée
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f" Dossier introuvable: {input_path}")
        sys.exit(1)
    
    # Déterminer le chemin du modèle
    if args.model:
        model_path = Path(args.model)
    else:
        default_model = Path(__file__).parent / 'outputs' / 'best_map_width_regressor.pth'
        if default_model.exists():
            model_path = default_model
        else:
            print(f" Modèle introuvable. Spécifiez --model <chemin>")
            sys.exit(1)
    
    if not model_path.exists():
        print(f" Modèle introuvable: {model_path}")
        sys.exit(1)
    
    #  BATCH PREDICTION 
    print("\n" + "="*70)
    print("   BATCH PREDICTIONS - MAP WIDTH")
    print("="*70 + "\n")
    
    batch_predict(
        str(input_path),
        str(model_path),
        args.output,
        device=args.device
    )


if __name__ == '__main__':
    main()
