"""
Script pour visualiser les prédictions avec les images
Usage: python visualize_predictions.py --samples 6
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import random
from predict_pipeline_presence import predict, preprocess_image


def visualize_predictions(model_path, data_dir, num_samples=6, device='cpu'):
    """
    Visualise des échantillons avec leurs prédictions
    """
    # Trouver tous les fichiers
    all_files = glob.glob(f"{data_dir}/*.npz")
    
    if len(all_files) == 0:
        print(f"❌ Aucun fichier trouvé dans {data_dir}")
        return
    
    # Sélectionner des échantillons aléatoires
    samples = random.sample(all_files, min(num_samples, len(all_files)))
    
    # Créer la grille de visualisation
    rows = (num_samples + 2) // 3
    cols = min(3, num_samples)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    print(f"Analyzing {len(samples)} samples...")
    
    for idx, file_path in enumerate(samples):
        # Charger l'image
        data = np.load(file_path, allow_pickle=True)
        image = data['data']
        
        # Convertir en float32 si nécessaire
        if image.dtype == np.float16:
            image = image.astype(np.float32)
        
        # Nettoyer les NaN
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Faire la prédiction
        try:
            probability, prediction = predict(model_path, file_path, device)
            
            # Label réel du fichier
            filename = file_path.split('/')[-1]
            true_label = 0 if 'no_pipe' in filename else 1
            
            # Déterminer si la prédiction est correcte
            is_correct = (prediction == true_label)
            
            # Afficher l'image (premier canal - Bx)
            im = axes[idx].imshow(image[:, :, 0], cmap='viridis', aspect='auto')
            
            # Titre avec les informations
            title = f"{filename[:30]}...\n"
            title += f"True: {'Pipeline' if true_label == 1 else 'No pipe'} | "
            title += f"Pred: {'Pipeline' if prediction == 1 else 'No pipe'}\n"
            title += f"Prob: {probability:.3f} | Conf: {max(probability, 1-probability):.3f}"
            
            # Couleur du titre selon la prédiction
            color = 'green' if is_correct else 'red'
            axes[idx].set_title(title, fontsize=9, color=color, weight='bold')
            axes[idx].axis('off')
            
            # Ajouter une colorbar
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
            
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error: {str(e)}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
    
    # Cacher les axes non utilisés
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Pipeline Presence Predictions\nGreen=Correct, Red=Incorrect', 
                 fontsize=14, weight='bold')
    plt.tight_layout()
    
    # Sauvegarder et afficher
    output_file = 'predictions_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to '{output_file}'")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize pipeline presence predictions')
    parser.add_argument('--data_dir', type=str, 
                        default='/home/tsaminadin/Documents/HETIC/SkipperNDT/Training_database_float16',
                        help='Directory containing .npz files')
    parser.add_argument('--model', type=str, default='best_pipeline_classifier.pth',
                        help='Path to trained model')
    parser.add_argument('--samples', type=int, default=6,
                        help='Number of samples to visualize')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    visualize_predictions(args.model, args.data_dir, args.samples, args.device)


if __name__ == "__main__":
    main()
