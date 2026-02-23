"""
Script pour faire des prédictions en batch sur plusieurs fichiers
Usage: python batch_predict.py --input_dir <directory> [--output results.csv]
"""

import torch
import numpy as np
import argparse
import glob
import os
import csv
from tqdm import tqdm
from predict_pipeline_presence import predict


def batch_predict(model_path, input_dir, output_file='batch_results.csv', device='cpu'):
    """
    Fait des prédictions sur tous les fichiers .npz d'un dossier
    """
    # Trouver tous les fichiers .npz
    file_paths = glob.glob(os.path.join(input_dir, '*.npz'))
    
    if len(file_paths) == 0:
        print(f"❌ Aucun fichier .npz trouvé dans {input_dir}")
        return
    
    print(f"Trouvé {len(file_paths)} fichiers à analyser")
    
    # Préparer le fichier de résultats
    results = []
    
    # Prédire sur chaque fichier
    for file_path in tqdm(file_paths, desc="Processing"):
        filename = os.path.basename(file_path)
        
        try:
            probability, prediction = predict(model_path, file_path, device)
            
            results.append({
                'filename': filename,
                'prediction': prediction,
                'probability': probability,
                'confidence': max(probability, 1-probability),
                'status': 'PIPELINE DETECTED' if prediction == 1 else 'NO PIPELINE'
            })
        except Exception as e:
            print(f"⚠️  Erreur pour {filename}: {e}")
            results.append({
                'filename': filename,
                'prediction': -1,
                'probability': 0.0,
                'confidence': 0.0,
                'status': 'ERROR'
            })
    
    # Sauvegarder les résultats
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'prediction', 'probability', 'confidence', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Afficher les statistiques
    print(f"\n{'='*60}")
    print(f"BATCH PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Total files processed: {len(results)}")
    
    num_with_pipe = sum(1 for r in results if r['prediction'] == 1)
    num_no_pipe = sum(1 for r in results if r['prediction'] == 0)
    num_errors = sum(1 for r in results if r['prediction'] == -1)
    
    print(f"\nPipeline detected: {num_with_pipe} ({num_with_pipe/len(results)*100:.2f}%)")
    print(f"No pipeline: {num_no_pipe} ({num_no_pipe/len(results)*100:.2f}%)")
    
    if num_errors > 0:
        print(f"Errors: {num_errors}")
    
    # Moyenne de confiance
    avg_confidence = sum(r['confidence'] for r in results if r['prediction'] != -1) / (len(results) - num_errors)
    print(f"\nAverage confidence: {avg_confidence*100:.2f}%")
    
    print(f"\n✓ Results saved to '{output_file}'")
    
    # Fichiers avec faible confiance
    low_confidence = [r for r in results if r['confidence'] < 0.7 and r['prediction'] != -1]
    if low_confidence:
        print(f"\n⚠️  Files with low confidence (<70%):")
        for r in low_confidence[:5]:  # Afficher les 5 premiers
            print(f"  - {r['filename']}: {r['confidence']*100:.2f}%")
        if len(low_confidence) > 5:
            print(f"  ... and {len(low_confidence)-5} more")


def main():
    parser = argparse.ArgumentParser(description='Batch prediction for pipeline presence')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing .npz files')
    parser.add_argument('--model', type=str, default='best_pipeline_classifier.pth',
                        help='Path to trained model')
    parser.add_argument('--output', type=str, default='batch_results.csv',
                        help='Output CSV file')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    print(f"Model: {args.model}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output}")
    print(f"Device: {args.device}")
    print()
    
    batch_predict(args.model, args.input_dir, args.output, args.device)


if __name__ == "__main__":
    main()
