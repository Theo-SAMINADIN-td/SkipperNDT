"""

     EVALUATE REGRESSION - Évaluation complète du modèle       
  Métriques: MAE, MSE, R² + Visualisations                    


Ce script évalue le modèle entraîné sur le test set et génère:
- Tableau des métriques (MAE, MSE, RMSE, R²)
- Scatter plot: prédictions vs réalité
- Histogramme de distribution des erreurs
- Fichier JSON avec tous les détails
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error
)

import torch
from torch.utils.data import DataLoader

# Importer le modèle et dataset
from map_width_regressor import MapWidthRegressor, MapWidthDataset, CONFIG


# 
# ÉVALUATION DU MODÈLE
# 

def evaluate_model(model_path, test_files, device='cpu'):
    """
    Évaluer le modèle sur un ensemble de test.
    
    Retourne: (predictions, targets, metrics_dict)
    """
    
    # Charger le modèle
    model = MapWidthRegressor(num_channels=4, num_output=1)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f" Modèle chargé: {model_path}")
    
    # Créer le dataset et dataloader
    test_dataset = MapWidthDataset(test_files, target_size=224)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Inférence sur le test set
    predictions = []
    targets = []
    
    print(f" Évaluation sur {len(test_dataset)} échantillons de test...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            # Clamp optionnel
            outputs = torch.clamp(outputs, CONFIG['WIDTH_MIN'], CONFIG['WIDTH_MAX'])
            
            predictions.extend(outputs.cpu().numpy().squeeze())
            targets.extend(labels.numpy().squeeze())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calcul des métriques
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    median_ae = median_absolute_error(targets, predictions)
    
    # Erreurs absolues et relatives
    abs_errors = np.abs(targets - predictions)
    rel_errors = abs_errors / targets
    
    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2_score': float(r2),
        'median_ae': float(median_ae),
        'mean_rel_error': float(np.mean(rel_errors)),
        'std_error': float(np.std(abs_errors)),
        'min_error': float(np.min(abs_errors)),
        'max_error': float(np.max(abs_errors)),
        'samples_within_05m': int(np.sum(abs_errors <= 0.5)),
        'samples_within_1m': int(np.sum(abs_errors <= 1.0)),
        'test_samples': len(targets),
    }
    
    return predictions, targets, metrics


# 
# VISUALISATIONS
# 

def plot_comprehensive_evaluation(predictions, targets, output_dir):
    """
    Créer un tableau de bord complet avec 4 visualisations.
    """
    
    errors = targets - predictions
    abs_errors = np.abs(errors)
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    #  1. SCATTER: Prédit vs Réel 
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(targets, predictions, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
    
    # Ligne de prédiction parfaite
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect Prediction')
    
    # Zones d'erreur acceptables
    ax1.fill_between([min_val, max_val], [min_val-0.5, max_val-0.5], [min_val+0.5, max_val+0.5],
                      alpha=0.1, color='green', label='±0.5m Tolerance')
    
    ax1.set_xlabel('Réel (mètres)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Prédit (mètres)', fontsize=11, fontweight='bold')
    ax1.set_title('Prédictions vs Réalité', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    #  2. RÉSIDUS 
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(targets, errors, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2.5, label='Zero Error')
    ax2.axhline(y=0.5, color='orange', linestyle=':', lw=2, label='±0.5m')
    ax2.axhline(y=-0.5, color='orange', linestyle=':', lw=2)
    
    ax2.set_xlabel('Réel (mètres)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Résidu: Réel - Prédit (m)', fontsize=11, fontweight='bold')
    ax2.set_title('Analyse des Résidus', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    #  3. HISTOGRAMME DES ERREURS 
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(errors, bins=40, alpha=0.7, color='skyblue', edgecolor='black', linewidth=1.2)
    ax3.axvline(0, color='r', linestyle='--', linewidth=2.5, label='Zero Error')
    ax3.axvline(np.mean(errors), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.3f}m')
    ax3.axvline(np.median(errors), color='orange', linestyle=':', linewidth=2, label=f'Median: {np.median(errors):.3f}m')
    
    ax3.set_xlabel('Erreur = Réel - Prédit (mètres)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Fréquence', fontsize=11, fontweight='bold')
    ax3.set_title('Distribution des Erreurs', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    #  4. BOX PLOT 
    ax4 = fig.add_subplot(gs[1, 1])
    
    bp = ax4.boxplot([abs_errors], labels=['Erreurs Absolues'], patch_artist=True,
                      widths=0.6, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    
    # Ajouter des statistiques textuelles
    q1 = np.percentile(abs_errors, 25)
    q2 = np.percentile(abs_errors, 50)
    q3 = np.percentile(abs_errors, 75)
    
    stats_text = f"Min: {abs_errors.min():.3f}m\nQ1: {q1:.3f}m\nMedian: {q2:.3f}m\nQ3: {q3:.3f}m\nMax: {abs_errors.max():.3f}m"
    ax4.text(1.35, q2, stats_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax4.set_ylabel('Erreur Absolue (mètres)', fontsize=11, fontweight='bold')
    ax4.set_title('Box Plot des Erreurs Absolues', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Évaluation Complète du Modèle de Régression', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    output_path = output_dir / 'evaluation_comprehensive.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f" Graphique complet sauvegardé: {output_path}")
    plt.close()


def plot_error_distribution(predictions, targets, output_dir):
    """Créer un histogramme dédié à la distribution des erreurs."""
    
    errors = targets - predictions
    abs_errors = np.abs(errors)
    
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)
    
    # Histogramme des erreurs signées
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(errors, bins=35, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(0, color='r', linestyle='--', linewidth=2.5, label='Erreur = 0')
    ax1.axvline(np.mean(errors), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.3f}m')
    ax1.set_xlabel('Erreur = Réel - Prédit (mètres)', fontsize=11)
    ax1.set_ylabel('Fréquence', fontsize=11)
    ax1.set_title('Distribution des Erreurs Signées', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Histogramme des erreurs absolues
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(abs_errors, bins=35, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(np.mean(abs_errors), color='b', linestyle='--', linewidth=2, label=f'Mean: {np.mean(abs_errors):.3f}m')
    ax2.axvline(0.5, color='orange', linestyle=':', linewidth=2.5, label='Tolerance (0.5m)')
    ax2.set_xlabel('Erreur Absolue (mètres)', fontsize=11)
    ax2.set_ylabel('Fréquence', fontsize=11)
    ax2.set_title('Distribution des Erreurs Absolues', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    output_path = output_dir / 'error_distribution.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f" Distribution des erreurs sauvegardée: {output_path}")
    plt.close()


# 
# RAPPORT TEXTUEL
# 

def print_evaluation_report(metrics, predictions, targets):
    """Afficher un rapport d'évaluation formaté."""
    
    errors = targets - predictions
    abs_errors = np.abs(errors)
    
    print("\n" + "="*70)
    print("   RAPPORT D'ÉVALUATION - RÉGRESSION MAP WIDTH")
    print("="*70)
    
    print(f"\n MÉTRIQUES PRINCIPALES:")
    print(f"   MAE (Mean Absolute Error):  {metrics['mae']:.4f}m {'' if metrics['mae'] < 1.0 else ''}")
    print(f"   RMSE (Root Mean Squared):   {metrics['rmse']:.4f}m")
    print(f"   MSE (Mean Squared Error):   {metrics['mse']:.4f}")
    print(f"   R² Score:                   {metrics['r2_score']:.4f}")
    
    print(f"\n STATISTIQUES D'ERREUR:")
    print(f"   Erreur médiane:             {metrics['median_ae']:.4f}m")
    print(f"   Erreur relative moyenne:    {metrics['mean_rel_error']:.4f} ({metrics['mean_rel_error']*100:.2f}%)")
    print(f"   Écart-type erreurs:         {metrics['std_error']:.4f}m")
    print(f"   Erreur min:                 {metrics['min_error']:.4f}m")
    print(f"   Erreur max:                 {metrics['max_error']:.4f}m")
    
    print(f"\n DISTRIBUTION DE L'ERREUR:")
    pct_05 = (metrics['samples_within_05m'] / metrics['test_samples']) * 100
    pct_1 = (metrics['samples_within_1m'] / metrics['test_samples']) * 100
    print(f"   {metrics['samples_within_05m']}/{metrics['test_samples']} ({pct_05:.1f}%) dans ±0.5m")
    print(f"   {metrics['samples_within_1m']}/{metrics['test_samples']} ({pct_1:.1f}%) dans ±1.0m")
    
    print(f"\n DISTRIBUTION DES PRÉDICTIONS:")
    print(f"   Min:  {predictions.min():.2f}m | Réel min: {targets.min():.2f}m")
    print(f"   Max:  {predictions.max():.2f}m | Réel max: {targets.max():.2f}m")
    print(f"   Mean: {predictions.mean():.2f}m | Réel mean: {targets.mean():.2f}m")
    print(f"   Std:  {predictions.std():.2f}m | Réel std: {targets.std():.2f}m")
    
    print("="*70 + "\n")


# 
# POINT D'ENTRÉE PRINCIPAL
# 

def main():
    print("\n" + "="*70)
    print("   EVALUATION - MAP WIDTH REGRESSOR")
    print("="*70 + "\n")
    
    # Chemins
    task_dir = Path(__file__).parent
    outputs_dir = task_dir / 'outputs'
    
    # Vérifier les fichiers essentiels
    model_path = outputs_dir / 'best_map_width_regressor.pth'
    if not model_path.exists():
        print(f" Modèle introuvable: {model_path}")
        print("   Veuillez d'abord entraîner le modèle avec map_width_regressor.py")
        return
    
    # Chercher le test set
    data_dir = task_dir / 'data'
    if not data_dir.exists():
        print(f" Dossier 'data' introuvable: {data_dir}")
        return
    
    npz_files = list(data_dir.glob('**/*.npz'))
    if len(npz_files) == 0:
        print(f" Aucun fichier .npz trouvé dans {data_dir}")
        return
    
    # Split dataset
    np.random.seed(42)
    np.random.shuffle(npz_files)
    
    n = len(npz_files)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    test_files = npz_files[n_train + n_val:]
    
    if len(test_files) == 0:
        print(f"  Pas de fichiers de test (dataset trop petit)")
        # Utiliser une partie pour le test
        test_files = npz_files[int(len(npz_files) * 0.8):]
    
    print(f" Utilisation de {len(test_files)} fichiers de test")
    
    #  ÉVALUATION 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictions, targets, metrics = evaluate_model(str(model_path), test_files, device=device)
    
    #  RAPPORT 
    print_evaluation_report(metrics, predictions, targets)
    
    #  VISUALISATIONS 
    outputs_dir.mkdir(exist_ok=True)
    plot_comprehensive_evaluation(predictions, targets, outputs_dir)
    plot_error_distribution(predictions, targets, outputs_dir)
    
    #  EXPORT JSON 
    results_json = outputs_dir / 'evaluation_results.json'
    with open(results_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f" Métriques sauvegardées: {results_json}")
    print(f"\n Évaluation complétée!\n")


if __name__ == '__main__':
    main()
