"""

     ANALYZE WIDTH DATASET - Analyse exploratoire du dataset   
  Distribution des largeurs, détection d'outliers, stats       


Ce script analyse le dataset brut avant l'entraînement:
- Distribution des largeurs magnétiques
- Détection d'outliers (IQR, Z-score)
- Corrélation avec les dimensions d'image
- Visualisations statistiques
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

# Importer le dataset
from map_width_regressor import MapWidthDataset, CONFIG


# 
# ANALYSE DU DATASET
# 

class DatasetAnalyzer:
    """
    Analyser les propriétés statistiques du dataset de régression.
    """
    
    def __init__(self, npz_files):
        """
        Charger et analyser les fichiers .npz.
        
        Paramètres:
        - npz_files: liste des chemins vers les fichiers .npz
        """
        self.npz_files = npz_files
        self.widths = []
        self.shapes = []
        self.valid_files = 0
        self.invalid_files = 0
        
        print(f" Analyse de {len(npz_files)} fichiers...")
        
        for i, npz_path in enumerate(npz_files):
            try:
                npz_data = np.load(npz_path, allow_pickle=True)
                
                # Extraire magnétique
                mag_data = None
                if 'data' in npz_data.files:
                    mag_data = npz_data['data']
                else:
                    for key in npz_data.files:
                        candidate = npz_data[key]
                        if isinstance(candidate, np.ndarray) and len(candidate.shape) == 3:
                            mag_data = candidate
                            break
                
                if mag_data is None or mag_data.shape[2] != 4:
                    self.invalid_files += 1
                    continue
                
                # Extraire label
                width_label = None
                for key in ['width', 'label', 'map_width', 'target']:
                    if key in npz_data.files:
                        width_label = float(npz_data[key])
                        break
                
                if width_label is None or np.isnan(width_label) or width_label <= 0:
                    self.invalid_files += 1
                    continue
                
                self.widths.append(width_label)
                self.shapes.append((mag_data.shape[0], mag_data.shape[1]))
                self.valid_files += 1
                
                if (i + 1) % 50 == 0:
                    print(f"   {i + 1} fichiers traités")
                
            except Exception as e:
                self.invalid_files += 1
                continue
        
        self.widths = np.array(self.widths)
        self.shapes = np.array(self.shapes)
        
        print(f" {self.valid_files} fichiers valides, {self.invalid_files} invalides\n")
    
    def compute_statistics(self):
        """Calculer les statistiques principales."""
        
        if len(self.widths) == 0:
            print(" Aucun échantillon valide!")
            return None
        
        # Statistiques descriptives
        stats_dict = {
            'n_samples': int(len(self.widths)),
            'min': float(np.min(self.widths)),
            'max': float(np.max(self.widths)),
            'mean': float(np.mean(self.widths)),
            'median': float(np.median(self.widths)),
            'std': float(np.std(self.widths)),
            'var': float(np.var(self.widths)),
            'q1': float(np.percentile(self.widths, 25)),
            'q3': float(np.percentile(self.widths, 75)),
            'iqr': float(np.percentile(self.widths, 75) - np.percentile(self.widths, 25)),
            'skewness': float(stats.skew(self.widths)),
            'kurtosis': float(stats.kurtosis(self.widths)),
        }
        
        # Détection d'outliers (IQR)
        q1 = stats_dict['q1']
        q3 = stats_dict['q3']
        iqr = stats_dict['iqr']
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers_iqr = self.widths[(self.widths < lower_bound) | (self.widths > upper_bound)]
        stats_dict['outliers_iqr_count'] = int(len(outliers_iqr))
        stats_dict['outliers_iqr_pct'] = float(100 * len(outliers_iqr) / len(self.widths))
        stats_dict['outliers_iqr_bounds'] = [float(lower_bound), float(upper_bound)]
        
        # Détection d'outliers (Z-score)
        z_scores = np.abs(stats.zscore(self.widths))
        outliers_zscore = self.widths[z_scores > 3]
        stats_dict['outliers_zscore_count'] = int(len(outliers_zscore))
        stats_dict['outliers_zscore_pct'] = float(100 * len(outliers_zscore) / len(self.widths))
        
        # Bandes d'intérêt
        stats_dict['within_5_80m'] = int(np.sum((self.widths >= 5) & (self.widths <= 80)))
        stats_dict['within_5_80m_pct'] = float(100 * stats_dict['within_5_80m'] / len(self.widths))
        
        return stats_dict
    
    def detect_outliers(self):
        """Identifier les outliers détaillés."""
        
        q1 = np.percentile(self.widths, 25)
        q3 = np.percentile(self.widths, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (self.widths < lower_bound) | (self.widths > upper_bound)
        outlier_indices = np.where(outlier_mask)[0]
        outlier_values = self.widths[outlier_mask]
        
        return {
            'outlier_indices': outlier_indices.tolist(),
            'outlier_values': outlier_values.tolist(),
            'bounds': [float(lower_bound), float(upper_bound)],
        }


# 
# VISUALISATIONS
# 

def plot_width_distribution(widths, output_dir):
    """Créer un tableau de visualisations sur la distribution des largeurs."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    #  1. HISTOGRAMME 
    ax1 = fig.add_subplot(gs[0, 0])
    n, bins, patches = ax1.hist(widths, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Colorier les barres selon l'intervalle
    for i, patch in enumerate(patches):
        if 5 <= bins[i] <= 80:
            patch.set_facecolor('lightgreen')
        else:
            patch.set_facecolor('lightcoral')
    
    ax1.axvline(np.mean(widths), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(widths):.2f}m')
    ax1.axvline(np.median(widths), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(widths):.2f}m')
    ax1.set_xlabel('Largeur (mètres)', fontsize=11)
    ax1.set_ylabel('Fréquence', fontsize=11)
    ax1.set_title('Distribution des Largeurs', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    #  2. BOÎTE À MOUSTACHES 
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot(widths, vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')
    
    q1, q2, q3 = np.percentile(widths, [25, 50, 75])
    iqr = q3 - q1
    
    stats_text = f"Min: {widths.min():.2f}m\nQ1: {q1:.2f}m\nMedian: {q2:.2f}m\nQ3: {q3:.2f}m\nMax: {widths.max():.2f}m\nIQR: {iqr:.2f}m"
    ax2.text(1.3, q2, stats_text, fontsize=9, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax2.set_ylabel('Largeur (mètres)', fontsize=11)
    ax2.set_title('Box Plot des Largeurs', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    #  3. COURBE CDF 
    ax3 = fig.add_subplot(gs[0, 2])
    sorted_widths = np.sort(widths)
    cdf = np.arange(1, len(sorted_widths) + 1) / len(sorted_widths)
    
    ax3.plot(sorted_widths, cdf, linewidth=2.5, color='darkblue')
    ax3.axvline(5, color='orange', linestyle='--', linewidth=2, label='Min acceptable (5m)')
    ax3.axvline(80, color='red', linestyle='--', linewidth=2, label='Max acceptable (80m)')
    
    ax3.set_xlabel('Largeur (mètres)', fontsize=11)
    ax3.set_ylabel('Cumulative Probability', fontsize=11)
    ax3.set_title('Fonction de Distribution Cumulative', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    #  4. Q-Q PLOT (Normalité) 
    ax4 = fig.add_subplot(gs[1, 0])
    stats.probplot(widths, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Test de Normalité)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    #  5. VIOLIN PLOT 
    ax5 = fig.add_subplot(gs[1, 1])
    parts = ax5.violinplot([widths], positions=[1], showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    
    ax5.set_ylabel('Largeur (mètres)', fontsize=11)
    ax5.set_xticks([1])
    ax5.set_xticklabels(['Largeurs'])
    ax5.set_title('Violin Plot Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    #  6. STATISTIQUES TEXTUELLES 
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    n_within_range = np.sum((widths >= 5) & (widths <= 80))
    stats_text = f"""
STATISTIQUES PRINCIPALES

Échantillons: {len(widths)}
Min: {widths.min():.3f}m
Max: {widths.max():.3f}m
Moyenne: {np.mean(widths):.3f}m
Médiane: {np.median(widths):.3f}m
Std Dev: {np.std(widths):.3f}m

IQR: {q3 - q1:.3f}m
Asymétrie: {stats.skew(widths):.3f}
Kurtosis: {stats.kurtosis(widths):.3f}

[5-80m]: {n_within_range}/{len(widths)}
({100*n_within_range/len(widths):.1f}%)

Outliers IQR: {np.sum((widths < q1 - 1.5*(q3-q1)) | (widths > q3 + 1.5*(q3-q1)))}
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Analyse Complète de la Distribution des Largeurs', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    output_path = output_dir / 'width_distribution.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f" Distribution sauvegardée: {output_path}")
    plt.close()


def plot_outliers_visualization(widths, output_dir):
    """Visualiser les outliers détectés."""
    
    q1 = np.percentile(widths, 25)
    q3 = np.percentile(widths, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter avec couleur selon outlier status
    colors = ['red' if (w < lower_bound or w > upper_bound) else 'blue' for w in widths]
    ax1.scatter(range(len(widths)), sorted(widths), c=colors, alpha=0.6, s=50)
    ax1.axhline(lower_bound, color='orange', linestyle='--', linewidth=2, label='Outlier Bounds')
    ax1.axhline(upper_bound, color='orange', linestyle='--', linewidth=2)
    ax1.set_xlabel('Index (triés)', fontsize=11)
    ax1.set_ylabel('Largeur (mètres)', fontsize=11)
    ax1.set_title('Identif. Outliers - Tri par Largeur', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Histogramme avec outliers en rouge
    outlier_mask = (widths < lower_bound) | (widths > upper_bound)
    ax2.hist(widths[~outlier_mask], bins=30, alpha=0.7, label='Normal', color='blue')
    ax2.hist(widths[outlier_mask], bins=10, alpha=0.7, label='Outliers', color='red')
    ax2.axvline(lower_bound, color='orange', linestyle='--', linewidth=2)
    ax2.axvline(upper_bound, color='orange', linestyle='--', linewidth=2)
    ax2.set_xlabel('Largeur (mètres)', fontsize=11)
    ax2.set_ylabel('Fréquence', fontsize=11)
    ax2.set_title('Outliers dans Histogramme', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    output_path = output_dir / 'outliers_visualization.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f" Visualisation outliers sauvegardée: {output_path}")
    plt.close()


# 
# RAPPORT D'ANALYSE
# 

def print_analysis_report(stats_dict, outliers_info):
    """Afficher un rapport formaté."""
    
    print("\n" + "="*70)
    print("   ANALYSE DU DATASET - MAP WIDTH REGRESSION")
    print("="*70)
    
    print(f"\n STATISTIQUES PRINCIPALES:")
    print(f"   Échantillons valides: {stats_dict['n_samples']}")
    print(f"   Min:   {stats_dict['min']:.3f}m")
    print(f"   Max:   {stats_dict['max']:.3f}m")
    print(f"   Mean:  {stats_dict['mean']:.3f}m")
    print(f"   Median: {stats_dict['median']:.3f}m")
    print(f"   Std Dev: {stats_dict['std']:.3f}m")
    
    print(f"\n QUARTILES & IQR:")
    print(f"   Q1 (25%):  {stats_dict['q1']:.3f}m")
    print(f"   Q2 (50%):  {stats_dict['median']:.3f}m")
    print(f"   Q3 (75%):  {stats_dict['q3']:.3f}m")
    print(f"   IQR:       {stats_dict['iqr']:.3f}m")
    
    print(f"\n DÉTECTION D'OUTLIERS:")
    print(f"   Outliers (IQR):  {stats_dict['outliers_iqr_count']} ({stats_dict['outliers_iqr_pct']:.2f}%)")
    print(f"   Bounds (IQR):    [{stats_dict['outliers_iqr_bounds'][0]:.3f}, {stats_dict['outliers_iqr_bounds'][1]:.3f}]")
    print(f"   Outliers (Z>3):  {stats_dict['outliers_zscore_count']} ({stats_dict['outliers_zscore_pct']:.2f}%)")
    
    print(f"\n VÉRIFICATION PLAGE:")
    print(f"   Dans [5, 80]m: {stats_dict['within_5_80m']}/{stats_dict['n_samples']} ({stats_dict['within_5_80m_pct']:.2f}%)")
    
    print(f"\n FORME DE LA DISTRIBUTION:")
    print(f"   Asymétrie (Skewness): {stats_dict['skewness']:.3f}", end="")
    if abs(stats_dict['skewness']) < 0.5:
        print(" (symétrique )")
    elif stats_dict['skewness'] > 0:
        print(" (décalée à droite)")
    else:
        print(" (décalée à gauche)")
    
    print(f"   Kurtosis: {stats_dict['kurtosis']:.3f}", end="")
    if abs(stats_dict['kurtosis']) < 1:
        print(" (normale )")
    else:
        print(" (distribution atypique)")
    
    print("\n" + "="*70 + "\n")


# 
# POINT D'ENTRÉE
# 

def main():
    print("\n" + "="*70)
    print("   ANALYZE WIDTH DATASET - TASK 2")
    print("="*70 + "\n")
    
    # Chemins
    task_dir = Path(__file__).parent
    data_dir = task_dir / 'data'
    output_dir = task_dir / 'outputs'
    
    # Vérifier le dossier data
    if not data_dir.exists():
        print(f" Dossier 'data' introuvable: {data_dir}")
        print("   Créez le dossier et placez vos fichiers .npz dedans")
        return
    
    npz_files = list(data_dir.glob('**/*.npz'))
    if len(npz_files) == 0:
        print(f" Aucun fichier .npz trouvé dans {data_dir}")
        return
    
    print(f" Trouvé {len(npz_files)} fichiers .npz\n")
    
    #  ANALYSE 
    analyzer = DatasetAnalyzer(npz_files)
    
    if len(analyzer.widths) == 0:
        print(" Aucun échantillon valide!")
        return
    
    # Statistiques
    stats_dict = analyzer.compute_statistics()
    outliers_info = analyzer.detect_outliers()
    
    # Rapport
    print_analysis_report(stats_dict, outliers_info)
    
    #  VISUALISATIONS 
    output_dir.mkdir(exist_ok=True)
    
    plot_width_distribution(analyzer.widths, output_dir)
    plot_outliers_visualization(analyzer.widths, output_dir)
    
    #  EXPORT JSON 
    analysis_path = output_dir / 'dataset_analysis.json'
    with open(analysis_path, 'w') as f:
        json.dump({
            'statistics': stats_dict,
            'outliers': outliers_info,
        }, f, indent=2)
    
    print(f" Analyse complète sauvegardée: {analysis_path}")
    print(f"\n Analyse terminée!\n")


if __name__ == '__main__':
    main()
