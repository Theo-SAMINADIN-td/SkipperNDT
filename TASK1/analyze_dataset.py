"""
Script pour analyser le dataset et vérifier la distribution des classes
"""

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def analyze_dataset(data_dir):
    """Analyse la distribution des labels dans le dataset"""
    
    all_files = glob.glob(os.path.join(data_dir, '*.npz'))
    
    labels = []
    label_names = []
    categories = Counter()
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        # Extraire la catégorie du nom de fichier
        if 'no_pipe' in filename:
            labels.append(0)
            label_names.append('No Pipe')
            categories['no_pipe'] += 1
        elif 'perfect' in filename:
            labels.append(1)
            label_names.append('Perfect')
            categories['perfect'] += 1
        elif 'missed' in filename:
            labels.append(1)
            label_names.append('Missed')
            categories['missed'] += 1
        elif 'parallel' in filename:
            labels.append(1)
            label_names.append('Parallel')
            categories['parallel'] += 1
        elif 'sample' in filename:
            # Extraire le type du nom
            if 'no_pipe' in filename:
                labels.append(0)
                label_names.append('No Pipe')
                categories['no_pipe'] += 1
            else:
                labels.append(1)
                label_names.append('Sample with pipe')
                categories['sample'] += 1
        else:
            # Par défaut, considérer comme présence de conduite
            labels.append(1)
            label_names.append('Other')
            categories['other'] += 1
    
    return labels, label_names, categories, all_files


def plot_distribution(categories, labels):
    """Affiche la distribution des classes"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graphique 1: Distribution des catégories détaillées
    categories_sorted = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))
    axes[0].bar(categories_sorted.keys(), categories_sorted.values())
    axes[0].set_title('Distribution par catégorie')
    axes[0].set_xlabel('Catégorie')
    axes[0].set_ylabel('Nombre d\'échantillons')
    axes[0].tick_params(axis='x', rotation=45)
    
    for i, (cat, count) in enumerate(categories_sorted.items()):
        axes[0].text(i, count, str(count), ha='center', va='bottom')
    
    # Graphique 2: Distribution binaire (0 vs 1)
    class_counts = Counter(labels)
    class_labels = ['No Pipe (0)', 'With Pipe (1)']
    class_values = [class_counts[0], class_counts[1]]
    
    colors = ['#ff6b6b', '#51cf66']
    axes[1].bar(class_labels, class_values, color=colors)
    axes[1].set_title('Distribution binaire')
    axes[1].set_xlabel('Classe')
    axes[1].set_ylabel('Nombre d\'échantillons')
    
    for i, count in enumerate(class_values):
        percentage = (count / sum(class_values)) * 100
        axes[1].text(i, count, f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=150)
    print("✓ Distribution plot saved to 'dataset_distribution.png'")
    plt.show()


def main():
    data_dir = '/home/tsaminadin/Documents/HETIC/SkipperNDT/Training_database_float16'
    
    print("Analyzing dataset...")
    labels, label_names, categories, all_files = analyze_dataset(data_dir)
    
    print(f"\n{'='*60}")
    print(f"DATASET ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\nTotal files: {len(all_files)}")
    
    print(f"\nDetailed categories:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    
    class_counts = Counter(labels)
    print(f"\nBinary classification:")
    print(f"  Class 0 (No pipe): {class_counts[0]} ({class_counts[0]/len(labels)*100:.2f}%)")
    print(f"  Class 1 (With pipe): {class_counts[1]} ({class_counts[1]/len(labels)*100:.2f}%)")
    
    # Vérifier si le dataset est suffisant (minimum 500 échantillons)
    print(f"\n{'='*60}")
    print(f"DATASET REQUIREMENTS CHECK")
    print(f"{'='*60}")
    
    min_required = 500
    if len(all_files) >= min_required:
        print(f"✓ Dataset size: {len(all_files)} samples (minimum: {min_required})")
    else:
        print(f"✗ Dataset size: {len(all_files)} samples (minimum: {min_required})")
        print(f"  WARNING: Dataset is smaller than recommended!")
    
    # Vérifier l'équilibre des classes
    imbalance_ratio = max(class_counts[0], class_counts[1]) / min(class_counts[0], class_counts[1])
    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 5:
        print(f"  ⚠️  High class imbalance detected!")
        print(f"  Consider using weighted loss or data augmentation")
    elif imbalance_ratio > 2:
        print(f"  ⚠️  Moderate class imbalance detected")
        print(f"  May benefit from weighted loss")
    else:
        print(f"  ✓ Classes are relatively balanced")
    
    # Afficher la distribution
    plot_distribution(categories, labels)
    
    # Échantillons par classe
    print(f"\nSample files:")
    print(f"\nClass 0 (No pipe) examples:")
    no_pipe_files = [f for f, l in zip(all_files, labels) if l == 0][:3]
    for f in no_pipe_files:
        print(f"  - {os.path.basename(f)}")
    
    print(f"\nClass 1 (With pipe) examples:")
    with_pipe_files = [f for f, l in zip(all_files, labels) if l == 1][:3]
    for f in with_pipe_files:
        print(f"  - {os.path.basename(f)}")


if __name__ == "__main__":
    main()
