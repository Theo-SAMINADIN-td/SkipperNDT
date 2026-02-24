"""
Analyse du dataset pour la Tâche 2 - Régression de largeur magnétique
"""

import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from map_width_regressor import estimate_magnetic_width


def main():
    DATA_DIR = '../Training_database_float16'
    PIXEL_SIZE = 0.2

    all_files = glob.glob(os.path.join(DATA_DIR, '*.npz'))
    pipe_files = [f for f in all_files if 'no_pipe' not in os.path.basename(f)]

    print(f"Total files       : {len(all_files)}")
    print(f"Files with pipe   : {len(pipe_files)}")
    print("\nEstimating widths via FWHM...")

    widths, valid_files, skipped = [], [], 0

    for f in tqdm(pipe_files, desc="Analysing"):
        try:
            data = np.load(f, allow_pickle=True)
            img  = data['data'].astype(np.float32)
            img  = np.nan_to_num(img)
            w    = estimate_magnetic_width(img, PIXEL_SIZE)
            if w is not None:
                widths.append(w)
                valid_files.append(f)
            else:
                skipped += 1
        except Exception:
            skipped += 1

    widths = np.array(widths)
    print(f"\n{'='*55}")
    print("DATASET ANALYSIS - TASK 2")
    print(f"{'='*55}")
    print(f"Usable samples    : {len(valid_files)}  (skipped: {skipped})")
    print(f"Width range       : {widths.min():.1f}m – {widths.max():.1f}m")
    print(f"Mean ± std        : {widths.mean():.1f}m ± {widths.std():.1f}m")
    print(f"Median            : {np.median(widths):.1f}m")
    print(f"Min 500 samples   : {'✓' if len(valid_files) >= 500 else '✗'}")

    bins = [(5, 20), (20, 40), (40, 60), (60, 80)]
    print("\nDistribution par tranche:")
    for lo, hi in bins:
        n = np.sum((widths >= lo) & (widths < hi))
        print(f"  [{lo:2d}–{hi:2d}m]: {n:4d} samples ({n/len(widths)*100:.1f}%)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(widths, bins=30, edgecolor='black', color='steelblue')
    axes[0].axvline(widths.mean(),   color='red',    linestyle='--', label=f'Mean {widths.mean():.1f}m')
    axes[0].axvline(np.median(widths), color='green', linestyle='--', label=f'Median {np.median(widths):.1f}m')
    axes[0].set_title('Distribution des largeurs magnétiques')
    axes[0].set_xlabel('Width (m)')
    axes[0].set_ylabel('Count')
    axes[0].legend(); axes[0].grid(True)

    axes[1].boxplot(widths, vert=True)
    axes[1].set_title('Boxplot des largeurs')
    axes[1].set_ylabel('Width (m)')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('dataset_distribution_task2.png', dpi=150)
    print("\n✓ Plot saved to 'dataset_distribution_task2.png'")
    plt.show()


if __name__ == "__main__":
    main()
