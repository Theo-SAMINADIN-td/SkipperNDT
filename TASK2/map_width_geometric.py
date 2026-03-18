"""
MAP WIDTH — SOLUTION GEOMETRIQUE V2 — TACHE 2
==============================================
Ameliorations vs V1 :
  1. PCA locale (centre 25%) -> meilleur angle sur tuyaux courbes
  2. 5 profils perpendiculaires -> mediane des FWHM (robustesse)
  3. Seuil adaptatif (30/40/50%) -> meilleur sur grands tuyaux
  4. Validation par coherence entre profils -> rejet des outliers

Resolution : 0.20 m/pixel (confirmee par prof)
Pas besoin du CSV pour predire — methode 100% geometrique
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/Users/macdekhail/Desktop/SkipperNDT/TASK2/data/Training_database_float16'
CSV_PATH = '/Users/macdekhail/Desktop/SkipperNDT/TASK2/data/Training_database_float16/pipe_detection_label.csv'

RESOLUTION_M_PER_PX = 0.20


def get_pipe_axis(ch_norm, use_center_only=True):
    """
    PCA pour trouver l'axe du tuyau.
    use_center_only=True -> PCA sur les 25% centraux seulement
    -> meilleur sur les tuyaux courbes
    """
    h, w = ch_norm.shape

    if use_center_only:
        # Garder seulement la zone centrale 50%x50%
        margin_y = h // 4
        margin_x = w // 4
        zone = ch_norm[margin_y:h-margin_y, margin_x:w-margin_x]
        offset_y, offset_x = margin_y, margin_x
    else:
        zone = ch_norm
        offset_y, offset_x = 0, 0

    threshold = np.percentile(zone[zone > 0], 85) if (zone > 0).any() else 0.5
    mask = zone > threshold
    ys, xs = np.where(mask)

    if len(xs) < 20:
        # Fallback : PCA globale
        mask = ch_norm > np.percentile(ch_norm[ch_norm > 0], 85)
        ys, xs = np.where(mask)
        offset_y, offset_x = 0, 0
        if len(xs) < 20:
            return None, None, None

    # Centre de masse (coordonnees dans l'image complete)
    cx = np.mean(xs) + offset_x
    cy = np.mean(ys) + offset_y

    # PCA
    coords = np.stack([xs - np.mean(xs), ys - np.mean(ys)], axis=1).astype(float)
    cov = np.cov(coords.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    main_vec = eigenvectors[:, np.argmax(eigenvalues)]

    return main_vec, cx, cy


def measure_fwhm_profile(ch, cx, cy, perp_vec, threshold_pct=0.5):
    """
    Mesure le FWHM sur un profil perpendiculaire au tuyau.
    threshold_pct : niveau auquel mesurer (0.5 = mi-hauteur)
    Retourne la largeur en pixels ou None.
    """
    h, w = ch.shape
    half = min(h, w) // 2
    t = np.arange(-half, half)

    xs_line = (cx + t * perp_vec[0]).astype(int)
    ys_line = (cy + t * perp_vec[1]).astype(int)
    valid = (xs_line >= 0) & (xs_line < w) & (ys_line >= 0) & (ys_line < h)

    if valid.sum() < 20:
        return None

    profile  = ch[ys_line[valid], xs_line[valid]].astype(float)
    smoothed = gaussian_filter1d(profile, sigma=3)

    p_min, p_max = smoothed.min(), smoothed.max()
    if p_max - p_min < 1e-6:
        return None

    norm  = (smoothed - p_min) / (p_max - p_min)
    above = np.where(norm > threshold_pct)[0]

    if len(above) < 2:
        return None

    return float(above[-1] - above[0])


def predict_width(npz_path):
    """
    Predit la largeur en metres pour un fichier NPZ.

    Pipeline :
    1. Canal 2 (le plus net)
    2. PCA locale (centre 25%) -> axe du tuyau
    3. 5 profils perpendiculaires le long du tuyau
    4. Pour chaque profil : seuil adaptatif (30/40/50%)
    5. Mediane des mesures valides x 0.20 m/pixel
    """
    img = np.load(npz_path, allow_pickle=True)['data'].astype(np.float32)
    ch  = img[:, :, 2].copy()
    ch  = np.nan_to_num(ch, nan=0.0)

    h, w = ch.shape
    ch_min, ch_max = ch.min(), ch.max()
    if ch_max - ch_min < 1e-6:
        return None

    ch_norm = (ch - ch_min) / (ch_max - ch_min)

    # 1. PCA locale pour trouver l'axe
    main_vec, cx, cy = get_pipe_axis(ch_norm, use_center_only=True)
    if main_vec is None:
        return None

    # Direction perpendiculaire
    perp_vec = np.array([-main_vec[1], main_vec[0]])

    # 2. 5 positions le long de l'axe du tuyau
    par_range = min(h, w) * 0.3  # 30% de la taille
    offsets   = np.linspace(-par_range, par_range, 5)

    all_widths = []

    for offset in offsets:
        # Centre de ce profil (deplace le long de l'axe)
        px = cx + offset * main_vec[0]
        py = cy + offset * main_vec[1]

        # Verifier que le centre est dans l'image
        if not (0 < px < w and 0 < py < h):
            continue

        # 3. Seuil adaptatif : tester 30%, 40%, 50%
        widths_for_thresholds = []
        for thr in [0.30, 0.40, 0.50]:
            fw = measure_fwhm_profile(ch, px, py, perp_vec, threshold_pct=thr)
            if fw is not None and fw > 3:  # minimum 3 pixels = 0.6m
                widths_for_thresholds.append(fw)

        if not widths_for_thresholds:
            continue

        # Prendre la mediane des seuils pour ce profil
        all_widths.append(np.median(widths_for_thresholds))

    if not all_widths:
        return None

    # 4. Rejeter les outliers (ecart > 50% de la mediane)
    median_w = np.median(all_widths)
    filtered = [w for w in all_widths if abs(w - median_w) / (median_w + 1e-6) < 0.5]

    if not filtered:
        filtered = all_widths

    final_px = np.median(filtered)
    return float(final_px * RESOLUTION_M_PER_PX)


def main():
    print("\n" + "="*70)
    print("   MAP WIDTH — SOLUTION GEOMETRIQUE V2 — TACHE 2")
    print("="*70 + "\n")

    out_dir = Path(__file__).parent / 'outputs'
    out_dir.mkdir(exist_ok=True)

    # Charger le CSV pour evaluation (optionnel)
    df      = pd.read_csv(CSV_PATH, sep=';')
    df_pipe = df[df['label'] == 1]
    lut     = dict(zip(df_pipe['field_file'], df_pipe['width_m']))
    meta    = df_pipe.set_index('field_file')

    npz_files = sorted(Path(DATA_DIR).glob('*.npz'))
    print(f"{len(npz_files)} fichiers NPZ | Resolution: {RESOLUTION_M_PER_PX} m/pixel\n")

    preds, targets, errors = [], [], []
    results_detail = []

    for i, npz_path in enumerate(npz_files):
        nom = npz_path.name
        if nom not in lut:
            continue
        true_w = float(lut[nom])

        try:
            pred_w = predict_width(str(npz_path))
            if pred_w is None:
                continue

            err = abs(pred_w - true_w)
            preds.append(pred_w)
            targets.append(true_w)
            errors.append(err)

            row = meta.loc[nom]
            results_detail.append({
                'file'    : nom,
                'true'    : true_w,
                'pred'    : pred_w,
                'error'   : err,
                'coverage': row['coverage_type'],
                'shape'   : row['shape'],
                'noisy'   : row['noisy'],
            })

        except Exception:
            continue

        if (i + 1) % 200 == 0:
            print(f"   {len(preds)} predictions | MAE courante: {np.mean(errors):.4f}m")

    preds   = np.array(preds)
    targets = np.array(targets)

    mae  = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2   = r2_score(targets, preds)

    print("\n" + "="*70)
    print("RESULTATS FINAUX")
    print("="*70)
    print(f"Echantillons : {len(preds)}")
    print(f"MAE  : {mae:.4f}m  {'OBJECTIF ATTEINT' if mae < 1.0 else 'OBJECTIF NON ATTEINT'}")
    print(f"RMSE : {rmse:.4f}m")
    print(f"R2   : {r2:.4f}")
    print("="*70)

    # MAE par categorie
    res = pd.DataFrame(results_detail)
    print("\nMAE par coverage_type:")
    print(res.groupby('coverage')['error'].mean().round(3).to_string())
    print("\nMAE par shape:")
    print(res.groupby('shape')['error'].mean().round(3).to_string())
    print("\nMAE par noisy:")
    print(res.groupby('noisy')['error'].mean().round(3).to_string())
    print()

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(targets, preds, alpha=0.4, s=15)
    v = [0, max(targets.max(), preds.max())]
    axes[0].plot(v, v, 'r--', lw=2, label='Parfait')
    axes[0].set_xlabel('Reel (m)'); axes[0].set_ylabel('Predit (m)')
    axes[0].set_title(f'Predictions vs Realite — MAE={mae:.3f}m R2={r2:.3f}')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    errs = np.abs(preds - targets)
    axes[1].hist(errs, bins=50, color='steelblue', edgecolor='white')
    axes[1].axvline(1.0, color='r', ls='--', lw=2, label='Objectif 1m')
    axes[1].axvline(mae, color='g', ls='--', lw=2, label=f'MAE={mae:.2f}m')
    axes[1].set_xlabel('Erreur absolue (m)'); axes[1].set_ylabel('Fichiers')
    axes[1].set_title('Distribution des erreurs')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'geometric_v2_results.png', dpi=100)
    plt.close()
    print("geometric_v2_results.png sauvegarde")

    # JSON
    results = {
        'method'       : 'geometric_pca_local_adaptive_fwhm',
        'resolution'   : RESOLUTION_M_PER_PX,
        'n_samples'    : len(preds),
        'test_metrics' : {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)},
        'objective_met': bool(mae < 1.0),
        'mae_by_coverage': res.groupby('coverage')['error'].mean().round(3).to_dict(),
        'mae_by_shape'   : res.groupby('shape')['error'].mean().round(3).to_dict(),
    }
    with open(out_dir / 'geometric_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Resultats sauvegardes dans outputs/")
    print("Tache 2 V2 terminee !\n")


if __name__ == '__main__':
    main()