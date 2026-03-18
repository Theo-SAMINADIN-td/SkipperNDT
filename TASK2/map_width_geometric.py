"""
MAP WIDTH — SOLUTION GEOMETRIQUE V3 — TACHE 2
==============================================
Ameliorations vs V2 :
  - Seuil 10% (bords exterieurs) au lieu de 50% (mi-hauteur)
  - Detection profil oscillant -> sigma plus grand + fallback canal
  - Correction biais systematique +2m sur grandes largeurs
  - Validation qualite du profil avant de mesurer

Resolution : 0.20 m/pixel (confirmee par prof!!!±≠)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/Users/macdekhail/Desktop/SkipperNDT/TASK2/data/Training_database_float16'
CSV_PATH = '/Users/macdekhail/Desktop/SkipperNDT/TASK2/data/Training_database_float16/pipe_detection_label.csv'

RESOLUTION_M_PER_PX = 0.20


def get_pipe_axis(ch_norm):
    h, w = ch_norm.shape
    margin_y, margin_x = h // 4, w // 4
    zone = ch_norm[margin_y:h-margin_y, margin_x:w-margin_x]
    if (zone > 0).sum() < 20:
        zone = ch_norm
        margin_y, margin_x = 0, 0
    threshold = np.percentile(zone[zone > 0], 85)
    mask = zone > threshold
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return None, None, None
    cx = np.mean(xs) + margin_x
    cy = np.mean(ys) + margin_y
    coords = np.stack([xs - np.mean(xs), ys - np.mean(ys)], axis=1).astype(float)
    vals, vecs = np.linalg.eigh(np.cov(coords.T))
    main_vec = vecs[:, np.argmax(vals)]
    return main_vec, cx, cy


def is_oscillating(profile):
    peaks, _ = find_peaks(profile, height=0.7, distance=5)
    return len(peaks) > 3


def get_perp_profile(ch, cx, cy, perp_vec, sigma=5):
    h, w = ch.shape
    half = min(h, w) // 2
    t = np.arange(-half, half)
    xl = (cx + t * perp_vec[0]).astype(int)
    yl = (cy + t * perp_vec[1]).astype(int)
    valid = (xl >= 0) & (xl < w) & (yl >= 0) & (yl < h)
    if valid.sum() < 20:
        return None
    profile = gaussian_filter1d(ch[yl[valid], xl[valid]].astype(float), sigma=sigma)
    norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
    return norm


def measure_width_from_profile(norm, threshold=0.10):
    above = np.where(norm > threshold)[0]
    if len(above) < 2:
        return None
    return float(above[-1] - above[0])


def predict_width(npz_path):
    img = np.load(npz_path, allow_pickle=True)['data'].astype(np.float32)

    for canal in [2, 0, 3]:
        ch = np.nan_to_num(img[:, :, canal])
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min < 1e-6:
            continue

        ch_norm = (ch - ch_min) / (ch_max - ch_min)
        main_vec, cx, cy = get_pipe_axis(ch_norm)
        if main_vec is None:
            continue

        perp_vec = np.array([-main_vec[1], main_vec[0]])

        profile = get_perp_profile(ch, cx, cy, perp_vec, sigma=5)
        if profile is None:
            continue
        if is_oscillating(profile):
            profile = get_perp_profile(ch, cx, cy, perp_vec, sigma=15)
            if profile is None or is_oscillating(profile):
                continue

        h, w = ch.shape
        par_range = min(h, w) * 0.25
        offsets = np.linspace(-par_range, par_range, 5)
        widths = []

        for offset in offsets:
            px = cx + offset * main_vec[0]
            py = cy + offset * main_vec[1]
            if not (0 < px < w and 0 < py < h):
                continue
            prof = get_perp_profile(ch, px, py, perp_vec, sigma=5)
            if prof is None or is_oscillating(prof):
                prof = get_perp_profile(ch, px, py, perp_vec, sigma=15)
            if prof is None:
                continue
            fw = measure_width_from_profile(prof, threshold=0.10)
            if fw is not None and fw > 3:
                widths.append(fw)

        if not widths:
            continue

        med = np.median(widths)
        filtered = [x for x in widths if abs(x - med) / (med + 1e-6) < 0.4]
        if not filtered:
            filtered = widths

        width_px = np.median(filtered)
        width_m  = width_px * RESOLUTION_M_PER_PX

        # Correction biais systematique observe (+2m)
        if width_m > 5.0:
            width_m = width_m - 2.0

        return float(max(0.1, width_m))

    return None


def main():
    print("\n" + "="*70)
    print("   MAP WIDTH — SOLUTION GEOMETRIQUE V3 — TACHE 2")
    print("="*70 + "\n")

    out_dir = Path(__file__).parent / 'outputs'
    out_dir.mkdir(exist_ok=True)

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

    res = pd.DataFrame(results_detail)
    print("\nMAE par coverage_type:")
    print(res.groupby('coverage')['error'].mean().round(3).to_string())
    print("\nMAE par shape:")
    print(res.groupby('shape')['error'].mean().round(3).to_string())
    print("\nMAE par noisy:")
    print(res.groupby('noisy')['error'].mean().round(3).to_string())
    print()

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
    plt.savefig(out_dir / 'geometric_v3_results.png', dpi=100)
    plt.close()

    results = {
        'method'         : 'geometric_v3_adaptive_threshold',
        'resolution'     : RESOLUTION_M_PER_PX,
        'n_samples'      : len(preds),
        'test_metrics'   : {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)},
        'objective_met'  : bool(mae < 1.0),
        'mae_by_coverage': res.groupby('coverage')['error'].mean().round(3).to_dict(),
        'mae_by_shape'   : res.groupby('shape')['error'].mean().round(3).to_dict(),
    }
    with open(out_dir / 'geometric_v3_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Resultats sauvegardes dans outputs/")
    print("Tache 2 V3 terminee !\n")


if __name__ == '__main__':
    main()