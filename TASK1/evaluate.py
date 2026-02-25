"""
evaluate.py
───────────
Compare predictions_task1.csv (model output)  vs
         pipe_detection_label.csv (ground truth)

Focus: Pipe detected (1) vs No Pipe (0) — predicted vs real.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
PRED_CSV  = ROOT / 'predictions_task1.csv'
LABEL_CSV = ROOT / 'pipe_detection_label.csv'
OUT_DIR   = ROOT

# ── Load & merge ───────────────────────────────────────────────────────────────
print("📂 Chargement des fichiers...")
df_pred  = pd.read_csv(PRED_CSV,  sep=';')
df_label = pd.read_csv(LABEL_CSV, sep=';')

df_pred['key']  = df_pred['filename'].apply(lambda x: Path(x).name)
df_label['key'] = df_label['field_file'].apply(lambda x: Path(x).name)

df = df_pred.merge(df_label[['key', 'label']], on='key', how='inner')

if df.empty:
    raise ValueError("❌ Aucune ligne en commun entre les deux fichiers CSV.")

print(f"✅ {len(df)} fichiers appariés.\n")

# Rename for clarity
df['Prédit']  = df['prediction'].map({1: 'Pipe (1)', 0: 'No Pipe (0)'})
df['Réel']    = df['label'].map({1: 'Pipe (1)', 0: 'No Pipe (0)'})

y_true = df['label'].astype(int).values
y_pred = df['prediction'].astype(int).values

# ── Console summary ────────────────────────────────────────────────────────────
pred_pipe   = (y_pred == 1).sum()
pred_nopipe = (y_pred == 0).sum()
real_pipe   = (y_true == 1).sum()
real_nopipe = (y_true == 0).sum()

correct = (y_pred == y_true).sum()
wrong   = (y_pred != y_true).sum()

print("━"*45)
print(f"  {'':20s}  {'Prédit':>8}  {'Réel':>8}")
print("━"*45)
print(f"  {'Pipe (1)':20s}  {pred_pipe:>8}  {real_pipe:>8}")
print(f"  {'No Pipe (0)':20s}  {pred_nopipe:>8}  {real_nopipe:>8}")
print("━"*45)
print(f"  Correct : {correct} / {len(df)}  ({correct/len(df)*100:.1f}%)")
print(f"  Erreurs : {wrong}   ({wrong/len(df)*100:.1f}%)")
print("━"*45)
print()
print(classification_report(y_true, y_pred,
                             target_names=['No Pipe (0)', 'Pipe (1)']))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Prédit vs Réel : comparaison des effectifs + confusion
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Prédit vs Réel — Pipe / No Pipe", fontsize=14, fontweight='bold')

COLORS = {'Pipe (1)': '#5b9bd5', 'No Pipe (0)': '#e07b54'}
labels_order = ['Pipe (1)', 'No Pipe (0)']

# ── 1A : Barres côte à côte Prédit vs Réel ────────────────────────────────────
ax = axes[0]
x   = np.arange(len(labels_order))
w   = 0.35

pred_counts = [pred_pipe, pred_nopipe]
real_counts = [real_pipe, real_nopipe]

bars_pred = ax.bar(x - w/2, pred_counts, w, label='Prédit',
                   color=[COLORS[l] for l in labels_order],
                   edgecolor='black', linewidth=0.6, alpha=0.9)
bars_real = ax.bar(x + w/2, real_counts, w, label='Réel (ground truth)',
                   color=[COLORS[l] for l in labels_order],
                   edgecolor='black', linewidth=0.6, alpha=0.45,
                   hatch='//')

for bar, val in zip(bars_pred, pred_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar, val in zip(bars_real, real_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha='center', va='bottom', fontsize=10, color='grey')

ax.set_xticks(x)
ax.set_xticklabels(labels_order, fontsize=11)
ax.set_ylabel("Nombre de fichiers")
ax.set_title("Effectifs : Prédit vs Réel")
ax.legend()
ax.set_ylim(0, max(max(pred_counts), max(real_counts)) * 1.18)

# ── 1B : Camembert Prédit ─────────────────────────────────────────────────────
ax = axes[1]
ax.pie(pred_counts,
       labels=[f"{l}\n{v} ({v/len(df)*100:.1f}%)" for l, v in zip(labels_order, pred_counts)],
       colors=[COLORS[l] for l in labels_order],
       startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.2},
       textprops={'fontsize': 10})
ax.set_title("Prédit par le modèle", fontsize=11)

# ── 1C : Camembert Réel ───────────────────────────────────────────────────────
ax = axes[2]
ax.pie(real_counts,
       labels=[f"{l}\n{v} ({v/len(df)*100:.1f}%)" for l, v in zip(labels_order, real_counts)],
       colors=[COLORS[l] for l in labels_order],
       startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.2},
       textprops={'fontsize': 10})
ax.set_title("Réel (ground truth)", fontsize=11)

plt.tight_layout()
out1 = OUT_DIR / 'compare_counts.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"✅ {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Matrice de confusion + distribution des probabilités
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("Détail des erreurs", fontsize=13, fontweight='bold')

# 2A : Matrice de confusion
ax = axes2[0]
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['No Pipe (0)', 'Pipe (1)'])
disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
ax.set_title("Matrice de confusion")
ax.set_xlabel("Prédit")
ax.set_ylabel("Réel")

# Annoter FP / FN / VP / VN
tn, fp, fn, tp = cm.ravel()
annotations = [
    (0, 0, f"VN\n{tn}", 'green'),
    (0, 1, f"FP\n{fp}", 'red'),
    (1, 0, f"FN\n{fn}", 'red'),
    (1, 1, f"VP\n{tp}", 'green'),
]
for row, col, txt, color in annotations:
    ax.text(col + 0.38, row + 0.38, txt,
            ha='right', va='bottom', fontsize=8,
            color=color, fontweight='bold')

# 2B : Histogramme des probabilités coloré Pipe vs No Pipe
ax = axes2[1]
bins = np.linspace(0, 1, 41)
ax.hist(df.loc[df['label'] == 0, 'probability'], bins=bins,
        color='#e07b54', alpha=0.75, label='Réel: No Pipe (0)',
        edgecolor='white', linewidth=0.4)
ax.hist(df.loc[df['label'] == 1, 'probability'], bins=bins,
        color='#5b9bd5', alpha=0.75, label='Réel: Pipe (1)',
        edgecolor='white', linewidth=0.4)
ax.axvline(0.5, color='red', linestyle='--', lw=1.5, label='Seuil 0.5')
ax.set_xlabel("Probabilité prédite (Pipe)")
ax.set_ylabel("Nombre de fichiers")
ax.set_title("Distribution des probabilités\n(coloré par vraie classe)")
ax.legend()

plt.tight_layout()
out2 = OUT_DIR / 'compare_detail.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"✅ {out2}")

print("\n🖼  Figures sauvegardées :")
print(f"   {out1.name}  ← comparaison des effectifs Prédit vs Réel")
print(f"   {out2.name}  ← matrice de confusion + distribution des probabilités")
