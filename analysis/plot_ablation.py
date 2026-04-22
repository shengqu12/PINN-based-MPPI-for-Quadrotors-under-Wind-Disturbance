"""
plot_ablation.py — Fig 5: Ablation Study Bar Chart

Layout (7.2 × 3.4 inches, 1 × 2):
    (a) In-distribution Val RMSE per axis (wind 0–8 m/s)
    (b) OOD RMSE per axis (wind 10–15 m/s)

Style matches fig3_controller_comparison:
  - White background, no panel highlight
  - Shared legend below figure
  - Consistent panel labels and suptitle

Run:
    python analysis/plot_ablation.py
"""

import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from analysis.plot_style import apply_style, COLORS, C_X, C_Y, C_Z, C_MEAN, panel_label, save

apply_style()

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, '..', 'checkpoints', 'ablation_results.csv')
OUT        = os.path.join(SCRIPT_DIR, '..', 'results', 'ablation_bar')

# ── Load data ─────────────────────────────────────────────────────────────────
with open(CSV_PATH, newline='') as f:
    rows = list(csv.DictReader(f))

ORDER = ['full', 'no_sym']
rows_sorted = sorted(rows, key=lambda r: ORDER.index(r['name']) if r['name'] in ORDER else 99)

METHOD_LABELS = {
    'full':   'Full PINN\n(ours)',
    'no_sym': 'w/o\nSymmetry',
}

methods  = [METHOD_LABELS.get(r['name'], r['name']) for r in rows_sorted]
val_x    = [float(r['val_x'])    for r in rows_sorted]
val_y    = [float(r['val_y'])    for r in rows_sorted]
val_z    = [float(r['val_z'])    for r in rows_sorted]
val_mean = [float(r['val_mean']) for r in rows_sorted]
ood_x    = [float(r['ood_x'])    for r in rows_sorted]
ood_y    = [float(r['ood_y'])    for r in rows_sorted]
ood_z    = [float(r['ood_z'])    for r in rows_sorted]
ood_mean = [float(r['ood_mean']) for r in rows_sorted]

n     = len(methods)
BAR_W = 0.20

# ── Draw one panel ────────────────────────────────────────────────────────────
def draw_panel(ax, x_v, y_v, z_v, m_v, title, panel_letter):
    idx     = np.arange(n)
    offsets = [-1.5*BAR_W, -0.5*BAR_W, 0.5*BAR_W, 1.5*BAR_W]
    data    = [(x_v, C_X, 'X-axis'),
               (y_v, C_Y, 'Y-axis'),
               (z_v, C_Z, 'Z-axis'),
               (m_v, C_MEAN, 'Mean')]

    for (vals, col, lbl), off in zip(data, offsets):
        for i in range(n):
            is_full = (i == 0)
            ax.bar(idx[i] + off, vals[i],
                   width=BAR_W,
                   color=col,
                   alpha=1.0 if is_full else 0.55,
                   hatch='' if is_full else '//',
                   linewidth=0.5,
                   edgecolor='black' if is_full else COLORS['grid'],
                   zorder=3)

    # mean value annotation on top of Mean bar
    ymax_data = max(max(x_v), max(y_v), max(z_v), max(m_v))
    for i, mv in enumerate(m_v):
        ax.text(idx[i] + offsets[3],
                mv + ymax_data * 0.025,
                f'{mv:.3f}',
                ha='center', va='bottom', fontsize=5.8,
                color=COLORS['text_dark'], rotation=90)

    ax.set_xticks(idx)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel('RMSE (m/s²)', labelpad=4)
    ax.set_title(title, pad=6, fontweight='bold')
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_ylim(bottom=0, top=ymax_data * 1.35)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.6, zorder=0)
    ax.grid(False, axis='x')

    panel_label(ax, panel_letter)


# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_val, ax_ood) = plt.subplots(
    1, 2, figsize=(7.2, 3.4),
    gridspec_kw={'wspace': 0.34}
)

draw_panel(ax_val, val_x, val_y, val_z, val_mean,
           'In-Distribution Val RMSE\n(wind 0–8 m/s)', 'a')
draw_panel(ax_ood, ood_x, ood_y, ood_z, ood_mean,
           'Out-of-Distribution RMSE\n(wind 10–15 m/s)', 'b')

# ── Shared suptitle ───────────────────────────────────────────────────────────
# fig.suptitle('Ablation Study: Effect of C4v Symmetry Constraint',
#              fontsize=10, fontweight='bold', y=1.03)

# ── Shared legend below (matching fig3 style) ─────────────────────────────────
axis_handles = [
    mpatches.Patch(facecolor=C_X,    edgecolor='black', lw=0.5, label='X-axis'),
    mpatches.Patch(facecolor=C_Y,    edgecolor='black', lw=0.5, label='Y-axis'),
    mpatches.Patch(facecolor=C_Z,    edgecolor='black', lw=0.5, label='Z-axis'),
    mpatches.Patch(facecolor=C_MEAN, edgecolor='black', lw=0.5, label='Mean'),
    mpatches.Patch(facecolor='#888', alpha=1.00, edgecolor='black', lw=0.5,
                   label='Full model (ours)'),
    mpatches.Patch(facecolor='#888', alpha=0.55, hatch='//',
                   edgecolor=COLORS['grid'], lw=0.5, label='Ablation variant'),
]
fig.legend(handles=axis_handles, loc='lower center', ncol=6,
           bbox_to_anchor=(0.5, -0.06),
           framealpha=0.9, edgecolor='#cccccc',
           fontsize=7.5, handlelength=1.2, columnspacing=1.0)

plt.tight_layout(rect=[0, 0.04, 1, 1])

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT), exist_ok=True)
save(fig, OUT)

# ── Print summary ─────────────────────────────────────────────────────────────
print("\nAblation Summary")
print(f"{'Method':<22} | {'Val X':>6} {'Val Y':>6} {'Val Z':>6} {'Val M':>6}"
      f" | {'OOD X':>6} {'OOD Y':>6} {'OOD Z':>6} {'OOD M':>6}")
print("-" * 78)
for r in rows_sorted:
    label = METHOD_LABELS.get(r['name'], r['name']).replace('\n', ' ')
    print(f"{label:<22} | {float(r['val_x']):6.3f} {float(r['val_y']):6.3f}"
          f" {float(r['val_z']):6.3f} {float(r['val_mean']):6.3f}"
          f" | {float(r['ood_x']):6.3f} {float(r['ood_y']):6.3f}"
          f" {float(r['ood_z']):6.3f} {float(r['ood_mean']):6.3f}")
