"""
plot_ablation.py — Publication-quality ablation study bar chart for RA-L submission.

Generates a 2-panel figure:
    (a) In-distribution Val RMSE per axis (wind 0-8 m/s)
    (b) OOD RMSE per axis (wind 10-15 m/s)

Usage:
    python analysis/plot_ablation.py
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, '..', 'checkpoints', 'ablation_results.csv')
OUT_PATH   = os.path.join(SCRIPT_DIR, '..', 'results',     'ablation_bar.pdf')
OUT_PNG    = os.path.join(SCRIPT_DIR, '..', 'results',     'ablation_bar.png')


# ── Load data ────────────────────────────────────────────────────────────────
def load_ablation(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

rows = load_ablation(CSV_PATH)

# Display order — Full model first, then ablation
ORDER = ['full', 'no_sym']
rows_sorted = sorted(rows, key=lambda r: ORDER.index(r['name'])
                     if r['name'] in ORDER else 99)

# Labels for x-axis ticks
METHOD_LABELS = {
    'full':   'Full PINN\n(ours)',
    'no_sym': 'w/o\nSymmetry',
}

methods   = [METHOD_LABELS.get(r['name'], r['name']) for r in rows_sorted]
val_x     = [float(r['val_x'])    for r in rows_sorted]
val_y     = [float(r['val_y'])    for r in rows_sorted]
val_z     = [float(r['val_z'])    for r in rows_sorted]
val_mean  = [float(r['val_mean']) for r in rows_sorted]
ood_x     = [float(r['ood_x'])    for r in rows_sorted]
ood_y     = [float(r['ood_y'])    for r in rows_sorted]
ood_z     = [float(r['ood_z'])    for r in rows_sorted]
ood_mean  = [float(r['ood_mean']) for r in rows_sorted]

n = len(methods)


# ── Style ─────────────────────────────────────────────────────────────────────
# Colorblind-safe palette (Wong 2011)
C_X    = '#0072B2'   # deep blue  — X axis
C_Y    = '#009E73'   # green      — Y axis
C_Z    = '#E69F00'   # amber      — Z axis
C_MEAN = '#CC79A7'   # pink       — Mean

ALPHA_FULL  = 1.00   # full model bars: fully opaque
ALPHA_ABL   = 0.55   # ablation bars: translucent

HATCH_FULL = ''
HATCH_ABL  = '//'    # hatching to distinguish ablation from full

BAR_W   = 0.20       # individual bar width
GROUP_W = 0.88       # total group width
EDGE    = 'white'

plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       8.5,
    'axes.titlesize':  9.5,
    'axes.labelsize':  8.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi':      200,
    'savefig.dpi':     300,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':  0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
})


# ── Helper: draw one panel ────────────────────────────────────────────────────
def draw_panel(ax, x_vals, y_vals, z_vals, mean_vals,
               ylabel, title, panel_label,
               show_legend=False, y_top_override=None):
    """
    Draws a grouped bar chart for one split (Val or OOD).

    x_vals, y_vals, z_vals, mean_vals : list of floats, length n
    """
    x = np.arange(n)

    offsets = [-1.5 * BAR_W, -0.5 * BAR_W, 0.5 * BAR_W, 1.5 * BAR_W]
    datasets = [
        (x_vals, C_X,    'X-axis'),
        (y_vals, C_Y,    'Y-axis'),
        (z_vals, C_Z,    'Z-axis'),
        (mean_vals, C_MEAN, 'Mean'),
    ]

    for (vals, color, label), offset in zip(datasets, offsets):
        for i in range(n):
            alpha   = ALPHA_FULL if i == 0 else ALPHA_ABL
            hatch   = HATCH_FULL if i == 0 else HATCH_ABL
            lw_edge = 0.5
            ax.bar(x[i] + offset, vals[i],
                   width=BAR_W,
                   color=color, alpha=alpha,
                   hatch=hatch, linewidth=lw_edge,
                   edgecolor=EDGE if i > 0 else 'black',
                   zorder=3)
        # legend proxy — only once
        if show_legend:
            ax.bar(0, 0, color=color, alpha=0.85, label=label)  # dummy for legend

    # ── Highlight the "Full model" group with a light background ──────────────
    ax.axvspan(x[0] - GROUP_W / 2, x[0] + GROUP_W / 2,
               ymin=0, ymax=1, color='#f0f4f8', zorder=1, linewidth=0)

    # ── Value annotations on top of the "Mean" bar ──────────────────────────
    for i, mv in enumerate(mean_vals):
        xpos = x[i] + offsets[3]
        ypos = mv + (max(max(z_vals), max(mean_vals)) * 0.02)
        ax.text(xpos, ypos, f'{mv:.3f}',
                ha='center', va='bottom', fontsize=6.0,
                color='#333333', rotation=90)

    # ── Formatting ─────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_title(title, pad=6, fontweight='bold')
    ax.yaxis.grid(True, linewidth=0.4, linestyle='--', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_ylim(bottom=0)

    if y_top_override:
        ax.set_ylim(top=y_top_override)
    else:
        ymax = max(max(x_vals), max(y_vals), max(z_vals), max(mean_vals))
        ax.set_ylim(top=ymax * 1.30)

    # Panel label (a), (b)
    ax.text(-0.12, 1.04, panel_label,
            transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        # deduplicate
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        # add style patches
        full_patch = mpatches.Patch(facecolor='#888', alpha=1.00,
                                    edgecolor='black', linewidth=0.5,
                                    label='Full model')
        abl_patch  = mpatches.Patch(facecolor='#888', alpha=0.55,
                                    hatch='//', edgecolor='white', linewidth=0.5,
                                    label='Ablation variant')
        leg = ax.legend(
            handles=list(seen.values()) + [full_patch, abl_patch],
            labels=list(seen.keys()) + ['Full model', 'Ablation variant'],
            loc='upper right', ncol=2,
            framealpha=0.9, edgecolor='#cccccc',
            handlelength=1.2, handleheight=0.8,
            columnspacing=0.8, labelspacing=0.3,
        )
        leg.get_frame().set_linewidth(0.5)


# ── Figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 2,
    figsize=(7.2, 3.0),          # RA-L double-column width ≈ 7 inches
    gridspec_kw={'wspace': 0.32}
)

draw_panel(
    axes[0],
    val_x, val_y, val_z, val_mean,
    ylabel='RMSE (m/s²)',
    title='(a) In-Distribution Val RMSE\n(wind 0–8 m/s)',
    panel_label='',
    show_legend=False,
    y_top_override=None,
)

draw_panel(
    axes[1],
    ood_x, ood_y, ood_z, ood_mean,
    ylabel='RMSE (m/s²)',
    title='(b) Out-of-Distribution RMSE\n(wind 10–15 m/s)',
    panel_label='',
    show_legend=True,
    y_top_override=None,
)

# ── Shared caption-style annotation ───────────────────────────────────────────
# fig.text(
#     0.5, -0.04,
#     'Ablation Study — Residual Acceleration Prediction RMSE (m/s²)\n'
#     'Bars: X (blue) / Y (green) / Z (amber) / Mean (pink). '
#     'Filled = Full model; hatched = ablation variant.',
#     ha='center', fontsize=7, color='#444444',
#     style='italic'
# )

plt.tight_layout(rect=[0, 0.00, 1, 1])

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PDF if (OUT_PDF := OUT_PATH).endswith('.pdf') else OUT_PATH,
            bbox_inches='tight', dpi=300)
fig.savefig(OUT_PNG, bbox_inches='tight', dpi=300)

print(f"Saved: {OUT_PATH}")
print(f"Saved: {OUT_PNG}")

# ── Print table summary ───────────────────────────────────────────────────────
print("\nAblation Summary Table")
print(f"{'Method':<22} | {'Val X':>6} {'Val Y':>6} {'Val Z':>6} {'Val M':>6}"
      f" | {'OOD X':>6} {'OOD Y':>6} {'OOD Z':>6} {'OOD M':>6}")
print("-" * 78)
for r in rows_sorted:
    print(f"{METHOD_LABELS.get(r['name'], r['name']).replace(chr(10),' '):22}"
          f" | {float(r['val_x']):6.3f} {float(r['val_y']):6.3f}"
          f" {float(r['val_z']):6.3f} {float(r['val_mean']):6.3f}"
          f" | {float(r['ood_x']):6.3f} {float(r['ood_y']):6.3f}"
          f" {float(r['ood_z']):6.3f} {float(r['ood_mean']):6.3f}")
