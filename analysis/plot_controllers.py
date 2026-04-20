"""
plot_controllers.py — Fig 3: Controller Comparison (error vs wind speed, 2×2)

Layout (figure*, ~7.2 × 4.0 inches):
    2 rows × 2 cols → 4 trajectories: Hover / Circle / Lemniscate / Spiral
    Each subplot: mean position error vs wind speed (0–12 m/s)
    OOD region (10–12 m/s) shaded

Run:
    python analysis/plot_controllers.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from analysis.plot_style import (
    apply_style, COLORS, C_SE3, C_NOM, C_PINN,
    panel_label, save
)

apply_style()

BASE = os.path.join(os.path.dirname(__file__), '..')
OUT  = os.path.join(BASE, 'results', 'fig3_controller_comparison')

# ── Data (from Table III in paper) ───────────────────────────────────────────
WIND_SPEEDS = [0, 2, 4, 6, 8, 10, 12]
OOD_START   = 10   # wind ≥ this is OOD

DATA = {
    'Hover': {
        'SE3 Only':      [0.000, 0.143, 0.299, 0.477, 0.701, 1.021, 1.518],
        'Nominal MPPI':  [0.001, 0.129, 0.269, 0.427, 0.621, 0.887, 1.276],
        'PINN-MPPI':     [0.000, 0.064, 0.130, 0.193, 0.264, 0.367, 0.627],
    },
    'Circle': {
        'SE3 Only':      [0.073, 0.157, 0.316, 0.503, 0.738, 1.061, 1.538],
        'Nominal MPPI':  [0.109, 0.164, 0.298, 0.459, 0.658, 0.896, 1.261],
        'PINN-MPPI':     [0.031, 0.088, 0.142, 0.199, 0.262, 0.389, 0.656],
    },
    'Lemniscate': {
        'SE3 Only':      [0.129, 0.203, 0.348, 0.520, 0.746, 1.081, 1.601],
        'Nominal MPPI':  [0.059, 0.148, 0.286, 0.447, 0.646, 0.920, 1.311],
        'PINN-MPPI':     [0.101, 0.120, 0.164, 0.220, 0.290, 0.406, 0.685],
    },
    'Spiral': {
        'SE3 Only':      [0.124, 0.194, 0.340, 0.522, 0.753, 1.093, 1.613],
        'Nominal MPPI':  [0.043, 0.144, 0.285, 0.448, 0.650, 0.933, 1.331],
        'PINN-MPPI':     [0.094, 0.101, 0.158, 0.225, 0.300, 0.413, 0.693],
    },
}

CTRL_STYLE = {
    'SE3 Only':     dict(color=C_SE3,  lw=1.4, ls='-',  marker='o', ms=4,
                         label='SE3 Only'),
    'Nominal MPPI': dict(color=C_NOM,  lw=1.4, ls='--', marker='s', ms=4,
                         label='Nominal MPPI'),
    'PINN-MPPI':    dict(color=C_PINN, lw=1.8, ls='-',  marker='*', ms=6,
                         label='PINN-MPPI (ours)', zorder=5),
}

TRAJ_ORDER  = ['Hover', 'Circle', 'Lemniscate', 'Spiral']
PANEL_ALPHA = ['a', 'b', 'c', 'd']

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.2),
                          gridspec_kw={'hspace': 0.46, 'wspace': 0.32})
axes_flat = axes.flatten()

ood_x_min = WIND_SPEEDS.index(OOD_START)  # index where OOD begins

for idx, (tname, ax) in enumerate(zip(TRAJ_ORDER, axes_flat)):
    tdata = DATA[tname]
    winds = np.array(WIND_SPEEDS)

    # OOD shading
    ax.axvspan(OOD_START - 1, WIND_SPEEDS[-1] + 1,
               color='#F5F5F5', alpha=0.85, zorder=0)
    ax.axvline(OOD_START - 1, color='#BBBBBB', lw=0.7, ls=':', zorder=1)
    if idx == 0:
        ax.text(OOD_START - 0.5, 1.45, 'OOD',
                fontsize=6.5, color='#999999', ha='center', va='top',
                style='italic')

    for ctrl, style in CTRL_STYLE.items():
        vals = np.array(tdata[ctrl])
        ax.plot(winds, vals, markerfacecolor='white',
                markeredgewidth=1.2, **style)

    # improvement annotation at wind = 8 m/s
    idx8    = WIND_SPEEDS.index(8)
    se3_8   = tdata['SE3 Only'][idx8]
    pinn_8  = tdata['PINN-MPPI'][idx8]
    improv  = (se3_8 - pinn_8) / se3_8 * 100
    ax.annotate(f'−{improv:.0f}%',
                xy=(8, pinn_8), xytext=(7.4, pinn_8 + 0.18),
                fontsize=6.0, color=C_PINN, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_PINN,
                                lw=0.6, mutation_scale=7))

    ax.set_xlim(-0.5, 13)
    ax.set_xticks(WIND_SPEEDS)
    ax.set_xticklabels([str(w) for w in WIND_SPEEDS], fontsize=7)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Wind speed (m/s)', labelpad=2)
    ax.set_ylabel('Mean pos. error (m)', labelpad=2)
    ax.set_title(tname, pad=4, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
    ax.grid(False, axis='x')

    panel_label(ax, PANEL_ALPHA[idx])

# shared legend below
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.04),
           framealpha=0.9, edgecolor='#cccccc',
           fontsize=8, handlelength=2.0,
           columnspacing=1.5)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
save(fig, OUT)
