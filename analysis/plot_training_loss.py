"""
plot_training_loss.py — Fig 2: Training Loss + λ_sym Cyclic Annealing Schedule

Single-panel figure (3.5 × 2.8 inches):
    Training loss (log scale, left y-axis) +
    λ_sym cyclic schedule (right y-axis, dashed)

Run:
    python analysis/plot_training_loss.py
"""

import os, sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from analysis.plot_style import apply_style, COLORS, panel_label, save

apply_style()

BASE     = os.path.join(os.path.dirname(__file__), '..')
LOG_PATH = os.path.join(BASE, 'checkpoints', 'train_log.csv')
OUT      = os.path.join(BASE, 'results', 'fig2_training_loss')

df     = pd.read_csv(LOG_PATH)
epochs = df['epoch'].values

fig, ax = plt.subplots(figsize=(3.5, 2.8))

# Training loss (log scale)
ax.plot(epochs, df['train_loss'].values,
        color=COLORS['text_dark'], lw=1.4, label='Train loss', zorder=3)
ax.set_yscale('log')
ax.set_xlabel('Epoch', labelpad=3)
ax.set_ylabel('Loss (log scale)', labelpad=4)
ax.set_title(r'Training Loss & $\lambda_{\mathrm{sym}}$ Schedule',
             pad=6, fontweight='bold')
ax.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.6, zorder=0)
ax.grid(False, axis='x')
ax.set_xlim(left=0)

# λ_sym cyclic schedule (right axis)
if 'lambda_sym' in df.columns:
    ax2 = ax.twinx()
    ax2.plot(epochs, df['lambda_sym'].values,
             color=COLORS['sky'], lw=1.1, linestyle='--', alpha=0.85,
             label='$\\lambda_{\\mathrm{sym}}$', zorder=2)
    ax2.set_ylabel('$\\lambda_{\\mathrm{sym}}$',
                   color=COLORS['sky'], labelpad=4)
    ax2.tick_params(axis='y', labelcolor=COLORS['sky'], labelsize=7)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylim(bottom=0)
    lines  = ax.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=7.5, loc='upper right',
              framealpha=0.9, edgecolor='#cccccc')
else:
    ax.legend(fontsize=7.5, loc='upper right')

panel_label(ax, 'a')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
save(fig, OUT)
