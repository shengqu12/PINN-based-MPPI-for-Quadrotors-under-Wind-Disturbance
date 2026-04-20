"""
plot_style.py — Shared publication style for all figures.

Matches ablation_bar.png:
  - Serif font, clean spines, dashed gridlines
  - Wong (2011) colorblind-safe palette
  - Bold panel labels (a)(b)(c)…
  - Light panel-background shading for emphasis

Usage:
    from analysis.plot_style import apply_style, COLORS, panel_label, add_panel_bg
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Colorblind-safe palette (Wong 2011) ──────────────────────────────────────
COLORS = {
    # axes / data series
    'blue':       '#0072B2',   # X-axis / SE3-Only
    'green':      '#009E73',   # Y-axis / Nominal MPPI
    'amber':      '#E69F00',   # Z-axis
    'pink':       '#CC79A7',   # Mean
    'vermillion': '#D55E00',   # PINN-MPPI (ours)  ← highlight color
    'sky':        '#56B4E9',   # secondary blue
    'black':      '#000000',
    # backgrounds / grids
    'panel_bg':   '#EDF3FB',   # light blue-gray panel fill (same as ablation_bar)
    'grid':       '#CCCCCC',
    'text_dark':  '#333333',
}

# Controller colors (3 baselines used consistently across all figures)
C_SE3   = COLORS['blue']        # SE3-Only
C_NOM   = COLORS['amber']       # Nominal MPPI
C_PINN  = COLORS['vermillion']  # PINN-MPPI (ours)

# Per-axis colors — same palette as controller comparison (no green)
C_X    = COLORS['blue']        # deep blue
C_Y    = COLORS['sky']         # sky blue  (replaces green for palette consistency)
C_Z    = COLORS['amber']       # amber
C_MEAN = COLORS['vermillion']  # vermillion (same highlight as PINN-MPPI)


def apply_style():
    """Call once at the top of every plotting script."""
    plt.rcParams.update({
        'font.family':          'serif',
        'font.serif':           ['Times New Roman', 'DejaVu Serif'],
        'font.size':            8.5,
        'axes.titlesize':       9.5,
        'axes.labelsize':       8.5,
        'xtick.labelsize':      8.0,
        'ytick.labelsize':      8.0,
        'legend.fontsize':      7.5,
        'legend.framealpha':    0.90,
        'legend.edgecolor':     '#cccccc',
        'legend.handlelength':  1.4,
        # spines
        'axes.spines.top':      False,
        'axes.spines.right':    False,
        'axes.linewidth':       0.7,
        'xtick.major.width':    0.7,
        'ytick.major.width':    0.7,
        # grid
        'axes.grid':            True,
        'grid.linestyle':       '--',
        'grid.linewidth':       0.4,
        'grid.alpha':           0.60,
        'axes.axisbelow':       True,
        # output
        'figure.dpi':           150,
        'savefig.dpi':          300,
        'savefig.bbox':         'tight',
    })


def panel_label(ax, letter, x=-0.13, y=1.05):
    """Add bold (a)/(b)/… label in the top-left corner of an axes."""
    ax.text(x, y, f'({letter})',
            transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left',
            color=COLORS['text_dark'])


def add_panel_bg(ax, color=COLORS['panel_bg']):
    """Shade the axes background (same light blue used for the 'ours' group)."""
    ax.set_facecolor(color)


def grid_only(ax, axis='y'):
    """Show dashed gridlines on one axis only (default: y)."""
    ax.grid(True, axis=axis, linestyle='--', linewidth=0.4, alpha=0.6)
    if axis == 'y':
        ax.grid(False, axis='x')
    else:
        ax.grid(False, axis='y')


def save(fig, path_no_ext):
    """Save as both PNG (300 dpi) and PDF."""
    fig.savefig(path_no_ext + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(path_no_ext + '.pdf', bbox_inches='tight')
    print(f"Saved: {path_no_ext}.png / .pdf")
