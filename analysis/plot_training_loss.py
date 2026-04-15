"""
plot_training_loss.py — Training Process Visualization

Reads checkpoints/train_log.csv and generates 4 plots:
    1. Training loss + lambda_sym schedule (dual y-axis)
    2. Validation RMSE per axis (X/Y/Z)
    3. OOD RMSE per axis (X/Y/Z)
    4. Val vs OOD mean RMSE comparison

CSV columns expected:
    epoch, train_loss, lambda_sym,
    val_rmse_x, val_rmse_y, val_rmse_z,
    ood_rmse_x, ood_rmse_y, ood_rmse_z

Run:
    python analysis/plot_training_loss.py
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
LOG_PATH   = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'train_log.csv')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(LOG_PATH)
print(f"Read training log: {len(df)} epochs")
print(f"Columns: {df.columns.tolist()}")

val_mean = (df['val_rmse_x'] + df['val_rmse_y'] + df['val_rmse_z']) / 3
ood_mean = (df['ood_rmse_x'] + df['ood_rmse_y'] + df['ood_rmse_z']) / 3

# ── Colors ────────────────────────────────────────────────────────────────────
C_LOSS = '#2c3e50'
C_LAM  = '#3498db'
C_X    = '#e74c3c'
C_Y    = '#3498db'
C_Z    = '#f39c12'
C_VAL  = '#8e44ad'
C_OOD  = '#e74c3c'

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PINN Training Curves (ResidualPINN, L_reg + λ·L_sym)',
             fontsize=13, y=1.01)

# ── Plot 1: Loss + lambda schedule ───────────────────────────────────────────

ax = axes[0, 0]
ax.plot(df['epoch'], df['train_loss'], color=C_LOSS, lw=1.5, label='Train loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log scale)', color=C_LOSS)
ax.set_yscale('log')
ax.tick_params(axis='y', labelcolor=C_LOSS)
ax.grid(True, alpha=0.3)
ax.set_title('Training Loss & λ_sym Schedule')

# lambda_sym on right y-axis (if column exists)
if 'lambda_sym' in df.columns:
    ax2 = ax.twinx()
    ax2.plot(df['epoch'], df['lambda_sym'], color=C_LAM, lw=1.2,
             linestyle='--', alpha=0.7, label='λ_sym')
    ax2.set_ylabel('λ_sym', color=C_LAM)
    ax2.tick_params(axis='y', labelcolor=C_LAM)
    ax2.set_ylim(bottom=0)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
else:
    ax.legend(fontsize=9)

# ── Plot 2: Validation RMSE per axis ─────────────────────────────────────────
ax = axes[0, 1]
ax.plot(df['epoch'], df['val_rmse_x'], color=C_X, lw=1.2, label='Val X')
ax.plot(df['epoch'], df['val_rmse_y'], color=C_Y, lw=1.2, label='Val Y')
ax.plot(df['epoch'], df['val_rmse_z'], color=C_Z, lw=1.5, label='Val Z', linestyle='--')
ax.plot(df['epoch'], val_mean, color='black', lw=1.5, label='Val mean', linestyle=':')

best_ep  = df.loc[val_mean.idxmin(), 'epoch']
best_val = val_mean.min()
ax.scatter([best_ep], [best_val], color='black', zorder=5, s=40)
ax.text(best_ep + 2, best_val, f'ep{int(best_ep)}\n{best_val:.4f}', fontsize=8)

ax.set_xlabel('Epoch')
ax.set_ylabel('RMSE (m/s²)')
ax.set_title('Validation RMSE per Axis (wind 0–8 m/s)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# ── Plot 3: OOD RMSE per axis ────────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(df['epoch'], df['ood_rmse_x'], color=C_X, lw=1.2, label='OOD X')
ax.plot(df['epoch'], df['ood_rmse_y'], color=C_Y, lw=1.2, label='OOD Y')
ax.plot(df['epoch'], df['ood_rmse_z'], color=C_Z, lw=1.5, label='OOD Z', linestyle='--')
ax.plot(df['epoch'], ood_mean, color='black', lw=1.5, label='OOD mean', linestyle=':')

best_ood_idx = ood_mean.idxmin()
best_ood_ep  = df.loc[best_ood_idx, 'epoch']
best_ood_val = ood_mean.min()
ax.scatter([best_ood_ep], [best_ood_val], color='black', zorder=5, s=40)
ax.text(best_ood_ep + 2, best_ood_val + 0.1,
        f'ep{int(best_ood_ep)}\n{best_ood_val:.3f}', fontsize=8)

ax.set_xlabel('Epoch')
ax.set_ylabel('RMSE (m/s²)')
ax.set_title('OOD RMSE per Axis (wind 10–15 m/s)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# ── Plot 4: Val vs OOD Mean RMSE ─────────────────────────────────────────────
ax = axes[1, 1]
ax.plot(df['epoch'], val_mean, color=C_VAL, lw=1.5, label='Val Mean RMSE')
ax.plot(df['epoch'], ood_mean, color=C_OOD, lw=1.5, label='OOD Mean RMSE', linestyle='--')

final     = df.iloc[-1]
final_val = (final['val_rmse_x'] + final['val_rmse_y'] + final['val_rmse_z']) / 3
final_ood = (final['ood_rmse_x'] + final['ood_rmse_y'] + final['ood_rmse_z']) / 3

ax.annotate(f'Final Val: {final_val:.4f}',
            xy=(final['epoch'], final_val),
            xytext=(final['epoch'] - 60, final_val + 0.5),
            arrowprops=dict(arrowstyle='->', color=C_VAL),
            fontsize=8, color=C_VAL)
ax.annotate(f'Final OOD: {final_ood:.3f}',
            xy=(final['epoch'], final_ood),
            xytext=(final['epoch'] - 80, final_ood + 1.0),
            arrowprops=dict(arrowstyle='->', color=C_OOD),
            fontsize=8, color=C_OOD)

ax.set_xlabel('Epoch')
ax.set_ylabel('Mean RMSE (m/s²)')
ax.set_title('Val vs OOD Mean RMSE')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()
save_path = os.path.join(RESULT_DIR, 'training_curves.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("TRAINING RESULT SUMMARY")
print(f"{'='*50}")
best_row = df.loc[val_mean.idxmin()]
print(f"Best epoch (lowest val mean) : {int(best_row['epoch'])}")
print(f"  Val  X/Y/Z : {best_row['val_rmse_x']:.4f} / "
      f"{best_row['val_rmse_y']:.4f} / {best_row['val_rmse_z']:.4f}")
print(f"  OOD  X/Y/Z : {best_row['ood_rmse_x']:.4f} / "
      f"{best_row['ood_rmse_y']:.4f} / {best_row['ood_rmse_z']:.4f}")
print(f"  Val  mean  : {val_mean.loc[best_row.name]:.4f} m/s²")
ood_at_best = (best_row['ood_rmse_x'] + best_row['ood_rmse_y'] + best_row['ood_rmse_z']) / 3
print(f"  OOD  mean  : {ood_at_best:.4f} m/s²")
print(f"  OOD/Val ratio : {ood_at_best / (val_mean.loc[best_row.name] + 1e-9):.1f}x")
