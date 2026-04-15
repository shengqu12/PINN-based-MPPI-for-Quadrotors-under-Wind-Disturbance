"""
ablation_pinn.py — Ablation study: does the C4v symmetry loss help?

Two variants answer one question:

  Q: Does the C4v symmetry loss improve OOD generalisation?
      full   : PINN (EDC + rotor) + constant lambda_sym
      no_sym : PINN (EDC + rotor), regression only (lambda_sym = 0)

Both use the same ResidualPINN architecture (29K params).
No cyclical annealing — lambda_sym is held constant throughout training.

Normalizer: std_omx == std_omy enforced via symmetrize_xy().

Outputs:
    checkpoints/ablation_{variant}/best_model.pt
    checkpoints/ablation_results.csv
    checkpoints/ablation_table.txt
"""

import os, sys, csv, time
import torch
import torch.optim as optim
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training.dataset import make_dataloaders
from training.losses import loss_regression, total_loss
from models.pinn import ResidualPINN

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')


# ── Variant definitions ──────────────────────────────────────────────────────
#
# lambda_sym > 0 enables the C4v symmetry loss (constant throughout training).
# Both variants use ResidualPINN — only the symmetry loss is toggled.

VARIANTS = {
    'full': {
        'description': 'Full PINN (EDC + C4v sym)',
        'lambda_sym':  0.1,
        'epochs':      300,
        'patience':    40,
    },
    'no_sym': {
        'description': 'PINN w/o symmetry loss',
        'lambda_sym':  0.0,
        'epochs':      300,
        'patience':    40,
    },
}

BATCH_SIZE = 512
LR         = 1e-3
SEED       = 42


# ── Helper functions ──────────────────────────────────────────────────

def evaluate(model, loader, norm_mean, norm_std, device):
    """Per-axis RMSE (m/s^2)."""
    model.eval()
    all_pred, all_target = [], []
    with torch.no_grad():
        for X, y, _, _ in loader:
            X     = X.to(device)
            v_rel = X[:, 0:3] * norm_std[0:3] + norm_mean[0:3]
            pred  = model(X, v_rel)
            all_pred.append(pred.cpu())
            all_target.append(y)
    pred   = torch.cat(all_pred)
    target = torch.cat(all_target)
    return ((pred - target) ** 2).mean(dim=0).sqrt().numpy()


# ── Single variant training ──────────────────────────────────────────────────

def train_variant(name, cfg, device):
    lambda_sym = cfg['lambda_sym']
    print(f"\n{'='*60}")
    print(f"Variant : {name}  ({cfg['description']})")
    print(f"  model      = ResidualPINN")
    print(f"  lambda_sym = {lambda_sym}  (constant)")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ckpt_dir = os.path.join(CKPT_DIR, f'ablation_{name}')
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader, val_loader, ood_loader, normalizer = make_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, wind_max_train=8.0
    )
    normalizer.save(os.path.join(ckpt_dir, 'normalizer.pt'))
    norm_mean = normalizer.mean.to(device)
    norm_std  = normalizer.std.to(device)

    model    = ResidualPINN(input_dim=17).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )

    best_val_loss    = float('inf')
    patience_counter = 0
    t_start          = time.time()

    for epoch in range(1, cfg['epochs'] + 1):

        model.train()
        total_loss_sum = 0.0
        nb             = 0

        for X, y, _, _ in train_loader:
            X = X.to(device)
            y = y.to(device)

            v_rel = X[:, 0:3] * norm_std[0:3] + norm_mean[0:3]
            pred  = model(X, v_rel)

            if lambda_sym > 0:
                loss, _ = total_loss(
                    pred=pred, target=y,
                    model=model, X_batch=X,
                    norm_mean=norm_mean, norm_std=norm_std,
                    lambda_sym=lambda_sym,
                )
            else:
                loss = loss_regression(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_sum += loss.item()
            nb             += 1

        avg = total_loss_sum / nb

        val_rmse = evaluate(model, val_loader, norm_mean, norm_std, device)
        ood_rmse = evaluate(model, ood_loader, norm_mean, norm_std, device)
        val_mean = val_rmse.mean()
        scheduler.step(val_mean)

        if epoch <= 3 or epoch % 50 == 0:
            print(f"  Epoch {epoch:3d} | "
                  f"loss={avg:.4f} lam={lambda_sym:.4f} | "
                  f"val=[{val_rmse[0]:.3f},{val_rmse[1]:.3f},{val_rmse[2]:.3f}] | "
                  f"ood=[{ood_rmse[0]:.3f},{ood_rmse[1]:.3f},{ood_rmse[2]:.3f}]")

        if val_mean < best_val_loss:
            best_val_loss    = val_mean
            patience_counter = 0
            os.makedirs(ckpt_dir, exist_ok=True)   # ensure dir exists before save
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_rmse':    val_rmse,
                'ood_rmse':    ood_rmse,
            }, os.path.join(ckpt_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t_start
    ckpt    = torch.load(
        os.path.join(ckpt_dir, 'best_model.pt'), weights_only=False
    )
    print(f"\n  Best epoch {ckpt['epoch']}  ({elapsed/60:.1f} min)")
    print(f"  Val RMSE : {ckpt['val_rmse'].round(4)}")
    print(f"  OOD RMSE : {ckpt['ood_rmse'].round(4)}")

    return {
        'name':        name,
        'description': cfg['description'],
        'epoch':       int(ckpt['epoch']),
        'val_x':       float(ckpt['val_rmse'][0]),
        'val_y':       float(ckpt['val_rmse'][1]),
        'val_z':       float(ckpt['val_rmse'][2]),
        'val_mean':    float(ckpt['val_rmse'].mean()),
        'ood_x':       float(ckpt['ood_rmse'][0]),
        'ood_y':       float(ckpt['ood_rmse'][1]),
        'ood_z':       float(ckpt['ood_rmse'][2]),
        'ood_mean':    float(ckpt['ood_rmse'].mean()),
        'time_min':    elapsed / 60,
    }


# ── Results table ──────────────────────────────────────────────────────────

def print_table(results):
    header = (
        f"{'Method':<32} | "
        f"{'Val x':>6} {'Val y':>6} {'Val z':>6} {'Val M':>6} | "
        f"{'OOD x':>6} {'OOD y':>6} {'OOD z':>6} {'OOD M':>6}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['description']:<32} | "
            f"{r['val_x']:6.3f} {r['val_y']:6.3f} {r['val_z']:6.3f} "
            f"{r['val_mean']:6.3f} | "
            f"{r['ood_x']:6.3f} {r['ood_y']:6.3f} {r['ood_z']:6.3f} "
            f"{r['ood_mean']:6.3f}"
        )
    print(sep)
    print("\nAblation analysis:")
    full   = next((r for r in results if r['name'] == 'full'),   None)
    no_sym = next((r for r in results if r['name'] == 'no_sym'), None)
    if full and no_sym:
        delta = no_sym['ood_mean'] - full['ood_mean']
        sign  = 'sym helps (+OOD)' if delta > 0 else 'sym hurts (-OOD)'
        print(f"  C4v sym effect (OOD mean): {delta:+.3f} m/s^2  [{sign}]")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Pre-create all checkpoint directories
    os.makedirs(CKPT_DIR, exist_ok=True)
    for name in VARIANTS:
        os.makedirs(os.path.join(CKPT_DIR, f'ablation_{name}'), exist_ok=True)

    print(f"Running {len(VARIANTS)} ablation variants...\n")

    results = []
    for name, cfg in VARIANTS.items():
        result = train_variant(name, cfg, device)
        results.append(result)

    # Save CSV
    csv_path = os.path.join(CKPT_DIR, 'ablation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved: {csv_path}")

    # Print and save table
    print_table(results)

    txt_path = os.path.join(CKPT_DIR, 'ablation_table.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Ablation Experiment -- Residual Prediction RMSE (m/s^2)\n")
        f.write("Val: wind 0-8 m/s (within training distribution)\n")
        f.write("OOD: wind 10-15 m/s (outside training distribution)\n\n")
        header = (
            f"{'Method':<32} | "
            f"{'Val x':>6} {'Val y':>6} {'Val z':>6} {'Val M':>6} | "
            f"{'OOD x':>6} {'OOD y':>6} {'OOD z':>6} {'OOD M':>6}\n"
        )
        f.write(header)
        f.write("-" * len(header) + "\n")
        for r in results:
            f.write(
                f"{r['description']:<32} | "
                f"{r['val_x']:6.3f} {r['val_y']:6.3f} {r['val_z']:6.3f} "
                f"{r['val_mean']:6.3f} | "
                f"{r['ood_x']:6.3f} {r['ood_y']:6.3f} {r['ood_z']:6.3f} "
                f"{r['ood_mean']:6.3f}\n"
            )
    print(f"Table saved: {txt_path}")


if __name__ == '__main__':
    main()
