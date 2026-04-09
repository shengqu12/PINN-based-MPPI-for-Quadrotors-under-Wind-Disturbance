"""
ablation.py  ----ablation of PINN

ablation variant:
    full        : full model (ours)
    no_sym      : no symmetry constraint (lambda_sym=0)
    no_curr     : no curriculum (train on full wind range from epoch 1)
    no_downwash : no downwash constraint (lambda_dw=0)


outputs:
    checkpoints/ablation_{variant}/best_model.pt
    checkpoints/ablation_results.csv
    checkpoints/ablation_table.txt
"""

import os, sys, csv, time
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training.dataset import make_dataloaders, ResidualDataset, Normalizer
from training.losses import total_loss
from models.pinn import ResidualPINN

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')

# ── Variant definitions ──────────────────────────────────────────────────────
# All variants use 17-dim inputs, only change loss weights and curriculum switch
VARIANTS = {
    'full': {
        'description': 'Full model (ours)',
        'lambda_sym':   0.05,
        'lambda_dw':    0.0,
        'curriculum':   True,
        'epochs':       300,
        'patience':     40,
    },
    'no_sym': {
        'description': 'w/o symmetry loss',
        'lambda_sym':   0.0,   # ← no symmetry constraint
        'lambda_dw':    0.0,
        'curriculum':   True,
        'epochs':       300,
        'patience':     40,
    },
    'no_curr': {
        'description': 'w/o curriculum',
        'lambda_sym':   0.5,
        'lambda_dw':    0.0,
        'curriculum':   False,  # ← no curriculum
        'epochs':       300,
        'patience':     40,
    },

        'with_downwash': {
        'description': 'w/ downwash loss',
        'lambda_sym':  0.5,
        'lambda_dw':   0.1,
        'curriculum':  True,
        'epochs':      300,
        'patience':    40,
    },
}

CURRICULUM_SCHEDULE = [
    (1,   60,  4.0),
    (61,  300, 8.0),
]

BATCH_SIZE = 512
LR         = 1e-3
SEED       = 42


# ── Helper functions ──────────────────────────────────────────────────

def get_wind_max(epoch):
    for start, end, wmax in CURRICULUM_SCHEDULE:
        if start <= epoch <= end:
            return wmax
    return 8.0


def build_train_loader(data_dir, wind_max, normalizer, batch_size):
    """build train loader for given wind_max (curriculum stage)"""
    ds = ResidualDataset(data_dir, wind_max_train=wind_max, split='train')
    ds.X = normalizer.transform(ds.X)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def evaluate(model, loader, norm_mean, norm_std, device):
    """calculate RMSE for each axis (m/s²)."""
    model.eval()
    all_pred, all_target = [], []
    with torch.no_grad():
        for X, y, _, motor_speeds in loader:   # dataset returns (X, y, wind_speed, motor_speeds)
            X     = X.to(device)
            v_rel = X[:, 0:3] * norm_std[0:3] + norm_mean[0:3]
            pred  = model(X, v_rel)
            all_pred.append(pred.cpu())
            all_target.append(y)
    pred   = torch.cat(all_pred)
    target = torch.cat(all_target)
    return ((pred - target) ** 2).mean(dim=0).sqrt().numpy()   # (3,)


# ── Single variant training ──────────────────────────────────────────────────────

def train_variant(name, cfg, device):
    print(f"\n{'='*60}")
    print(f"Variant: {name}  ({cfg['description']})")
    print(f"  lambda_sym={cfg['lambda_sym']}  "
          f"lambda_dw={cfg['lambda_dw']}  "
          f"curriculum={cfg['curriculum']}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ckpt_dir = os.path.join(CKPT_DIR, f'ablation_{name}')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Normalizer fit on training set (wind 0-8 m/s) and save
    _, val_loader, ood_loader, normalizer = make_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, wind_max_train=8.0
    )
    normalizer.save(os.path.join(ckpt_dir, 'normalizer.pt'))
    norm_mean = normalizer.mean.to(device)
    norm_std  = normalizer.std.to(device)

    # All variants: 17-dim inputs
    model     = ResidualPINN(input_dim=17).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )

    best_val_loss    = float('inf')
    patience_counter = 0
    current_wind_max = None
    train_loader     = None
    t_start          = time.time()

    for epoch in range(1, cfg['epochs'] + 1):

        # Curriculum learning: adjust wind_max according to schedule
        if cfg['curriculum']:
            wind_max = get_wind_max(epoch)
        else:
            wind_max = 8.0   # No curriculum: use full dataset from epoch 1

        if wind_max != current_wind_max:
            current_wind_max = wind_max
            train_loader = build_train_loader(
                DATA_DIR, wind_max, normalizer, BATCH_SIZE
            )
            print(f"  [Curriculum] epoch {epoch}: wind_max={wind_max} m/s  "
                  f"n={len(train_loader.dataset)}")

        model.train()
        sums = {'reg': 0., 'sym': 0., 'dw': 0., 'total': 0.}
        nb   = 0

        for X, y, _, motor_speeds in train_loader:
            X            = X.to(device)
            y            = y.to(device)
            motor_speeds = motor_speeds.to(device)
            v_rel        = X[:, 0:3] * norm_std[0:3] + norm_mean[0:3]

            pred = model(X, v_rel)
            loss, bd = total_loss(
                pred=pred, target=y,
                model=model, X_batch=X,
                normalizer_mean=norm_mean,
                normalizer_std=norm_std,
                motor_speeds_phys=motor_speeds,
                lambda_sym=cfg['lambda_sym'],
                lambda_dw=cfg['lambda_dw'],
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            sums['reg']   += bd['reg']
            sums['sym']   += bd['sym']
            sums['dw']    += bd['dw']
            sums['total'] += loss.item()
            nb            += 1

        avg = {k: v / nb for k, v in sums.items()}

        val_rmse = evaluate(model, val_loader, norm_mean, norm_std, device)
        ood_rmse = evaluate(model, ood_loader, norm_mean, norm_std, device)
        val_mean = val_rmse.mean()
        scheduler.step(val_mean)

        if epoch <= 3 or epoch % 50 == 0:
            print(f"  Epoch {epoch:3d} | "
                  f"total={avg['total']:.4f} "
                  f"reg={avg['reg']:.4f} "
                  f"sym={avg['sym']:.4f} "
                  f"dw={avg['dw']:.4f} | "
                  f"val=[{val_rmse[0]:.3f},{val_rmse[1]:.3f},{val_rmse[2]:.3f}] | "
                  f"ood=[{ood_rmse[0]:.3f},{ood_rmse[1]:.3f},{ood_rmse[2]:.3f}]")

        if val_mean < best_val_loss:
            best_val_loss    = val_mean
            patience_counter = 0
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


# ── Results printing ──────────────────────────────────────────────────────────

def print_table(results):
    header = (
        f"{'Method':<28} | "
        f"{'Val x':>6} {'Val y':>6} {'Val z':>6} {'Val μ':>6} | "
        f"{'OOD x':>6} {'OOD y':>6} {'OOD z':>6} {'OOD μ':>6}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['description']:<28} | "
            f"{r['val_x']:6.3f} {r['val_y']:6.3f} {r['val_z']:6.3f} "
            f"{r['val_mean']:6.3f} | "
            f"{r['ood_x']:6.3f} {r['ood_y']:6.3f} {r['ood_z']:6.3f} "
            f"{r['ood_mean']:6.3f}"
        )
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Running {len(VARIANTS)} ablation variants...\n")

    results = []
    for name, cfg in VARIANTS.items():
        result = train_variant(name, cfg, device)
        results.append(result)

    # save CSV
    csv_path = os.path.join(CKPT_DIR, 'ablation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved: {csv_path}")

    # print table
    print_table(results)

    # save table text
    txt_path = os.path.join(CKPT_DIR, 'ablation_table.txt')
    with open(txt_path, 'w') as f:
        f.write("Ablation Experiment — Residual Prediction RMSE (m/s²)\n")
        f.write("Val: wind 0-8 m/s (within training distribution)\n")
        f.write("OOD: wind 10-15 m/s (outside training distribution)\n\n")
        header = (
            f"{'Method':<28} | "
            f"{'Val x':>6} {'Val y':>6} {'Val z':>6} {'Val μ':>6} | "
            f"{'OOD x':>6} {'OOD y':>6} {'OOD z':>6} {'OOD μ':>6}\n"
        )
        f.write(header)
        f.write("-" * len(header) + "\n")
        for r in results:
            f.write(
                f"{r['description']:<28} | "
                f"{r['val_x']:6.3f} {r['val_y']:6.3f} {r['val_z']:6.3f} "
                f"{r['val_mean']:6.3f} | "
                f"{r['ood_x']:6.3f} {r['ood_y']:6.3f} {r['ood_z']:6.3f} "
                f"{r['ood_mean']:6.3f}\n"
            )
    print(f"Table saved: {txt_path}")


if __name__ == '__main__':
    main()