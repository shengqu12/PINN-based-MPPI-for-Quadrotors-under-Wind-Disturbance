"""
train.py — PINN training script with cyclical symmetry annealing.

Loss:
    L_total = L_reg + lambda_sym(epoch) * L_sym

lambda_sym is cyclically annealed (Gu et al. 2024 style):
  - First R fraction of each cycle : lambda stays at lambda_max
      → network learns under physics constraint
  - Remaining (1-R) fraction       : lambda decays linearly to 0
      → network focuses on MSE, warm-starts next cycle

C4v symmetry loss:
    L_sym = (1/3) sum_{k in {90,180,270}} MSE(f(R_k·x), R_k·f(x))
    omega (body-frame) is NOT rotated — invariant under world rotation.

Normalizer symmetry:
    std_omx = std_omy (and other x/y pairs) are enforced via
    Normalizer.symmetrize_xy() in dataset.py.
"""

import os, sys
import torch
import torch.optim as optim
import numpy as np
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.pinn import ResidualPINN
from training.dataset import make_dataloaders
from training.losses import loss_regression, total_loss, get_lambda_sym


CFG = {
    'data_dir':       os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'),
    'ckpt_dir':       os.path.join(os.path.dirname(__file__), '..', 'checkpoints'),
    'batch_size':     512,
    'lr':             1e-3,
    'epochs':         300,
    'patience':       40,
    'seed':           42,
    # Cyclical symmetry annealing
    'lambda_sym_max': 0.1,   # peak weight for L_sym
    'n_cycles':       5,     # annealing cycles over total epochs
    'R':              0.5,   # fraction of each cycle at lambda_max
}

torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"training device: {device}")


def evaluate(model, loader, norm_mean, norm_std, device):
    """Evaluate per-axis RMSE on val or OOD set."""
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


def main():
    os.makedirs(CFG['ckpt_dir'], exist_ok=True)

    print("Initializing dataset and normalizer...")
    train_loader, val_loader, ood_loader, normalizer = make_dataloaders(
        CFG['data_dir'], batch_size=CFG['batch_size'], wind_max_train=8.0
    )
    normalizer.save(os.path.join(CFG['ckpt_dir'], 'normalizer.pt'))
    norm_mean = normalizer.mean.to(device)
    norm_std  = normalizer.std.to(device)
    print(f"Input dimensions: {len(norm_mean)}")
    print(f"Train samples:    {len(train_loader.dataset)}")
    print(f"Sym annealing:    lambda_max={CFG['lambda_sym_max']}, "
          f"n_cycles={CFG['n_cycles']}, R={CFG['R']}\n")

    model = ResidualPINN(input_dim=17).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=CFG['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )

    log_path = os.path.join(CFG['ckpt_dir'], 'train_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'epoch', 'train_loss', 'lambda_sym',
            'val_rmse_x', 'val_rmse_y', 'val_rmse_z',
            'ood_rmse_x', 'ood_rmse_y', 'ood_rmse_z',
        ])

    best_val_loss    = float('inf')
    patience_counter = 0

    for epoch in range(1, CFG['epochs'] + 1):

        # Cyclically-annealed lambda_sym
        lambda_sym = get_lambda_sym(
            epoch,
            total_epochs=CFG['epochs'],
            lambda_max=CFG['lambda_sym_max'],
            n_cycles=CFG['n_cycles'],
            R=CFG['R'],
        )

        model.train()
        total_loss_sum = 0.0
        nb = 0

        for X, y, _, _ in train_loader:
            X = X.to(device)
            y = y.to(device)

            v_rel = X[:, 0:3] * norm_std[0:3] + norm_mean[0:3]
            pred  = model(X, v_rel)

            if lambda_sym > 0:
                loss, _ = total_loss(pred, y, model, X,
                                     norm_mean, norm_std,
                                     lambda_sym=lambda_sym)
            else:
                loss = loss_regression(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_sum += loss.item()
            nb             += 1

        avg_loss = total_loss_sum / nb

        val_rmse = evaluate(model, val_loader, norm_mean, norm_std, device)
        ood_rmse = evaluate(model, ood_loader, norm_mean, norm_std, device)
        val_mean = val_rmse.mean()
        scheduler.step(val_mean)

        if epoch <= 5 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}/{CFG['epochs']} | "
                f"loss={avg_loss:.4f} lam={lambda_sym:.4f} | "
                f"val=[{val_rmse[0]:.3f},{val_rmse[1]:.3f},{val_rmse[2]:.3f}] | "
                f"ood=[{ood_rmse[0]:.3f},{ood_rmse[1]:.3f},{ood_rmse[2]:.3f}]"
            )

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, avg_loss, round(lambda_sym, 6),
                val_rmse[0], val_rmse[1], val_rmse[2],
                ood_rmse[0], ood_rmse[1], ood_rmse[2],
            ])

        if val_mean < best_val_loss:
            best_val_loss    = val_mean
            patience_counter = 0
            os.makedirs(CFG['ckpt_dir'], exist_ok=True)
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_rmse':    val_rmse,
                'ood_rmse':    ood_rmse,
                'cfg':         CFG,
            }, os.path.join(CFG['ckpt_dir'], 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= CFG['patience']:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    ckpt = torch.load(
        os.path.join(CFG['ckpt_dir'], 'best_model.pt'), weights_only=False
    )
    model.load_state_dict(ckpt['model_state'])
    print(f"\n{'='*60}")
    print(f"Best model (epoch {ckpt['epoch']}):")
    print(f"  Val RMSE : {ckpt['val_rmse'].round(4)} m/s^2")
    print(f"  OOD RMSE : {ckpt['ood_rmse'].round(4)} m/s^2")


if __name__ == '__main__':
    main()
