"""
train.py — PINN training script.

Total_Loss :
    L_total = L_reg + λ_sym * L_sym + λ_dw * L_downwash

Curriculum Learning:
    epoch 1-60:   wind <= 4 m/s (learn basic aerodynamics
    epoch 61-300: wind <= 8 m/s (generalize to stronger winds)

"""

import os, sys
import torch
import torch.optim as optim
import numpy as np
import csv
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.pinn import ResidualPINN
from training.dataset import make_dataloaders, ResidualDataset
from training.losses import total_loss


# === setup ==============================================
# curriculum learning:
# - epoch 1-60:   wind <= 4 m/s (learn basic aerodynamics)
# - epoch 61-300: wind <= 8 m/s (generalize to stronger winds)

CURRICULUM = [
    (1,   60,  4.0),
    (61,  300, 8.0),
]


CFG = {
    'data_dir':   os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'),
    'ckpt_dir':   os.path.join(os.path.dirname(__file__), '..', 'checkpoints'),
    'batch_size': 512,
    'lr':         1e-3,
    'epochs':     300,
    'lambda_sym': 0.5,   # symmetry
    'lambda_dw':  0.0,   # downwash
    'patience':   40,
    'seed':       42,
}

torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"training device: {device}")




def get_wind_max(epoch):
    for start, end, wmax in CURRICULUM:
        if start <= epoch <= end:
            return wmax
    return 8.0


def build_train_loader(data_dir, wind_max, normalizer, batch_size):
    """using fixed normalizer to build train loader for different wind_max"""
    ds = ResidualDataset(data_dir, wind_max_train=wind_max, split='train')
    ds.X = normalizer.transform(ds.X)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def evaluate(model, loader, norm_mean, norm_std, device):
    """evaluate RMSE on val or OOD set"""
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

    print("initializing Normalizer...")
    _, val_loader, ood_loader, normalizer = make_dataloaders(
        CFG['data_dir'], batch_size=CFG['batch_size'], wind_max_train=8.0
    )
    normalizer.save(os.path.join(CFG['ckpt_dir'], 'normalizer.pt'))
    norm_mean = normalizer.mean.to(device)
    norm_std  = normalizer.std.to(device)
    print(f"input dimensions: {len(norm_mean)}\n")


    model = ResidualPINN(input_dim=17).to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=CFG['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )

    log_path = os.path.join(CFG['ckpt_dir'], 'train_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'epoch', 'wind_max', 'train_total', 'l_reg', 'l_sym', 'l_dw',
            'val_rmse_x', 'val_rmse_y', 'val_rmse_z',
            'ood_rmse_x', 'ood_rmse_y', 'ood_rmse_z',
        ])

    best_val_loss    = float('inf')
    patience_counter = 0
    current_wind_max = None
    train_loader     = None

    for epoch in range(1, CFG['epochs'] + 1):

        wind_max = get_wind_max(epoch)
        if wind_max != current_wind_max:
            current_wind_max = wind_max
            train_loader = build_train_loader(
                CFG['data_dir'], wind_max, normalizer, CFG['batch_size']
            )
            print(f"\n[curriculum] epoch {epoch}: wind_max={wind_max} m/s  "
                  f"sample={len(train_loader.dataset)}")

        model.train()
        sums = {'reg': 0., 'sym': 0., 'dw': 0., 'total': 0.}
        nb   = 0

        for X, y, _, motor_speeds in train_loader:   
            X            = X.to(device)
            y            = y.to(device)
            motor_speeds = motor_speeds.to(device)  

            v_rel = X[:, 0:3] * norm_std[0:3] + norm_mean[0:3]
            pred  = model(X, v_rel)

            loss, bd = total_loss(
                pred=pred, target=y,
                model=model, X_batch=X,
                normalizer_mean=norm_mean,
                normalizer_std=norm_std,
                motor_speeds_phys=motor_speeds,
                lambda_sym=CFG['lambda_sym'],
                lambda_dw=CFG['lambda_dw'],
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

        if epoch <= 5 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}/{CFG['epochs']} "
                f"[w<={wind_max:.0f}] | "
                f"total={avg['total']:.4f} "
                f"reg={avg['reg']:.4f} "
                f"sym={avg['sym']:.4f} "
                f"dw={avg['dw']:.4f} | "
                f"val=[{val_rmse[0]:.3f},{val_rmse[1]:.3f},{val_rmse[2]:.3f}] | "
                f"ood=[{ood_rmse[0]:.3f},{ood_rmse[1]:.3f},{ood_rmse[2]:.3f}]"
            )

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, wind_max, avg['total'],
                avg['reg'], avg['sym'], avg['dw'],
                val_rmse[0], val_rmse[1], val_rmse[2],
                ood_rmse[0], ood_rmse[1], ood_rmse[2],
            ])

        if val_mean < best_val_loss:
            best_val_loss    = val_mean
            patience_counter = 0
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
    print(f"Best model(epoch {ckpt['epoch']}):")
    print(f"  Val RMSE : {ckpt['val_rmse'].round(4)} m/s²")
    print(f"  OOD RMSE : {ckpt['ood_rmse'].round(4)} m/s²")


if __name__ == '__main__':
    main()