"""
dataset.py — dataset loading and normalization
normalization : the input features are in the same scale.

input features (17 dimensions):
    v_rel(3)  relative velocity = vel - wind
    quat(4)   quaternions
    omega(3)  body angular velocity
    u(4)      motor speed rad/s
    wind(3)   wind speed vector

labels (3 dimensions):
    residual acceleration = a_actual - a_nominal

when training, the dataset returns motor_speeds, which is used to calculate v_induced in losses.py.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os, glob

# Hummingbird physical parameters
# =========================================================

K_ETA        = 5.57e-6
ROTOR_RADIUS = 0.1
RHO          = 1.225
ROTOR_AREA   = np.pi * ROTOR_RADIUS ** 2

# =========================================================



# calculate the downwash induced velocity for each rotor, used in L_downwash loss
def compute_induced_velocity(motor_speeds):
    """
    BEM induced velocity (each rotor).
    v_induced_i = sqrt(T_i / (2ρA)) = sqrt(k_eta * ω_i² / (2ρA))

    Args:
        motor_speeds: (N,4) rad/s
    Returns:
        (N,4) m/s
    """
    T_i = K_ETA * motor_speeds ** 2
    return np.sqrt(np.maximum(T_i / (2 * RHO * ROTOR_AREA), 0.0))

# =========================================================
def build_features(states, controls):
    """
    Build 17-dimensional input features.

    Args:
        states:   (N,16) [pos(3),vel(3),quat(4),omega(3),wind(3)]
        controls: (N,4)   motor speed rad/s
    Returns:
        X: (N,17)   input features
    """
    vel   = states[:, 3:6]
    quat  = states[:, 6:10]
    omega = states[:, 10:13]
    wind  = states[:, 13:16]
    v_rel = vel - wind   # relative velocity

    # v_rel(3) | quat(4) | omega(3) | u(4) | wind(3) = 17 dimensions
    return np.concatenate([v_rel, quat, omega, controls, wind], axis=1)


class ResidualDataset(Dataset):
    """
    residual dataset for PINN training, based on the collected data from collect_rotorpy.py.    

    each sample returns: (X, y, wind_speed, motor_speeds)
        X:            (17,) normalized input features (v_rel, quat, omega, u, wind)
        y:            (3,)  residual acceleration labels m/s²
        wind_speed:   scalar wind speed
        motor_speeds: (4,)  motor speeds (for L_downwash calculation)
    """

    def __init__(self, data_dir, wind_max_train=8.0, split='train'):
        self.split = split
        all_X, all_y, all_ws, all_u = [], [], [], []

        for fpath in sorted(glob.glob(os.path.join(data_dir, '*.npz'))):
            d         = np.load(fpath)
            states    = d['states'].astype(np.float32)
            controls  = d['controls'].astype(np.float32)
            residuals = d['residuals'].astype(np.float32)

            wind       = states[:, 13:16]
            wind_speed = np.linalg.norm(wind, axis=1)
            is_ood     = np.mean(wind_speed) > wind_max_train

            if split == 'ood' and not is_ood:
                continue
            if split in ('train', 'val') and is_ood:
                continue

            X = build_features(states, controls)
            all_X.append(X)
            all_y.append(residuals)
            all_ws.append(wind_speed)
            all_u.append(controls)   

        assert len(all_X) > 0, f"not find the data for split='{split}', directory={data_dir}"

        X_all = np.concatenate(all_X).astype(np.float32)
        y_all = np.concatenate(all_y).astype(np.float32)
        w_all = np.concatenate(all_ws).astype(np.float32)
        u_all = np.concatenate(all_u).astype(np.float32)

        if split in ('train', 'val'):
            N   = len(X_all)
            idx = np.random.default_rng(42).permutation(N)
            cut = int(N * 0.8)
            idx = idx[:cut] if split == 'train' else idx[cut:]
            X_all, y_all, w_all, u_all = (
                X_all[idx], y_all[idx], w_all[idx], u_all[idx]
            )

        self.X            = torch.from_numpy(X_all)
        self.y            = torch.from_numpy(y_all)
        self.wind_speed   = torch.from_numpy(w_all)
        self.motor_speeds = torch.from_numpy(u_all)   

        print(f"[Dataset] split={split:5s}  sample={len(self.X):6d}  "
              f"wind speed={w_all.min():.1f}-{w_all.max():.1f} m/s")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx],
                self.wind_speed[idx], self.motor_speeds[idx])


class Normalizer:
    """normalize input features (zero-mean unit-variance)"""

    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, dataset):
        X         = dataset.X
        self.mean = X.mean(dim=0)
        self.std  = X.std(dim=0).clamp(min=1e-6)
        print(f"[Normalizer] fit completed, input dimensions={X.shape[1]}")
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def save(self, path):
        torch.save({'mean': self.mean, 'std': self.std}, path)

    def load(self, path):
        ckpt      = torch.load(path, weights_only=False)
        self.mean = ckpt['mean']
        self.std  = ckpt['std']
        return self

# =========================================================
# data split(train/val/ood)
# =========================================================

def make_dataloaders(data_dir, batch_size=512, num_workers=0,
                     wind_max_train=8.0):
    train_ds = ResidualDataset(data_dir, wind_max_train=wind_max_train,
                               split='train')
    val_ds   = ResidualDataset(data_dir, wind_max_train=8.0, split='val')
    ood_ds   = ResidualDataset(data_dir, wind_max_train=8.0, split='ood')

    normalizer = Normalizer().fit(train_ds)
    for ds in [train_ds, val_ds, ood_ds]:
        ds.X = normalizer.transform(ds.X)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    ood_loader   = DataLoader(ood_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, ood_loader, normalizer


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    train_loader, val_loader, ood_loader, norm = make_dataloaders(data_dir)
    X, y, w, u = next(iter(train_loader))
    print(f"\nBatch Verification：")
    print(f"  X shape: {X.shape}   range [{X.min():.2f}, {X.max():.2f}]")
    print(f"  y shape: {y.shape}")
    print(f"  u shape: {u.shape}")