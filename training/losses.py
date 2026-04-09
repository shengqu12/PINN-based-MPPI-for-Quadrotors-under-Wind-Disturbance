"""
losses.py — PINN loss functions.

Total Loss:
    L_total = L_reg + λ_sym * L_sym + λ_dw * L_downwash

L_reg: ||f_pred - f_target||²
L_sym: ||f(R90·x) - R90·f(x)||²
L_downwash: || violation of monotonicity between |f_rotor_z| and Σv_induced_i ||²

"""

import torch
import torch.nn.functional as F
import math
import numpy as np

# Hummingbird physical parameters
K_ETA        = 5.57e-6
ROTOR_RADIUS = 0.1
RHO          = 1.225
ROTOR_AREA   = math.pi * ROTOR_RADIUS ** 2


# ── 1. Regression Loss ──────────────────────────────────────────────────────

def loss_regression(pred, target):
    """
    MSE regression loss.
    pred, target: (B,3) residual acceleration m/s²
    """
    return F.mse_loss(pred, target)


# ── 2. C4v symmetry Loss ───────────────────────────────────────────────

def _rot90_z():
    """90° rotation matrix around Z axis"""
    return torch.tensor([
        [ 0., -1.,  0.],
        [ 1.,  0.,  0.],
        [ 0.,  0.,  1.],
    ], dtype=torch.float32)


def _quat_multiply(q1, q2):
    """Hamilton product of two quaternions q1 and q2, both in (x,y,z,w) format."""
    x1,y1,z1,w1 = q1[:,0],q1[:,1],q1[:,2],q1[:,3]
    x2,y2,z2,w2 = q2[:,0],q2[:,1],q2[:,2],q2[:,3]
    return torch.stack([
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2,
        w1*w2-x1*x2-y1*y2-z1*z2,
    ], dim=1)


def loss_symmetry(model, X_batch, normalizer_mean, normalizer_std):
    """
    C4v symmetry loss: f(R90·x) = R90·f(x)
    """
    R90 = _rot90_z().to(X_batch.device)
    B   = X_batch.shape[0]

    # inverse normalization to get physical values for symmetry transformation
    X_phys = X_batch * normalizer_std + normalizer_mean

    v_rel = X_phys[:, 0:3]
    quat  = X_phys[:, 3:7]
    omega = X_phys[:, 7:10]
    u     = X_phys[:, 10:14]
    wind  = X_phys[:, 14:17]

    # rotate the physical features
    v_rel_rot = (R90 @ v_rel.T).T
    omega_rot = (R90 @ omega.T).T
    wind_rot  = (R90 @ wind.T).T

    # quaternion rotation: q_rot * quat, where q_rot represents 90° rotation around Z axis
    q_rot    = torch.tensor(
        [0., 0., math.sin(math.pi/4), math.cos(math.pi/4)],
        device=X_batch.device
    ).expand(B, -1)
    quat_rot = _quat_multiply(q_rot, quat)

    # rotate the motor speeds
    u_rot = u[:, [3, 0, 1, 2]]


    X_rot_phys = torch.cat(
        [v_rel_rot, quat_rot, omega_rot, u_rot, wind_rot], dim=1
    )
    X_rot = (X_rot_phys - normalizer_mean) / normalizer_std

    v_rel_phys = X_phys[:, 0:3]

    f_orig         = model(X_batch, v_rel_phys)   # f(x)
    f_rot_input    = model(X_rot,   v_rel_rot)    # f(R90·x)
    f_orig_rotated = (R90 @ f_orig.T).T           # R90·f(x)

    return F.mse_loss(f_rot_input, f_orig_rotated)


# ── 3. downwash Loss ──────────────────────────────────────

def loss_downwash(model, X_batch, motor_speeds_phys):
    """
    constraint: rotor_net absolute Z output  total induced velocity from all rotors.

    """
    T_i     = K_ETA * motor_speeds_phys ** 2           # (B,4) 
    v_ind_i = torch.sqrt(torch.clamp(
        T_i / (2 * RHO * ROTOR_AREA), min=0.0
    ))                                                   # (B,4) 
    v_total = v_ind_i.sum(dim=1)                        # (B,) 


    f_rotor_z = model.forward_rotor_z(X_batch)          # (B,)
    


    idx1    = torch.randperm(len(v_total), device=X_batch.device)
    idx2    = torch.randperm(len(v_total), device=X_batch.device)

    v1, v2  = v_total[idx1], v_total[idx2]
    f1, f2  = f_rotor_z[idx1].abs(), f_rotor_z[idx2].abs()

    
    violation = F.relu(v1 - v2) * F.relu(f2 - f1)  # (B,) violation only when v1 > v2 but f1 <= f2
    return violation.mean() 


# ── total Loss ───────────────────────────────────────────────────────────

def total_loss(pred, target, model, X_batch,
               normalizer_mean, normalizer_std,
               motor_speeds_phys,
               lambda_sym=0.5, lambda_dw=0.1):
    """
    L_total = L_reg + λ_sym * L_sym + λ_dw * L_downwash

    """
    l_reg = loss_regression(pred, target)
    l_sym = loss_symmetry(model, X_batch, normalizer_mean, normalizer_std)
    l_dw  = loss_downwash(model, X_batch, motor_speeds_phys)

    loss = l_reg + lambda_sym * l_sym + lambda_dw * l_dw
    return loss, {
        'reg': l_reg.item(),
        'sym': l_sym.item(),
        'dw':  l_dw.item(),
    }


if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.pinn import ResidualPINN

    model     = ResidualPINN(input_dim=17)
    B         = 32
    X         = torch.randn(B, 17)
    v_rel     = torch.randn(B, 3)
    target    = torch.randn(B, 3)
    norm_mean = torch.zeros(17)
    norm_std  = torch.ones(17)
    u_phys    = torch.ones(B, 4) * 1788.0

    pred     = model(X, v_rel)
    loss, bd = total_loss(pred, target, model, X,
                          norm_mean, norm_std, u_phys)
    print(f"L_reg = {bd['reg']:.4f}")
    print(f"L_sym = {bd['sym']:.4f}")
    print(f"L_dw  = {bd['dw']:.4f}")
    print(f"Total = {loss.item():.4f}")
    print("Loss computation successful!")