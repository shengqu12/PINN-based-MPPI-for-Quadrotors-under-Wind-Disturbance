"""
losses.py — PINN loss functions.

Total Loss:
    L_total = L_reg + lambda_sym * L_sym

L_reg : MSE regression loss
L_sym : C4v symmetry residual  ||f(R90·x) - R90·f(x)||^2

Physical rotation conventions for C4v (world-z rotation):
  - v_rel, wind : world-frame vectors        → rotated by R_z
  - omega       : BODY-FRAME angular velocity → UNCHANGED (invariant under world rotation)
  - quat        : body-to-world attitude      → left-multiplied by q_R  (q_new = q_R * q)
  - u (motors)  : cyclic permutation X-config (0=FR, 1=RR, 2=RL, 3=FL)
  - label y     : world-frame residual accel  → rotated by R_z

BUG fixed vs v1: the original code rotated omega as a world-frame vector.
omega is body-frame angular velocity, which is invariant under world-frame
rotations. Rotating it was enforcing a physically incorrect constraint.

PREREQUISITE: training data must be C4v-symmetric (wind collected in all
4 cardinal directions +x/-x/+y/-y).  See collection/collect_rotorpy.py.
"""

import torch
import torch.nn.functional as F
import math


# ── 1. Regression Loss ─────────────────────────────────────────────────

def loss_regression(pred, target):
    return F.mse_loss(pred, target)


# ── 2. Rotation utilities ───────────────────────────────────────────────

def _rot_z(angle_deg: float) -> torch.Tensor:
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], dtype=torch.float32)


def _quat_z(angle_deg: float) -> torch.Tensor:
    """Unit quaternion for rotation of angle_deg around z axis, [qx,qy,qz,qw]."""
    half = math.radians(angle_deg) / 2.0
    return torch.tensor([0., 0., math.sin(half), math.cos(half)], dtype=torch.float32)


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1*q2, [x,y,z,w] convention, inputs (B,4)."""
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dim=1)


def _apply_rotation(X_norm: torch.Tensor,
                    norm_mean: torch.Tensor,
                    norm_std:  torch.Tensor,
                    angle_deg: int) -> tuple:
    """
    Rotate normalised input features by angle_deg around world z axis.
    Returns (X_rot_norm, v_rel_rot_phys).

    Motor permutations — X-config quadrotor:
      layout (top view): FR(0) at (+x,-y), RR(1) at (-x,-y),
                         RL(2) at (-x,+y), FL(3) at (+x,+y)
      Under +90 CCW world rotation:
        new position 0 (FR) <- old motor 3 (FL)
        new position 1 (RR) <- old motor 0 (FR)
        ...  => perm = [3,0,1,2]
      180 deg => perm = [2,3,0,1]
      270 deg => perm = [1,2,3,0]
    """
    _PERM = {90: [3,0,1,2], 180: [2,3,0,1], 270: [1,2,3,0]}

    dev   = X_norm.device
    R     = _rot_z(angle_deg).to(dev)
    B     = X_norm.shape[0]

    # Denormalise
    X_phys = X_norm * norm_std + norm_mean

    v_rel = X_phys[:, 0:3]
    quat  = X_phys[:, 3:7]
    omega = X_phys[:, 7:10]   # body-frame — DO NOT rotate
    u     = X_phys[:, 10:14]
    wind  = X_phys[:, 14:17]

    # Rotate world-frame vectors
    v_rel_rot = (R @ v_rel.T).T
    wind_rot  = (R @ wind.T).T

    # Rotate attitude quaternion: q_new = q_R * q  (body-to-world, left-multiply)
    q_R      = _quat_z(angle_deg).to(dev).expand(B, -1)
    quat_rot = _quat_multiply(q_R, quat)
    quat_rot = quat_rot / quat_rot.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Permute motor speeds
    u_rot = u[:, _PERM[angle_deg]]

    # Renormalise
    X_rot_phys = torch.cat([v_rel_rot, quat_rot, omega, u_rot, wind_rot], dim=1)
    X_rot_norm = (X_rot_phys - norm_mean) / norm_std

    return X_rot_norm, v_rel_rot


# ── 3. C4v Symmetry Loss ───────────────────────────────────────────────

def loss_symmetry(model, X_batch, norm_mean, norm_std):
    """
    C4v symmetry constraint (fixed version):
        L_sym = (1/3) * sum_{k in {90,180,270}} MSE(f(R_k·x), R_k·f(x))

    Averages over all three non-trivial rotations for a stronger signal.

    REQUIRES: training data collected with all 4 cardinal wind directions
              (+x, -x, +y, -y) to be meaningful.
    """
    dev = X_batch.device
    total = torch.tensor(0.0, device=dev)

    # Physical v_rel for the original batch
    v_rel_orig = X_batch[:, 0:3] * norm_std[0:3] + norm_mean[0:3]
    f_orig     = model(X_batch, v_rel_orig)

    for angle in (90, 180, 270):
        R      = _rot_z(angle).to(dev)
        X_rot, v_rel_rot = _apply_rotation(X_batch, norm_mean, norm_std, angle)

        f_rot_input    = model(X_rot, v_rel_rot)
        f_orig_rotated = (R @ f_orig.T).T

        total = total + F.mse_loss(f_rot_input, f_orig_rotated)

    return total / 3.0


# ── 4. Cyclical Annealing Schedule for lambda_sym ─────────────────────
#
# Motivation (Gu et al. 2024): at high wind speeds L_sym and L_reg can
# conflict, causing gradient fights that hurt OOD performance when lambda
# is fixed.  Cyclical annealing lets the model alternate between:
#   Phase 1 (beta <= R)   : lambda = lambda_max  → enforce physics symmetry
#   Phase 2 (beta > R)    : lambda decays to 0   → focus on MSE, warm-start
#                                                    for next cycle
#
# Parameters:
#   lambda_max : peak symmetry weight (default 0.1)
#   n_cycles   : number of annealing cycles over total_epochs (default 5)
#   R          : fraction of each cycle at lambda_max (default 0.5)

def get_lambda_sym(epoch: int, total_epochs: int = 300,
                   lambda_max: float = 0.1,
                   n_cycles: int = 5,
                   R: float = 0.5) -> float:
    """Return the cyclically-annealed lambda_sym for the current epoch."""
    T_cycle = total_epochs / n_cycles
    beta    = (epoch % T_cycle) / T_cycle
    if beta <= R:
        return lambda_max
    else:
        return lambda_max * (1.0 - beta) / (1.0 - R)


# ── 6. Total Loss ──────────────────────────────────────────────────────

def total_loss(pred, target, model, X_batch,
               norm_mean, norm_std,
               lambda_sym=0.01,
               **kwargs):
    """
    L_total = L_reg + lambda_sym * L_sym

    Args:
        pred, target : (B,3) model prediction and ground truth
        model        : ResidualPINN instance
        X_batch      : (B,17) normalised input features
        norm_mean/std: normaliser parameters (on device)
        lambda_sym   : weight for symmetry loss (default 0.01)
        **kwargs     : absorbs legacy parameters for backward compatibility
    Returns:
        loss  : scalar total loss
        breakdown : dict with 'reg' and 'sym' values
    """
    l_reg = loss_regression(pred, target)
    l_sym = loss_symmetry(model, X_batch, norm_mean, norm_std)

    loss = l_reg + lambda_sym * l_sym
    return loss, {'reg': l_reg.item(), 'sym': l_sym.item()}


# ── 7. Rotation augmentation utilities (available if needed) ─────────────────

def augment_batch(X_norm, y, norm_mean, norm_std):
    """
    4x batch via 90/180/270-deg rotations (for offline data augmentation).
    Only use when training data is C4v-symmetric.
    """
    Xs, ys = [X_norm], [y]
    for angle in (90, 180, 270):
        dev = X_norm.device
        R   = _rot_z(angle).to(dev)
        Xr, _ = _apply_rotation(X_norm, norm_mean, norm_std, angle)
        yr    = (R @ y.T).T
        Xs.append(Xr); ys.append(yr)
    return torch.cat(Xs, dim=0), torch.cat(ys, dim=0)


# ── Quick test ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.pinn import ResidualPINN

    torch.manual_seed(0)
    B         = 32
    X         = torch.randn(B, 17)
    y         = torch.randn(B, 3)
    norm_mean = torch.zeros(17)
    norm_std  = torch.ones(17)
    model     = ResidualPINN(input_dim=17)

    # Test: omega is unchanged under any rotation
    for ang in (90, 180, 270):
        Xr, _ = _apply_rotation(X, norm_mean, norm_std, ang)
        err = (Xr[:, 7:10] - X[:, 7:10]).abs().max().item()
        assert err < 1e-6, f"omega changed under {ang}: err={err}"
    print("omega invariance OK")

    # Test: v_rel and wind are correctly rotated
    R90 = _rot_z(90)
    X90, _ = _apply_rotation(X, norm_mean, norm_std, 90)
    assert ((R90 @ X[:, 0:3].T).T - X90[:, 0:3]).abs().max() < 1e-6
    assert ((R90 @ X[:, 14:17].T).T - X90[:, 14:17]).abs().max() < 1e-6
    print("v_rel and wind rotation OK")

    # Test total_loss
    v_rel = X[:, 0:3]
    pred  = model(X, v_rel)
    loss, bd = total_loss(pred, y, model, X, norm_mean, norm_std, lambda_sym=0.01)
    print(f"L_reg = {bd['reg']:.4f}  L_sym = {bd['sym']:.4f}  Total = {loss.item():.4f}")
    print("All tests OK")
