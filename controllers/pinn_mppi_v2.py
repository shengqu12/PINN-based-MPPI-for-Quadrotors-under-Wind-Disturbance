"""
pinn_mppi_v2.py — PINN-MPPI Controller (α-MPPI with full quaternion RK4 + CUDA Graph)

Architecture:
    PINN queries drag once per step at actual drone velocity (velocity-aware).
    MPPI optimizes per-axis feedforward gain α ∈ [0, 1.5]³ over H future steps.
    SE3Control executes with PINN-scaled feedforward:
        flat['x_ddot'] = traj_x_ddot + (-α_opt ⊙ r̂_ema)

Rollout model (full quaternion RK4 with embedded SE3 — same as Minarik et al.):
    SE3 position controller:
        a_cmd = -Kp*ep - Kd*ev + a_traj - α⊙r̂
        Ft    = m * ‖a_cmd + g‖,  z_des = normalize(a_cmd + g)
    SE3 attitude controller (linearized, embedded):
        q_des     = quat_from_z(z_des)
        q_err     = q_des_inv ⊗ q
        ω_cmd     = -Kr * q_err_vec * sign(q_err_w)
    RK4 dynamics:
        dp = v
        dv = (Ft/m) * R(q)@[0,0,1] - g + (1-α)⊙r̂    ← PINN wind residual
        dq = 0.5 * q ⊗ [ω_cmd; 0]

CUDA Graph acceleration:
    Noise sampling and host→device copies happen outside the graph each step.
    The deterministic rollout (RK4 × H steps × K samples) is captured once
    and replayed via a single CUDA Graph call, eliminating ~15 μs/kernel of
    Python dispatch overhead. Target: ~100 Hz on RTX 5070 (K=896, H=15).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F
import sys
import time

torch.set_float32_matmul_precision('high')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.pinn import ResidualPINN
from training.dataset import Normalizer

from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.trajectories.hover_traj import HoverTraj

CKPT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# ── physics constants ──────────────────────────────────────────────────────────
K_ETA        = quad_params['k_eta']
K_M          = quad_params['k_m']
MASS         = quad_params['mass']
IXX          = quad_params['Ixx']
IYY          = quad_params['Iyy']
IZZ          = quad_params['Izz']
G            = 9.81
OMEGA_MIN    = float(quad_params['rotor_speed_min'])
OMEGA_MAX    = float(quad_params['rotor_speed_max'])
T_MIN        = K_ETA * OMEGA_MIN ** 2
T_MAX        = K_ETA * OMEGA_MAX ** 2
HOVER_OMEGA  = float(np.sqrt(MASS * G / (4 * K_ETA)))
HOVER_THRUST = MASS * G

J_np = np.array([IXX, IYY, IZZ], dtype=np.float32)

OMEGA_XY_MAX = 10.0   # rad/s  (kept for external imports)
OMEGA_Z_MAX  =  2.0   # rad/s

# ── rotor geometry & allocation matrix ────────────────────────────────────────
_rp = quad_params['rotor_pos']
ROTOR_POS = np.array(list(_rp.values()) if isinstance(_rp, dict)
                     else _rp, dtype=np.float32)
_rd = quad_params['rotor_directions']
ROTOR_DIR = np.array(list(_rd.values()) if isinstance(_rd, dict)
                     else _rd, dtype=np.float32)


def _build_gamma():
    G_ = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        G_[0, i] = 1.0
        G_[1, i] = ROTOR_POS[i, 1]
        G_[2, i] = ROTOR_POS[i, 0]
        G_[3, i] = ROTOR_DIR[i] * (K_M / K_ETA)
    return G_


GAMMA_np     = _build_gamma()
GAMMA_INV_np = np.linalg.pinv(GAMMA_np).astype(np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    print(f"[PINN-MPPI v2] GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[PINN-MPPI v2] CPU mode")


# ── quaternion helpers (batch, GPU-compatible, CUDA-Graph-safe) ────────────────

def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
    """(K,4) → (K,4) unit quaternions [qx,qy,qz,qw]."""
    return q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)


def _quat_rotate_z(q: torch.Tensor) -> torch.Tensor:
    """Third column of R(q): world-frame body z-axis.
    q: (K,4) [qx,qy,qz,qw]  →  (K,3)  = R(q) @ [0,0,1]
    """
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rx = 2.0 * (qx * qz + qy * qw)
    ry = 2.0 * (qy * qz - qx * qw)
    rz = 1.0 - 2.0 * (qx * qx + qy * qy)
    return torch.stack([rx, ry, rz], dim=1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion product q1 ⊗ q2. Both (K,4) [qx,qy,qz,qw]."""
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dim=1)


def _quat_from_z(b3: torch.Tensor) -> torch.Tensor:
    """Minimal-rotation quaternion: world-z [0,0,1] → b3.
    b3: (K,3) unit vectors  →  (K,4) [qx,qy,qz,qw].
    Handles near-singular cases (b3 ≈ ±z).
    """
    K    = b3.shape[0]
    dev  = b3.device
    dt   = b3.dtype
    dot  = b3[:, 2:3].clamp(-1.0 + 1e-6, 1.0 - 1e-6)  # cos θ, (K,1)

    hc = torch.sqrt((1.0 + dot) * 0.5 + 1e-8)   # cos(θ/2)
    hs = torch.sqrt((1.0 - dot) * 0.5 + 1e-8)   # sin(θ/2)

    # rotation axis: cross([0,0,1], b3) = [-b3y, b3x, 0]
    ax = -b3[:, 1:2]
    ay =  b3[:, 0:1]
    az = torch.zeros_like(ax)

    axis_len = (ax*ax + ay*ay + az*az).clamp(min=1e-12).sqrt()
    ax = ax / axis_len * hs
    ay = ay / axis_len * hs
    az = az / axis_len * hs

    q = torch.cat([ax, ay, az, hc], dim=1)   # (K,4)

    # Degenerate: b3≈+z → identity, b3≈-z → 180° around x
    ident = torch.zeros(K, 4, device=dev, dtype=dt); ident[:, 3] = 1.0
    flipx = torch.zeros(K, 4, device=dev, dtype=dt); flipx[:, 0] = 1.0

    q = torch.where((b3[:, 2:3] >  0.9999).expand(-1, 4), ident, q)
    q = torch.where((b3[:, 2:3] < -0.9999).expand(-1, 4), flipx, q)
    return q


# ── PINN inference ─────────────────────────────────────────────────────────────

def pinn_infer(model, normalizer, state, wind_vec, device):
    """Query PINN with v=0 to isolate wind-only drag.
    Returns (3,) numpy residual acceleration [m/s²], world frame."""
    vel   = np.array(state['v'], dtype=np.float32)
    quat  = np.array(state['q'], dtype=np.float32)
    omega = np.array(state['w'], dtype=np.float32)
    wind  = np.array(wind_vec,   dtype=np.float32)
    u_hov = np.full(4, HOVER_OMEGA, dtype=np.float32)

    v_rel = vel - wind
    X     = np.concatenate([v_rel, quat, omega, u_hov, wind])
    X_t   = torch.tensor(X,     dtype=torch.float32, device=device).unsqueeze(0)
    vr_t  = torch.tensor(v_rel, dtype=torch.float32, device=device).unsqueeze(0)
    X_n   = (X_t - normalizer.mean) / normalizer.std
    with torch.no_grad():
        return model(X_n, vr_t).squeeze(0).cpu().numpy()


# ── PINN-MPPI Controller ───────────────────────────────────────────────────────

class PINNMPPIv2:
    """
    α-MPPI with full 12-DOF quaternion RK4 rollout + CUDA Graph acceleration.

    Control variable: α ∈ [0, 1.5]³  (per-axis PINN feedforward gain).
    Rollout: SE3 position + attitude embedded in quaternion RK4 dynamics.
    Execution: flat['x_ddot'] = traj_x_ddot + (-α_opt ⊙ r̂_ema)  (unchanged).

    CUDA Graph: on CUDA devices the entire H-step rollout is captured once
    and replayed without Python overhead, targeting ~100 Hz (K=896, H=15).
    """

    def __init__(self, model, normalizer, wind_vec,
                 K=896, H=15, dt=0.01, n_interp=10,
                 sigma=0.25, lam=5.0,
                 c_p=200., c_v=20.,
                 alpha_default=0.75, alpha_min=0.0, alpha_max=1.5,
                 use_pinn=True, device=None,
                 kp_pos=(6.5, 6.5, 15.0),
                 kd_pos=(5.0, 5.0,  8.0),
                 kr_att=12.0):

        self.device    = device or DEVICE
        self.use_pinn  = use_pinn
        self.wind_vec  = np.array(wind_vec, dtype=np.float32)
        self.K         = K
        self.H         = H
        self.dt        = dt
        self.n_interp  = n_interp
        self.dt_pred   = dt * n_interp
        self.sigma     = sigma
        self.lam       = lam
        self.c_p       = c_p
        self.c_v       = c_v
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_def = alpha_default
        self.Kr        = kr_att

        if use_pinn:
            self.model = model.to(self.device)
            self.model.eval()
            self.normalizer      = normalizer
            self.normalizer.mean = normalizer.mean.to(self.device)
            self.normalizer.std  = normalizer.std.to(self.device)
        else:
            self.model      = None
            self.normalizer = None

        dev = self.device
        self.Kp    = torch.tensor(kp_pos, dtype=torch.float32, device=dev)
        self.Kd    = torch.tensor(kd_pos, dtype=torch.float32, device=dev)
        self.G_VEC = torch.tensor([0., 0., G], dtype=torch.float32, device=dev)

        # Warm-start: α sequence (H, 3)
        self.alpha_seq = np.full((H, 3), alpha_default, dtype=np.float32)

        self._compute_times = []
        self._graph         = None

        # Pre-allocate static tensors; build CUDA Graph on CUDA devices
        self._alloc_static()
        if dev.type == 'cuda':
            try:
                self._build_graph()
            except Exception as e:
                print(f"  [PINNMPPIv2] CUDA Graph capture failed ({e}), "
                      f"using eager mode")
                self._graph = None

    # ── static tensor allocation ──────────────────────────────────────────────

    def _alloc_static(self):
        """Pre-allocate all tensors used in _rollout_inplace."""
        K, H = self.K, self.H
        dev  = self.device

        # Inputs — filled by update() before each call
        self._s_p0     = torch.zeros(K, 3, device=dev)
        self._s_v0     = torch.zeros(K, 3, device=dev)
        self._s_q0     = torch.zeros(K, 4, device=dev)
        self._s_q0[:, 3] = 1.0            # identity quaternion
        self._s_r_hat  = torch.zeros(K, 3, device=dev)
        self._s_alpha  = torch.zeros(K, H, 3, device=dev)
        self._s_p_des  = torch.zeros(H, 3, device=dev)
        self._s_v_des  = torch.zeros(H, 3, device=dev)
        self._s_a_traj = torch.zeros(H, 3, device=dev)

        # Outputs — filled by _rollout_inplace
        self._s_costs     = torch.zeros(K, device=dev)
        self._s_alpha_opt = torch.zeros(H, 3, device=dev)

    # ── dynamics: SE3 + quaternion kinematics ─────────────────────────────────

    def _deriv(self, p, v, q, alpha_h, p_des_h, v_des_h, a_traj_h, r_hat_k):
        """
        Compute (dp, dv, dq) for one derivative evaluation.

        p, v     : (K,3)   position, velocity
        q        : (K,4)   quaternion [qx,qy,qz,qw]
        alpha_h  : (K,3)   feedforward gain
        p_des_h, v_des_h, a_traj_h : (1,3) references (broadcast over K)
        r_hat_k  : (K,3)   PINN residual
        """
        Kp = self.Kp.unsqueeze(0)
        Kd = self.Kd.unsqueeze(0)
        g  = self.G_VEC.unsqueeze(0)

        # ── SE3 position controller ───────────────────────────────────────
        ep    = p - p_des_h
        ev    = v - v_des_h
        a_cmd = -Kp * ep - Kd * ev + a_traj_h - alpha_h * r_hat_k   # (K,3)

        a_total = a_cmd + g                                       # (K,3)
        Ft      = MASS * a_total.norm(dim=1, keepdim=True).clamp(min=0.5)
        z_des   = F.normalize(a_total, dim=1)                     # (K,3)

        # ── SE3 attitude controller (embedded) ────────────────────────────
        q_des     = _quat_from_z(z_des)
        q_des_inv = torch.cat([-q_des[:, :3], q_des[:, 3:4]], dim=1)
        q_err     = _quat_mul(q_des_inv, q)
        sign      = (q_err[:, 3:4] >= 0).to(q.dtype) * 2.0 - 1.0  # ±1, no zero
        omega_cmd = (-self.Kr) * q_err[:, :3] * sign               # (K,3)

        # ── 12-DOF dynamics ───────────────────────────────────────────────
        dp = v
        z_world = _quat_rotate_z(q)
        dv = (Ft / MASS) * z_world - g + (1.0 - alpha_h) * r_hat_k  # (K,3)

        # dq = 0.5 * q ⊗ [omega_cmd; 0]
        omega_q = torch.cat(
            [omega_cmd, torch.zeros(p.shape[0], 1, device=p.device)], dim=1)
        dq = 0.5 * _quat_mul(q, omega_q)

        return dp, dv, dq

    def _rk4_step(self, p, v, q, alpha_h,
                  p_des_h, v_des_h, a_traj_h, r_hat_k, dt):
        """One RK4 integration step (all args fixed over [t, t+dt])."""
        k1p, k1v, k1q = self._deriv(p, v, q,
                                     alpha_h, p_des_h, v_des_h, a_traj_h, r_hat_k)
        k2p, k2v, k2q = self._deriv(
            p + 0.5*dt*k1p, v + 0.5*dt*k1v,
            _quat_normalize(q + 0.5*dt*k1q),
            alpha_h, p_des_h, v_des_h, a_traj_h, r_hat_k)
        k3p, k3v, k3q = self._deriv(
            p + 0.5*dt*k2p, v + 0.5*dt*k2v,
            _quat_normalize(q + 0.5*dt*k2q),
            alpha_h, p_des_h, v_des_h, a_traj_h, r_hat_k)
        k4p, k4v, k4q = self._deriv(
            p + dt*k3p, v + dt*k3v,
            _quat_normalize(q + dt*k3q),
            alpha_h, p_des_h, v_des_h, a_traj_h, r_hat_k)

        c = dt / 6.0
        p_new = p + c * (k1p + 2.0*k2p + 2.0*k3p + k4p)
        v_new = v + c * (k1v + 2.0*k2v + 2.0*k3v + k4v)
        q_new = _quat_normalize(q + c * (k1q + 2.0*k2q + 2.0*k3q + k4q))
        return p_new, v_new, q_new

    # ── graph-capturable rollout ──────────────────────────────────────────────

    def _rollout_inplace(self):
        """
        Full K-sample, H-step rollout using static tensors.
        Pure GPU tensor ops — safe for CUDA Graph capture and replay.

        Reads:  _s_p0, _s_v0, _s_q0, _s_r_hat, _s_alpha,
                _s_p_des, _s_v_des, _s_a_traj
        Writes: _s_costs, _s_alpha_opt
        """
        dt = self.dt_pred
        H  = self.H
        K  = self.K

        p = self._s_p0.clone()   # (K,3) working copies — fixed memory on replay
        v = self._s_v0.clone()
        q = self._s_q0.clone()

        self._s_costs.zero_()

        for h in range(H):
            alpha_h  = self._s_alpha[:, h, :]       # (K,3)
            p_des_h  = self._s_p_des[h:h+1]         # (1,3) broadcasts over K
            v_des_h  = self._s_v_des[h:h+1]
            a_traj_h = self._s_a_traj[h:h+1]

            ep = p - p_des_h
            ev = v - v_des_h
            self._s_costs.add_(
                (ep * ep).sum(1) * self.c_p + (ev * ev).sum(1) * self.c_v
            )

            p, v, q = self._rk4_step(
                p, v, q, alpha_h,
                p_des_h, v_des_h, a_traj_h,
                self._s_r_hat, dt)

        # MPPI weighting
        c_min  = self._s_costs.min()
        w      = torch.exp(-(self._s_costs - c_min) / self.lam)
        w_norm = w / w.sum().clamp(min=1e-8)

        # Weighted sum → optimal alpha sequence (H,3)
        self._s_alpha_opt.copy_(
            (w_norm.view(K, 1, 1) * self._s_alpha).sum(0))

    # ── CUDA Graph build ──────────────────────────────────────────────────────

    def _build_graph(self):
        """Warmup (fills CUDA caches) then capture rollout as a CUDA Graph."""
        # 3 warmup runs — needed to initialise cuBLAS/cuDNN caches
        for _ in range(3):
            self._rollout_inplace()
        torch.cuda.synchronize(self.device)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._rollout_inplace()
        torch.cuda.synchronize(self.device)
        print(f"  [PINNMPPIv2] CUDA Graph captured  K={self.K}  H={self.H}")

    # ── core MPPI update ──────────────────────────────────────────────────────

    def update(self, state, ref_seq, r_hat):
        """
        Parameters
        ----------
        state   : dict with 'x'(3,), 'v'(3,), 'q'(4,)
        ref_seq : list of H dicts, each with 'x', optionally 'x_dot', 'x_ddot'
        r_hat   : (3,) numpy — EMA-smoothed PINN wind residual

        Returns
        -------
        alpha_opt : (3,) numpy
        dt_ms     : float [ms]
        """
        t0   = time.perf_counter()
        K, H = self.K, self.H
        dev  = self.device

        # ── fill reference buffers (outside graph) ───────────────────────
        p_des  = torch.stack([
            torch.tensor(ref_seq[h]['x'], dtype=torch.float32, device=dev)
            for h in range(H)])
        v_des  = torch.stack([
            torch.tensor(ref_seq[h].get('x_dot',  np.zeros(3)),
                         dtype=torch.float32, device=dev)
            for h in range(H)])
        a_traj = torch.stack([
            torch.tensor(ref_seq[h].get('x_ddot', np.zeros(3)),
                         dtype=torch.float32, device=dev)
            for h in range(H)])

        self._s_p_des.copy_(p_des)
        self._s_v_des.copy_(v_des)
        self._s_a_traj.copy_(a_traj)

        # ── fill state buffers (outside graph) ───────────────────────────
        p0 = torch.tensor(state['x'], dtype=torch.float32, device=dev)
        v0 = torch.tensor(state['v'], dtype=torch.float32, device=dev)
        q0 = torch.tensor(state.get('q', [0., 0., 0., 1.]),
                          dtype=torch.float32, device=dev)

        self._s_p0.copy_(p0.unsqueeze(0).expand(K, -1))
        self._s_v0.copy_(v0.unsqueeze(0).expand(K, -1))
        self._s_q0.copy_(q0.unsqueeze(0).expand(K, -1))

        # ── fill PINN residual (outside graph) ───────────────────────────
        r_t = torch.tensor(r_hat, dtype=torch.float32, device=dev)
        self._s_r_hat.copy_(r_t.unsqueeze(0).expand(K, -1))

        # ── sample α sequences with fresh noise (outside graph) ──────────
        U_ws  = torch.tensor(self.alpha_seq, dtype=torch.float32, device=dev)
        noise = torch.randn(K, H, 3, device=dev) * self.sigma
        alpha = (U_ws.unsqueeze(0) + noise).clamp(self.alpha_min, self.alpha_max)
        self._s_alpha.copy_(alpha)

        # ── rollout + MPPI weighting (graph replay or eager) ─────────────
        if self._graph is not None:
            self._graph.replay()
        else:
            self._rollout_inplace()

        # ── read result and warm-start update ────────────────────────────
        alpha_opt_np = self._s_alpha_opt.cpu().numpy()   # (H,3); syncs GPU→CPU
        self.alpha_seq[:-1] = alpha_opt_np[1:]
        self.alpha_seq[-1]  = self.alpha_def

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._compute_times.append(dt_ms)
        return alpha_opt_np[0], dt_ms

    # ── diagnostics ───────────────────────────────────────────────────────────

    def compute_frequency_stats(self):
        if not self._compute_times:
            return {}
        times = np.array(self._compute_times)
        hz    = 1000.0 / (times + 1e-9)
        return {
            'mean_ms': float(times.mean()),
            'std_ms':  float(times.std()),
            'mean_hz': float(hz.mean()),
            'min_hz':  float(hz.min()),
            'max_hz':  float(hz.max()),
        }


# ── Traditional (Nominal) MPPI — direct acceleration feedforward ───────────────

class NominalMPPIv2(PINNMPPIv2):
    """
    Traditional MPPI baseline — no PINN, no wind knowledge.

    Optimization variable: Δa ∈ [−da_max, da_max]³  (per-axis acceleration feedforward)
    Rollout model:
        a_cmd = −Kp·ep − Kd·ev + a_traj + Δa      (controller applies Δa directly)
        dv    = (Ft/m)·z_world − g                  (NO wind term — nominal dynamics)
    Execution:
        flat['x_ddot'] = traj_x_ddot + Δa_opt

    Demonstrates MPPI's multi-step lookahead benefit without wind knowledge.
    Expected ordering:  SE3-Only  <  NominalMPPI  <  PINN-MPPI
    """

    def __init__(self, wind_vec, da_max=5.0, sigma=0.5, K=896, H=15,
                 dt=0.01, n_interp=10, lam=5.0, c_p=200., c_v=20.,
                 device=None, **kwargs):
        # Pass dummy model=None; use_pinn=False skips model.to(device) etc.
        super().__init__(
            model=None, normalizer=None, wind_vec=wind_vec,
            K=K, H=H, dt=dt, n_interp=n_interp, sigma=sigma, lam=lam,
            c_p=c_p, c_v=c_v,
            alpha_default=0.0,          # no perturbation at warm-start tail
            alpha_min=-da_max,
            alpha_max=da_max,
            use_pinn=False,
            device=device,
            **kwargs,
        )

    def _deriv(self, p, v, q, alpha_h, p_des_h, v_des_h, a_traj_h, r_hat_k):
        """
        alpha_h  : (K,3) — treated as direct Δa feedforward (NOT a gain)
        r_hat_k  : (K,3) — IGNORED (no wind model in nominal rollout)
        """
        Kp = self.Kp.unsqueeze(0)
        Kd = self.Kd.unsqueeze(0)
        g  = self.G_VEC.unsqueeze(0)

        ep    = p - p_des_h
        ev    = v - v_des_h
        # Controller injects Δa directly (positive addition, matching execution)
        a_cmd = -Kp * ep - Kd * ev + a_traj_h + alpha_h  # (K,3)

        a_total = a_cmd + g
        Ft      = MASS * a_total.norm(dim=1, keepdim=True).clamp(min=0.5)
        z_des   = F.normalize(a_total, dim=1)

        q_des     = _quat_from_z(z_des)
        q_des_inv = torch.cat([-q_des[:, :3], q_des[:, 3:4]], dim=1)
        q_err     = _quat_mul(q_des_inv, q)
        sign      = (q_err[:, 3:4] >= 0).to(q.dtype) * 2.0 - 1.0
        omega_cmd = (-self.Kr) * q_err[:, :3] * sign

        dp = v
        z_world = _quat_rotate_z(q)
        dv = (Ft / MASS) * z_world - g   # nominal dynamics: NO wind residual

        omega_q = torch.cat(
            [omega_cmd, torch.zeros(p.shape[0], 1, device=p.device)], dim=1)
        dq = 0.5 * _quat_mul(q, omega_q)

        return dp, dv, dq


# ── nominal MPPI episode ───────────────────────────────────────────────────────

def run_nominal_episode(wind_vec, trajectory_fn,
                        sim_time=15.0, dt=0.01, K=896, H=15, n_interp=10,
                        da_max=5.0, sigma=0.5, verbose=False):
    """
    Traditional MPPI: directly optimises per-axis acceleration Δa ∈ [−da_max, da_max]³.
    No PINN, no wind knowledge in rollout model.
    Execution: flat['x_ddot'] = traj_x_ddot + Δa_opt
    """
    trajectory = trajectory_fn()
    start_pos  = trajectory.update(0.0)['x'].copy()

    state = {
        'x':            start_pos,
        'v':            np.zeros(3),
        'q':            np.array([0., 0., 0., 1.]),
        'w':            np.zeros(3),
        'wind':         wind_vec.copy(),
        'rotor_speeds': np.ones(4) * HOVER_OMEGA,
    }

    vehicle = Multirotor(quad_params, state)
    ctrl    = SE3Control(quad_params)
    mppi    = NominalMPPIv2(wind_vec, da_max=da_max, sigma=sigma,
                             K=K, H=H, dt=dt, n_interp=n_interp)

    N         = int(sim_time / dt)
    times     = np.zeros(N)
    positions = np.zeros((N, 3))
    des_pos   = np.zeros((N, 3))
    errors    = np.zeros(N)
    comp_ms   = np.zeros(N)

    _traj = trajectory_fn()

    for i in range(N):
        t = i * dt
        state['wind'] = wind_vec.copy()

        ref_seq = [_traj.update(t + j * mppi.dt_pred) for j in range(H)]

        # update() accepts r_hat but NominalMPPIv2._deriv ignores it
        delta_a_opt, dt_ms = mppi.update(state, ref_seq, np.zeros(3))
        comp_ms[i] = dt_ms

        flat        = dict(ref_seq[0])
        traj_x_ddot = np.array(ref_seq[0].get('x_ddot', np.zeros(3)),
                                dtype=np.float32)
        # Direct acceleration feedforward (no r_hat scaling)
        flat['x_ddot'] = traj_x_ddot + delta_a_opt

        cmd   = ctrl.update(t, state, flat)
        state = vehicle.step(state, cmd, dt)

        times[i]     = t
        positions[i] = state['x']
        des_pos[i]   = ref_seq[0]['x']
        errors[i]    = np.linalg.norm(positions[i] - des_pos[i])

        if verbose and i % 50 == 0:
            hz = 1000.0 / dt_ms if dt_ms > 0 else 0
            print(f"  t={t:.1f}s  err={errors[i]:.4f}m  "
                  f"{dt_ms:.1f}ms ({hz:.0f}Hz)  "
                  f"da=[{delta_a_opt[0]:.2f},{delta_a_opt[1]:.2f},{delta_a_opt[2]:.2f}]",
                  end='\r')

    if verbose:
        print()

    return {
        'times':      times,
        'positions':  positions,
        'des_pos':    des_pos,
        'errors':     errors,
        'comp_ms':    comp_ms,
        'mean_err':   float(errors[100:].mean()),
        'freq_stats': mppi.compute_frequency_stats(),
    }


# ── quaternion → rotation matrix (numpy, diagnostics only) ────────────────────

def _quat_to_rotmat(q):
    """[qx,qy,qz,qw] → 3×3 rotation matrix (body→world)."""
    qx, qy, qz, qw = q
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)  ],
        [2*(qx*qy+qz*qw),   1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)  ],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx**2+qy**2)],
    ], dtype=np.float64)


# ── closed-loop episode ────────────────────────────────────────────────────────

def run_episode(model, normalizer, wind_vec, trajectory_fn,
                sim_time=15.0, dt=0.01, K=896, H=15, n_interp=10,
                use_pinn=True, verbose=False):
    """
    Run one closed-loop episode with α-MPPI + PINN feedforward.

    PINN is queried at actual drone velocity; r̂ is clamped to ±6 m/s² per axis.
    MPPI (full quaternion RK4 rollout) optimises α ∈ [0,1.5]³.
    SE3Control executes with PINN-scaled feedforward:
        flat['x_ddot'] = traj_x_ddot + (-α_opt ⊙ r̂_ema)

    use_pinn=False: r̂_ema=0 → flat['x_ddot']=traj_x_ddot → SE3-Only baseline.
    """
    trajectory = trajectory_fn()
    start_pos  = trajectory.update(0.0)['x'].copy()

    state = {
        'x':            start_pos,
        'v':            np.zeros(3),
        'q':            np.array([0., 0., 0., 1.]),
        'w':            np.zeros(3),
        'wind':         wind_vec.copy(),
        'rotor_speeds': np.ones(4) * HOVER_OMEGA,
    }

    vehicle = Multirotor(quad_params, state)
    ctrl    = SE3Control(quad_params)
    mppi    = PINNMPPIv2(model, normalizer, wind_vec,
                          K=K, H=H, dt=dt, n_interp=n_interp,
                          use_pinn=use_pinn)
    dev     = mppi.device

    N         = int(sim_time / dt)
    times     = np.zeros(N)
    positions = np.zeros((N, 3))
    des_pos   = np.zeros((N, 3))
    errors    = np.zeros(N)
    comp_ms   = np.zeros(N)

    _traj = trajectory_fn()

    # EMA on PINN output — smooths initial transient kick
    # α=0.05 → ~0.2 s time constant at 100 Hz
    ALPHA_EMA = 0.05
    r_ema     = np.zeros(3, dtype=np.float32)

    for i in range(N):
        t = i * dt
        state['wind'] = wind_vec.copy()

        ref_seq = [_traj.update(t + j * mppi.dt_pred) for j in range(H)]

        # PINN wind-residual estimate (velocity-aware)
        # Query at actual drone velocity to capture drag variation during dynamic
        # manoeuvres (figure-eight max speed ±1.41 m/s → ±25% drag change vs v=0).
        # r_ema is clamped to R_EMA_MAX before feedforward to prevent motor
        # saturation when moving hard against wind (v_rel_x can reach −9.4 m/s).
        R_EMA_MAX = 6.0   # m/s² — safe feedforward ceiling (alpha_max=1.5 → 9 m/s²)
        if use_pinn:
            s_pinn = dict(state)
            res_np = pinn_infer(mppi.model, mppi.normalizer,
                                s_pinn, wind_vec, dev)
            r_ema = ALPHA_EMA * res_np + (1.0 - ALPHA_EMA) * r_ema
        else:
            r_ema = np.zeros(3, dtype=np.float32)

        # Clamp EMA before it reaches MPPI / feedforward
        r_ema_clamped = np.clip(r_ema, -R_EMA_MAX, R_EMA_MAX)

        # MPPI: optimise α with full quaternion RK4 rollout.
        # Zero out z component before MPPI/feedforward: horizontal wind creates
        # a tiny z aerodynamic effect (r_ema_z ≈ 0.35 m/s²) that SE3 Kp_z=15
        # handles with only 0.023 m steady-state error.  Letting MPPI optimise
        # alpha_z causes more z oscillation than it removes.
        r_ema_xy       = r_ema_clamped.copy()
        r_ema_xy[2]    = 0.0
        alpha_opt, dt_ms = mppi.update(state, ref_seq, r_ema_xy)
        comp_ms[i] = dt_ms

        # SE3Control feedforward: inject PINN compensation into x_ddot
        flat        = dict(ref_seq[0])
        traj_x_ddot = np.array(ref_seq[0].get('x_ddot', np.zeros(3)),
                                dtype=np.float32)
        flat['x_ddot'] = traj_x_ddot + (-alpha_opt * r_ema_xy)

        cmd   = ctrl.update(t, state, flat)
        state = vehicle.step(state, cmd, dt)

        times[i]     = t
        positions[i] = state['x']
        des_pos[i]   = ref_seq[0]['x']
        errors[i]    = np.linalg.norm(positions[i] - des_pos[i])

        if verbose and i % 50 == 0:
            hz = 1000.0 / dt_ms if dt_ms > 0 else 0
            print(f"  t={t:.1f}s  err={errors[i]:.4f}m  "
                  f"{dt_ms:.1f}ms ({hz:.0f}Hz)  "
                  f"α=[{alpha_opt[0]:.2f},{alpha_opt[1]:.2f},{alpha_opt[2]:.2f}]",
                  end='\r')

    if verbose:
        print()

    return {
        'times':      times,
        'positions':  positions,
        'des_pos':    des_pos,
        'errors':     errors,
        'comp_ms':    comp_ms,
        'mean_err':   float(errors[100:].mean()),
        'freq_stats': mppi.compute_frequency_stats(),
    }


# ── actuator layer (kept for external API compatibility) ──────────────────────

def body_rate_pd_to_motors(omega_des, omega_cur, Ft, Kp=10.0):
    """Body-rate PD → motor speeds (not used by run_episode; kept for API compat)."""
    e_w   = omega_des - omega_cur
    tau   = Kp * J_np * e_w
    b_raw = np.array([Ft, tau[0], tau[1], tau[2]])
    T_raw = GAMMA_INV_np @ b_raw

    T_base = Ft / 4.0
    T_tau  = T_raw - T_base
    scale  = 1.0
    pos    = T_tau[T_tau > 0]
    neg    = T_tau[T_tau < 0]
    if pos.size:
        scale = min(scale, float(((T_MAX - T_base) / pos).min()))
    if neg.size:
        scale = min(scale, float(((T_base - T_MIN) / (-neg)).min()))
    scale = max(0.0, scale)

    tau_s = tau * scale
    b     = np.array([Ft, tau_s[0], tau_s[1], tau_s[2]])
    T     = GAMMA_INV_np @ b
    T     = np.clip(T, T_MIN + 1e-8, T_MAX)
    return np.clip(np.sqrt(T / K_ETA), OMEGA_MIN, OMEGA_MAX)


# ── model loading ──────────────────────────────────────────────────────────────

def load_pinn_model(ckpt_dir=CKPT_DIR):
    ckpt       = torch.load(os.path.join(ckpt_dir, 'best_model.pt'),
                            weights_only=False)
    normalizer = Normalizer()
    normalizer.load(os.path.join(ckpt_dir, 'normalizer.pt'))
    model = ResidualPINN(input_dim=17)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  PINN loaded  epoch={ckpt['epoch']}  "
          f"val_rmse={ckpt['val_rmse'].round(3)}  "
          f"ood_rmse={ckpt['ood_rmse'].round(3)}")
    return model, normalizer


# ── standalone test ────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    model, normalizer = load_pinn_model()

    print(f"\nα-MPPI (full quat RK4 + CUDA Graph)  K=896 H=15 n_interp=10")
    print(f"Device: {DEVICE}")

    trajectories = {
        'hover':  lambda: HoverTraj(x0=np.array([0., 0., 1.5])),
        'circle': lambda: ThreeDCircularTraj(
            center=np.array([0., 0., 1.5]),
            radius=np.array([1.5, 1.5, 0.]),
            freq=np.array([0.2, 0.2, 0.]),
        ),
    }

    wind_configs = [
        (np.array([0.,  0., 0.]), 'wind=0  m/s (no wind)'),
        (np.array([4.,  0., 0.]), 'wind=4  m/s (train)  '),
        (np.array([8.,  0., 0.]), 'wind=8  m/s (train)  '),
        (np.array([10., 0., 0.]), 'wind=10 m/s (OOD)    '),
        (np.array([12., 0., 0.]), 'wind=12 m/s (OOD)    '),
    ]

    print(f"\n{'':>7}  {'Wind':>22}  {'SE3-only':>8}  {'PINN-MPPI':>10}  "
          f"{'Improv':>8}     Hz")
    print('-' * 80)

    for tname, tfn in trajectories.items():
        for wv, wlabel in wind_configs:
            r_nom  = run_episode(model, normalizer, wv, tfn,
                                 sim_time=12., K=896, H=15,
                                 use_pinn=False, verbose=False)
            r_pinn = run_episode(model, normalizer, wv, tfn,
                                 sim_time=12., K=896, H=15,
                                 use_pinn=True, verbose=False)
            improv = (r_nom['mean_err'] - r_pinn['mean_err']) \
                     / (r_nom['mean_err'] + 1e-9) * 100
            hz     = r_pinn['freq_stats'].get('mean_hz', 0)
            print(f"{tname:>7}  {wlabel}  "
                  f"{r_nom['mean_err']:8.4f}  "
                  f"{r_pinn['mean_err']:10.4f}  "
                  f"{improv:+7.1f}%  {hz:6.1f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
