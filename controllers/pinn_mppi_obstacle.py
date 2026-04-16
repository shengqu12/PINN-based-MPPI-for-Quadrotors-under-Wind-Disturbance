"""
pinn_mppi_obstacle.py — PINN-MPPI with obstacle avoidance

Extends PINNMPPIv2 by adding a cylindrical obstacle repulsion cost
directly inside the MPPI rollout. Works with CUDA Graph.

Usage:
    from controllers.pinn_mppi_obstacle import PINNMPPIObstacle, run_obstacle_episode

    obstacles = [
        (np.array([1.0, 0.0, 1.5]), 0.3),   # (center_xyz, radius_m)
    ]
    mppi = PINNMPPIObstacle(model, normalizer, wind_vec,
                             obstacles=obstacles, c_obs=3000.)
"""

import os, sys
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from controllers.pinn_mppi_v2 import (
    PINNMPPIv2, pinn_infer, load_pinn_model,
    HOVER_OMEGA, DEVICE,
)
from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj


# ── Obstacle-aware MPPI ────────────────────────────────────────────────────────

class PINNMPPIObstacle(PINNMPPIv2):
    """
    PINNMPPIv2 + cylindrical obstacle avoidance cost.

    Obstacle model: vertical cylinder — only x-y distance matters.
    Cost: c_obs * relu(r + margin - dist_xy)^2  for each obstacle.

    CUDA Graph compatible: obstacle tensors are pre-allocated static buffers.
    Obstacle positions/radii are fixed for the lifetime of the controller instance.

    Parameters
    ----------
    obstacles   : list of (center: np.array(3,), radius: float)
    c_obs       : obstacle cost weight  (default 3000)
    obs_margin  : soft safety buffer beyond physical radius [m]  (default 0.30)
    """

    def __init__(self, model, normalizer, wind_vec,
                 obstacles=None,
                 c_obs=3000.,
                 obs_margin=0.30,
                 **kwargs):
        # ── must set these BEFORE super().__init__() ──────────────────────
        # super().__init__() calls self._alloc_static() and self._build_graph()
        # via Python MRO → our overridden versions will be called.
        self._obs_list  = list(obstacles) if obstacles else []
        self._N_obs     = len(self._obs_list)
        self.c_obs      = float(c_obs)
        self.obs_margin = float(obs_margin)

        super().__init__(model, normalizer, wind_vec, **kwargs)

    # ── override: add obstacle static tensors ─────────────────────────────

    def _alloc_static(self):
        super()._alloc_static()
        dev = self.device
        N   = max(1, self._N_obs)   # ≥1 to avoid empty-tensor issues

        centers = torch.zeros(N, 3, device=dev)
        radii   = torch.zeros(N,    device=dev)

        for i, (center, radius) in enumerate(self._obs_list):
            centers[i] = torch.tensor(np.asarray(center, dtype=np.float32), device=dev)
            radii[i]   = float(radius)

        self._s_obs_centers = centers   # (N, 3)  x/y/z of obstacle centres
        self._s_obs_r       = radii     # (N,)    physical radii

    # ── override: rollout with obstacle cost ──────────────────────────────

    def _rollout_inplace(self):
        """
        Full rollout (same as PINNMPPIv2) + obstacle repulsion cost at each
        horizon step.  All GPU tensor ops — CUDA Graph safe.
        """
        dt = self.dt_pred
        H  = self.H
        K  = self.K

        p = self._s_p0.clone()
        v = self._s_v0.clone()
        q = self._s_q0.clone()

        self._s_costs.zero_()

        for h in range(H):
            alpha_h  = self._s_alpha[:, h, :]
            p_des_h  = self._s_p_des[h:h+1]
            v_des_h  = self._s_v_des[h:h+1]
            a_traj_h = self._s_a_traj[h:h+1]

            ep = p - p_des_h
            ev = v - v_des_h

            # ── tracking cost ─────────────────────────────────────────────
            self._s_costs.add_(
                (ep * ep).sum(1) * self.c_p + (ev * ev).sum(1) * self.c_v
            )

            # ── obstacle repulsion cost (xy-only: cylinders are vertical) ─
            if self._N_obs > 0:
                N_obs = self._N_obs
                # (K,1,2) vs (1,N_obs,2) → (K,N_obs,2)
                p_xy  = p[:, :2].unsqueeze(1)
                c_xy  = self._s_obs_centers[:N_obs, :2].unsqueeze(0)
                dist  = (p_xy - c_xy).norm(dim=2)                          # (K,N_obs)
                pen   = (self._s_obs_r[:N_obs].unsqueeze(0)
                         + self.obs_margin - dist).clamp(min=0.)           # (K,N_obs)
                self._s_costs.add_((pen * pen).sum(1) * self.c_obs)

            # ── RK4 step ──────────────────────────────────────────────────
            p, v, q = self._rk4_step(
                p, v, q, alpha_h,
                p_des_h, v_des_h, a_traj_h,
                self._s_r_hat, dt)

        # ── MPPI weighting ────────────────────────────────────────────────
        c_min  = self._s_costs.min()
        w      = torch.exp(-(self._s_costs - c_min) / self.lam)
        w_norm = w / w.sum().clamp(min=1e-8)
        self._s_alpha_opt.copy_(
            (w_norm.view(K, 1, 1) * self._s_alpha).sum(0))


# ── closed-loop episode with obstacles ─────────────────────────────────────────

def run_obstacle_episode(model, normalizer, wind_vec, trajectory_fn,
                         obstacles=None,
                         sim_time=20.0, dt=0.01, K=896, H=15, n_interp=10,
                         use_pinn=True, c_obs=3000., obs_margin=0.30,
                         verbose=False):
    """
    Same as pinn_mppi_v2.run_episode but with obstacle avoidance.

    Returns dict with 'positions', 'des_pos', 'errors', 'freq_stats',
    'collisions' (bool array, True when inside any obstacle).
    """
    obstacles = obstacles or []
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

    if obstacles and use_pinn:
        mppi = PINNMPPIObstacle(model, normalizer, wind_vec,
                                 obstacles=obstacles,
                                 K=K, H=H, dt=dt, n_interp=n_interp,
                                 c_obs=c_obs, obs_margin=obs_margin,
                                 use_pinn=True)
    else:
        from controllers.pinn_mppi_v2 import PINNMPPIv2
        mppi = PINNMPPIv2(model, normalizer, wind_vec,
                           K=K, H=H, dt=dt, n_interp=n_interp,
                           use_pinn=use_pinn)

    dev   = mppi.device
    N     = int(sim_time / dt)
    _traj = trajectory_fn()

    positions  = np.zeros((N, 3))
    des_pos    = np.zeros((N, 3))
    errors     = np.zeros(N)
    collisions = np.zeros(N, dtype=bool)

    ALPHA_EMA = 0.05
    r_ema     = np.zeros(3, dtype=np.float32)

    for i in range(N):
        t = i * dt
        state['wind'] = wind_vec.copy()

        ref_seq = [_traj.update(t + j * mppi.dt_pred) for j in range(H)]

        if use_pinn:
            s_pinn      = dict(state)
            s_pinn['v'] = np.zeros(3, dtype=np.float32)
            res_np      = pinn_infer(mppi.model, mppi.normalizer,
                                     s_pinn, wind_vec, dev)
            r_ema = ALPHA_EMA * res_np + (1.0 - ALPHA_EMA) * r_ema
        else:
            r_ema = np.zeros(3, dtype=np.float32)

        alpha_opt, dt_ms = mppi.update(state, ref_seq, r_ema)

        flat          = dict(ref_seq[0])
        traj_x_ddot   = np.array(ref_seq[0].get('x_ddot', np.zeros(3)),
                                  dtype=np.float32)
        flat['x_ddot'] = traj_x_ddot + (-alpha_opt * r_ema)

        cmd   = ctrl.update(t, state, flat)
        state = vehicle.step(state, cmd, dt)

        positions[i] = state['x']
        des_pos[i]   = ref_seq[0]['x']
        errors[i]    = np.linalg.norm(positions[i] - des_pos[i])

        # collision check (xy distance to each obstacle)
        for center, radius in obstacles:
            xy_dist = np.linalg.norm(positions[i, :2] - np.asarray(center)[:2])
            if xy_dist < radius:
                collisions[i] = True
                break

        if verbose and i % 100 == 0:
            hz = 1000.0 / dt_ms if dt_ms > 0 else 0
            nc = collisions[:i+1].sum()
            print(f"  t={t:.1f}s  err={errors[i]:.3f}m  {hz:.0f}Hz  "
                  f"collisions={nc}", end='\r')

    if verbose:
        print()

    return {
        'positions':  positions,
        'des_pos':    des_pos,
        'errors':     errors,
        'collisions': collisions,
        'mean_err':   float(errors[100:].mean()),
        'n_collide':  int(collisions.sum()),
        'freq_stats': mppi.compute_frequency_stats(),
    }
