"""
run_minarik_fig.py — Minarik-style 2-panel obstacle avoidance figure

Reproduces the style of Fig. 4 in Minarik et al.: two side-by-side 2D
top-down x-y panels.

    Panel (a): slanted circle — 3 cylindrical obstacles on the elliptical path
    Panel (b): diagonal line  — 2 cylindrical obstacles blocking the path

Controller: PINN-MPPI (α-MPPI for wind compensation) + potential-field
reference modification for obstacle avoidance.  The potential field shifts
the desired position away from obstacles so that SE3Control steers clear.

Blue dashed = reference (nominal trajectory, ignores obstacles).
Orange solid = PINN-MPPI + obstacle avoidance.
Black circles = physical obstacles; grey halos = safety margins.

Usage:
    python experiments/run_minarik_fig.py
    python experiments/run_minarik_fig.py --wind 8
"""

import os, sys, argparse, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patches as mpatches

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from controllers.pinn_mppi_v2 import (
    load_pinn_model, pinn_infer, PINNMPPIv2, HOVER_OMEGA, DEVICE)

from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import SE3Control

CKPT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


# ── Trajectory classes ─────────────────────────────────────────────────────────

class SlantedCircleTraj:
    """
    Circle tilted 30° out of horizontal plane.
    xy-projection is an ellipse: semi-axes R (x) and R*cos(30°) (y).
    """
    def __init__(self, center=np.array([0., 0., 1.5]),
                 R=2.0, freq=0.15, tilt_deg=30.):
        self.cx, self.cy, self.cz = center
        self.R  = R
        self.w  = 2 * np.pi * freq
        self.ct = np.cos(np.radians(tilt_deg))
        self.st = np.sin(np.radians(tilt_deg))

    def update(self, t):
        x  = self.cx + self.R * np.cos(self.w * t)
        y  = self.cy + self.R * np.sin(self.w * t) * self.ct
        z  = self.cz + self.R * np.sin(self.w * t) * self.st
        vx = -self.R * self.w * np.sin(self.w * t)
        vy =  self.R * self.w * np.cos(self.w * t) * self.ct
        vz =  self.R * self.w * np.cos(self.w * t) * self.st
        ax = -self.R * self.w**2 * np.cos(self.w * t)
        ay = -self.R * self.w**2 * np.sin(self.w * t) * self.ct
        az = -self.R * self.w**2 * np.sin(self.w * t) * self.st
        return {
            'x': np.array([x, y, z]),
            'x_dot': np.array([vx, vy, vz]),
            'x_ddot': np.array([ax, ay, az]),
            'x_dddot': np.zeros(3), 'x_ddddot': np.zeros(3),
            'yaw': 0., 'yaw_dot': 0.,
        }


class LineTraj:
    """
    Constant-velocity straight-line trajectory; hovers at endpoint.
    direction is normalised automatically.
    """
    def __init__(self, start=np.array([0., 0., 1.5]),
                 direction=np.array([1., 1., 0.]),
                 speed=0.4, length=6.0):
        self.start = np.array(start, dtype=float)
        d = np.array(direction, dtype=float)
        self.direction = d / np.linalg.norm(d)
        self.speed  = float(speed)
        self.length = float(length)

    def update(self, t):
        s      = min(t * self.speed, self.length)
        moving = (t * self.speed) < self.length
        pos = self.start + s * self.direction
        vel = (self.speed * self.direction) if moving else np.zeros(3)
        return {
            'x': pos, 'x_dot': vel,
            'x_ddot': np.zeros(3), 'x_dddot': np.zeros(3),
            'x_ddddot': np.zeros(3), 'yaw': 0., 'yaw_dot': 0.,
        }


# ── Obstacle avoidance: potential-field reference correction ──────────────────

def obstacle_repulsion(pos_xy, obstacles, obs_margin, k_rep=1.5,
                        sigma=0.8):
    """
    Gaussian repulsion from obstacles — smooth, no hard cutoff.

    rep = k_rep * exp(−d² / (2σ²)) * unit_outward
    where d = max(dist_from_centre − radius, 0)  [distance from obstacle surface].

    obs_margin is used to extend the effective radius (shifts Gaussian peak).
    Returns a 3-vector (x, y, 0).
    """
    rep = np.zeros(2)
    for (center, radius) in obstacles:
        c_xy = np.asarray(center)[:2]
        diff = pos_xy - c_xy
        dist = np.linalg.norm(diff)
        if dist > 1e-6:
            # Distance beyond effective surface (physical radius + margin/2)
            eff_r = radius + obs_margin * 0.5
            d     = max(dist - eff_r, 0.0)
            mag   = k_rep * np.exp(-d ** 2 / (2.0 * sigma ** 2))
            rep  += mag * (diff / dist)
    return np.array([rep[0], rep[1], 0.0])


# ── Episode runner with PINN-MPPI + potential field obstacle avoidance ─────────

def run_pf_episode(model, normalizer, wind_vec, traj_fn,
                   obstacles, obs_margin,
                   k_rep=3.0, sigma=0.8,
                   rep_source='reference',
                   sim_time=25.0, dt=0.01,
                   K=896, H=15, n_interp=10,
                   use_pinn=True, verbose=False):
    """
    PINN-MPPI wind compensation + potential-field obstacle avoidance.

    rep_source='reference'  — feedforward: repulsion computed from NOMINAL
        reference position.  Gives a smooth, pre-shaped detour the drone just
        follows.  Stable for circular trajectories (no feedback oscillations).
    rep_source='drone'      — reactive: repulsion from drone's current position.
        Gives reactive steering.  Works well for straight-line trajectories
        but can oscillate on curves.

    Returns dict with 'positions', 'des_pos', 'errors', 'collisions',
    'mean_err', 'n_collide', 'mean_hz'.
    """
    obstacles  = list(obstacles or [])
    trajectory = traj_fn()
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

    dev   = mppi.device
    N     = int(sim_time / dt)
    _traj = traj_fn()

    positions  = np.zeros((N, 3))
    des_pos    = np.zeros((N, 3))
    errors     = np.zeros(N)
    collisions = np.zeros(N, dtype=bool)

    ALPHA_EMA  = 0.05
    r_ema      = np.zeros(3, dtype=np.float32)
    t_compute  = []

    for i in range(N):
        t = i * dt
        state['wind'] = wind_vec.copy()

        ref_seq = [_traj.update(t + j * mppi.dt_pred) for j in range(H)]

        # ── PINN wind compensation (EMA) ────────────────────────────────
        if use_pinn:
            s_pinn = dict(state)
            r_new  = pinn_infer(mppi.model, mppi.normalizer,
                                s_pinn, wind_vec, dev)
            r_ema = ALPHA_EMA * r_new + (1.0 - ALPHA_EMA) * r_ema
        else:
            r_ema = np.zeros(3, dtype=np.float32)

        # ── α-MPPI optimisation ─────────────────────────────────────────
        t0 = time.perf_counter()
        alpha_opt, _ = mppi.update(state, ref_seq, r_ema)
        t_compute.append(time.perf_counter() - t0)

        # ── Obstacle avoidance (potential-field reference modification) ──
        # rep_source='reference': feedforward — repulsion from NOMINAL reference.
        #   Deterministic, smooth detour; best for circular / curved trajectories.
        # rep_source='drone':     reactive  — repulsion from drone current position.
        #   Works for straight-line trajectories without oscillation risk.
        ref_pos = ref_seq[0]['x']
        if rep_source == 'reference':
            src_xy = ref_pos[:2]
        else:  # 'drone'
            src_xy = state['x'][:2]
        rep = obstacle_repulsion(src_xy, obstacles, obs_margin, k_rep, sigma)

        flat = dict(ref_seq[0])
        flat['x']     = ref_pos + rep
        flat['x_dot'] = ref_seq[0]['x_dot']          # velocity ref unchanged
        flat['x_ddot'] = ref_seq[0].get('x_ddot', np.zeros(3)) + (-alpha_opt * r_ema)

        # ── SE3 execution ───────────────────────────────────────────────
        cmd   = ctrl.update(t, state, flat)
        state = vehicle.step(state, cmd, dt)

        positions[i] = state['x']
        des_pos[i]   = ref_pos              # log nominal (unmodified) reference
        errors[i]    = np.linalg.norm(positions[i] - des_pos[i])

        for center, radius in obstacles:
            if np.linalg.norm(positions[i, :2] - np.asarray(center)[:2]) < radius:
                collisions[i] = True
                break

        if verbose and i % 200 == 0:
            nc = collisions[:i+1].sum()
            print(f"  t={t:.1f}s  err={errors[i]:.3f}m  collisions={nc}", end='\r')

    if verbose:
        print()

    mean_hz = 1.0 / np.mean(t_compute) if t_compute else 0.
    return {
        'positions':  positions,
        'des_pos':    des_pos,
        'errors':     errors,
        'collisions': collisions,
        'mean_err':   float(errors[100:].mean()),
        'n_collide':  int(collisions.sum()),
        'mean_hz':    float(mean_hz),
    }


# ── Reference geometry for plotting ───────────────────────────────────────────

def _slanted_circle_ref(R=2.0, freq=0.15, tilt_deg=30., n=500):
    w  = 2 * np.pi * freq
    ct = np.cos(np.radians(tilt_deg))
    t  = np.linspace(0., 1. / freq, n)
    return R * np.cos(w * t), R * np.sin(w * t) * ct


def _line_ref(start, direction, length, n=300):
    d   = np.array(direction, dtype=float)
    d  /= np.linalg.norm(d)
    s   = np.linspace(0., length, n)
    pts = np.array(start[:2]) + np.outer(s, d[:2])
    return pts[:, 0], pts[:, 1]


# ── Figure ─────────────────────────────────────────────────────────────────────

def _add_obstacle_patches(ax, obstacles, obs_margin):
    OBS_COL    = '#111111'
    MARGIN_COL = '#aaaaaa'
    for (cx, cy, _), r in [(np.asarray(c), r) for c, r in obstacles]:
        # safety margin halo (grey, behind)
        ax.add_patch(plt.Circle((cx, cy), r + obs_margin,
                                color=MARGIN_COL, alpha=0.30, zorder=3))
        ax.add_patch(plt.Circle((cx, cy), r + obs_margin,
                                color=MARGIN_COL, alpha=0.65, zorder=3,
                                fill=False, lw=0.9))
        # physical obstacle (solid black)
        ax.add_patch(plt.Circle((cx, cy), r,
                                color=OBS_COL, alpha=1.0, zorder=5))


def _square_limits(ax, ref_xy, pos, obstacles, obs_margin, pad=0.5):
    """Set equal-aspect square limits encompassing everything."""
    all_x = list(ref_xy[0]) + list(pos[:, 0])
    all_y = list(ref_xy[1]) + list(pos[:, 1])
    for (c, r) in obstacles:
        all_x += [c[0] - r - obs_margin, c[0] + r + obs_margin]
        all_y += [c[1] - r - obs_margin, c[1] + r + obs_margin]
    xc = (min(all_x) + max(all_x)) / 2
    yc = (min(all_y) + max(all_y)) / 2
    half = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) / 2 + pad
    ax.set_xlim(xc - half, xc + half)
    ax.set_ylim(yc - half, yc + half)


def plot_minarik(res_a, ref_a, obs_a, margin_a,
                 res_b, ref_b, obs_b, margin_b,
                 wind_mag, save_path):

    REF_COL  = '#1F77B4'
    PINN_COL = '#FF7F0E'

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    entries = [
        (axes[0], res_a, ref_a, obs_a, margin_a, '(a) slanted circle'),
        (axes[1], res_b, ref_b, obs_b, margin_b, '(b) line'),
    ]
    for ax, res, ref_xy, obstacles, margin, label in entries:
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.18, zorder=0)

        # reference
        ax.plot(ref_xy[0], ref_xy[1],
                color=REF_COL, lw=1.4, ls='--', alpha=0.75,
                label='Reference', zorder=2)

        # PINN-MPPI trajectory (skip transient)
        pos  = res['positions']
        skip = 80
        ax.plot(pos[skip:, 0], pos[skip:, 1],
                color=PINN_COL, lw=2.0, alpha=0.92,
                label='PINN-MPPI + Obs.', zorder=4)
        ax.scatter(pos[skip, 0], pos[skip, 1],
                   s=45, c=PINN_COL, zorder=6,
                   edgecolors='k', linewidths=0.6)

        # obstacles
        _add_obstacle_patches(ax, obstacles, margin)
        _square_limits(ax, ref_xy, pos[skip:], obstacles, margin)

        ax.set_xlabel('x  [m]', fontsize=10)
        ax.set_ylabel('y  [m]', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold', pad=6)
        ax.tick_params(labelsize=9)

    # shared legend
    ref_ln  = matplotlib.lines.Line2D([], [], color=REF_COL,  lw=1.4, ls='--',
                                      label='Reference')
    pinn_ln = matplotlib.lines.Line2D([], [], color=PINN_COL, lw=2.0,
                                      label='PINN-MPPI + Obs. avoid.')
    obs_p   = mpatches.Patch(color='#111111', label='Obstacle')
    saf_p   = mpatches.Patch(color='#aaaaaa', alpha=0.45, label='Safety margin')
    fig.legend(handles=[ref_ln, pinn_ln, obs_p, saf_p],
               loc='lower center', ncol=4, fontsize=8.5,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.02))


    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Minarik-style 2-panel obstacle avoidance figure')
    p.add_argument('--wind',       type=float, default=8.0)
    p.add_argument('--K',          type=int,   default=896,
                   help='MPPI samples (default 896)')
    p.add_argument('--k-rep',      type=float, default=4.0,
                   help='potential-field repulsion strength [m] (default 4.0)')
    p.add_argument('--sim-time-a', type=float, default=25.)
    p.add_argument('--sim-time-b', type=float, default=22.)
    args = p.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    wind_vec = np.array([args.wind, 0., 0.])
    wind_mag = float(np.linalg.norm(wind_vec))

    print(f"\n{'='*60}")
    print(f"  Minarik-style figure  |  wind={args.wind} m/s  K={args.K}  "
          f"k_rep={args.k_rep}")
    print(f"{'='*60}\n")

    model, normalizer = load_pinn_model(CKPT_DIR)

    # ── Scenario A: Slanted circle + 3 obstacles ──────────────────────────────
    # R=2.5, tilt=30°, freq=0.12 Hz.
    # Obstacles placed 0.30 m INSIDE the ellipse along the outward ellipse normal
    # at θ=90°, 210°, 330°.  This puts the obstacle surface exactly on the
    # reference ellipse (dist from reference = r=0.30 m), so the blue dashed
    # line visually clips each obstacle while the drone deviates outward.
    #
    # Ellipse: x = R*cos(θ),  y = R*sin(θ)*cos30°,  a=R, b=R*cos30°
    # Outward normal at θ: n = (b*cos(θ), a*sin(θ)) / |(b*cos(θ), a*sin(θ))|
    # Reference point:  p = (R*cos(θ), R*sin(θ)*cos30°)
    # Obstacle center:  p - 0.30*n_hat
    R_a    = 2.5
    ct_a   = np.cos(np.radians(30.))
    r_obs_a = 0.30
    d_inset = 0.30   # inset distance (= physical radius, so ref just clips obstacle)

    def _ellipse_obstacle(theta_deg, R, ct, r_obs, d):
        """Obstacle centre d metres inside ellipse at angle theta, in 3D (z=1.5)."""
        θ    = np.radians(theta_deg)
        a, b = R, R * ct            # semi-axes
        ref  = np.array([a * np.cos(θ), b * np.sin(θ)])          # ref. point
        n_raw = np.array([b * np.cos(θ), a * np.sin(θ)])          # ellipse normal
        n    = n_raw / np.linalg.norm(n_raw)
        center = ref - d * n         # inset from reference toward centre
        return np.array([center[0], center[1], 1.5]), r_obs

    obs_a = [_ellipse_obstacle(θ, R_a, ct_a, r_obs_a, d_inset)
             for θ in [90., 210., 330.]]
    obs_margin_a = 0.30   # only used for eff_r in Gaussian (margin/2 added to radius)
    k_rep_a      = 1.5   # Gaussian peak repulsion [m] — wide smooth detour
    sigma_rep_a  = 0.85  # Gaussian std [m] — smooth activation zone

    def traj_fn_a():
        return SlantedCircleTraj(center=np.array([0., 0., 1.5]),
                                  R=R_a, freq=0.12, tilt_deg=30.)

    print("Running scenario A: slanted circle + 3 obstacles...")
    t0 = time.perf_counter()
    res_a = run_pf_episode(
        model, normalizer, wind_vec, traj_fn_a,
        obstacles=obs_a, obs_margin=obs_margin_a,
        k_rep=k_rep_a, sigma=sigma_rep_a,
        rep_source='reference',
        sim_time=args.sim_time_a, K=args.K, H=15, n_interp=10,
        use_pinn=True, verbose=False)
    print(f"  mean_err={res_a['mean_err']:.4f}m  "
          f"collisions={res_a['n_collide']}  "
          f"{res_a['mean_hz']:.0f} Hz  ({time.perf_counter()-t0:.1f}s)")
    ref_a = _slanted_circle_ref(R=R_a, freq=0.12, tilt_deg=30.)

    # ── Scenario B: Diagonal line + 2 obstacles ────────────────────────────────
    # Line: start=(0,0,1.5), dir=(1,1,0)/√2, speed=0.4 m/s, length=6 m.
    # Obstacles are placed SLIGHTLY OFF the diagonal (0.15 m to the lower-right,
    # i.e. in the +x/-y direction perpendicular to the diagonal).  This ensures
    # diff = ref_pos - obs_center is NEVER zero, preventing the singularity that
    # occurs when an obstacle is exactly on the reference line.
    #
    # Obs_1 at diagonal distance 2.0 m → nominal (√2, √2) + offset
    # Obs_2 at diagonal distance 4.5 m → nominal (4.5/√2, 4.5/√2) + offset
    # With r_obs=0.28m > offset=0.15m the diagonal still passes through each
    # obstacle, so the reference visually clips the black circles while the drone
    # deviates to the upper-left (away from obstacles).
    inv_rt2  = 1.0 / np.sqrt(2)
    perp_off = 0.15 * np.array([inv_rt2, -inv_rt2])   # lower-right perpendicular
    obs_b = [
        (np.array([2.0 * inv_rt2 + perp_off[0],
                   2.0 * inv_rt2 + perp_off[1], 1.5]),  0.28),
        (np.array([4.5 * inv_rt2 + perp_off[0],
                   4.5 * inv_rt2 + perp_off[1], 1.5]),  0.28),
    ]
    obs_margin_b = 0.25
    k_rep_b      = 1.5   # same as scenario A — well-calibrated Gaussian peak
    sigma_rep_b  = 0.85  # wider activation zone for smooth early avoidance

    def traj_fn_b():
        return LineTraj(start=np.array([0., 0., 1.5]),
                        direction=np.array([1., 1., 0.]),
                        speed=0.4, length=6.0)

    print("Running scenario B: diagonal line + 2 obstacles...")
    t0 = time.perf_counter()
    res_b = run_pf_episode(
        model, normalizer, wind_vec, traj_fn_b,
        obstacles=obs_b, obs_margin=obs_margin_b,
        k_rep=k_rep_b, sigma=sigma_rep_b,
        rep_source='reference',
        sim_time=args.sim_time_b, K=args.K, H=15, n_interp=10,
        use_pinn=True, verbose=False)
    print(f"  mean_err={res_b['mean_err']:.4f}m  "
          f"collisions={res_b['n_collide']}  "
          f"{res_b['mean_hz']:.0f} Hz  ({time.perf_counter()-t0:.1f}s)")
    ref_b = _line_ref(start=np.array([0., 0., 1.5]),
                      direction=np.array([1., 1., 0.]),
                      length=6.0)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'Scenario':<28} {'Mean err':>9} {'Collisions':>11} {'Hz':>6}")
    print('-' * 60)
    print(f"{'(a) Slanted circle':<28} {res_a['mean_err']:9.4f} "
          f"{res_a['n_collide']:11d} {res_a['mean_hz']:6.0f}")
    print(f"{'(b) Diagonal line':<28} {res_b['mean_err']:9.4f} "
          f"{res_b['n_collide']:11d} {res_b['mean_hz']:6.0f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    npy_path = os.path.join(RESULT_DIR, 'minarik_fig.npy')
    np.save(npy_path, {
        'res_a': res_a, 'ref_a': ref_a, 'obs_a': obs_a, 'margin_a': obs_margin_a,
        'res_b': res_b, 'ref_b': ref_b, 'obs_b': obs_b, 'margin_b': obs_margin_b,
        'wind_vec': wind_vec,
    }, allow_pickle=True)
    print(f"\nSaved: {npy_path}")

    png_path = os.path.join(RESULT_DIR, 'minarik_fig.png')
    plot_minarik(res_a, ref_a, obs_a, obs_margin_a,
                 res_b, ref_b, obs_b, obs_margin_b,
                 wind_mag, png_path)
    print("Done.")


if __name__ == '__main__':
    main()
