"""
plot_trajectories.py — Fig 4: 2D Top-View Trajectories + Obstacle Avoidance

Layout (~7.2 × 2.8 inches, 1 row × 4 cols):
    (a)  wind=0 m/s   — SE3 vs PINN-MPPI, circle
    (b)  wind=8 m/s   — SE3 vs PINN-MPPI, circle  (training limit)
    (c)  wind=12 m/s  — SE3 vs PINN-MPPI, circle  (OOD)
    (d)  Obstacle avoidance — SE3 / PINN / PINN+Obs, wind=8 m/s

Run:
    conda run -n drone python analysis/plot_trajectories.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from analysis.plot_style import apply_style, COLORS, C_SE3, C_NOM, C_PINN, panel_label, save

apply_style()

BASE = os.path.join(os.path.dirname(__file__), '..')
OUT  = os.path.join(BASE, 'results', 'fig4_traj_obstacle')

# ── Try importing simulation helpers ──────────────────────────────────────────
try:
    from controllers.pinn_mppi_v2 import run_episode, load_pinn_model
    from controllers.pinn_mppi_obstacle import run_obstacle_episode
    from rotorpy.vehicles.hummingbird_params import quad_params
    from rotorpy.vehicles.multirotor import Multirotor
    from rotorpy.controllers.quadrotor_control import SE3Control
    from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
    import torch

    _K_ETA      = quad_params['k_eta']
    _MASS       = quad_params['mass']
    HOVER_OMEGA = float(np.sqrt(_MASS * 9.81 / (4 * _K_ETA)))
    CKPT_DIR    = os.path.join(BASE, 'checkpoints')

    model, normalizer = load_pinn_model(CKPT_DIR)

    def _circle_traj():
        return ThreeDCircularTraj(
            center=np.array([0., 0., 1.5]),
            radius=np.array([1.5, 1.5, 0.]),
            freq=np.array([0.2, 0.2, 0.]))

    def simulate_pinn(wind_vec, sim_time=8.):
        r = run_episode(model, normalizer, wind_vec, _circle_traj,
                        sim_time=sim_time, dt=0.01, K=896, H=15,
                        use_pinn=True, verbose=False)
        return r['positions'], r['des_pos']

    def simulate_se3(wind_vec, sim_time=8.):
        state = {'x': np.array([0., 0., 1.5]), 'v': np.zeros(3),
                 'q': np.array([0., 0., 0., 1.]), 'w': np.zeros(3),
                 'wind': wind_vec.copy(),
                 'rotor_speeds': np.ones(4) * HOVER_OMEGA}
        vehicle = Multirotor(quad_params, state)
        ctrl    = SE3Control(quad_params)
        traj    = _circle_traj()
        N       = int(sim_time / 0.01)
        pos     = np.zeros((N, 3)); ref = np.zeros((N, 3))
        for i in range(N):
            t          = i * 0.01
            state['wind'] = wind_vec.copy()
            flat       = traj.update(t)
            cmd        = ctrl.update(t, state, flat)
            state      = vehicle.step(state, cmd, 0.01)
            pos[i]     = state['x']; ref[i] = flat['x']
        return pos, ref

    # Obstacle: cylinder at (1.9, 0.0), r=0.25 m
    _OBS_LIST = [(np.array([1.9, 0.0, 1.5]), 0.25)]

    def simulate_obs_pinn(wind_vec, sim_time=8.):
        """PINN-MPPI tracking only, no obstacle cost."""
        r = run_obstacle_episode(model, normalizer, wind_vec, _circle_traj,
                                 obstacles=[],
                                 sim_time=sim_time, dt=0.01, K=896, H=15,
                                 use_pinn=True, verbose=False)
        return r['positions'], r['des_pos']

    def simulate_obs_avoid(wind_vec, sim_time=8.):
        """PINN-MPPI + obstacle avoidance cost."""
        r = run_obstacle_episode(model, normalizer, wind_vec, _circle_traj,
                                 obstacles=_OBS_LIST,
                                 sim_time=sim_time, dt=0.01, K=896, H=15,
                                 use_pinn=True, c_obs=3000., obs_margin=0.30,
                                 verbose=False)
        return r['positions'], r['des_pos']

    SIM_AVAILABLE = True
    print("Simulation available — running live episodes.")

except ImportError as e:
    SIM_AVAILABLE = False
    print(f"Simulation not available ({e}). Loading saved .npz data.")


# ── Load pre-saved data (fallback) ────────────────────────────────────────────
def load_saved(tag):
    path = os.path.join(BASE, 'results', f'traj_{tag}.npz')
    if os.path.exists(path):
        d = np.load(path)
        return d['pos'], d['ref']
    return None, None


def get_traj(tag, wind_vec, mode='pinn', sim_time=8.):
    if SIM_AVAILABLE:
        if mode == 'pinn':
            return simulate_pinn(wind_vec, sim_time)
        elif mode == 'se3':
            return simulate_se3(wind_vec, sim_time)
        elif mode == 'obs_pinn':
            return simulate_obs_pinn(wind_vec, sim_time)
        elif mode == 'obs_avoid':
            return simulate_obs_avoid(wind_vec, sim_time)
    return load_saved(tag)


# ── Plot helpers ──────────────────────────────────────────────────────────────
def _draw_traj_2d(ax, ref, actual, color, label, lw=1.6):
    if ref is not None:
        ax.plot(ref[:, 0], ref[:, 1],
                color='#BBBBBB', lw=1.0, ls='--', label='Reference', zorder=2)
    if actual is not None:
        ax.plot(actual[:, 0], actual[:, 1],
                color=color, lw=lw, label=label, zorder=3)
        ax.scatter(actual[0, 0], actual[0, 1],
                   color=color, s=18, zorder=5, edgecolors='white', linewidths=0.5)


def _format_2d(ax, title, wind_ms, xlim=(-2.3, 2.8), ylim=(-2.2, 2.2)):
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', labelpad=2)
    ax.set_ylabel('y (m)', labelpad=2)
    ax.set_title(title, pad=4, fontweight='bold')
    # wind direction arrow (top-right corner)
    ax.annotate('', xy=(xlim[1] - 0.15, ylim[1] - 0.35),
                xytext=(xlim[1] - 0.85, ylim[1] - 0.35),
                arrowprops=dict(arrowstyle='->', color=COLORS['sky'],
                                lw=1.2, mutation_scale=10))
    ax.text(xlim[1] - 0.50, ylim[1] - 0.15,
            f'{wind_ms} m/s', fontsize=6, color=COLORS['sky'], ha='center')
    ax.yaxis.grid(True, linestyle='--', linewidth=0.35, alpha=0.5, zorder=0)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.35, alpha=0.5, zorder=0)


def _error_box(ax, lines):
    ax.text(0.03, 0.03, '\n'.join(lines),
            transform=ax.transAxes, fontsize=5.5,
            color=COLORS['text_dark'], va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec='#cccccc', lw=0.5))


def _rmse(pos, ref):
    if pos is None or ref is None:
        return '—'
    L = min(len(pos), len(ref))
    return f'{np.mean(np.linalg.norm(pos[:L] - ref[:L], axis=1)):.3f} m'


# ── Run / load simulations ─────────────────────────────────────────────────────
w0  = np.array([0., 0., 0.])
w8  = np.array([8., 0., 0.])
w12 = np.array([12., 0., 0.])

print("Panel (a): wind=0 …")
pos_se3_0,  ref0  = get_traj('se3_w0',  w0,  mode='se3')
pos_pinn_0, _     = get_traj('pinn_w0', w0,  mode='pinn')

print("Panel (b): wind=8 …")
pos_se3_8,  ref8  = get_traj('se3_w8',  w8,  mode='se3')
pos_pinn_8, _     = get_traj('pinn_w8', w8,  mode='pinn')

print("Panel (c): wind=12 (OOD) …")
pos_se3_12,  ref12 = get_traj('se3_w12',  w12, mode='se3')
pos_pinn_12, _     = get_traj('pinn_w12', w12, mode='pinn')

print("Panel (d): obstacle avoidance …")
pos_se3_obs,   ref_obs = get_traj('se3_obs',    w8, mode='se3')
pos_pinn_obs,  _       = get_traj('pinn_obs',   w8, mode='obs_pinn')
pos_avoid_obs, _       = get_traj('pinn_av',    w8, mode='obs_avoid')

# ── Figure layout (1 × 4) ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.9),
                          gridspec_kw={'wspace': 0.42})

# ── (a) wind = 0 m/s ─────────────────────────────────────────────────────────
ax = axes[0]
_draw_traj_2d(ax, ref0,  pos_se3_0,  C_SE3,  'SE3 Only')
_draw_traj_2d(ax, None,  pos_pinn_0, C_PINN, 'PINN-MPPI (ours)')
_format_2d(ax, 'No Wind', wind_ms=0)
panel_label(ax, 'a')
ax.legend(fontsize=5.8, loc='lower right', framealpha=0.9,
          handlelength=1.4, borderpad=0.4, edgecolor='#cccccc')
_error_box(ax, [f'SE3:  {_rmse(pos_se3_0,  ref0)}',
                f'PINN: {_rmse(pos_pinn_0, ref0)}'])

# ── (b) wind = 8 m/s ─────────────────────────────────────────────────────────
ax = axes[1]
_draw_traj_2d(ax, ref8,  pos_se3_8,  C_SE3,  'SE3 Only')
_draw_traj_2d(ax, None,  pos_pinn_8, C_PINN, 'PINN-MPPI (ours)')
_format_2d(ax, 'Wind 8 m/s\n(train limit)', wind_ms=8)
panel_label(ax, 'b')
_error_box(ax, [f'SE3:  {_rmse(pos_se3_8,  ref8)}',
                f'PINN: {_rmse(pos_pinn_8, ref8)}'])

# ── (c) wind = 12 m/s (OOD) ──────────────────────────────────────────────────
ax = axes[2]
_draw_traj_2d(ax, ref12, pos_se3_12,  C_SE3,  'SE3 Only')
_draw_traj_2d(ax, None,  pos_pinn_12, C_PINN, 'PINN-MPPI (ours)')
_format_2d(ax, 'Wind 12 m/s (OOD)', wind_ms=12)
ax.text(0.97, 0.97, 'OOD', transform=ax.transAxes, fontsize=7,
        color='white', fontweight='bold', ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.25', fc=COLORS['vermillion'], ec='none'))
panel_label(ax, 'c')
_error_box(ax, [f'SE3:  {_rmse(pos_se3_12,  ref12)}',
                f'PINN: {_rmse(pos_pinn_12, ref12)}'])

# ── (d) Obstacle avoidance ────────────────────────────────────────────────────
ax = axes[3]
OBS_C  = np.array([1.9, 0.0])
OBS_R  = 0.25
OBS_M  = 0.30

_draw_traj_2d(ax, ref_obs,   pos_se3_obs,   C_SE3,            'SE3 Only')
_draw_traj_2d(ax, None,      pos_pinn_obs,  COLORS['amber'],  'PINN-MPPI')
_draw_traj_2d(ax, None,      pos_avoid_obs, COLORS['green'],  'PINN+Obs')

ax.add_patch(Circle(OBS_C, OBS_R,
                    fc='#CC3311', ec='#990000', lw=0.8, zorder=6))
ax.add_patch(Circle(OBS_C, OBS_R + OBS_M,
                    fc='none', ec='#CC3311', lw=0.8, ls='--', zorder=5))

_format_2d(ax, 'Obstacle Avoidance\n(wind 8 m/s)', wind_ms=8)
panel_label(ax, 'd')
ax.legend(fontsize=5.5, loc='lower left', framealpha=0.9,
          handlelength=1.4, borderpad=0.4, edgecolor='#cccccc')
_error_box(ax, [f'SE3:   {_rmse(pos_se3_obs,   ref_obs)}',
                f'PINN:  {_rmse(pos_pinn_obs,  ref_obs)}',
                f'Avoid: {_rmse(pos_avoid_obs, ref_obs)}'])

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle('Trajectory Tracking and Obstacle Avoidance under Wind Disturbance',
             fontsize=10, fontweight='bold', y=1.02)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT), exist_ok=True)
save(fig, OUT)
