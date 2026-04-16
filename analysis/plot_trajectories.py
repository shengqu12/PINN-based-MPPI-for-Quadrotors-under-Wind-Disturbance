"""
plot_trajectories.py — 3D Trajectory Tracking Visualization

Generates three figures:
    Figure 1: 2 rows × 4 cols — SE3/Nominal MPPI vs PINN-MPPI (4 trajectories, wind=8)
    Figure 2: 1 row × 5 cols — PINN-MPPI circle at various wind speeds (0/4/8/10/12 m/s)
    Figure 3: 1 row × 2 cols — SE3 vs PINN-MPPI side-by-side (circle, wind=8)

NOTE: In our framework, SE3 Only == Nominal MPPI (same execution, no PINN feedforward).
      PINN-MPPI uses SE3Control + PINN feedforward (x_ddot += -0.75 * PINN(v=0, wind)).
"""

import numpy as np
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from controllers.pinn_mppi_v2 import run_episode, load_pinn_model
from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.trajectories.hover_traj import HoverTraj

CKPT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

_K_ETA      = quad_params['k_eta']
_MASS       = quad_params['mass']
_G          = 9.81
HOVER_OMEGA = float(np.sqrt(_MASS * _G / (4 * _K_ETA)))


# ── Trajectory definitions ─────────────────────────────────────────────────────

class FigureEightTraj:
    """Figure-eight (Lissajous): x=Ax*sin(wx*t), y=Ay*sin(2wx*t)"""
    def __init__(self, center=np.array([0.,0.,1.5]),
                 Ax=1.5, Ay=0.75, freq_x=0.15):
        self.cx, self.cy, self.cz = center
        self.Ax = Ax; self.Ay = Ay
        self.wx = 2 * np.pi * freq_x
        self.wy = 2 * self.wx

    def update(self, t):
        x  = self.cx + self.Ax * np.sin(self.wx * t)
        y  = self.cy + self.Ay * np.sin(self.wy * t)
        vx = self.Ax * self.wx * np.cos(self.wx * t)
        vy = self.Ay * self.wy * np.cos(self.wy * t)
        ax = -self.Ax * self.wx**2 * np.sin(self.wx * t)
        ay = -self.Ay * self.wy**2 * np.sin(self.wy * t)
        return {
            'x': np.array([x, y, self.cz]), 'x_dot': np.array([vx, vy, 0.]),
            'x_ddot': np.array([ax, ay, 0.]), 'x_dddot': np.zeros(3),
            'x_ddddot': np.zeros(3), 'yaw': 0., 'yaw_dot': 0.,
        }


class SlantedCircleTraj:
    """Slanted circle tilted 30° out of horizontal plane."""
    def __init__(self, center=np.array([0.,0.,1.5]),
                 R=1.5, freq=0.2, tilt_deg=30.):
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
            'x': np.array([x, y, z]), 'x_dot': np.array([vx, vy, vz]),
            'x_ddot': np.array([ax, ay, az]), 'x_dddot': np.zeros(3),
            'x_ddddot': np.zeros(3), 'yaw': 0., 'yaw_dot': 0.,
        }


# ── Simulation helpers ─────────────────────────────────────────────────────────

def simulate(model, normalizer, wind_vec, trajectory_fn,
             use_pinn=True, sim_time=15.0, K=896, H=15):
    """
    Wrapper around run_episode.
    Returns (positions (N,3), des_pos (N,3)).
    use_pinn=True  → SE3Control + PINN feedforward (x_ddot += -0.75 * PINN(v=0))
    use_pinn=False → SE3Control only, no feedforward (identical to SE3 Only)
    """
    r = run_episode(model, normalizer, wind_vec, trajectory_fn,
                    sim_time=sim_time, dt=0.01, K=K, H=H,
                    use_pinn=use_pinn, verbose=False)
    return r['positions'], r['des_pos']


def se3_only_sim(wind_vec, traj_fn, sim_time=15., dt=0.01):
    """Pure SE3Control with no MPPI and no PINN — fastest baseline."""
    state = {'x': np.array([0.,0.,1.5]), 'v': np.zeros(3),
             'q': np.array([0.,0.,0.,1.]), 'w': np.zeros(3),
             'wind': wind_vec.copy(), 'rotor_speeds': np.ones(4)*HOVER_OMEGA}
    vehicle = Multirotor(quad_params, state)
    ctrl    = SE3Control(quad_params)
    traj    = traj_fn()
    N = int(sim_time / dt)
    pos = np.zeros((N, 3)); ref = np.zeros((N, 3))
    for i in range(N):
        t = i * dt
        state['wind'] = wind_vec.copy()
        flat  = traj.update(t)
        cmd   = ctrl.update(t, state, flat)
        state = vehicle.step(state, cmd, dt)
        pos[i] = state['x']; ref[i] = flat['x']
    return pos, ref


# ── Plot helper ────────────────────────────────────────────────────────────────

def plot_traj(ax, ref, actual, color, label,
              xlim=None, ylim=None, zlim=None,
              elev=25, azim=-60):
    ax.plot(ref[:,0], ref[:,1], ref[:,2],
            color='#7EC8E3', linestyle='--', linewidth=1.0,
            alpha=0.7, label='Reference')
    ax.plot(actual[:,0], actual[:,1], actual[:,2],
            color=color, linewidth=1.8, alpha=0.9, label=label)
    ax.scatter(*actual[0], color=color, s=25, zorder=5)
    ax.set_xlabel('x [m]', fontsize=7, labelpad=1)
    ax.set_ylabel('y [m]', fontsize=7, labelpad=1)
    ax.set_zlabel('z [m]', fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.7)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.15)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if zlim: ax.set_zlim(zlim)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    model, normalizer = load_pinn_model(CKPT_DIR)

    wind8 = np.array([8., 0., 0.])

    trajs = {
        'hover':   lambda: HoverTraj(x0=np.array([0.,0.,1.5])),
        'circle':  lambda: ThreeDCircularTraj(
                       center=np.array([0.,0.,1.5]),
                       radius=np.array([1.5,1.5,0.]),
                       freq=np.array([0.2,0.2,0.])),
        'slanted': lambda: SlantedCircleTraj(
                       center=np.array([0.,0.,1.5]), R=1.5, freq=0.2, tilt_deg=30),
        'eight':   lambda: FigureEightTraj(
                       center=np.array([0.,0.,1.5]), Ax=1.5, Ay=0.75, freq_x=0.15),
    }
    traj_titles = {
        'hover': 'Hover', 'circle': 'Circle',
        'slanted': 'Slanted Circle', 'eight': 'Figure-Eight',
    }
    axis_limits = {
        'hover':   dict(xlim=[-0.8,1.2], ylim=[-0.5,0.5], zlim=[0.5,2.5]),
        'circle':  dict(xlim=[-2.5,3.0], ylim=[-2.5,2.5], zlim=[0.5,2.5]),
        'slanted': dict(xlim=[-2.5,3.0], ylim=[-2.5,2.5], zlim=[0.0,3.5]),
        'eight':   dict(xlim=[-2.5,3.0], ylim=[-2.0,2.0], zlim=[0.5,2.5]),
    }

    # ── Figure 1: No-PINN vs PINN-MPPI (2 rows × 4 cols) ─────────────────────
    print("Figure 1: No-PINN vs PINN-MPPI (4 trajectories, wind=8)...")
    fig1, axes1 = plt.subplots(2, 4, figsize=(18, 8),
                                subplot_kw={'projection': '3d'})
    fig1.suptitle(
        'SE3/Nominal MPPI (no PINN) vs PINN-MPPI under Wind Disturbance (8 m/s)',
        fontsize=13, fontweight='bold', y=1.01
    )

    for col, tname in enumerate(trajs.keys()):
        print(f"  Simulating {tname}...")
        lims     = axis_limits[tname]
        pos_nom, ref  = simulate(model, normalizer, wind8, trajs[tname],
                                 use_pinn=False, sim_time=7.)
        pos_pinn, _   = simulate(model, normalizer, wind8, trajs[tname],
                                 use_pinn=True, sim_time=7.)

        plot_traj(axes1[0, col], ref, pos_nom, '#5B9BD5',
                  'No PINN (SE3/Nom.MPPI)', **lims)
        axes1[0, col].set_title(
            f'({chr(97+col)}) No PINN\n({traj_titles[tname]})', fontsize=9)

        plot_traj(axes1[1, col], ref, pos_pinn, '#E87B2D',
                  'PINN-MPPI (ours)', **lims)
        axes1[1, col].set_title(
            f'({chr(97+4+col)}) PINN-MPPI\n({traj_titles[tname]})', fontsize=9)

    plt.tight_layout()
    p1 = os.path.join(RESULT_DIR, 'traj_comparison.png')
    fig1.savefig(p1, dpi=150, bbox_inches='tight')
    print(f"Saved: {p1}")

    # ── Figure 2: PINN-MPPI wind speed sweep (circle) ─────────────────────────
    print("\nFigure 2: wind speed sweep (circle)...")
    wind_list = [
        (np.array([0.,0.,0.]),  'wind=0 m/s'),
        (np.array([4.,0.,0.]),  'wind=4 m/s'),
        (np.array([8.,0.,0.]),  'wind=8 m/s'),
        (np.array([10.,0.,0.]), 'wind=10 m/s\n(OOD)'),
        (np.array([12.,0.,0.]), 'wind=12 m/s\n(OOD)'),
    ]
    fig2, axes2 = plt.subplots(1, 5, figsize=(22, 4),
                                subplot_kw={'projection': '3d'})
    fig2.suptitle('PINN-MPPI Circle Tracking at Various Wind Speeds',
                  fontsize=12, fontweight='bold')

    for col, (wv, wlabel) in enumerate(wind_list):
        print(f"  {wlabel.replace(chr(10),' ')}...")
        pos, ref = simulate(model, normalizer, wv, trajs['circle'],
                            use_pinn=True, sim_time=7.)
        xmax = 2.5 + float(np.linalg.norm(wv)) * 0.1
        plot_traj(axes2[col], ref, pos, '#E87B2D', 'PINN-MPPI',
                  xlim=[-xmax, xmax+0.5], ylim=[-2.5,2.5], zlim=[0.5,2.5])
        axes2[col].set_title(f'({chr(97+col)}) {wlabel}', fontsize=9)

    plt.tight_layout()
    p2 = os.path.join(RESULT_DIR, 'traj_wind_sweep.png')
    fig2.savefig(p2, dpi=150, bbox_inches='tight')
    print(f"Saved: {p2}")

    # ── Figure 3: SE3 vs PINN-MPPI side-by-side (circle, wind=8) ─────────────
    print("\nFigure 3: controller comparison (circle, wind=8)...")
    pos_se3,  ref_c = se3_only_sim(wind8, trajs['circle'], sim_time=7.)
    pos_pinn_c, _   = simulate(model, normalizer, wind8, trajs['circle'],
                                use_pinn=True, sim_time=7.)
    lims_c = axis_limits['circle']

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5),
                                subplot_kw={'projection': '3d'})
    fig3.suptitle('Controller Comparison: circle trajectory, wind=8 m/s',
                  fontsize=12, fontweight='bold')

    plot_traj(axes3[0], ref_c, pos_se3, '#5B9BD5',
              'SE3 / Nominal MPPI', **lims_c)
    axes3[0].set_title('(a) SE3 / Nominal MPPI  (no PINN feedforward)', fontsize=10)

    plot_traj(axes3[1], ref_c, pos_pinn_c, '#E87B2D',
              'PINN-MPPI (ours)', **lims_c)
    axes3[1].set_title('(b) PINN-MPPI  (SE3 + PINN x_ddot feedforward)', fontsize=10)

    plt.tight_layout()
    p3 = os.path.join(RESULT_DIR, 'traj_controller_compare.png')
    fig3.savefig(p3, dpi=150, bbox_inches='tight')
    print(f"Saved: {p3}")

    print(f"\nAll figures saved:\n  {p1}\n  {p2}\n  {p3}")


if __name__ == '__main__':
    main()
