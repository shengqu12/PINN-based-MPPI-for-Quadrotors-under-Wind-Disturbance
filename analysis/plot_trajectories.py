"""
plot_trajectories.py — 3D Trajectory Tracking Visualization

Fixes:
    1. Figure-eight trajectory starts from origin (phi=0)
    2. Fixed axis limits for Hover to avoid numerical noise
    3. Included slanted circle trajectory
    4. Unified perspective for all subplots
"""

import numpy as np
import torch
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.pinn import ResidualPINN
from training.dataset import Normalizer
from controllers.pinn_mppi_cpu import (MPPIController, pinn_predict,
                                    load_pinn_model, EMA_ALPHA, DP_CALC_MAX)

from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.trajectories.hover_traj import HoverTraj

CKPT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
K_ETA       = quad_params['k_eta']
MASS        = quad_params['mass']
G           = 9.81
HOVER_OMEGA = float(np.sqrt(MASS * G / (4 * K_ETA)))
KP_POS      = np.array([6.5, 6.5, 15.0])


# ── Trajectory Definitions ───────────────────────────────────────────

class FigureEightTraj:
    """
    Figure-eight trajectory (Lissajous), starting from the origin:
        x(t) = Ax * sin(wx * t)
        y(t) = Ay * sin(wy * t)    phi=0 -> starts from (0,0)
    """
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
            'x_ddddot': np.zeros(3), 'yaw': 0., 'yaw_dot': 0.
        }


class SlantedCircleTraj:
    """
    Slanted circle trajectory (aligned with Minarik Fig.5c):
        x(t) = R * cos(w * t)
        y(t) = R * sin(w * t) * cos(theta)
        z(t) = z0 + R * sin(w * t) * sin(theta)
    theta=30° -> Slight tilt, periodic change in z-axis
    """
    def __init__(self, center=np.array([0.,0.,1.5]),
                 R=1.5, freq=0.2, tilt_deg=30.0):
        self.cx, self.cy, self.cz = center
        self.R   = R
        self.w   = 2 * np.pi * freq
        self.ct  = np.cos(np.radians(tilt_deg))
        self.st  = np.sin(np.radians(tilt_deg))

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
            'x_ddddot': np.zeros(3), 'yaw': 0., 'yaw_dot': 0.
        }


# ── Simulation ───────────────────────────────────────────────────────

def simulate(model, normalizer, wind_vec, trajectory_fn,
             use_pinn=True, sim_time=15.0, dt=0.01, K=1000, H=20):
    _traj_init = trajectory_fn()
    start_pos  = _traj_init.update(0.0)["x"].copy()
    state = {
        "x":            start_pos,
        "v":            np.zeros(3),
        "q":            np.array([0.,0.,0.,1.]),
        "w":            np.zeros(3),
        "wind":         wind_vec.copy(),
        "rotor_speeds": np.ones(4) * HOVER_OMEGA,
    }
    vehicle    = Multirotor(quad_params, state)
    controller = SE3Control(quad_params)
    trajectory = trajectory_fn()
    mppi       = MPPIController(
        model, normalizer, wind_vec,
        K=K, H=H, dt=dt, use_pinn=use_pinn
    )

    N = int(sim_time/dt)
    positions = np.zeros((N,3))
    ref_pos   = np.zeros((N,3))
    t = 0.

    for i in range(N):
        state['wind'] = wind_vec.copy()
        flat_ref  = trajectory.update(t)
        vel_des   = flat_ref['x_dot']
        pds = np.array([trajectory.update(t + h * dt)['x'] for h in range(H)])
        vds = np.array([trajectory.update(t + h * dt)['x_dot'] for h in range(H)])
        mu  = np.array(state.get('rotor_speeds', np.ones(4) * HOVER_OMEGA))

        if use_pinn:
            res     = pinn_predict(model, normalizer, state, mu, wind_vec, vel_ref=vel_des)
            a, dpc, _ = mppi.update(state, pds, vds, mu, res)
            dp      = a * dpc
        else:
            _, dpc, _ = mppi.update(state, pds, vds, mu, np.zeros(3))
            dp      = np.zeros(3)

        fm = dict(flat_ref)
        fm['x'] = flat_ref['x'] + dp
        fm['x_dot'] = flat_ref['x_dot']
        cmd   = controller.update(t, state, fm)
        state = vehicle.step(state, cmd, dt)

        positions[i] = state['x']
        ref_pos[i]   = flat_ref['x']
        t += dt

    return positions, ref_pos


# ── Plotting Tools ───────────────────────────────────────────────────

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

    # Fix axis limits to avoid scaling issues from numerical noise
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if zlim: ax.set_zlim(zlim)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    os.makedirs(RESULT_DIR, exist_ok=True)
    model, normalizer = load_pinn_model()

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
        'hover':   'hover',
        'circle':  'circle',
        'slanted': 'slanted circle',
        'eight':   'figure-eight',
    }

    # Axis limits for each trajectory type
    axis_limits = {
        'hover':   dict(xlim=[-0.5,0.8], ylim=[-0.3,0.3], zlim=[0.5,2.5]),
        'circle':  dict(xlim=[-2.0,2.5], ylim=[-2.0,2.0], zlim=[0.5,2.5]),
        'slanted': dict(xlim=[-2.0,2.5], ylim=[-2.0,2.0], zlim=[0.0,3.5]),
        'eight':   dict(xlim=[-2.0,2.5], ylim=[-1.5,1.5], zlim=[0.5,2.5]),
    }

    # ── Figure 1: Nominal MPPI vs PINN-MPPI (2 rows × 4 columns) ────────
    print("Generating Figure 1: Nominal vs PINN-MPPI (4 Trajectories)...")
    fig1, axes1 = plt.subplots(
        2, 4, figsize=(18, 8),
        subplot_kw={'projection': '3d'}
    )
    fig1.suptitle(
        'Nominal MPPI vs PINN-MPPI under Wind Disturbance (8 m/s)',
        fontsize=13, fontweight='bold', y=1.01
    )

    for col, tname in enumerate(trajs.keys()):
        print(f"  Simulating {tname}...")
        lims = axis_limits[tname]

        pos_nom,  ref = simulate(model, normalizer, wind8,
                                  trajs[tname], use_pinn=False, sim_time=15.)
        pos_pinn, _   = simulate(model, normalizer, wind8,
                                  trajs[tname], use_pinn=True,  sim_time=15.)

        ax = axes1[0, col]
        plot_traj(ax, ref, pos_nom, '#5B9BD5', 'Nominal MPPI', **lims)
        ax.set_title(f'({chr(97+col)}) Nominal MPPI\n({traj_titles[tname]})',
                     fontsize=9)

        ax = axes1[1, col]
        plot_traj(ax, ref, pos_pinn, '#E87B2D', 'PINN-MPPI (ours)', **lims)
        ax.set_title(f'({chr(97+4+col)}) PINN-MPPI\n({traj_titles[tname]})',
                     fontsize=9)

    plt.tight_layout()
    p1 = os.path.join(RESULT_DIR, 'traj_comparison.png')
    fig1.savefig(p1, dpi=150, bbox_inches='tight')
    print(f"Figure 1 saved: {p1}")

    # ── Figure 2: PINN-MPPI under different wind speeds (circle) ────────
    print("\nGenerating Figure 2: Wind Speed Sweep...")
    wind_list = [
        (np.array([0.,0.,0.]),  'wind = 0 m/s'),
        (np.array([4.,0.,0.]),  'wind = 4 m/s'),
        (np.array([8.,0.,0.]),  'wind = 8 m/s'),
        (np.array([10.,0.,0.]), 'wind = 10 m/s (OOD)'),
        (np.array([12.,0.,0.]), 'wind = 12 m/s (OOD)'),
    ]

    fig2, axes2 = plt.subplots(
        1, 5, figsize=(22, 4),
        subplot_kw={'projection': '3d'}
    )
    fig2.suptitle(
        'PINN-MPPI Tracking under Various Wind Conditions (circle trajectory)',
        fontsize=12, fontweight='bold'
    )

    for col, (wv, wlabel) in enumerate(wind_list):
        print(f"  Simulating {wlabel}...")
        pos, ref = simulate(model, normalizer, wv,
                            trajs['circle'], use_pinn=True, sim_time=15.)
        # Expand X range as wind speed increases
        xmax = 2.0 + float(np.linalg.norm(wv)) * 0.1
        ax = axes2[col]
        plot_traj(ax, ref, pos, '#E87B2D', 'PINN-MPPI',
                  xlim=[-xmax, xmax+0.5], ylim=[-2.,2.], zlim=[0.5,2.5])
        ax.set_title(f'({chr(97+col)}) {wlabel}', fontsize=9)

    plt.tight_layout()
    p2 = os.path.join(RESULT_DIR, 'traj_wind_sweep.png')
    fig2.savefig(p2, dpi=150, bbox_inches='tight')
    print(f"Figure 2 saved: {p2}")

    # ── Figure 3: Four Controllers Comparison (circle, wind=8) ──────────
    print("\nGenerating Figure 3: Controller Comparison...")

    def se3_only_sim(wind_vec, traj_fn, sim_time=15., dt=0.01):
        state = {'x':np.array([0.,0.,1.5]),'v':np.zeros(3),
                 'q':np.array([0.,0.,0.,1.]),'w':np.zeros(3),
                 'wind':wind_vec.copy(),'rotor_speeds':np.ones(4)*HOVER_OMEGA}
        vehicle = Multirotor(quad_params, state)
        ctrl    = SE3Control(quad_params)
        traj    = traj_fn()
        N       = int(sim_time/dt)
        pos = np.zeros((N,3)); ref = np.zeros((N,3)); t=0.
        for i in range(N):
            state['wind'] = wind_vec.copy()
            flat = traj.update(t)
            cmd  = ctrl.update(t, state, flat)
            state = vehicle.step(state, cmd, dt)
            pos[i]=state['x']; ref[i]=flat['x']; t+=dt
        return pos, ref

    def oracle_sim(model, normalizer, wind_vec, traj_fn, sim_time=15., dt=0.01):
        state = {'x':np.array([0.,0.,1.5]),'v':np.zeros(3),
                 'q':np.array([0.,0.,0.,1.]),'w':np.zeros(3),
                 'wind':wind_vec.copy(),'rotor_speeds':np.ones(4)*HOVER_OMEGA}
        vehicle = Multirotor(quad_params, state)
        ctrl    = SE3Control(quad_params)
        traj    = traj_fn()
        N = int(sim_time/dt)
        pos=np.zeros((N,3)); ref=np.zeros((N,3)); res_ema=None; t=0.
        for i in range(N):
            state['wind'] = wind_vec.copy()
            flat    = traj.update(t)
            vel_des = flat['x_dot']
            mu      = np.array(state.get('rotor_speeds', np.ones(4)*HOVER_OMEGA))
            r       = pinn_predict(model, normalizer, state, mu,
                                   wind_vec, vel_ref=vel_des)
            if res_ema is None: res_ema=r.copy()
            else: res_ema = EMA_ALPHA*res_ema+(1-EMA_ALPHA)*r
            dp = np.clip(-res_ema/KP_POS, -DP_CALC_MAX, DP_CALC_MAX)
            fm=dict(flat); fm['x']=flat['x']+dp; fm['x_dot']=flat['x_dot']
            cmd   = ctrl.update(t, state, fm)
            state = vehicle.step(state, cmd, dt)
            pos[i]=state['x']; ref[i]=flat['x']; t+=dt
        return pos, ref

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5),
                                subplot_kw={'projection': '3d'})
    fig3.suptitle('Controller Comparison (circle, wind=8 m/s)',
                  fontsize=12, fontweight='bold')

    lims_c = axis_limits['circle']

    print("  Simulating SE3 only...")
    pos_se3, ref_c = se3_only_sim(wind8, trajs['circle'])
    print("  Simulating Oracle...")
    pos_ora, _     = oracle_sim(model, normalizer, wind8, trajs['circle'])
    print("  Simulating Nominal MPPI...")
    pos_nom_c, _   = simulate(model, normalizer, wind8, trajs['circle'],
                               use_pinn=False)
    print("  Simulating PINN-MPPI...")
    pos_pinn_c, _  = simulate(model, normalizer, wind8, trajs['circle'],
                               use_pinn=True)

    # Left: SE3 vs Oracle
    ax = axes3[0]
    ax.plot(ref_c[:,0], ref_c[:,1], ref_c[:,2],
            color='#7EC8E3', linestyle='--', linewidth=1., alpha=0.7,
            label='Reference')
    ax.plot(pos_se3[:,0], pos_se3[:,1], pos_se3[:,2],
            color='#5B9BD5', linewidth=1.8, label='SE3 only')
    ax.plot(pos_ora[:,0], pos_ora[:,1], pos_ora[:,2],
            color='#70AD47', linewidth=1.8, label='Oracle (alpha=1)')
    ax.set_title('(a) SE3 only vs Oracle', fontsize=10)
    ax.legend(fontsize=8); ax.view_init(25,-60); ax.grid(True, alpha=0.15)
    ax.set_xlabel('x [m]',fontsize=8); ax.set_ylabel('y [m]',fontsize=8)
    ax.set_zlabel('z [m]',fontsize=8); ax.tick_params(labelsize=7)
    ax.set_xlim(lims_c['xlim']); ax.set_ylim(lims_c['ylim'])
    ax.set_zlim(lims_c['zlim'])

    # Right: Nominal vs PINN-MPPI
    ax = axes3[1]
    ax.plot(ref_c[:,0], ref_c[:,1], ref_c[:,2],
            color='#7EC8E3', linestyle='--', linewidth=1., alpha=0.7,
            label='Reference')
    ax.plot(pos_nom_c[:,0], pos_nom_c[:,1], pos_nom_c[:,2],
            color='#5B9BD5', linewidth=1.8, alpha=0.7, label='Nominal MPPI')
    ax.plot(pos_pinn_c[:,0], pos_pinn_c[:,1], pos_pinn_c[:,2],
            color='#E87B2D', linewidth=1.8, label='PINN-MPPI (ours)')
    ax.set_title('(b) Nominal MPPI vs PINN-MPPI', fontsize=10)
    ax.legend(fontsize=8); ax.view_init(25,-60); ax.grid(True, alpha=0.15)
    ax.set_xlabel('x [m]',fontsize=8); ax.set_ylabel('y [m]',fontsize=8)
    ax.set_zlabel('z [m]',fontsize=8); ax.tick_params(labelsize=7)
    ax.set_xlim(lims_c['xlim']); ax.set_ylim(lims_c['ylim'])
    ax.set_zlim(lims_c['zlim'])

    plt.tight_layout()
    p3 = os.path.join(RESULT_DIR, 'traj_controller_compare.png')
    fig3.savefig(p3, dpi=150, bbox_inches='tight')
    print(f"Figure 3 saved: {p3}")

    print(f"\nAll tasks completed!\n  {p1}\n  {p2}\n  {p3}")


if __name__ == '__main__':
    main()