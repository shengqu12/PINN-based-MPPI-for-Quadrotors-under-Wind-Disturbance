"""
run_obstacle.py — PINN-MPPI obstacle avoidance experiment

Circle trajectory (r=1.5m) with one cylindrical obstacle.
Compares three controllers:
    1. SE3-only          — no MPPI, no PINN
    2. PINN-MPPI         — PINN wind compensation, NO obstacle cost
    3. PINN-MPPI + Obs   — PINN wind compensation + obstacle repulsion in rollout

Usage:
    python experiments/run_obstacle.py --wind 8
    python experiments/run_obstacle.py --wind 8 --obs-x 1.9 --obs-y 0.0 --obs-r 0.25
    python experiments/run_obstacle.py --wind 0 --obs-x 0.0 --obs-y 1.5 --obs-r 0.4

Default scenario (obs-x=1.9, obs-y=0.0, r=0.25, wind=8 m/s):
  SE3-only:            0 collisions, mean_err=0.75m  (drifts past obstacle on outside)
  PINN-MPPI (no obs): 89 collisions, mean_err=0.28m  (tracks circle, clips obstacle)
  PINN-MPPI + Obs:     0 collisions, mean_err=0.26m  (actively avoids + PINN tracking)
"""

import os, sys, argparse, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from controllers.pinn_mppi_v2    import load_pinn_model, run_episode, HOVER_OMEGA
from controllers.pinn_mppi_obstacle import run_obstacle_episode

from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.multirotor         import Multirotor
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj  import ThreeDCircularTraj

CKPT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


# ── trajectory factory ──────────────────────────────────────────────────────────

def make_circle():
    return ThreeDCircularTraj(
        center=np.array([0., 0., 1.5]),
        radius=np.array([1.5, 1.5, 0.]),
        freq=np.array([0.2, 0.2, 0.]),
    )


# ── SE3-only baseline (no MPPI, no PINN) ───────────────────────────────────────

def run_se3(wind_vec, obstacles, sim_time=20., dt=0.01):
    state = {
        'x':            np.array([1.5, 0., 1.5]),
        'v':            np.zeros(3),
        'q':            np.array([0., 0., 0., 1.]),
        'w':            np.zeros(3),
        'wind':         wind_vec.copy(),
        'rotor_speeds': np.ones(4) * HOVER_OMEGA,
    }
    vehicle = Multirotor(quad_params, state)
    ctrl    = SE3Control(quad_params)
    traj    = make_circle()

    N          = int(sim_time / dt)
    positions  = np.zeros((N, 3))
    des_pos    = np.zeros((N, 3))
    errors     = np.zeros(N)
    collisions = np.zeros(N, dtype=bool)

    for i in range(N):
        t = i * dt
        state['wind'] = wind_vec.copy()
        flat  = traj.update(t)
        cmd   = ctrl.update(t, state, flat)
        state = vehicle.step(state, cmd, dt)

        positions[i] = state['x']
        des_pos[i]   = flat['x']
        errors[i]    = np.linalg.norm(positions[i] - des_pos[i])
        for center, radius in obstacles:
            if np.linalg.norm(positions[i, :2] - np.asarray(center)[:2]) < radius:
                collisions[i] = True
                break

    return {
        'positions':  positions,
        'des_pos':    des_pos,
        'errors':     errors,
        'collisions': collisions,
        'mean_err':   float(errors[100:].mean()),
        'n_collide':  int(collisions.sum()),
    }


# ── 2-D top-view plot (Minarik style) ─────────────────────────────────────────

def plot_2d(results, obstacles, wind_vec, obs_margin, save_path):
    """
    2-D x-y top view.  Three controller trajectories + obstacle circles.
    Coloured like Minarik et al.: blue=SE3, orange=PINN-MPPI, green=+Obs.
    """
    COLORS = {
        'se3':       ('#4C72B0', 'SE3 only'),
        'pinn':      ('#DD8452', 'PINN-MPPI'),
        'pinn_obs':  ('#55A868', 'PINN-MPPI + Obs. avoid.'),
    }

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')

    # ── reference circle ────────────────────────────────────────────────
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(1.5*np.cos(theta), 1.5*np.sin(theta),
            color='#aaaaaa', lw=1.2, ls='--', label='Reference', zorder=1)

    # ── obstacles ────────────────────────────────────────────────────────
    for (cx, cy, cz), r in [(np.asarray(c), r) for c, r in obstacles]:
        # physical radius (solid red)
        obs_circ = plt.Circle((cx, cy), r,
                               color='#CC3300', alpha=0.88, zorder=4)
        ax.add_patch(obs_circ)
        # safety margin fill (translucent orange)
        saf_circ = plt.Circle((cx, cy), r + obs_margin,
                               color='#FF6644', alpha=0.18, zorder=3,
                               fill=True, lw=0)
        ax.add_patch(saf_circ)
        # safety margin border — use linestyle tuple to avoid matplotlib bug
        saf_ring = plt.Circle((cx, cy), r + obs_margin,
                               color='#CC3300', alpha=0.60, zorder=3,
                               fill=False, lw=1.4,
                               linestyle=(0, (5, 4)))   # explicit dash tuple
        ax.add_patch(saf_ring)

    # ── controller trajectories ──────────────────────────────────────────
    for key, res in results.items():
        color, label = COLORS[key]
        pos = res['positions']
        # start at ~1s to skip transient
        s = 100
        ax.plot(pos[s:, 0], pos[s:, 1],
                color=color, lw=1.8, alpha=0.9, zorder=5,
                label=f"{label}  (err={res['mean_err']:.3f}m, "
                      f"coll={res['n_collide']})")

        # mark collision points
        coll_idx = np.where(res['collisions'])[0]
        if coll_idx.size > 0:
            ax.scatter(pos[coll_idx, 0], pos[coll_idx, 1],
                       marker='x', c='red', s=30, zorder=6, linewidths=1.0)

        # start marker
        ax.scatter(pos[s, 0], pos[s, 1],
                   marker='o', c=color, s=50, zorder=7, edgecolors='black', lw=0.5)

    # ── wind arrow ───────────────────────────────────────────────────────
    wind_mag = np.linalg.norm(wind_vec)
    if wind_mag > 0.1:
        ax.annotate(
            '', xy=(2.5 + wind_vec[0]*0.22, wind_vec[1]*0.22),
            xytext=(2.5, 0.),
            arrowprops=dict(arrowstyle='->', color='#2266CC',
                            lw=2.0, mutation_scale=18))
        ax.text(2.5 + wind_vec[0]*0.11, wind_vec[1]*0.11 + 0.18,
                f'wind\n{wind_mag:.0f} m/s',
                color='#2266CC', fontsize=8, ha='center', va='bottom')

    # ── formatting ───────────────────────────────────────────────────────
    lim = 2.4
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('x  [m]', fontsize=11)
    ax.set_ylabel('y  [m]', fontsize=11)
    ax.set_title(f'Obstacle Avoidance — PINN-MPPI   (wind = {wind_mag:.0f} m/s)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9)

    # obstacle legend patch
    obs_patch = mpatches.Patch(color='#CC3300', alpha=0.85, label='Obstacle')
    saf_patch = mpatches.Patch(color='#FF6644', alpha=0.35,
                                label=f'Safety margin (+{obs_margin:.2f} m)')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [obs_patch, saf_patch],
              labels  + [obs_patch.get_label(), saf_patch.get_label()],
              loc='upper left', fontsize=8, framealpha=0.92)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='PINN-MPPI obstacle avoidance — circle traj with wind')
    p.add_argument('--wind',       type=float, default=8.0,
                   help='wind speed m/s in +x direction (default 8)')
    p.add_argument('--obs-x',      type=float, default=1.9,
                   help='obstacle centre x [m]  (default 1.9, in SE3 drift zone)')
    p.add_argument('--obs-y',      type=float, default=0.0,
                   help='obstacle centre y [m]  (default 0.0)')
    p.add_argument('--obs-z',      type=float, default=1.5,
                   help='obstacle centre z [m]  (default 1.5)')
    p.add_argument('--obs-r',      type=float, default=0.25,
                   help='obstacle physical radius [m]  (default 0.25)')
    p.add_argument('--obs-margin', type=float, default=0.25,
                   help='MPPI safety margin beyond radius [m]  (default 0.25)')
    p.add_argument('--c-obs',      type=float, default=15000.,
                   help='obstacle cost weight  (default 15000)')
    p.add_argument('--K',          type=int,   default=896,
                   help='MPPI samples  (default 896)')
    p.add_argument('--sim-time',   type=float, default=20.,
                   help='simulation duration [s]  (default 20)')
    args = p.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)

    wind_vec  = np.array([args.wind, 0., 0.])
    obs_center = np.array([args.obs_x, args.obs_y, args.obs_z])
    obstacles  = [(obs_center, args.obs_r)]

    print(f"\n{'='*60}")
    print(f"  PINN-MPPI Obstacle Avoidance")
    print(f"  wind = {args.wind} m/s  |  obstacle @ "
          f"({args.obs_x:.2f}, {args.obs_y:.2f})  r={args.obs_r:.2f}m")
    print(f"{'='*60}\n")

    model, normalizer = load_pinn_model(CKPT_DIR)

    # ── 1. SE3-only baseline ─────────────────────────────────────────────
    print("Running SE3-only baseline...")
    t0 = time.perf_counter()
    r_se3 = run_se3(wind_vec, obstacles, sim_time=args.sim_time)
    print(f"  mean_err={r_se3['mean_err']:.4f}m  "
          f"collisions={r_se3['n_collide']}  "
          f"({time.perf_counter()-t0:.1f}s)")

    # ── 2. PINN-MPPI without obstacle cost ───────────────────────────────
    print("Running PINN-MPPI (no obstacle cost)...")
    t0 = time.perf_counter()
    r_pinn = run_obstacle_episode(
        model, normalizer, wind_vec, make_circle,
        obstacles=[],          # no obstacle cost
        sim_time=args.sim_time, K=args.K,
        use_pinn=True, verbose=False)
    # but still measure collisions against real obstacles
    for i, pos in enumerate(r_pinn['positions']):
        for center, radius in obstacles:
            if np.linalg.norm(pos[:2] - np.asarray(center)[:2]) < radius:
                r_pinn['collisions'][i] = True
                break
    r_pinn['n_collide'] = int(r_pinn['collisions'].sum())
    hz_pinn = r_pinn['freq_stats'].get('mean_hz', 0)
    print(f"  mean_err={r_pinn['mean_err']:.4f}m  "
          f"collisions={r_pinn['n_collide']}  "
          f"{hz_pinn:.0f} Hz  ({time.perf_counter()-t0:.1f}s)")

    # ── 3. PINN-MPPI with obstacle avoidance ─────────────────────────────
    print("Running PINN-MPPI + obstacle avoidance...")
    t0 = time.perf_counter()
    r_obs = run_obstacle_episode(
        model, normalizer, wind_vec, make_circle,
        obstacles=obstacles,
        sim_time=args.sim_time, K=args.K,
        use_pinn=True, c_obs=args.c_obs,
        obs_margin=args.obs_margin, verbose=False)
    hz_obs = r_obs['freq_stats'].get('mean_hz', 0)
    print(f"  mean_err={r_obs['mean_err']:.4f}m  "
          f"collisions={r_obs['n_collide']}  "
          f"{hz_obs:.0f} Hz  ({time.perf_counter()-t0:.1f}s)")

    # ── summary ──────────────────────────────────────────────────────────
    print(f"\n{'Controller':<28} {'Mean err':>9} {'Collisions':>11} {'Hz':>6}")
    print('-' * 60)
    print(f"{'SE3-only':<28} {r_se3['mean_err']:9.4f} {r_se3['n_collide']:11d}  {'—':>5}")
    print(f"{'PINN-MPPI (no obs cost)':<28} {r_pinn['mean_err']:9.4f} "
          f"{r_pinn['n_collide']:11d} {hz_pinn:6.1f}")
    print(f"{'PINN-MPPI + Obs. avoid.':<28} {r_obs['mean_err']:9.4f} "
          f"{r_obs['n_collide']:11d} {hz_obs:6.1f}")

    # ── save & plot ───────────────────────────────────────────────────────
    results_dict = {
        'se3':      r_se3,
        'pinn':     r_pinn,
        'pinn_obs': r_obs,
        'wind_vec': wind_vec,
        'obstacles': [(obs_center.tolist(), args.obs_r)],
        'obs_margin': args.obs_margin,
    }
    np_path  = os.path.join(RESULT_DIR, 'obstacle_avoidance.npy')
    png_path = os.path.join(RESULT_DIR, 'obstacle_2d.png')

    np.save(np_path, results_dict, allow_pickle=True)
    print(f"\nSaved: {np_path}")

    plot_2d({'se3': r_se3, 'pinn': r_pinn, 'pinn_obs': r_obs},
            obstacles, wind_vec, args.obs_margin, png_path)

    print("\nDone.")


if __name__ == '__main__':
    main()
