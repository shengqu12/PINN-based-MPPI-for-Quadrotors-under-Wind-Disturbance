"""
experiments/ablation_controllers.py — Controller comparison

Controllers compared:
    1. SE3 Only       — SE3Control tracking, no MPPI, no PINN (baseline)
    2. Nominal MPPI   — Traditional MPPI: directly optimises Δa ∈ [−5,5]³ per axis.
                        No PINN, no wind knowledge. Rollout uses nominal dynamics (no wind term).
                        Shows MPPI's multi-step lookahead benefit without wind knowledge.
    3. PINN-MPPI      — Proposed: α-MPPI optimises PINN feedforward gain α∈[0,1.5]³.
                        MPPI rollout uses SE3 closed-loop + PINN wind residual.
                        Execution: SE3Control with flat['x_ddot'] += -α_opt ⊙ r̂_ema.

Logic chain:  SE3-Only  <  Nominal MPPI  <  PINN-MPPI
    SE3 → Nominal MPPI:   MPPI's multi-step lookahead adds value
    Nominal MPPI → PINN:  PINN wind knowledge is the key additional gain

Trajectories: hover, circle, lemniscate, spiral
Wind:         0, 2, 4, 6, 8 m/s (train),  10, 12 m/s (OOD)
Metric:       mean position error (m), control Hz
"""

import os, sys, time, argparse
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from controllers.pinn_mppi_v2 import (
    PINNMPPIv2, body_rate_pd_to_motors,
    load_pinn_model, run_episode as pinn_run_episode,
    run_nominal_episode,
    HOVER_OMEGA, T_MAX, OMEGA_XY_MAX, OMEGA_Z_MAX,
)

from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj

CKPT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

_HOVER_OMEGA = float(np.sqrt(quad_params['mass'] * 9.81 / (4 * quad_params['k_eta'])))


# ── trajectory factories ───────────────────────────────────────────────────────

def make_lemniscate(scale=1.5, period=10.0):
    class _T:
        def update(self, t):
            w   = 2*np.pi/period;  s = w*t;  d = 1+np.sin(s)**2
            x   = scale*np.cos(s)/d
            y   = scale*np.sin(s)*np.cos(s)/d
            eps = 1e-4;  d2 = 1+np.sin(s+eps*w)**2
            x2  = scale*np.cos(s+eps*w)/d2
            y2  = scale*np.sin(s+eps*w)*np.cos(s+eps*w)/d2
            dx  = (x2-x)/eps;  dy = (y2-y)/eps
            return {'x': np.array([x,y,1.5]), 'x_dot': np.array([dx,dy,0.]),
                    'x_ddot': np.zeros(3), 'yaw': 0., 'yaw_dot': 0.}
    return _T()

def make_spiral(radius=1.5, height=1.0, period=10.0):
    class _T:
        def update(self, t):
            w  = 2*np.pi/period;  f = min(t/period, 1.0)
            x  = radius*np.cos(w*t);  y = radius*np.sin(w*t)
            z  = 1.5 + height*f
            dx = -radius*w*np.sin(w*t);  dy = radius*w*np.cos(w*t)
            return {'x': np.array([x,y,z]), 'x_dot': np.array([dx,dy,height/period]),
                    'x_ddot': np.zeros(3), 'yaw': 0., 'yaw_dot': 0.}
    return _T()

TRAJECTORIES = {
    'hover':      lambda: HoverTraj(x0=np.array([0.,0.,1.5])),
    'circle':     lambda: ThreeDCircularTraj(
                      center=np.array([0.,0.,1.5]),
                      radius=np.array([1.5,1.5,0.]),
                      freq=np.array([0.2,0.2,0.])),
    'lemniscate': make_lemniscate,
    'spiral':     make_spiral,
}

WIND_CONFIGS = [
    (0.,  'train'), (2.,  'train'), (4.,  'train'),
    (6.,  'train'), (8.,  'train'),
    (10., 'OOD'),   (12., 'OOD'),
]


# ── controller runners ─────────────────────────────────────────────────────────

def run_se3_only(wind_vec, traj_fn, sim_time=15., dt=0.01):
    """SE3Control baseline — no MPPI, no PINN."""
    state = {'x': np.array([0.,0.,1.5]), 'v': np.zeros(3),
             'q': np.array([0.,0.,0.,1.]), 'w': np.zeros(3),
             'wind': wind_vec.copy(), 'rotor_speeds': np.ones(4)*_HOVER_OMEGA}
    vehicle = Multirotor(quad_params, state)
    ctrl    = SE3Control(quad_params)
    traj    = traj_fn()
    N = int(sim_time/dt);  errs = [];  ms_list = []
    for i in range(N):
        t = i*dt;  state['wind'] = wind_vec.copy()
        flat  = traj.update(t)
        t0    = time.perf_counter()
        cmd   = ctrl.update(t, state, flat)
        ms_list.append((time.perf_counter()-t0)*1000)
        state = vehicle.step(state, cmd, dt)
        errs.append(np.linalg.norm(state['x'] - flat['x']))
    return {'mean_err': float(np.mean(errs[100:])),
            'errors':   np.array(errs),
            'mean_hz':  1000./np.mean(ms_list[10:]) if len(ms_list)>10 else 0}


def run_nominal_mppi(wind_vec, traj_fn,
                     sim_time=15., dt=0.01, K=896, H=15, n_interp=10):
    """Traditional MPPI — directly optimises Δa ∈ [−5, 5]³ per axis.

    No PINN, no wind knowledge in rollout model.
    MPPI uses multi-step lookahead to find the best acceleration feedforward,
    but cannot predict wind direction → reactive rather than proactive.
    Demonstrates MPPI's lookahead benefit independently of PINN.

    Expected:  SE3-Only  <  Nominal MPPI  <  PINN-MPPI
    """
    r = run_nominal_episode(wind_vec, traj_fn,
                            sim_time=sim_time, dt=dt, K=K, H=H,
                            n_interp=n_interp, verbose=False)
    r['mean_hz'] = r['freq_stats'].get('mean_hz', 0)
    return r


def run_pinn_mppi(model, normalizer, wind_vec, traj_fn,
                  sim_time=15., dt=0.01, K=896, H=15, n_interp=10,
                  use_pinn=True):
    """PINN-MPPI v2 — PINN residual inside rollout dv."""
    r = pinn_run_episode(model, normalizer, wind_vec, traj_fn,
                         sim_time=sim_time, dt=dt, K=K, H=H, n_interp=n_interp,
                         use_pinn=use_pinn, verbose=False)
    r['mean_hz'] = r['freq_stats'].get('mean_hz', 0)
    return r


# ── main comparison ────────────────────────────────────────────────────────────

def run_all(model, normalizer, trajs, wind_cfgs,
            sim_time=15., K=896):
    all_res = {}
    for tname, tfn in trajs.items():
        print(f"\n{'='*90}")
        print(f"Trajectory: {tname}")
        print(f"{'='*90}")
        print(f"\n{'Wind':>5} | {'Split':>5} | {'SE3':>8} | "
              f"{'NomMPPI':>8} | {'PINN-MPPI':>10} | "
              f"{'vs SE3':>8} | {'vs Nom':>8} | {'Hz':>6}")
        print("-"*74)

        tres = {}
        for ws, split in wind_cfgs:
            wv     = np.array([ws, 0., 0.])
            r_se3  = run_se3_only(wv, tfn, sim_time)
            r_nom  = run_nominal_mppi(wv, tfn, sim_time, K=K)
            r_pinn = run_pinn_mppi(model, normalizer, wv, tfn, sim_time,
                                   K=K, use_pinn=True)

            vs_se3 = (r_se3['mean_err'] - r_pinn['mean_err']) \
                     / (r_se3['mean_err'] + 1e-9) * 100
            vs_nom = (r_nom['mean_err'] - r_pinn['mean_err']) \
                     / (r_nom['mean_err'] + 1e-9) * 100

            print(f"{ws:5.0f} | {split:>5} | "
                  f"{r_se3['mean_err']:8.4f} | "
                  f"{r_nom['mean_err']:8.4f} | "
                  f"{r_pinn['mean_err']:10.4f} | "
                  f"{vs_se3:+7.1f}% | "
                  f"{vs_nom:+7.1f}% | "
                  f"{r_pinn['mean_hz']:6.1f}")

            tres[ws] = {'se3': r_se3, 'nom_mppi': r_nom,
                        'pinn_mppi': r_pinn, 'split': split}
        all_res[tname] = tres
    return all_res


def save_table(all_res, path):
    """Write a plain-text summary table."""
    lines = []
    for tname, tres in all_res.items():
        lines.append(f"\nTrajectory: {tname}")
        lines.append(f"{'Wind':>5} | {'Split':>5} | {'SE3':>8} | "
                     f"{'NomMPPI':>8} | {'PINN-MPPI':>10} | "
                     f"{'vs SE3':>8} | {'vs Nom':>8} | {'Hz':>6}")
        lines.append("-"*74)
        for ws in sorted(tres.keys()):
            d      = tres[ws]
            r_se3  = d['se3']['mean_err']
            r_nom  = d['nom_mppi']['mean_err']
            r_pinn = d['pinn_mppi']['mean_err']
            hz     = d['pinn_mppi']['mean_hz']
            vs_se3 = (r_se3 - r_pinn) / (r_se3 + 1e-9) * 100
            vs_nom = (r_nom - r_pinn) / (r_nom + 1e-9) * 100
            lines.append(f"{ws:5.0f} | {d['split']:>5} | "
                         f"{r_se3:8.4f} | {r_nom:8.4f} | "
                         f"{r_pinn:10.4f} | "
                         f"{vs_se3:+7.1f}% | {vs_nom:+7.1f}% | {hz:6.1f}")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Table saved: {path}")


def plot_results(all_res, path):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        tnames = list(all_res.keys());  nt = len(tnames)
        fig, axes = plt.subplots(nt, 2, figsize=(14, 4*nt))
        if nt == 1:
            axes = [axes]
        fig.suptitle('Controller Comparison: SE3 vs Nominal-MPPI vs PINN-MPPI v2',
                     fontsize=11)

        C = {'SE3': 'steelblue', 'Nom-MPPI': 'mediumpurple',
             'PINN-MPPI': 'darkorange'}

        for row, tname in enumerate(tnames):
            res     = all_res[tname]
            ws_list = sorted(res.keys())
            ax      = axes[row][0]
            x       = np.arange(len(ws_list));  w = 0.25

            for ci, (lbl, key) in enumerate([
                    ('SE3', 'se3'), ('Nom-MPPI', 'nom_mppi'),
                    ('PINN-MPPI', 'pinn_mppi')]):
                vals = [res[ws][key]['mean_err'] for ws in ws_list]
                ax.bar(x + (ci-1)*w, vals, w, label=lbl,
                       color=C[lbl], alpha=0.85)

            # vertical line between train and OOD
            ood_start = next((i for i,ws in enumerate(ws_list)
                              if res[ws]['split']=='OOD'), None)
            if ood_start is not None:
                ax.axvline(x=ood_start-0.5, color='gray', ls='--', alpha=0.5,
                           label='train|OOD')

            ax.set_xticks(x)
            ax.set_xticklabels([f'{ws}m/s' for ws in ws_list])
            ax.set_xlabel('Wind speed'); ax.set_ylabel('Mean error (m)')
            ax.set_title(f'{tname}'); ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

            # time-series at wind=8
            ax2 = axes[row][1]
            ws  = 8.
            if ws in res:
                ts = np.arange(len(res[ws]['se3']['errors'])) * 0.01
                for lbl, key in [('SE3', 'se3'), ('Nom-MPPI', 'nom_mppi'),
                                  ('PINN-MPPI', 'pinn_mppi')]:
                    e = res[ws][key]['errors']
                    ax2.plot(ts[:len(e)], e, label=lbl, color=C[lbl], alpha=0.85)
                ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Error (m)')
                ax2.set_title(f'{tname}  wind=8 m/s')
                ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {path}")
    except Exception as e:
        print(f"Plot skipped: {e}")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='hover + circle only, fewer wind speeds')
    parser.add_argument('--K', type=int, default=896)
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)

    model, normalizer = load_pinn_model(CKPT_DIR)

    trajs = ({'hover': TRAJECTORIES['hover'],
              'circle': TRAJECTORIES['circle']}
             if args.quick else TRAJECTORIES)

    wind_cfgs = ([(0.,'train'),(4.,'train'),(8.,'train'),
                  (10.,'OOD'),(12.,'OOD')]
                 if args.quick else WIND_CONFIGS)

    all_res = run_all(model, normalizer, trajs, wind_cfgs,
                      sim_time=15., K=args.K)

    np.save(os.path.join(RESULT_DIR, 'ablation_controllers.npy'),
            all_res, allow_pickle=True)
    save_table(all_res,
               os.path.join(RESULT_DIR, 'ablation_controllers.txt'))
    plot_results(all_res,
                 os.path.join(RESULT_DIR, 'ablation_controllers.png'))
    print("\nDone.")


if __name__ == '__main__':
    main()
