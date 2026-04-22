"""
test_generalization.py — generalization experiments

Method:
    Nominal MPPI: K=896, H=15, no PINN
    PINN-MPPI:    K=896, H=15, PINN-augmented SE3 feedforward

Test 1: Wind direction generalization (hover, 8 m/s).
    Training used all 4 cardinal directions (0°/90°/180°/270°).
    The 45° diagonal is the only truly unseen direction.

Test 2: Trajectory generalization (X-wind 8 m/s).
    Hover/Circle/Lissajous were used in training.
    Lemniscate and Spiral are unseen trajectories.

Test 3: OOD wind speed on Lemniscate (unseen trajectory).
"""

import numpy as np
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from controllers.pinn_mppi_v2 import (
    run_episode as pinn_run_episode,
    load_pinn_model,
)
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.lissajous_traj import TwoDLissajous

CKPT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

MPPI_K = 896
MPPI_H = 15


# ── Unseen trajectory definitions ─────────────────────────────────────────────

def make_lemniscate(scale=1.5, period=10.0):
    class _T:
        def update(self, t):
            w   = 2 * np.pi / period;  s = w * t;  d = 1 + np.sin(s) ** 2
            x   = scale * np.cos(s) / d
            y   = scale * np.sin(s) * np.cos(s) / d
            eps = 1e-4;  d2 = 1 + np.sin(s + eps * w) ** 2
            x2  = scale * np.cos(s + eps * w) / d2
            y2  = scale * np.sin(s + eps * w) * np.cos(s + eps * w) / d2
            dx  = (x2 - x) / eps;  dy = (y2 - y) / eps
            return {
                'x': np.array([x, y, 1.5]),
                'x_dot': np.array([dx, dy, 0.]),
                'x_ddot': np.zeros(3),
                'x_dddot': np.zeros(3), 'x_ddddot': np.zeros(3),
                'yaw': 0., 'yaw_dot': 0.,
            }
    return _T()


def make_spiral(radius=1.5, height=1.0, period=10.0):
    class _T:
        def update(self, t):
            w  = 2 * np.pi / period;  f = min(t / period, 1.0)
            x  = radius * np.cos(w * t);  y = radius * np.sin(w * t)
            z  = 1.5 + height * f
            dx = -radius * w * np.sin(w * t);  dy = radius * w * np.cos(w * t)
            return {
                'x': np.array([x, y, z]),
                'x_dot': np.array([dx, dy, height / period]),
                'x_ddot': np.zeros(3),
                'x_dddot': np.zeros(3), 'x_ddddot': np.zeros(3),
                'yaw': 0., 'yaw_dot': 0.,
            }
    return _T()


# ── thin wrapper ───────────────────────────────────────────────────────────────

def run_ep(model, normalizer, wind_vec, traj_fn, use_pinn=True, sim_time=15.0):
    """Call pinn_mppi_v2.run_episode and return scalar mean error."""
    r = pinn_run_episode(
        model, normalizer, wind_vec, traj_fn,
        sim_time=sim_time, dt=0.01, K=MPPI_K, H=MPPI_H,
        use_pinn=use_pinn, verbose=False,
    )
    return float(r['mean_err'])


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    model, normalizer = load_pinn_model(CKPT_DIR)

    print(f"Generalization Test  (K={MPPI_K}, H={MPPI_H})")
    print("Only difference between Nominal and PINN-MPPI: PINN feedforward\n")

    speed = 8.0

    # ── Test 1: Wind direction generalization (hover) ──────────────────────────
    print("=" * 62)
    print("Test 1: Wind Direction Generalization  (hover, 8 m/s)")
    print("=" * 62)

    wind_configs = {
        '+X (trained)':  np.array([ 1.,  0., 0.]) * speed,
        '+Y (trained)':  np.array([ 0.,  1., 0.]) * speed,
        '45° (unseen)':  np.array([ 1.,  1., 0.]) / np.sqrt(2) * speed,
        '-X (trained)':  np.array([-1.,  0., 0.]) * speed,
    }
    traj_hover = lambda: HoverTraj(x0=np.array([0., 0., 1.5]))

    print(f"\n{'Wind direction':>14} | {'Nominal':>8} | {'PINN-MPPI':>10} | {'Improve':>8}")
    print("-" * 52)

    wind_results = {}
    for name, wv in wind_configs.items():
        e_nom  = run_ep(model, normalizer, wv, traj_hover, use_pinn=False)
        e_pinn = run_ep(model, normalizer, wv, traj_hover, use_pinn=True)
        improv = (e_nom - e_pinn) / (e_nom + 1e-9) * 100
        tag    = '✓' if improv > 0 else '✗'
        print(f"{name:>14} | {e_nom:8.4f} | {e_pinn:10.4f} | {improv:+7.1f}% {tag}")
        wind_results[name] = {'nom': e_nom, 'pinn': e_pinn}

    # ── Test 2: Trajectory generalization (X-wind 8 m/s) ──────────────────────
    print("\n" + "=" * 62)
    print("Test 2: Trajectory Generalization  (X-wind 8 m/s)")
    print("=" * 62)

    wind_x = np.array([speed, 0., 0.])
    traj_configs = {
        'Lissajous (train)':    (lambda: TwoDLissajous(
                                     A=1.5, B=1.5, a=1, b=2,
                                     delta=1.5708, height=1.5),
                                 'train'),
        'Circle r=1.5 (train)': (lambda: ThreeDCircularTraj(
                                     center=np.array([0., 0., 1.5]),
                                     radius=np.array([1.5, 1.5, 0.]),
                                     freq=np.array([0.2, 0.2, 0.])),
                                 'train'),
        'Lemniscate (unseen)':  (lambda: make_lemniscate(scale=1.5, period=10.0),
                                 'unseen'),
        'Spiral (unseen)':      (lambda: make_spiral(radius=1.5, height=1.0, period=10.0),
                                 'unseen'),
    }

    print(f"\n{'Trajectory':>24} | {'Cat':>6} | {'Nominal':>8} | "
          f"{'PINN-MPPI':>10} | {'Improve':>8}")
    print("-" * 68)

    traj_results = {}
    for tname, (tfn, ttype) in traj_configs.items():
        e_nom  = run_ep(model, normalizer, wind_x, tfn, use_pinn=False)
        e_pinn = run_ep(model, normalizer, wind_x, tfn, use_pinn=True)
        improv = (e_nom - e_pinn) / (e_nom + 1e-9) * 100
        tag    = '✓' if improv > 0 else '✗'
        print(f"{tname:>24} | {ttype:>6} | {e_nom:8.4f} | "
              f"{e_pinn:10.4f} | {improv:+7.1f}% {tag}")
        traj_results[tname] = {'nom': e_nom, 'pinn': e_pinn, 'type': ttype}

    # ── Test 3: OOD wind speed on Lemniscate ──────────────────────────────────
    print("\n" + "=" * 62)
    print("Test 3: OOD Wind Speed on Lemniscate (unseen trajectory)")
    print("=" * 62)

    traj_lemn = lambda: make_lemniscate(scale=1.5, period=10.0)

    print(f"\n{'Wind':>6} | {'Split':>5} | {'Nominal':>8} | "
          f"{'PINN-MPPI':>10} | {'Improve':>8}")
    print("-" * 52)

    ood_results = {}
    for ws, label in [(8., 'train'), (10., 'OOD'), (12., 'OOD')]:
        wv     = np.array([ws, 0., 0.])
        e_nom  = run_ep(model, normalizer, wv, traj_lemn, use_pinn=False)
        e_pinn = run_ep(model, normalizer, wv, traj_lemn, use_pinn=True)
        improv = (e_nom - e_pinn) / (e_nom + 1e-9) * 100
        tag    = '✓' if improv > 0 else '✗'
        print(f"{ws:6.0f} | {label:>5} | {e_nom:8.4f} | "
              f"{e_pinn:10.4f} | {improv:+7.1f}% {tag}")
        ood_results[ws] = {'nom': e_nom, 'pinn': e_pinn}

    # ── save ──────────────────────────────────────────────────────────────────
    all_results = {
        'wind_dir':    wind_results,
        'trajectory':  traj_results,
        'ood_wind':    ood_results,
        'mppi_params': {'K': MPPI_K, 'H': MPPI_H},
    }
    np.save(os.path.join(RESULT_DIR, 'generalization.npy'),
            all_results, allow_pickle=True)

    # ── plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        C_NOM  = '#5B9BD5'
        C_PINN = '#FFC000'

        def add_improv_labels(ax, xs, noms, pinns):
            ymax = ax.get_ylim()[1]
            for i, (n, p) in enumerate(zip(noms, pinns)):
                imp = (n - p) / (n + 1e-9) * 100
                color = '#2E7D32' if imp > 0 else '#C62828'
                ax.text(xs[i], max(n, p) + ymax * 0.03,
                        f'{imp:+.0f}%', ha='center', fontsize=8,
                        color=color, fontweight='bold')

        # Graph 1 — wind direction
        ax    = axes[0]
        names = list(wind_results.keys())
        noms  = [wind_results[n]['nom']  for n in names]
        pinns = [wind_results[n]['pinn'] for n in names]
        x     = np.arange(len(names))
        ax.bar(x - 0.2, noms,  0.4, label='Nominal MPPI', color=C_NOM,  alpha=0.85)
        ax.bar(x + 0.2, pinns, 0.4, label='PINN-MPPI',    color=C_PINN, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(['+X\n(train)', '+Y\n(train)', '45°\n(unseen)', '-X\n(train)'],
                           fontsize=9)
        ax.set_ylabel('Mean Error (m)'); ax.set_xlabel('Wind Direction')
        ax.set_title('Wind Direction\n(hover, 8 m/s)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
        add_improv_labels(ax, x, noms, pinns)

        # Graph 2 — trajectory
        ax    = axes[1]
        names = list(traj_results.keys())
        noms  = [traj_results[n]['nom']  for n in names]
        pinns = [traj_results[n]['pinn'] for n in names]
        x     = np.arange(len(names))
        ax.bar(x - 0.2, noms,  0.4, label='Nominal MPPI', color=C_NOM,  alpha=0.85)
        ax.bar(x + 0.2, pinns, 0.4, label='PINN-MPPI',    color=C_PINN, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(['Lissajous\n(train)', 'Circle\n(train)',
                            'Lemnis-\ncate', 'Spiral'],
                           fontsize=8)
        ax.set_ylabel('Mean Error (m)'); ax.set_xlabel('Trajectory')
        ax.set_title('Trajectory Generalization\n(X-wind 8 m/s)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
        add_improv_labels(ax, x, noms, pinns)

        # Graph 3 — OOD wind speed
        ax      = axes[2]
        ws_list = list(ood_results.keys())
        noms    = [ood_results[ws]['nom']  for ws in ws_list]
        pinns   = [ood_results[ws]['pinn'] for ws in ws_list]
        x       = np.arange(len(ws_list))
        ax.bar(x - 0.2, noms,  0.4, label='Nominal MPPI', color=C_NOM,  alpha=0.85)
        ax.bar(x + 0.2, pinns, 0.4, label='PINN-MPPI',    color=C_PINN, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{ws:.0f} m/s\n({"train" if ws == 8 else "OOD"})'
                            for ws in ws_list], fontsize=9)
        ax.set_ylabel('Mean Error (m)'); ax.set_xlabel('Wind Speed')
        ax.set_title('OOD Wind Speed\n(Lemniscate, unseen)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
        add_improv_labels(ax, x, noms, pinns)

        plt.tight_layout()
        path = os.path.join(RESULT_DIR, 'generalization.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved: {path}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    print("\nDone.")


if __name__ == '__main__':
    main()
