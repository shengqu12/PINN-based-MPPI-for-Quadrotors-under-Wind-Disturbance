"""
test_generalization.py — generalization experiments

Method:
    Nominal MPPI: K=896, H=15, no PINN
    PINN-MPPI:    K=896, H=15, PINN-augmented SE3 feedforward

Test 1: Wind direction generalization (hover, 8 m/s).
    Training used all 4 cardinal directions (0°/90°/180°/270°).
    The 45° diagonal is the only truly unseen direction.

Test 2: Trajectory generalization (X-wind 8 m/s).
    Hover/Circle were used in training. Lissajous-vertical and larger/smaller
    circles test same-family and out-of-family generalization.

Test 3: OOD wind speed on Lissajous figure-8.
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

CKPT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

MPPI_K = 896
MPPI_H = 15


# ── Lissajous figure-8 trajectory (vertical orientation) ──────────────────────

class LissajousVerticalTraj:
    """
    Vertical-orientation Lissajous (1:2 frequency ratio):
        x(t) = Ax * sin(wx * t)
        y(t) = Ay * sin(2*wx * t + phi)
    """
    def __init__(self, center=np.array([0., 0., 1.5]),
                 Ax=1.5, Ay=1.5, freq_x=0.15):
        self.cx, self.cy, self.cz = center
        self.Ax = Ax;  self.Ay = Ay
        self.wx = 2 * np.pi * freq_x
        self.wy = 2 * self.wx
        self.phi = np.pi / 2

    def update(self, t):
        x  = self.cx + self.Ax * np.sin(self.wx * t)
        y  = self.cy + self.Ay * np.sin(self.wy * t + self.phi)
        vx = self.Ax * self.wx * np.cos(self.wx * t)
        vy = self.Ay * self.wy * np.cos(self.wy * t + self.phi)
        ax = -self.Ax * self.wx**2 * np.sin(self.wx * t)
        ay = -self.Ay * self.wy**2 * np.sin(self.wy * t + self.phi)
        return {
            'x':       np.array([x, y, self.cz]),
            'x_dot':   np.array([vx, vy, 0.]),
            'x_ddot':  np.array([ax, ay, 0.]),
            'x_dddot': np.zeros(3), 'x_ddddot': np.zeros(3),
            'yaw': 0., 'yaw_dot': 0.,
        }


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
        'Hover (train)':          (lambda: HoverTraj(x0=np.array([0., 0., 1.5])),
                                   'train'),
        'Circle r=1.5 (train)':   (lambda: ThreeDCircularTraj(
                                       center=np.array([0., 0., 1.5]),
                                       radius=np.array([1.5, 1.5, 0.]),
                                       freq=np.array([0.2, 0.2, 0.])),
                                   'train'),
        'Lissajous-V (unseen)':   (lambda: LissajousVerticalTraj(
                                       center=np.array([0., 0., 1.5]),
                                       Ax=1.5, Ay=1.5, freq_x=0.15),
                                   'unseen'),
        'Circle r=0.5 (unseen)':  (lambda: ThreeDCircularTraj(
                                       center=np.array([0., 0., 1.5]),
                                       radius=np.array([0.5, 0.5, 0.]),
                                       freq=np.array([0.3, 0.3, 0.])),
                                   'unseen'),
        'Circle r=3.0 (unseen)':  (lambda: ThreeDCircularTraj(
                                       center=np.array([0., 0., 1.5]),
                                       radius=np.array([3.0, 3.0, 0.]),
                                       freq=np.array([0.1, 0.1, 0.])),
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

    # ── Test 3: OOD wind speed on Lissajous ───────────────────────────────────
    print("\n" + "=" * 62)
    print("Test 3: OOD Wind Speed on Lissajous Figure-8")
    print("=" * 62)

    traj_eight = lambda: LissajousVerticalTraj(
        center=np.array([0., 0., 1.5]), Ax=1.5, Ay=1.5, freq_x=0.15)

    print(f"\n{'Wind':>6} | {'Split':>5} | {'Nominal':>8} | "
          f"{'PINN-MPPI':>10} | {'Improve':>8}")
    print("-" * 52)

    ood_results = {}
    for ws, label in [(8., 'train'), (10., 'OOD'), (12., 'OOD')]:
        wv     = np.array([ws, 0., 0.])
        e_nom  = run_ep(model, normalizer, wv, traj_eight, use_pinn=False)
        e_pinn = run_ep(model, normalizer, wv, traj_eight, use_pinn=True)
        improv = (e_nom - e_pinn) / (e_nom + 1e-9) * 100
        tag    = '✓' if improv > 0 else '✗'
        print(f"{ws:6.0f} | {label:>5} | {e_nom:8.4f} | "
              f"{e_pinn:10.4f} | {improv:+7.1f}% {tag}")
        ood_results[ws] = {'nom': e_nom, 'pinn': e_pinn}

    # ── save ──────────────────────────────────────────────────────────────────
    import numpy as np_
    all_results = {
        'wind_dir':   wind_results,
        'trajectory': traj_results,
        'ood_wind':   ood_results,
        'mppi_params': {'K': MPPI_K, 'H': MPPI_H},
    }
    np_.save(os.path.join(RESULT_DIR, 'generalization.npy'),
             all_results, allow_pickle=True)

    # ── plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f'Generalization Test  (K={MPPI_K}, H={MPPI_H})\n'
            'Nominal MPPI (SE3, no PINN)  vs  PINN-MPPI (SE3 + PINN feedforward)',
            fontsize=10, fontweight='bold',
        )
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
        ax.set_xticklabels(['Hover', 'Circle\nr=1.5', 'Lissajous', 'Circle\nr=0.5', 'Circle\nr=3.0'],
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
        ax.set_title('OOD Wind Speed\n(Lissajous figure-8)')
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
