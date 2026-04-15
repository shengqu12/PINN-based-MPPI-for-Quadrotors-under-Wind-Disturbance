"""
test_generalization.py — ablation experiment

Method:
    Nominal MPPI:K=896, H=15, no PINN (H corespond to n in the paper)
    PINN-MPPI:   K=896, H=15, PINN


Test1: 
    Y direct wind (testify the generalization ability of using sysmetry constraint)
    Training is only based on x direction wind.

Test2: 
    Lissajous figure-8 trajectory (vertical orientation)
    Training used horizontal-orientation Lissajous; this tests same-family generalization.

Test3:
   OOD wind speed + Lissajous figure-8 (trajectory seen in training family)

"""

import numpy as np
import torch
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.pinn import ResidualPINN
from training.dataset import Normalizer
from controllers.pinn_mppi_GPU import MPPIController, pinn_predict

from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.trajectories.hover_traj import HoverTraj

CKPT_DIR    = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
RESULT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'results')
K_ETA       = quad_params['k_eta']
MASS        = quad_params['mass']
G           = 9.81
OMEGA_MIN   = quad_params['rotor_speed_min']
OMEGA_MAX   = quad_params['rotor_speed_max']
HOVER_OMEGA = float(np.sqrt(MASS * G / (4 * K_ETA)))
KP_POS      = np.array([6.5, 6.5, 15.0])

# MPPI parameters: K = num trajectories, H = horizon
MPPI_K = 896
MPPI_H = 15


def load_pinn():
    ckpt       = torch.load(os.path.join(CKPT_DIR, 'best_model.pt'),
                            weights_only=False)
    normalizer = Normalizer()
    normalizer.load(os.path.join(CKPT_DIR, 'normalizer.pt'))
    model = ResidualPINN(input_dim=17)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, normalizer


# ── Lissajous figure-8 traj (vertical orientation) ─────────────────────────────────────────────────────────

class LissajousVerticalTraj:
    """
    Vertical-orientation Lissajous (1:2 frequency ratio):
        x(t) = Ax * sin(ωx * t)
        y(t) = Ay * cos(2ωx * t)
    NOTE: Training used TwoDLissajous (horizontal variant from RotorPy).
    This tests generalization within the same curve family.
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
            'x': np.array([x, y, self.cz]),
            'x_dot': np.array([vx, vy, 0.]),
            'x_ddot': np.array([ax, ay, 0.]),
            'x_dddot': np.zeros(3), 'x_ddddot': np.zeros(3),
            'yaw': 0., 'yaw_dot': 0.,
        }


# ── simulation ──────────────────────────────────────────────────────

def run_episode(model, normalizer, wind_vec, trajectory_fn,
                use_pinn=True, sim_time=15.0, dt=0.01):
    """
    MPPI simulation: Nominal vs PINN 

    Nominal (use_pinn=False): MPPI 
        rollout based on nominal model, no outer info

    PINN-MPPI (use_pinn=True): 
        rollout based on nominal model and PINN residual knowledge
    """
    state = {
        'x':            np.array([0., 0., 1.5]),
        'v':            np.zeros(3),
        'q':            np.array([0., 0., 0., 1.]),
        'w':            np.zeros(3),
        'wind':         wind_vec.copy(),
        'rotor_speeds': np.ones(4) * HOVER_OMEGA,
    }

    vehicle    = Multirotor(quad_params, state)
    controller = SE3Control(quad_params)
    trajectory = trajectory_fn()

    # same parameter expect last one
    mppi = MPPIController(
        model, normalizer, wind_vec,
        K=MPPI_K, H=MPPI_H, dt=dt,
        use_pinn=use_pinn   # sole difference
    )

    N      = int(sim_time / dt)
    errors = np.zeros(N)
    t      = 0.0

    for i in range(N):
        state['wind'] = wind_vec.copy()
        flat_ref      = trajectory.update(t)
        vel_des       = flat_ref['x_dot']

        pos_des_seq = np.array([
            trajectory.update(t + h*dt)['x'] for h in range(mppi.H)
        ])
        vel_des_seq = np.array([
            trajectory.update(t + h*dt)['x_dot'] for h in range(mppi.H)
        ])

        motor_speeds_est = np.array(
            state.get('rotor_speeds', np.ones(4)*HOVER_OMEGA)
        )

        if use_pinn:
            # PINN-MPPI
            residual_raw = pinn_predict(
                model, normalizer, state,
                motor_speeds_est, wind_vec, vel_ref=vel_des
            )
        else:
            # Nominal MPPI
            residual_raw = np.zeros(3)

        alpha, delta_p_calc, _ = mppi.update(
            state, pos_des_seq, vel_des_seq,
            motor_speeds_est, residual_raw
        )
        delta_p = alpha * delta_p_calc   

        flat_modified          = dict(flat_ref)
        flat_modified['x']     = flat_ref['x'] + delta_p
        flat_modified['x_dot'] = flat_ref['x_dot']

        cmd   = controller.update(t, state, flat_modified)
        state = vehicle.step(state, cmd, dt)

        errors[i] = np.linalg.norm(state['x'] - flat_ref['x'])
        t        += dt

    return errors[100:].mean()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    model, normalizer = load_pinn()

    print(f"Test of generalization:(K={MPPI_K}, H={MPPI_H})")
    print(f"One difference: PINN residual =? 0\n")

    speed = 8.0

    # ── test 1: Y direction wind ───────────────────────────────────────────────
    print("=" * 62)
    print("Test1: Generalization to Unseen Wind Directions (hover,wind=8 m/s)")
    print("=" * 62)

    wind_configs = {
        'X(trained)': np.array([ 1.,  0., 0.]) * speed,
        'Y(tested)': np.array([ 0.,  1., 0.]) * speed,
        '45° (XY direction)':      np.array([ 1.,  1., 0.]) / np.sqrt(2) * speed,
        '-X (direction)':       np.array([-1.,  0., 0.]) * speed,
    }

    traj_hover = lambda: HoverTraj(x0=np.array([0., 0., 1.5]))

    print(f"\n{'Wind direction':>14} | {'Nominal MPPI':>12} | "
          f"{'PINN-MPPI':>10} | {'Improvement':>8}")
    print("-" * 54)

    wind_results = {}
    for name, wv in wind_configs.items():
        e_nom  = run_episode(model, normalizer, wv,
                             traj_hover, use_pinn=False)
        e_pinn = run_episode(model, normalizer, wv,
                             traj_hover, use_pinn=True)
        improv = (e_nom - e_pinn) / (e_nom + 1e-9) * 100
        tag    = '✓' if improv > 0 else '✗'
        print(f"{name:>14} | {e_nom:12.4f} | "
              f"{e_pinn:10.4f} | {improv:+7.1f}% {tag}")
        wind_results[name] = {'nom': e_nom, 'pinn': e_pinn}

    # ── Test 2: trajectory generalization ────────────────────────────────────────
    print("\n" + "=" * 62)
    print("Test2:traj generalization(under X 8 m/s)")
    print("=" * 62)

    wind_x = np.array([speed, 0., 0.])

    traj_configs = {
        'Hover(trained)':    (lambda: HoverTraj(x0=np.array([0., 0., 1.5])),
                             'trained'),
        'Circle(trained)':   (lambda: ThreeDCircularTraj(
                                center=np.array([0., 0., 1.5]),
                                radius=np.array([1.5, 1.5, 0.]),
                                freq=np.array([0.2, 0.2, 0.])),
                             'trained'),
        'Lissajous-vertical(Similar)':    (lambda: LissajousVerticalTraj(
                                center=np.array([0., 0., 1.5]),
                                Ax=1.5, Ay=1.5, freq_x=0.15),
                             'similiar'),
        'Smaller circle r=0.5m(Unseen)': (lambda: ThreeDCircularTraj(
                                center=np.array([0., 0., 1.5]),
                                radius=np.array([0.5, 0.5, 0.]),
                                freq=np.array([0.3, 0.3, 0.])),
                             'test'),
        'Bigger circle r=3.0m(Unseen)': (lambda: ThreeDCircularTraj(
                                center=np.array([0., 0., 1.5]),
                                radius=np.array([3.0, 3.0, 0.]),
                                freq=np.array([0.1, 0.1, 0.])),
                             'Unseen'),
    }

    print(f"\n{'Traj':>20} | {'Category':>4} | {'Nominal MPPI':>12} | "
          f"{'PINN-MPPI':>10} | {'Improvement':>8}")
    print("-" * 66)

    traj_results = {}
    for traj_name, (traj_fn, traj_type) in traj_configs.items():
        e_nom  = run_episode(model, normalizer, wind_x,
                             traj_fn, use_pinn=False)
        e_pinn = run_episode(model, normalizer, wind_x,
                             traj_fn, use_pinn=True)
        improv = (e_nom - e_pinn) / (e_nom + 1e-9) * 100
        tag    = '✓' if improv > 0 else '✗'
        print(f"{traj_name:>20} | {traj_type:>4} | {e_nom:12.4f} | "
              f"{e_pinn:10.4f} | {improv:+7.1f}% {tag}")
        traj_results[traj_name] = {
            'nom': e_nom, 'pinn': e_pinn, 'type': traj_type
        }

    # ── Test 3: Double Generalization Test ───────────────────────────────────────
    print("\n" + "=" * 62)
    print("Test3: Wind-Speed OOD on Lissajous figure-8 (trajectory seen in training family)")
    print("=" * 62)

    traj_eight = lambda: LissajousVerticalTraj(
        center=np.array([0., 0., 1.5]), Ax=1.5, Ay=1.5, freq_x=0.15
    )

    print(f"\n{'Wind':>6} | {'Category':>5} | {'Nominal MPPI':>12} | "
          f"{'PINN-MPPI':>10} | {'Improvement':>8}")
    print("-" * 54)

    ood_results = {}
    for ws, label in [(8., 'train'), (10., 'OOD'), (12., 'OOD')]:
        wv     = np.array([ws, 0., 0.])
        e_nom  = run_episode(model, normalizer, wv,
                             traj_eight, use_pinn=False)
        e_pinn = run_episode(model, normalizer, wv,
                             traj_eight, use_pinn=True)
        improv = (e_nom - e_pinn) / (e_nom + 1e-9) * 100
        tag    = '✓' if improv > 0 else '✗'
        print(f"{ws:6.0f} | {label:>5} | {e_nom:12.4f} | "
              f"{e_pinn:10.4f} | {improv:+7.1f}% {tag}")
        ood_results[ws] = {'nom': e_nom, 'pinn': e_pinn}

    all_results = {
        'wind_dir':   wind_results,
        'trajectory': traj_results,
        'double_ood': ood_results,
        'mppi_params': {'K': MPPI_K, 'H': MPPI_H},
    }
    np.save(os.path.join(RESULT_DIR, 'generalization.npy'),
            all_results, allow_pickle=True)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f'Generalizaton Test(K={MPPI_K}, H={MPPI_H})\n'
            'Nominal MPPI vs PINN-MPPI',
            fontsize=11, fontweight='bold'
        )

        C_NOM  = '#5B9BD5'
        C_PINN = '#FFC000'

        def add_improv_labels(ax, x_vals, noms, pinns):
            ymax = ax.get_ylim()[1]
            for i, (n, p) in enumerate(zip(noms, pinns)):
                imp = (n-p)/(n+1e-9)*100
                color = '#2E7D32' if imp > 0 else '#C62828'
                ax.text(x_vals[i], max(n,p) + ymax*0.03,
                        f'{imp:+.0f}%', ha='center',
                        fontsize=8, color=color, fontweight='bold')

        # Graph 1 : wind direction generalization
        ax    = axes[0]
        names = list(wind_results.keys())
        noms  = [wind_results[n]['nom']  for n in names]
        pinns = [wind_results[n]['pinn'] for n in names]
        x     = np.arange(len(names))
        ax.bar(x-0.2, noms,  0.4, label=f'Nominal MPPI (K={MPPI_K})',
               color=C_NOM,  alpha=0.85, edgecolor='white')
        ax.bar(x+0.2, pinns, 0.4, label='PINN-MPPI',
               color=C_PINN, alpha=0.85, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(['X\n(trained)', 'Y\n(unseen)', '45°\n(unseen)', '-X\n(unseen)'],
                           fontsize=9)
        ax.set_ylabel('Mean Error (m)'); ax.set_xlabel('Wind Direction')
        ax.set_title('Wind Generalization\n(hover, 8m/s)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
        add_improv_labels(ax, x, noms, pinns)

        # Graph2 : traj generalization
        ax    = axes[1]
        names = list(traj_results.keys())
        noms  = [traj_results[n]['nom']  for n in names]
        pinns = [traj_results[n]['pinn'] for n in names]
        types = [traj_results[n]['type'] for n in names]
        x     = np.arange(len(names))
        ax.bar(x-0.2, noms,  0.4, label=f'Nominal MPPI (K={MPPI_K})',
               color=C_NOM,  alpha=0.85, edgecolor='white')
        ax.bar(x+0.2, pinns, 0.4, label='PINN-MPPI',
               color=C_PINN, alpha=0.85, edgecolor='white')

        for i, tp in enumerate(types):
            if tp == 'Unseen':
                ax.axvspan(i-0.5, i+0.5, alpha=0.06, color='orange',
                           label='_unseen_traj' if i==2 else '')
        ax.set_xticks(x)
        ax.set_xticklabels(['Hover\n(Trained)', 'Circle\n(Trained)',
                            'Lissajous\n(Unseen)', 'Smaller Circle\n(Unseen)', 'Bigger Circle\n(Unseen)'],
                           fontsize=8)
        ax.set_ylabel('Mean Error (m)'); ax.set_xlabel('Trajectory Category')
        ax.set_title('Trajectory Generalization\n(X-wind 8m/s)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
        add_improv_labels(ax, x, noms, pinns)

        # Graph3 : double Generalization
        ax     = axes[2]
        ws_list = list(ood_results.keys())
        noms   = [ood_results[ws]['nom']  for ws in ws_list]
        pinns  = [ood_results[ws]['pinn'] for ws in ws_list]
        x      = np.arange(len(ws_list))
        ax.bar(x-0.2, noms,  0.4, label=f'Nominal MPPI (K={MPPI_K})',
               color=C_NOM,  alpha=0.85, edgecolor='white')
        ax.bar(x+0.2, pinns, 0.4, label='PINN-MPPI',
               color=C_PINN, alpha=0.85, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{ws:.0f}m/s\n({"train" if ws==8 else "OOD"})'
                            for ws in ws_list], fontsize=9)
        ax.set_ylabel('Mean Error (m)'); ax.set_xlabel('Wind Speed')
        ax.set_title('Wind-Speed OOD\n(Lissajous figure-8 trajectory)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
        add_improv_labels(ax, x, noms, pinns)

        plt.tight_layout()
        path = os.path.join(RESULT_DIR, 'generalization.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\n figure has been saved: {path}")
    except Exception as e:
        print(f"Jump Ploting: {e}")


if __name__ == '__main__':
    main()