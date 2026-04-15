"""
collect_rotorpy.py — data collection script v2.0

Changes from v1:
  - Training wind is now FULLY C4v-SYMMETRIC: 4 cardinal directions
    (+x, -x, +y, -y) at every training speed. This is required for the
    C4v symmetry loss in train.py to be meaningful.
  - Diagonal winds (45°/135°/225°/315°) are intentionally LEFT OUT of
    training so that test_generalization.py Test1 can still demonstrate
    wind-direction generalisation to unseen diagonals.
  - Filenames use angle notation  wind_spd{S}_ang{A}_{traj}.npz
    to avoid sign-in-filename issues on Windows.
  - Already-existing files are skipped (resume-friendly).

Output format (each .npz):
    states:    (N,16) — [pos(3), vel(3), quat(4), omega(3), wind(3)]
    controls:  (N,4)  — cmd_motor_speeds rad/s
    a_actual:  (N,3)  — numerically differentiated actual acceleration
    a_nominal: (N,3)  — analytically predicted acceleration
    residuals: (N,3)  — PINN labels = a_actual - a_nominal

Data split (determined by wind speed, NOT direction):
    train / val : wind speed 0-8 m/s
    OOD         : wind speed > 8 m/s
"""

import numpy as np
import os, sys, math

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.quadrotor_nominal import QuadrotorNominal

from rotorpy.world import World
from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.wind.default_winds import ConstantWind, NoWind
from rotorpy.sensors.imu import Imu
from rotorpy.sensors.external_mocap import MotionCapture
from rotorpy.estimators.nullestimator import NullEstimator
from rotorpy.simulate import simulate

# ── Hummingbird params info ──────────────────────────────────────────────────
print("=== Hummingbird params ===")
for k in ['mass', 'Ixx', 'Iyy', 'Izz', 'k_eta', 'c_Dx', 'c_Dy', 'c_Dz']:
    print(f"  {k}: {quad_params[k]}")

nominal_model = QuadrotorNominal(quad_params)


# ── World / sensor factories ──────────────────────────────────────────────────

def make_world():
    return World.empty(([-8, 8], [-8, 8], [0, 12]))

def make_sensors():
    return Imu(), MotionCapture(sampling_rate=100)


# ── Trajectory factory ────────────────────────────────────────────────────────

def make_trajs():
    return {
        'hover': HoverTraj(x0=np.array([0.0, 0.0, 1.5])),
        'circle': ThreeDCircularTraj(
            center=np.array([0.0, 0.0, 1.5]),
            radius=np.array([2.0, 2.0, 0.0]),
            freq=np.array([0.2, 0.2, 0.0]),
        ),
        'lissajous': TwoDLissajous(
            A=1.5, B=1.5, a=1, b=2,
            delta=1.5708, height=1.5,
        ),
    }


# ── Wind config: symmetric C4v training set ──────────────────────────────────
#
# Training (0-8 m/s): 4 cardinal directions per speed → C4v symmetric
#   angle 0  = +x direction
#   angle 90 = +y direction
#   angle 180= -x direction
#   angle 270= -y direction
#
# Diagonals (45°/135°/225°/315°) are intentionally EXCLUDED from training
# so that test_generalization.py Test1 can evaluate diagonal-wind generalisation.
#
# OOD (>8 m/s): +x only (backward-compatible with existing evaluation scripts)

TRAIN_ANGLES  = [0, 90, 180, 270]         # 4 cardinal directions
TRAIN_SPEEDS  = [0.0, 2.0, 4.0, 6.0, 8.0]
OOD_CONFIGS   = [(10.0, 0), (12.0, 0), (15.0, 0), (15.0, 45)]  # (speed, angle_deg)

# Produce (speed, angle_deg) pairs for training
WIND_CONFIGS_TRAIN = []
for spd in TRAIN_SPEEDS:
    if spd == 0.0:
        WIND_CONFIGS_TRAIN.append((0.0, 0))   # zero-wind: direction irrelevant
    else:
        for ang in TRAIN_ANGLES:
            WIND_CONFIGS_TRAIN.append((spd, ang))

WIND_CONFIGS_OOD = list(OOD_CONFIGS)


def wind_vec_from_angle(speed, angle_deg):
    """Wind vector [wx, wy, 0] from speed (m/s) and angle (degrees CCW from +x)."""
    rad = math.radians(angle_deg)
    return np.array([speed * math.cos(rad), speed * math.sin(rad), 0.0])


def fname_for(speed, angle_deg, traj_name):
    """Filename: wind_spd{S}_ang{A:03d}_{traj}.npz"""
    return f"wind_spd{speed:.0f}_ang{angle_deg:03d}_{traj_name}.npz"


# ── Episode collection ────────────────────────────────────────────────────────

def collect_episode(wind_vec, trajectory, sim_time=20.0, dt=0.01):
    wx, wy, wz = wind_vec
    wind   = ConstantWind(wx, wy, wz) if np.any(wind_vec != 0) else NoWind()
    imu, mocap = make_sensors()

    initial_state = {
        'x':            np.array([0.0, 0.0, 1.5]),
        'v':            np.zeros(3),
        'q':            np.array([0.0, 0.0, 0.0, 1.0]),
        'w':            np.zeros(3),
        'wind':         wind_vec.copy(),
        'rotor_speeds': np.array([1788.53] * 4),
    }

    result = simulate(
        world         = make_world(),
        initial_state = initial_state,
        vehicle       = Multirotor(quad_params, initial_state),
        controller    = SE3Control(quad_params),
        trajectory    = trajectory,
        wind_profile  = wind,
        imu           = imu,
        mocap         = mocap,
        estimator     = NullEstimator(),
        t_final       = sim_time,
        t_step        = dt,
        safety_margin = 0.25,
        use_mocap     = False,
        terminate     = False,
    )

    state_dict = result[1]
    ctrl_dict  = result[2]

    pos        = state_dict['x']
    vel        = state_dict['v']
    quat       = state_dict['q']
    omega      = state_dict['w']
    wind_rec   = state_dict['wind']
    cmd_speeds = ctrl_dict['cmd_motor_speeds']

    N         = len(vel) - 1
    states    = np.zeros((N, 16))
    a_actual  = np.zeros((N, 3))
    a_nominal = np.zeros((N, 3))

    for i in range(N):
        a_act = (vel[i + 1] - vel[i]) / dt
        a_nom = nominal_model.nominal_acceleration(
            v_world          = vel[i],
            q                = quat[i],
            cmd_motor_speeds = cmd_speeds[i],
        )
        states[i]    = np.concatenate([pos[i], vel[i], quat[i], omega[i], wind_rec[i]])
        a_actual[i]  = a_act
        a_nominal[i] = a_nom

    residuals = a_actual - a_nominal
    WARMUP = 50
    return {
        'states':    states[WARMUP:],
        'controls':  cmd_speeds[WARMUP:N],
        'a_actual':  a_actual[WARMUP:],
        'a_nominal': a_nominal[WARMUP:],
        'residuals': residuals[WARMUP:],
    }


# ── Sanity check ──────────────────────────────────────────────────────────────

def quick_sanity_check():
    traj = HoverTraj(x0=np.array([0.0, 0.0, 1.5]))
    data = collect_episode(np.zeros(3), traj, sim_time=5.0)
    print(f"  samples:       {len(data['residuals'])}")
    print(f"  residual mean: {np.mean(data['residuals'], axis=0).round(4)}")
    print(f"  residual std:  {np.std(data['residuals'],  axis=0).round(4)}")
    print("  [OK] pipeline works!\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(out_dir, exist_ok=True)

    print("Step 1: Sanity check (hover, no wind, 5 s)")
    quick_sanity_check()

    all_configs = [
        ('TRAIN', WIND_CONFIGS_TRAIN),
        ('OOD',   WIND_CONFIGS_OOD),
    ]

    total_new = 0
    total_skip = 0

    for split_label, configs in all_configs:
        print(f"\n{'='*60}")
        print(f"Collecting {split_label} data ({len(configs)} wind configs "
              f"x 3 trajs = {len(configs)*3} episodes)")
        print(f"{'='*60}")

        for speed, angle_deg in configs:
            wind_vec = wind_vec_from_angle(speed, angle_deg)

            for traj_name, traj in make_trajs().items():
                fname = fname_for(speed, angle_deg, traj_name)
                fpath = os.path.join(out_dir, fname)

                if os.path.exists(fpath):
                    total_skip += 1
                    continue   # resume-friendly: skip existing files

                print(f"[{split_label}] speed={speed:.0f} m/s  "
                      f"angle={angle_deg:3d} deg  traj={traj_name}")
                try:
                    data = collect_episode(wind_vec, traj, sim_time=20.0)
                    np.savez(fpath, **data)
                    n     = len(data['residuals'])
                    r_mag = np.mean(np.linalg.norm(data['residuals'], axis=1))
                    print(f"  -> {n} samples  mean|residual|={r_mag:.4f} m/s^2  "
                          f"saved: {fname}")
                    total_new += 1
                except Exception as e:
                    print(f"  FAILED: {e}")

    print(f"\n{'='*60}")
    print(f"New files collected : {total_new}")
    print(f"Existing files skipped: {total_skip}")
    print(f"Data directory: {out_dir}")
    print()
    print("NOTE: Old asymmetric files (w*_d*.npz) are still present.")
    print("For a clean symmetric dataset, delete them manually:")
    print(f"  del {out_dir}\\w*.npz")


if __name__ == '__main__':
    main()
