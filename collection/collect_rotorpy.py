"""
data collection script v1.0
output format (each .npz file):
    states:    (N, 16) — [pos(3), vel(3), quat(4), omega(3), wind(3)]
    controls:  (N, 4)  — cmd_motor_speeds rad/s
    a_actual:  (N, 3)  — numerically differentiated actual acceleration
    a_nominal: (N, 3)  — analytically predicted acceleration
    residuals: (N, 3)  — PINN training labels = a_actual - a_nominal
"""

import numpy as np
import os, sys

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

# print params
# using hummingbird as my drone model
print("=== Hummingbird params ===")
for k in ['mass', 'Ixx', 'Iyy', 'Izz', 'k_eta', 'c_Dx', 'c_Dy', 'c_Dz']: # Ixx rotational inertia, k_eta thrust coefficient, c_Dx/Dy/Dz linear drag coefficients
    print(f"  {k}: {quad_params[k]}")

nominal_model = QuadrotorNominal(quad_params)

# create an empty box: x [-8,8], y [-8,8], z [0,12]
def make_world():
    return World.empty(([-8, 8], [-8, 8], [0, 12]))

# set imu and mocap sampling rates, and other sensor configs if needed.
# IMU: (acc, gyro)
# Mocap:(pos)
def make_sensors():
    imu   = Imu()
    mocap = MotionCapture(sampling_rate=100)
    return imu, mocap

# collect one episode of data with given wind and trajectory, return dict of arrays (states, controls, a_actual, a_nominal, residuals)
def collect_episode(wind_vec, trajectory, sim_time=20.0, dt=0.01):
    """
    run episode with given wind and trajectory, 
    """
    wx, wy, wz = wind_vec
    wind = ConstantWind(wx, wy, wz) if np.any(wind_vec != 0) else NoWind()

    initial_state = {
        'x':            np.array([0.0, 0.0, 1.5]),
        'v':            np.zeros(3),
        'q':            np.array([0.0, 0.0, 0.0, 1.0]),
        'w':            np.zeros(3),
        'wind':         wind_vec.copy(),
        'rotor_speeds': np.array([1788.53] * 4),
    }

    vehicle    = Multirotor(quad_params, initial_state) # dynamic model
    controller = SE3Control(quad_params) # controller
    imu, mocap = make_sensors()

    result = simulate(
        world         = make_world(),
        initial_state = initial_state,
        vehicle       = vehicle,
        controller    = controller,
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

    # decouple tuple
    state_dict = result[1]
    ctrl_dict  = result[2]

    pos        = state_dict['x']                # (N,3)
    vel        = state_dict['v']                # (N,3)
    quat       = state_dict['q']                # (N,4) [qx,qy,qz,qw]
    omega      = state_dict['w']                # (N,3)
    wind_rec   = state_dict['wind']             # (N,3)
    cmd_speeds = ctrl_dict['cmd_motor_speeds']  # (N,4)

    N = len(vel) - 1  #  terminal has no next state

    states    = np.zeros((N, 16))
    a_actual  = np.zeros((N, 3))
    a_nominal = np.zeros((N, 3))

    for i in range(N):
        # a_actual: vel[i+1] - vel[i] / dt
        a_act = (vel[i + 1] - vel[i]) / dt

        # a_nominal: without wind disturbance.
        a_nom = nominal_model.nominal_acceleration(
            v_world          = vel[i],
            q                = quat[i],
            cmd_motor_speeds = cmd_speeds[i],
        )

        states[i]    = np.concatenate([pos[i], vel[i], quat[i], omega[i], wind_rec[i]])
        a_actual[i]  = a_act
        a_nominal[i] = a_nom
    
    # the error between real world model and nominal model.
    residuals = a_actual - a_nominal

    WARMUP = 50 # skip first 0.5 seconds to avoid large transient effects from initialization
    return {
        'states':    states[WARMUP:],          # (N,16)
        'controls':  cmd_speeds[WARMUP:N],  # (N,4)
        'a_actual':  a_actual[WARMUP:],        # (N,3)
        'a_nominal': a_nominal[WARMUP:],       # (N,3)
        'residuals': residuals[WARMUP:],       # (N,3) PINN training labels = a_actual - a_nominal
    }


def quick_sanity_check():
    """sanity check: hover + no wind, confirm pipeline works. 
    should see small residuals (mostly from numerical differentiation noise) and no obvious bias. 
    """
    traj = HoverTraj(x0=np.array([0.0, 0.0, 1.5])) # hover at 1.5 m, should be stable without wind, residuals should be small.
    data = collect_episode(
        wind_vec   = np.zeros(3),
        trajectory = traj,
        sim_time   = 5.0,
    )
    print(f"  samples:          {len(data['residuals'])}")
    print(f"  residual mean:    {np.mean(data['residuals'], axis=0).round(4)}")
    print(f"  residual std:     {np.std(data['residuals'],  axis=0).round(4)}")
    print(f"  residual max_abs: {np.max(np.abs(data['residuals']), axis=0).round(4)}")
    print("  [OK] pipeline works!\n")
    return data

#  three trajectories: hover, circle, lissajous.
def make_trajs():
    return {
        'hover':     HoverTraj(x0=np.array([0.0, 0.0, 1.5])),
        'circle':    ThreeDCircularTraj(
                         center=np.array([0.0, 0.0, 1.5]),
                         radius=np.array([2.0, 2.0, 0.0]),
                         freq=np.array([0.2, 0.2, 0.0]),
                     ),
        'lissajous': TwoDLissajous(
                         A=1.5, B=1.5, a=1, b=2,
                         delta=1.5708,
                         height=1.5,
                     ),
    }


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: sanity check
    quick_sanity_check()

    # Step 2: collect data with different wind conditions and trajectories.
    # 0-8 m/s → training set   |   10-15 m/s → OOD test set
    wind_configs = [
        # training set
        (0.0,  [1, 0, 0]),
        (2.0,  [1, 0, 0]),
        (4.0,  [1, 0, 0]),
        (4.0,  [0, 1, 0]),
        (6.0,  [1, 0, 0]),
        (6.0,  [0.7, 0.7, 0]),
        (8.0,  [1, 0, 0]),
        (8.0,  [0, 1, 0]),
        # OOD test set
        (10.0, [1, 0, 0]),
        (12.0, [1, 0, 0]),
        (15.0, [1, 0, 0]),
        (15.0, [0.7, 0.7, 0]),
    ]

    total = 0

    for wind_speed, wind_dir in wind_configs:
        wind_vec = wind_speed * np.array(wind_dir, dtype=float)
        wind_tag = f"w{wind_speed:.0f}_d{wind_dir[0]:.1f}{wind_dir[1]:.1f}"

        for traj_name, traj in make_trajs().items():
            label = "TRAIN" if wind_speed <= 8 else "OOD"
            print(f"[{label}] wind={wind_speed} m/s  traj={traj_name}")

            try:
                data = collect_episode(
                    wind_vec   = wind_vec,
                    trajectory = traj,
                    sim_time   = 20.0,
                )
                fname = os.path.join(out_dir, f"{wind_tag}_{traj_name}.npz")
                np.savez(fname, **data)

                n     = len(data['residuals'])
                total += n
                r_mag = np.mean(np.linalg.norm(data['residuals'], axis=1))
                print(f"  -> {n} samples, mean |residual| = {r_mag:.4f} m/s2")

            except Exception as e:
                print(f"  FAILED: {e}")

    print(f"\n{'='*40}")
    print(f"Total samples: {total}")
    print(f"Saved to: {out_dir}")


if __name__ == '__main__':
    main()