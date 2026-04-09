"""
quadrotor_env.py — MuJoCo quadrotor environment

Two visualization modes:
    Hummingbird (default): hummingbird.xml, mass 0.5 kg
    Skydio X2 (--skydio): skydio_x2/scene.xml, visualization only
"""

import numpy as np
import mujoco
import os
from rotorpy.vehicles.hummingbird_params import quad_params as _hummingbird_params

# ── Physical parameters (consistent with RotorPy / PINN training) ─────────────────

K_ETA      = 5.57e-6
K_M        = 1.36e-7
MASS       = 0.5
G          = 9.81

# Hummingbird quadrotor rotor positions
ROTOR_POS  = np.array([
    [ 0.1202,  0.1202, 0.0],   # M1: (+x+y) CW
    [ 0.1202, -0.1202, 0.0],   # M2: (+x-y) CCW
    [-0.1202, -0.1202, 0.0],   # M3: (-x-y) CW
    [-0.1202,  0.1202, 0.0],   # M4: (-x+y) CCW
])
ROTOR_DIR  = np.array([1, -1, 1, -1])

C_DX, C_DY, C_DZ = 0.005, 0.005, 0.010

OMEGA_MIN   = 0.0
OMEGA_MAX   = 1500.0
HOVER_OMEGA = float(np.sqrt(MASS * G / (4 * K_ETA)))
KP_POS      = np.array([6.5, 6.5, 15.0])
KD_POS      = np.array([4.0, 4.0,  9.0])
TAU_MOTOR   = 0.015   # actuator time constant (15 ms)


class QuadrotorMuJoCoEnv:

    def __init__(self, xml_path=None, use_skydio=False):
        sim_dir = os.path.dirname(os.path.abspath(__file__))

        if xml_path is not None:
            pass
        elif use_skydio:
            candidate = os.path.join(sim_dir, 'skydio_x2', 'scene.xml')
            if os.path.exists(candidate):
                xml_path = candidate
                print("[Notice] Skydio X2 visualization model + Hummingbird physical parameters")
            else:
                print("[Warning] skydio_x2/scene.xml not found, fallback to Hummingbird")
                use_skydio = False

        if xml_path is None:
            xml_path = os.path.join(sim_dir, 'hummingbird.xml')

        self.use_skydio = use_skydio
        self.model      = mujoco.MjModel.from_xml_path(xml_path)
        self.data       = mujoco.MjData(self.model)

        # Hummingbird inertia parameters
        IXX = _hummingbird_params['Ixx']
        IYY = _hummingbird_params['Iyy']
        IZZ = _hummingbird_params['Izz']

        # ── Force total mass to MASS=0.5 kg ─────────────────────────────
        total_mass = float(self.model.body_mass.sum())
        if abs(total_mass - MASS) > 0.01:
            scale = MASS / total_mass
            self.model.body_mass[:]    *= scale
            self.model.body_inertia[:] *= scale
            print(f"  [Mass Rescaled] {total_mass:.3f}kg → {MASS}kg (×{scale:.3f})")

        # ── Find main body id ────────────────────────────────────────────
        self.body_id    = 1
        self._body_name = 'body[1]'
        for name in ['quadrotor', 'x2', 'body', 'base_link']:
            try:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid > 0:
                    self.body_id    = bid
                    self._body_name = name
                    break
            except Exception:
                pass

        # ── Locate free joint ────────────────────────────────────────────
        self._qpos_start = 0
        self._qvel_start = 0
        for i in range(self.model.njnt):
            if (self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
                    and int(self.model.jnt_bodyid[i]) == self.body_id):
                self._qpos_start = int(self.model.jnt_qposadr[i])
                self._qvel_start = int(self.model.jnt_dofadr[i])
                break

        # Override inertia tensor (ensure consistency with Hummingbird)
        self.model.body_inertia[self.body_id, :] = [IXX, IYY, IZZ]
        if use_skydio:
            print(f"  [Inertia Override] Using Hummingbird inertia: Ixx={IXX}, Iyy={IYY}, Izz={IZZ}")

        self._actual_speeds = np.ones(4) * HOVER_OMEGA
        self._wind_vec      = np.zeros(3)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        label = 'Skydio X2 (visual only)' if use_skydio else 'Hummingbird'
        print(f"[QuadrotorMuJoCoEnv] {label} | MuJoCo {mujoco.__version__}")
        print(f"  body='{self._body_name}' qpos_start={self._qpos_start}"
              f" hover={HOVER_OMEGA:.1f} rad/s τ={TAU_MOTOR*1000:.0f} ms")

        if use_skydio and self.model.nu > 0:
            print(f"  Skydio actuators={self.model.nu} (zeroed every step)")

    def reset(self, pos=None, vel=None, quat=None, omega=None):
        mujoco.mj_resetData(self.model, self.data)
        s  = self._qpos_start
        sv = self._qvel_start

        self.data.qpos[s:s+3]   = pos if pos is not None else [0., 0., 1.5]
        self.data.qpos[s+3:s+7] = [1., 0., 0., 0.]   # unit quaternion [qw,qx,qy,qz]

        if quat is not None:
            # Input [qx,qy,qz,qw] → MuJoCo [qw,qx,qy,qz]
            self.data.qpos[s+3:s+7] = [quat[3], quat[0], quat[1], quat[2]]

        self.data.qvel[sv:sv+3]   = vel   if vel   is not None else [0.,0.,0.]
        self.data.qvel[sv+3:sv+6] = omega if omega is not None else [0.,0.,0.]

        # Zero Skydio actuators
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0

        self._actual_speeds = np.ones(4) * HOVER_OMEGA
        self._wind_vec      = np.zeros(3)

        mujoco.mj_forward(self.model, self.data)
        return self._get_state()

    def _apply_forces(self, motor_speeds, wind_vec):
        """
        Thrust + torque → xfrc_applied (Hummingbird physics)

        Torque computation (r × F):
            τ_x =  ROTOR_POS[:,1] × F_i
            τ_y = -ROTOR_POS[:,0] × F_i  (note the negative sign)
            τ_z =  ROTOR_DIR × (K_M/K_ETA) × F_i
        """
        F_i   = K_ETA * motor_speeds ** 2
        T     = F_i.sum()

        tau_x =  np.sum(ROTOR_POS[:, 1] * F_i)
        tau_y = -np.sum(ROTOR_POS[:, 0] * F_i)
        tau_z =  np.sum(ROTOR_DIR * F_i) * (K_M / K_ETA)

        s       = self._qpos_start
        quat_mj = self.data.qpos[s+3:s+7].copy()
        R       = self._quat_to_rot(quat_mj)

        # Body frame → world frame
        F_world   = R @ np.array([0., 0., T])
        tau_world = R @ np.array([tau_x, tau_y, tau_z])

        # Aerodynamic drag
        sv    = self._qvel_start
        v_rel = self.data.qvel[sv:sv+3] - wind_vec

        F_drag = np.array([
            -C_DX * v_rel[0] * abs(v_rel[0]),
            -C_DY * v_rel[1] * abs(v_rel[1]),
            -C_DZ * v_rel[2] * abs(v_rel[2]),
        ])

        self.data.xfrc_applied[self.body_id, 0:3] = F_world + F_drag
        self.data.xfrc_applied[self.body_id, 3:6] = tau_world

    def step(self, cmd_motor_speeds, wind_vec=None, dt=0.01):
        cmd_motor_speeds = np.clip(cmd_motor_speeds, OMEGA_MIN, OMEGA_MAX)

        if wind_vec is None:
            wind_vec = np.zeros(3)

        self._wind_vec = wind_vec.copy()

        n_steps = max(1, int(round(dt / self.model.opt.timestep)))
        dt_sub  = float(self.model.opt.timestep)

        for _ in range(n_steps):
            # First-order motor dynamics (introduces realistic hover oscillation)
            self._actual_speeds += ((cmd_motor_speeds - self._actual_speeds)
                                    * dt_sub / TAU_MOTOR)
            self._actual_speeds  = np.clip(self._actual_speeds, OMEGA_MIN, OMEGA_MAX)

            # Zero Skydio actuators to avoid interference
            if self.model.nu > 0:
                self.data.ctrl[:] = 0.0

            self._apply_forces(self._actual_speeds, wind_vec)
            mujoco.mj_step(self.model, self.data)

        return self._get_state()

    def _get_state(self):
        s  = self._qpos_start
        sv = self._qvel_start

        pos     = self.data.qpos[s:s+3].copy()
        quat_mj = self.data.qpos[s+3:s+7].copy()

        # MuJoCo [qw,qx,qy,qz] → RotorPy [qx,qy,qz,qw]
        quat = np.array([quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]])

        vel = self.data.qvel[sv:sv+3].copy()

        # IMPORTANT: MuJoCo free joint angular velocity is already in body frame
        omega_b = self.data.qvel[sv+3:sv+6].copy()

        return {
            'x':            pos,
            'v':            vel,
            'q':            quat,
            'w':            omega_b,
            'wind':         self._wind_vec.copy(),
            'rotor_speeds': self._actual_speeds.copy(),
        }

    @staticmethod
    def _quat_to_rot(quat_mj):
        """[qw,qx,qy,qz] → rotation matrix (body → world)"""
        w, x, y, z = quat_mj
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
            [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
            [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])


def test_env(use_skydio=False):
    import sys
    print(f"\n=== Testing {'Skydio X2' if use_skydio else 'Hummingbird'} ===")

    env   = QuadrotorMuJoCoEnv(use_skydio=use_skydio)
    state = env.reset(pos=[0., 0., 1.5])

    print(f"  Initial position: {state['x'].round(4)}")

    hover  = np.ones(4) * HOVER_OMEGA
    errors = []

    for i in range(500):
        state = env.step(hover)
        errors.append(np.linalg.norm(state['x'] - np.array([0.,0.,1.5])))

        if i % 100 == 99:
            print(f"  t={i*0.01+0.01:.1f}s "
                  f"pos={state['x'].round(3)} "
                  f"err={errors[-1]:.5f} m "
                  f"ω={state['rotor_speeds'][0]:.1f}")

    avg = np.mean(errors[-100:])
    print(f"  Average hover error: {avg:.5f} m  {'✓ PASS' if avg < 0.05 else '✗ FAIL'}")


if __name__ == '__main__':
    import sys
    test_env(use_skydio='--skydio' in sys.argv)