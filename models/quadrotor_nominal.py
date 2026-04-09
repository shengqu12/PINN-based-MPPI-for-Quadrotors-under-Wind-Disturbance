import numpy as np

class QuadrotorNominal:
    """
    quadrotor nominal dynamics
    input: state(12): [x, y, z, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz], cmd_motor_speeds(4): torque command for 4 motors (rad/s)
    output: a_nominal(3) — world frame linear acceleration
    a_nominal is based on the nominal model(physical equations) without any disturbance or model mismatch. It is used as the "reference" for the disturbance observer to estimate the disturbance acceleration.
    a_actual is the actual linear acceleration of the quadrotor, which is be measured by sensors (e.g., IMU) or obtained from high-fidelity simulation. 
    The difference between a_actual and a_nominal is the disturbance acceleration, which includes all unmodeled dynamics, external disturbances (e.g., wind), and model mismatch. 
    """

    def __init__(self, quad_params):
        self.mass = quad_params['mass']
        self.g    = 9.81
        self.k_eta = quad_params['k_eta']   # thrust 
        # linear drag coefficients (body frame)
        self.c_Dx = quad_params['c_Dx']
        self.c_Dy = quad_params['c_Dy']
        self.c_Dz = quad_params['c_Dz']

    # thrust from each motor F = k_eta * omega^2
    def motor_speeds_to_thrust(self, cmd_motor_speeds):
        """cmd_motor_speeds (4,) rad/s → total thrust (scalar)"""
        forces = self.k_eta * cmd_motor_speeds**2   
        return np.sum(forces)
    
    # calcute Rotration matrix from quaternion: objective: body frame → world frame
    def quat_to_R(self, q):
        """quaternion [qx,qy,qz,qw] → rotation matrix R (body→world)"""
        qx, qy, qz, qw = q
        R = np.array([
            [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
            [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
        ])
        return R

    # calculate accerlation based on the nominal model (thrust, gravity, linear drag) without any disturbance or model mismatch.
    def nominal_acceleration(self, v_world, q, cmd_motor_speeds):
        """
        calculate nominal linear acceleration (world frame)

        Args:
            v_world:          (3,) world frame velocity
            q:                (4,) quaternion [qx,qy,qz,qw]
            cmd_motor_speeds: (4,) motor speeds rad/s

        Returns:
            a_nominal: (3,) world frame linear acceleration
        """
        R = self.quat_to_R(q)

        # thrust (body frame z-axis positive direction)
        T = self.motor_speeds_to_thrust(cmd_motor_speeds)
        thrust_world = R @ np.array([0.0, 0.0, T]) / self.mass # a = F/m, and thrust is in body frame, so transform to world frame using R.

        # gravity
        gravity = np.array([0.0, 0.0, -self.g])

        # linear drag : assume drag is linear proportional to velocity
        v_body = R.T @ v_world
        drag_body = -np.array([
            self.c_Dx * v_body[0],
            self.c_Dy * v_body[1],
            self.c_Dz * v_body[2],
        ]) / self.mass
        drag_world = R @ drag_body

        return thrust_world + gravity + drag_world # a = F_thrust/m + F_gravity + F_drag/m