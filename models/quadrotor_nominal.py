import numpy as np

class QuadrotorNominal:
    """
    quadrotor nominal dynamics
    input: state(12): [x, y, z, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz], cmd_motor_speeds(4): torque command for 4 motors (rad/s)
    output: a_nominal(3) — world frame linear acceleration
    a_nominal = thrust/m + gravity (no drag).
    Drag is excluded so the PINN residual (a_actual - a_nominal) captures both
    aerodynamic drag and wind disturbance, matching the Minarik et al. rollout model.
    """

    def __init__(self, quad_params):
        self.mass  = quad_params['mass']
        self.g     = 9.81
        self.k_eta = quad_params['k_eta']   # thrust coefficient

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

    def nominal_acceleration(self, v_world, q, cmd_motor_speeds):
        """
        calculate nominal linear acceleration (world frame): thrust/m + gravity.

        Args:
            v_world:          (3,) world frame velocity (unused, kept for API compatibility)
            q:                (4,) quaternion [qx,qy,qz,qw]
            cmd_motor_speeds: (4,) motor speeds rad/s

        Returns:
            a_nominal: (3,) world frame linear acceleration
        """
        R = self.quat_to_R(q)

        T = self.motor_speeds_to_thrust(cmd_motor_speeds)
        thrust_world = R @ np.array([0.0, 0.0, T]) / self.mass

        gravity = np.array([0.0, 0.0, -self.g])

        return thrust_world + gravity