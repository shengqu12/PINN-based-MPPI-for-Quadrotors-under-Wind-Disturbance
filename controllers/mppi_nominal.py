"""
mppi_nominal.py 
— reproduce 
- "Model Predictive Path Integral Control for Agile Unmanned Aerial  Vehicles" 
- Michal Minarik, 2024 (MPPI paper)
- using pd controller as controller, mppi as planner (Minarik uses PX4 as controller, mppi as planner)


time step and horizon:
    predicted step = n*Δt = 10*0.01 = 0.1s
    N=15 → horizon = 1.5s
    k = 896 samples per iteration
    dt = 10ms


    original:m=1.21kg, device:1024 core NVIDIA GPU
    ours:m=0.5kg,CPU
"""

import numpy as np
import torch
import os, sys, time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.trajectories.hover_traj import HoverTraj

RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# ── platform parameters ─────────────────────────────────────────────────────────
K_ETA       = quad_params['k_eta'] # thrust coefficient: T = k_eta * ω²
K_M         = quad_params['k_m']   # torque coefficient: τ = k_m * ω²
MASS        = quad_params['mass'] # drone mass (kg)
IXX         = quad_params['Ixx'] # inertia tensor (kg·m²)
IYY         = quad_params['Iyy']
IZZ         = quad_params['Izz']
G           = 9.81 
OMEGA_MIN   = float(quad_params['rotor_speed_min']) # lowest rotor speed (rad/s)
OMEGA_MAX   = float(quad_params['rotor_speed_max']) # highest rotor speed (rad/s)
T_MIN       = K_ETA * OMEGA_MIN ** 2          # 0 N rotor thrust
T_MAX       = K_ETA * OMEGA_MAX ** 2          # ~1.254 N / rotor
HOVER_OMEGA = float(np.sqrt(MASS * G / (4 * K_ETA)))
HOVER_THRUST = MASS * G                        # ~4.905 N

# J_t (equation[9]) : inertia matrix diagonal
J_np = np.array([IXX, IYY, IZZ])
J_t  = torch.tensor(J_np, dtype=torch.float32)

# omega limits (TABLE I: Drone parameters and MPPI control limits.)
OMEGA_XY_MAX = 10.0   # rad/s
OMEGA_Z_MAX  =  2.0   # rad/s

# rotor_pos format
# where's the rotor?
_rp = quad_params['rotor_pos']
ROTOR_POS = np.array(list(_rp.values()) if isinstance(_rp, dict)
                     else _rp, dtype=np.float32)   # (4,3)

# where's the rotor spinning direction? (1 or -1)
_rd = quad_params['rotor_directions']
ROTOR_DIR = np.array(list(_rd.values()) if isinstance(_rd, dict)
                     else _rd, dtype=np.float32)   # (4,)

#====== allocation matrix Γ (equation[11]) ======
# Γ = [ 1  1  1  1
#       0  l/√2  -l/√2  -l/√2  l/√2
#       -l/√2  -l/√2  l/√2  l/√2
#       -c_tf  c_tf  -c_tf  c_tf ]
# ----- c_tf = K_M / K_ETA
def _build_gamma():
    G_ = np.zeros((4, 4))
    for i in range(4):
        G_[0, i] = 1.0
        G_[1, i] = ROTOR_POS[i, 1]
        G_[2, i] = ROTOR_POS[i, 0]
        G_[3, i] = ROTOR_DIR[i] * (K_M / K_ETA)
    return G_

GAMMA_np  = _build_gamma()
# matrix convert to torch tensor
GAMMA_t   = torch.tensor(GAMMA_np,              dtype=torch.float32)
# compute pseudo-inverse for later use (equation[10])
GAMMA_INV = torch.tensor(np.linalg.pinv(GAMMA_np), dtype=torch.float32)


# ── quaternion tool function ────────────────────────────────────────────────────

def quat_rotate_batch(q, v):
    """q:(K,4)[qx,qy,qz,qw], v:(K,3) → (K,3)"""
    qx,qy,qz,qw = q[:,0],q[:,1],q[:,2],q[:,3]
    vx,vy,vz    = v[:,0],v[:,1],v[:,2]
    cx = qy*vz - qz*vy
    cy = qz*vx - qx*vz
    cz = qx*vy - qy*vx
    return torch.stack([
        vx + 2*(qw*cx + qy*cz - qz*cy),
        vy + 2*(qw*cy + qz*cx - qx*cz),
        vz + 2*(qw*cz + qx*cy - qy*cx),
    ], dim=1)

def quat_mul_batch(q1, q2):
    """quaternion q1⊗q2, (K,4)"""
    x1,y1,z1,w1 = q1[:,0],q1[:,1],q1[:,2],q1[:,3]
    x2,y2,z2,w2 = q2[:,0],q2[:,1],q2[:,2],q2[:,3]
    return torch.stack([
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2,
        w1*w2-x1*x2-y1*y2-z1*z2,
    ], dim=1)


# ── internal (equation: 8-15) ──────────────────────────────────────────

def motor_limit_pipeline(Ft, omega_des, omega_cur, dt):
    """
    equation: (8)-(15) 

    Ft:        (K,)  collective thrust (N)
    omega_des: (K,3) MPPI output desired body rates (rad/s)
    omega_cur: (K,3) current body rates (rad/s)
    dt:        scalar time step

    return Ft_clip(K,), tau_clip(K,3), omega_clip(K,3)
    """
    # equation[8]: desired angular acceleration
    omega_dot_d = (omega_des - omega_cur) / dt              # (K,3)

    # equation[9]: desired torque (with Coriolis)
    Jw   = J_t * omega_cur
    cori = torch.cross(omega_cur, Jw, dim=1)
    tau_d = J_t * omega_dot_d + cori                        # (K,3)

    # equation[10]: individual rotor thrust T_d = Γ⁻¹ × [Ft; τ_d]
    b = torch.cat([Ft.unsqueeze(1), tau_d], dim=1)          # (K,4)
    T_d = (GAMMA_INV @ b.T).T                               # (K,4)

    # equation[12]: thrust saturation
    T_clip = torch.clamp(T_d, T_MIN, T_MAX)                 # (K,4)

    # equation[13]: saturated collective thrust and torque
    FT_clip  = (GAMMA_t @ T_clip.T).T                       # (K,4)
    Ft_clip  = FT_clip[:, 0]                                 # (K,)
    tau_clip = FT_clip[:, 1:]                                # (K,3)

    # equation[14]-[15]: feasible angular rates
    Jw2         = J_t * omega_cur
    cori2       = torch.cross(omega_cur, Jw2, dim=1)
    omega_dot_c = (tau_clip - cori2) / J_t
    omega_clip  = omega_cur + omega_dot_c * dt               # (K,3)

    return Ft_clip, tau_clip, omega_clip


# ── building dynamics[6] and RK4 integration [2] ────────────────────────────────────────────────

def _deriv(pos, vel, quat, omega, Ft_clip, tau_clip):
    """continuous derivative dx/dt"""
    K = pos.shape[0]
    thrust_body      = torch.zeros(K, 3)
    thrust_body[:,2] = Ft_clip
    thrust_world     = quat_rotate_batch(quat, thrust_body)
    gravity          = torch.zeros(K, 3); gravity[:,2] = -G
    acc              = thrust_world / MASS + gravity
    Jw    = J_t * omega
    cori  = torch.cross(omega, Jw, dim=1)
    alpha = (tau_clip - cori) / J_t
    omega_q      = torch.zeros(K, 4)
    omega_q[:,:3] = omega * 0.5
    dqdt = quat_mul_batch(quat, omega_q)
    return vel, acc, dqdt, alpha   # dp, dv, dq, dω

def rk4_step(pos, vel, quat, omega, Ft_clip, tau_clip, dt):
    """RK4 integration for one step: x_{n+1} = x_n + dt/6*(k1 + 2*k2 + 2*k3 + k4)"""
    dp1,dv1,dq1,dw1 = _deriv(pos, vel, quat, omega, Ft_clip, tau_clip)

    p2=pos+dp1*(dt/2); v2=vel+dv1*(dt/2)
    q2=quat+dq1*(dt/2); w2=omega+dw1*(dt/2)
    q2=q2/(q2.norm(dim=1,keepdim=True)+1e-8)
    dp2,dv2,dq2,dw2 = _deriv(p2, v2, q2, w2, Ft_clip, tau_clip)

    p3=pos+dp2*(dt/2); v3=vel+dv2*(dt/2)
    q3=quat+dq2*(dt/2); w3=omega+dw2*(dt/2)
    q3=q3/(q3.norm(dim=1,keepdim=True)+1e-8)
    dp3,dv3,dq3,dw3 = _deriv(p3, v3, q3, w3, Ft_clip, tau_clip)

    p4=pos+dp3*dt; v4=vel+dv3*dt
    q4=quat+dq3*dt; w4=omega+dw3*dt
    q4=q4/(q4.norm(dim=1,keepdim=True)+1e-8)
    dp4,dv4,dq4,dw4 = _deriv(p4, v4, q4, w4, Ft_clip, tau_clip)

    pos_n  = pos   + dt/6*(dp1+2*dp2+2*dp3+dp4)
    vel_n  = vel   + dt/6*(dv1+2*dv2+2*dv3+dv4)
    quat_n = quat  + dt/6*(dq1+2*dq2+2*dq3+dq4)
    omg_n  = omega + dt/6*(dw1+2*dw2+2*dw3+dw4)
    quat_n = quat_n/(quat_n.norm(dim=1,keepdim=True)+1e-8)
    return pos_n, vel_n, quat_n, omg_n


# ── MPPI controller ───────────────────────────────────────────────────────

class MPPINominal:
    """

    parameter (Table II Parameters used in MPPI control):
        K=896, H=15, Δt=10ms, λ=1e-4
        n=15 → n*Δt = 1.5s
        Σ = diag(0.60, 0.15, 0.15, 0.05) ------ sigma (sampling noise applied to control input u)
        R = diag(0.01, 0.05, 0.05, 0.10) ------ R (control cost weight for u)
        R_Δ = diag(0.05, 0.10, 0.10, 0.30) ------ R_Δ (control cost weight for Δu)
        c_p=400, c_v=40, c_q=20, c_ω=20
    """

    def __init__(self, K=896, H=15, dt=0.01, n_interp=10, lam=1e-4,
                 c_p=400.0, c_v=40.0, c_q=20.0, c_w=20.0):
        self.K       = K
        self.H       = H
        self.dt      = dt
        self.n       = n_interp
        self.dt_pred = dt * n_interp    
        self.lam     = lam
        self.c_p     = c_p
        self.c_v     = c_v
        self.c_q     = c_q
        self.c_w     = c_w

        # Σ , R , R_Δ
        self.sigma = np.array([0.60, 0.15, 0.15, 0.05], dtype=np.float32)
        self.R     = torch.tensor([0.01, 0.05, 0.05, 0.10])
        self.R_dlt = torch.tensor([0.05, 0.10, 0.10, 0.30])

        # warm up with nominal hover thrust and zero body rates
        self.U = np.zeros((self.H, 4), dtype=np.float32)
        self.U[:, 0] = HOVER_THRUST

        self._compute_times = []

    def _rollout(self, state, U_sampled, ref_seq):
        """
        K rollouts in parallel, each with a control sequence of length H.

        U_sampled: (K,H,4) = [Ft, ωx_des, ωy_des, ωz_des]
        ref_seq:   (H,) = list of reference states for each step in the horizon
        """
        K = self.K

        pos   = torch.tensor(state['x'], dtype=torch.float32).expand(K,-1).clone()
        vel   = torch.tensor(state['v'], dtype=torch.float32).expand(K,-1).clone()
        quat  = torch.tensor(state['q'], dtype=torch.float32).expand(K,-1).clone()
        omega = torch.tensor(state['w'], dtype=torch.float32).expand(K,-1).clone()
        U_t   = torch.tensor(U_sampled, dtype=torch.float32) # U_sampled: (896,15,4) control added with noise 896 kind of control sequence

        costs = torch.zeros(K)

        for j in range(self.H):
            u     = U_t[:, j, :]           # (K,4)
            Ft    = u[:, 0]                 # (K,)
            w_des = u[:, 1:]               # (K,3) expexted body rates from MPPI output

            # angular limit (Table I)
            w_des = torch.cat([
                torch.clamp(w_des[:,:2], -OMEGA_XY_MAX, OMEGA_XY_MAX),
                torch.clamp(w_des[:,2:], -OMEGA_Z_MAX, OMEGA_Z_MAX),
            ], dim=1)

            # ω_des → [Ft_clip, tau_clip, omega_clip]
            Ft_clip, tau_clip, omega_clip = motor_limit_pipeline(
                Ft, w_des, omega, self.dt_pred
            )

            # Using RK4 to predict next state (equation[6]) with the limited control input
            pos, vel, quat, omega = rk4_step(
                pos, vel, quat, omega, Ft_clip, tau_clip, self.dt_pred
            )
            omega = omega_clip

            # NaN and inf protection: any abnormal value falls back to a reasonable default
            pos   = torch.nan_to_num(pos,   nan=0.0, posinf=1e3, neginf=-1e3)
            vel   = torch.nan_to_num(vel,   nan=0.0, posinf=1e3, neginf=-1e3)
            quat  = torch.nan_to_num(quat,  nan=0.0)
            omega = torch.nan_to_num(omega, nan=0.0, posinf=10., neginf=-10.)
            # quaternion normalization
            qn = quat.norm(dim=1, keepdim=True).clamp(min=1e-8)
            quat = quat / qn

            # ── reference state ──
            p_ref = torch.tensor(ref_seq[j]['x'],     dtype=torch.float32)
            v_ref = torch.tensor(ref_seq[j]['x_dot'], dtype=torch.float32)
            # reference quaternion: use current reference acceleration to infer orientation (simplified: hover case q_ref=[0,0,0,1])
            q_ref = torch.tensor([0., 0., 0., 1.],    dtype=torch.float32)
            w_ref = torch.zeros(3)

            # ── state tracking cost (equation [17-18]) ──
            e_p  = pos - p_ref
            e_v  = vel - v_ref
            e_w  = omega - w_ref
            # quaternion distance: d_q = 1 - <q1,q2>² equation[21]
            qdot = (quat * q_ref).sum(dim=1)
            d_q  = 1.0 - qdot**2

            costs += (self.c_p * (e_p**2).sum(dim=1)
                    + self.c_v * (e_v**2).sum(dim=1)
                    + self.c_q * d_q
                    + self.c_w * (e_w**2).sum(dim=1))

            # ── control penalty (equation[16]): ||u||^2_R ──
            u_nom       = torch.zeros_like(u)
            u_nom[:,0]  = HOVER_THRUST
            costs      += (self.R * (u - u_nom)**2).sum(dim=1)

            # ── control rate penalty (equation[16]): ||Delta_u||^2_R_delta ──
            if j > 0:
                du     = U_t[:,j,:] - U_t[:,j-1,:]
                costs += (self.R_dlt * du**2).sum(dim=1)

        return costs.numpy()

    def update(self, state, ref_seq):
        """
        calculate optimal control u* = [Ft, ωx_des, ωy_des, ωz_des]
        return (u_opt (4,), dt_ms)
        """
        t0 = time.perf_counter()

        # sampling noise (equation [2])
        noise       = (np.random.randn(self.K, self.H, 4).astype(np.float32)
                       * self.sigma[None,None,:])
        noise[0]    = 0.0
        U_sampled   = self.U[None,:,:] + noise

        # thrust limit (equation [12]): clip the collective thrust Ft to [0, 4*T_MAX]
        U_sampled[:,:,0] = np.clip(U_sampled[:,:,0], 0.0,
                                   4 * T_MAX)

        # Rollout costs estimation (equation [3])
        costs = self._rollout(state, U_sampled, ref_seq)

        # MPPI weights (equation [4])
        rho     = costs.min()
        weights = np.exp(-(costs - rho) / self.lam)
        weights = weights / (weights.sum() + 1e-8)

        # weighted control update (equation [5])
        delta_u = np.einsum('k,khd->hd', weights,
                            U_sampled - self.U[None,:,:])
        U_new   = self.U + delta_u

        # warm up for next step: shift control sequence and append the last one
        self.U[:-1] = U_new[1:]
        self.U[-1]  = U_new[-1]
        self.U[-1, 0] = HOVER_THRUST    

        t1    = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0
        self._compute_times.append(dt_ms)

        return U_new[0].copy(), dt_ms

    def compute_frequency_stats(self):
        if not self._compute_times:
            return {}
        times = np.array(self._compute_times)
        hz    = 1000.0 / times
        return {
            'mean_ms': float(times.mean()),
            'std_ms':  float(times.std()),
            'mean_hz': float(hz.mean()),
            'min_hz':  float(hz.min()),
        }


# ── control - actuator────────────────────────────────────────

def u_to_motor_speeds(u_opt, omega_cur):
    """
    Convert MPPI output [Ft, wx_des, wy_des, wz_des] to motor speeds.
    NaN protection: fall back to hover speed on any invalid value.
    """
    if np.any(np.isnan(u_opt)) or np.any(np.isinf(u_opt)):
        return np.ones(4) * HOVER_OMEGA

    Ft    = torch.tensor([float(np.clip(u_opt[0], 0.0, 4*T_MAX))],
                         dtype=torch.float32)
    w_des = torch.tensor(u_opt[1:][None,:], dtype=torch.float32)
    w_cur = torch.tensor(omega_cur[None,:],  dtype=torch.float32)
    w_des = torch.nan_to_num(w_des, nan=0.0)
    w_cur = torch.nan_to_num(w_cur, nan=0.0)

    w_des = torch.cat([
        torch.clamp(w_des[:,:2], -OMEGA_XY_MAX, OMEGA_XY_MAX),
        torch.clamp(w_des[:,2:], -OMEGA_Z_MAX,   OMEGA_Z_MAX),
    ], dim=1)

    Ft_clip, tau_clip, _ = motor_limit_pipeline(Ft, w_des, w_cur, dt=0.01)

    b     = torch.cat([Ft_clip.unsqueeze(1), tau_clip], dim=1)
    T     = (GAMMA_INV @ b.T).T.squeeze(0)
    T     = torch.clamp(T, T_MIN + 1e-8, T_MAX)
    omega = torch.sqrt(T / K_ETA)
    result = np.clip(omega.numpy(), OMEGA_MIN, OMEGA_MAX)
    if np.any(np.isnan(result)):
        return np.ones(4) * HOVER_OMEGA
    return result


# ── Body rate inner ring PD controller ────────────────────────────────────────

def body_rate_pd_to_motors(omega_des, omega_cur, Ft,
                            Kp=0.5, Kd=0.01):
    """
    inner body rate PD controller (simulate PX4 body rate controller)
    """
    e_w = omega_des - omega_cur          # (3,) desired body rate error
    tau = Kp * J_np * e_w               # (3,) desired torque from PD control

    b = np.array([Ft, tau[0], tau[1], tau[2]])
    T = np.linalg.solve(GAMMA_np, b)    # (4,) per-rotor thrust
    T = np.clip(T, T_MIN + 1e-8, T_MAX)
    omega_motors = np.sqrt(T / K_ETA)
    return np.clip(omega_motors, OMEGA_MIN, OMEGA_MAX)


# ── closed-loop simulation ─────────────────────────────────────────────────

def run_episode(wind_vec, trajectory_fn,
                K=896, H=15, n_interp=10,
                sim_time=15.0, dt=0.01,
                verbose=False):
    """Minarik 2024 MPPI closed-loop simulation with nominal MPPI controller."""
    state = {
        'x':            np.array([0., 0., 1.5]),
        'v':            np.zeros(3),
        'q':            np.array([0., 0., 0., 1.]),
        'w':            np.zeros(3),
        'wind':         wind_vec.copy(),
        'rotor_speeds': np.ones(4) * HOVER_OMEGA,
    }

    vehicle    = Multirotor(quad_params, state)
    trajectory = trajectory_fn()
    mppi       = MPPINominal(K=K, H=H, dt=dt, n_interp=n_interp)

    N_steps   = int(sim_time / dt)
    times     = np.zeros(N_steps)
    positions = np.zeros((N_steps, 3))
    des_pos   = np.zeros((N_steps, 3))
    errors    = np.zeros(N_steps)
    comp_ms   = np.zeros(N_steps)

    t = 0.0
    for i in range(N_steps):
        state['wind'] = wind_vec.copy()

        # reference sequence for MPPI rollouts: predict the reference state at each step in the horizon
        ref_seq = [
            trajectory.update(t + j * mppi.dt_pred)
            for j in range(mppi.H)
        ]

        u_opt, dt_ms = mppi.update(state, ref_seq)
        comp_ms[i]   = dt_ms

        # MPPI output → motor commands (with NaN protection and limits)
        Ft_des    = float(np.clip(u_opt[0], 0.0, 4*T_MAX))
        omega_des = np.clip(u_opt[1:],
                            [-OMEGA_XY_MAX,-OMEGA_XY_MAX,-OMEGA_Z_MAX],
                            [ OMEGA_XY_MAX, OMEGA_XY_MAX, OMEGA_Z_MAX])

        # inner body rate PD (simulate PX4 body rate controller)
        motor_speeds = body_rate_pd_to_motors(
            omega_des, state['w'], Ft_des
        )

        cmd   = {'cmd_motor_speeds': motor_speeds}
        state = vehicle.step(state, cmd, dt)

        times[i]     = t
        positions[i] = state['x']
        des_pos[i]   = ref_seq[0]['x']
        errors[i]    = np.linalg.norm(positions[i] - des_pos[i])
        t           += dt

        if verbose and i % 100 == 0:
            hz = 1000.0 / dt_ms if dt_ms > 0 else 0
            print(f"  t={t:.1f}s  err={errors[i]:.4f}m  {hz:.0f}Hz", end='\r')

    if verbose:
        print()

    return {
        'times':      times,
        'positions':  positions,
        'des_pos':    des_pos,
        'errors':     errors,
        'comp_ms':    comp_ms,
        'mean_err':   errors[100:].mean(),
        'freq_stats': mppi.compute_frequency_stats(),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("Minarik 2024 MPPI parameters:")
    print(f"  K={896}, H={15}, n_interp={10}")
    print(f"  prediction: {10*0.01:.2f}s  horizon: {15*10*0.01:.1f}s")
    print(f"  hover speed: {HOVER_OMEGA:.1f} rad/s")
    print(f"  speed limits: [{OMEGA_MIN}, {OMEGA_MAX}] rad/s")
    print(f"  individual motor thrust limits: [{T_MIN:.4f}, {T_MAX:.4f}] N")

    trajectories = {
        'hover':  lambda: HoverTraj(x0=np.array([0., 0., 1.5])),
        'circle': lambda: ThreeDCircularTraj(
            center=np.array([0., 0., 1.5]),
            radius=np.array([1.5, 1.5, 0.]),
            freq=np.array([0.2, 0.2, 0.]),
        ),
    }

    test_configs = [
        (0.0,  'train'),
        (4.0,  'train'),
        (8.0,  'train'),
        (10.0, 'OOD'),
        (12.0, 'OOD'),
    ]

    for traj_name, traj_fn in trajectories.items():
        print(f"\n{'='*52}")
        print(f"trajectory: {traj_name}")
        print(f"{'='*52}")
        print(f"\n{'Wind':>6} | {'Split':>5} | {'Error (m)':>10} | {'Hz':>7}")
        print("-"*36)

        for wind_speed, split in test_configs:
            wind_vec = np.array([wind_speed, 0., 0.])
            r  = run_episode(wind_vec, traj_fn, verbose=False)
            hz = r['freq_stats']['mean_hz']
            print(f"{wind_speed:6.0f} | {split:>5} | "
                  f"{r['mean_err']:10.4f} | {hz:7.1f}")


if __name__ == '__main__':
    main()