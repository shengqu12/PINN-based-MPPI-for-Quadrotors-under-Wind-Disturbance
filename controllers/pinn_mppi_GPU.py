"""
pinn_mppi_GPU.py — PINN-MPPI controller 

Comparison to Minarik's:
                    Minarik (2024)                  This implementation(ours)
controll input:     [Ft, wx, wy, wz]                [αx, αy, αz]
inner controller:   PX4 body rate PID             SE3Control (PID)
dynamics:           RK4 with Coriolis           RK4 with Coriolis(same)
hardware:           GPU(Jetson Orin)               GPU (CUDA Graph)
learning model:     None                       PINN residual model (same)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import sys, time

torch.set_float32_matmul_precision('high')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.pinn import ResidualPINN
from training.dataset import Normalizer

from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.trajectories.hover_traj import HoverTraj

CKPT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

K_ETA       = quad_params['k_eta']
K_M         = quad_params['k_m']
MASS        = quad_params['mass']
IXX         = quad_params['Ixx']
IYY         = quad_params['Iyy']
IZZ         = quad_params['Izz']
G           = 9.81
OMEGA_MIN   = quad_params['rotor_speed_min']
OMEGA_MAX   = quad_params['rotor_speed_max']
HOVER_OMEGA = float(np.sqrt(MASS * G / (4 * K_ETA)))

_rp = quad_params['rotor_pos']
ROTOR_POS = np.array(list(_rp.values()) if isinstance(_rp, dict)
                     else _rp, dtype=np.float32)
_rd = quad_params['rotor_directions']
ROTOR_DIR = np.array(list(_rd.values()) if isinstance(_rd, dict)
                     else _rd, dtype=np.float32)

KP_POS = np.array([6.5, 6.5, 15.0])
KD_POS = np.array([4.0, 4.0,  9.0])

EMA_ALPHA   = 0.85
DP_CALC_MAX = 0.25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    print(f"[MPPI] GPU is activated: {torch.cuda.get_device_name(0)}")
else:
    print("[MPPI] Using CPU (CUDA Graph not available)")


# ── tensors ────────────────────────────────────────────────────────

_ARM = _RDIR = _I = None

def _get_tensors(device):
    global _ARM, _RDIR, _I
    if _ARM is None or _ARM.device != device:
        _ARM  = torch.tensor(ROTOR_POS, dtype=torch.float32, device=device)
        _RDIR = torch.tensor(ROTOR_DIR, dtype=torch.float32, device=device)
        _I    = torch.tensor([IXX, IYY, IZZ], dtype=torch.float32, device=device)
    return _ARM, _RDIR, _I


# ── RK4 dynamics (with Coriolis, corresponding to Minarik's formula 6) ────────────────────────

def quat_rotate_batch(q, v):
    qx,qy,qz,qw = q[:,0],q[:,1],q[:,2],q[:,3]
    vx,vy,vz    = v[:,0],v[:,1],v[:,2]
    cx = qy*vz - qz*vy; cy = qz*vx - qx*vz; cz = qx*vy - qy*vx
    return torch.stack([
        vx + 2*(qw*cx + qy*cz - qz*cy),
        vy + 2*(qw*cy + qz*cx - qx*cz),
        vz + 2*(qw*cz + qx*cy - qy*cx),
    ], dim=1)

def quat_mul_batch(q1, q2):
    x1,y1,z1,w1 = q1[:,0],q1[:,1],q1[:,2],q1[:,3]
    x2,y2,z2,w2 = q2[:,0],q2[:,1],q2[:,2],q2[:,3]
    return torch.stack([
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2,
        w1*w2-x1*x2-y1*y2-z1*z2,
    ], dim=1)

def _dynamics_deriv(pos, vel, quat, omega, u, device,
                    arm=None, rdir=None, I=None):
    """
    calculate derivatives of state variables (dp, dv, dq, dw) according to Minarik's formula 6.
    """
    K = pos.shape[0]
    if arm is None:
        arm, rdir, I = _get_tensors(device)

    F_i = K_ETA * u**2
    T   = F_i.sum(dim=1)

    # using zeros_like to ensure the same dtype and device
    thrust_body = torch.zeros_like(pos)
    thrust_body[:,2] = T
    thrust_world = quat_rotate_batch(quat, thrust_body)

    gravity = torch.zeros_like(pos)
    gravity[:,2] = -G
    dp  = vel
    dv  = thrust_world / MASS + gravity

    tau_x = (arm[:,1] * F_i).sum(dim=1)
    tau_y = (arm[:,0] * F_i).sum(dim=1)
    tau_z = (rdir * F_i * K_M).sum(dim=1)
    tau   = torch.stack([tau_x, tau_y, tau_z], dim=1)
    Jw    = I * omega
    cori  = torch.cross(omega, Jw, dim=1)
    dw    = (tau - cori) / I

    omega_q = torch.zeros(K, 4, device=device)
    omega_q[:,:3] = omega * 0.5
    dq = quat_mul_batch(quat, omega_q)
    return dp, dv, dq, dw

def nominal_step_batch(pos, vel, quat, omega, u, dt, device=None,
                       arm=None, rdir=None, I=None):
    if device is None:
        device = pos.device
    dp1,dv1,dq1,dw1 = _dynamics_deriv(pos,vel,quat,omega,u,device,arm,rdir,I)
    p2=pos+dp1*(dt/2); v2=vel+dv1*(dt/2)
    q2=quat+dq1*(dt/2); w2=omega+dw1*(dt/2)
    q2=q2/(q2.norm(dim=1,keepdim=True)+1e-8)
    dp2,dv2,dq2,dw2 = _dynamics_deriv(p2,v2,q2,w2,u,device,arm,rdir,I)
    p3=pos+dp2*(dt/2); v3=vel+dv2*(dt/2)
    q3=quat+dq2*(dt/2); w3=omega+dw2*(dt/2)
    q3=q3/(q3.norm(dim=1,keepdim=True)+1e-8)
    dp3,dv3,dq3,dw3 = _dynamics_deriv(p3,v3,q3,w3,u,device,arm,rdir,I)
    p4=pos+dp3*dt; v4=vel+dv3*dt
    q4=quat+dq3*dt; w4=omega+dw3*dt
    q4=q4/(q4.norm(dim=1,keepdim=True)+1e-8)
    dp4,dv4,dq4,dw4 = _dynamics_deriv(p4,v4,q4,w4,u,device,arm,rdir,I)
    pos_new  = pos   + dt/6*(dp1+2*dp2+2*dp3+dp4)
    vel_new  = vel   + dt/6*(dv1+2*dv2+2*dv3+dv4)
    quat_new = quat  + dt/6*(dq1+2*dq2+2*dq3+dq4)
    omg_new  = omega + dt/6*(dw1+2*dw2+2*dw3+dw4)
    quat_new = quat_new/(quat_new.norm(dim=1,keepdim=True)+1e-8)
    return pos_new, vel_new, quat_new, omg_new


# ── PINN inference ────────────────────────────────────────────────────────

def pinn_batch_refvel(model, normalizer, vel_ref, quat, omega,
                      motor_speeds, wind_vec, device=None):
    if device is None:
        device = vel_ref.device
    K    = vel_ref.shape[0]
    wind = torch.tensor(wind_vec, dtype=torch.float32, device=device).expand(K, 3)
    v_rel  = vel_ref - wind
    X      = torch.cat([v_rel, quat, omega, motor_speeds, wind], dim=1)
    X_norm = (X - normalizer.mean) / normalizer.std
    with torch.no_grad():
        return model(X_norm, v_rel)


def pinn_predict(model, normalizer, state, motor_speeds,
                 wind_vec, vel_ref=None):
    vel   = vel_ref.astype(np.float32) if vel_ref is not None \
            else state['v'].astype(np.float32)
    wind  = wind_vec.astype(np.float32)
    v_rel = vel - wind
    X = np.concatenate([v_rel, state['q'].astype(np.float32),
                        state['w'].astype(np.float32),
                        motor_speeds.astype(np.float32), wind])
    dev    = normalizer.mean.device
    X_t    = torch.as_tensor(X, dtype=torch.float32, device=dev).unsqueeze(0)
    vr_t   = torch.as_tensor(v_rel, dtype=torch.float32, device=dev).unsqueeze(0)
    X_norm = (X_t - normalizer.mean) / normalizer.std
    with torch.no_grad():
        return model(X_norm, vr_t).squeeze(0).cpu().numpy()


# ── MPPI controller（CUDA Graph acceleration）────────────────────────────────────

class MPPIController:
    """
    MPPI controller with optional PINN residual compensation, designed for GPU execution and accelerated by CUDA Graphs.
    CUDA Graph mechanism:
        First update → allocate static tensors + capture rollout computation graph
        Subsequent updates → memcpy to update inputs + single graph.replay()

    control variables:
        α = [αx, αy, αz] ∈ [0, alpha_max]³
        Δp_i = α_i × clip(-r_ema_i / kp_i, -0.25, 0.25)
        corresponding to the diagonal disturbance adaptation matrix Λ = diag(αx, αy, αz)
    """

    def __init__(self, model, normalizer, wind_vec,
                 K=896, H=15, dt=0.01,
                 sigma=0.15, lam=0.1, alpha_max=1.2,
                 ema_alpha=EMA_ALPHA,
                 dp_calc_max=DP_CALC_MAX,
                 use_pinn=True,
                 device=None):
        self.device = device or DEVICE
        self.model  = model.to(self.device)
        self.normalizer      = normalizer
        self.normalizer.mean = normalizer.mean.to(self.device)
        self.normalizer.std  = normalizer.std.to(self.device)
        self.wind_vec    = np.array(wind_vec, dtype=np.float32)
        self.K           = K
        self.H           = H
        self.dt          = dt
        self.sigma       = sigma
        self.lam         = lam
        self.alpha_max   = alpha_max
        self.ema_alpha   = ema_alpha
        self.dp_calc_max = dp_calc_max
        self.use_pinn    = use_pinn

        # self.U in the GPU
        self.U = torch.full((H, 3), 0.3, dtype=torch.float32, device=self.device)
        self.residual_ema = None

        self.Q_pos  = 100.0
        self.Q_vel  = 2.0
        self.R_ctrl = 0.0001

        self._compute_times = []

        # CUDA Graph 
        self._graph         = None   
        self._graph_ready   = False

        # static tensor（CUDA Graph ）
        self._st = {}

    def update_ema(self, residual):
        if self.residual_ema is None:
            self.residual_ema = residual.copy()
        else:
            self.residual_ema = (self.ema_alpha * self.residual_ema
                                 + (1 - self.ema_alpha) * residual)
        dp_calc = -self.residual_ema / KP_POS
        return np.clip(dp_calc, -self.dp_calc_max, self.dp_calc_max)

    def _allocate_static_tensors(self):
        K, H, dev = self.K, self.H, self.device
        self._st = {
            'pos':     torch.zeros(K, 3,    device=dev),
            'vel':     torch.zeros(K, 3,    device=dev),
            'quat':    torch.zeros(K, 4,    device=dev),
            'omega':   torch.zeros(K, 3,    device=dev),
            'U':       torch.zeros(K, H, 3, device=dev),
            'pos_des': torch.zeros(H, 3,    device=dev),
            'dp_base': torch.zeros(3,       device=dev),
            'res':     torch.zeros(K, 3,    device=dev),
            'u_hover': torch.full((K, 4), HOVER_OMEGA, device=dev),
            'costs':   torch.zeros(K,       device=dev),
        }
        self._gravity     = torch.zeros(K, 3, device=dev); self._gravity[:,2] = -G
        self._thrust_body = torch.zeros(K, 3, device=dev)
        self._omega_q     = torch.zeros(K, 4, device=dev)

    def _rollout_static(self):
        st  = self._st
        dev = self.device
        arm, rdir, I = self._cached_arm, self._cached_rdir, self._cached_I

        pos   = st['pos'].clone()
        vel   = st['vel'].clone()
        quat  = st['quat'].clone()
        omega = st['omega'].clone()

        st['costs'].zero_()

        for h in range(self.H):
            alpha = st['U'][:, h, :]
            dp    = alpha * st['dp_base']

            pos, vel, quat, omega = nominal_step_batch(
                pos, vel, quat, omega, st['u_hover'], self.dt,
                device=dev, arm=arm, rdir=rdir, I=I
            )
            vel = vel + st['res'] * self.dt

            e_ref = pos - st['pos_des'][h]
            st['costs'].add_(self.Q_pos  * (e_ref**2).sum(dim=1))
            st['costs'].add_(self.Q_vel  * (vel**2).sum(dim=1))
            st['costs'].add_(self.R_ctrl * (alpha**2).sum(dim=1))

    def _setup_graph(self):
        self._allocate_static_tensors()

        self._cached_arm, self._cached_rdir, self._cached_I =             _get_tensors(self.device)

        # warmup
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                self._rollout_static()
        torch.cuda.current_stream().wait_stream(s)

        # capture graph
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._rollout_static()

        self._graph_ready = True
        # print(f"[CUDA Graph] has captured rollout graph (K={self.K}, H={self.H})")

    def _rollout_graph(self, state_gpu, U_sampled, pos_des_gpu,
                       vel_des_gpu, delta_p_calc, res_const):

        st = self._st

        st['pos'].copy_(state_gpu['x_gpu'].expand(self.K, -1))
        st['vel'].copy_(state_gpu['v_gpu'].expand(self.K, -1))
        st['quat'].copy_(state_gpu['q_gpu'].expand(self.K, -1))
        st['omega'].copy_(state_gpu['w_gpu'].expand(self.K, -1))
        st['U'].copy_(U_sampled)
        st['pos_des'].copy_(pos_des_gpu)
        st['dp_base'].copy_(torch.as_tensor(delta_p_calc,
                            dtype=torch.float32, device=self.device))
        st['res'].copy_(res_const)

        self._graph.replay()

        return st['costs']

    def _rollout_cpu(self, state_gpu, U_sampled, pos_des_gpu,
                     vel_des_gpu, delta_p_calc, res_const):
        K, dev = self.K, self.device
        pos   = state_gpu['x_gpu'].expand(K,-1).clone()
        vel   = state_gpu['v_gpu'].expand(K,-1).clone()
        quat  = state_gpu['q_gpu'].expand(K,-1).clone()
        omega = state_gpu['w_gpu'].expand(K,-1).clone()
        u_hover = torch.full((K,4), HOVER_OMEGA, device=dev)
        dp_base = torch.as_tensor(delta_p_calc, dtype=torch.float32, device=dev)
        costs   = torch.zeros(K, device=dev)
        for h in range(self.H):
            alpha = U_sampled[:,h,:]
            dp    = alpha * dp_base
            pos,vel,quat,omega = nominal_step_batch(
                pos,vel,quat,omega,u_hover,self.dt,device=dev)
            vel = vel + res_const * self.dt
            e_ref  = pos - pos_des_gpu[h]
            costs += self.Q_pos  * (e_ref**2).sum(dim=1)
            costs += self.Q_vel  * (vel**2).sum(dim=1)
            costs += self.R_ctrl * (alpha**2).sum(dim=1)
        return costs

    def update(self, state, pos_des_seq, vel_des_seq,
               motor_speeds_est, residual_raw):
        t0  = time.perf_counter()
        dev = self.device

        delta_p_calc = self.update_ema(residual_raw)

        state_gpu = {
            'x_gpu': torch.as_tensor(state['x'], dtype=torch.float32, device=dev).unsqueeze(0),
            'v_gpu': torch.as_tensor(state['v'], dtype=torch.float32, device=dev).unsqueeze(0),
            'q_gpu': torch.as_tensor(state['q'], dtype=torch.float32, device=dev).unsqueeze(0),
            'w_gpu': torch.as_tensor(state['w'], dtype=torch.float32, device=dev).unsqueeze(0),
        }

        pos_des_gpu = torch.as_tensor(pos_des_seq, dtype=torch.float32, device=dev)
        vel_des_gpu = torch.as_tensor(vel_des_seq, dtype=torch.float32, device=dev)
        noise     = torch.randn(self.K, self.H, 3,
                                dtype=torch.float32, device=dev) * self.sigma
        noise[0]  = 0.0
        U_sampled = torch.clamp(self.U.unsqueeze(0) + noise, 0.0, self.alpha_max)
        

        if self.use_pinn:
            vel_ref_0 = vel_des_gpu[0:1].expand(self.K, -1)
            u_hover   = torch.full((self.K, 4), HOVER_OMEGA, device=dev)
            quat_now  = state_gpu['q_gpu'].expand(self.K, -1)
            omg_now   = state_gpu['w_gpu'].expand(self.K, -1)
            res_const = pinn_batch_refvel(
                self.model, self.normalizer,
                vel_ref_0, quat_now, omg_now, u_hover,
                self.wind_vec, device=dev
            )
        else:
            res_const = torch.zeros(self.K, 3, device=dev)

        # Rollout（CUDA Graph or CPU fallback）
        if dev.type == 'cuda':
            if not self._graph_ready:
                self._setup_graph()
            costs = self._rollout_graph(
                state_gpu, U_sampled, pos_des_gpu,
                vel_des_gpu, delta_p_calc, res_const
            )
        else:
            costs = self._rollout_cpu(
                state_gpu, U_sampled, pos_des_gpu,
                vel_des_gpu, delta_p_calc, res_const
            )

        # calculate weight
        beta    = costs.min()
        weights = torch.exp(-(costs - beta) / self.lam)
        weights = weights / (weights.sum() + 1e-8)

        U_new = torch.einsum('k,khd->hd', weights, U_sampled)
        U_new = torch.clamp(U_new, 0.0, self.alpha_max)

        self.U[:-1] = U_new[1:]
        self.U[-1]  = U_new[-1]

        t1    = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0
        self._compute_times.append(dt_ms)

        return U_new[0].cpu().numpy(), delta_p_calc, dt_ms

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
            'max_hz':  float(hz.max()),
        }


# ── closed loop simulation ─────────────────────────────────────────────────────────

def run_episode(model, normalizer, wind_vec, trajectory_fn,
                sim_time=15.0, dt=0.01,
                use_pinn=True, K=1000, H=20,
                alpha_max=1.2, ema_alpha=EMA_ALPHA,
                dp_calc_max=DP_CALC_MAX,
                verbose=True):

    trajectory = trajectory_fn()
    start_pos  = trajectory.update(0.0)['x'].copy()

    state = {
        'x':            start_pos,
        'v':            np.zeros(3),
        'q':            np.array([0., 0., 0., 1.]),
        'w':            np.zeros(3),
        'wind':         wind_vec.copy(),
        'rotor_speeds': np.ones(4) * HOVER_OMEGA,
    }

    vehicle    = Multirotor(quad_params, state)
    controller = SE3Control(quad_params)
    mppi       = MPPIController(
        model, normalizer, wind_vec,
        K=K, H=H, dt=dt,
        alpha_max=alpha_max, ema_alpha=ema_alpha,
        dp_calc_max=dp_calc_max,
        use_pinn=use_pinn
    )

    N = int(sim_time / dt)

    # generate whole traj in advance to avoid for cycle in python
    _traj = trajectory_fn()
    all_refs = [_traj.update(i * dt) for i in range(N + H + 1)]
    all_pos  = np.array([r['x']     for r in all_refs], dtype=np.float32)
    all_vel  = np.array([r['x_dot'] for r in all_refs], dtype=np.float32)

    times     = np.zeros(N)
    positions = np.zeros((N, 3))
    des_pos   = np.zeros((N, 3))
    errors    = np.zeros(N)
    comp_ms   = np.zeros(N)
    alphas    = np.zeros((N, 3))

    for i in range(N):
        t = i * dt
        state['wind'] = wind_vec.copy()

        pos_des_seq = all_pos[i:i+H]
        vel_des_seq = all_vel[i:i+H]
        vel_des     = all_vel[i]

        motor_speeds_est = np.array(
            state.get('rotor_speeds', np.ones(4)*HOVER_OMEGA)
        )

        if use_pinn:
            residual_raw = pinn_predict(
                model, normalizer, state, motor_speeds_est,
                wind_vec, vel_ref=vel_des
            )
        else:
            residual_raw = np.zeros(3)

        alpha, delta_p_calc, dt_ms = mppi.update(
            state, pos_des_seq, vel_des_seq,
            motor_speeds_est, residual_raw
        )
        delta_p = alpha * delta_p_calc

        comp_ms[i] = dt_ms
        alphas[i]  = alpha

        flat_modified = dict(all_refs[i])
        flat_modified['x']     = all_pos[i] + delta_p
        flat_modified['x_dot'] = all_vel[i]

        cmd   = controller.update(t, state, flat_modified)
        state = vehicle.step(state, cmd, dt)

        times[i]     = t
        positions[i] = state['x']
        des_pos[i]   = all_pos[i]
        errors[i]    = np.linalg.norm(positions[i] - des_pos[i])

        if verbose and i % 50 == 0:
            hz = 1000.0 / dt_ms if dt_ms > 0 else 0
            print(f"  t={t:.1f}s  err={errors[i]:.4f}m  "
                  f"α=[{alpha[0]:.2f},{alpha[1]:.2f},{alpha[2]:.2f}]"
                  f"  {dt_ms:.1f}ms ({hz:.0f}Hz)", end='\r')

    if verbose:
        print()

    return {
        'times':      times,
        'positions':  positions,
        'des_pos':    des_pos,
        'errors':     errors,
        'comp_ms':    comp_ms,
        'alphas':     alphas,
        'mean_err':   errors[100:].mean(),
        'mean_alpha': alphas[100:].mean(axis=0),
        'freq_stats': mppi.compute_frequency_stats(),
    }


def load_pinn_model():
    ckpt       = torch.load(os.path.join(CKPT_DIR, 'best_model.pt'),
                            weights_only=False)
    normalizer = Normalizer()
    normalizer.load(os.path.join(CKPT_DIR, 'normalizer.pt'))
    model = ResidualPINN(input_dim=17)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"load PINN(epoch {ckpt['epoch']})")
    print(f"  Val RMSE : {ckpt['val_rmse'].round(3)}")
    print(f"  OOD RMSE : {ckpt['ood_rmse'].round(3)}")
    return model, normalizer


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    model, normalizer = load_pinn_model()
    print(f"\nSE3 improvement: kp={KP_POS.tolist()}, kd={KD_POS.tolist()}")
    print(f"EMA={EMA_ALPHA}  DP_CALC_MAX={DP_CALC_MAX}m")
    print(f"MPPI: K=896, H=15(RK4 + CUDA Graph)")

    print("\n── control frequency test (3 sec)──")
    _r = run_episode(model, normalizer, np.zeros(3),
                     lambda: HoverTraj(x0=np.array([0.,0.,1.5])),
                     sim_time=3.0, K=896, H=15,
                     use_pinn=True, verbose=False)
    fs = _r['freq_stats']
    print(f"  mean: {fs['mean_hz']:.1f}Hz  Lowest: {fs['min_hz']:.1f}Hz")
    print(f"  {'✓ statisfy real time demand' if fs['mean_hz'] > 50 else '✗ too slow, lower K'}")

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
        (2.0,  'train'),
        (4.0,  'train'),
        (6.0,  'train'),
        (8.0,  'train'),
        (10.0, 'OOD'),
        (12.0, 'OOD'),
    ]

    all_results = {}
    for traj_name, traj_fn in trajectories.items():
        print(f"\n{'='*72}")
        print(f"traj: {traj_name}")
        print(f"{'='*72}")
        print(f"\n{'Wind':>6} | {'Split':>5} | "
              f"{'Nominal':>10} | {'PINN-MPPI':>10} | "
              f"{'improvement':>8} | {'α mean value[x,y,z]':>22} | {'Hz':>6}")
        print("-"*82)

        results = {}
        for wind_speed, split in test_configs:
            wind_vec = np.array([wind_speed, 0., 0.])
            r_nom  = run_episode(model, normalizer, wind_vec,
                                 traj_fn, use_pinn=False,
                                 K=896, H=15, verbose=False)
            r_pinn = run_episode(model, normalizer, wind_vec,
                                 traj_fn, use_pinn=True,
                                 K=896, H=15, verbose=False)
            improv     = (r_nom['mean_err'] - r_pinn['mean_err']) \
                         / (r_nom['mean_err'] + 1e-9) * 100
            hz         = r_pinn['freq_stats']['mean_hz']
            ma         = r_pinn['mean_alpha']
            print(f"{wind_speed:6.0f} | {split:>5} | "
                  f"{r_nom['mean_err']:10.4f} | "
                  f"{r_pinn['mean_err']:10.4f} | "
                  f"{improv:+7.1f}% | "
                  f"[{ma[0]:.2f},{ma[1]:.2f},{ma[2]:.2f}] | "
                  f"{hz:6.1f}")
            results[wind_speed] = {'nom': r_nom, 'pinn': r_pinn}
        all_results[traj_name] = results

    np.save(os.path.join(RESULT_DIR, 'mppi_results.npy'),
            all_results, allow_pickle=True)

    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle('PINN-MPPI（K=896, H=15)', fontsize=10)

        for row, (traj_name, results) in enumerate(all_results.items()):
            ws_list = sorted(results.keys())
            nom_e   = [results[ws]['nom']['mean_err']  for ws in ws_list]
            pinn_e  = [results[ws]['pinn']['mean_err'] for ws in ws_list]
            x = np.arange(len(ws_list))

            ax = axes[row, 0]
            ax.bar(x-0.2, nom_e,  0.4, label='Nominal', color='steelblue')
            ax.bar(x+0.2, pinn_e, 0.4, label='PINN-MPPI', color='darkorange')
            ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5)
            ax.set_xticks(x); ax.set_xticklabels([f'{ws}' for ws in ws_list])
            ax.set_xlabel('Wind (m/s)'); ax.set_ylabel('mean error (m)')
            ax.set_title(f'{traj_name}'); ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

            ax = axes[row, 1]
            ws = 8.0
            ax.plot(results[ws]['nom']['times'],
                    results[ws]['nom']['errors'], 'b-', label='Nominal', alpha=0.8)
            ax.plot(results[ws]['pinn']['times'],
                    results[ws]['pinn']['errors'], 'r--', label='PINN-MPPI', alpha=0.8)
            ax.set_xlabel('time (s)'); ax.set_ylabel('error (m)')
            ax.set_title(f'{traj_name} wind=8m/s')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

            ax = axes[row, 2]
            hz_list = []
            for ws in ws_list:
                hz_list.extend([1000./m for m in
                                 results[ws]['pinn']['comp_ms'] if m > 0])
            if hz_list:
                ax.hist(hz_list, bins=30, color='teal', alpha=0.7, edgecolor='white')
                ax.axvline(x=np.mean(hz_list), color='red', linestyle='--',
                           label=f'mean value={np.mean(hz_list):.0f}Hz')
            ax.set_xlabel('Hz'); ax.set_title(f'{traj_name} control frequency')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(RESULT_DIR, 'mppi_pinn.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\ngraph saved: {path}")
    except Exception as e:
        print(f"jump: {e}")


if __name__ == '__main__':
    main()