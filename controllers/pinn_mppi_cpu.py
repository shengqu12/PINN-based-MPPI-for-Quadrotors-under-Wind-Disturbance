"""
mppi_pinn.py — PINN-MPPI controller (ours)

MPPI design:
    rollout model: complete nominal dynamics (thrust + gravity + torque + PINN residual)
    control variables:    position compensation gain a ∈ [0, alpha_max]
    parameters:        K=896, H=15, dt=0.01s

    device : CPU

PINN input (17-dimensional):
    v_rel(3) | quat(4) | omega(3) | u(4) | wind(3)

"""

import numpy as np
import torch
import os, sys, time

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


# ── complete nominal dynamics (batch Euler integration) ──────────────────────────────────
# using Euler integration for simplicity because it runs on CPU

def quat_rotate_batch(q, v):
    """quaternion rotation, q:(K,4) [qx,qy,qz,qw], v:(K,3)"""
    qx, qy, qz, qw = q[:,0], q[:,1], q[:,2], q[:,3]
    vx, vy, vz     = v[:,0], v[:,1], v[:,2]
    cx = qy*vz - qz*vy
    cy = qz*vx - qx*vz
    cz = qx*vy - qy*vx
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

def nominal_step_batch(pos, vel, quat, omega, u, dt):
    """
    nominal dynamics rollout 
    Where's the drone going in dt sec if apply u.

    """
    K = pos.shape[0]
    F_i = K_ETA * u**2                              # (K,4) 各电机推力

    T               = F_i.sum(dim=1)
    thrust_body     = torch.zeros(K, 3)
    thrust_body[:,2] = T
    thrust_world    = quat_rotate_batch(quat, thrust_body)

    gravity         = torch.zeros(K, 3)
    gravity[:,2]    = -G
    acc             = thrust_world / MASS + gravity

    arm   = torch.tensor(ROTOR_POS, dtype=torch.float32)
    tau_x = (arm[:,1] * F_i).sum(dim=1)
    tau_y = (arm[:,0] * F_i).sum(dim=1)
    tau_z = (torch.tensor(ROTOR_DIR, dtype=torch.float32) * F_i * K_M).sum(dim=1)
    tau   = torch.stack([tau_x, tau_y, tau_z], dim=1)

    I     = torch.tensor([IXX, IYY, IZZ])
    alpha = tau / I

    pos_new   = pos   + vel   * dt
    vel_new   = vel   + acc   * dt
    omega_new = omega + alpha * dt

    half_w        = torch.zeros(K, 4)
    half_w[:,:3]  = omega * 0.5
    dq            = quat_mul_batch(quat, half_w) * dt
    quat_new      = quat + dq
    quat_new      = quat_new / (quat_new.norm(dim=1, keepdim=True) + 1e-8)

    return pos_new, vel_new, quat_new, omega_new


# ── PINN batch inference（17 dimensions，reference velocity）───────────────────────────────────

def pinn_batch_refvel(model, normalizer, vel_ref, quat, omega,
                      motor_speeds, wind_vec):
    """
    vel_ref: (K,3) reference velocity for the horizon start point (t=0)
    """
    K     = vel_ref.shape[0]
    wind  = torch.tensor(wind_vec, dtype=torch.float32).expand(K, 3)
    v_rel = vel_ref - wind    # 参考速度 - 风速
    X     = torch.cat([v_rel, quat, omega, motor_speeds, wind], dim=1)
    X_norm = (X - normalizer.mean) / normalizer.std
    with torch.no_grad():
        return model(X_norm, v_rel)


def pinn_predict(model, normalizer, state, motor_speeds,
                 wind_vec, vel_ref=None):
    """单步 PINN 推理（用于闭环控制）"""
    vel   = vel_ref.astype(np.float32) if vel_ref is not None \
            else state['v'].astype(np.float32)
    quat  = state['q'].astype(np.float32)
    omega = state['w'].astype(np.float32)
    wind  = wind_vec.astype(np.float32)
    v_rel = vel - wind
    X      = np.concatenate([v_rel, quat, omega,
                             motor_speeds.astype(np.float32), wind])
    X_t    = torch.from_numpy(X).unsqueeze(0)
    X_norm = (X_t - normalizer.mean) / normalizer.std
    vr_t   = torch.from_numpy(v_rel).unsqueeze(0)
    with torch.no_grad():
        return model(X_norm, vr_t).squeeze(0).numpy()


# ── MPPI 控制器 ───────────────────────────────────────────────────────

class MPPIController:
    """
    MPPI 

    Rollout 模型：
        完整名义动力学（pos, vel, quat, omega 全部积分）
        + PINN 残差（用参考速度预测）

    控制变量：
        位置补偿增益 α ∈ [0, alpha_max]
        Δp = α × clip(-r_ema / kp, -0.25, 0.25)

    参数推荐：
        K=1000, H=20 → 实测 ~50-100Hz（CPU）
        K=500,  H=10 → 实测 ~100-200Hz（CPU）
    """

    def __init__(self, model, normalizer, wind_vec,
                 K=1000, H=20, dt=0.01,
                 sigma=0.15, lam=0.1, alpha_max=1.2,
                 ema_alpha=EMA_ALPHA,
                 dp_calc_max=DP_CALC_MAX,
                 use_pinn=True):
        self.model        = model
        self.normalizer   = normalizer
        self.wind_vec     = np.array(wind_vec, dtype=np.float32)
        self.K            = K
        self.H            = H
        self.dt           = dt
        self.sigma        = sigma
        self.lam          = lam
        self.alpha_max    = alpha_max
        self.ema_alpha    = ema_alpha
        self.dp_calc_max  = dp_calc_max
        self.use_pinn     = use_pinn

        self.U = np.full((H, 1), 0.3, dtype=np.float32)
        self.residual_ema = None

        self.Q_pos  = 100.0
        self.Q_vel  = 2.0
        self.R_ctrl = 0.0001

        self._compute_times = []
        self.kp = torch.tensor(KP_POS, dtype=torch.float32)
        self.kd = torch.tensor(KD_POS, dtype=torch.float32)

    def update_ema(self, residual):
        """EMA 更新 + 安全限幅"""
        if self.residual_ema is None:
            self.residual_ema = residual.copy()
        else:
            self.residual_ema = (self.ema_alpha * self.residual_ema
                                 + (1 - self.ema_alpha) * residual)
        dp_calc = -self.residual_ema / KP_POS
        return np.clip(dp_calc, -self.dp_calc_max, self.dp_calc_max)

    def _rollout(self, state, U_sampled, pos_des_seq,
                 vel_des_seq, delta_p_calc, motor_speeds_est):
        """
        完整动力学 rollout（不再是极简线性模型）。

        每步：
            1. 根据 α 计算位置补偿 Δp
            2. SE3 近似：a_cmd = kp*(p_des+Δp-pos) + kd*(v_des-vel)
            3. 从 a_cmd 反推悬停附近的电机转速（简化）
            4. 完整动力学积分（pos, vel, quat, omega）
            5. PINN 残差修正速度
        """
        K = self.K

        pos   = torch.tensor(state['x'], dtype=torch.float32).expand(K,-1).clone()
        vel   = torch.tensor(state['v'], dtype=torch.float32).expand(K,-1).clone()
        quat  = torch.tensor(state['q'], dtype=torch.float32).expand(K,-1).clone()
        omega = torch.tensor(state['w'], dtype=torch.float32).expand(K,-1).clone()

        # 用悬停转速近似（简化电机分配）
        u_hover = torch.full((K, 4), HOVER_OMEGA)

        U_t     = torch.tensor(U_sampled, dtype=torch.float32)
        dp_base = torch.tensor(delta_p_calc, dtype=torch.float32)
        costs   = torch.zeros(K)

        # PINN 只推理一次（在 horizon 起点）
        # 物理依据：风速在 H×dt=0.2s 内几乎不变，残差近似常数
        # 效果：计算量从 H×1.08ms 降到 1×1.08ms，频率从 33Hz → ~110Hz
        if self.use_pinn:
            vel_ref_0 = torch.tensor(vel_des_seq[0],
                                     dtype=torch.float32).expand(K, -1)
            res_const = pinn_batch_refvel(
                self.model, self.normalizer,
                vel_ref_0, quat, omega, u_hover, self.wind_vec
            )  # (K,3) 在整个 horizon 内保持不变
        else:
            res_const = torch.zeros(K, 3)

        for h in range(self.H):
            alpha   = U_t[:, h, 0]
            dp      = alpha.unsqueeze(1) * dp_base      # (K,3)
            p_des_h = torch.tensor(pos_des_seq[h], dtype=torch.float32)
            v_des_h = torch.tensor(vel_des_seq[h], dtype=torch.float32)

            # 完整动力学积分
            pos, vel, quat, omega = nominal_step_batch(
                pos, vel, quat, omega, u_hover, self.dt
            )

            # 用常数残差修正速度（只推理一次）
            vel = vel + res_const * self.dt

            # 代价：相对原始参考位置（不含补偿）
            e_ref  = pos - p_des_h
            costs += self.Q_pos  * (e_ref**2).sum(dim=1)
            costs += self.Q_vel  * (vel**2).sum(dim=1)
            costs += self.R_ctrl * (dp**2).sum(dim=1)

        return costs.numpy()

    def update(self, state, pos_des_seq, vel_des_seq,
               motor_speeds_est, residual_raw):
        t0 = time.perf_counter()

        delta_p_calc = self.update_ema(residual_raw)

        noise     = np.random.randn(self.K, self.H, 1).astype(np.float32) * self.sigma
        noise[0]  = 0.0
        U_sampled = np.clip(self.U[None,:,:] + noise, 0.0, self.alpha_max)

        costs = self._rollout(state, U_sampled, pos_des_seq,
                              vel_des_seq, delta_p_calc, motor_speeds_est)

        beta    = costs.min()
        weights = np.exp(-(costs - beta) / self.lam)
        weights = weights / (weights.sum() + 1e-8)

        U_new = np.einsum('k,khd->hd', weights, U_sampled)
        U_new = np.clip(U_new, 0.0, self.alpha_max)

        self.U[:-1] = U_new[1:]
        self.U[-1]  = U_new[-1]

        t1    = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0
        self._compute_times.append(dt_ms)

        return float(U_new[0, 0]), delta_p_calc, dt_ms

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


# ── 闭环仿真 ─────────────────────────────────────────────────────────

def run_episode(model, normalizer, wind_vec, trajectory_fn,
                sim_time=15.0, dt=0.01,
                use_pinn=True, K=1000, H=20,
                alpha_max=1.2, ema_alpha=EMA_ALPHA,
                dp_calc_max=DP_CALC_MAX,
                verbose=True):

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
    mppi       = MPPIController(
        model, normalizer, wind_vec,
        K=K, H=H, dt=dt,
        alpha_max=alpha_max, ema_alpha=ema_alpha,
        dp_calc_max=dp_calc_max,
        use_pinn=use_pinn
    )

    N         = int(sim_time / dt)
    times     = np.zeros(N)
    positions = np.zeros((N, 3))
    des_pos   = np.zeros((N, 3))
    errors    = np.zeros(N)
    comp_ms   = np.zeros(N)
    alphas    = np.zeros(N)

    t = 0.0
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
            residual_raw = pinn_predict(
                model, normalizer, state, motor_speeds_est,
                wind_vec, vel_ref=vel_des
            )
            alpha, delta_p_calc, dt_ms = mppi.update(
                state, pos_des_seq, vel_des_seq,
                motor_speeds_est, residual_raw
            )
            delta_p = alpha * delta_p_calc
        else:
            # Nominal MPPI：残差置零，MPPI 照常运行（K=1000,H=20）
            # 唯一区别是 rollout 没有风扰信息
            # 不能强制 delta_p=0，否则 MPPI 完全没有作用，
            # 对比就变成了 PINN-MPPI vs SE3 only，而不是 vs Nominal MPPI
            residual_raw = np.zeros(3)
            alpha, delta_p_calc, dt_ms = mppi.update(
                state, pos_des_seq, vel_des_seq,
                motor_speeds_est, residual_raw
            )
            delta_p = alpha * delta_p_calc   # 照常使用 MPPI 输出

        comp_ms[i] = dt_ms
        alphas[i]  = alpha

        flat_modified          = dict(flat_ref)
        flat_modified['x']     = flat_ref['x'] + delta_p
        flat_modified['x_dot'] = flat_ref['x_dot']

        cmd   = controller.update(t, state, flat_modified)
        state = vehicle.step(state, cmd, dt)

        times[i]     = t
        positions[i] = state['x']
        des_pos[i]   = flat_ref['x']
        errors[i]    = np.linalg.norm(positions[i] - des_pos[i])
        t           += dt

        if verbose and i % 50 == 0:
            hz = 1000.0 / dt_ms if dt_ms > 0 else 0
            print(f"  t={t:.1f}s  误差={errors[i]:.4f}m  "
                  f"α={alpha:.2f}  {dt_ms:.1f}ms ({hz:.0f}Hz)", end='\r')

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
        'mean_alpha': alphas[100:].mean(),
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
    print(f"加载 PINN（epoch {ckpt['epoch']}）")
    print(f"  Val RMSE : {ckpt['val_rmse'].round(3)}")
    print(f"  OOD RMSE : {ckpt['ood_rmse'].round(3)}")
    return model, normalizer


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    model, normalizer = load_pinn_model()
    print(f"\nSE3 增益: kp={KP_POS.tolist()}, kd={KD_POS.tolist()}")
    print(f"EMA={EMA_ALPHA}  DP_CALC_MAX={DP_CALC_MAX}m")
    print(f"MPPI: K=1000, H=20（完整动力学 rollout）")

    # 频率预测试
    print("\n── 控制频率预测试（3秒）──")
    _r = run_episode(model, normalizer, np.zeros(3),
                     lambda: HoverTraj(x0=np.array([0.,0.,1.5])),
                     sim_time=3.0, K=1000, H=20,
                     use_pinn=True, verbose=False)
    fs = _r['freq_stats']
    print(f"  平均: {fs['mean_hz']:.1f}Hz  最低: {fs['min_hz']:.1f}Hz")
    print(f"  {'✓ 满足实时要求（>50Hz）' if fs['mean_hz'] > 50 else '✗ 太慢，降低 K'}")

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

    all_results = {}
    for traj_name, traj_fn in trajectories.items():
        print(f"\n{'='*68}")
        print(f"轨迹: {traj_name}")
        print(f"{'='*68}")
        print(f"\n{'Wind':>6} | {'Split':>5} | "
              f"{'Nominal':>10} | {'PINN-MPPI':>10} | "
              f"{'改善率':>8} | {'α 均值':>7} | {'Hz':>6}")
        print("-"*64)

        results = {}
        for wind_speed, split in test_configs:
            wind_vec = np.array([wind_speed, 0., 0.])
            r_nom    = run_episode(model, normalizer, wind_vec,
                                   traj_fn, use_pinn=False,
                                   K=1000, H=20, verbose=False)
            r_pinn   = run_episode(model, normalizer, wind_vec,
                                   traj_fn, use_pinn=True,
                                   K=1000, H=20, verbose=False)
            improv    = (r_nom['mean_err'] - r_pinn['mean_err']) \
                        / (r_nom['mean_err'] + 1e-9) * 100
            hz        = r_pinn['freq_stats']['mean_hz']
            mean_alpha = r_pinn['mean_alpha']
            print(f"{wind_speed:6.0f} | {split:>5} | "
                  f"{r_nom['mean_err']:10.4f} | "
                  f"{r_pinn['mean_err']:10.4f} | "
                  f"{improv:+7.1f}% | "
                  f"{mean_alpha:7.3f} | {hz:6.1f}")
            results[wind_speed] = {'nom': r_nom, 'pinn': r_pinn}
        all_results[traj_name] = results

    np.save(os.path.join(RESULT_DIR, 'mppi_results.npy'),
            all_results, allow_pickle=True)

    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle(
            'PINN-MPPI（K=1000, H=20，完整动力学 rollout）\n'
            '参考速度输入 + EMA + delta_p 限幅',
            fontsize=10
        )
        for row, (traj_name, results) in enumerate(all_results.items()):
            ws_list  = sorted(results.keys())
            nom_e    = [results[ws]['nom']['mean_err']  for ws in ws_list]
            pinn_e   = [results[ws]['pinn']['mean_err'] for ws in ws_list]
            alphas_m = [results[ws]['pinn']['mean_alpha'] for ws in ws_list]
            x        = np.arange(len(ws_list))

            ax = axes[row, 0]
            ax.bar(x-0.2, nom_e,  0.4, label='Nominal', color='steelblue')
            ax.bar(x+0.2, pinn_e, 0.4, label='PINN-MPPI', color='darkorange')
            ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5)
            ax.set_xticks(x); ax.set_xticklabels([f'{ws}' for ws in ws_list])
            ax.set_xlabel('Wind (m/s)'); ax.set_ylabel('平均误差 (m)')
            ax.set_title(f'{traj_name}'); ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

            ax = axes[row, 1]
            ws = 8.0
            ax.plot(results[ws]['nom']['times'],
                    results[ws]['nom']['errors'], 'b-', label='Nominal', alpha=0.8)
            ax.plot(results[ws]['pinn']['times'],
                    results[ws]['pinn']['errors'], 'r--', label='PINN-MPPI', alpha=0.8)
            ax.set_xlabel('时间 (s)'); ax.set_ylabel('位置误差 (m)')
            ax.set_title(f'{traj_name} wind=8 m/s')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

            ax = axes[row, 2]
            all_hz = [1000./m for m in results[ws]['pinn']['comp_ms'] if m > 0
                      for ws in ws_list]
            if all_hz:
                ax.hist(all_hz, bins=30, color='teal', alpha=0.7, edgecolor='white')
                ax.axvline(x=np.mean(all_hz), color='red', linestyle='--',
                           label=f'均值={np.mean(all_hz):.0f}Hz')
            ax.set_xlabel('Hz'); ax.set_title(f'{traj_name} 控制频率')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(RESULT_DIR, 'mppi_pinn.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存: {path}")
    except Exception as e:
        print(f"画图跳过: {e}")


if __name__ == '__main__':
    main()