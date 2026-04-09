"""
mppi_pinn_mujoco.py — MuJoCo PINN-MPPI 完整可视化


    python simulation/mppi_pinn_mujoco.py --render --traj circle --wind 8

    python simulation/mppi_pinn_mujoco.py --render --skydio --traj circle --wind 8

    python simulation/mppi_pinn_mujoco.py --render --skydio --traj circle --wind 8 --no_pinn
"""

import numpy as np
import torch
import os, sys, time, argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.pinn import ResidualPINN
from training.dataset import Normalizer
from simulation.quadrotor_env import QuadrotorMuJoCoEnv, HOVER_OMEGA
from controllers.pinn_mppi_GPU import (MPPIController, pinn_predict,
                                        nominal_step_batch, DEVICE)

from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.trajectories.hover_traj import HoverTraj

CKPT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def load_pinn():
    ckpt = torch.load(os.path.join(CKPT_DIR, 'best_model.pt'),
                            weights_only=False)
    normalizer = Normalizer()
    normalizer.load(os.path.join(CKPT_DIR, 'normalizer.pt'))
    model = ResidualPINN(input_dim=17)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"PINN has loaded (epoch {ckpt['epoch']})")
    return model, normalizer


# ── Geometry tool ────────────────────────────────────────────────────────

def _sphere(scn, pos, r, rgba):
    import mujoco
    if scn.ngeom >= scn.maxgeom: return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([r, 0., 0.]),
        np.array(pos, dtype=np.float64),
        np.eye(3).flatten(),
        np.array(rgba, dtype=np.float32))
    scn.ngeom += 1


def _capsule(scn, p1, p2, w, rgba):
    import mujoco
    if scn.ngeom >= scn.maxgeom: return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3), np.zeros(3), np.eye(3).flatten(),
        np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE, float(w),
        np.array(p1, dtype=np.float64),
        np.array(p2, dtype=np.float64))
    scn.ngeom += 1


def err_to_rgba(err, max_err=1.2, alpha=0.85):
    """
    green = less error
    red = bigger error 
    """
    t = float(np.clip(err / max_err, 0., 1.))
    return [t, 1. - t, 0.05, alpha]


def cost_to_rgba(cost, min_c, max_c, alpha=0.30):
    t = float(np.clip((cost - min_c) / (max_c - min_c + 1e-8), 0., 1.))
    return [t, 1. - t, 0., alpha]


# ── MPPI rollout visualization ───────────────────────────────────────────────

def get_rollout_positions(mppi, state, pos_des_gpu, n_show=20):
    dev = mppi.device
    K, H = mppi.K, mppi.H

    with torch.no_grad():
        noise     = torch.randn(K, H, 3, device=dev) * mppi.sigma
        noise[0]  = 0.0
        U_sampled = torch.clamp(mppi.U.unsqueeze(0) + noise, 0., mppi.alpha_max)

        pos   = torch.tensor(state['x'], dtype=torch.float32, device=dev).expand(K,-1).clone()
        vel   = torch.tensor(state['v'], dtype=torch.float32, device=dev).expand(K,-1).clone()
        quat  = torch.tensor(state['q'], dtype=torch.float32, device=dev).expand(K,-1).clone()
        omega = torch.tensor(state['w'], dtype=torch.float32, device=dev).expand(K,-1).clone()
        u_hov = torch.full((K, 4), HOVER_OMEGA, device=dev)

        traj_pos = []
        costs    = torch.zeros(K, device=dev)
        for h in range(H):
            pos, vel, quat, omega = nominal_step_batch(
                pos, vel, quat, omega, u_hov, mppi.dt, device=dev)
            traj_pos.append(pos.cpu().numpy())
            if h < pos_des_gpu.shape[0]:
                e = pos.cpu().numpy() - pos_des_gpu[h].cpu().numpy()
                costs += torch.tensor((e**2).sum(axis=1), device=dev)

    idx      = np.linspace(0, K-1, n_show, dtype=int)
    traj_arr = np.stack(traj_pos, axis=1)
    return traj_arr[idx], costs.cpu().numpy()[idx]


# ── visualization function ─────────────────────────────────────────────────────

class CameraController:
    """follow the quadrotor"""
    def __init__(self, offset=np.array([3.5, -2.5, 2.0])):
        self.offset   = offset
        self._lookat  = None
        self._azimuth = 135.0
        self._t       = 0
        self.use_auto_cam = True

    def update(self, viewer, drone_pos, dt):
        if not self.use_auto_cam:
            return
        
        target = np.array(drone_pos, dtype=float)
        if self._lookat is None:
            self._lookat = target.copy()
        alpha = 1.0 - np.exp(-dt * 3.0)
        self._lookat += (target - self._lookat) * alpha

        self._azimuth += dt * 4.0
        self._t       += dt

        viewer.cam.lookat[:]  = self._lookat
        viewer.cam.distance   = 5.5
        viewer.cam.elevation  = -22
        viewer.cam.azimuth    = self._azimuth % 360


def update_scene(scn, state, full_ref, cur_ref,
                 history_pos, history_err,
                 des_history,
                 wind_vec, alpha_norm,t,
                 rollout_pos=None, rollout_costs=None):
    """
    update geom defined by viewer.user_scn 

    history_pos: list of (3,) 
    history_err: list of float 
    des_history: list of (3,) desired position
    alpha_norm:  float [0,1] 
    """
    scn.ngeom = 0
    max_err   = 1.2

    # ── 1. reference traj (blue little ball)──────────────────────────────
    step = max(1, len(full_ref) // 70)
    for p in full_ref[::step]:
        _sphere(scn, p, 0.016, [0.0, 0.85, 0.85, 0.30])

    # ── 2. reference current location (yellow ball)──────────────────────────────────
    if len(cur_ref) > 0:
        _sphere(scn, cur_ref[0], 0.016, [1.0, 0.90, 0.0, 0.95])  #(scn, current location, radius, color)

    # ── 3. expected traj history(blue)──────────────────────────────────
    skip = max(1, len(des_history) // 80)
    for j in range(skip, len(des_history), skip):
        _capsule(scn, des_history[j-skip], des_history[j],
                 0.008, [0.3, 0.5, 1.0, 0.50])

    # ── 4. real traj history───────────────────────────
    # Green means less error
    # Red means Big error
    pos_hist = history_pos[-100:] # recording the 100 points before current state
    err_hist = history_err[-100:]
    skip2    = max(1, len(pos_hist) // 60)
    prev_p   = None
    for j in range(0, len(pos_hist), skip2):
        rgba = err_to_rgba(err_hist[j], max_err, alpha=0.85)
        _sphere(scn, pos_hist[j], 0.018, rgba)
        if prev_p is not None:
            _capsule(scn, prev_p, pos_hist[j], 0.012, rgba)
        prev_p = pos_hist[j]

    # # ── 5. α gain ball────────────────────────
    # # The bigger the ball, the bigger it's effort to fight against the wind
    # alpha_r = 0.035 + alpha_norm * 0.055
    # _sphere(scn, state['x'], alpha_r, [1.0, 0.55, 0.0, 0.70])

    # ── 6. wind field────────────────────────
    wind_mag = np.linalg.norm(wind_vec)
    if wind_mag > 0.1:
        drone = np.array(state['x'])
        scale = 0.28 / (wind_mag + 1e-6) 

        flow_speed = wind_mag

        flow_offset = (t*flow_speed)%2.5  
    
        x_range = np.linspace(-10.0,10.0,5)
        z_range = np.linspace(0.0,4.0,3)
        y_range = np.linspace(-8.0,8.0,8)

        for dx in x_range:
            for dy in y_range:
                for dz in z_range:
                    origin = np.array([dx+ flow_offset, dy , dz])
                    tip    = origin + wind_vec * scale
                    _capsule(scn, origin, tip, 0.009,
                            [0.25, 0.55, 1.0, 0.50])
                    _sphere(scn, tip, 0.025, [0.25, 0.55, 1.0, 0.65])

        # # arrow on drone
        # tip_main = drone + wind_vec * scale * 1.5
        # _capsule(scn, drone, tip_main, 0.010, [0.10, 0.40, 1.0, 0.90])
        # _sphere(scn, tip_main, 0.015, [0.10, 0.40, 1.0, 0.95])

    # ── 7. MPPI rollout──────────────────────────────
    if rollout_pos is not None and rollout_costs is not None:
        min_c = rollout_costs.min()
        max_c = rollout_costs.max()
        order = np.argsort(rollout_costs)[::-1]   
        for k in order:
            rgba  = cost_to_rgba(rollout_costs[k], min_c, max_c, alpha=0.32)
            traj  = rollout_pos[k]
            step3 = max(1, len(traj) // 8)
            for h in range(step3, len(traj), step3):
                _capsule(scn, traj[h-step3], traj[h], 0.007, rgba)


# ── simulation ─────────────────────────────────────────────────────────

def run_episode(model, normalizer, wind_vec, trajectory_fn,
                sim_time=15.0, dt=0.01,
                use_pinn=True, K=896, H=15,
                alpha_max=1.2, ema_alpha=0.85,
                render=False, show_rollout=True,
                use_skydio=False, verbose=True):

    env        = QuadrotorMuJoCoEnv(use_skydio=use_skydio)
    trajectory = trajectory_fn()
    start_pos  = trajectory.update(0.0)['x'].copy()
    state      = env.reset(pos=start_pos)

    controller = SE3Control(quad_params)
    mppi       = MPPIController(
        model, normalizer, wind_vec,
        K=K, H=H, dt=dt,
        alpha_max=alpha_max, ema_alpha=ema_alpha,
        use_pinn=use_pinn
    )

    N = int(sim_time / dt)

    # generate traj 
    _traj    = trajectory_fn()
    all_refs = [_traj.update(i * dt) for i in range(N + H + 1)]
    all_pos  = np.array([r['x']     for r in all_refs], dtype=np.float32)
    all_vel  = np.array([r['x_dot'] for r in all_refs], dtype=np.float32)

    # reference traj (static)
    _tf2     = trajectory_fn()
    n_full   = min(N, int(1.0/0.2/dt)*2 + 1)   
    full_ref = np.array([_tf2.update(i*dt)['x'] for i in range(n_full)],
                        dtype=np.float32)

    times     = np.zeros(N)
    positions = np.zeros((N, 3))
    des_pos   = np.zeros((N, 3))
    errors    = np.zeros(N)
    comp_ms   = np.zeros(N)
    alphas    = np.zeros((N, 3))

    def key_callback(keycode):
        # chr(keycode) c: change the view
        if chr(keycode).upper() == 'C':
            cam_ctrl.use_auto_cam = not cam_ctrl.use_auto_cam
            status = "Automatically follow" if cam_ctrl.use_auto_cam else "Free Camera"
            print(f"[Camera] Mode Change: {status}")    
    # visualization initialization
    viewer   = None
    cam_ctrl = CameraController()

    if render:
        try:
            import mujoco
            import mujoco.viewer as mj_viewer
            viewer = mj_viewer.launch_passive(env.model, env.data,key_callback=key_callback)
            viewer.cam.lookat[:] = list(start_pos)
            viewer.cam.distance  = 5.5
            viewer.cam.azimuth   = 135
            viewer.cam.elevation = -22
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT]        = False
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
            print("[Viewer] Started Successfully")
            print("  Blue line=Expected Trajectory  Green/Red Line = Real Traj")
            print("Push c to change camera")
            if show_rollout:
                print("  Green= Good MPPI Sampling  Red line = Bad MPPI Sampling")
        except Exception as e:
            print(f"[Viewer] Start failed: {e}")
            render = False

    history_pos = []
    history_err = []
    des_history = []
    rollout_pos_vis   = None
    rollout_costs_vis = None

    for i in range(N):
        t = i * dt

        pos_des_seq = all_pos[i:i+H]
        vel_des_seq = all_vel[i:i+H]
        vel_des     = all_vel[i]

        motor_speeds_est = np.array(
            state.get('rotor_speeds', np.ones(4)*HOVER_OMEGA))

        residual_raw = pinn_predict(
            model, normalizer, state, motor_speeds_est,
            wind_vec, vel_ref=vel_des
        ) if use_pinn else np.zeros(3)

        alpha, delta_p_calc, dt_ms = mppi.update(
            state, pos_des_seq, vel_des_seq,
            motor_speeds_est, residual_raw)
        delta_p    = alpha * delta_p_calc
        comp_ms[i] = dt_ms
        alphas[i]  = alpha

        flat_modified          = dict(all_refs[i])
        flat_modified['x']     = all_pos[i] + delta_p
        flat_modified['x_dot'] = all_vel[i]

        cmd          = controller.update(t, state, flat_modified)
        motor_speeds = cmd['cmd_motor_speeds'].copy()
        state        = env.step(motor_speeds, wind_vec=wind_vec, dt=dt)

        times[i]     = t
        positions[i] = state['x']
        des_pos[i]   = all_pos[i]
        errors[i]    = np.linalg.norm(positions[i] - des_pos[i])

        history_pos.append(state['x'].copy())
        history_err.append(float(errors[i]))
        des_history.append(all_pos[i].copy())

        # update rollout visualization
        if render and show_rollout and i % 8 == 0:
            try:
                pos_des_gpu = torch.as_tensor(
                    pos_des_seq, dtype=torch.float32, device=DEVICE)
                rollout_pos_vis, rollout_costs_vis = get_rollout_positions(
                    mppi, state, pos_des_gpu, n_show=200)
            except Exception:
                pass

        if render and viewer is not None:
            if not viewer.is_running():
                print("\n[Viewer] Window is closed")
                break

            alpha_norm = float(np.linalg.norm(alpha)) / (np.sqrt(3) * alpha_max)

            with viewer.lock():
                update_scene(
                    viewer.user_scn, state,
                    full_ref,
                    all_pos[i:i+min(H, N-i)],
                    history_pos, history_err,
                    des_history, wind_vec,
                    alpha_norm,
                    t,
                    rollout_pos_vis, rollout_costs_vis
                )
                cam_ctrl.update(viewer, state['x'], dt)

            viewer.sync()

        if verbose and i % 20 == 0:
            hz = 1000.0 / dt_ms if dt_ms > 0 else 0
            print(f"  t={t:5.1f}s | err={errors[i]:.3f}m | "
                  f"wind={np.linalg.norm(wind_vec):.1f}m/s | "
                  f"α=[{alpha[0]:.2f},{alpha[1]:.2f},{alpha[2]:.2f}] | "
                  f"{hz:.0f}Hz", end='\r')

    if verbose: print()
    if viewer is not None: viewer.close()

    return {
        'times':      times,
        'positions':  positions,
        'des_pos':    des_pos,
        'errors':     errors,
        'comp_ms':    comp_ms,
        'alphas':     alphas,
        'mean_err':   errors[100:].mean(),
        'mean_alpha': alphas[100:].mean(axis=0),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render',     action='store_true')
    parser.add_argument('--skydio',     action='store_true',
                        help='using skydio model(same pramater with humming bird)')
    parser.add_argument('--wind',       type=float, default=8.0)
    parser.add_argument('--traj',       type=str,   default='circle',
                        choices=['hover', 'circle'])
    parser.add_argument('--sim_time',   type=float, default=20.0)
    parser.add_argument('--no_pinn',    action='store_true')
    parser.add_argument('--no_rollout', action='store_true')
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    model, normalizer = load_pinn()

    trajs = {
        'hover':  lambda: HoverTraj(x0=np.array([0., 0., 1.5])),
        'circle': lambda: ThreeDCircularTraj(
            center=np.array([0., 0., 1.5]),
            radius=np.array([1.5, 1.5, 0.]),
            freq=np.array([0.2, 0.2, 0.]),
        ),
    }

    if args.render:
        use_pinn = not args.no_pinn
        label    = 'PINN-MPPI' if use_pinn else 'Nominal MPPI'
        model_l  = 'Skydio X2 (visual)' if args.skydio else 'Hummingbird'
        print(f"\nVisualization:{label} | {args.traj} | wind={args.wind}m/s | {model_l}")
        wind_vec = np.array([args.wind, 0., 0.])
        r = run_episode(
            model, normalizer, wind_vec, trajs[args.traj],
            sim_time=args.sim_time, K=896, H=15,
            use_pinn=use_pinn,
            show_rollout=not args.no_rollout,
            use_skydio=args.skydio,
            render=True, verbose=True
        )
        print(f"\n Mean Error: {r['mean_err']:.4f}m")
        ma = r['mean_alpha']
        print(f"Average α:   [{ma[0]:.2f},{ma[1]:.2f},{ma[2]:.2f}]")
        return

    test_configs = [
        (0.,  'train'), (2., 'train'), (4.,  'train'),(6., 'train'), (8.,  'train'),
        (10., 'OOD'),   (12., 'OOD'),
    ]
    all_results = {}
    for tname, tfn in trajs.items():
        print(f"\n{'='*68}\ntrajectory: {tname}  [MuJoCo]\n{'='*68}")
        print(f"\n{'Wind':>6}|{'Split':>5}|{'Nominal':>10}|"
              f"{'PINN-MPPI':>10}|{'Improvement':>8}|{'α mean value':>20}")
        print("-"*65)
        results = {}
        for ws, split in test_configs:
            wv    = np.array([ws, 0., 0.])
            r_nom = run_episode(model, normalizer, wv, tfn,
                                K=896, H=15, use_pinn=False,
                                use_skydio=args.skydio, verbose=False)
            r_pin = run_episode(model, normalizer, wv, tfn,
                                K=896, H=15, use_pinn=True,
                                use_skydio=args.skydio, verbose=False)
            imp = (r_nom['mean_err']-r_pin['mean_err'])/(r_nom['mean_err']+1e-9)*100
            ma  = r_pin['mean_alpha']
            print(f"{ws:6.0f}|{split:>5}|{r_nom['mean_err']:10.4f}|"
                  f"{r_pin['mean_err']:10.4f}|{imp:+7.1f}%|"
                  f"[{ma[0]:.2f},{ma[1]:.2f},{ma[2]:.2f}]")
            results[ws] = {'nom': r_nom, 'pinn': r_pin}
        all_results[tname] = results

    np.save(os.path.join(RESULT_DIR,'mppi_mujoco.npy'),
            all_results, allow_pickle=True)
    print("\nResult Saved")


if __name__ == '__main__':
    main()