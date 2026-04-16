"""
mppi_pinn_mujoco.py — MuJoCo PINN-MPPI full visualization

    python simulation/mppi_pinn_mujoco.py --render --traj circle --wind 8

    python simulation/mppi_pinn_mujoco.py --render --skydio --traj circle --wind 8

    python simulation/mppi_pinn_mujoco.py --render --skydio --traj circle --wind 8 --no_pinn

    python simulation/mppi_pinn_mujoco.py --render --obstacle --traj circle --wind 8
"""

import numpy as np
import torch
import os, sys, time, argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from controllers.pinn_mppi_v2 import (
    load_pinn_model, PINNMPPIv2, pinn_infer, HOVER_OMEGA, DEVICE,
)
from controllers.pinn_mppi_obstacle import PINNMPPIObstacle
from simulation.quadrotor_env import QuadrotorMuJoCoEnv

from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.trajectories.hover_traj import HoverTraj

CKPT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

ALPHA_EMA = 0.05
R_EMA_MAX = 6.0   # m/s² clamp — prevents motor saturation

# Default obstacle (matches obstacle_scene.xml)
DEFAULT_OBSTACLES = [
    (np.array([1.9, 0.0, 1.5]), 0.25),   # (center_xyz, radius_m)
]


def load_pinn():
    model, normalizer = load_pinn_model(CKPT_DIR)
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


def _cylinder_obs(scn, cx, cy, radius, height=3.0,
                  rgba_body=(0.85, 0.20, 0.10, 0.70),
                  rgba_margin=(1.0, 0.45, 0.25, 0.18),
                  margin=0.25):
    """Draw cylindrical obstacle: solid body + translucent safety margin."""
    import mujoco
    half_h = height / 2.0
    cz     = half_h
    if scn.ngeom < scn.maxgeom:
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.array([radius, half_h, 0.]),
            np.array([cx, cy, cz], dtype=np.float64),
            np.eye(3).flatten(),
            np.array(rgba_body, dtype=np.float32))
        scn.ngeom += 1
    if scn.ngeom < scn.maxgeom:
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.array([radius + margin, half_h + 0.01, 0.]),
            np.array([cx, cy, cz + 0.01], dtype=np.float64),
            np.eye(3).flatten(),
            np.array(rgba_margin, dtype=np.float32))
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

def get_rollout_positions(mppi, state, ref_seq, r_hat, n_show=30):
    """
    Sample K_vis alpha sequences and roll out the SE3 dynamics for H steps.
    Returns (n_show, H, 3) positions and (n_show,) costs.
    Uses PINNMPPIv2._rk4_step directly — no CUDA Graph overhead.
    """
    dev  = mppi.device
    K_v  = min(n_show * 10, mppi.K)

    with torch.no_grad():
        U_ws  = torch.tensor(mppi.alpha_seq, dtype=torch.float32, device=dev)
        noise = torch.randn(K_v, mppi.H, 3, device=dev) * mppi.sigma
        alpha = (U_ws.unsqueeze(0) + noise).clamp(mppi.alpha_min, mppi.alpha_max)

        p = torch.tensor(state['x'], dtype=torch.float32,
                         device=dev).unsqueeze(0).expand(K_v, -1).clone()
        v = torch.tensor(state['v'], dtype=torch.float32,
                         device=dev).unsqueeze(0).expand(K_v, -1).clone()
        q = torch.tensor(state.get('q', [0., 0., 0., 1.]),
                         dtype=torch.float32,
                         device=dev).unsqueeze(0).expand(K_v, -1).clone()
        r_t = torch.tensor(r_hat, dtype=torch.float32,
                           device=dev).unsqueeze(0).expand(K_v, -1).contiguous()

        traj_pos = []
        costs    = torch.zeros(K_v, device=dev)

        for h in range(mppi.H):
            p_des  = torch.tensor(ref_seq[h]['x'],
                                  dtype=torch.float32, device=dev).unsqueeze(0)
            v_des  = torch.tensor(ref_seq[h].get('x_dot',  np.zeros(3)),
                                  dtype=torch.float32, device=dev).unsqueeze(0)
            a_traj = torch.tensor(ref_seq[h].get('x_ddot', np.zeros(3)),
                                  dtype=torch.float32, device=dev).unsqueeze(0)

            alpha_h = alpha[:, h, :]
            p, v, q = mppi._rk4_step(p, v, q, alpha_h,
                                      p_des, v_des, a_traj, r_t,
                                      mppi.dt_pred)
            traj_pos.append(p.cpu().numpy())

            ep = p - p_des; ev = v - v_des
            costs += mppi.c_p * (ep * ep).sum(1) + mppi.c_v * (ev * ev).sum(1)

    idx      = np.linspace(0, K_v - 1, n_show, dtype=int)
    traj_arr = np.stack(traj_pos, axis=1)   # (K_v, H, 3)
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
                 wind_vec, alpha_norm, t,
                 rollout_pos=None, rollout_costs=None,
                 obstacles=None):
    """
    update geom defined by viewer.user_scn

    history_pos: list of (3,)
    history_err: list of float
    des_history: list of (3,) desired position
    alpha_norm:  float [0,1]
    obstacles:   list of (center_xyz, radius) — drawn as red cylinders
    """
    scn.ngeom = 0
    max_err   = 1.2

    # ── 1. reference traj (blue little ball)──────────────────────────────
    step = max(1, len(full_ref) // 70)
    for p in full_ref[::step]:
        _sphere(scn, p, 0.016, [0.0, 0.85, 0.85, 0.30])

    # ── 2. reference current location (yellow ball)──────────────────────────────────
    if len(cur_ref) > 0:
        _sphere(scn, cur_ref[0], 0.016, [1.0, 0.90, 0.0, 0.95])

    # ── 3. expected traj history(blue)──────────────────────────────────
    skip = max(1, len(des_history) // 80)
    for j in range(skip, len(des_history), skip):
        _capsule(scn, des_history[j-skip], des_history[j],
                 0.008, [0.3, 0.5, 1.0, 0.50])

    # ── 4. real traj history───────────────────────────
    # Green means less error
    # Red means Big error
    pos_hist = history_pos[-100:]
    err_hist = history_err[-100:]
    skip2    = max(1, len(pos_hist) // 60)
    prev_p   = None
    for j in range(0, len(pos_hist), skip2):
        rgba = err_to_rgba(err_hist[j], max_err, alpha=0.85)
        _sphere(scn, pos_hist[j], 0.018, rgba)
        if prev_p is not None:
            _capsule(scn, prev_p, pos_hist[j], 0.012, rgba)
        prev_p = pos_hist[j]

    # ── 5. wind field────────────────────────
    wind_mag = np.linalg.norm(wind_vec)
    if wind_mag > 0.1:
        scale      = 0.28 / (wind_mag + 1e-6)
        flow_speed = wind_mag
        flow_offset = (t * flow_speed) % 2.5

        x_range = np.linspace(-10.0, 10.0, 5)
        z_range = np.linspace(0.0, 4.0, 3)
        y_range = np.linspace(-8.0, 8.0, 8)

        for dx in x_range:
            for dy in y_range:
                for dz in z_range:
                    origin = np.array([dx + flow_offset, dy, dz])
                    tip    = origin + wind_vec * scale
                    _capsule(scn, origin, tip, 0.009,
                             [0.25, 0.55, 1.0, 0.50])
                    _sphere(scn, tip, 0.025, [0.25, 0.55, 1.0, 0.65])

    # ── 6. MPPI rollout──────────────────────────────
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

    # ── 7. obstacles (red cylinders + safety margin)─────────────────────
    if obstacles:
        for (center, radius) in obstacles:
            _cylinder_obs(scn, center[0], center[1], radius)


# ── simulation ─────────────────────────────────────────────────────────

def run_episode(model, normalizer, wind_vec, trajectory_fn,
                sim_time=15.0, dt=0.01,
                use_pinn=True, K=896, H=15,
                render=False, show_rollout=True,
                use_skydio=False,
                obstacles=None,
                verbose=True):

    env        = QuadrotorMuJoCoEnv(use_skydio=use_skydio)
    trajectory = trajectory_fn()
    start_pos  = trajectory.update(0.0)['x'].copy()
    state      = env.reset(pos=start_pos)
    controller = SE3Control(quad_params)

    # Choose MPPI controller (with or without obstacle cost)
    if obstacles:
        mppi = PINNMPPIObstacle(
            model, normalizer, wind_vec,
            K=K, H=H, dt=dt, use_pinn=use_pinn,
            obstacles=obstacles,
        )
    else:
        mppi = PINNMPPIv2(
            model, normalizer, wind_vec,
            K=K, H=H, dt=dt, use_pinn=use_pinn,
        )

    dev   = mppi.device
    r_ema = np.zeros(3, dtype=np.float32)

    N = int(sim_time / dt)

    # Pre-compute dense reference (for scene rendering)
    _traj    = trajectory_fn()
    all_refs = [_traj.update(i * dt) for i in range(N + H + 1)]
    all_pos  = np.array([r['x'] for r in all_refs], dtype=np.float32)

    # Full reference traj for static scene dots
    _tf2     = trajectory_fn()
    n_full   = min(N, int(1.0 / 0.2 / dt) * 2 + 1)
    full_ref = np.array([_tf2.update(i * dt)['x'] for i in range(n_full)],
                        dtype=np.float32)

    # Dedicated trajectory instance for MPPI ref_seq (avoids creating N*H objects)
    _traj_ref = trajectory_fn()

    times     = np.zeros(N)
    positions = np.zeros((N, 3))
    des_pos   = np.zeros((N, 3))
    errors    = np.zeros(N)
    comp_ms   = np.zeros(N)
    alphas    = np.zeros((N, 3))

    def key_callback(keycode):
        if chr(keycode).upper() == 'C':
            cam_ctrl.use_auto_cam = not cam_ctrl.use_auto_cam
            status = "Automatically follow" if cam_ctrl.use_auto_cam else "Free Camera"
            print(f"[Camera] Mode Change: {status}")

    viewer   = None
    cam_ctrl = CameraController()

    if render:
        try:
            import mujoco
            import mujoco.viewer as mj_viewer
            viewer = mj_viewer.launch_passive(
                env.model, env.data, key_callback=key_callback)
            viewer.cam.lookat[:] = list(start_pos)
            viewer.cam.distance  = 5.5
            viewer.cam.azimuth   = 135
            viewer.cam.elevation = -22
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT]        = False
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
            print("[Viewer] Started Successfully")
            print("  Blue line=Expected Trajectory  Green/Red Line = Real Traj")
            print("  Push C to toggle camera mode")
            if show_rollout:
                print("  Green=Good MPPI Sampling  Red=Bad MPPI Sampling")
            if obstacles:
                print(f"  Obstacle avoidance ON ({len(obstacles)} obstacle(s))")
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
        state['wind'] = wind_vec.copy()

        # MPPI reference horizon (at dt_pred intervals)
        ref_seq = [_traj_ref.update(t + j * mppi.dt_pred)
                   for j in range(H)]

        # PINN wind-residual (velocity-aware + clamp)
        if use_pinn:
            res_np = pinn_infer(mppi.model, mppi.normalizer,
                                state, wind_vec, dev)
            r_ema = ALPHA_EMA * res_np + (1.0 - ALPHA_EMA) * r_ema
            r_ema_clamped = np.clip(r_ema, -R_EMA_MAX, R_EMA_MAX)
        else:
            r_ema_clamped = np.zeros(3, dtype=np.float32)

        # Zero z-axis (SE3 handles z residual cleanly)
        r_ema_xy    = r_ema_clamped.copy()
        r_ema_xy[2] = 0.0

        # MPPI -> optimal alpha
        alpha_opt, dt_ms = mppi.update(state, ref_seq, r_ema_xy)
        comp_ms[i] = dt_ms
        alphas[i]  = alpha_opt

        # SE3Control with PINN x_ddot feedforward
        flat           = dict(ref_seq[0])
        traj_x_ddot    = np.array(ref_seq[0].get('x_ddot', np.zeros(3)),
                                   dtype=np.float32)
        flat['x_ddot'] = traj_x_ddot + (-alpha_opt * r_ema_xy)

        cmd          = controller.update(t, state, flat)
        motor_speeds = cmd['cmd_motor_speeds'].copy()
        state        = env.step(motor_speeds, wind_vec=wind_vec, dt=dt)

        times[i]     = t
        positions[i] = state['x']
        des_pos[i]   = all_pos[i]
        errors[i]    = np.linalg.norm(positions[i] - des_pos[i])

        history_pos.append(state['x'].copy())
        history_err.append(float(errors[i]))
        des_history.append(all_pos[i].copy())

        # Update rollout visualization every 8 steps
        if render and show_rollout and i % 8 == 0:
            try:
                rollout_pos_vis, rollout_costs_vis = get_rollout_positions(
                    mppi, state, ref_seq, r_ema_xy, n_show=30)
            except Exception:
                pass

        if render and viewer is not None:
            if not viewer.is_running():
                print("\n[Viewer] Window is closed")
                break

            alpha_norm = float(np.linalg.norm(alpha_opt)) / (np.sqrt(3) * mppi.alpha_max)

            with viewer.lock():
                update_scene(
                    viewer.user_scn, state,
                    full_ref,
                    all_pos[i:i + min(H, N - i)],
                    history_pos, history_err,
                    des_history, wind_vec,
                    alpha_norm, t,
                    rollout_pos_vis, rollout_costs_vis,
                    obstacles=obstacles,
                )
                cam_ctrl.update(viewer, state['x'], dt)

            viewer.sync()

        if verbose and i % 20 == 0:
            hz = 1000.0 / dt_ms if dt_ms > 0 else 0
            print(f"  t={t:5.1f}s | err={errors[i]:.3f}m | "
                  f"wind={np.linalg.norm(wind_vec):.1f}m/s | "
                  f"alpha=[{alpha_opt[0]:.2f},{alpha_opt[1]:.2f},{alpha_opt[2]:.2f}] | "
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
        'mean_err':   float(errors[100:].mean()),
        'mean_alpha': alphas[100:].mean(axis=0),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render',     action='store_true')
    parser.add_argument('--skydio',     action='store_true',
                        help='using skydio model (same parameters as hummingbird)')
    parser.add_argument('--wind',       type=float, default=8.0)
    parser.add_argument('--traj',       type=str,   default='circle',
                        choices=['hover', 'circle'])
    parser.add_argument('--sim_time',   type=float, default=20.0)
    parser.add_argument('--no_pinn',    action='store_true')
    parser.add_argument('--no_rollout', action='store_true')
    parser.add_argument('--obstacle',   action='store_true',
                        help='Enable obstacle avoidance (cylinder at x=1.9, r=0.25m)')
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

    obstacles = DEFAULT_OBSTACLES if args.obstacle else None

    if args.render:
        use_pinn = not args.no_pinn
        label    = 'PINN-MPPI' if use_pinn else 'Nominal MPPI'
        model_l  = 'Skydio X2 (visual)' if args.skydio else 'Hummingbird'
        obs_l    = f' | obstacle ON (x=1.9)' if args.obstacle else ''
        print(f"\nVisualization: {label} | {args.traj} | wind={args.wind}m/s | {model_l}{obs_l}")
        wind_vec = np.array([args.wind, 0., 0.])
        r = run_episode(
            model, normalizer, wind_vec, trajs[args.traj],
            sim_time=args.sim_time, K=896, H=15,
            use_pinn=use_pinn,
            show_rollout=not args.no_rollout,
            use_skydio=args.skydio,
            obstacles=obstacles,
            render=True, verbose=True,
        )
        print(f"\n Mean Error: {r['mean_err']:.4f}m")
        ma = r['mean_alpha']
        print(f"Average alpha: [{ma[0]:.2f},{ma[1]:.2f},{ma[2]:.2f}]")
        return

    # Headless batch test
    test_configs = [
        (0.,  'train'), (2., 'train'), (4.,  'train'), (6., 'train'), (8.,  'train'),
        (10., 'OOD'),   (12., 'OOD'),
    ]
    all_results = {}
    for tname, tfn in trajs.items():
        print(f"\n{'='*68}\ntrajectory: {tname}  [MuJoCo]\n{'='*68}")
        print(f"\n{'Wind':>6}|{'Split':>5}|{'Nominal':>10}|"
              f"{'PINN-MPPI':>10}|{'Improvement':>8}|{'alpha mean':>20}")
        print("-"*65)
        results = {}
        for ws, split in test_configs:
            wv    = np.array([ws, 0., 0.])
            r_nom = run_episode(model, normalizer, wv, tfn,
                                K=896, H=15, use_pinn=False,
                                use_skydio=args.skydio,
                                obstacles=obstacles, verbose=False)
            r_pin = run_episode(model, normalizer, wv, tfn,
                                K=896, H=15, use_pinn=True,
                                use_skydio=args.skydio,
                                obstacles=obstacles, verbose=False)
            imp = (r_nom['mean_err'] - r_pin['mean_err']) / (r_nom['mean_err'] + 1e-9) * 100
            ma  = r_pin['mean_alpha']
            print(f"{ws:6.0f}|{split:>5}|{r_nom['mean_err']:10.4f}|"
                  f"{r_pin['mean_err']:10.4f}|{imp:+7.1f}%|"
                  f"[{ma[0]:.2f},{ma[1]:.2f},{ma[2]:.2f}]")
            results[ws] = {'nom': r_nom, 'pinn': r_pin}
        all_results[tname] = results

    np.save(os.path.join(RESULT_DIR, 'mppi_mujoco.npy'),
            all_results, allow_pickle=True)
    print("\nResult Saved")


if __name__ == '__main__':
    main()
