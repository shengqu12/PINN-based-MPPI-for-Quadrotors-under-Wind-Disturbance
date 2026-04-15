"""
experiments/all_controllers.py — full controller comparison and control frequency comparison

controller:
    1. SE3 Only         
    2. SE3 + PINN FF    constant feedforward (α=0.4)
    3. Oracle           α=1.0 fully compensate(should be the upper limit)
    4. Nominal MPPI     MPPI without PINN using nominal dynamics
    5. PINN-MPPI        main method(MPPI online optimization α=[αx,αy,αz])

traj: hover, circle, lemniscate, spiral
sampling efficiency: K=200 vs K=896 (H=15) 
"""

import os, sys, argparse, time
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.pinn import ResidualPINN
from training.dataset import Normalizer
from controllers.pinn_mppi_GPU import MPPIController, pinn_predict
from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj

CKPT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

K_ETA       = quad_params['k_eta']
MASS        = quad_params['mass']
G           = 9.81
HOVER_OMEGA = float(np.sqrt(MASS * G / (4 * K_ETA)))
KP_POS      = np.array([6.5, 6.5, 15.0])


# ── traj ─────────────────────────────────────────────────────────────
# lemniscate
def make_lemniscate(scale=1.5, period=10.0):
    class _T:
        def update(self, t):
            w     = 2*np.pi/period
            s     = w*t
            d     = 1+np.sin(s)**2
            x     = scale*np.cos(s)/d
            y     = scale*np.sin(s)*np.cos(s)/d
            eps   = 1e-4
            d2    = 1+np.sin(s+eps*w)**2
            x2    = scale*np.cos(s+eps*w)/d2
            y2    = scale*np.sin(s+eps*w)*np.cos(s+eps*w)/d2
            dx    = (x2-x)/eps; dy = (y2-y)/eps
            return {'x':np.array([x,y,1.5]),'x_dot':np.array([dx,dy,0.]),
                    'x_ddot':np.zeros(3),'yaw':0.,'yaw_dot':0.}
    return _T()

# spiral
def make_spiral(radius=1.5, height=1.0, period=10.0):
    class _T:
        def update(self, t):
            w  = 2*np.pi/period
            f  = min(t/period, 1.0)
            x  = radius*np.cos(w*t); y = radius*np.sin(w*t)
            z  = 1.5+height*f
            dx = -radius*w*np.sin(w*t); dy = radius*w*np.cos(w*t)
            dz = height/period
            return {'x':np.array([x,y,z]),'x_dot':np.array([dx,dy,dz]),
                    'x_ddot':np.zeros(3),'yaw':0.,'yaw_dot':0.}
    return _T()

TRAJECTORIES = {
    'hover':      lambda: HoverTraj(x0=np.array([0.,0.,1.5])),
    'circle':     lambda: ThreeDCircularTraj(
                      center=np.array([0.,0.,1.5]),
                      radius=np.array([1.5,1.5,0.]),
                      freq=np.array([0.2,0.2,0.])),
    'lemniscate': make_lemniscate,
    'spiral':     make_spiral,
}

WIND_CONFIGS = [
    (0.,  'train'), (2., 'train'), (4.,  'train'), (6., 'train'), (8.,  'train'),
    (10., 'OOD'),   (12., 'OOD'),
]


# ── controller ────────────────────────────────────────────────────────────

def _init(wind_vec, traj_fn):
    state = {'x':np.array([0.,0.,1.5]),'v':np.zeros(3),
             'q':np.array([0.,0.,0.,1.]),'w':np.zeros(3),
             'wind':wind_vec.copy(),'rotor_speeds':np.ones(4)*HOVER_OMEGA}
    return (Multirotor(quad_params,state), SE3Control(quad_params),
            traj_fn(), state)

# se3_controller
def run_se3_only(wind_vec, traj_fn, sim_time=15., dt=0.01):
    vehicle, ctrl, traj, state = _init(wind_vec, traj_fn)
    N = int(sim_time/dt); errs=[]
    for i in range(N):
        t=i*dt; state['wind']=wind_vec.copy()
        flat=traj.update(t)
        state=vehicle.step(state, ctrl.update(t,state,flat), dt)
        errs.append(np.linalg.norm(state['x']-flat['x']))
    return {'mean_err':np.mean(errs[100:]),'errors':np.array(errs)}

# pinn_feedforward_se3_controller
def run_se3_pinn_ff(model, norm, wind_vec, traj_fn,
                    sim_time=15., dt=0.01, alpha=0.4):
    vehicle, ctrl, traj, state = _init(wind_vec, traj_fn)
    N=int(sim_time/dt); errs=[]; r_ema=np.zeros(3)
    for i in range(N):
        t=i*dt; state['wind']=wind_vec.copy()
        flat=traj.update(t)
        mu=np.array(state.get('rotor_speeds',np.ones(4)*HOVER_OMEGA))
        r_raw=pinn_predict(model,norm,state,mu,wind_vec,vel_ref=flat['x_dot'])
        r_ema=0.85*r_ema+0.15*r_raw
        dp=np.clip(-r_ema/KP_POS,-0.25,0.25)
        fm=dict(flat); fm['x']=flat['x']+alpha*dp; fm['x_dot']=flat['x_dot']
        state=vehicle.step(state, ctrl.update(t,state,fm), dt)
        errs.append(np.linalg.norm(state['x']-flat['x']))
    return {'mean_err':np.mean(errs[100:]),'errors':np.array(errs)}

# oracle a=1
def run_oracle(model, norm, wind_vec, traj_fn, sim_time=15., dt=0.01):
    return run_se3_pinn_ff(model,norm,wind_vec,traj_fn,sim_time,dt,alpha=1.0)

# MPPI 
def run_mppi(model, norm, wind_vec, traj_fn,
             sim_time=15., dt=0.01, K=896, H=15, use_pinn=True):
    state={'x':np.array([0.,0.,1.5]),'v':np.zeros(3),
           'q':np.array([0.,0.,0.,1.]),'w':np.zeros(3),
           'wind':wind_vec.copy(),'rotor_speeds':np.ones(4)*HOVER_OMEGA}
    vehicle=Multirotor(quad_params,state); ctrl=SE3Control(quad_params)
    mppi=MPPIController(model,norm,wind_vec,K=K,H=H,dt=dt,use_pinn=use_pinn)
    N=int(sim_time/dt)
    _t=traj_fn()
    all_refs=[_t.update(i*dt) for i in range(N+H+1)]
    all_pos=np.array([r['x'] for r in all_refs],dtype=np.float32)
    all_vel=np.array([r['x_dot'] for r in all_refs],dtype=np.float32)
    errs=[]; ms_list=[]
    for i in range(N):
        t=i*dt; state['wind']=wind_vec.copy()
        mu=np.array(state.get('rotor_speeds',np.ones(4)*HOVER_OMEGA))
        res=pinn_predict(model,norm,state,mu,wind_vec,vel_ref=all_vel[i]) \
            if use_pinn else np.zeros(3)
        t0=time.perf_counter()
        alpha,dpc,_=mppi.update(state,all_pos[i:i+H],all_vel[i:i+H],mu,res)
        ms_list.append((time.perf_counter()-t0)*1000)
        fm=dict(all_refs[i])
        fm['x']=all_pos[i]+alpha*dpc; fm['x_dot']=all_vel[i]
        state=vehicle.step(state,ctrl.update(t,state,fm),dt)
        errs.append(np.linalg.norm(state['x']-all_pos[i]))
    return {'mean_err':np.mean(errs[100:]),'errors':np.array(errs),
            'mean_hz':1000./np.mean(ms_list[10:]) if len(ms_list)>10 else 0}


# ── main experiment ────────────────────────────────────────────────────────────

def run_all(model, norm, trajs, wind_cfgs, sim_time=15., K=896):
    all_res={}
    for tname, tfn in trajs.items():
        print(f"\n{'='*82}\ntraj: {tname}\n{'='*82}")
        print(f"\n{'Wind':>5}|{'Split':>5}|{'SE3':>8}|{'SE3+FF':>8}|"
              f"{'Oracle':>8}|{'NomMPPI':>8}|{'PINN-MPPI':>10}|{'improvement%':>7}|{'Hz':>6}")
        print("-"*74)
        tres={}
        for ws, split in wind_cfgs:
            wv=np.array([ws,0.,0.])
            r_se3 =run_se3_only(wv,tfn,sim_time)
            r_ff  =run_se3_pinn_ff(model,norm,wv,tfn,sim_time)
            r_orc =run_oracle(model,norm,wv,tfn,sim_time)
            r_nom =run_mppi(model,norm,wv,tfn,sim_time,K=K,use_pinn=False)
            r_pin =run_mppi(model,norm,wv,tfn,sim_time,K=K,use_pinn=True)
            base=r_se3['mean_err']+1e-9
            imp=(base-r_pin['mean_err'])/base*100
            print(f"{ws:5.0f}|{split:>5}|{r_se3['mean_err']:8.4f}|"
                  f"{r_ff['mean_err']:8.4f}|{r_orc['mean_err']:8.4f}|"
                  f"{r_nom['mean_err']:8.4f}|{r_pin['mean_err']:10.4f}|"
                  f"{imp:+6.1f}%|{r_pin['mean_hz']:6.1f}")
            tres[ws]={'se3':r_se3,'se3_pinn':r_ff,'oracle':r_orc,
                      'nom_mppi':r_nom,'pinn_mppi':r_pin,'split':split}
        all_res[tname]=tres
    return all_res

# control efficiency 
def run_sampling_efficiency(model, norm):
    print(f"\n{'='*50}\nsampling efficiency: K=200 vs K=896\n{'='*50}")
    print(f"\n{'K':>6}|{'Hz':>8}|{'hover w=8':>11}|{'circle w=8':>12}")
    print("-"*42)
    wv=np.array([8.,0.,0.])
    res={}
    for K in [200,896]:
        rh=run_mppi(model,norm,wv,TRAJECTORIES['hover'],K=K)
        rc=run_mppi(model,norm,wv,TRAJECTORIES['circle'],K=K)
        res[K]={'hover':rh,'circle':rc}
        print(f"{K:6d}|{rh['mean_hz']:8.1f}|{rh['mean_err']:11.4f}|{rc['mean_err']:12.4f}")
    for traj in ['hover','circle']:
        imp=(res[200][traj]['mean_err']-res[896][traj]['mean_err']) \
            /(res[896][traj]['mean_err']+1e-9)*100
        print(f"  {traj}: K=200 vs K=896 error {imp:+.1f}%")
    return res


def plot_results(all_res, path):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        tnames=list(all_res.keys()); nt=len(tnames)
        fig,axes=plt.subplots(nt,2,figsize=(14,4*nt))
        if nt==1: axes=[axes]
        fig.suptitle('Controller Comparison Experiment',fontsize=13)
        C={'SE3':'steelblue','SE3+PINN':'mediumseagreen',
           'Oracle':'gold','Nom-MPPI':'mediumpurple','PINN-MPPI':'darkorange'}
        for row,tname in enumerate(tnames):
            res=all_res[tname]; ws_list=sorted(res.keys())
            ax=axes[row][0]; x=np.arange(len(ws_list)); w=0.16
            for ci,(lbl,key) in enumerate([
                ('SE3','se3'),('SE3+PINN','se3_pinn'),
                ('Oracle','oracle'),('Nom-MPPI','nom_mppi'),
                ('PINN-MPPI','pinn_mppi')]):
                vals=[res[ws][key]['mean_err'] for ws in ws_list]
                ax.bar(x+ci*w-2*w,vals,w,label=lbl,color=C[lbl],alpha=0.85)
            ax.axvline(x=1.5,color='gray',ls='--',alpha=0.5)
            ax.set_xticks(x+w); ax.set_xticklabels([f'{ws}m/s' for ws in ws_list])
            ax.set_xlabel('Wind'); ax.set_ylabel('Error (m)')
            ax.set_title(tname); ax.legend(fontsize=7); ax.grid(True,alpha=0.3,axis='y')
            ax2=axes[row][1]; ws=8.
            if ws in res:
                ts=np.arange(len(res[ws]['se3']['errors']))*0.01
                for lbl,key in [('SE3','se3'),('SE3+PINN','se3_pinn'),
                                 ('Oracle','oracle'),('PINN-MPPI','pinn_mppi')]:
                    e=res[ws][key]['errors']
                    ax2.plot(ts[:len(e)],e,label=lbl,color=C[lbl],alpha=0.85)
                ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Error (m)')
                ax2.set_title(f'{tname} wind=8m/s')
                ax2.legend(fontsize=7); ax2.grid(True,alpha=0.3)
        plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches='tight')
        print(f"\nGraph: {path}")
    except Exception as e:
        print(f"Jump: {e}")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--quick',       action='store_true',help='only hover+circle')
    parser.add_argument('--no_sampling', action='store_true',help='no sampling efficiency comparison')
    args=parser.parse_args()
    os.makedirs(RESULT_DIR,exist_ok=True)

    ckpt=torch.load(os.path.join(CKPT_DIR,'best_model.pt'),weights_only=False)
    norm=Normalizer(); norm.load(os.path.join(CKPT_DIR,'normalizer.pt'))
    model=ResidualPINN(input_dim=17)
    model.load_state_dict(ckpt['model_state']); model.eval()
    print(f"PINN epoch={ckpt['epoch']}  Val={ckpt['val_rmse'].round(3)}")

    trajs=({k:TRAJECTORIES[k] for k in ['hover','circle']} if args.quick
           else TRAJECTORIES)
    all_res=run_all(model,norm,trajs,WIND_CONFIGS,K=1000)

    if not args.no_sampling:
        run_sampling_efficiency(model,norm)

    np.save(os.path.join(RESULT_DIR,'all_controllers.npy'),all_res,allow_pickle=True)
    plot_results(all_res,os.path.join(RESULT_DIR,'all_controllers.png'))
    print("\nCompleted ✓")

if __name__=='__main__':
    main()