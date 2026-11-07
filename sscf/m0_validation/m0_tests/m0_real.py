"""Run the M0 gate on REAL traces.
Input: a CSV/NPY matrix, rows = workloads (VM/container), cols = time bins (uniform sampling).
  python m0_real.py trace.csv --racks 10 --warmup 256 --utils 0.6 0.75 0.88
Prepare the matrix from: Azure VM trace (avg CPU per 5 min), Alibaba 2018 container_usage (cpu_util_percent, 10 s),
Google 2019 (instance_usage average_usage.cpus, 5 min). Resample to equal length and drop all-zero rows.
Note: 5-min/10-s sampling tests macro/minute-scale structure only; ms-scale needs local perf captures (M0-B).
"""
import argparse, numpy as np
from m0_pasted import evaluate_spectral_viability, simulate_nonlinear_lindley_queue as lind
from m0_robust import place, methods
ap=argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--racks",type=int,default=10)
ap.add_argument("--warmup",type=int,default=256); ap.add_argument("--utils",type=float,nargs="+",default=[0.6,0.75,0.88])
ap.add_argument("--reps",type=int,default=10, help="random workload subsamples for CI")
ap.add_argument("--n",type=int,default=120, help="workloads per subsample")
a=ap.parse_args()
X=np.load(a.path) if a.path.endswith(".npy") else np.loadtxt(a.path,delimiter=",")
X=X[X.std(1)>0]; print("workloads x bins:",X.shape)
rng=np.random.default_rng(0)
for util in a.utils:
    res={m:[] for m in methods}; sers=[]; cohs=[]
    for r in range(a.reps):
        idx=rng.choice(len(X),min(a.n,len(X)),replace=False); tr=X[idx]
        s,c=evaluate_spectral_viability(tr,window_size=min(128,tr.shape[1]//4)); sers.append(s); cohs.append(c)
        for m in methods:
            R,cap=place(tr,a.racks,a.warmup,util,m,r); res[m].append(lind(R,cap))
    eb=np.array(res["effbw"])
    print(f"\nutil={util}: SER3={np.mean(sers):.1%} coherence={np.mean(cohs):.2f}")
    for m in methods:
        v=np.array(res[m]); print(f"  {m:11s} median={np.median(v):9.3f} mean={v.mean():9.3f} beats_effbw={np.mean(v<eb):.0%}")
    h=np.array(res["harm_k2"]); print(f"  harm_k2 vs effbw median change: {(np.median(eb)-np.median(h))/(np.median(eb)+1e-9):+.1%}")
