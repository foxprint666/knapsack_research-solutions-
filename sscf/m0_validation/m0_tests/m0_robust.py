import numpy as np
from m0_pasted import generate_realistic_cloud_traces as gen, evaluate_spectral_viability, simulate_nonlinear_lindley_queue as lind

def place(traces, n_racks, warmup, util, method, seed):
    n,T = traces.shape; train=traces[:,:warmup]; test=traces[:,warmup:]
    cap = (traces.sum(0).mean()/n_racks)/util
    R = np.zeros((n_racks,T-warmup)); rng=np.random.default_rng(seed)
    dc=np.zeros(n_racks); var=np.zeros(n_racks); ser=np.zeros((n_racks,warmup)); spec=np.zeros((n_racks,warmup//2+1),complex)
    peaks=np.zeros(n_racks); cnt=np.zeros(n_racks)
    order = np.argsort(-train.mean(1)) if method!="po2c" else range(n)
    for i in order:
        mu=train[i].mean(); v=train[i].var()
        if method=="random": b=rng.integers(n_racks)
        elif method=="po2c":
            c=rng.choice(n_racks,2,replace=False); b=c[0] if dc[c[0]]<=dc[c[1]] else c[1]
        elif method=="static_p99":
            p=np.percentile(train[i],99); b=int(np.argmin(peaks)); peaks[b]+=p
        elif method=="effbw":
            b=int(np.argmin((dc+mu)+2.33*np.sqrt(var+v)))
        elif method=="corr_full":   # time-domain: minimise peak/std of summed TRAIN series (upper bound for any compression)
            sc=[(ser[r]+train[i]).std()*2.33+(dc[r]+mu) for r in range(n_racks)]; b=int(np.argmin(sc))
        elif method=="harm_k2":
            s=np.fft.rfft(train[i]-mu); p=np.abs(s[1:])**2; t=np.argsort(p)[-2:]+1
            sp=np.zeros_like(s); sp[t]=s[t]
            sc=[(dc[r]+mu)+1.5*np.sum(np.abs((spec[r]+sp).real)+np.abs((spec[r]+sp).imag))/warmup for r in range(n_racks)]
            b=int(np.argmin(sc)); spec[b]+=sp
        dc[b]+=mu; var[b]+=v; ser[b]+=train[i]; R[b]+=test[i]
    return R,cap

methods=["random","po2c","static_p99","effbw","corr_full","harm_k2"]
if __name__=="__main__":
    for regime in ["ideal_harmonic","realistic_bursty"]:
        for util in [0.6,0.75,0.88]:
            res={m:[] for m in methods}; sers=[]; cohs=[]
            for seed in range(20):
                tr=gen(regime=regime,seed=seed); s,c=evaluate_spectral_viability(tr); sers.append(s); cohs.append(c)
                for m in methods:
                    R,cap=place(tr,10,256,util,m,seed); res[m].append(lind(R,cap))
            print(f"\n[{regime}] util={util}  SER3={np.mean(sers):.1%} coh={np.mean(cohs):.2f}  (median P99 backlog over 20 seeds; [wins vs effbw])")
            eb=np.array(res["effbw"])
            for m in methods:
                a=np.array(res[m]); print(f"   {m:11s} median={np.median(a):8.3f}  mean={a.mean():8.3f}  better_than_effbw={np.mean(a<eb):.0%}")
