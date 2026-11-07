import numpy as np
from m0_pasted import generate_realistic_cloud_traces as gen, simulate_nonlinear_lindley_queue as lind

def mixed(n, T, p_per, seed):
    rng=np.random.default_rng(seed)
    P=gen(n,T,"ideal_harmonic",seed); B=gen(n,T,"realistic_bursty",seed+1000)
    isper=rng.random(n)<p_per
    return np.where(isper[:,None],P,B), isper

def shape(tr, deferrable, w):
    out=tr.copy(); k=np.ones(w)/w
    for i in np.where(deferrable)[0]:
        out[i]=np.convolve(np.concatenate([np.full(w-1,tr[i,0]),tr[i]]),k,mode="valid")  # causal MA: mass-preserving, adds ~(w-1)/2 bins delay
    return out

def gate(x_train):
    h=len(x_train)//2; seg=lambda s:s-s.mean()
    s1=np.fft.rfft(seg(x_train[:h])); s2=np.fft.rfft(seg(x_train[h:2*h]))
    p=np.abs(s1[1:])**2; ser=np.sort(p)[-3:].sum()/(p.sum()+1e-12); k=int(np.argmax(p))+1
    stab=abs(np.exp(1j*(np.angle(s1[k])-np.angle(s2[k]))).real)  # |cos| of phase change (same freq, integer-cycle halves)
    return ser>0.85 and np.cos(np.angle(s1[k])-np.angle(s2[k]))>0.9

def periodic_part(x_train, warmup):
    mu=x_train.mean(); s=np.fft.rfft(x_train-mu); p=np.abs(s[1:])**2; t=np.argsort(p)[-2:]+1
    sp=np.zeros_like(s); sp[t]=s[t]; return np.fft.irfft(sp,n=warmup)

def place(tr, n_racks, warmup, util, method, seed):
    n,T=tr.shape; train=tr[:,:warmup]; test=tr[:,warmup:]
    cap=(tr.sum(0).mean()/n_racks)/util; R=np.zeros((n_racks,T-warmup))
    dc=np.zeros(n_racks); var=np.zeros(n_racks); Pser=np.zeros((n_racks,warmup)); Vnp=np.zeros(n_racks)
    for i in np.argsort(-train.mean(1)):
        mu=train[i].mean(); v=train[i].var()
        if method=="effbw":
            sc=(dc+mu)+2.33*np.sqrt(var+v)
        else:  # cancellation-aware effective bandwidth
            per=gate(train[i]); 
            if per:
                pp=periodic_part(train[i],warmup); resid=max(v-pp.var(),0)
                sc=np.array([(dc[r]+mu)+2.33*np.sqrt(Vnp[r]+resid+(Pser[r]+pp).var()) for r in range(n_racks)])
            else:
                sc=(dc+mu)+2.33*np.sqrt(Vnp+Pser.var(1)+v)
        b=int(np.argmin(sc)); dc[b]+=mu; var[b]+=v
        if method!="effbw":
            if per: Pser[b]+=pp; Vnp[b]+=resid
            else: Vnp[b]+=v
        R[b]+=test[i]
    return R,cap

def ci(d,rng):
    bs=[np.mean(rng.choice(d,len(d))) for _ in range(2000)]; return np.percentile(bs,[2.5,97.5])

if __name__=="__main__":
    rng=np.random.default_rng(1); S=30; W=8
    for util in [0.75,0.88]:
        for p_per in [0.0,0.3,0.6]:
            r={k:[] for k in ["effbw","hyb","effbw+shape","hyb+shape"]}
            for s in range(S):
                tr,isper=mixed(120,1024,p_per,s)
                dfr=(~isper)&(np.random.default_rng(s+777).random(120)<0.5)
                sh=shape(tr,dfr,W)
                for name,t,m in [("effbw",tr,"effbw"),("hyb",tr,"hyb"),("effbw+shape",sh,"effbw"),("hyb+shape",sh,"hyb")]:
                    R,cap=place(t,10,256,util,m,s); r[name].append(lind(R,cap))
            base=np.array(r["effbw"]); print(f"\nutil={util} periodic_fraction={p_per}  (backlog, 30 seeds; paired change vs effbw, negative=better)")
            for k,v in r.items():
                v=np.array(v); d=(v-base)/(base+1e-9)
                lo,hi=ci(v-base,rng); print(f"  {k:12s} median={np.median(v):8.1f} mean={v.mean():8.1f}  mean_diff={np.mean(v-base):+8.1f} 95%CI[{lo:+.1f},{hi:+.1f}]  wins={np.mean(v<base):.0%}")
