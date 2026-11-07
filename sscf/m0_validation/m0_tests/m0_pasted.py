import numpy as np
from dataclasses import dataclass

def generate_realistic_cloud_traces(n_workloads=120, T=1024, regime="realistic_bursty", seed=42):
    rng = np.random.default_rng(seed)
    traces = np.zeros((n_workloads, T)); t = np.arange(T)
    for i in range(n_workloads):
        base_load = rng.uniform(0.05, 0.15)
        if regime == "ideal_harmonic":
            freq = rng.choice([8, 16, 32, 64]) / T
            phase = rng.uniform(0, 2*np.pi)
            wave = 0.08*np.sin(2*np.pi*freq*t + phase)
            noise = rng.normal(0, 0.015, size=T)
            traces[i] = np.clip(base_load + wave + noise, 0, None)
        else:
            white = rng.standard_normal(T)
            fft_white = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(T); freqs[0] = 1.0
            pink = np.fft.irfft(fft_white/(freqs**0.5), n=T)
            pink = 0.04*(pink/(np.std(pink)+1e-8))
            freq = rng.choice([12, 24, 48])/T
            phase_walk = np.cumsum(rng.normal(0, 0.08, size=T))
            quasi_wave = 0.04*np.sin(2*np.pi*freq*t + phase_walk)
            bursts = (rng.pareto(a=2.2, size=T)-1.0)*(rng.random(T) < 0.05)*0.06
            traces[i] = np.clip(base_load + pink + quasi_wave + bursts, 0, None)
    return traces

def evaluate_spectral_viability(traces, window_size=128, top_k=3):
    n_workloads, T = traces.shape
    n_windows = T // window_size
    ser_list, coherence_list = [], []
    for w in range(n_workloads):
        x = traces[w]; prev_phases=None; dom_bin=None; phase_diffs=[]
        for m in range(n_windows):
            seg = x[m*window_size:(m+1)*window_size]
            spec = np.fft.rfft(seg - np.mean(seg))
            power = np.abs(spec[1:])**2
            total = np.sum(power)+1e-12
            top = np.argsort(power)[-top_k:]
            ser_list.append(np.sum(power[top])/total)
            if dom_bin is None: dom_bin = int(np.argmax(power))+1
            cp = np.angle(spec[dom_bin])
            if prev_phases is not None: phase_diffs.append(np.exp(1j*(cp-prev_phases)))
            prev_phases = cp
        if phase_diffs: coherence_list.append(np.abs(np.mean(phase_diffs)))
    return float(np.mean(ser_list)), float(np.mean(coherence_list))

def simulate_nonlinear_lindley_queue(rack_loads, capacity):
    n_racks, T = rack_loads.shape
    queues = np.zeros((n_racks, T))
    for r in range(n_racks):
        q = 0.0
        for t in range(T):
            load = rack_loads[r, t]; u = load/capacity
            eff = load*(1.0+0.5*((u-0.85)/0.15)**2) if u > 0.85 else load
            q = max(0.0, q+eff-capacity); queues[r, t] = q
    return float(np.percentile(queues, 99))

def run_placement_competition(traces, n_racks=10, warmup=256):
    n_workloads, T = traces.shape
    train = traces[:, :warmup]; test = traces[:, warmup:]
    total_mean_demand = np.sum(np.mean(traces, axis=1))
    rack_capacity = (total_mean_demand/n_racks)/0.88
    racks_harm = np.zeros((n_racks, T-warmup)); rack_dc = np.zeros(n_racks)
    rack_spec = np.zeros((n_racks, warmup//2+1), dtype=np.complex128)
    for i in range(n_workloads):
        dc = np.mean(train[i]); spec = np.fft.rfft(train[i]-dc)
        power = np.abs(spec[1:])**2; top2 = np.argsort(power)[-2:]+1
        sparse = np.zeros_like(spec); sparse[top2] = spec[top2]
        best_r, best_score = 0, float("inf")
        for r in range(n_racks):
            if rack_dc[r]+dc > rack_capacity*0.98: score = 1e9+rack_dc[r]
            else:
                comb = rack_spec[r]+sparse
                ac = np.sum(np.abs(comb.real)+np.abs(comb.imag))/warmup
                score = (rack_dc[r]+dc)+1.5*ac
            if score < best_score: best_score, best_r = score, r
        rack_dc[best_r] += dc; rack_spec[best_r] += sparse; racks_harm[best_r] += test[i]
    racks_eb = np.zeros((n_racks, T-warmup)); eb_mu=np.zeros(n_racks); eb_var=np.zeros(n_racks)
    for i in range(n_workloads):
        mu, var = np.mean(train[i]), np.var(train[i])
        scores = (eb_mu+mu)+2.33*np.sqrt(eb_var+var); b=int(np.argmin(scores))
        eb_mu[b]+=mu; eb_var[b]+=var; racks_eb[b]+=test[i]
    rng = np.random.default_rng(99)
    racks_po2c = np.zeros((n_racks, T-warmup)); load=np.zeros(n_racks)
    for i in range(n_workloads):
        mu=np.mean(train[i]); c1,c2=rng.choice(n_racks,size=2,replace=False)
        b = c1 if load[c1]<=load[c2] else c2
        load[b]+=mu; racks_po2c[b]+=test[i]
    return (simulate_nonlinear_lindley_queue(racks_harm,rack_capacity),
            simulate_nonlinear_lindley_queue(racks_eb,rack_capacity),
            simulate_nonlinear_lindley_queue(racks_po2c,rack_capacity))

if __name__ == "__main__":
    for regime in ["ideal_harmonic","realistic_bursty"]:
        tr = generate_realistic_cloud_traces(regime=regime)
        s,c = evaluate_spectral_viability(tr); qh,qe,qp = run_placement_competition(tr)
        print(f"== {regime}: SER3={s:.2%} coh={c:.2f} Q99 harm={qh:.4f} effbw={qe:.4f} po2c={qp:.4f} gain_vs_eb={(qe-qh)/(qe+1e-9):+.1%}")
