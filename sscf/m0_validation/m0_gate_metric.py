import argparse
import numpy as np

def compute_diurnal_metrics(X, bin_seconds=300):
    """
    Per workload, measures:
      1. Energy in the 24h, 12h, and 8h diurnal harmonics (SER_diurnal).
      2. Day-to-day phase consistency (phase coherence across daily cycles).
      3. Mean 24h phase angle (hour of peak).
    """
    n_workloads, T = X.shape
    bins_per_day = 86400 // bin_seconds
    D = T // bins_per_day
    if D < 1:
        raise ValueError(f"Trace length ({T} bins) is less than 1 day ({bins_per_day} bins).")
    
    X_cut = X[:, :D * bins_per_day]
    ser_list, stab_list, phase_list = [], [], []
    
    # Target diurnal harmonics across the D-day window
    # 24h = D cycles, 12h = 2D cycles, 8h = 3D cycles
    f24, f12, f8 = D, 2 * D, 3 * D
    
    for i in range(n_workloads):
        x = X_cut[i]
        mu = np.mean(x)
        spec = np.fft.rfft(x - mu)
        power = np.abs(spec[1:])**2
        total_power = np.sum(power) + 1e-12
        
        # Diurnal power at 24h, 12h, 8h (indices in power are f-1)
        diurnal_bins = [f24 - 1, f12 - 1, f8 - 1]
        p_diurnal = np.sum(power[diurnal_bins])
        ser_diurnal = p_diurnal / total_power
        ser_list.append(ser_diurnal)
        
        # Measure 24h fundamental phase day-by-day
        daily_phases = []
        for d in range(D):
            day_seg = x[d * bins_per_day : (d + 1) * bins_per_day]
            day_spec = np.fft.rfft(day_seg - np.mean(day_seg))
            # fundamental 24h harmonic is bin 1 in a 1-day slice
            daily_phases.append(np.angle(day_spec[1]))
            
        phasors = np.exp(1j * np.array(daily_phases))
        phase_coherence = float(np.abs(np.mean(phasors)))
        mean_phase = float(np.angle(np.mean(phasors)))
        
        stab_list.append(phase_coherence)
        phase_list.append(mean_phase)
        
    return np.array(ser_list), np.array(stab_list), np.array(phase_list), D

def evaluate_threshold_grid(sers, stabs, phases, weights=None,
                            energy_thresholds=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85),
                            phase_thresholds=(0.5, 0.7, 0.8, 0.9)):
    """
    Sweeps a grid of energy and phase stability thresholds.
    Returns table of periodic shares (by count & weight) and phase spread R.
    """
    n = len(sers)
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.array(weights, dtype=np.float64)
        if len(weights) != n:
            raise ValueError(f"Weights length ({len(weights)}) does not match workloads count ({n}).")
    total_weight = np.sum(weights) + 1e-12

    results = []
    for eth in energy_thresholds:
        for pth in phase_thresholds:
            passing = (sers >= eth) & (stabs >= pth)
            count = int(np.sum(passing))
            share_count = count / n
            share_weight = float(np.sum(weights[passing]) / total_weight)
            
            if count > 0:
                pass_weights = weights[passing]
                # Phase spread R: resultant vector magnitude of periodic phases
                R = float(np.abs(np.sum(pass_weights * np.exp(1j * phases[passing])) / np.sum(pass_weights)))
            else:
                R = np.nan
                
            results.append({
                "energy_thresh": eth,
                "phase_thresh": pth,
                "count": count,
                "share_count": share_count,
                "share_weight": share_weight,
                "R": R
            })
    return results

def generate_sanity_data(n=300, days=7, bin_sec=300, p_per=0.4, seed=42):
    """
    Generates 300 synthetic workloads:
      - 40% planted noisy diurnal with uniformly spread phases
      - 60% bursty noise (pink noise + Pareto bursts)
    """
    rng = np.random.default_rng(seed)
    bins_per_day = 86400 // bin_sec
    T = days * bins_per_day
    t = np.arange(T)
    X = np.zeros((n, T))
    is_per = rng.random(n) < p_per
    
    for i in range(n):
        base = rng.uniform(0.1, 0.3)
        if is_per[i]:
            phase = rng.uniform(0, 2 * np.pi)
            f24 = 1.0 / bins_per_day
            diurnal = 0.15 * np.sin(2 * np.pi * f24 * t + phase)
            h12 = 0.05 * np.sin(2 * np.pi * (2 * f24) * t + 2 * phase)
            noise = rng.normal(0, 0.08, size=T)
            X[i] = np.clip(base + diurnal + h12 + noise, 0, None)
        else:
            white = rng.standard_normal(T)
            fft_w = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(T)
            freqs[0] = 1.0
            pink = np.fft.irfft(fft_w / (freqs**0.5), n=T)
            pink = 0.08 * (pink / (np.std(pink) + 1e-8))
            bursts = (rng.pareto(a=2.2, size=T) - 1.0) * (rng.random(T) < 0.05) * 0.15
            X[i] = np.clip(base + pink + bursts, 0, None)
            
    cores = rng.choice([2, 4, 8, 16, 32, 64], size=n)
    weights = cores * (T * bin_sec / 3600.0)
    return X, weights, is_per

def print_decision_report(results, n_total, D, bin_seconds):
    print("=" * 90)
    print("M0 GATE DECISION METRIC REPORT: REAL TRACE DIURNAL PERIODICITY SWEEP")
    print("=" * 90)
    print(f"Workloads evaluated : {n_total}")
    print(f"Duration            : {D} full days ({D * 86400 // bin_seconds} bins @ {bin_seconds}s/bin)")
    print("-" * 90)
    print(f"{'Energy Thresh':>13} | {'Phase Thresh':>12} | {'Count':>6} | {'Share (Count)':>13} | {'Share (Weight)':>14} | {'Phase Spread R':>14}")
    print("-" * 90)
    
    for r in results:
        r_str = f"{r['R']:14.3f}" if not np.isnan(r['R']) else f"{'N/A':>14}"
        print(f"{r['energy_thresh']:13.2f} | {r['phase_thresh']:12.2f} | {r['count']:6d} | {r['share_count']:12.1%} | {r['share_weight']:13.1%} | {r_str}")
    print("=" * 90)
    print("DECISION RULE INTERPRETATION:")
    print("  * Below 50% periodic share:")
    print("      -> DROP harmonic placement. Build Active Shaper (Pivot B), Effective Bandwidth, and CXL (Plane 3).")
    print("  * 50% - 60% periodic share:")
    print("      -> INDETERMINATE (collect more data / longer observation window).")
    print("  * 60% or more periodic share, AND R << 1.0, AND high cluster utilization:")
    print("      -> RETAIN cancellation-aware scorer as an optional add-on.")
    print("  * Phase Spread R:")
    print("      -> Near 1.0 means all workloads peak at the same time (constructive collision, cannot cancel).")
    print("      -> Near 0.0 means complementary phases exist across the day (destructive cancellation viable).")
    print("=" * 90)

def main():
    parser = argparse.ArgumentParser(description="M0 Gate Metric: Real Trace Diurnal Periodicity & Phase Sweep")
    parser.add_argument("path", nargs="?", default=None, help="Path to workload matrix (.npy or .csv)")
    parser.add_argument("--bin-seconds", type=int, default=300, help="Time bin size in seconds (default: 300 for 5 min)")
    parser.add_argument("--weights", type=str, default=None, help="Path to workload weights (.npy or .csv, e.g. core-hours)")
    parser.add_argument("--min-days", type=int, default=3, help="Minimum required days of trace duration (default: 3)")
    parser.add_argument("--sanity-check", action="store_true", help="Run on 300-workload synthetic sanity check dataset")
    args = parser.parse_args()

    if args.sanity_check or args.path is None:
        print("[*] Running sanity check on 300-workload synthetic trace (40% noisy diurnal, 60% bursty)...")
        X, weights, ground_truth = generate_sanity_data(n=300, days=7, bin_sec=args.bin_seconds, p_per=0.4, seed=42)
        true_share = np.mean(ground_truth)
        print(f"[*] Ground truth planted periodic share: {true_share:.1%}")
    else:
        if args.path.endswith(".npy"):
            X = np.load(args.path)
        else:
            X = np.loadtxt(args.path, delimiter=",")
        
        valid = (np.std(X, axis=1) > 0)
        X = X[valid]
        print(f"[*] Loaded trace matrix: {X.shape[0]} active workloads x {X.shape[1]} bins")
        
        weights = None
        if args.weights:
            weights = np.load(args.weights) if args.weights.endswith(".npy") else np.loadtxt(args.weights, delimiter=",")
            weights = weights[valid]

    bins_per_day = 86400 // args.bin_seconds
    total_days = X.shape[1] // bins_per_day
    if total_days < args.min_days:
        print(f"[!] ERROR: Trace length {X.shape[1]} bins = {total_days} days. Minimum required is {args.min_days} days.")
        return

    sers, stabs, phases, D = compute_diurnal_metrics(X, bin_seconds=args.bin_seconds)
    results = evaluate_threshold_grid(sers, stabs, phases, weights=weights)
    print_decision_report(results, len(sers), D, args.bin_seconds)

if __name__ == "__main__":
    main()
