"""
m0_prime_checks.py
Rigorous validation of the Effective Bandwidth hypothesis on real Azure traces:
1. Rolling train-one-day / test-next-day evaluation across time-aligned 24h splits (Day 0->1, Day 1->2).
2. Direct ablation of variance: Compares Mean-Cap-Only (Sum(mu) <= U*C) vs EffBW (Sum(mu) + z*sqrt(Sum(var)) <= C)
   vs EffBW+Cap (both) vs Static P99 vs Borg-like rule vs Host-Sum Percentile rule (using actual sum series).
3. Variance Inflation Factor (VIF): Var(Sum_i X_i) / Sum_i Var(X_i) on test hosts to measure real cross-VM correlation.
4. Min-max error bounds over splits, raw overload rates (no shaping assumptions), and true VIF ratios.
"""

import argparse
import sys
import numpy as np
from scipy.stats import norm

BINS_PER_DAY = 288 # 86400 / 300 = 288 bins per day

def compute_vif(host_traces):
    """
    Computes the Variance Inflation Factor (VIF):
    VIF = Var( Sum_i X_i ) / Sum_i Var( X_i )
    VIF == 1.0 means perfect independence (covariances = 0).
    VIF > 1.0 means positive cross-VM covariance / co-bursting.
    """
    # host_traces is shape (N_vms_on_host, T)
    if host_traces.shape[0] <= 1:
        return 1.0
    summed_series = np.sum(host_traces, axis=0) # (T,)
    var_of_sum = float(np.var(summed_series))
    sum_of_vars = float(np.sum(np.var(host_traces, axis=1)))
    if sum_of_vars < 1e-9:
        return 1.0
    return var_of_sum / sum_of_vars

def pack_and_evaluate(trace, train_slice, test_slice, host_capacity, u_cap, z_val):
    N, T_total = trace.shape
    train_data = trace[:, train_slice]
    test_data = trace[:, test_slice]
    T_test = test_data.shape[1]
    
    # Train statistics
    mu_tr = np.mean(train_data, axis=1)
    std_tr = np.std(train_data, axis=1)
    var_tr = std_tr ** 2
    p95_tr = np.percentile(train_data, 95, axis=1)
    p99_tr = np.percentile(train_data, 99, axis=1)
    max_tr = np.max(train_data, axis=1)
    
    # Borg-like heuristic: max(1.5*mean, 1.15*p95)
    borg_size = np.maximum(mu_tr * 1.5, p95_tr * 1.15)
    
    order = np.argsort(-mu_tr) # Best-Fit Decreasing by mean demand
    
    policies = [
        "static_p99",
        "borg_like",
        "effbw_raw",
        "mean_cap_only",
        "effbw_plus_cap",
        "host_sum_p99"
    ]
    
    split_results = {}
    
    for pol in policies:
        hosts_vms = []
        host_mu = []
        host_var = []
        host_peak = []
        host_borg = []
        
        for idx in order:
            v_mu = mu_tr[idx]
            v_var = var_tr[idx]
            v_p99 = p99_tr[idx]
            v_borg = borg_size[idx]
            
            placed = False
            for h in range(len(hosts_vms)):
                cand_mu = host_mu[h] + v_mu
                cand_var = host_var[h] + v_var
                cand_peak = host_peak[h] + v_p99
                cand_borg = host_borg[h] + v_borg
                
                fits = False
                if pol == "static_p99":
                    fits = (cand_peak <= host_capacity)
                elif pol == "borg_like":
                    fits = (cand_borg <= host_capacity)
                elif pol == "effbw_raw":
                    # Pure Effective Bandwidth with NO mean cap: Sum(mu) + z*sqrt(Sum(var)) <= C
                    fits = (cand_mu + z_val * np.sqrt(cand_var) <= host_capacity)
                elif pol == "mean_cap_only":
                    # ABLATION: Mean-Cap alone with NO variance term: Sum(mu) <= u_cap * C
                    fits = (cand_mu <= u_cap * host_capacity)
                elif pol == "effbw_plus_cap":
                    # Combined: EffBW AND Mean-Cap
                    fits = (cand_mu + z_val * np.sqrt(cand_var) <= host_capacity) and (cand_mu <= u_cap * host_capacity)
                elif pol == "host_sum_p99":
                    # Host-sum percentile rule: evaluated using the actual historical summed series
                    # Candidate summed series on train data:
                    cand_series = np.sum(train_data[hosts_vms[h] + [idx], :], axis=0)
                    fits = (np.percentile(cand_series, 99) <= host_capacity)
                    
                if fits:
                    hosts_vms[h].append(idx)
                    host_mu[h] = cand_mu
                    host_var[h] = cand_var
                    host_peak[h] = cand_peak
                    host_borg[h] = cand_borg
                    placed = True
                    break
                    
            if not placed:
                hosts_vms.append([idx])
                host_mu.append(v_mu)
                host_var.append(v_var)
                host_peak.append(v_p99)
                host_borg.append(v_borg)
                
        num_hosts = len(hosts_vms)
        tot_test_intervals = num_hosts * T_test
        raw_overload_count = 0
        vif_list = []
        
        for h in range(num_hosts):
            vms = hosts_vms[h]
            host_test_traces = test_data[vms, :] # (len(vms), T_test)
            agg_test_load = np.sum(host_test_traces, axis=0)
            
            # Check overload against host capacity (raw unshaped demand)
            raw_overload_count += np.sum(agg_test_load > host_capacity)
            
            # Compute VIF on this host
            vif = compute_vif(host_test_traces)
            vif_list.append(vif)
            
        overload_rate = (raw_overload_count / max(1, tot_test_intervals)) * 100.0
        mean_vif = float(np.mean(vif_list)) if vif_list else 1.0
        
        split_results[pol] = {
            "hosts": num_hosts,
            "vms_per_host": N / num_hosts,
            "overload_pct": overload_rate,
            "vif": mean_vif
        }
        
    return split_results

def main():
    parser = argparse.ArgumentParser(description="M0' Rigorous Checks: Rolling Splits, Variance Ablation & VIF")
    parser.add_argument("trace", type=str, help="Path to trace.npy")
    parser.add_argument("--capacity", type=float, default=1600.0, help="Host capacity in CPU percent (default 1600)")
    parser.add_argument("--train-bins", type=int, default=288, help="Bins per train slice (default 288 = 1 day)")
    parser.add_argument("--u-cap", type=float, default=0.70, help="Mean utilization cap (default 0.70)")
    parser.add_argument("--z", type=float, default=2.33, help="Effective bandwidth z factor (default 2.33)")
    args = parser.parse_args()
    
    trace = np.load(args.trace)
    N, T = trace.shape
    bins_per_split = args.train_bins
    n_splits = (T // bins_per_split) - 1
    
    if n_splits < 1:
        print(f"ERROR: Trace has {T} bins; needs at least {bins_per_split * 2} bins for 1 rolling train/test split.")
        sys.exit(1)
        
    print("=" * 115)
    print(f"M0' CHECKS: ROLLING TRAIN-ONE-DAY / TEST-NEXT-DAY VALIDATION (Capacity = {args.capacity:.0f}%, u_cap = {args.u_cap:.2f}, z = {args.z:.2f})")
    print(f"Workloads: {N} VMs  |  Total Bins: {T} ({T/BINS_PER_DAY:.1f} days)  |  Rolling Splits: {n_splits}")
    print("=" * 115)
    
    all_splits_data = []
    for s in range(n_splits):
        tr_start = s * bins_per_split
        tr_end = (s + 1) * bins_per_split
        te_start = tr_end
        te_end = te_start + bins_per_split
        
        print(f"\n[Split {s+1}/{n_splits}] Train: bins [{tr_start}:{tr_end}] (Day {tr_start/BINS_PER_DAY:.1f}-{tr_end/BINS_PER_DAY:.1f}) -> Test: bins [{te_start}:{te_end}] (Day {te_start/BINS_PER_DAY:.1f}-{te_end/BINS_PER_DAY:.1f})")
        res = pack_and_evaluate(trace, slice(tr_start, tr_end), slice(te_start, te_end),
                                host_capacity=args.capacity, u_cap=args.u_cap, z_val=args.z)
        all_splits_data.append(res)
        
    # Aggregate over splits
    policies = list(all_splits_data[0].keys())
    pol_names = {
        "static_p99": "Static P99",
        "borg_like": "Borg-like Predictor",
        "effbw_raw": f"EffBW Raw (z={args.z})",
        "mean_cap_only": f"Mean-Cap-Only (u<={args.u_cap:.2f})",
        "effbw_plus_cap": f"EffBW + Cap (z={args.z}, u<={args.u_cap:.2f})",
        "host_sum_p99": "Host-Sum P99 Series"
    }
    
    print("\n" + "=" * 115)
    print("AGGREGATED MULTI-SPLIT COMPARISON & VARIANCE ABLATION")
    print("=" * 115)
    print(f"{'Policy':<30} | {'Hosts (mean)':<12} | {'Savings vs P99':<16} | {'Overload % [min, max]':<26} | {'Mean VIF'}")
    print("-" * 115)
    
    base_p99_hosts = np.mean([s["static_p99"]["hosts"] for s in all_splits_data])
    
    for pol in policies:
        hosts_vals = [s[pol]["hosts"] for s in all_splits_data]
        over_vals = [s[pol]["overload_pct"] for s in all_splits_data]
        vif_vals = [s[pol]["vif"] for s in all_splits_data]
        
        mean_h = np.mean(hosts_vals)
        mean_over = np.mean(over_vals)
        min_over = np.min(over_vals)
        max_over = np.max(over_vals)
        mean_v = np.mean(vif_vals)
        
        savings = ((base_p99_hosts - mean_h) / base_p99_hosts) * 100.0
        
        over_str = f"{mean_over:.2f}% [{min_over:.2f}%, {max_over:.2f}%]"
        print(f"{pol_names[pol]:<30} | {mean_h:<12.1f} | {savings:>+14.1f}% | {over_str:<26} | {mean_v:.2f}")

    print("=" * 115)
    print("INTERPRETATION GUIDE:")
    print("  * VIF (Variance Inflation Factor) = Var(Sum_i X_i) / Sum_i Var(X_i):")
    print("      VIF ~ 1.0  -> Workloads are uncorrelated; Sum(var) quadrature rule is valid.")
    print("      VIF >> 1.0 -> Workloads co-burst; Sum(var) severely underestimates aggregate tail risk.")
    print("  * ABLATION TEST (Mean-Cap-Only vs EffBW):")
    print("      If Mean-Cap-Only matches EffBW+Cap in hosts & overload, the variance term adds no value.")
    print("      If EffBW+Cap achieves lower overload or fewer hosts, the statistical term does active work.")
    print("=" * 115)

if __name__ == "__main__":
    main()
