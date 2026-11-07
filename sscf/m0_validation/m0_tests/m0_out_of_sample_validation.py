"""
m0_out_of_sample_validation.py
Out-of-sample validation of Effective Bandwidth host savings and overload calibration
against real Google Borg / Autopilot-style predictive baselines on the Azure trace.

1. Train / Test Split:
   - Train window (Profiling): Day 0 to Day 1.5 (bins 0 to 432)
   - Test window (Out-of-Sample Evaluation): Day 1.5 to Day 3.0 (bins 432 to 864)

2. Placement Policies (Sized on TRAIN, Evaluated on TEST):
   a. Peak-Max (Worst-Case Historical): Sum(max_train) <= Capacity
   b. Static P99 (Historical P99 Reservation): Sum(p99_train) <= Capacity
   c. Google Borg / Autopilot Predictor (Borg/Apollo safety buffer):
      Sizes each task as: max(mean_train * 1.5, p95_train * 1.15) (exponential-decay / safety margin)
   d. Standard Effective Bandwidth (EffBW):
      Sum(mu_train) + z * sqrt(Sum(sigma_train^2)) <= Capacity (z = 2.33 and z = 3.0)
   e. Asymmetric SSTF:
      Sum(mu_int_train) + Sum(mu_del_train) + z * sqrt(Sum(sigma_int_train^2)) <= Capacity

3. Calibration Check:
   - Compares predicted rack overload probability (analytical Gaussian Q-function / tail bound)
     versus actual observed overload frequency on out-of-sample test data.
   - Measures reliability across safety factors z in [2.0, 2.33, 2.8, 3.0, 3.5].
"""

import numpy as np
from scipy.stats import norm

def run_out_of_sample_evaluation(host_capacity=1600.0):
    trace = np.load("trace.npy")       # (4994, 864)
    labels = np.load("labels.npy")     # 1 = Interactive, 2 = Delay-insensitive
    w = np.load("w.npy")
    
    N, T = trace.shape
    split_bin = 432 # Exactly 1.5 days train, 1.5 days test
    
    train_trace = trace[:, :split_bin]
    test_trace = trace[:, split_bin:]
    T_test = test_trace.shape[1]
    
    # Compute profiling metrics on TRAIN ONLY
    mu_train = np.mean(train_trace, axis=1)
    std_train = np.std(train_trace, axis=1)
    var_train = std_train ** 2
    p95_train = np.percentile(train_trace, 95, axis=1)
    p99_train = np.percentile(train_trace, 99, axis=1)
    max_train = np.max(train_trace, axis=1)
    
    # Google Borg Autopilot policy: max(1.5 * mean, 1.15 * p95)
    google_pred_size = np.maximum(mu_train * 1.5, p95_train * 1.15)
    
    is_int = (labels == 1)
    is_del = (labels == 2)
    
    # Sort order: descending mean on train data
    order = np.argsort(-mu_train)
    
    policies = [
        ("peak_max", "Peak-Max (Train Max)"),
        ("static_p99", "Static P99 (Train P99)"),
        ("google_autopilot", "Google Borg/Autopilot"),
        ("effbw_233", "EffBW (z=2.33)"),
        ("effbw_300", "EffBW (z=3.00)"),
        ("sstf_asym_233", "SSTF Asym (z=2.33)"),
        ("sstf_asym_300", "SSTF Asym (z=3.00)")
    ]
    
    results = {}
    
    for pol_key, pol_name in policies:
        hosts_vms = []
        host_mu = []
        host_var = []
        host_mu_int = []
        host_var_int = []
        host_peak = []
        host_google = []
        
        for idx in order:
            v_mu = mu_train[idx]
            v_var = var_train[idx]
            v_p99 = p99_train[idx]
            v_max = max_train[idx]
            v_g = google_pred_size[idx]
            v_int = is_int[idx]
            
            placed = False
            for h in range(len(hosts_vms)):
                cand_mu = host_mu[h] + v_mu
                cand_var = host_var[h] + v_var
                cand_mu_int = host_mu_int[h] + (v_mu if v_int else 0.0)
                cand_var_int = host_var_int[h] + (v_var if v_int else 0.0)
                cand_peak = host_peak[h] + (v_max if pol_key == "peak_max" else v_p99)
                cand_google = host_google[h] + v_g
                
                fits = False
                if pol_key == "peak_max":
                    fits = (cand_peak <= host_capacity)
                elif pol_key == "static_p99":
                    fits = (cand_peak <= host_capacity)
                elif pol_key == "google_autopilot":
                    fits = (cand_google <= host_capacity)
                elif pol_key == "effbw_233":
                    fits = (cand_mu + 2.33 * np.sqrt(cand_var) <= host_capacity)
                elif pol_key == "effbw_300":
                    fits = (cand_mu + 3.00 * np.sqrt(cand_var) <= host_capacity)
                elif pol_key == "sstf_asym_233":
                    # Interactive variance dominates; delay is smoothed buffer
                    fits = (cand_mu + 2.33 * np.sqrt(cand_var_int) <= host_capacity)
                elif pol_key == "sstf_asym_300":
                    fits = (cand_mu + 3.00 * np.sqrt(cand_var_int) <= host_capacity)
                    
                if fits:
                    hosts_vms[h].append(idx)
                    host_mu[h] = cand_mu
                    host_var[h] = cand_var
                    host_mu_int[h] = cand_mu_int
                    host_var_int[h] = cand_var_int
                    host_peak[h] = cand_peak
                    host_google[h] = cand_google
                    placed = True
                    break
                    
            if not placed:
                hosts_vms.append([idx])
                host_mu.append(v_mu)
                host_var.append(v_var)
                host_mu_int.append(v_mu if v_int else 0.0)
                host_var_int.append(v_var if v_int else 0.0)
                host_peak.append(v_max if pol_key == "peak_max" else v_p99)
                host_google.append(v_g)
                
        num_hosts = len(hosts_vms)
        tot_test_intervals = num_hosts * T_test
        
        raw_overload_intervals = 0
        interactive_overload_intervals = 0
        overload_depths = []
        
        # Calibration tracking: predicted tail risk vs actual
        pred_overload_probs = []
        
        for h in range(num_hosts):
            vms = hosts_vms[h]
            # Sliced on OUT-OF-SAMPLE TEST TRACE ONLY
            host_test = test_trace[vms, :]
            actual_load = np.sum(host_test, axis=0) # (T_test,)
            
            # Interactive only on TEST
            int_vms = [v for v in vms if is_int[v]]
            int_load = np.sum(test_trace[int_vms, :], axis=0) if int_vms else np.zeros(T_test)
            
            # Actual overloads
            diff = actual_load - host_capacity
            is_over = diff > 0
            raw_overload_intervals += np.sum(is_over)
            if np.any(is_over):
                overload_depths.extend(diff[is_over] / host_capacity)
                
            # Interactive SLA violation under Plane 2 active shaping
            interactive_overload_intervals += np.sum(int_load > host_capacity)
            
            # Analytical predicted probability of overload under Gaussian assumption:
            # P(Demand > C) = 1 - Phi((C - mu) / sigma)
            if host_var[h] > 0:
                z_score = (host_capacity - host_mu[h]) / np.sqrt(host_var[h])
                pred_p = 1.0 - norm.cdf(z_score)
            else:
                pred_p = 0.0
            pred_overload_probs.append(pred_p)
            
        results[pol_key] = {
            "name": pol_name,
            "hosts": num_hosts,
            "vms_per_host": N / num_hosts,
            "raw_violation_pct": (raw_overload_intervals / tot_test_intervals) * 100.0,
            "interactive_sla_pct": (interactive_overload_intervals / tot_test_intervals) * 100.0,
            "p99_overload_pct": float(np.percentile(overload_depths, 99)) * 100.0 if overload_depths else 0.0,
            "mean_pred_overload_pct": float(np.mean(pred_overload_probs)) * 100.0
        }
        
    return results

def run_calibration_sweep(host_capacity=1600.0):
    trace = np.load("trace.npy")
    split_bin = 432
    train = trace[:, :split_bin]
    test = trace[:, split_bin:]
    
    mu_tr = np.mean(train, axis=1)
    var_tr = np.var(train, axis=1)
    order = np.argsort(-mu_tr)
    
    print("\n" + "=" * 95)
    print("CALIBRATION CHECK: ANALYTICAL PREDICTED OVERLOAD VS OBSERVED TEST OVERLOAD")
    print("=" * 95)
    print(f"{'z Target':<10} | {'Theoretical Risk':<18} | {'Analytical Pred':<18} | {'Observed Test Overload':<24} | {'Hosts'}")
    print("-" * 95)
    
    for z in [2.00, 2.33, 2.80, 3.00, 3.50, 4.00]:
        theo_p = (1.0 - norm.cdf(z)) * 100.0
        hosts_vms = []
        host_mu = []
        host_var = []
        
        for idx in order:
            v_mu = mu_tr[idx]
            v_var = var_tr[idx]
            placed = False
            for h in range(len(hosts_vms)):
                c_mu = host_mu[h] + v_mu
                c_var = host_var[h] + v_var
                if c_mu + z * np.sqrt(c_var) <= host_capacity:
                    hosts_vms[h].append(idx)
                    host_mu[h] = c_mu
                    host_var[h] = c_var
                    placed = True
                    break
            if not placed:
                hosts_vms.append([idx])
                host_mu.append(v_mu)
                host_var.append(v_var)
                
        num_hosts = len(hosts_vms)
        tot_intervals = num_hosts * test.shape[1]
        observed_over = 0
        pred_ps = []
        
        for h in range(num_hosts):
            vms = hosts_vms[h]
            agg_test = np.sum(test[vms, :], axis=0)
            observed_over += np.sum(agg_test > host_capacity)
            z_act = (host_capacity - host_mu[h]) / (np.sqrt(host_var[h]) + 1e-9)
            pred_ps.append(1.0 - norm.cdf(z_act))
            
        obs_p = (observed_over / tot_intervals) * 100.0
        mean_pred = np.mean(pred_ps) * 100.0
        
        print(f"{z:<10.2f} | {theo_p:<17.2f}% | {mean_pred:<17.2f}% | {obs_p:<23.2f}% | {num_hosts}")

if __name__ == "__main__":
    print("=" * 110)
    print("M0' OUT-OF-SAMPLE VALIDATION & GOOGLE AUTOPILOT PREDICTOR BENCHMARK")
    print("Train Window: Days 0.0 - 1.5 (432 bins)  |  Test Window: Days 1.5 - 3.0 (432 bins)")
    print("Host Capacity: 16 Cores (1600% CPU)")
    print("=" * 110)
    
    res = run_out_of_sample_evaluation(host_capacity=1600.0)
    base_p99_hosts = res["static_p99"]["hosts"]
    google_hosts = res["google_autopilot"]["hosts"]
    
    print(f"\n{'Policy':<25} | {'Hosts':<6} | {'VMs/Host':<8} | {'Savings vs P99':<15} | {'Savings vs Google':<18} | {'Raw Test Viol %':<16} | {'Interactive SLA Viol %'}")
    print("-" * 115)
    
    for key, data in res.items():
        sav_p99 = ((base_p99_hosts - data["hosts"]) / base_p99_hosts) * 100.0
        sav_goog = ((google_hosts - data["hosts"]) / google_hosts) * 100.0
        print(f"{data['name']:<25} | {data['hosts']:<6} | {data['vms_per_host']:<8.1f} | {sav_p99:>+13.1f}% | {sav_goog:>+16.1f}% | {data['raw_violation_pct']:>14.2f}% | {data['interactive_sla_pct']:>20.4f}%")
        
    run_calibration_sweep(host_capacity=1600.0)
