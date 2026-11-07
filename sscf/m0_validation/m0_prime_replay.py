"""
m0_prime_replay.py
Tests whether Effective Bandwidth packing saves physical hosts on the REAL Azure trace
at an acceptable SLA violation rate (the M0' Replay Test).

Evaluates 4 placement policies across the 4,994 real Azure VMs (864 bins = 3 full days):
1. Static P99 Reservation: Sum(P99_i) <= Capacity
2. Peak-Load Reservation (Worst-case provisioning): Sum(Max_i) <= Capacity
3. Standard Effective Bandwidth (Full Fleet): Sum(mu_i) + 2.33 * sqrt(Sum(sigma_i^2)) <= Capacity
4. Asymmetric SSTF Effective Bandwidth (Interactive variance only, deferrable buffer):
   Sum(mu_int) + 0.95 * Sum(mu_delay) + 2.33 * sqrt(Sum(sigma_int^2)) <= Capacity

Measures:
- Host count required (and host savings relative to static reservations)
- Fleet average packing density (VMs per host)
- Measured violation rate: % of 5-min intervals where actual aggregate demand exceeds host capacity
- Tail violation depth: P99 and Max overload (% above host capacity)
- Interactive SLA violation rate under Plane 2 active shaping priority
"""

import sys
import numpy as np

def run_m0_prime_replay(host_capacity=800.0, target_violation_budget=0.01):
    """
    Simulates packing 4,994 real Azure VMs onto hosts of given capacity.
    host_capacity = 800.0 corresponds to an 8-core host (each core is 100% CPU, so 800% capacity)
    or an 8x normalized host slice.
    """
    trace = np.load("trace.npy")       # (4994, 864)
    weights = np.load("w.npy")         # (4994,)
    labels = np.load("labels.npy")     # 1 = Interactive, 2 = Delay-insensitive
    
    N, T = trace.shape
    
    # Precompute per-VM statistics
    mu = np.mean(trace, axis=1)
    sigma = np.std(trace, axis=1)
    var = sigma ** 2
    p99 = np.percentile(trace, 99, axis=1)
    p_max = np.max(trace, axis=1)
    
    is_interactive = (labels == 1)
    is_delay = (labels == 2)
    
    # Sort order: descending mean demand (standard Bin-Packing Best-Fit Decreasing heuristic)
    order = np.argsort(-mu)
    
    policies = ["peak_max", "static_p99", "effbw_standard", "sstf_asymmetric"]
    results = {}
    
    for pol in policies:
        # Each host tracks:
        # - list of VM indices
        # - allocated metric
        # - total mu
        # - total var (or interactive var)
        hosts_vms = []
        host_mu = []
        host_mu_int = []
        host_mu_delay = []
        host_var = []
        host_var_int = []
        host_peak = []
        
        for idx in order:
            v_mu = mu[idx]
            v_var = var[idx]
            v_p99 = p99[idx]
            v_max = p_max[idx]
            v_int = is_interactive[idx]
            
            placed = False
            # Try to place on existing host
            for h in range(len(hosts_vms)):
                cand_mu = host_mu[h] + v_mu
                cand_mu_int = host_mu_int[h] + (v_mu if v_int else 0.0)
                cand_mu_delay = host_mu_delay[h] + (0.0 if v_int else v_mu)
                cand_var = host_var[h] + v_var
                cand_var_int = host_var_int[h] + (v_var if v_int else 0.0)
                cand_peak = host_peak[h] + (v_max if pol == "peak_max" else v_p99)
                
                # Check capacity condition
                fits = False
                if pol == "peak_max":
                    fits = (cand_peak <= host_capacity)
                elif pol == "static_p99":
                    fits = (cand_peak <= host_capacity)
                elif pol == "effbw_standard":
                    # Sum(mu) + 2.33 * sqrt(Sum(var)) <= Capacity
                    eff_demand = cand_mu + 2.33 * np.sqrt(cand_var)
                    fits = (eff_demand <= host_capacity)
                elif pol == "sstf_asymmetric":
                    # Plane 1: Interactive variance dominates; delay-insensitive runs as buffer
                    eff_demand = cand_mu_int + cand_mu_delay + 2.33 * np.sqrt(cand_var_int)
                    fits = (eff_demand <= host_capacity)
                    
                if fits:
                    hosts_vms[h].append(idx)
                    host_mu[h] = cand_mu
                    host_mu_int[h] = cand_mu_int
                    host_mu_delay[h] = cand_mu_delay
                    host_var[h] = cand_var
                    host_var_int[h] = cand_var_int
                    host_peak[h] = cand_peak
                    placed = True
                    break
                    
            if not placed:
                # Open a new host
                hosts_vms.append([idx])
                host_mu.append(v_mu)
                host_mu_int.append(v_mu if v_int else 0.0)
                host_mu_delay.append(0.0 if v_int else v_mu)
                host_var.append(v_var)
                host_var_int.append(v_var if v_int else 0.0)
                host_peak.append(v_max if pol == "peak_max" else v_p99)
                
        num_hosts = len(hosts_vms)
        
        # Now replay the real 864-bin trace across every host and measure violations
        total_intervals = num_hosts * T
        overflow_intervals = 0
        overflow_amounts = []
        interactive_violations_under_shaping = 0
        
        for h in range(num_hosts):
            vms_in_host = hosts_vms[h]
            host_trace = trace[vms_in_host, :] # (vms, T)
            aggregate_load = np.sum(host_trace, axis=0) # (T,)
            
            # Interactive load alone
            int_vms = [v for v in vms_in_host if is_interactive[v]]
            if len(int_vms) > 0:
                int_load = np.sum(trace[int_vms, :], axis=0)
            else:
                int_load = np.zeros(T)
                
            # Unshaped host overflow
            over = aggregate_load - host_capacity
            is_over = over > 0
            overflow_intervals += np.sum(is_over)
            if np.any(is_over):
                overflow_amounts.extend(over[is_over] / host_capacity)
                
            # Plane 2 Active Shaping: Can the host absorb the burst by throttling delay-insensitive VMs?
            # If int_load > host_capacity, interactive SLA is violated regardless of shaping
            int_over = int_load > host_capacity
            interactive_violations_under_shaping += np.sum(int_over)
            
        violation_rate = overflow_intervals / max(1, total_intervals)
        p99_overflow = float(np.percentile(overflow_amounts, 99)) if len(overflow_amounts) > 0 else 0.0
        max_overflow = float(np.max(overflow_amounts)) if len(overflow_amounts) > 0 else 0.0
        sla_violation_rate = interactive_violations_under_shaping / max(1, total_intervals)
        
        results[pol] = {
            "hosts_needed": num_hosts,
            "vms_per_host": N / num_hosts,
            "raw_violation_rate": violation_rate,
            "p99_overflow_pct": p99_overflow * 100.0,
            "max_overflow_pct": max_overflow * 100.0,
            "interactive_sla_violation_rate": sla_violation_rate,
        }
        
    return results

if __name__ == "__main__":
    print("=" * 80)
    print("M0' REPLAY: REAL AZURE TRACE EFFECTIVE-BANDWIDTH HOST PACKING VALIDATION")
    print("=" * 80)
    # Test across multiple host slice capacities:
    # 800 (e.g. 8 cores @ 100%), 1600 (16 cores), 3200 (32 cores)
    for cap in [800.0, 1600.0, 3200.0]:
        cores = int(cap / 100)
        print(f"\n--- HOST CAPACITY: {cap:.0f}% CPU (~{cores} Cores per Host) ---")
        res = run_m0_prime_replay(host_capacity=cap)
        base_hosts = res["static_p99"]["hosts_needed"]
        
        print(f"{'Policy':<22} | {'Hosts':<6} | {'VMs/Host':<8} | {'Host Savings':<12} | {'Raw Viol %':<10} | {'Int SLA Viol %':<14} | {'P99 Overload'}")
        print("-" * 100)
        for pol, data in res.items():
            savings = ((base_hosts - data["hosts_needed"]) / base_hosts) * 100.0
            print(f"{pol:<22} | {data['hosts_needed']:<6} | {data['vms_per_host']:<8.1f} | {savings:>+10.1f}% | {data['raw_violation_rate']*100:>9.2f}% | {data['interactive_sla_violation_rate']*100:>13.4f}% | +{data['p99_overflow_pct']:.1f}%")
