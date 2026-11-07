"""
lbhf-ref/sim/eval_harness.py
Task T0.3 - Baseline Simulator & Effective-Bandwidth Scheduling Engine
Implements placement schedulers (Random, LeastLoaded, Po2C, StaticP99, EffectiveBandwidth)
and queueing evaluation (Lindley queue with nonlinear tail saturation).
"""

import numpy as np

class LindleyQueueSimulator:
    """
    Simulates physical rack queue dynamics under multi-tenant load.
    Includes quadratic degradation when utilization exceeds 85% to model tail-latency cliffs.
    """
    @staticmethod
    def simulate(rack_loads, capacity, penalty_threshold=0.85, penalty_factor=0.5):
        n_racks, T = rack_loads.shape
        queues = np.zeros((n_racks, T), dtype=np.float64)
        for r in range(n_racks):
            q = 0.0
            for t in range(T):
                load = rack_loads[r, t]
                u = load / (capacity + 1e-9)
                if u > penalty_threshold:
                    eff_load = load * (1.0 + penalty_factor * ((u - penalty_threshold) / (1.0 - penalty_threshold))**2)
                else:
                    eff_load = load
                q = max(0.0, q + eff_load - capacity)
                queues[r, t] = q
        
        return {
            "p99_backlog": float(np.percentile(queues, 99)),
            "p95_backlog": float(np.percentile(queues, 95)),
            "mean_backlog": float(np.mean(queues)),
            "max_backlog": float(np.max(queues)),
            "queues": queues
        }

class PlacementScheduler:
    """
    Candidate placement algorithms for cluster hosts/racks.
    """
    def __init__(self, n_racks, target_util=0.88, seed=42):
        self.n_racks = n_racks
        self.target_util = target_util
        self.rng = np.random.default_rng(seed)

    def compute_capacity(self, traces):
        total_mean_demand = np.sum(np.mean(traces, axis=1))
        return (total_mean_demand / self.n_racks) / self.target_util

    def place(self, traces, method="effbw", warmup_bins=256, labels=None):
        n_workloads, T = traces.shape
        warmup_bins = min(warmup_bins, T // 4)
        train = traces[:, :warmup_bins]
        test = traces[:, warmup_bins:]
        T_test = test.shape[1]
        
        capacity = self.compute_capacity(traces)
        rack_loads = np.zeros((self.n_racks, T_test), dtype=np.float64)
        
        # Rack statistics
        dc = np.zeros(self.n_racks)
        dc_interactive = np.zeros(self.n_racks)
        var = np.zeros(self.n_racks)
        peaks = np.zeros(self.n_racks)
        
        if labels is None:
            labels = np.zeros(n_workloads, dtype=np.int8)
            
        # Order by descending mean demand
        order = np.argsort(-np.mean(train, axis=1)) if method != "po2c" else np.arange(n_workloads)
        
        for idx in order:
            mu = float(np.mean(train[idx]))
            v = float(np.var(train[idx]))
            p99 = float(np.percentile(train[idx], 99))
            is_int = bool(labels[idx] == 1)
            
            if method == "random":
                b = self.rng.integers(self.n_racks)
            elif method == "least_loaded":
                b = int(np.argmin(dc))
            elif method == "po2c":
                c1, c2 = self.rng.choice(self.n_racks, size=2, replace=False)
                b = c1 if dc[c1] <= dc[c2] else c2
            elif method == "static_p99":
                b = int(np.argmin(peaks))
                peaks[b] += p99
            elif method == "effbw":
                # Statistical Multiplexing: mu + 2.33 * sigma (99% Gaussian confidence limit)
                scores = (dc + mu) + 2.33 * np.sqrt(var + v)
                b = int(np.argmin(scores))
            elif method == "mix_aware":
                # Dual-constraint: target 70% total util, 30% interactive util
                # Find racks where adding mu does not exceed capacity, prioritizing least loaded
                valid = np.where((dc + mu <= capacity * self.target_util) & 
                                 (dc_interactive + (mu if is_int else 0.0) <= capacity * 0.30))[0]
                if len(valid) > 0:
                    b = valid[np.argmin(dc[valid])]
                else:
                    b = int(np.argmin(dc))
            else:
                raise ValueError(f"Unknown placement method: {method}")
                
            dc[b] += mu
            if is_int:
                dc_interactive[b] += mu
            var[b] += v
            rack_loads[b] += test[idx]
            
        metrics = LindleyQueueSimulator.simulate(rack_loads, capacity)
        metrics["capacity_per_rack"] = capacity
        metrics["target_utilization"] = self.target_util
        metrics["method"] = method
        return metrics, rack_loads
