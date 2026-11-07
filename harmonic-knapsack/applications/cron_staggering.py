"""
harmonic-knapsack/applications/cron_staggering.py
Application Model: Thundering-Herd Elimination via Cron / Batch Job Staggering.

Models:
  - Thousands of enterprise servers running recurring hourly cron jobs (backups, cache warming).
  - Top-of-the-hour coincidence (all nodes firing at minute 00) vs. Modular Hash Staggering.
  - Concurrency reduction, database connection pool protection, and network egress smoothing.
"""

import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class CronStaggerSimulator:
    """
    Simulates large-scale server fleet cron execution across a 60-minute cycle.
    """
    def __init__(self, num_nodes: int = 5000,
                 cycle_minutes: int = 60,
                 job_duration_minutes: float = 2.0,
                 egress_per_job_mbps: float = 100.0):
        self.num_nodes = int(num_nodes)
        self.cycle_minutes = int(cycle_minutes)
        self.job_duration_minutes = float(job_duration_minutes)
        self.egress_per_job_mbps = float(egress_per_job_mbps)

    def run_comparison(self, seed: int = 42) -> Dict[str, Any]:
        """
        Compares:
          1. Naive Synchronized Cron: all nodes execute at minute 0.
          2. Uniform Hash-Staggered: offset = hash(node_id) % cycle_minutes.
        """
        # Time grid in 1-second resolution over the 60-minute cycle
        total_seconds = self.cycle_minutes * 60
        duration_seconds = int(round(self.job_duration_minutes * 60))
        
        # 1. Naive Synchronized: all start at second 0
        naive_concurrency = np.zeros(total_seconds, dtype=np.int32)
        naive_concurrency[:duration_seconds] = self.num_nodes
        naive_peak_concurrency = self.num_nodes
        naive_peak_egress_gbps = (naive_peak_concurrency * self.egress_per_job_mbps) / 1000.0

        # 2. Hash-Staggered: offset based on SHA256 of hostname
        staggered_concurrency = np.zeros(total_seconds, dtype=np.int32)
        node_start_seconds = []
        for i in range(self.num_nodes):
            h = int(hashlib.sha256(f"node_{i:06d}.cluster.local".encode()).hexdigest(), 16)
            start_sec = h % total_seconds
            node_start_seconds.append(start_sec)
            
            # Add job duration window (with wrap-around modulo cycle)
            for s in range(duration_seconds):
                t_idx = (start_sec + s) % total_seconds
                staggered_concurrency[t_idx] += 1

        staggered_peak_concurrency = int(np.max(staggered_concurrency))
        staggered_mean_concurrency = float(np.mean(staggered_concurrency))
        staggered_peak_egress_gbps = (staggered_peak_concurrency * self.egress_per_job_mbps) / 1000.0

        # Theoretical ideal expectation: N * (duration / period)
        theoretical_mean = self.num_nodes * (self.job_duration_minutes / self.cycle_minutes)
        concurrency_reduction_pct = ((naive_peak_concurrency - staggered_peak_concurrency) / naive_peak_concurrency) * 100.0

        return {
            "num_nodes": self.num_nodes,
            "cycle_minutes": self.cycle_minutes,
            "job_duration_minutes": self.job_duration_minutes,
            "naive_synchronized": {
                "peak_concurrency": naive_peak_concurrency,
                "peak_egress_gbps": naive_peak_egress_gbps
            },
            "hash_staggered": {
                "peak_concurrency": staggered_peak_concurrency,
                "mean_concurrency": staggered_mean_concurrency,
                "theoretical_mean": theoretical_mean,
                "peak_egress_gbps": staggered_peak_egress_gbps
            },
            "concurrency_reduction_pct": concurrency_reduction_pct
        }
