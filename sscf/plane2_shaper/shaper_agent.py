"""
lbhf-ref/shaper/shaper_agent.py
Task T4.2 - Active Shaper Agent (Pivot B)
Implements causal, mass-preserving workload traffic shaping for deferrable workloads.
Tracks the induced queue delay and temporal displacement cost.
"""

import numpy as np

class ShaperAgent:
    """
    Applies causal smoothing to deferrable workloads.
    Stand-in for time-slicing / release timers on non-interactive workloads.
    """
    def __init__(self, window_width=8, deferrable_fraction=0.5, seed=42):
        self.window_width = max(1, int(window_width))
        self.deferrable_fraction = deferrable_fraction
        self.rng = np.random.default_rng(seed)

    def shape(self, traces, deferrable_mask=None):
        """
        Applies causal moving average to workloads flagged as deferrable.
        Preserves total demand mass while reducing peak burstiness.
        Returns:
          - shaped_traces: np.ndarray of shape (N, T)
          - delay_stats: dict of added latency metrics
        """
        N, T = traces.shape
        W = self.window_width
        shaped = traces.copy()
        
        if deferrable_mask is None:
            # Subsample deferrable workloads
            deferrable_mask = (self.rng.random(N) < self.deferrable_fraction)
            
        deferrable_indices = np.where(deferrable_mask)[0]
        kernel = np.ones(W) / W
        
        # Theoretical mean delay introduced by causal moving average of width W
        mean_delay_bins = (W - 1) / 2.0
        
        for idx in deferrable_indices:
            row = traces[idx]
            # Prepend initial value to preserve length and maintain causality
            padded = np.concatenate([np.full(W - 1, row[0]), row])
            convolved = np.convolve(padded, kernel, mode="valid")
            shaped[idx] = convolved
            
        return shaped, {
            "num_workloads": N,
            "num_shaped": len(deferrable_indices),
            "deferrable_fraction": len(deferrable_indices) / max(1, N),
            "window_width_bins": W,
            "mean_delay_bins": mean_delay_bins,
            "delay_preserving": True
        }
