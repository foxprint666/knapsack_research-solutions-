"""
harmonic-knapsack/signal_processing/anc_fxlms_simulator.py
Signal Processing Model: Active Noise Cancellation (ANC) via Filtered-X LMS (FxLMS).

Demonstrates:
  1. Destructive wave interference: e(n) = d(n) - y_filtered(n).
  2. The FxLMS algorithm with primary path P(z) and secondary path S(z).
  3. The Strict Causality Constraint:
       tau_electronic < tau_acoustic (causal) vs tau_electronic > tau_acoustic (acausal failure).
  4. Gaussian vs. Impulsive Pareto Disturbance:
       LMS gradient explosion on Pareto bursts vs. Robust Normalized Sign-LMS.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class FxLMSSimulator:
    """
    Simulates Feedforward Active Noise Cancellation using Filtered-X LMS.
    """
    def __init__(self, filter_order: int = 32, mu: float = 0.005):
        self.filter_order = int(filter_order)
        self.mu = float(mu)

    def run_simulation(self, num_samples: int = 2000,
                       noise_type: str = "gaussian",
                       causal_delay_samples: int = 4,
                       use_sign_lms: bool = False,
                       seed: int = 42) -> Dict[str, Any]:
        """
        Runs ANC FxLMS simulation.
        
        Args:
          noise_type: 'gaussian' or 'pareto' (impulsive heavy-tailed bursts)
          causal_delay_samples: Positive integer = causal reference lead time;
                                Negative integer = acausal latency violation.
          use_sign_lms: Whether to use Normalized Sign-LMS for impulsive stability.
        """
        rng = np.random.default_rng(seed)
        N = num_samples
        L = self.filter_order

        # Generate reference noise x(n)
        if noise_type == "gaussian":
            # Bandpass-filtered stationary Gaussian noise
            raw_noise = rng.normal(0.0, 1.0, size=N)
            b = np.array([0.2, 0.5, 0.7, 0.5, 0.2])
            x = np.convolve(raw_noise, b, mode="same")
        elif noise_type == "pareto":
            # Impulsive Pareto bursts (heavy-tailed, alpha=1.5)
            pareto_samples = (rng.pareto(1.5, size=N) - 1.0) * 0.4
            sign_flip = rng.choice([-1.0, 1.0], size=N)
            x = pareto_samples * sign_flip
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        # Primary path P(z): physical acoustic delay + transfer response
        # Acoustic propagation delay = max(1, 8 + causal_delay_samples)
        p_delay = max(1, 8 + causal_delay_samples)
        p_fir = np.zeros(20)
        p_fir[p_delay] = 1.0
        p_fir[min(19, p_delay + 1)] = 0.4
        d = np.convolve(x, p_fir, mode="full")[:N]

        # Secondary path S(z): electronics + speaker + error mic
        s_delay = 8  # Fixed electronic group delay (ADC + DAC + speaker)
        s_fir = np.zeros(16)
        s_fir[s_delay] = 0.8
        s_fir[min(15, s_delay + 1)] = -0.3

        # Filtered reference signal x'(n) = S_hat(z) * x(n)
        s_hat = s_fir.copy()
        x_filtered = np.convolve(x, s_hat, mode="full")[:N]

        # Simulation state
        w = np.zeros(L, dtype=np.float64)
        y = np.zeros(N, dtype=np.float64)
        e = np.zeros(N, dtype=np.float64)

        x_buf = np.zeros(L, dtype=np.float64)
        x_filt_buf = np.zeros(L, dtype=np.float64)
        s_state = np.zeros(len(s_fir), dtype=np.float64)

        diverged = False

        for n in range(N):
            # Update tapped delay lines
            x_buf[1:] = x_buf[:-1]
            x_buf[0] = x[n]

            x_filt_buf[1:] = x_filt_buf[:-1]
            x_filt_buf[0] = x_filtered[n]

            # Generate anti-noise output: y(n) = w^T * x
            y_n = float(np.dot(w, x_buf))
            y[n] = y_n

            # Anti-noise passes through physical secondary path S(z)
            s_state[1:] = s_state[:-1]
            s_state[0] = y_n
            y_at_mic = float(np.dot(s_fir, s_state))

            # Error microphone: e(n) = d(n) - y_at_mic(n)
            e_n = d[n] - y_at_mic
            e[n] = e_n

            # Weight update
            norm_factor = np.dot(x_filt_buf, x_filt_buf) + 1e-4
            if use_sign_lms:
                # Normalized Sign-LMS (robust to impulsive Pareto bursts)
                grad = np.sign(e_n) * (x_filt_buf / norm_factor)
            else:
                # Standard FxLMS (MSE gradient)
                grad = e_n * x_filt_buf

            w += self.mu * grad

            # Check for filter weight explosion / divergence
            if np.any(np.abs(w) > 50.0) or np.any(np.isnan(w)):
                diverged = True
                break

        # Performance evaluation (steady state: second half of trace)
        steady_slice = slice(N // 2, N)
        initial_power = float(np.mean(d[steady_slice] ** 2))
        residual_power = float(np.mean(e[steady_slice] ** 2)) if not diverged else float('inf')
        
        attenuation_db = 10.0 * np.log10(initial_power / max(1e-12, residual_power)) if not diverged else -99.9

        return {
            "num_samples": N,
            "noise_type": noise_type,
            "causal_delay_samples": causal_delay_samples,
            "is_causal": causal_delay_samples >= 0,
            "diverged": diverged,
            "attenuation_db": attenuation_db,
            "initial_noise_power": initial_power,
            "residual_error_power": residual_power,
            "use_sign_lms": use_sign_lms
        }
