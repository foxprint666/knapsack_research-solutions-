"""
harmonic-knapsack/applications/ai_training_power_cap.py
Application Model: Co-Located AI Training Jobs Under a Shared Power Cap.

Based on:
  - Brieuc Le Roux Tardif, "Do Co-Located AI Training Jobs Synchronize?
    Load-Dependent Throttling as a Coupling Mechanism for Phase-Locking
    Behind a Shared Power Cap" (arXiv:2607.19638).
  - Choukse et al., "Power Stabilization for AI Training Datacenters"
    (Microsoft / OpenAI / NVIDIA, arXiv:2508.14318).

Simulates:
  - Synchronous mini-batch SGD iterations (Compute phase @ high wattage, AllReduce @ low wattage).
  - Emergent phase-locking synchronization under shared power constraint.
  - Phase-Scattering Scheduling (scattering phases across the iteration cycle).
  - Phase Drift over time and Closed-Loop Phase Re-centering.
"""

import math
from typing import List, Dict, Any, Optional
import numpy as np
from core.harmonic_representation import HarmonicProfile, HarmonicComponent
from core.phase_shift_math import PhaseShiftEngine

class AITrainingJob:
    """Represents a distributed DNN training job running on an accelerator rack."""
    def __init__(self, job_id: int, name: str,
                 iteration_period_ms: float = 1000.0,
                 compute_duration_ms: float = 750.0,
                 compute_power_kw: float = 6.0,
                 comm_power_kw: float = 1.5):
        self.job_id = job_id
        self.name = name
        self.iteration_period_sec = iteration_period_ms / 1000.0
        self.compute_duration_sec = compute_duration_ms / 1000.0
        self.compute_power_kw = float(compute_power_kw)
        self.comm_power_kw = float(comm_power_kw)
        
        # Build harmonic profile
        duty_cycle = self.compute_duration_sec / self.iteration_period_sec
        dc_mean = (duty_cycle * self.compute_power_kw) + ((1.0 - duty_cycle) * self.comm_power_kw)
        pulse = self.compute_power_kw - self.comm_power_kw
        f0 = 1.0 / self.iteration_period_sec
        
        harmonics = []
        for k in range(1, 6):
            amp = 2.0 * (pulse / (k * np.pi)) * np.abs(np.sin(k * np.pi * duty_cycle))
            harmonics.append(HarmonicComponent(
                harmonic_order=k,
                frequency_hz=k * f0,
                amplitude=amp,
                phase_rad=0.0
            ))
            
        self.profile = HarmonicProfile(
            period_seconds=self.iteration_period_sec,
            dc_mean=dc_mean,
            harmonics=harmonics,
            periodicity_confidence=0.99
        )


class AIPowerCapScheduler:
    """
    Schedules and coordinates AI training execution phases to flatten aggregate power draw.
    """
    def __init__(self, rack_power_cap_kw: float = 38.0):
        self.rack_power_cap_kw = float(rack_power_cap_kw)

    def evaluate_phase_scattering(self, jobs: List[AITrainingJob]) -> Dict[str, Any]:
        """
        Compares:
          1. Phase-Locked / Uncoordinated In-Phase Alignment (Worst Case)
          2. Phase-Scattering Scheduling (Uniform Phase Distribution)
        """
        M = len(jobs)
        assert M > 0, "Need at least one job"
        profiles = [j.profile for j in jobs]
        period = jobs[0].iteration_period_sec

        # 1. Synchronized / Phase-Locked alignment (offsets = 0)
        zero_offsets = [0.0] * M
        _, uncoord_curve, uncoord_peak = PhaseShiftEngine.combine_harmonic_profiles(
            profiles, zero_offsets, eval_points=500
        )

        # 2. Phase-Scattering: distribute offsets uniformly tau_k = k * (T / M)
        scattered_offsets = [k * (period / M) for k in range(M)]
        _, scattered_curve, scattered_peak = PhaseShiftEngine.combine_harmonic_profiles(
            profiles, scattered_offsets, eval_points=500
        )

        peak_reduction_pct = ((uncoord_peak - scattered_peak) / uncoord_peak) * 100.0

        return {
            "num_jobs": M,
            "power_cap_kw": self.rack_power_cap_kw,
            "uncoordinated": {
                "peak_power_kw": uncoord_peak,
                "trips_power_cap": uncoord_peak > self.rack_power_cap_kw,
                "excess_kw": max(0.0, uncoord_peak - self.rack_power_cap_kw)
            },
            "phase_scattered": {
                "peak_power_kw": scattered_peak,
                "trips_power_cap": scattered_peak > self.rack_power_cap_kw,
                "excess_kw": max(0.0, scattered_peak - self.rack_power_cap_kw),
                "offsets_ms": [tau * 1000.0 for tau in scattered_offsets]
            },
            "peak_reduction_pct": peak_reduction_pct
        }

    def simulate_phase_drift_and_recentering(self, jobs: List[AITrainingJob],
                                            num_iterations: int = 200,
                                            jitter_std_ms: float = 8.0,
                                            enable_closed_loop: bool = True,
                                            recenter_interval: int = 25,
                                            seed: int = 42) -> Dict[str, Any]:
        """
        Simulates long-term execution where network jitter causes open-loop phase drift,
        and demonstrates how closed-loop phase re-centering maintains stability.
        """
        rng = np.random.default_rng(seed)
        M = len(jobs)
        period_ms = jobs[0].iteration_period_sec * 1000.0
        
        # Initial scattered offsets in ms
        offsets_ms = np.array([k * (period_ms / M) for k in range(M)])
        
        peak_history = []
        drift_max_history = []
        
        for it in range(num_iterations):
            # Accumulate random timing jitter per job
            jitter = rng.normal(0.0, jitter_std_ms, size=M)
            offsets_ms = (offsets_ms + jitter) % period_ms

            # Closed-loop re-centering check
            if enable_closed_loop and (it > 0) and (it % recenter_interval == 0):
                # Re-align phases toward uniform spacing
                ideal_offsets = np.array([k * (period_ms / M) for k in range(M)])
                # Nudge phases by 50% toward ideal to model gentle control correction
                offsets_ms = (0.5 * offsets_ms + 0.5 * ideal_offsets) % period_ms

            # Evaluate instantaneous peak power at this iteration
            offsets_sec = [t / 1000.0 for t in offsets_ms]
            _, _, peak_kw = PhaseShiftEngine.combine_harmonic_profiles(
                [j.profile for j in jobs], offsets_sec, eval_points=250
            )
            peak_history.append(peak_kw)
            
            # Phase clustering metric: variance of inter-job spacing
            sorted_offs = np.sort(offsets_ms)
            spacings = np.diff(np.append(sorted_offs, sorted_offs[0] + period_ms))
            drift_max_history.append(float(np.std(spacings)))

        return {
            "num_iterations": num_iterations,
            "enable_closed_loop": enable_closed_loop,
            "mean_peak_power_kw": float(np.mean(peak_history)),
            "max_peak_power_kw": float(np.max(peak_history)),
            "power_cap_kw": self.rack_power_cap_kw,
            "cap_violation_count": int(np.sum(np.array(peak_history) > self.rack_power_cap_kw)),
            "peak_history": peak_history,
            "drift_history": drift_max_history
        }
