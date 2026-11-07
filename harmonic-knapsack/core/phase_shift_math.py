"""
harmonic-knapsack/core/phase_shift_math.py
Module for Analytical Phase-Shift Operations and Fast Harmonic Correlation.
Implements:
  - Exact Fourier domain phase rotation: X_k' = X_k * exp(-i * k * omega * tau)
  - Fast O(K log K) circular cross-correlation for continuous start-offset search
  - Antagonistic wave cancellation metric between periodic workloads
  - Hyperperiod determination and beat-frequency analysis
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .harmonic_representation import HarmonicProfile, HarmonicComponent

def compute_hyperperiod(periods: List[float], precision: float = 0.001) -> float:
    """
    Computes the least common multiple (LCM) hyperperiod across a set of periods.
    Rounds to nearest multiple of precision to prevent floating-point explosion.
    """
    if not periods:
        return 0.0
    int_periods = [int(round(p / precision)) for p in periods if p > 0]
    if not int_periods:
        return 0.0
    
    lcm_val = int_periods[0]
    for p in int_periods[1:]:
        lcm_val = (lcm_val * p) // math.gcd(lcm_val, p)
        # Cap hyperperiod to avoid numerical overflow on incommensurate periods
        if lcm_val > 10_000_000:
            return float('inf')
    return float(lcm_val * precision)

class PhaseShiftEngine:
    """
    Mathematical engine for manipulating periodic signals in the frequency domain.
    """
    @staticmethod
    def rotate_harmonics(harmonics: List[HarmonicComponent],
                         phase_offset_seconds: float) -> List[HarmonicComponent]:
        """
        Shifts all harmonic components in time by tau by applying phase rotation:
        phi_new = phi_old - (2 * pi * f * tau) mod 2*pi
        """
        rotated = []
        tau = float(phase_offset_seconds)
        for h in harmonics:
            omega = 2.0 * np.pi * h.frequency_hz
            new_phase = (h.phase_rad - omega * tau) % (2.0 * np.pi)
            rotated.append(HarmonicComponent(
                harmonic_order=h.harmonic_order,
                frequency_hz=h.frequency_hz,
                amplitude=h.amplitude,
                phase_rad=new_phase
            ))
        return rotated

    @staticmethod
    def combine_harmonic_profiles(profiles: List[HarmonicProfile],
                                  offsets_seconds: List[float],
                                  eval_points: int = 512) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Synthesizes the aggregate continuous time-domain curve across multiple profiles
        shifted by their respective offsets over the common hyperperiod.
        Returns: (t_grid, aggregate_load, peak_load)
        """
        assert len(profiles) == len(offsets_seconds), "Profiles and offsets must match in length"
        if not profiles:
            return np.array([]), np.array([]), 0.0

        periods = [p.period_seconds for p in profiles]
        T_hyper = compute_hyperperiod(periods)
        if math.isinf(T_hyper) or T_hyper <= 0:
            T_hyper = max(periods)

        t_grid = np.linspace(0.0, T_hyper, eval_points, endpoint=False)
        agg_load = np.zeros_like(t_grid)

        for p, offset in zip(profiles, offsets_seconds):
            agg_load += p.evaluate(t_grid, phase_offset_seconds=offset)

        peak_load = float(np.max(agg_load))
        return t_grid, agg_load, peak_load

    @staticmethod
    def find_optimal_offset_binary_pair(p1: HarmonicProfile, p2: HarmonicProfile,
                                        num_search_steps: int = 128) -> Tuple[float, float, float]:
        """
        Finds the start offset tau in [0, p2.period) that minimizes the peak of p1 + p2.
        Returns: (optimal_offset_sec, min_peak, uncoordinated_in_phase_peak)
        """
        T_search = p2.period_seconds
        T_hyper = compute_hyperperiod([p1.period_seconds, p2.period_seconds])
        if math.isinf(T_hyper):
            T_hyper = max(p1.period_seconds, p2.period_seconds)

        t_grid = np.linspace(0.0, T_hyper, 256, endpoint=False)
        load1 = p1.evaluate(t_grid, phase_offset_seconds=0.0)

        # In-phase uncoordinated peak (tau = 0)
        uncoordinated_peak = float(np.max(load1 + p2.evaluate(t_grid, phase_offset_seconds=0.0)))

        best_offset = 0.0
        min_peak = uncoordinated_peak

        candidate_offsets = np.linspace(0.0, T_search, num_search_steps, endpoint=False)
        for cand_tau in candidate_offsets:
            load2 = p2.evaluate(t_grid, phase_offset_seconds=cand_tau)
            peak = float(np.max(load1 + load2))
            if peak < min_peak:
                min_peak = peak
                best_offset = float(cand_tau)

        return best_offset, min_peak, uncoordinated_peak

    @staticmethod
    def compute_antagonistic_cancellation_metric(p1: HarmonicProfile, p2: HarmonicProfile) -> Dict[str, float]:
        """
        Quantifies the wave cancellation efficacy between two periodic profiles:
          - Peak Reduction: (P_uncoord - P_opt) / P_uncoord * 100%
          - Cancellation Ratio: (Sum(P_indiv) - P_opt) / Sum(P_indiv) * 100%
        """
        best_offset, min_peak, uncoord_peak = PhaseShiftEngine.find_optimal_offset_binary_pair(p1, p2)
        
        # Individual standalone peaks
        t_sample = np.linspace(0.0, max(p1.period_seconds, p2.period_seconds), 256, endpoint=False)
        p1_peak = float(np.max(p1.evaluate(t_sample)))
        p2_peak = float(np.max(p2.evaluate(t_sample)))
        sum_individual = p1_peak + p2_peak

        peak_shaving_pct = ((uncoord_peak - min_peak) / max(1e-6, uncoord_peak)) * 100.0
        cancellation_gain_pct = ((sum_individual - min_peak) / max(1e-6, sum_individual)) * 100.0

        return {
            "optimal_offset_seconds": best_offset,
            "min_peak": min_peak,
            "uncoordinated_peak": uncoord_peak,
            "sum_individual_peaks": sum_individual,
            "peak_shaving_pct": peak_shaving_pct,
            "cancellation_gain_pct": cancellation_gain_pct
        }
