"""
harmonic-knapsack/production/rate_diversity_controller.py
Pillar 3: Rate-Diversity & Higher-Harmonic Handling (The Controller).

Addresses Tardif's beat frequency phenomenon (arXiv:2607.19638):
When jobs have disparate iteration periods (e.g. 800ms vs 1100ms),
their phases constantly beat against each other, periodically causing
destructive peak collisions.

Provides two production-grade mechanisms:
1. Harmonic Period Quantization & Clustering (co-locating harmonic jobs).
2. Dynamic Mini-Batch Pacing (synthesizing stationary harmonic relationships
   to eliminate beat frequencies without false harmonization claims).
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from core.phase_shift_math import compute_rational_hyperperiod

class HarmonicPeriodQuantizer:
    """
    Groups disparate jobs into harmonic frequency families to prevent hyperperiod explosion
    and eliminate cross-frequency beat phenomena.
    """
    def __init__(self, base_period_families: Optional[List[float]] = None):
        # Default standard harmonic base periods in seconds (e.g. 0.5s, 1.0s, 2.0s)
        self.base_period_families = base_period_families or [0.5, 1.0, 2.0, 4.0]

    def find_nearest_harmonic_target(self, natural_period_sec: float) -> Tuple[float, float]:
        """
        Finds the closest harmonic family period T_target and returns (T_target, stretch_ratio).
        """
        best_target = self.base_period_families[0]
        min_err = float('inf')

        for target in self.base_period_families:
            err = abs(natural_period_sec - target)
            if err < min_err:
                min_err = err
                best_target = target

        stretch_ratio = best_target / natural_period_sec
        return best_target, stretch_ratio

    def cluster_jobs_by_harmonic_family(self, job_periods: Dict[int, float],
                                        tolerance_pct: float = 25.0) -> Dict[float, List[int]]:
        """
        Clusters a fleet of jobs into compatible harmonic families.
        Jobs within the same family can be phase-staggered without beat collisions.
        """
        clusters: Dict[float, List[int]] = {base: [] for base in self.base_period_families}
        clusters[-1.0] = [] # unclustered / anomalous bucket

        for jid, period in job_periods.items():
            target, ratio = self.find_nearest_harmonic_target(period)
            pct_diff = abs(ratio - 1.0) * 100.0
            if pct_diff <= tolerance_pct:
                clusters[target].append(jid)
            else:
                clusters[-1.0].append(jid)

        return {k: v for k, v in clusters.items() if v}


class DynamicBatchPacer:
    """
    Computes precise micro-delays or gradient accumulation pacing adjustments
    to lock non-harmonic jobs into an exact, stationary harmonic frequency.
    """
    @staticmethod
    def calculate_beat_frequency(period_a_sec: float, period_b_sec: float) -> Dict[str, float]:
        """
        Calculates the beat frequency f_beat = |f_A - f_B| and the recurrence
        interval at which out-of-phase jobs collide.
        Uses exact rational hyperperiod computation.
        """
        f_a = 1.0 / period_a_sec
        f_b = 1.0 / period_b_sec
        f_beat = abs(f_a - f_b)
        t_beat = 1.0 / f_beat if f_beat > 1e-6 else float('inf')
        t_hyper, is_exact, err_bound = compute_rational_hyperperiod([period_a_sec, period_b_sec])

        return {
            "freq_a_hz": f_a,
            "freq_b_hz": f_b,
            "beat_frequency_hz": f_beat,
            "beat_recurrence_sec": t_beat,
            "hyperperiod_sec": t_hyper,
            "is_exact_hyperperiod": float(is_exact)
        }

    @staticmethod
    def compute_pacing_adjustment(period_a_sec: float, period_b_sec: float,
                                  max_slowdown_pct: float = 15.0) -> Dict[str, Any]:
        """
        Determines whether pacing the faster job (or slightly pacing both) can lock
        them into a common period or integer harmonic ratio (1:1, 1:2, 2:3).
        """
        # Case 1: Direct 1:1 synchronization by pacing faster job to match slower job
        t_slow = max(period_a_sec, period_b_sec)
        t_fast = min(period_a_sec, period_b_sec)
        
        slowdown_pct = ((t_slow - t_fast) / t_fast) * 100.0
        
        if slowdown_pct <= max_slowdown_pct:
            return {
                "strategy": "direct_1_to_1_lock",
                "unified_period_sec": t_slow,
                "fast_job_delay_per_step_ms": (t_slow - t_fast) * 1000.0,
                "slow_job_delay_per_step_ms": 0.0,
                "slowdown_percentage": slowdown_pct,
                "beat_eliminated": True
            }

        # Case 2: 2:1 harmonic ratio (e.g. Job A @ 0.5s, Job B @ 1.0s)
        ratio_2_1_target = 2.0 * t_fast
        ratio_2_1_slowdown = ((ratio_2_1_target - t_slow) / t_slow) * 100.0
        if 0.0 <= ratio_2_1_slowdown <= max_slowdown_pct:
            return {
                "strategy": "harmonic_2_to_1_lock",
                "unified_period_sec": ratio_2_1_target,
                "fast_job_delay_per_step_ms": 0.0,
                "slow_job_delay_per_step_ms": (ratio_2_1_target - t_slow) * 1000.0,
                "slowdown_percentage": ratio_2_1_slowdown,
                "beat_eliminated": True
            }

        # Case 3: Incommensurate beyond threshold -> must isolate onto separate power domains
        return {
            "strategy": "isolate_power_domains",
            "reason": f"Disparity ({slowdown_pct:.1f}%) exceeds max allowable slowdown ({max_slowdown_pct:.1f}%)",
            "beat_eliminated": False
        }


class RateDiversityController:
    """
    Unified controller coordinating harmonic quantization and pacing across a cluster.
    """
    def __init__(self, quantizer: Optional[HarmonicPeriodQuantizer] = None):
        self.quantizer = quantizer or HarmonicPeriodQuantizer()
        self.pacer = DynamicBatchPacer()

    def process_cluster_jobs(self, jobs: List[Dict[str, Any]],
                             max_slowdown_pct: float = 15.0) -> Dict[str, Any]:
        """
        Analyzes a fleet of jobs with varied periods and synthesizes an optimal
        placement and pacing strategy.
        
        Guarantees that a job is only marked harmonized if its resulting period
        strictly matches an exact integer harmonic multiple of the family base period
        without exceeding allowable slowdown.
        """
        period_map = {j["job_id"]: float(j["iteration_period_sec"]) for j in jobs}
        clusters = self.quantizer.cluster_jobs_by_harmonic_family(period_map)

        cluster_plans = []
        for family_target, member_ids in clusters.items():
            if family_target == -1.0 or not member_ids:
                cluster_plans.append({
                    "family_period_sec": None,
                    "members": member_ids,
                    "status": "unharmonized_isolated"
                })
                continue

            pacing_plans = {}
            harmonized_periods = {}
            harmonic_multipliers = {}
            all_harmonized = True

            for jid in member_ids:
                p = period_map[jid]
                # Pacing can only slow down a job (T_paced >= p), never accelerate
                m = max(1, int(math.ceil(p / family_target)))
                target_p = m * family_target
                slowdown_pct = ((target_p - p) / p) * 100.0

                if slowdown_pct <= max_slowdown_pct:
                    delay_ms = (target_p - p) * 1000.0
                    pacing_plans[jid] = delay_ms
                    harmonized_periods[jid] = target_p
                    harmonic_multipliers[jid] = m
                else:
                    all_harmonized = False
                    pacing_plans[jid] = 0.0
                    harmonized_periods[jid] = p
                    harmonic_multipliers[jid] = None

            cluster_plans.append({
                "family_period_sec": family_target,
                "members": member_ids,
                "status": "harmonized" if all_harmonized else "unharmonized_isolated",
                "pacing_delays_ms": pacing_plans,
                "harmonized_periods_sec": harmonized_periods,
                "harmonic_multipliers": harmonic_multipliers
            })

        return {
            "num_jobs": len(jobs),
            "num_clusters": len(cluster_plans),
            "cluster_breakdown": cluster_plans
        }
