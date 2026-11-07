"""
harmonic-knapsack/core/temporal_knapsack_solver.py
Phase-Offset Gated Harmonic Knapsack & Peak-Constrained Bin Packing Solver.

Implements:
  1. Problem Formulation A: Peak-Minimizing Harmonic Scheduling (Fixed Item Set)
       min_{tau_1 ... tau_N} max_t sum_i L_i(t - tau_i)
  2. Problem Formulation B: Value-Maximizing Temporal Harmonic Knapsack
       max sum_i v_i x_i  s.t. max_t sum_i x_i L_i(t - tau_i) <= C
  3. Solvers:
       - Uncoordinated Baseline (in-phase peak collision)
       - Greedy Harmonic Staggering (fast sequential phase search)
       - Simulated Annealing Phase Optimizer (global multi-harmonic optimization)
       - Static Peak Reservation Baseline comparison (demonstrating admission gain)
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .harmonic_representation import HarmonicProfile
from .phase_shift_math import PhaseShiftEngine, compute_hyperperiod

class HarmonicItem:
    """Represents a periodic workload item available for scheduling/admission."""
    def __init__(self, item_id: int, name: str, value: float,
                 profile: HarmonicProfile):
        self.item_id = item_id
        self.name = name
        self.value = float(value)
        self.profile = profile
        
        # Precompute standalone peak and mean
        t_sample = np.linspace(0.0, profile.period_seconds, 256, endpoint=False)
        load_sample = profile.evaluate(t_sample)
        self.peak_load = float(np.max(load_sample))
        self.mean_load = profile.dc_mean
        
        # Value density per unit peak demand
        self.value_density = self.value / max(1e-6, self.peak_load)

    def __repr__(self) -> str:
        return (f"HarmonicItem({self.name}, val={self.value:.1f}, "
                f"peak={self.peak_load:.2f}, mean={self.mean_load:.2f}, "
                f"period={self.profile.period_seconds:.1f}s)")


class TemporalHarmonicKnapsackSolver:
    """
    Main solver class for Peak-Minimization and Value-Maximization under harmonic constraints.
    """
    def __init__(self, search_resolution_bins: int = 64):
        self.search_resolution_bins = search_resolution_bins

    def solve_peak_minimization(self, items: List[HarmonicItem],
                                method: str = "greedy_harmonic",
                                seed: int = 42) -> Dict[str, Any]:
        """
        Solves Formulation A: finds start offsets tau_i to minimize the peak aggregate load.
        """
        if not items:
            return {"peak_load": 0.0, "offsets": [], "uncoordinated_peak": 0.0}

        profiles = [item.profile for item in items]
        periods = [p.period_seconds for p in profiles]
        T_hyper = compute_hyperperiod(periods)
        if math.isinf(T_hyper) or T_hyper <= 0:
            T_hyper = max(periods)

        # Baseline: Uncoordinated (all offsets = 0)
        zero_offsets = [0.0] * len(items)
        t_grid, uncoord_curve, uncoord_peak = PhaseShiftEngine.combine_harmonic_profiles(
            profiles, zero_offsets, eval_points=512
        )

        if method == "uncoordinated":
            return {
                "method": "uncoordinated",
                "peak_load": uncoord_peak,
                "uncoordinated_peak": uncoord_peak,
                "peak_reduction_pct": 0.0,
                "offsets_seconds": zero_offsets,
                "hyperperiod_seconds": T_hyper,
                "aggregate_curve": uncoord_curve
            }

        elif method == "greedy_harmonic":
            # Sort items descending by standalone peak demand
            sorted_indices = sorted(range(len(items)), key=lambda i: items[i].peak_load, reverse=True)
            offsets = [0.0] * len(items)
            
            # Place first dominant item at offset 0
            current_scheduled_profiles = [profiles[sorted_indices[0]]]
            current_scheduled_offsets = [0.0]

            for idx in sorted_indices[1:]:
                item = items[idx]
                p = item.profile
                
                # Search candidate offsets in [0, p.period_seconds)
                candidate_taus = np.linspace(0.0, p.period_seconds, self.search_resolution_bins, endpoint=False)
                best_tau = 0.0
                best_peak = float('inf')

                for tau in candidate_taus:
                    test_profiles = current_scheduled_profiles + [p]
                    test_offsets = current_scheduled_offsets + [tau]
                    _, _, cand_peak = PhaseShiftEngine.combine_harmonic_profiles(
                        test_profiles, test_offsets, eval_points=256
                    )
                    if cand_peak < best_peak:
                        best_peak = cand_peak
                        best_tau = float(tau)

                offsets[idx] = best_tau
                current_scheduled_profiles.append(p)
                current_scheduled_offsets.append(best_tau)

            # Evaluate final curve
            t_grid, final_curve, final_peak = PhaseShiftEngine.combine_harmonic_profiles(
                profiles, offsets, eval_points=512
            )
            reduction_pct = ((uncoord_peak - final_peak) / max(1e-6, uncoord_peak)) * 100.0

            return {
                "method": "greedy_harmonic",
                "peak_load": final_peak,
                "uncoordinated_peak": uncoord_peak,
                "peak_reduction_pct": reduction_pct,
                "offsets_seconds": offsets,
                "hyperperiod_seconds": T_hyper,
                "aggregate_curve": final_curve
            }

        elif method == "simulated_annealing":
            rng = np.random.default_rng(seed)
            # Initialize with greedy solution
            greedy_res = self.solve_peak_minimization(items, method="greedy_harmonic")
            current_offsets = list(greedy_res["offsets_seconds"])
            current_peak = greedy_res["peak_load"]
            best_offsets = list(current_offsets)
            best_peak = current_peak

            temperature = current_peak * 0.20
            cooling_rate = 0.92
            iterations_per_temp = 20

            for step in range(30):
                for _ in range(iterations_per_temp):
                    # Pick a random item (except first) and perturb offset
                    target_i = rng.integers(0, len(items))
                    p = profiles[target_i]
                    delta = rng.uniform(-0.25, 0.25) * p.period_seconds
                    new_offset = (current_offsets[target_i] + delta) % p.period_seconds
                    
                    candidate_offsets = list(current_offsets)
                    candidate_offsets[target_i] = new_offset

                    _, _, cand_peak = PhaseShiftEngine.combine_harmonic_profiles(
                        profiles, candidate_offsets, eval_points=256
                    )

                    d_e = cand_peak - current_peak
                    if d_e < 0 or (temperature > 1e-4 and rng.random() < math.exp(-d_e / temperature)):
                        current_offsets = candidate_offsets
                        current_peak = cand_peak
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_offsets = list(current_offsets)

                temperature *= cooling_rate

            t_grid, final_curve, final_peak = PhaseShiftEngine.combine_harmonic_profiles(
                profiles, best_offsets, eval_points=512
            )
            reduction_pct = ((uncoord_peak - final_peak) / max(1e-6, uncoord_peak)) * 100.0

            return {
                "method": "simulated_annealing",
                "peak_load": final_peak,
                "uncoordinated_peak": uncoord_peak,
                "peak_reduction_pct": reduction_pct,
                "offsets_seconds": best_offsets,
                "hyperperiod_seconds": T_hyper,
                "aggregate_curve": final_curve
            }

        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def solve_value_maximization(self, candidate_items: List[HarmonicItem],
                                 capacity: float) -> Dict[str, Any]:
        """
        Solves Formulation B: Maximize sum(v_i * x_i) subject to max_t sum_i x_i L_i(t - tau_i) <= C.
        Compares Phase-Staggered Harmonic Knapsack vs Static Peak Knapsack.
        """
        # Baseline: Static Peak 0/1 Knapsack (items ordered by value density)
        sorted_by_density = sorted(candidate_items, key=lambda it: it.value_density, reverse=True)
        
        # 1. Static Peak Knapsack
        static_admitted = []
        static_total_peak = 0.0
        static_total_value = 0.0
        for it in sorted_by_density:
            if static_total_peak + it.peak_load <= capacity:
                static_admitted.append(it)
                static_total_peak += it.peak_load
                static_total_value += it.value

        # 2. Harmonic Phase-Staggered Knapsack
        harmonic_admitted: List[HarmonicItem] = []
        harmonic_offsets: List[float] = []
        harmonic_total_value = 0.0

        for it in sorted_by_density:
            p = it.profile
            candidate_taus = np.linspace(0.0, p.period_seconds, self.search_resolution_bins, endpoint=False)
            
            best_tau = None
            min_resulting_peak = float('inf')

            for tau in candidate_taus:
                test_profiles = [item.profile for item in harmonic_admitted] + [p]
                test_offsets = harmonic_offsets + [tau]
                _, _, cand_peak = PhaseShiftEngine.combine_harmonic_profiles(
                    test_profiles, test_offsets, eval_points=256
                )
                if cand_peak <= capacity and cand_peak < min_resulting_peak:
                    min_resulting_peak = cand_peak
                    best_tau = float(tau)

            if best_tau is not None:
                harmonic_admitted.append(it)
                harmonic_offsets.append(best_tau)
                harmonic_total_value += it.value

        # Compute realized aggregate peak of admitted harmonic set
        if harmonic_admitted:
            _, _, realized_harmonic_peak = PhaseShiftEngine.combine_harmonic_profiles(
                [it.profile for it in harmonic_admitted], harmonic_offsets, eval_points=512
            )
        else:
            realized_harmonic_peak = 0.0

        value_gain_pct = ((harmonic_total_value - static_total_value) / max(1e-6, static_total_value)) * 100.0
        item_gain_pct = ((len(harmonic_admitted) - len(static_admitted)) / max(1, len(static_admitted))) * 100.0

        return {
            "capacity": float(capacity),
            "static_baseline": {
                "admitted_count": len(static_admitted),
                "admitted_items": [it.name for it in static_admitted],
                "total_value": static_total_value,
                "reserved_peak": static_total_peak
            },
            "harmonic_staggered": {
                "admitted_count": len(harmonic_admitted),
                "admitted_items": [it.name for it in harmonic_admitted],
                "offsets_seconds": harmonic_offsets,
                "total_value": harmonic_total_value,
                "realized_peak": realized_harmonic_peak
            },
            "value_gain_pct": value_gain_pct,
            "admitted_items_gain_pct": item_gain_pct
        }
