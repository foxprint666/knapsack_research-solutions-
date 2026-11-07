"""
harmonic-knapsack/applications/hpc_checkpoint_ftio.py
Application Model: HPC Checkpoint I/O Staging (FTIO Model).

Based on:
  Tarraf et al., "FTIO: Detecting I/O Periodicity in HPC Applications" (TU Darmstadt).

Simulates:
  - Multiple parallel simulation jobs alternating between compute phases and checkpoint writes.
  - Shared burst buffer / parallel file system with finite bandwidth ceiling B_max.
  - Uncoordinated collision vs FTIO-guided harmonic phase-staggering.
  - Metrics: Peak bandwidth demand, I/O contention slowdown, and effective system utilization.
"""

import numpy as np
from core.harmonic_representation import HarmonicProfile, HarmonicComponent
from core.phase_shift_math import PhaseShiftEngine
from core.temporal_knapsack_solver import HarmonicItem, TemporalHarmonicKnapsackSolver

def create_synthetic_checkpoint_profile(period_minutes: float,
                                        checkpoint_duration_minutes: float,
                                        checkpoint_bandwidth_gbps: float,
                                        idle_bandwidth_gbps: float = 0.5) -> HarmonicProfile:
    """
    Creates a Fourier profile approximating a periodic rectangular pulsed checkpoint write.
    """
    period_sec = period_minutes * 60.0
    duration_sec = checkpoint_duration_minutes * 60.0
    duty_cycle = duration_sec / period_sec
    
    # DC mean bandwidth
    dc_mean = (duty_cycle * checkpoint_bandwidth_gbps) + ((1.0 - duty_cycle) * idle_bandwidth_gbps)
    pulse_height = checkpoint_bandwidth_gbps - idle_bandwidth_gbps
    
    # Fourier series expansion of rectangular pulse of width duration_sec
    f0 = 1.0 / period_sec
    harmonics = []
    for k in range(1, 6):
        # A_k = 2 * (pulse_height / (k * pi)) * sin(k * pi * duty_cycle)
        amp = 2.0 * (pulse_height / (k * np.pi)) * np.abs(np.sin(k * np.pi * duty_cycle))
        harmonics.append(HarmonicComponent(
            harmonic_order=k,
            frequency_hz=k * f0,
            amplitude=amp,
            phase_rad=0.0
        ))
        
    return HarmonicProfile(
        period_seconds=period_sec,
        dc_mean=dc_mean,
        harmonics=harmonics,
        periodicity_confidence=0.98
    )

class HPCCheckpointScheduler:
    """
    Schedules HPC job checkpoint phases to eliminate storage contention.
    """
    def __init__(self, storage_bandwidth_capacity_gbps: float = 40.0):
        self.capacity_gbps = storage_bandwidth_capacity_gbps
        self.solver = TemporalHarmonicKnapsackSolver(search_resolution_bins=120)

    def run_benchmark(self, job_configs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Runs comparative benchmark: Uncoordinated Checkpoints vs FTIO Harmonic Staggering.
        """
        if job_configs is None:
            # 6 concurrent scientific simulation jobs with periodic checkpointing
            job_configs = [
                {"name": "Astrophysics_3D", "period_m": 30.0, "duration_m": 4.0, "bw_gbps": 12.0, "val": 100},
                {"name": "Climate_Global",  "period_m": 60.0, "duration_m": 5.0, "bw_gbps": 15.0, "val": 120},
                {"name": "Molecular_Dyn",   "period_m": 30.0, "duration_m": 3.0, "bw_gbps": 10.0, "val": 80},
                {"name": "Turbulence_DNS",  "period_m": 60.0, "duration_m": 6.0, "bw_gbps": 14.0, "val": 110},
                {"name": "Quantum_Chem",   "period_m": 120.0, "duration_m": 8.0, "bw_gbps": 16.0, "val": 90},
                {"name": "Plasma_Fusion",   "period_m": 60.0, "duration_m": 4.0, "bw_gbps": 11.0, "val": 85},
            ]

        items = []
        for i, cfg in enumerate(job_configs):
            prof = create_synthetic_checkpoint_profile(
                period_minutes=cfg["period_m"],
                checkpoint_duration_minutes=cfg["duration_m"],
                checkpoint_bandwidth_gbps=cfg["bw_gbps"]
            )
            items.append(HarmonicItem(
                item_id=i,
                name=cfg["name"],
                value=cfg["val"],
                profile=prof
            ))

        # Solve Peak-Minimization
        uncoord_res = self.solver.solve_peak_minimization(items, method="uncoordinated")
        staggered_res = self.solver.solve_peak_minimization(items, method="greedy_harmonic")

        # Solve Knapsack Admission under Storage Capacity Ceiling
        knapsack_res = self.solver.solve_value_maximization(items, capacity=self.capacity_gbps)

        uncoord_peak = uncoord_res["peak_load"]
        staggered_peak = staggered_res["peak_load"]
        peak_reduction_pct = ((uncoord_peak - staggered_peak) / uncoord_peak) * 100.0

        # Contention severity: duration and magnitude exceeding capacity
        storage_overload_uncoord = max(0.0, uncoord_peak - self.capacity_gbps)
        storage_overload_staggered = max(0.0, staggered_peak - self.capacity_gbps)

        return {
            "num_jobs": len(items),
            "storage_capacity_gbps": self.capacity_gbps,
            "uncoordinated": {
                "peak_bandwidth_gbps": uncoord_peak,
                "overload_excess_gbps": storage_overload_uncoord,
                "is_contended": uncoord_peak > self.capacity_gbps
            },
            "harmonic_staggered": {
                "peak_bandwidth_gbps": staggered_peak,
                "overload_excess_gbps": storage_overload_staggered,
                "is_contended": staggered_peak > self.capacity_gbps,
                "offsets_minutes": [t / 60.0 for t in staggered_res["offsets_seconds"]]
            },
            "peak_reduction_pct": peak_reduction_pct,
            "knapsack_admission": knapsack_res
        }
