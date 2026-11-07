"""
harmonic-knapsack/tests/test_solver_benchmarks.py
Comprehensive benchmark suite testing:
  1. Temporal Harmonic Knapsack (Peak Minimization & Value Maximization).
  2. HPC Checkpoint Staging (FTIO Model).
  3. AI Training Power-Cap Phase Scattering & Drift Recentering.
  4. Enterprise Cron Thundering-Herd Staggering.
  5. Time-Sensitive Networking (TSN 802.1Qbv) GCL Synthesis.
"""

import sys
import os
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.harmonic_representation import HarmonicComponent, HarmonicProfile
from core.temporal_knapsack_solver import HarmonicItem, TemporalHarmonicKnapsackSolver
from applications.hpc_checkpoint_ftio import HPCCheckpointScheduler
from applications.ai_training_power_cap import AIPowerCapScheduler, AITrainingJob
from applications.cron_staggering import CronStaggerSimulator
from applications.tsn_cyclic_scheduler import TSNCyclicScheduler, TSNFlow

def test_knapsack_peak_minimization():
    # Create 5 periodic items with period 60s
    solver = TemporalHarmonicKnapsackSolver(search_resolution_bins=60)
    items = []
    for i in range(5):
        prof = HarmonicProfile(
            period_seconds=60.0,
            dc_mean=10.0,
            harmonics=[HarmonicComponent(1, 1.0/60.0, amplitude=6.0, phase_rad=0.0)]
        )
        items.append(HarmonicItem(item_id=i, name=f"Item_{i}", value=100.0, profile=prof))

    uncoord = solver.solve_peak_minimization(items, method="uncoordinated")
    greedy = solver.solve_peak_minimization(items, method="greedy_harmonic")
    sa = solver.solve_peak_minimization(items, method="simulated_annealing", seed=42)

    # In uncoordinated, all peaks align: 5 * (10 + 6) = 80.0
    assert abs(uncoord["peak_load"] - 80.0) < 0.1, f"Expected 80.0, got {uncoord['peak_load']}"
    # Greedy achieves pairwise antagonistic cancellation (56.0, 30% reduction)
    assert greedy["peak_load"] <= 56.1, f"Greedy peak too high: {greedy['peak_load']}"
    assert greedy["peak_reduction_pct"] >= 30.0, f"Low reduction: {greedy['peak_reduction_pct']}"
    # Simulated Annealing finds the 5-way uniform phase constellation (near theoretical min 50.0)
    assert sa["peak_load"] < 52.0, f"SA peak too high: {sa['peak_load']}"
    print(f"[PASS] test_knapsack_peak_minimization: uncoord={uncoord['peak_load']:.1f} -> greedy={greedy['peak_load']:.1f} (-{greedy['peak_reduction_pct']:.1f}%), SA={sa['peak_load']:.1f}")

def test_knapsack_value_maximization_capacity_gain():
    # Demonstrate Formulation B: Phase-Staggered Knapsack admits significantly more items than Static Knapsack
    solver = TemporalHarmonicKnapsackSolver(search_resolution_bins=60)
    items = []
    # 8 items, each with peak = 16.0 (mean=10, amp=6), value=100
    for i in range(8):
        prof = HarmonicProfile(
            period_seconds=60.0,
            dc_mean=10.0,
            harmonics=[HarmonicComponent(1, 1.0/60.0, amplitude=6.0, phase_rad=0.0)]
        )
        items.append(HarmonicItem(item_id=i, name=f"Task_{i}", value=100.0, profile=prof))

    # Resource capacity C = 58.0
    # Under static peak reservation: max 3 items fit (3 * 16 = 48 <= 58; 4 * 16 = 64 > 58) -> Value = 300
    # Under harmonic phase staggering: 5 items fit (peak 56.0 <= 58) -> Value = 500 (+66.7% gain!)
    res = solver.solve_value_maximization(items, capacity=58.0)

    static_admitted = res["static_baseline"]["admitted_count"]
    harmonic_admitted = res["harmonic_staggered"]["admitted_count"]
    val_gain = res["value_gain_pct"]

    assert static_admitted == 3, f"Expected 3 static items, got {static_admitted}"
    assert harmonic_admitted >= 5, f"Expected at least 5 harmonic items, got {harmonic_admitted}"
    assert val_gain >= 60.0, f"Expected >= 60% value gain, got {val_gain:.1f}%"
    print(f"[PASS] test_knapsack_value_maximization: Static={static_admitted} items (val={res['static_baseline']['total_value']:.0f}) -> Harmonic={harmonic_admitted} items (val={res['harmonic_staggered']['total_value']:.0f}) (+{val_gain:.1f}% value gain)")

def test_hpc_checkpoint_ftio_benchmark():
    scheduler = HPCCheckpointScheduler(storage_bandwidth_capacity_gbps=40.0)
    res = scheduler.run_benchmark()

    # Uncoordinated peak hits storage contention (> 40 Gbps)
    assert res["uncoordinated"]["is_contended"] is True
    assert res["uncoordinated"]["peak_bandwidth_gbps"] > 40.0
    # Staggered eliminates or significantly reduces peak
    assert res["peak_reduction_pct"] > 25.0
    print(f"[PASS] test_hpc_checkpoint_ftio: Uncoord Peak={res['uncoordinated']['peak_bandwidth_gbps']:.1f} Gbps (CONTENDED) -> Staggered={res['harmonic_staggered']['peak_bandwidth_gbps']:.1f} Gbps (-{res['peak_reduction_pct']:.1f}%)")

def test_ai_training_power_cap_benchmark():
    scheduler = AIPowerCapScheduler(rack_power_cap_kw=38.0)
    jobs = [
        AITrainingJob(0, "Llama-70B-Worker-1"),
        AITrainingJob(1, "Llama-70B-Worker-2"),
        AITrainingJob(2, "Mixtral-Worker-3"),
        AITrainingJob(3, "Mixtral-Worker-4"),
        AITrainingJob(4, "Qwen-Worker-5"),
        AITrainingJob(5, "Qwen-Worker-6")
    ]

    res = scheduler.evaluate_phase_scattering(jobs)
    assert res["uncoordinated"]["trips_power_cap"] is True
    assert res["phase_scattered"]["trips_power_cap"] is False
    assert res["peak_reduction_pct"] > 15.0

    # Test phase drift with and without closed-loop recentering
    drift_open = scheduler.simulate_phase_drift_and_recentering(jobs, num_iterations=100, enable_closed_loop=False)
    drift_closed = scheduler.simulate_phase_drift_and_recentering(jobs, num_iterations=100, enable_closed_loop=True)

    assert drift_closed["cap_violation_count"] == 0, "Closed-loop should have 0 power cap violations"
    assert drift_closed["mean_peak_power_kw"] <= drift_open["mean_peak_power_kw"]
    print(f"[PASS] test_ai_training_power_cap: Uncoord={res['uncoordinated']['peak_power_kw']:.1f} kW (TRIPPED) -> Scattered={res['phase_scattered']['peak_power_kw']:.1f} kW (-{res['peak_reduction_pct']:.1f}%) | Closed-loop mean={drift_closed['mean_peak_power_kw']:.1f} kW vs Open={drift_open['mean_peak_power_kw']:.1f} kW.")

def test_cron_thundering_herd_staggering():
    sim = CronStaggerSimulator(num_nodes=5000, cycle_minutes=60, job_duration_minutes=2.0)
    res = sim.run_comparison()

    assert res["naive_synchronized"]["peak_concurrency"] == 5000
    assert res["hash_staggered"]["peak_concurrency"] < 350
    assert res["concurrency_reduction_pct"] > 93.0
    print(f"[PASS] test_cron_thundering_herd_staggering: Naive concurrency=5000 -> Staggered peak={res['hash_staggered']['peak_concurrency']} (-{res['concurrency_reduction_pct']:.1f}%)")

def test_tsn_cyclic_scheduler():
    scheduler = TSNCyclicScheduler(cycle_time_us=1000.0, guard_band_us=10.0)
    flows = [
        TSNFlow(1, "Brake_by_Wire", period_us=1000.0, transmission_duration_us=120.0),
        TSNFlow(2, "Steering_Angle", period_us=1000.0, transmission_duration_us=80.0),
        TSNFlow(3, "Lidar_Scan", period_us=1000.0, transmission_duration_us=250.0),
        TSNFlow(4, "Radar_Dwell", period_us=1000.0, transmission_duration_us=150.0),
    ]

    sched = scheduler.synthesize_schedule(flows)
    assert sched["schedulable"] is True
    assert len(sched["schedule"]) == 4

    # Clock drift test
    drift_safe = scheduler.simulate_clock_drift_collision(sched, drift_rate_ppm=5.0, elapsed_seconds=1.0)
    assert drift_safe["collision_detected"] is False

    drift_danger = scheduler.simulate_clock_drift_collision(sched, drift_rate_ppm=50.0, elapsed_seconds=1.0)
    assert drift_danger["collision_detected"] is True
    print(f"[PASS] test_tsn_cyclic_scheduler: GCL synthesized ({sched['utilization_pct']:.1f}% util); clock drift collision detected when drift ({drift_danger['accumulated_drift_us']:.1f}us) > guard ({scheduler.guard_band_us}us).")

if __name__ == "__main__":
    test_knapsack_peak_minimization()
    test_knapsack_value_maximization_capacity_gain()
    test_hpc_checkpoint_ftio_benchmark()
    test_ai_training_power_cap_benchmark()
    test_cron_thundering_herd_staggering()
    test_tsn_cyclic_scheduler()
    print("\nAll solver and application benchmark tests passed successfully!")
