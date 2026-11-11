"""
harmonic-knapsack/tests/test_production_runtime.py
Unit tests verifying the three production-grade engineering components:
1. Real-Time Telemetry Ingestion (The Sensor)
2. Non-Intrusive Actuation Hook (The Actuator)
3. Rate-Diversity & Higher-Harmonic Handling (The Controller)
"""

import sys
import os
import time
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from production.telemetry_sensor import (
    PowerSampleRingBuffer,
    StreamingSpectralAnalyzer,
    TelemetrySensor
)
from production.nccl_actuator_hook import (
    ActuationPlan,
    MicroBarrierPhaseController,
    NCCLActuatorHook
)
from production.rate_diversity_controller import (
    HarmonicPeriodQuantizer,
    DynamicBatchPacer,
    RateDiversityController
)

def test_telemetry_sensor_ring_buffer_and_analyzer():
    sensor = TelemetrySensor(job_id=42, sampling_rate_hz=50.0)
    dt = 0.02  # 50 Hz
    period = 1.0  # 1.0s period
    t0 = 100.0

    # Stream 250 samples (5 full periods) of synthetic GPU power waveform: 5.0 + 3.0*cos(2*pi*t)
    for i in range(250):
        t = t0 + i * dt
        p = 5.0 + 3.0 * np.cos(2.0 * np.pi * (1.0 / period) * t)
        sensor.ingest_reading(power_kw=p, timestamp_sec=t)

    assert len(sensor.buffer) == 250

    state = sensor.poll_spectral_state(assumed_period_sec=1.0)
    assert state["has_profile"] is True
    assert state["confidence"] > 0.85
    assert abs(state["detected_period_sec"] - 1.0) < 0.1
    print(f"[PASS] test_telemetry_sensor: Period={state['detected_period_sec']:.2f}s, Confidence={state['confidence']:.2f}")

def test_nccl_actuator_hook_throughput_retention():
    # Simulate a 1000-step training job with 1000ms iteration period
    controller = MicroBarrierPhaseController(job_id=1, iteration_period_ms=1000.0)
    hook = NCCLActuatorHook(controller=controller, emulate_hardware=True)

    # Request a 150ms phase shift
    plan = controller.request_phase_shift(target_offset_ms=150.0, max_delay_ms=150.0)
    assert plan.shift_budget_ms == 150.0

    # Execute 100 iterations of training
    for step in range(100):
        hook()

    # Verify phase is shifted to 150ms
    assert abs(controller.current_estimated_offset_ms - 150.0) < 1e-4

    # Calculate throughput retention over 10,000 steps
    stats = controller.compute_throughput_overhead(total_training_steps=10000)
    assert stats["slowdown_percentage"] < 0.02  # Less than 0.02% slowdown
    assert stats["effective_throughput_retention_pct"] > 99.98
    print(f"[PASS] test_nccl_actuator_hook: Injected {stats['total_delay_injected_sec']:.3f}s delay -> Slowdown={stats['slowdown_percentage']:.4f}%, Throughput Retention={stats['effective_throughput_retention_pct']:.4f}%")

def test_rate_diversity_beat_frequency_and_pacing():
    # Case 1: Job A (800ms) and Job B (1100ms)
    beat = DynamicBatchPacer.calculate_beat_frequency(0.8, 1.1)
    assert abs(beat["beat_frequency_hz"] - 0.3409) < 0.01
    assert abs(beat["beat_recurrence_sec"] - 2.93) < 0.1
    # LCM hyperperiod = 8.8s
    assert abs(beat["hyperperiod_sec"] - 8.8) < 0.1

    # Case 2: Pacing adjustment
    # Pacing 800ms to 900ms vs 1100ms
    adj = DynamicBatchPacer.compute_pacing_adjustment(period_a_sec=0.95, period_b_sec=1.0, max_slowdown_pct=10.0)
    assert adj["beat_eliminated"] is True
    assert adj["strategy"] == "direct_1_to_1_lock"
    assert abs(adj["fast_job_delay_per_step_ms"] - 50.0) < 1e-4

    # Case 3: Cluster quantization
    quantizer = HarmonicPeriodQuantizer(base_period_families=[0.5, 1.0, 2.0])
    jobs = {1: 0.98, 2: 1.02, 3: 0.51, 4: 1.95, 5: 3.5}
    clusters = quantizer.cluster_jobs_by_harmonic_family(jobs, tolerance_pct=15.0)

    assert 1 in clusters[1.0] and 2 in clusters[1.0]
    assert 3 in clusters[0.5]
    assert 4 in clusters[2.0]
    assert 5 in clusters[-1.0]  # Isolated
    print(f"[PASS] test_rate_diversity: Beat freq={beat['beat_frequency_hz']:.3f}Hz, Recurrence={beat['beat_recurrence_sec']:.2f}s, Clusters={clusters}")

if __name__ == "__main__":
    test_telemetry_sensor_ring_buffer_and_analyzer()
    test_nccl_actuator_hook_throughput_retention()
    test_rate_diversity_beat_frequency_and_pacing()
    print("\nAll production runtime tests passed successfully!")
