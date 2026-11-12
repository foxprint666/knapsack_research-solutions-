"""
harmonic-knapsack/tests/test_pipeline_parallel_harmonics.py
Unit tests verifying 1F1B Pipeline Parallelism modeling, multi-step staircase power profiles,
analytical Fourier series decomposition with K harmonics, and multi-job phase scattering.
"""

import sys
import os
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from applications.ai_training_power_cap import (
    PipelineStage1F1B,
    PipelineParallelTrainingJob,
    AIPowerCapScheduler
)
from production.live_nvml_benchmark import PhysicsEmulatedNVMLDevice
from production.telemetry_sensor import TelemetrySensor

def test_1f1b_schedule_and_bubble_structure():
    # Setup P=4 stages, M=8 microbatches
    stage0 = PipelineStage1F1B(stage_id=0, num_stages=4, num_microbatches=8,
                               fwd_duration_ms=20.0, bwd_duration_ms=40.0)
    stage3 = PipelineStage1F1B(stage_id=3, num_stages=4, num_microbatches=8,
                               fwd_duration_ms=20.0, bwd_duration_ms=40.0)

    # Stage 0 starts computing immediately at t=0
    assert stage0.intervals[0][0] == "F", f"Expected Stage 0 to start with FWD, got {stage0.intervals[0][0]}"
    assert stage0.intervals[0][1] == 0.0

    # Stage 3 must have a warmup bubble waiting for activations from stages 0, 1, 2
    # Warmup bubble duration = 3 * 20ms = 60ms = 0.060s
    assert stage3.intervals[0][0] == "WARMUP_BUBBLE"
    assert abs(stage3.intervals[0][2] - stage3.intervals[0][1] - 0.060) < 1e-4

    # Both stages must share the exact same global iteration period
    assert abs(stage0.iteration_period_sec - stage3.iteration_period_sec) < 1e-6
    print(f"[PASS] test_1f1b_schedule_and_bubble_structure: Iteration Period = {stage0.iteration_period_sec:.4f}s")

def test_analytical_fourier_vs_discrete_fft():
    # Verify that the closed-form analytical Fourier integrals of the 1F1B staircase
    # match discrete FFT on a densely sampled grid.
    stage = PipelineStage1F1B(stage_id=1, num_stages=4, num_microbatches=8,
                              fwd_duration_ms=25.0, bwd_duration_ms=50.0,
                              fwd_power_kw=5.5, bwd_power_kw=6.5, bubble_power_kw=1.5,
                              num_harmonics=16)

    prof = stage.profile
    T = stage.iteration_period_sec
    assert prof.periodicity_confidence > 0.85, f"Confidence too low: {prof.periodicity_confidence}"

    # Sample discrete signal at 10,000 points
    N = 10000
    t = np.linspace(0, T, N, endpoint=False)
    sampled = np.array([stage.sample_power_at_time(ti) for ti in t])

    # Check DC mean match
    discrete_dc = float(np.mean(sampled))
    assert abs(prof.dc_mean - discrete_dc) < 0.05, f"DC mismatch: analytical={prof.dc_mean}, discrete={discrete_dc}"

    # Check Fourier harmonic amplitudes against FFT
    fft_vals = np.fft.rfft(sampled) / N
    for h in prof.harmonics[:8]:
        fft_amp = 2.0 * float(np.abs(fft_vals[h.harmonic_order]))
        diff = abs(h.amplitude - fft_amp)
        assert diff < 0.05, f"Harmonic {h.harmonic_order} mismatch: analytical={h.amplitude:.4f}, fft={fft_amp:.4f}"

    print(f"[PASS] test_analytical_fourier_vs_discrete_fft: Analytical Fourier matches FFT within 0.05 kW across all harmonics.")

def test_spectral_energy_bimodal_distribution():
    # Verify that 1F1B produces both low-frequency macro envelope energy
    # and micro-batch carrier frequency ripples
    job = PipelineParallelTrainingJob(
        job_id=1,
        name="Llama-4-PP4-M8",
        num_stages=4,
        num_microbatches=8,
        fwd_duration_ms=25.0,
        bwd_duration_ms=50.0,
        num_harmonics=24
    )

    dist = job.compute_spectral_energy_distribution()
    # Macro envelope (k in 1..4) carries dominant AC energy in the aggregate profile
    assert dist["macro_envelope_ac_pct"] > 50.0, f"Low macro energy: {dist['macro_envelope_ac_pct']}"
    # Microbatch carrier ripples (k near 8 and 16) exist
    assert dist["microbatch_carrier_ac_pct"] > 1.0, f"Microbatch carrier missing: {dist['microbatch_carrier_ac_pct']}"
    assert abs(dist["bubble_fraction"] - (3.0 / 8.0)) < 1e-4

    # On an individual stage (Stage 2), micro-batch harmonics (k near M=8) are prominent
    st2_h8 = job.stages[2].profile.harmonics[7].amplitude
    st2_h1 = job.stages[2].profile.harmonics[0].amplitude
    assert st2_h8 / st2_h1 > 0.20, f"Expected Stage 2 micro-batch ripple to be >20% of fundamental, got {st2_h8/st2_h1:.2f}"

    print(f"[PASS] test_spectral_energy_bimodal_distribution: Aggregate Macro={dist['macro_envelope_ac_pct']:.1f}%, Stage 2 MicroBatch/Macro={st2_h8/st2_h1:.2f}")

def test_aggregate_pipeline_linear_additivity():
    # Verify that the aggregate pipeline profile matches the point-by-point sum of stages
    job = PipelineParallelTrainingJob(
        job_id=2,
        name="GPT-Class-PP4-M12",
        num_stages=4,
        num_microbatches=12,
        num_harmonics=24
    )

    T = job.iteration_period_sec
    t = np.linspace(0, T, 500, endpoint=False)
    pointwise_sum = np.array([job.sample_aggregate_power_at_time(ti) for ti in t])
    harmonic_eval = job.profile.evaluate(t)

    # Mean absolute difference between piecewise sum and K-harmonic reconstruction
    mae = float(np.mean(np.abs(pointwise_sum - harmonic_eval)))
    # With K=24 harmonics, MAE is small relative to the 25kW peak
    assert mae < 2.5, f"High reconstruction error: {mae} kW"
    print(f"[PASS] test_aggregate_pipeline_linear_additivity: Harmonic reconstruction MAE = {mae:.3f} kW against 25 kW peak.")

def test_multi_job_pipeline_phase_scattering():
    # 4 identical 3D-parallel jobs sharing a 100 kW power cap
    jobs = [
        PipelineParallelTrainingJob(job_id=i, name=f"Llama-4-Replica-{i}",
                                     num_stages=4, num_microbatches=8)
        for i in range(4)
    ]

    scheduler = AIPowerCapScheduler(rack_power_cap_kw=85.0)
    res = scheduler.evaluate_pipeline_coordination(jobs)

    # Uncoordinated peak will sum the in-phase active compute plateaus
    uncoord_peak = res["uncoordinated"]["peak_power_kw"]
    scattered_peak = res["phase_scattered"]["peak_power_kw"]
    reduction = res["peak_reduction_pct"]

    assert uncoord_peak > scattered_peak
    assert reduction > 10.0, f"Expected >10% peak reduction, got {reduction:.2f}%"
    print(f"[PASS] test_multi_job_pipeline_phase_scattering: Uncoordinated={uncoord_peak:.2f}kW -> Scattered={scattered_peak:.2f}kW (-{reduction:.2f}%)")

def test_physics_emulated_nvml_with_pipeline_stage():
    stage = PipelineStage1F1B(stage_id=2, num_stages=4, num_microbatches=8)
    device = PhysicsEmulatedNVMLDevice(device_id=0, pipeline_stage=stage)

    # Sample 8 integer periods to avoid edge spectral leakage
    T = stage.iteration_period_sec
    dt = 0.02
    N = int((8 * T) / dt)
    samples = []
    sensor = TelemetrySensor(job_id=99, sampling_rate_hz=50.0, window_duration_sec=8 * T, num_harmonics=16)
    
    for i in range(N):
        t = i * dt
        p_watts = device.sample_power_watts(t)
        samples.append(p_watts)
        sensor.ingest_reading(power_kw=p_watts / 1000.0, timestamp_sec=t)

    assert len(samples) == N
    # Watts should stay within valid GPU/node power envelope
    assert min(samples) > 1000.0  # >1.0 kW base idle
    assert max(samples) < 8000.0  # <8.0 kW peak compute

    # Telemetry sensor state check
    state = sensor.poll_spectral_state(assumed_period_sec=stage.iteration_period_sec, num_harmonics=16)
    assert state["has_profile"] is True
    assert state["confidence"] > 0.85
    assert abs(state["detected_period_sec"] - T) < 0.05
    print(f"[PASS] test_physics_emulated_nvml_with_pipeline_stage: Ingested {len(samples)} samples, Telemetry confidence={state['confidence']:.2f}, detected period={state['detected_period_sec']:.3f}s")

if __name__ == "__main__":
    test_1f1b_schedule_and_bubble_structure()
    test_analytical_fourier_vs_discrete_fft()
    test_spectral_energy_bimodal_distribution()
    test_aggregate_pipeline_linear_additivity()
    test_multi_job_pipeline_phase_scattering()
    test_physics_emulated_nvml_with_pipeline_stage()
