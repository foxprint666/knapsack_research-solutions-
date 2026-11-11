"""
harmonic-knapsack/tests/test_harmonic_math.py
Unit tests verifying spectral decomposition, Fourier series reconstruction,
phase rotation mathematics, and antagonistic wave cancellation.
"""

import sys
import os
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.harmonic_representation import HarmonicComponent, HarmonicProfile
from core.phase_shift_math import PhaseShiftEngine, compute_hyperperiod

def test_fourier_decomposition_and_reconstruction():
    # Construct a synthetic periodic signal: 10 + 4*cos(2*pi*f*t) + 2*sin(4*pi*f*t)
    f0 = 1.0 / 60.0  # period = 60 seconds
    t = np.linspace(0.0, 180.0, 1800, endpoint=False) # 3 full periods, dt=0.1s
    pure_signal = 10.0 + 4.0 * np.cos(2 * np.pi * f0 * t) + 2.0 * np.sin(4 * np.pi * f0 * t)

    prof = HarmonicProfile.from_timeseries(pure_signal, sampling_interval_sec=0.1, assumed_period_sec=60.0, num_harmonics=2)
    
    # Check DC component
    assert abs(prof.dc_mean - 10.0) < 0.1, f"DC mean mismatch: {prof.dc_mean}"
    # Check fundamental harmonic amplitude
    assert abs(prof.harmonics[0].amplitude - 4.0) < 0.15, f"Harmonic 1 amplitude mismatch: {prof.harmonics[0].amplitude}"
    # Check confidence score
    assert prof.periodicity_confidence > 0.95, f"Low confidence: {prof.periodicity_confidence}"

    # Reconstruct signal
    t_recon = np.linspace(0.0, 60.0, 600, endpoint=False)
    recon = prof.evaluate(t_recon)
    target = 10.0 + 4.0 * np.cos(2 * np.pi * f0 * t_recon) + 2.0 * np.sin(4 * np.pi * f0 * t_recon)
    max_recon_err = float(np.max(np.abs(recon - target)))
    assert max_recon_err < 0.3, f"High reconstruction error: {max_recon_err}"
    print(f"[PASS] test_fourier_decomposition_and_reconstruction: DC={prof.dc_mean:.2f}, Amp1={prof.harmonics[0].amplitude:.2f}, max_err={max_recon_err:.4f}")

def test_phase_shift_frequency_vs_time_domain():
    # Verify that frequency-domain phase rotation exp(-i * k * omega * tau)
    # exactly matches a time-domain physical circular roll.
    period = 100.0
    f0 = 1.0 / period
    t = np.linspace(0.0, period, 1000, endpoint=False)
    
    prof = HarmonicProfile(
        period_seconds=period,
        dc_mean=5.0,
        harmonics=[
            HarmonicComponent(1, f0, amplitude=3.0, phase_rad=0.5),
            HarmonicComponent(2, 2 * f0, amplitude=1.5, phase_rad=-0.8)
        ]
    )

    tau = 25.0  # shift by quarter period
    # Method 1: Evaluate with phase offset parameter
    shifted_eval = prof.evaluate(t, phase_offset_seconds=tau)

    # Method 2: Manually rotate harmonics via PhaseShiftEngine
    rotated_harmonics = PhaseShiftEngine.rotate_harmonics(prof.harmonics, tau)
    prof_rotated = HarmonicProfile(period, prof.dc_mean, rotated_harmonics)
    rotated_eval = prof_rotated.evaluate(t, phase_offset_seconds=0.0)

    # Method 3: Direct time domain shift (circular roll by 250 samples)
    unshifted_eval = prof.evaluate(t, phase_offset_seconds=0.0)
    time_rolled = np.roll(unshifted_eval, 250)

    diff_eval_rotated = float(np.max(np.abs(shifted_eval - rotated_eval)))
    diff_eval_rolled = float(np.max(np.abs(shifted_eval - time_rolled)))

    assert diff_eval_rotated < 1e-10, f"Rotation mismatch: {diff_eval_rotated}"
    assert diff_eval_rolled < 0.05, f"Time roll mismatch: {diff_eval_rolled}"
    print(f"[PASS] test_phase_shift_frequency_vs_time_domain: analytical rotation diff={diff_eval_rotated:.2e}, time roll diff={diff_eval_rolled:.4f}")

def test_antagonistic_pairing_cancellation():
    # Two identical cosine waves: L1 = 10 + 5*cos(omega*t), L2 = 10 + 5*cos(omega*t)
    period = 60.0
    f0 = 1.0 / period
    p1 = HarmonicProfile(period, 10.0, [HarmonicComponent(1, f0, amplitude=5.0, phase_rad=0.0)])
    p2 = HarmonicProfile(period, 10.0, [HarmonicComponent(1, f0, amplitude=5.0, phase_rad=0.0)])

    res = PhaseShiftEngine.compute_antagonistic_cancellation_metric(p1, p2)
    # When shifted by tau = 30s (half period = pi rad), cos(wt) and cos(wt + pi) = -cos(wt) exactly cancel!
    # Uncoordinated peak: 10 + 10 + 5 + 5 = 30.0
    # Optimal peak: 10 + 10 + 0 = 20.0 (a 33.3% peak reduction!)
    assert abs(res["optimal_offset_seconds"] - 30.0) < 0.5, f"Expected 30.0s offset, got {res['optimal_offset_seconds']}"
    assert abs(res["min_peak"] - 20.0) < 0.1, f"Expected 20.0 min peak, got {res['min_peak']}"
    assert abs(res["peak_shaving_pct"] - 33.33) < 0.5, f"Peak reduction mismatch: {res['peak_shaving_pct']}"
    print(f"[PASS] test_antagonistic_pairing_cancellation: offset={res['optimal_offset_seconds']:.1f}s, uncoord={res['uncoordinated_peak']:.1f} -> min_peak={res['min_peak']:.1f} (-{res['peak_shaving_pct']:.1f}%)")

def test_hyperperiod_computation():
    # Test harmonic periods: 10, 20, 40 -> LCM = 40
    assert compute_hyperperiod([10.0, 20.0, 40.0]) == 40.0
    # Test non-harmonic: 10.0 and 15.0 -> LCM = 30.0
    assert compute_hyperperiod([10.0, 15.0]) == 30.0
    # Test small fractional: 10.0 and 10.5 -> LCM = 210.0
    assert abs(compute_hyperperiod([10.0, 10.5]) - 210.0) < 0.1
    print("[PASS] test_hyperperiod_computation: all LCM calculations passed.")

def test_fft_cross_correlation_and_greedy_fft():
    # Verify O(M log M) circular cross-correlation vs discrete grid search
    period = 60.0
    f0 = 1.0 / period
    p1 = HarmonicProfile(period, 10.0, [HarmonicComponent(1, f0, amplitude=5.0, phase_rad=0.0)])
    p2 = HarmonicProfile(period, 10.0, [HarmonicComponent(1, f0, amplitude=5.0, phase_rad=0.0)])

    # 1. Test PhaseShiftEngine.find_optimal_offset_fft
    best_tau, min_peak, uncoord_peak = PhaseShiftEngine.find_optimal_offset_fft(p1, p2, eval_points=512)
    assert abs(best_tau - 30.0) < 0.5, f"Expected 30.0s offset, got {best_tau}"
    assert abs(min_peak - 20.0) < 0.1, f"Expected min peak 20.0, got {min_peak}"
    assert abs(uncoord_peak - 30.0) < 0.1, f"Expected uncoord peak 30.0, got {uncoord_peak}"

    # 2. Test solver greedy_fft vs greedy_harmonic
    from core.temporal_knapsack_solver import HarmonicItem, TemporalHarmonicKnapsackSolver
    solver = TemporalHarmonicKnapsackSolver(search_resolution_bins=60)
    items = [
        HarmonicItem(0, "Item0", 100.0, p1),
        HarmonicItem(1, "Item1", 100.0, p2),
    ]
    res_fft = solver.solve_peak_minimization(items, method="greedy_fft")
    res_harmonic = solver.solve_peak_minimization(items, method="greedy_harmonic")

    assert abs(res_fft["peak_load"] - 20.0) < 0.1
    assert abs(res_fft["peak_load"] - res_harmonic["peak_load"]) < 0.2
    assert res_fft["peak_reduction_pct"] >= 33.0
    print(f"[PASS] test_fft_cross_correlation_and_greedy_fft: FFT solver peak={res_fft['peak_load']:.1f} vs Harmonic solver={res_harmonic['peak_load']:.1f}")

if __name__ == "__main__":
    test_fourier_decomposition_and_reconstruction()
    test_phase_shift_frequency_vs_time_domain()
    test_antagonistic_pairing_cancellation()
    test_hyperperiod_computation()
    test_fft_cross_correlation_and_greedy_fft()
    print("\nAll harmonic math tests passed successfully!")
