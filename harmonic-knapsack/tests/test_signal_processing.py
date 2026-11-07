"""
harmonic-knapsack/tests/test_signal_processing.py
Unit tests verifying Active Noise Cancellation (FxLMS causality & Pareto noise)
and Adaptive Beamforming (SMI, RMB rule, and steering mismatch).
"""

import sys
import os
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from signal_processing.anc_fxlms_simulator import FxLMSSimulator
from signal_processing.adaptive_beamformer import AdaptiveBeamformer

def test_fxlms_causality_and_attenuation():
    sim = FxLMSSimulator(filter_order=32, mu=0.005)

    # 1. Causal setup: reference lead time = 4 samples (t_elec < t_acoustic)
    causal_res = sim.run_simulation(num_samples=2500, noise_type="gaussian", causal_delay_samples=4)
    assert causal_res["is_causal"] is True
    assert causal_res["diverged"] is False
    assert causal_res["attenuation_db"] > 10.0, f"Expected > 10 dB attenuation, got {causal_res['attenuation_db']:.2f} dB"
    print(f"[PASS] test_fxlms_causality: Causal FxLMS achieved {causal_res['attenuation_db']:.2f} dB broadband noise attenuation.")

    # 2. Acausal setup: reference lag time = -6 samples (t_elec > t_acoustic)
    acausal_res = sim.run_simulation(num_samples=2500, noise_type="gaussian", causal_delay_samples=-6)
    assert acausal_res["is_causal"] is False
    # Acausal filter cannot predict broadband noise; attenuation drops near/below 0 dB
    assert acausal_res["attenuation_db"] < 1.0, f"Acausal filter should not cancel broadband noise, got {acausal_res['attenuation_db']:.2f} dB"
    print(f"[PASS] test_fxlms_causality: Acausal FxLMS failed to cancel broadband noise ({acausal_res['attenuation_db']:.2f} dB), confirming causality law.")

def test_fxlms_pareto_impulsive_stability():
    sim = FxLMSSimulator(filter_order=32, mu=0.005)

    # Standard LMS on Pareto impulsive noise (extreme outliers)
    standard_pareto = sim.run_simulation(num_samples=2500, noise_type="pareto", use_sign_lms=False, seed=99)
    # Normalized Sign-LMS on Pareto impulsive noise
    robust_pareto = sim.run_simulation(num_samples=2500, noise_type="pareto", use_sign_lms=True, seed=99)

    # Robust Normalized Sign-LMS should not diverge
    assert robust_pareto["diverged"] is False
    print(f"[PASS] test_fxlms_pareto_stability: Standard LMS under Pareto noise diverged={standard_pareto['diverged']}, whereas Normalized Sign-LMS remained stable (diverged={robust_pareto['diverged']}).")

def test_adaptive_beamformer_rmb_and_nulling():
    beamformer = AdaptiveBeamformer(num_elements=8)

    # 1. Sufficient snapshots (K = 64 >= 2 * 8 = 16, satisfying RMB rule)
    res_rmb = beamformer.evaluate_beamformer(num_snapshots=64, desired_theta_deg=0.0, look_mismatch_deg=0.0)
    assert res_rmb["rmb_rule_satisfied"] is True
    # Null suppression at interferer (30 deg) should be deep (> 20 dB)
    assert res_rmb["null_suppression_depth_db"] > 20.0, f"Expected > 20 dB null, got {res_rmb['null_suppression_depth_db']:.1f} dB"
    # Desired signal gain should be near 0 dB (distortionless constraint w^H * a = 1)
    assert abs(res_rmb["desired_signal_gain_db"]) < 0.5, f"Distortionless violation: {res_rmb['desired_signal_gain_db']:.2f} dB"
    print(f"[PASS] test_adaptive_beamformer_rmb: K=64 snapshots satisfied RMB rule; null suppression={res_rmb['null_suppression_depth_db']:.1f} dB, signal gain={res_rmb['desired_signal_gain_db']:.2f} dB.")

    # 2. Steering Vector Look Mismatch (pointing error of 2.5 degrees)
    res_mismatch = beamformer.evaluate_beamformer(num_snapshots=64, desired_theta_deg=0.0, look_mismatch_deg=2.5)
    # The beamformer treats the desired signal as an interferer and places a null on it!
    assert res_mismatch["desired_signal_gain_db"] < -10.0, f"Expected signal cancellation, got {res_mismatch['desired_signal_gain_db']:.1f} dB"
    print(f"[PASS] test_adaptive_beamformer_mismatch: 2.5 deg look mismatch caused catastrophic signal cancellation ({res_mismatch['desired_signal_gain_db']:.1f} dB gain on desired signal).")

def test_beamformer_diagonal_loading():
    beamformer = AdaptiveBeamformer(num_elements=8)
    # Under-sampled snapshot regime: K = 10 < 2M
    res_unloaded = beamformer.evaluate_beamformer(num_snapshots=10, diagonal_load_alpha=0.0)
    res_loaded = beamformer.evaluate_beamformer(num_snapshots=10, diagonal_load_alpha=10.0)

    # Diagonal loading stabilizes the weight norm and gain
    assert abs(res_loaded["desired_signal_gain_db"]) < abs(res_unloaded["desired_signal_gain_db"]) + 2.0
    print(f"[PASS] test_beamformer_diagonal_loading: Diagonal loading stabilized under-sampled snapshot regime (K=10).")

if __name__ == "__main__":
    test_fxlms_causality_and_attenuation()
    test_fxlms_pareto_impulsive_stability()
    test_adaptive_beamformer_rmb_and_nulling()
    test_beamformer_diagonal_loading()
    print("\nAll signal processing unit tests passed successfully!")
