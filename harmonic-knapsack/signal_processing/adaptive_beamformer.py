"""
harmonic-knapsack/signal_processing/adaptive_beamformer.py
Signal Processing Model: Adaptive Beamforming & Spatial Nulling (MVDR / Capon).

Demonstrates:
  1. Antenna array spatial filtering: y(n) = w^H * x(n).
  2. Sample Matrix Inversion (SMI) and the Reed-Mallett-Brennan (RMB) Rule (K >= 2M).
  3. Diagonal Loading Regularization: R_loaded = R_hat + alpha * I.
  4. Catastrophic Signal Cancellation under Steering Vector Mismatch.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class AdaptiveBeamformer:
    """
    Uniform Linear Array (ULA) MVDR / Capon Adaptive Beamformer.
    """
    def __init__(self, num_elements: int = 8, element_spacing_wavelengths: float = 0.5):
        self.M = int(num_elements)
        self.d_lambda = float(element_spacing_wavelengths)

    def steering_vector(self, theta_deg: float) -> np.ndarray:
        """
        Computes the spatial steering vector a(theta) for a plane wave at angle theta.
        a_m = exp(-j * 2 * pi * d_lambda * m * sin(theta_rad))
        """
        theta_rad = np.radians(theta_deg)
        m = np.arange(self.M)
        phase = -2.0 * np.pi * self.d_lambda * m * np.sin(theta_rad)
        return np.exp(1j * phase)

    def generate_snapshots(self, num_snapshots: int,
                           desired_theta_deg: float = 0.0,
                           interferers: Optional[List[Tuple[float, float]]] = None,
                           snr_db: float = 10.0,
                           seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes snapshot matrix X in C^(M x K).
        Returns: (X_total, s_desired)
        """
        rng = np.random.default_rng(seed)
        K = int(num_snapshots)
        
        # Default interferers: (angle_deg, inr_db)
        if interferers is None:
            interferers = [(30.0, 25.0), (-45.0, 20.0)]

        # Desired signal s(n)
        s_power = 10.0 ** (snr_db / 10.0)
        s = np.sqrt(s_power / 2.0) * (rng.normal(size=K) + 1j * rng.normal(size=K))
        a_sig = self.steering_vector(desired_theta_deg)

        # Build total received array snapshot
        X = np.outer(a_sig, s)

        # Add interferers
        for theta_i, inr_db in interferers:
            i_power = 10.0 ** (inr_db / 10.0)
            int_sig = np.sqrt(i_power / 2.0) * (rng.normal(size=K) + 1j * rng.normal(size=K))
            a_int = self.steering_vector(theta_i)
            X += np.outer(a_int, int_sig)

        # Add spatially white thermal noise (sigma^2 = 1.0)
        noise = (rng.normal(size=(self.M, K)) + 1j * rng.normal(size=(self.M, K))) / np.sqrt(2.0)
        X += noise

        return X, s

    def compute_mvdr_weights(self, X: np.ndarray,
                             look_direction_deg: float = 0.0,
                             diagonal_load_alpha: float = 0.0) -> np.ndarray:
        """
        Computes Capon / MVDR weight vector:
          w = (R^-1 * a) / (a^H * R^-1 * a)
        """
        K = X.shape[1]
        # Sample covariance matrix
        R_hat = (X @ X.conj().T) / K

        # Optional diagonal loading regularization
        if diagonal_load_alpha > 0.0:
            R_hat += diagonal_load_alpha * np.eye(self.M)

        a_look = self.steering_vector(look_direction_deg)
        
        try:
            R_inv = np.linalg.pinv(R_hat)
            numerator = R_inv @ a_look
            denominator = a_look.conj().T @ numerator
            w = numerator / denominator
            return w
        except np.linalg.LinAlgError:
            return np.ones(self.M, dtype=complex) / self.M

    def evaluate_beamformer(self, num_snapshots: int = 64,
                            desired_theta_deg: float = 0.0,
                            look_mismatch_deg: float = 0.0,
                            diagonal_load_alpha: float = 0.0,
                            interferers: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Evaluates beamformer output SINR and spatial nulling depth.
        """
        if interferers is None:
            interferers = [(30.0, 25.0), (-45.0, 20.0)]

        X, s = self.generate_snapshots(
            num_snapshots=num_snapshots,
            desired_theta_deg=desired_theta_deg,
            interferers=interferers,
            snr_db=10.0
        )

        look_angle = desired_theta_deg + look_mismatch_deg
        w = self.compute_mvdr_weights(X, look_direction_deg=look_angle, diagonal_load_alpha=diagonal_load_alpha)

        # Compute beampattern across angles [-90, +90] deg
        angles = np.linspace(-90.0, 90.0, 361)
        beampattern_db = []
        for ang in angles:
            a = self.steering_vector(ang)
            response = float(np.abs(w.conj().T @ a) ** 2)
            beampattern_db.append(10.0 * np.log10(max(1e-12, response)))

        beampattern_db = np.array(beampattern_db)
        
        # Gain in desired signal direction
        gain_desired = float(np.interp(desired_theta_deg, angles, beampattern_db))
        
        # Null depth at primary interferer
        primary_int_angle = interferers[0][0]
        gain_interferer = float(np.interp(primary_int_angle, angles, beampattern_db))
        null_depth_db = gain_desired - gain_interferer

        rmb_satisfied = num_snapshots >= (2 * self.M)

        return {
            "num_elements": self.M,
            "num_snapshots": num_snapshots,
            "rmb_rule_satisfied": rmb_satisfied,
            "desired_angle_deg": desired_theta_deg,
            "look_mismatch_deg": look_mismatch_deg,
            "diagonal_load_alpha": diagonal_load_alpha,
            "desired_signal_gain_db": gain_desired,
            "interferer_null_gain_db": gain_interferer,
            "null_suppression_depth_db": null_depth_db,
            "signal_cancelled": (look_mismatch_deg != 0.0 and gain_desired < -10.0)
        }
