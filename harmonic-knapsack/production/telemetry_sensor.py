"""
harmonic-knapsack/production/telemetry_sensor.py
Pillar 1: Real-Time Telemetry Ingestion (The Sensor).

Collects sub-second GPU power telemetry (NVIDIA DCGM / NVML or Rack PDU)
into a sliding ring-buffer and continuously extracts HarmonicProfiles
and phase drift using sliding-window FFT.
"""

import time
import math
import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from core.harmonic_representation import HarmonicProfile, HarmonicComponent

class PowerSampleRingBuffer:
    """
    Fixed-size circular ring buffer for sub-second power measurements.
    Thread-safe for concurrent sensor ingestion and FFT background analysis.
    """
    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self.timestamps = np.zeros(capacity, dtype=np.float64)
        self.power_kw = np.zeros(capacity, dtype=np.float64)
        self.head = 0
        self.count = 0
        self._lock = threading.Lock()

    def append(self, timestamp_sec: float, power_kw: float) -> None:
        """Appends a single (timestamp, power_kw) sample."""
        with self._lock:
            idx = (self.head + self.count) % self.capacity
            self.timestamps[idx] = timestamp_sec
            self.power_kw[idx] = power_kw
            if self.count < self.capacity:
                self.count += 1
            else:
                self.head = (self.head + 1) % self.capacity

    def get_snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns ordered copies of (timestamps, power_kw)."""
        with self._lock:
            if self.count == 0:
                return np.array([]), np.array([])
            indices = (self.head + np.arange(self.count)) % self.capacity
            return self.timestamps[indices].copy(), self.power_kw[indices].copy()

    def __len__(self) -> int:
        with self._lock:
            return self.count


class StreamingSpectralAnalyzer:
    """
    Computes online Fourier series decomposition and tracks phase drift over time.
    """
    def __init__(self, sampling_interval_sec: float = 0.02,
                 window_duration_sec: float = 5.0,
                 ema_alpha: float = 0.25):
        self.dt = float(sampling_interval_sec)
        self.window_sec = float(window_duration_sec)
        self.ema_alpha = float(ema_alpha)
        self.current_profile: Optional[HarmonicProfile] = None
        self.smoothed_confidence = 0.0
        self.smoothed_period_sec: Optional[float] = None
        self.smoothed_phase_rad: Optional[float] = None

    def update_from_buffer(self, ring_buffer: PowerSampleRingBuffer,
                           assumed_period_sec: Optional[float] = None) -> Optional[HarmonicProfile]:
        """
        Extracts recent window of samples, resamples to uniform grid,
        and computes online HarmonicProfile.
        """
        ts, powers = ring_buffer.get_snapshot()
        if len(ts) < 16:
            return None

        # Filter to the recent window
        t_max = ts[-1]
        t_min = t_max - self.window_sec
        mask = ts >= t_min
        if np.sum(mask) < 16:
            return None

        ts_win = ts[mask]
        p_win = powers[mask]

        # Resample onto strictly uniform grid with dt
        t_uniform = np.arange(ts_win[0], ts_win[-1], self.dt)
        if len(t_uniform) < 16:
            return None

        p_uniform = np.interp(t_uniform, ts_win, p_win)

        # Online FFT decomposition
        profile = HarmonicProfile.from_timeseries(
            series=p_uniform,
            sampling_interval_sec=self.dt,
            assumed_period_sec=assumed_period_sec,
            num_harmonics=4
        )

        # Update EMA smoothed estimates
        if self.current_profile is None:
            self.smoothed_confidence = profile.periodicity_confidence
            self.smoothed_period_sec = profile.period_seconds
        else:
            self.smoothed_confidence = (
                (1.0 - self.ema_alpha) * self.smoothed_confidence
                + self.ema_alpha * profile.periodicity_confidence
            )
            self.smoothed_period_sec = (
                (1.0 - self.ema_alpha) * self.smoothed_period_sec
                + self.ema_alpha * profile.period_seconds
            )

        if profile.harmonics:
            curr_phase = profile.harmonics[0].phase_rad
            if self.smoothed_phase_rad is None:
                self.smoothed_phase_rad = curr_phase
            else:
                # Circular phase smoothing
                diff = (curr_phase - self.smoothed_phase_rad + np.pi) % (2.0 * np.pi) - np.pi
                self.smoothed_phase_rad = (self.smoothed_phase_rad + self.ema_alpha * diff) % (2.0 * np.pi)

        self.current_profile = profile
        return profile

    def compute_phase_drift(self, target_phase_rad: float) -> float:
        """
        Calculates instantaneous signed phase error delta_phi in [-pi, pi].
        Positive delta means running ahead, negative means lagging.
        """
        if self.smoothed_phase_rad is None:
            return 0.0
        delta = (self.smoothed_phase_rad - target_phase_rad + np.pi) % (2.0 * np.pi) - np.pi
        return float(delta)


class TelemetrySensor:
    """
    High-level Sensor abstraction representing an active telemetry stream
    from an NVML/DCGM daemon or rack PDU.
    """
    def __init__(self, job_id: int, sampling_rate_hz: float = 50.0):
        self.job_id = job_id
        self.sampling_interval_sec = 1.0 / max(1.0, sampling_rate_hz)
        self.buffer = PowerSampleRingBuffer(capacity=2048)
        self.analyzer = StreamingSpectralAnalyzer(
            sampling_interval_sec=self.sampling_interval_sec,
            window_duration_sec=4.0
        )

    def ingest_reading(self, power_kw: float, timestamp_sec: Optional[float] = None) -> None:
        """Simulates ingestion of one hardware telemetry reading."""
        t = time.time() if timestamp_sec is None else timestamp_sec
        self.buffer.append(t, power_kw)

    def poll_spectral_state(self, assumed_period_sec: Optional[float] = None) -> Dict[str, Any]:
        """Polls the latest spectral estimate and telemetry status."""
        prof = self.analyzer.update_from_buffer(self.buffer, assumed_period_sec=assumed_period_sec)
        return {
            "job_id": self.job_id,
            "sample_count": len(self.buffer),
            "confidence": self.analyzer.smoothed_confidence,
            "detected_period_sec": self.analyzer.smoothed_period_sec,
            "detected_phase_rad": self.analyzer.smoothed_phase_rad,
            "has_profile": prof is not None
        }
