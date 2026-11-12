"""
harmonic-knapsack/core/harmonic_representation.py
Module for Spectral Decomposition, Fourier Series Representation, and Periodicity Extraction.

Implements:
  - Complex and Trigonometric Fourier Series decomposition
  - Extraction of top-K dominant harmonics (frequency, amplitude, phase)
  - Reconstruction of continuous/discrete periodic load profiles
  - FTIO-style periodicity confidence score from Power Spectral Density (PSD)
  - Spectral windowing (Hann/Boxcar) and sub-harmonic fundamental validation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set

class HarmonicComponent:
    """Represents a single sinusoidal Fourier harmonic."""
    def __init__(self, harmonic_order: int, frequency_hz: float,
                 amplitude: float, phase_rad: float):
        self.harmonic_order = int(harmonic_order)
        self.frequency_hz = float(frequency_hz)
        self.amplitude = float(amplitude)
        self.phase_rad = float(phase_rad)

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluates A * cos(2*pi*f*t + phi)."""
        omega = 2.0 * np.pi * self.frequency_hz
        return self.amplitude * np.cos(omega * t + self.phase_rad)

    def __repr__(self) -> str:
        return (f"Harmonic(order={self.harmonic_order}, f={self.frequency_hz:.4f}Hz, "
                f"amp={self.amplitude:.4f}, phase={self.phase_rad:.4f}rad)")


class HarmonicProfile:
    """
    Compact spectral representation of a periodic workload curve.
    Stores DC mean component and top-K dominant harmonic components.
    """
    def __init__(self, period_seconds: float, dc_mean: float,
                 harmonics: List[HarmonicComponent],
                 periodicity_confidence: float = 1.0):
        self.period_seconds = float(period_seconds)
        self.fundamental_frequency = 1.0 / self.period_seconds if self.period_seconds > 0 else 0.0
        self.dc_mean = float(dc_mean)
        self.harmonics = harmonics
        self.periodicity_confidence = float(periodicity_confidence)

    def evaluate(self, t: np.ndarray, phase_offset_seconds: float = 0.0) -> np.ndarray:
        """
        Reconstructs the load curve at time points t with an optional start offset tau.
        
        Note:
          The output is clamped using np.maximum(0.0, load) to enforce physical
          non-negativity of resource utilization and power draw.
          Clipping is a non-linear operation, meaning the clamped time-domain signal
          contains residual higher-order distortion harmonics compared to the
          untruncated linear Fourier series.
        """
        load = np.full_like(t, self.dc_mean, dtype=np.float64)
        tau = float(phase_offset_seconds)
        for h in self.harmonics:
            omega = 2.0 * np.pi * h.frequency_hz
            # Time shift t -> t - tau corresponds to phase angle shift -omega * tau
            load += h.amplitude * np.cos(omega * (t - tau) + h.phase_rad)
        return np.maximum(0.0, load)

    @classmethod
    def from_timeseries(cls, series: np.ndarray, sampling_interval_sec: float,
                        assumed_period_sec: Optional[float] = None,
                        num_harmonics: int = 4,
                        window: Optional[str] = None) -> 'HarmonicProfile':
        """
        Decomposes a discrete timeseries into its dominant Fourier components.
        Uses FFT, optional windowing (e.g. 'hann'), sub-harmonic fundamental verification,
        and unique-bin energy tracking to prevent inflated confidence scores.
        """
        N = len(series)
        assert N > 0, "Time series must contain at least 1 sample"
        T_total = N * sampling_interval_sec
        dc_mean = float(np.mean(series))
        
        zero_mean = series - dc_mean
        freqs = np.fft.rfftfreq(N, d=sampling_interval_sec)

        # Spectral windowing to mitigate leakage
        if window == "hann":
            win = np.hanning(N)
            coherent_gain = float(np.mean(win))
            fft_vals = np.fft.rfft(zero_mean * win)
            scale = 2.0 / (N * coherent_gain)
        else:
            fft_vals = np.fft.rfft(zero_mean)
            scale = 2.0 / N
        
        # Power Spectral Density
        psd = np.abs(fft_vals) ** 2
        
        if assumed_period_sec is not None:
            period_sec = float(assumed_period_sec)
            f0 = 1.0 / period_sec
        else:
            # Auto-detect fundamental frequency with sub-harmonic overtone checking
            if len(psd) > 2:
                peak_idx = 1 + int(np.argmax(psd[1:]))
                f_peak = freqs[peak_idx]

                # Sub-harmonic check: check if half-frequency (f_peak/2) has strong energy
                # indicating that the peak is the 2nd harmonic rather than fundamental
                sub_half_idx = int(round(peak_idx / 2))
                if sub_half_idx > 0 and psd[sub_half_idx] > 0.25 * psd[peak_idx]:
                    f0 = freqs[sub_half_idx]
                else:
                    f0 = f_peak
                period_sec = 1.0 / f0 if f0 > 0 else T_total
            else:
                f0 = 1.0 / T_total
                period_sec = T_total

        total_ac_energy = float(np.sum(psd[1:])) + 1e-12
        
        harmonics = []
        unique_extracted_bins: Set[int] = set()

        for k in range(1, num_harmonics + 1):
            target_f = k * f0
            # Find closest FFT frequency bin
            closest_bin = int(np.argmin(np.abs(freqs - target_f)))
            if closest_bin < len(fft_vals):
                amp = float(scale * np.abs(fft_vals[closest_bin]))
                phase = float(np.angle(fft_vals[closest_bin]))
                harmonics.append(HarmonicComponent(
                    harmonic_order=k,
                    frequency_hz=target_f,
                    amplitude=amp,
                    phase_rad=phase
                ))
                if closest_bin > 0:
                    unique_extracted_bins.add(closest_bin)

        # Confidence: Sum energy over UNIQUE bins to strictly prevent double-counting
        extracted_energy = sum(float(psd[b]) for b in unique_extracted_bins if b < len(psd))
        confidence = float(np.clip(extracted_energy / total_ac_energy, 0.0, 1.0))

        return cls(
            period_seconds=period_sec,
            dc_mean=dc_mean,
            harmonics=harmonics,
            periodicity_confidence=confidence
        )
