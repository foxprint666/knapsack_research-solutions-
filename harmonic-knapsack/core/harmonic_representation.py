"""
harmonic-knapsack/core/harmonic_representation.py
Module for Spectral Decomposition, Fourier Series Representation, and Periodicity Extraction.
Implements:
  - Complex and Trigonometric Fourier Series decomposition
  - Extraction of top-K dominant harmonics (frequency, amplitude, phase)
  - Reconstruction of continuous/discrete periodic load profiles
  - FTIO-style periodicity confidence score from Power Spectral Density (PSD)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

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
                            num_harmonics: int = 4) -> 'HarmonicProfile':
        """
        Decomposes a discrete timeseries into its dominant Fourier components.
        Uses FFT and peak picking to compute amplitude and phase.
        """
        N = len(series)
        T_total = N * sampling_interval_sec
        dc_mean = float(np.mean(series))
        
        # Zero-mean signal for FFT
        zero_mean = series - dc_mean
        fft_vals = np.fft.rfft(zero_mean)
        freqs = np.fft.rfftfreq(N, d=sampling_interval_sec)
        
        # Power Spectral Density
        psd = np.abs(fft_vals) ** 2
        
        if assumed_period_sec is not None:
            period_sec = float(assumed_period_sec)
            f0 = 1.0 / period_sec
        else:
            # Auto-detect dominant fundamental frequency (excluding DC bin 0)
            if len(psd) > 1:
                peak_idx = 1 + np.argmax(psd[1:])
                f0 = freqs[peak_idx]
                period_sec = 1.0 / f0 if f0 > 0 else T_total
            else:
                f0 = 1.0 / T_total
                period_sec = T_total

        # FTIO-style periodicity confidence score: ratio of energy in dominant harmonics vs total AC energy
        total_ac_energy = np.sum(psd[1:]) + 1e-12
        
        harmonics = []
        for k in range(1, num_harmonics + 1):
            target_f = k * f0
            # Find closest FFT frequency bin
            closest_bin = int(np.argmin(np.abs(freqs - target_f)))
            if closest_bin < len(fft_vals):
                # Normalized amplitude for single-sided spectrum
                amp = (2.0 * np.abs(fft_vals[closest_bin])) / N
                # Phase angle
                phase = np.angle(fft_vals[closest_bin])
                harmonics.append(HarmonicComponent(
                    harmonic_order=k,
                    frequency_hz=target_f,
                    amplitude=amp,
                    phase_rad=phase
                ))

        # Sum of energy captured by the extracted harmonics
        extracted_bins = [int(np.argmin(np.abs(freqs - h.frequency_hz))) for h in harmonics]
        extracted_energy = np.sum([psd[b] for b in extracted_bins if b < len(psd)])
        confidence = float(np.clip(extracted_energy / total_ac_energy, 0.0, 1.0))

        return cls(
            period_seconds=period_sec,
            dc_mean=dc_mean,
            harmonics=harmonics,
            periodicity_confidence=confidence
        )
