"""
harmonic-knapsack/core/__init__.py
Core engine for Harmonic Temporal Knapsack and Phase-Offset Optimization.
"""

from .harmonic_representation import HarmonicComponent, HarmonicProfile
from .phase_shift_math import PhaseShiftEngine, compute_hyperperiod
from .temporal_knapsack_solver import HarmonicItem, TemporalHarmonicKnapsackSolver

__all__ = [
    "HarmonicComponent",
    "HarmonicProfile",
    "PhaseShiftEngine",
    "compute_hyperperiod",
    "HarmonicItem",
    "TemporalHarmonicKnapsackSolver"
]
