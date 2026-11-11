"""
harmonic-knapsack/production/__init__.py
Production-grade runtime modules for real-world deployment:
1. Real-Time Telemetry Ingestion (The Sensor)
2. Non-Intrusive Actuation Hook (The Actuator)
3. Rate-Diversity & Higher-Harmonic Handling (The Controller)
"""

from .telemetry_sensor import TelemetrySensor, StreamingSpectralAnalyzer, PowerSampleRingBuffer
from .nccl_actuator_hook import NCCLActuatorHook, MicroBarrierPhaseController, ActuationPlan
from .rate_diversity_controller import RateDiversityController, HarmonicPeriodQuantizer, DynamicBatchPacer

__all__ = [
    "TelemetrySensor",
    "StreamingSpectralAnalyzer",
    "PowerSampleRingBuffer",
    "NCCLActuatorHook",
    "MicroBarrierPhaseController",
    "ActuationPlan",
    "RateDiversityController",
    "HarmonicPeriodQuantizer",
    "DynamicBatchPacer"
]
