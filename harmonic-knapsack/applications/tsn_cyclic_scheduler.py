"""
harmonic-knapsack/applications/tsn_cyclic_scheduler.py
Application Model: Time-Sensitive Networking (IEEE 802.1Qbv) Cyclic Queue Gating.

Models:
  - IEEE 802.1Qbv Time-Aware Shaper (TAS) with repeating Gate Control List (GCL).
  - Synthesis of non-overlapping transmission windows for critical periodic flows.
  - Phase-Drift impact: IEEE 1588 PTP clock skew causing gate-boundary violations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class TSNFlow:
    """Represents a time-triggered critical real-time network flow."""
    def __init__(self, flow_id: int, name: str,
                 period_us: float,
                 transmission_duration_us: float,
                 payload_bytes: int = 1500):
        self.flow_id = flow_id
        self.name = name
        self.period_us = float(period_us)
        self.duration_us = float(transmission_duration_us)
        self.payload_bytes = payload_bytes

class TSNCyclicScheduler:
    """
    Synthesizes Gate Control Lists (GCL) and simulates schedule execution under clock drift.
    """
    def __init__(self, cycle_time_us: float = 1000.0, guard_band_us: float = 5.0):
        self.cycle_time_us = float(cycle_time_us)
        self.guard_band_us = float(guard_band_us)

    def synthesize_schedule(self, flows: List[TSNFlow]) -> Dict[str, Any]:
        """
        Synthesizes start offsets for flows to ensure non-overlapping transmission windows.
        """
        sorted_flows = sorted(flows, key=lambda f: f.duration_us, reverse=True)
        schedule_entries = []
        current_cursor_us = 0.0

        for f in sorted_flows:
            allocated_start_us = current_cursor_us
            allocated_end_us = allocated_start_us + f.duration_us
            
            if allocated_end_us + self.guard_band_us > self.cycle_time_us:
                # Cycle full: cannot admit without overlap
                return {
                    "schedulable": False,
                    "cycle_time_us": self.cycle_time_us,
                    "failed_flow": f.name,
                    "schedule": schedule_entries
                }

            schedule_entries.append({
                "flow_id": f.flow_id,
                "name": f.name,
                "start_us": allocated_start_us,
                "end_us": allocated_end_us,
                "duration_us": f.duration_us
            })
            current_cursor_us = allocated_end_us + self.guard_band_us

        return {
            "schedulable": True,
            "cycle_time_us": self.cycle_time_us,
            "guard_band_us": self.guard_band_us,
            "total_allocated_us": current_cursor_us,
            "utilization_pct": (current_cursor_us / self.cycle_time_us) * 100.0,
            "schedule": schedule_entries
        }

    def simulate_clock_drift_collision(self, schedule_info: Dict[str, Any],
                                       drift_rate_ppm: float = 100.0,
                                       elapsed_seconds: float = 1.0) -> Dict[str, Any]:
        """
        Simulates how PTP clock drift accumulates over time, potentially causing packets
        to spill past closed gates into the adjacent flow's window.
        """
        if not schedule_info.get("schedulable", False):
            return {"error": "Schedule is not schedulable"}

        # Drift in microseconds accumulated: seconds * (PPM * 1e-6) * 1e6 = seconds * PPM
        accumulated_drift_us = elapsed_seconds * drift_rate_ppm
        guard_band = schedule_info["guard_band_us"]
        
        # Check if drift exceeds guard band
        collision_detected = accumulated_drift_us > guard_band
        overflow_us = max(0.0, accumulated_drift_us - guard_band)

        return {
            "elapsed_seconds": elapsed_seconds,
            "clock_drift_ppm": drift_rate_ppm,
            "accumulated_drift_us": accumulated_drift_us,
            "guard_band_us": guard_band,
            "collision_detected": collision_detected,
            "overflow_into_adjacent_gate_us": overflow_us,
            "determinism_preserved": not collision_detected
        }
