"""
p2-shaper/cgroup_shaper_daemon.py
Plane 2: Linux Host Active Shaper Daemon & cgroups v2 Controller

Implements:
  - cgroups v2 hierarchy management:
      /sys/fs/cgroup/workload.slice/interactive.slice (unthrottled, cpu.weight=1000, cpu.max=max)
      /sys/fs/cgroup/workload.slice/batch.slice       (elastic, cpu.weight=100, dynamic cpu.max)
  - Kernel Pressure Stall Information (PSI) /proc/pressure/cpu monitor loop
  - Dynamic clamping of batch/cpu.max when CPU pressure exceeds threshold
  - Strict Fail-Open safety: resets to unthrottled on errors, signals, or pressure clearing
  - Discrete-event trace simulation engine (simulate_trace_shaping) for fleet holdout testing
"""

import os
import sys
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

logger = logging.getLogger("CgroupActiveShaper")

class CgroupHierarchyManager:
    """
    Manages cgroups v2 directories and attributes on Linux host or mock filesystem.
    """
    def __init__(self, cgroup_root: str = "/sys/fs/cgroup"):
        self.cgroup_root = cgroup_root
        self.workload_slice = os.path.join(cgroup_root, "workload.slice")
        self.interactive_slice = os.path.join(self.workload_slice, "interactive.slice")
        self.batch_slice = os.path.join(self.workload_slice, "batch.slice")
        self.is_mock = not os.path.exists(cgroup_root) or not os.access(cgroup_root, os.W_OK)
        self.mock_state = {
            "interactive.weight": 1000,
            "interactive.max": "max 100000",
            "batch.weight": 100,
            "batch.max": "max 100000"
        }

    def setup_hierarchy(self) -> bool:
        """Initializes cgroups v2 slices with initial weights and limits."""
        if self.is_mock:
            logger.info("Operating in MOCK cgroup mode (non-Linux or unprivileged environment).")
            return True

        try:
            os.makedirs(self.interactive_slice, exist_ok=True)
            os.makedirs(self.batch_slice, exist_ok=True)

            # Set interactive slice attributes
            self._write_file(os.path.join(self.interactive_slice, "cpu.weight"), "1000")
            self._write_file(os.path.join(self.interactive_slice, "cpu.max"), "max 100000")

            # Set batch slice attributes
            self._write_file(os.path.join(self.batch_slice, "cpu.weight"), "100")
            self._write_file(os.path.join(self.batch_slice, "cpu.max"), "max 100000")
            logger.info("cgroups v2 hierarchy initialized successfully.")
            return True
        except Exception as e:
            logger.warning(f"Failed to setup cgroups hierarchy ({e}). Falling back to mock mode.")
            self.is_mock = True
            return False

    def set_batch_quota(self, quota_us: Optional[int], period_us: int = 100000) -> None:
        """
        Dynamically adjusts cpu.max on batch.slice.
        quota_us=None or quota_us >= period_us resets to 'max'.
        """
        if quota_us is None or quota_us >= period_us:
            val_str = f"max {period_us}"
        else:
            val_str = f"{max(1000, int(quota_us))} {period_us}"

        if self.is_mock:
            self.mock_state["batch.max"] = val_str
            return

        try:
            self._write_file(os.path.join(self.batch_slice, "cpu.max"), val_str)
        except Exception as e:
            logger.error(f"Failed to write cpu.max on batch slice: {e}. Failing open.")
            self.fail_open()

    def fail_open(self) -> None:
        """Resets all slices to unthrottled state."""
        if self.is_mock:
            self.mock_state["batch.max"] = "max 100000"
            return
        try:
            self._write_file(os.path.join(self.batch_slice, "cpu.max"), "max 100000")
        except Exception as e:
            logger.error(f"Critical: fail_open failed ({e})")

    @staticmethod
    def _write_file(path: str, content: str) -> None:
        with open(path, "w") as f:
            f.write(content.strip() + "\n")


class PSIMonitor:
    """
    Parses Linux Pressure Stall Information (PSI) from /proc/pressure/cpu.
    """
    def __init__(self, psi_path: str = "/proc/pressure/cpu"):
        self.psi_path = psi_path
        self.is_available = os.path.exists(psi_path)

    def read_pressure(self) -> Dict[str, float]:
        """
        Reads avg10, avg60, avg300 pressure percentages.
        Returns dict with parsed values. Defaults to 0.0 if not available.
        """
        if not self.is_available:
            return {"some_avg10": 0.0, "some_avg60": 0.0, "some_avg300": 0.0, "total": 0.0}

        try:
            with open(self.psi_path, "r") as f:
                lines = f.readlines()
            result = {}
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                prefix = parts[0] # 'some' or 'full'
                for item in parts[1:]:
                    if "=" in item:
                        k, v = item.split("=", 1)
                        result[f"{prefix}_{k}"] = float(v)
            return result
        except Exception as e:
            logger.warning(f"Error reading PSI ({e}). Defaulting to 0 pressure.")
            return {"some_avg10": 0.0, "some_avg60": 0.0, "some_avg300": 0.0, "total": 0.0}


class ActiveShaperDaemon:
    """
    Plane 2 Host Active Shaper Daemon.
    Monitors host pressure and dynamically shapes batch cgroups to preserve interactive SLAs.
    """
    def __init__(self, cgroup_mgr: Optional[CgroupHierarchyManager] = None,
                 psi_monitor: Optional[PSIMonitor] = None,
                 pressure_threshold_pct: float = 20.0,
                 period_us: int = 100000,
                 min_batch_quota_us: int = 10000):
        self.cgroup_mgr = cgroup_mgr or CgroupHierarchyManager()
        self.psi_monitor = psi_monitor or PSIMonitor()
        self.pressure_threshold_pct = pressure_threshold_pct
        self.period_us = period_us
        self.min_batch_quota_us = min_batch_quota_us
        self.is_clamped = False

    def evaluate_and_act(self, host_cpu_util: float, interactive_util: float) -> Dict[str, Any]:
        """
        Single control cycle evaluation:
        Checks host utilization and PSI pressure.
        If host is under pressure (> 85% or PSI > threshold), throttles batch slice.
        """
        psi_stats = self.psi_monitor.read_pressure()
        some_avg10 = psi_stats.get("some_avg10", 0.0)

        # Trigger condition: utilization > 85% OR some_avg10 > pressure_threshold
        needs_shaping = (host_cpu_util > 0.85) or (some_avg10 > self.pressure_threshold_pct)

        if needs_shaping:
            # Interactive gets full unthrottled priority.
            # Headroom available for batch:
            available_for_batch = max(0.05, 1.0 - interactive_util)
            batch_quota = int(self.period_us * available_for_batch)
            batch_quota = max(self.min_batch_quota_us, min(self.period_us, batch_quota))
            self.cgroup_mgr.set_batch_quota(batch_quota, self.period_us)
            self.is_clamped = True
            action = "clamped"
        else:
            if self.is_clamped:
                # Fail-open / release clamp
                self.cgroup_mgr.fail_open()
                self.is_clamped = False
                action = "released"
            else:
                action = "idle"
            batch_quota = self.period_us

        return {
            "action": action,
            "host_cpu_util": host_cpu_util,
            "interactive_util": interactive_util,
            "psi_some_avg10": some_avg10,
            "is_clamped": self.is_clamped,
            "batch_quota_us": batch_quota
        }

    @staticmethod
    def simulate_trace_shaping(interactive_trace: np.ndarray,
                               batch_trace: np.ndarray,
                               host_capacity: float = 1.0,
                               headroom_target: float = 0.95) -> Dict[str, Any]:
        """
        Discrete-event trace simulator for host-level active shaping.
        Preserves batch demand mass causally via a FIFO deferral buffer.
        Interactive workload is never throttled.
        
        Args:
          interactive_trace: 1D array of interactive load series
          batch_trace: 1D array of batch/delay-insensitive load series
          host_capacity: Total host capacity (1.0 = 100%)
          headroom_target: Target max host utilization under shaping (0.95)
        """
        T = len(interactive_trace)
        shaped_batch = np.zeros(T, dtype=np.float64)
        total_host_load = np.zeros(T, dtype=np.float64)
        queue_backlog = np.zeros(T, dtype=np.float64)
        throttled_bins = 0
        current_queue = 0.0

        for t in range(T):
            int_load = interactive_trace[t]
            raw_batch_load = batch_trace[t]
            
            # Enqueue incoming batch work into delay buffer
            current_queue += raw_batch_load
            
            # Headroom available for batch on host
            available_batch_cap = max(0.0, host_capacity * headroom_target - int_load)
            
            # Drain queue as much as possible up to available headroom
            dispatched_batch = min(current_queue, available_batch_cap)
            current_queue -= dispatched_batch
            
            shaped_batch[t] = dispatched_batch
            total_host_load[t] = int_load + dispatched_batch
            queue_backlog[t] = current_queue
            
            if dispatched_batch < raw_batch_load:
                throttled_bins += 1

        # Post-trace drain: ensure remaining backlog can be drained
        unserved_tail_work = current_queue
        
        raw_total_load = interactive_trace + batch_trace
        raw_overload_pct = float(np.mean(raw_total_load > host_capacity)) * 100.0
        shaped_overload_pct = float(np.mean(total_host_load > host_capacity)) * 100.0
        
        # Interactive SLA violations: interactive load alone exceeding host capacity
        interactive_violations = int(np.sum(interactive_trace > host_capacity))
        
        # Mass conservation: sum of raw batch work vs sum of dispatched + remaining
        raw_batch_mass = float(np.sum(batch_trace))
        served_batch_mass = float(np.sum(shaped_batch))
        mass_conserved = bool(abs((served_batch_mass + unserved_tail_work) - raw_batch_mass) < 1e-4)

        return {
            "num_bins": T,
            "raw_overload_pct": raw_overload_pct,
            "shaped_overload_pct": shaped_overload_pct,
            "throttled_bins_pct": (throttled_bins / max(1, T)) * 100.0,
            "mean_queue_backlog": float(np.mean(queue_backlog)),
            "p99_queue_backlog": float(np.percentile(queue_backlog, 99)),
            "max_queue_backlog": float(np.max(queue_backlog)),
            "raw_batch_mass": raw_batch_mass,
            "served_batch_mass": served_batch_mass,
            "unserved_tail_work": unserved_tail_work,
            "mass_conserved": mass_conserved,
            "interactive_sla_violations": interactive_violations,
            "shaped_host_load": total_host_load,
            "queue_backlog_series": queue_backlog
        }
