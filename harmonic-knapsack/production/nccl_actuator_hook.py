"""
harmonic-knapsack/production/nccl_actuator_hook.py
Pillar 2: The Non-Intrusive Actuation Hook (The Actuator).

Implements phase shifting (tau_i) in PyTorch distributed training jobs
via micro-barrier delays at the completion of collective communication (AllReduce)
or NCCL pacing adjustments, amortizing the timing shift to achieve near-zero (<0.1%)
throughput degradation.
"""

import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

@dataclass
class ActuationPlan:
    """Defines an upcoming phase-shift actuation strategy."""
    target_offset_ms: float
    current_offset_ms: float
    shift_budget_ms: float
    stagger_over_iterations: int = 1
    delay_per_step_ms: float = 0.0

    @classmethod
    def create(cls, current_offset_ms: float, target_offset_ms: float,
               period_ms: float, max_single_step_delay_ms: float = 150.0) -> 'ActuationPlan':
        """
        Calculates the minimal forward phase displacement required modulo period_ms.
        Can either execute as a single micro-barrier or smoothly staggered over K steps.
        """
        # Forward displacement needed to reach target_offset
        delta_ms = (target_offset_ms - current_offset_ms) % period_ms
        if delta_ms < 0:
            delta_ms += period_ms

        if delta_ms <= max_single_step_delay_ms:
            stagger = 1
            delay_per_step = delta_ms
        else:
            # Smoothly spread over multiple iterations (e.g. 10ms per step)
            stagger = int(math.ceil(delta_ms / 15.0))
            delay_per_step = delta_ms / stagger

        return cls(
            target_offset_ms=target_offset_ms,
            current_offset_ms=current_offset_ms,
            shift_budget_ms=delta_ms,
            stagger_over_iterations=stagger,
            delay_per_step_ms=delay_per_step
        )


class MicroBarrierPhaseController:
    """
    Manages non-intrusive phase adjustments for a distributed training worker.
    """
    def __init__(self, job_id: int, iteration_period_ms: float = 1000.0):
        self.job_id = job_id
        self.period_ms = float(iteration_period_ms)
        self.current_estimated_offset_ms = 0.0
        self.active_plan: Optional[ActuationPlan] = None
        self.iterations_completed_in_plan = 0
        self.total_actuations_applied = 0
        self.cumulative_delay_injected_ms = 0.0

    def request_phase_shift(self, target_offset_ms: float,
                            max_delay_ms: float = 150.0) -> ActuationPlan:
        """Schedules a phase shift to target_offset_ms."""
        plan = ActuationPlan.create(
            current_offset_ms=self.current_estimated_offset_ms,
            target_offset_ms=target_offset_ms,
            period_ms=self.period_ms,
            max_single_step_delay_ms=max_delay_ms
        )
        self.active_plan = plan
        self.iterations_completed_in_plan = 0
        return plan

    def step_boundary(self) -> float:
        """
        Called at iteration boundary (e.g. after AllReduce or before next forward pass).
        Returns the micro-barrier delay in milliseconds to inject on this step.
        """
        if self.active_plan is None:
            return 0.0

        delay_ms = self.active_plan.delay_per_step_ms
        self.iterations_completed_in_plan += 1
        self.cumulative_delay_injected_ms += delay_ms
        self.current_estimated_offset_ms = (
            self.current_estimated_offset_ms + delay_ms
        ) % self.period_ms

        if self.iterations_completed_in_plan >= self.active_plan.stagger_over_iterations:
            self.active_plan = None
            self.total_actuations_applied += 1

        return delay_ms

    def compute_throughput_overhead(self, total_training_steps: int) -> Dict[str, float]:
        """
        Calculates the theoretical and realized throughput overhead of the actuation.
        Demonstrates why phase injection has negligible impact on total training time.
        """
        nominal_duration_ms = total_training_steps * self.period_ms
        actual_duration_ms = nominal_duration_ms + self.cumulative_delay_injected_ms
        slowdown_pct = (self.cumulative_delay_injected_ms / nominal_duration_ms) * 100.0

        return {
            "total_training_steps": float(total_training_steps),
            "nominal_runtime_sec": nominal_duration_ms / 1000.0,
            "actual_runtime_sec": actual_duration_ms / 1000.0,
            "total_delay_injected_sec": self.cumulative_delay_injected_ms / 1000.0,
            "slowdown_percentage": slowdown_pct,
            "effective_throughput_retention_pct": 100.0 - slowdown_pct
        }


class NCCLActuatorHook:
    """
    Simulates or hooks into PyTorch DDP / FSDP communication hooks:
    torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook
    
    Injects micro-barrier sleep at the AllReduce completion boundary.
    """
    def __init__(self, controller: MicroBarrierPhaseController,
                 emulate_hardware: bool = True):
        self.controller = controller
        self.emulate_hardware = emulate_hardware
        self.hook_invocations = 0

    def __call__(self, state: Any = None, bucket: Any = None) -> None:
        """
        Hook entry point triggered when gradient bucket finishes AllReduce.
        """
        self.hook_invocations += 1
        delay_ms = self.controller.step_boundary()
        if delay_ms > 0.0 and not self.emulate_hardware:
            # High-resolution delay in actual execution
            time.sleep(delay_ms / 1000.0)
