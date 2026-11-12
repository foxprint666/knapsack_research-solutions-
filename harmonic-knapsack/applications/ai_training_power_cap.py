"""
harmonic-knapsack/applications/ai_training_power_cap.py
Application Model: Co-Located AI Training Jobs Under a Shared Power Cap.

Based on:
  - Brieuc Le Roux Tardif, "Do Co-Located AI Training Jobs Synchronize?
    Load-Dependent Throttling as a Coupling Mechanism for Phase-Locking
    Behind a Shared Power Cap" (arXiv:2607.19638).
  - Choukse et al., "Power Stabilization for AI Training Datacenters"
    (Microsoft / OpenAI / NVIDIA, arXiv:2508.14318).

Simulates:
  - Synchronous mini-batch SGD iterations (Compute phase @ high wattage, AllReduce @ low wattage).
  - Emergent phase-locking synchronization under shared power constraint.
  - Phase-Scattering Scheduling (scattering phases across the iteration cycle).
  - Phase Drift over time and Closed-Loop Phase Re-centering.
"""

import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from core.harmonic_representation import HarmonicProfile, HarmonicComponent
from core.phase_shift_math import PhaseShiftEngine

class AITrainingJob:
    """Represents a distributed DNN training job running on an accelerator rack (Data Parallelism / DDP)."""
    def __init__(self, job_id: int, name: str,
                 iteration_period_ms: float = 1000.0,
                 compute_duration_ms: float = 750.0,
                 compute_power_kw: float = 6.0,
                 comm_power_kw: float = 1.5):
        self.job_id = job_id
        self.name = name
        self.iteration_period_sec = iteration_period_ms / 1000.0
        self.compute_duration_sec = compute_duration_ms / 1000.0
        self.compute_power_kw = float(compute_power_kw)
        self.comm_power_kw = float(comm_power_kw)
        
        # Build harmonic profile
        duty_cycle = self.compute_duration_sec / self.iteration_period_sec
        dc_mean = (duty_cycle * self.compute_power_kw) + ((1.0 - duty_cycle) * self.comm_power_kw)
        pulse = self.compute_power_kw - self.comm_power_kw
        f0 = 1.0 / self.iteration_period_sec
        
        harmonics = []
        for k in range(1, 6):
            amp = 2.0 * (pulse / (k * np.pi)) * np.abs(np.sin(k * np.pi * duty_cycle))
            harmonics.append(HarmonicComponent(
                harmonic_order=k,
                frequency_hz=k * f0,
                amplitude=amp,
                phase_rad=0.0
            ))
            
        self.profile = HarmonicProfile(
            period_seconds=self.iteration_period_sec,
            dc_mean=dc_mean,
            harmonics=harmonics,
            periodicity_confidence=0.99
        )


class PipelineStage1F1B:
    """
    Represents an individual stage p of a 1F1B (One Forward, One Backward) pipeline parallel schedule.
    Accurately models:
      - Startup/warmup forward passes and pipeline bubble wait states.
      - 1F1B steady-state alternating Forward/Backward micro-batches.
      - Cooldown backward passes.
      - Inter-stage P2P transfer gaps and AllReduce barrier synchronization.
      - Analytical Fourier series decomposition of the multi-step staircase power waveform.
    """
    def __init__(self, stage_id: int, num_stages: int, num_microbatches: int,
                 fwd_duration_ms: float = 25.0,
                 bwd_duration_ms: float = 50.0,
                 fwd_power_kw: float = 5.5,
                 bwd_power_kw: float = 6.5,
                 bubble_power_kw: float = 1.5,
                 allreduce_duration_ms: float = 50.0,
                 allreduce_power_kw: float = 2.0,
                 iteration_period_ms: Optional[float] = None,
                 num_harmonics: int = 16):
        assert 0 <= stage_id < num_stages, f"Invalid stage_id {stage_id} for num_stages {num_stages}"
        assert num_microbatches >= num_stages, "1F1B requires micro-batches >= num_stages"
        
        self.stage_id = int(stage_id)
        self.num_stages = int(num_stages)
        self.num_microbatches = int(num_microbatches)
        self.fwd_duration_sec = fwd_duration_ms / 1000.0
        self.bwd_duration_sec = bwd_duration_ms / 1000.0
        self.fwd_power_kw = float(fwd_power_kw)
        self.bwd_power_kw = float(bwd_power_kw)
        self.bubble_power_kw = float(bubble_power_kw)
        self.allreduce_duration_sec = allreduce_duration_ms / 1000.0
        self.allreduce_power_kw = float(allreduce_power_kw)
        self.num_harmonics = int(num_harmonics)

        # Compute full dependency-respecting schedule across all stages
        self.all_stage_intervals, max_active_time = self._compute_pipeline_schedule()
        
        # Settle iteration period
        min_period = max_active_time + self.allreduce_duration_sec
        if iteration_period_ms is not None:
            self.iteration_period_sec = max(min_period, iteration_period_ms / 1000.0)
        else:
            self.iteration_period_sec = min_period
            
        # Finalize intervals for this stage, appending AllReduce and idle padding
        self.intervals: List[Tuple[str, float, float, float]] = []
        for tag, t1, t2, pwr in self.all_stage_intervals[self.stage_id]:
            self.intervals.append((tag, t1, t2, pwr))

        # Add cooldown idle wait if this stage finished compute earlier than others
        stage_compute_end = self.intervals[-1][2] if self.intervals else 0.0
        if stage_compute_end < max_active_time:
            self.intervals.append(("COOLDOWN_BUBBLE", stage_compute_end, max_active_time, self.bubble_power_kw))

        # Add collective AllReduce / optimizer step
        ar_start = max_active_time
        ar_end = max_active_time + self.allreduce_duration_sec
        self.intervals.append(("ALLREDUCE", ar_start, ar_end, self.allreduce_power_kw))

        # Add remaining padding if iteration_period_sec > ar_end
        if self.iteration_period_sec > ar_end:
            self.intervals.append(("IDLE_PADDING", ar_end, self.iteration_period_sec, self.bubble_power_kw))

        # Analytical Fourier Profile
        self.profile = self.to_harmonic_profile(self.num_harmonics)

    def _compute_pipeline_schedule(self) -> Tuple[List[List[Tuple[str, float, float, float]]], float]:
        """
        Exact discrete-event dependency scheduler simulating 1F1B forward/backward micro-batch execution
        across all P stages.
        """
        P = self.num_stages
        M = self.num_microbatches
        f_end: Dict[Tuple[int, int], float] = {}
        b_end: Dict[Tuple[int, int], float] = {}
        stage_time = [0.0] * P
        all_stage_intervals: List[List[Tuple[str, float, float, float]]] = [[] for _ in range(P)]

        # Megatron-LM 1F1B Task order per stage
        tasks_per_stage = []
        for p in range(P):
            warmup_count = P - p
            order = [("F", m) for m in range(warmup_count)]
            f_next = warmup_count
            b_next = 0
            while f_next < M:
                order.append(("B", b_next))
                b_next += 1
                order.append(("F", f_next))
                f_next += 1
            while b_next < M:
                order.append(("B", b_next))
                b_next += 1
            tasks_per_stage.append(order)

        pending_idx = [0] * P
        while any(pending_idx[p] < len(tasks_per_stage[p]) for p in range(P)):
            progress = False
            for p in range(P):
                if pending_idx[p] >= len(tasks_per_stage[p]):
                    continue
                ttype, m = tasks_per_stage[p][pending_idx[p]]
                can_run = False
                dep_time = 0.0
                if ttype == "F":
                    if p == 0:
                        can_run, dep_time = True, 0.0
                    elif (p - 1, m) in f_end:
                        can_run, dep_time = True, f_end[(p - 1, m)]
                elif ttype == "B":
                    if (p, m) in f_end:
                        if p == P - 1:
                            can_run, dep_time = True, f_end[(p, m)]
                        elif (p + 1, m) in b_end:
                            can_run, dep_time = True, max(f_end[(p, m)], b_end[(p + 1, m)])
                if can_run:
                    start_t = max(stage_time[p], dep_time)
                    dur = self.fwd_duration_sec if ttype == "F" else self.bwd_duration_sec
                    end_t = start_t + dur
                    if start_t > stage_time[p]:
                        bubble_tag = "WARMUP_BUBBLE" if pending_idx[p] == 0 else "INTER_STAGE_BUBBLE"
                        all_stage_intervals[p].append((bubble_tag, stage_time[p], start_t, self.bubble_power_kw))
                    pwr = self.fwd_power_kw if ttype == "F" else self.bwd_power_kw
                    all_stage_intervals[p].append((ttype, start_t, end_t, pwr))
                    stage_time[p] = end_t
                    if ttype == "F":
                        f_end[(p, m)] = end_t
                    else:
                        b_end[(p, m)] = end_t
                    pending_idx[p] += 1
                    progress = True
            if not progress:
                raise RuntimeError("Deadlock in 1F1B schedule execution simulation")

        max_active_time = max(stage_time)
        return all_stage_intervals, max_active_time

    def sample_power_at_time(self, t_sec: float) -> float:
        """Evaluates instantaneous piecewise wattage at time t within the iteration."""
        t_mod = t_sec % self.iteration_period_sec
        for _, t1, t2, pwr in self.intervals:
            if t1 <= t_mod < t2:
                return pwr
        return self.bubble_power_kw

    def to_harmonic_profile(self, num_harmonics: int = 16) -> HarmonicProfile:
        """
        Computes the EXACT, closed-form Fourier Series representation for this stage's
        piecewise constant power waveform:
          c0 = (1/T) sum(V_j * delta_t_j)
          ak = (2/T) sum(V_j * [sin(w_k*t2) - sin(w_k*t1)] / w_k)
          bk = (2/T) sum(V_j * [-cos(w_k*t2) + cos(w_k*t1)] / w_k)
        """
        T = self.iteration_period_sec
        f0 = 1.0 / T
        
        # DC Mean
        dc_mean = sum(pwr * (t2 - t1) for _, t1, t2, pwr in self.intervals) / T
        total_ac_var = sum(((pwr - dc_mean) ** 2) * (t2 - t1) for _, t1, t2, pwr in self.intervals) / T

        harmonics = []
        captured_energy = 0.0
        for k in range(1, num_harmonics + 1):
            omega = 2.0 * np.pi * k * f0
            ak = (2.0 / T) * sum(pwr * (np.sin(omega * t2) - np.sin(omega * t1)) / omega
                                 for _, t1, t2, pwr in self.intervals)
            bk = (2.0 / T) * sum(pwr * (-np.cos(omega * t2) + np.cos(omega * t1)) / omega
                                 for _, t1, t2, pwr in self.intervals)
            amp = float(np.sqrt(ak**2 + bk**2))
            phi = float(np.arctan2(-bk, ak))
            captured_energy += 0.5 * (amp ** 2)
            harmonics.append(HarmonicComponent(
                harmonic_order=k,
                frequency_hz=k * f0,
                amplitude=amp,
                phase_rad=phi
            ))

        confidence = float(np.clip(captured_energy / max(1e-9, total_ac_var), 0.0, 1.0))
        return HarmonicProfile(
            period_seconds=T,
            dc_mean=dc_mean,
            harmonics=harmonics,
            periodicity_confidence=confidence
        )


class PipelineParallelTrainingJob:
    """
    Represents a full 3D-parallel frontier LLM training job (e.g., Llama-4 or GPT-class)
    partitioned across P pipeline stages with M micro-batches under a 1F1B schedule.
    """
    def __init__(self, job_id: int, name: str,
                 num_stages: int = 4,
                 num_microbatches: int = 8,
                 fwd_duration_ms: float = 25.0,
                 bwd_duration_ms: float = 50.0,
                 fwd_power_kw: float = 5.5,
                 bwd_power_kw: float = 6.5,
                 bubble_power_kw: float = 1.5,
                 allreduce_duration_ms: float = 50.0,
                 allreduce_power_kw: float = 2.0,
                 iteration_period_ms: Optional[float] = None,
                 num_harmonics: int = 24):
        self.job_id = int(job_id)
        self.name = str(name)
        self.num_stages = int(num_stages)
        self.num_microbatches = int(num_microbatches)
        self.num_harmonics = int(num_harmonics)

        # Instantiate all pipeline stages
        self.stages: List[PipelineStage1F1B] = [
            PipelineStage1F1B(
                stage_id=p,
                num_stages=self.num_stages,
                num_microbatches=self.num_microbatches,
                fwd_duration_ms=fwd_duration_ms,
                bwd_duration_ms=bwd_duration_ms,
                fwd_power_kw=fwd_power_kw,
                bwd_power_kw=bwd_power_kw,
                bubble_power_kw=bubble_power_kw,
                allreduce_duration_ms=allreduce_duration_ms,
                allreduce_power_kw=allreduce_power_kw,
                iteration_period_ms=iteration_period_ms,
                num_harmonics=self.num_harmonics
            )
            for p in range(self.num_stages)
        ]
        self.iteration_period_sec = self.stages[0].iteration_period_sec
        self.stage_profiles = [s.profile for s in self.stages]
        self.bubble_fraction = float(self.num_stages - 1) / float(self.num_microbatches)

        # Aggregate Model Profile: sum of Fourier components across all P stages
        self.profile = self._build_aggregate_harmonic_profile()

    def _build_aggregate_harmonic_profile(self) -> HarmonicProfile:
        """
        By linearity of Fourier Series, the sum of P stage waveforms is the sum of their Fourier components:
          c0_agg = sum(c0_p)
          ak_agg = sum(ak_p), bk_agg = sum(bk_p)
        """
        T = self.iteration_period_sec
        f0 = 1.0 / T
        dc_agg = sum(sp.dc_mean for sp in self.stage_profiles)

        agg_harmonics = []
        for k_idx in range(self.num_harmonics):
            ak_sum = 0.0
            bk_sum = 0.0
            order = k_idx + 1
            for sp in self.stage_profiles:
                h = sp.harmonics[k_idx]
                # h.amplitude * cos(w*t + h.phase_rad) = a*cos(w*t) + b*sin(w*t)
                ak = h.amplitude * np.cos(h.phase_rad)
                bk = -h.amplitude * np.sin(h.phase_rad)
                ak_sum += ak
                bk_sum += bk

            amp = float(np.sqrt(ak_sum**2 + bk_sum**2))
            phi = float(np.arctan2(-bk_sum, ak_sum))
            agg_harmonics.append(HarmonicComponent(
                harmonic_order=order,
                frequency_hz=order * f0,
                amplitude=amp,
                phase_rad=phi
            ))

        # Overall confidence is the mean confidence of the stages
        mean_conf = float(np.mean([sp.periodicity_confidence for sp in self.stage_profiles]))
        return HarmonicProfile(
            period_seconds=T,
            dc_mean=dc_agg,
            harmonics=agg_harmonics,
            periodicity_confidence=mean_conf
        )

    def compute_spectral_energy_distribution(self) -> Dict[str, float]:
        """
        Calculates spectral energy breakdown of the aggregate 1F1B power profile:
          1. DC Mean Baseline Energy
          2. Macro Iteration Envelope (Harmonics k in 1..4)
          3. Micro-Batch Switching Ripples (Harmonics clustered near M and 2M)
          4. Higher Harmonics Residual
        """
        M = self.num_microbatches
        macro_energy = 0.0
        mb_energy = 0.0
        residual_energy = 0.0

        for h in self.profile.harmonics:
            e = 0.5 * (h.amplitude ** 2)
            k = h.harmonic_order
            if k <= 4:
                macro_energy += e
            elif abs(k - M) <= 1 or abs(k - 2 * M) <= 1:
                mb_energy += e
            else:
                residual_energy += e

        total_ac = macro_energy + mb_energy + residual_energy
        return {
            "dc_power_kw": self.profile.dc_mean,
            "macro_envelope_ac_pct": (macro_energy / max(1e-9, total_ac)) * 100.0,
            "microbatch_carrier_ac_pct": (mb_energy / max(1e-9, total_ac)) * 100.0,
            "residual_ac_pct": (residual_energy / max(1e-9, total_ac)) * 100.0,
            "microbatch_frequency_hz": M / self.iteration_period_sec,
            "bubble_fraction": self.bubble_fraction
        }

    def sample_aggregate_power_at_time(self, t_sec: float) -> float:
        """Samples the combined instantaneous wattage of all P pipeline stages at time t."""
        return sum(stage.sample_power_at_time(t_sec) for stage in self.stages)


class AIPowerCapScheduler:
    """
    Schedules and coordinates AI training execution phases to flatten aggregate power draw.
    """
    def __init__(self, rack_power_cap_kw: float = 38.0):
        self.rack_power_cap_kw = float(rack_power_cap_kw)

    def evaluate_phase_scattering(self, jobs: List[Any]) -> Dict[str, Any]:
        """
        Compares:
          1. Phase-Locked / Uncoordinated In-Phase Alignment (Worst Case)
          2. Phase-Scattering Scheduling (Uniform Phase Distribution)
        Supports both single-harmonic DDP jobs (AITrainingJob) and 1F1B 3D-parallel jobs (PipelineParallelTrainingJob).
        """
        M = len(jobs)
        assert M > 0, "Need at least one job"
        profiles = [j.profile for j in jobs]
        period = jobs[0].iteration_period_sec

        # 1. Synchronized / Phase-Locked alignment (offsets = 0)
        zero_offsets = [0.0] * M
        _, uncoord_curve, uncoord_peak = PhaseShiftEngine.combine_harmonic_profiles(
            profiles, zero_offsets, eval_points=500
        )

        # 2. Phase-Scattering: distribute offsets uniformly tau_k = k * (T / M)
        scattered_offsets = [k * (period / M) for k in range(M)]
        _, scattered_curve, scattered_peak = PhaseShiftEngine.combine_harmonic_profiles(
            profiles, scattered_offsets, eval_points=500
        )

        peak_reduction_pct = ((uncoord_peak - scattered_peak) / uncoord_peak) * 100.0

        return {
            "num_jobs": M,
            "power_cap_kw": self.rack_power_cap_kw,
            "uncoordinated": {
                "peak_power_kw": uncoord_peak,
                "trips_power_cap": uncoord_peak > self.rack_power_cap_kw,
                "excess_kw": max(0.0, uncoord_peak - self.rack_power_cap_kw)
            },
            "phase_scattered": {
                "peak_power_kw": scattered_peak,
                "trips_power_cap": scattered_peak > self.rack_power_cap_kw,
                "excess_kw": max(0.0, scattered_peak - self.rack_power_cap_kw),
                "offsets_ms": [tau * 1000.0 for tau in scattered_offsets]
            },
            "peak_reduction_pct": peak_reduction_pct
        }

    def evaluate_pipeline_coordination(self, pipeline_jobs: List[PipelineParallelTrainingJob]) -> Dict[str, Any]:
        """
        Specialized evaluation for co-located multi-stage 3D/pipeline-parallel training jobs:
          1. Evaluates individual pipeline profiles vs aggregate uncoordinated peak.
          2. Applies optimal phase scattering across 3D-parallel job replicas.
          3. Quantifies intra-pipeline bubble interleaving and peak shaving under a shared rack power cap.
        """
        base_res = self.evaluate_phase_scattering(pipeline_jobs)
        base_res["pipeline_stage_count"] = pipeline_jobs[0].num_stages
        base_res["microbatch_count"] = pipeline_jobs[0].num_microbatches
        base_res["bubble_fraction"] = pipeline_jobs[0].bubble_fraction
        base_res["spectral_distributions"] = [
            j.compute_spectral_energy_distribution() for j in pipeline_jobs
        ]
        return base_res

    def simulate_phase_drift_and_recentering(self, jobs: List[AITrainingJob],
                                            num_iterations: int = 200,
                                            jitter_std_ms: float = 8.0,
                                            enable_closed_loop: bool = True,
                                            recenter_interval: int = 25,
                                            seed: int = 42) -> Dict[str, Any]:
        """
        Simulates long-term execution where network jitter causes open-loop phase drift,
        and demonstrates how closed-loop phase re-centering maintains stability.
        """
        rng = np.random.default_rng(seed)
        M = len(jobs)
        period_ms = jobs[0].iteration_period_sec * 1000.0
        
        # Initial scattered offsets in ms
        offsets_ms = np.array([k * (period_ms / M) for k in range(M)])
        
        peak_history = []
        drift_max_history = []
        
        for it in range(num_iterations):
            # Accumulate random timing jitter per job
            jitter = rng.normal(0.0, jitter_std_ms, size=M)
            offsets_ms = (offsets_ms + jitter) % period_ms

            # Closed-loop re-centering check
            if enable_closed_loop and (it > 0) and (it % recenter_interval == 0):
                # Re-align phases toward uniform spacing
                ideal_offsets = np.array([k * (period_ms / M) for k in range(M)])
                # Nudge phases by 50% toward ideal to model gentle control correction
                offsets_ms = (0.5 * offsets_ms + 0.5 * ideal_offsets) % period_ms

            # Evaluate instantaneous peak power at this iteration
            offsets_sec = [t / 1000.0 for t in offsets_ms]
            _, _, peak_kw = PhaseShiftEngine.combine_harmonic_profiles(
                [j.profile for j in jobs], offsets_sec, eval_points=250
            )
            peak_history.append(peak_kw)
            
            # Phase clustering metric: variance of inter-job spacing
            sorted_offs = np.sort(offsets_ms)
            spacings = np.diff(np.append(sorted_offs, sorted_offs[0] + period_ms))
            drift_max_history.append(float(np.std(spacings)))

        return {
            "num_iterations": num_iterations,
            "enable_closed_loop": enable_closed_loop,
            "mean_peak_power_kw": float(np.mean(peak_history)),
            "max_peak_power_kw": float(np.max(peak_history)),
            "power_cap_kw": self.rack_power_cap_kw,
            "cap_violation_count": int(np.sum(np.array(peak_history) > self.rack_power_cap_kw)),
            "peak_history": peak_history,
            "drift_history": drift_max_history
        }
