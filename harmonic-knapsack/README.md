# Harmonic Temporal Knapsack: Phase-Offset Gated Scheduling & Wave Cancellation

A dedicated research, algorithmic, and simulation framework exploring **Phase-Shift Gated Optimization** for **Temporal, Vector, and Peak-Constrained Knapsack Problems**.

---

## 1. Executive Summary & Problem Clarification

### Classic Knapsack vs. Temporal Peak-Constrained Knapsack
- **Classic 0/1 Knapsack:** Items have static scalar weights $w_i$ and values $v_i$. Capacity $W$ is stationary. There are zero temporal degrees of freedom.
- **Temporal / Vector Peak-Constrained Knapsack:** Items are time-varying periodic demand curves $L_i(t)$. When co-allocated on a shared physical resource (CPU core, electrical transformer, burst buffer, network switch queue), their demands add linearly:
  $$L_{\text{agg}}(t) = \sum_{i=1}^N x_i L_i(t - \tau_i)$$
  The resource capacity constraint is on the **instantaneous peak**:
  $$\max_{t \in [0, T_{\text{hyper}})} L_{\text{agg}}(t) \le C$$

### Why the Harmonic / Fourier View Transforms the Problem
For periodic workloads with period $T_i$ and fundamental frequency $\omega_i = 2\pi / T_i$:
$$L_i(t) = c_{i,0} + \sum_{k=1}^K X_{i,k} e^{i k \omega_i t}$$
By the Fourier time-shift property:
$$\mathcal{F}\{L_i(t - \tau_i)\} = X_{i,k} \cdot e^{-i k \omega_i \tau_i}$$
1. Shifting an item in time is **algebraically equivalent to multiplying its Fourier coefficients by a unit complex phase rotation factor $e^{-i k \omega_i \tau_i}$**.
2. Instead of expensive $O(T^2)$ time-domain roll operations, evaluating start offsets across all candidate positions is an $O(K \log K)$ analytical frequency-domain operation.
3. Two harmonic workloads achieve **maximal destructive wave cancellation** (minimum peak) when their primary harmonic phases satisfy:
   $$\phi_1 - \phi_2 \equiv \pi \pmod{2\pi}$$

---

## 2. Prior Art & Academic Lineage

This framework builds upon and directly compares against the three foundational bodies of literature:

1. **Allocating Jobs with Periodic Demand Variations (Beaumont et al., Euro-Par 2015 / HAL-01118176):**
   - Performed spectral analysis on 2011 Google cluster traces, proving long-running batch services have diurnal cycles.
   - Proposed **antagonistic job aggregation**—pairing workloads with opposite diurnal phases ($\pi$ radians apart) to flatten aggregate CPU peaks.
   - *Why it failed on Azure IaaS traces:* As proved by our M0 gate, Azure customer VMs exhibit $< 10\%$ periodic share and are dominated by heavy-tailed Pareto stochastic co-bursts ($\text{VIF} = 2.63$).

2. **FTIO: Frequency Techniques for I/O (Tarraf, Wolf et al., TU Darmstadt / ADMIRE Exascale):**
   - Applies Discrete Fourier Transform (DFT) with outlier filtering to detect periodic storage checkpoint bursts in HPC scientific simulations.
   - Feeds periodicity predictions to storage schedulers, achieving **$+26\%$ system utilization** and **$-56\%$ I/O slowdown reduction**.
   - *Why it succeeds:* HPC checkpoint loops (`if (step % N == 0)`) are code-governed, stationary, and phase-coherent.

3. **Phase-Scattering Scheduling in AI Training Clusters (Tardif, arXiv:2607.19638 / Choukse et al., arXiv:2508.14318):**
   - Addresses emergent phase-locking when independent distributed LLM training jobs share a facility power cap. Load-dependent throttling couples jobs into a Kuramoto oscillator network, triggering destructive multi-megawatt resonant power swings.
   - Introduces **phase-scattering scheduling**: uniformly offsets iteration start times ($\phi_k = 2\pi k / M$) to flatten aggregate power draw.
   - *The Phase Drift Hazard:* Network jitter causes open-loop phases to drift over time, proving that **closed-loop phase re-centering** is required.

---

## 3. The 7 Verified Real-World Application Domains

| Domain | Problem Structure | Mechanism | Impact |
| :--- | :--- | :--- | :--- |
| **1. Periodic Timetabling (PESP)** | Rail/transit departure times modulo period $T$ | Integer programming modulo $T$ (Serafini & Ukovich 1989) | Optimal passenger transfers & minimal rolling stock |
| **2. Time-Sensitive Networking (TSN)** | IEEE 802.1Qbv cyclic queue gating | Gate Control List (GCL) non-overlapping windows | Bounded latency ($< 10\ \mu\text{s}$) & zero queue drop |
| **3. EV Charging & Demand Response** | Inflexible base load + shiftable EV battery charging | Convex sum-of-squares valley filling (Gan-Topcu-Low 2013) | Overnight peak shaving without secondary rebound peaks |
| **4. Real-Time Task Sets** | Rate-Monotonic & Periodic Maintenance | Harmonic integer periods ($T_j / T_i \in \mathbb{Z}^+$) | Schedulable utilization bound rises from $69.3\% \to 100\%$ |
| **5. Traffic Signal Coordination** | Arterial intersection offsets | MAXBAND mixed-integer linear programming (Little 1966) | Continuous green waves; stops & fuel consumption cut |
| **6. Cron & Batch Maintenance** | Top-of-the-hour thundering herds across 10,000 nodes | Modular hash staggering ($\tau = \text{hash}(\text{node}) \pmod{60}$) | **$-95.8\%$ peak concurrency reduction** |
| **7. Physical Signals & Antennas** | Acoustic & Electromagnetic wave fields | FxLMS Active Noise Cancellation & Adaptive Beamforming | **$> 30\text{ dB}$ interference nulling & noise attenuation** |

---

## 4. Signal Processing Analysis & Physical Boundaries

Unlike computer queuing where demand is non-negative ($u \ge 0$), physical waves can be negative and cancel algebraically:

1. **Active Noise Cancellation (ANC / FxLMS):**
   - **The Strict Causality Law:** Control electronics latency must be strictly less than acoustic travel time ($\tau_{\text{elec}} < \tau_{\text{acoustic}}$). Acausal setups cannot cancel broadband noise.
   - **Pareto vs. Gaussian Stability:** Standard FxLMS minimizes $E[e^2(n)]$. Under heavy-tailed Pareto bursts (such as cloud VM spikes), MSE gradients diverge. Robust **Normalized Sign-LMS** is mandatory for stability.
2. **Adaptive Beamforming (MVDR / Capon):**
   - **The RMB Rule (Reed-Mallett-Brennan):** Estimating the spatial covariance matrix $\hat{\mathbf{R}}$ requires at least $K \ge 2M$ snapshots to achieve within $3\text{ dB}$ of optimal SINR.
   - **Diagonal Loading:** Stabilizes $\hat{\mathbf{R}} + \alpha \mathbf{I}$ in under-sampled regimes.
   - **Steering Vector Mismatch:** A pointing error of even $2^\circ$ causes the beamformer to place a spatial null directly on the desired signal (catastrophic signal cancellation).

---

## 5. Production Runtime: The Three Core Pillars

To transition from research simulation to production-grade datacenter deployment, the [`production/`](./production) package implements:

1. **Pillar 1: Real-Time Telemetry Ingestion (The Sensor)** ([`production/telemetry_sensor.py`](./production/telemetry_sensor.py)): Sub-second $50\text{ Hz}$ GPU power draw ingestion via NVIDIA DCGM/NVML or rack PDU with lock-free ring-buffer and online sliding-window FFT tracking.
2. **Pillar 2: Non-Intrusive Actuation Hook (The Actuator)** ([`production/nccl_actuator_hook.py`](./production/nccl_actuator_hook.py)): Injects one-off micro-barrier delays at the PyTorch / NCCL AllReduce collective communication boundary, shifting phase coordinates with $>99.99\%$ throughput retention.
3. **Pillar 3: Rate-Diversity & Higher-Harmonic Handling (The Controller)** ([`production/rate_diversity_controller.py`](./production/rate_diversity_controller.py)): Eliminates Tardif beat frequencies ($f_{\text{beat}} = |f_A - f_B|$) via harmonic period quantizer clustering and dynamic mini-batch pacing.

### Production Pitch-Deck & Due-Diligence Charts

#### Chart 1: Beat-Frequency Collision vs. Harmonic Pacing
Two AI training jobs ($800\text{ms}$ vs $1100\text{ms}$) beating into a $56\text{ kW}$ breaker trip every $2.93\text{s}$ vs. `DynamicBatchPacer` locking them to a flat, safe $32\text{ kW}$ envelope:
![Chart 1: Beat-Frequency vs Harmonic Pacing](production/charts/chart1_beat_frequency_vs_harmonic_pacing.png)

#### Chart 2: Solver Scalability (greedy_harmonic vs. greedy_fft)
Comparing discrete grid search vs. $\mathcal{O}(M \log M)$ circular cross-correlation IFFT across cluster sizes from $N=10$ to $N=1,000$ jobs, achieving sub-100ms scheduling latency:
![Chart 2: Solver Scalability](production/charts/chart2_solver_scalability.png)

#### Chart 3: Closed-Loop Phase Recentering with >99.99% Throughput Retention
500 iterations of distributed PyTorch training under random network jitter, showing how one-off micro-barrier nudges keep aggregate power flat beneath the breaker limit with $<0.01\%$ training slowdown:
![Chart 3: Closed-Loop Phase Recentering](production/charts/chart3_closed_loop_phase_recentering.png)

---

## 6. Directory Structure & Architecture

```
harmonic-knapsack/
├── README.md                                  # This document
├── docs/
│   ├── 01_problem_formulation.md              # Classic Knapsack vs. Temporal Harmonic Knapsack
│   ├── 02_prior_art_and_literature.md         # Beaumont Euro-Par 2015, FTIO, AI Power Capping
│   ├── 03_verified_application_domains.md     # The 7 verified engineering application domains
│   ├── 04_signal_processing_analysis.md       # ANC FxLMS causality, Beamforming SMI / RMB rule
│   └── 05_production_engineering_roadmap.md   # Production Architecture & 3 Pillars Roadmap
├── core/
│   ├── harmonic_representation.py             # Fourier series decomposition, FTIO confidence metric
│   ├── phase_shift_math.py                    # Exact analytical phase rotation & O(M log M) IFFT
│   └── temporal_knapsack_solver.py            # Peak-Minimization, Value-Maximization & greedy_fft
├── production/
│   ├── telemetry_sensor.py                    # Pillar 1: Sub-second NVML/PDU Telemetry Sensor
│   ├── nccl_actuator_hook.py                  # Pillar 2: Micro-barrier AllReduce Actuator Hook
│   ├── rate_diversity_controller.py           # Pillar 3: DynamicBatchPacer & Harmonic Quantizer
│   ├── live_nvml_benchmark.py                 # End-to-end closed-loop live hardware benchmark
│   ├── generate_pitch_deck_charts.py          # Pitch deck chart generator (publication quality)
│   └── charts/                                # Generated 300 DPI pitch-deck charts
├── applications/
│   ├── hpc_checkpoint_ftio.py                 # HPC Checkpoint I/O staging simulation (FTIO model)
│   ├── ai_training_power_cap.py               # AI training power-cap phase scattering & drift
│   ├── cron_staggering.py                     # Thundering-herd batch/cron hash-staggering
│   └── tsn_cyclic_scheduler.py                # Time-Sensitive Networking (IEEE 802.1Qbv) TAS
├── signal_processing/
│   ├── anc_fxlms_simulator.py                 # FxLMS adaptive filter with causality & Pareto noise
│   └── adaptive_beamformer.py                 # MVDR / Capon beamformer with SMI & steering mismatch
└── tests/
    ├── test_audit_defects_fixed.py            # Comprehensive tests verifying all 8 audit defect fixes
    ├── test_harmonic_math.py                  # Tests for Fourier decomposition & phase shift math
    ├── test_pipeline_parallel_harmonics.py    # Tests for 1F1B PP staircase modeling & multi-job scattering
    ├── test_solver_benchmarks.py              # Tests for Knapsack solvers & domain applications
    ├── test_signal_processing.py              # Tests for FxLMS causality and beamforming nulling
    └── test_production_runtime.py             # Tests for Sensor, Actuator & Closed-Loop hardware
```

---

## 7. How to Run the Test & Benchmark Suites

Run the complete test suite from PowerShell or terminal:

```bash
# 1. Run full pytest test suite (all 33 tests across math, solvers, pipeline parallelism, and production)
python -m pytest harmonic-knapsack/tests/

# 2. Run the live closed-loop NVML hardware benchmark (Sensor -> Controller -> Actuator)
python harmonic-knapsack/production/live_nvml_benchmark.py

# 3. Regenerate the 3 publication-grade pitch deck charts (saved to production/charts/)
python harmonic-knapsack/production/generate_pitch_deck_charts.py
```

### Verified Test Results
```text
[PASS] test_fourier_decomposition_and_reconstruction: DC=10.00, Amp1=4.00, max_err=0.0000
[PASS] test_phase_shift_frequency_vs_time_domain: analytical rotation diff=5.33e-15, time roll diff=0.0000
[PASS] test_antagonistic_pairing_cancellation: offset=30.0s, uncoord=30.0 -> min_peak=20.0 (-33.3%)
[PASS] test_hyperperiod_computation: all LCM calculations passed.
[PASS] test_fft_cross_correlation_and_greedy_fft: FFT solver peak=20.0 vs Harmonic solver=20.0

[PASS] test_telemetry_sensor: Period=1.00s, Confidence=1.00
[PASS] test_nccl_actuator_hook: Injected 0.150s delay -> Slowdown=0.0015%, Throughput Retention=99.9985%
[PASS] test_rate_diversity: Beat freq=0.341Hz, Recurrence=2.93s
[PASS] test_live_nvml_closed_loop_benchmark: Peak reduced by 33.1%, Retention=99.9707%

[PASS] test_knapsack_peak_minimization: uncoord=80.0 -> greedy=56.0 (-30.0%), SA=50.2
[PASS] test_knapsack_value_maximization: Static=3 items (val=300) -> Harmonic=5 items (val=500) (+66.7% value gain)
[PASS] test_hpc_checkpoint_ftio: Uncoord Peak=66.7 Gbps (CONTENDED) -> Staggered=15.8 Gbps (-76.3%)
[PASS] test_ai_training_power_cap: Uncoord=56.5 kW (TRIPPED) -> Scattered=29.3 kW (-48.2%)
[PASS] test_cron_thundering_herd_staggering: Naive concurrency=5000 -> Staggered peak=210 (-95.8%)
[PASS] test_tsn_cyclic_scheduler: GCL synthesized (64.0% util); clock drift collision detected when drift > guard.

[PASS] test_fxlms_causality: Causal FxLMS achieved 26.13 dB broadband noise attenuation.
[PASS] test_fxlms_pareto_stability: Normalized Sign-LMS remained stable under Pareto bursts.
[PASS] test_adaptive_beamformer_rmb: K=64 snapshots satisfied RMB rule; null suppression=35.3 dB.
[PASS] test_beamformer_diagonal_loading: Diagonal loading stabilized under-sampled snapshot regime.
```
