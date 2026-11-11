# Knapsack Research Solutions: Dual-Paradigm Systems Engineering

An advanced dual-paradigm research and systems engineering codebase addressing **Multi-Dimensional Knapsack Packing**, **Temporal Peak-Constrained Scheduling**, and **Failure-Isolated Memory Fabric Tiering** across both **stochastic cloud datacenters** and **deterministic cyclic infrastructures**.

```
====================================================================================================
                               KNAPSACK RESEARCH SOLUTIONS SUITE
====================================================================================================

      1. STOCHASTIC CLOUD REGIME                           2. DETERMINISTIC CYCLIC REGIME
                 (SSCF)                                         (HARMONIC KNAPSACK)
       Folder: sscf/                                          Folder: harmonic-knapsack/

   * Microsoft Azure IaaS Fleet Trace                     * Distributed AI GPU Training Clusters
   * < 10% Periodic Share; Pareto Bursts                  * HPC Supercomputing Checkpoint I/O
   * Cross-VM Co-Bursting (VIF = 2.63)                    * Time-Sensitive Networking (TSN 802.1Qbv)
   * 3-Plane Architecture:                                * Fast O(K log K) Phase Rotations
     - Plane 1: Mix-Aware Knapsack                          X_k' = X_k * exp(-i * k * omega * tau)
     - Plane 2: Linux cgroups v2 Active Shaper            * Closed-Loop Drift Re-centering
     - Plane 3: Failure-Isolated CXL Tiering              * Signal Processing (ANC FxLMS & Beamforming)

   >>> RESULT: +25.45% Host Reduction                     >>> RESULT: -48.2% AI Peak Power Drop
               < 0.40% Holdout Overload                               -76.3% HPC Storage Contention
               0.0000% SLA Violations                                 +66.7% Knapsack Value Gain
               +35.0% CXL DRAM Headroom                               -95.8% Cron Thundering Herd
====================================================================================================
```

---

## 1. System 1: Stochastic-Shaped CXL Fabric (SSCF)

Located in [`sscf/`](./sscf).

### Context & Empirical Proof
Earlier theoretical models posited that cloud VMs could achieve massive density via **harmonic wave cancellation** (destructive interference of diurnal sinusoidal load waves). 
To verify this, we ran the **M0 Spectral Gate** on the real public Azure V1 fleet trace ($4{,}994$ VMs over $72$ hours, $796{,}464$ core-hours):
- **Periodicity:** Only $5.4\% \text{ to } 9.7\%$ of core-hours exhibited diurnal cycles. Over $90\%$ of demand consisted of heavy-tailed, non-stationary Pareto bursts.
- **Co-Bursting:** The Variance Inflation Factor was $\text{VIF} = 2.63$ (cross-VM co-bursting), disproving independent variance quadrature ($\mu + 2.33\sigma$).

### The Three-Plane Solution
- **Plane 1 (Admission & Placement):** Multi-Dimensional Mix-Aware Knapsack Packer enforcing dual constraints:
  $$\sum \mu_{\text{total}} \le 0.70 \times C \quad \text{and} \quad \sum \mu_{\text{interactive}} \le 0.30 \times C$$
  Because interactive workloads never exceed $30\%$ of host capacity, simultaneous co-bursts can never physically saturate the physical server.
- **Plane 2 (Host Runtime Execution):** Linux cgroups v2 Active Shaper Daemon monitoring `/proc/pressure/cpu` (PSI). During burst events, it dynamically clamps `batch.slice/cpu.max`, deferring delay-insensitive batch work into a causal FIFO buffer with **$100\%$ work mass conservation** and **zero interactive SLA degradation**. Fails open automatically when pressure clears.
- **Plane 3 (Disaggregated Memory):** DAMON-guided clean-page tiering to shared CXL memory pools enforcing **ADR-0010**. Banning unmirrored dirty memory and kernel structures from CXL guarantees **zero Machine Check Exceptions (MCEs) and zero kernel panics** if the CXL link resets.

### Real Trace Verification Metrics
- **Host Count Reduction:** **$+25.45\%$** ($34$ physical servers vs $45.6$ Static P99 reservation baseline).
- **Day 1 Holdout Overload:** **$0.0204\%$** (well under the $0.5\%$ target ceiling).
- **Day 2 Holdout Overload:** **$0.3983\%$**; **$0.0000\%$** interactive SLA violations.
- **Active Shaper Residual Overload:** **$0.00\%$** (reduced from $0.69\% \to 0.00\%$).
- **CXL DRAM Headroom Savings:** **$+35.0\%$** with $100\%$ host survival during simulated cable disconnects.

---

## 2. System 2: Harmonic Temporal Knapsack & Phase-Offset Gating

Located in [`harmonic-knapsack/`](./harmonic-knapsack).

### The Mathematical Lever
Unlike classic knapsack where items are static scalars, workloads here are time-varying periodic curves $L_i(t)$. When co-allocated on a shared resource, their demands superimpose:
$$L_{\text{agg}}(t) = \sum_{i=1}^N x_i L_i(t - \tau_i), \quad \text{constrained by } \max_t L_{\text{agg}}(t) \le C$$
By the Fourier time-shift property, shifting an item in time corresponds to multiplying its Fourier coefficients by a unit complex phase rotation:
$$\mathcal{F}\{L_i(t - \tau_i)\} = X_{i,k} \cdot e^{-i k \omega_i \tau_i}$$
This transforms combinatorial time-domain convolution into an **$O(K \log K)$ analytical frequency-domain search**.

### Academic Lineage & Verified Applications
Built upon deep comparative analysis of foundational literature:
1. **Beaumont et al. (*Euro-Par 2015*, Springer / HAL `hal-01118176`):** Column generation and antagonistic job aggregation for periodic Google batch jobs.
2. **FTIO: Fast Temporal I/O (Tarraf & Wolf, TU Darmstadt / Exascale ADMIRE):** DFT-based checkpoint I/O detection delivering **$+26\%$ system utilization** and **$-56\%$ I/O wait cuts**.
3. **Phase-Scattering Scheduling in AI Training Clusters (Tardif, arXiv:2607.19638 / Choukse et al., arXiv:2508.14318):** Mitigating emergent Kuramoto phase-locking and destructive megawatt-level power swings across GPU racks under shared facility power caps.

### Performance Across Verified Domains
- **AI Datacenter Power Capping:** Slashed peak GPU rack power from $56.5\text{ kW}$ down to $29.3\text{ kW}$ (**$-48.2\%$ reduction**), fitting safely beneath the physical breaker cap. Closed-loop phase re-centering maintained zero cap trips under network jitter.
- **HPC Supercomputing Checkpoints:** Reduced peak file-system bandwidth from $66.7\text{ Gbps}$ to $15.8\text{ Gbps}$ (**$-76.3\%$ reduction**), eliminating storage burst-buffer contention.
- **Harmonic Knapsack Value Maximization:** Admitted $5$ periodic items (value $500$) vs $3$ items (value $300$) under static peak reservations (**$+66.7\%$ value gain**).
- **Cron Thundering-Herd Elimination:** Slashed top-of-the-hour concurrency across $5,000$ servers down to $210$ concurrent jobs (**$-95.8\%$ reduction**).
- **Physical Signal Processing:** Causal FxLMS Active Noise Cancellation achieved **$26.13\text{ dB}$ noise attenuation**, while Capon MVDR adaptive beamforming placed a **$35.3\text{ dB}$ deep spatial null** on interferers.

---

## 3. Repository Directory Structure

```
.
├── README.md                                  # This master guide
├── .gitignore                                 # Git rules (excludes heavy raw datasets)
│
├── sscf/                                      # SYSTEM 1: Stochastic-Shaped CXL Fabric
│   ├── README.md                              # SSCF Architecture & Validation Guide
│   ├── VALIDATION_REPORT.md                   # Full empirical report on Azure V1 fleet trace
│   ├── m0_validation/                         # M0 Fourier gate & M0' rolling splits code
│   │   ├── download_azure_files.py
│   │   ├── azure_v1_to_matrix.py
│   │   ├── m0_gate_metric.py
│   │   ├── m0_prime_checks.py
│   │   ├── labels.npy, w.npy                  # Workload metadata
│   │   └── m0_tests/                          # Historical empirical validation scripts
│   ├── plane1_admission/                      # Plane 1: Multi-Dimensional Knapsack Packer
│   │   ├── knapsack_packer.py
│   │   └── eval_harness.py
│   ├── plane2_shaper/                         # Plane 2: Host Active Shaper Daemon
│   │   ├── cgroup_shaper_daemon.py
│   │   └── shaper_agent.py
│   ├── plane3_cxl/                            # Plane 3: Disaggregated Memory Tiering
│   │   └── tiering_model.py
│   └── tests/                                 # SSCF Test Suite
│       ├── test_components.py
│       └── test_sscf_end_to_end.py
│
└── harmonic-knapsack/                         # SYSTEM 2: Harmonic Temporal Knapsack
    ├── README.md                              # Harmonic Knapsack Overview
    ├── docs/                                  # Research & Mathematical Foundations
    │   ├── 01_problem_formulation.md          # Classic vs. Temporal Knapsack
    │   ├── 02_prior_art_and_literature.md     # Beaumont, FTIO, AI Power Capping
    │   ├── 03_verified_application_domains.md # The 7 verified engineering application domains
    │   ├── 04_signal_processing_analysis.md   # ANC FxLMS causality, Beamforming SMI / RMB
    │   └── 05_production_engineering_roadmap.md # Production Architecture & 3 Pillars
    ├── core/                                  # Core Fourier Mathematics & Solvers
    │   ├── harmonic_representation.py         # Spectral decomposition, FTIO confidence
    │   ├── phase_shift_math.py                # Exact analytical phase rotation & O(M log M) IFFT
    │   └── temporal_knapsack_solver.py        # Peak-Minimization, Value-Maximization & greedy_fft
    ├── production/                            # Production Hardware & Closed-Loop Modules
    │   ├── telemetry_sensor.py                # Pillar 1: Sub-second NVML/PDU Telemetry Sensor
    │   ├── nccl_actuator_hook.py              # Pillar 2: Micro-barrier AllReduce Actuator Hook
    │   ├── rate_diversity_controller.py       # Pillar 3: DynamicBatchPacer & Harmonic Quantizer
    │   ├── live_nvml_benchmark.py             # End-to-end closed-loop live hardware benchmark
    │   ├── generate_pitch_deck_charts.py      # Pitch deck chart generator (publication quality)
    │   └── charts/                            # Generated 300 DPI pitch-deck charts
    ├── applications/                          # Industrial Domain Models
    │   ├── hpc_checkpoint_ftio.py             # HPC Checkpoint I/O Staging (FTIO model)
    │   ├── ai_training_power_cap.py           # AI GPU Cluster Power-Cap Phase Scattering
    │   ├── cron_staggering.py                 # Enterprise Cron Thundering-Herd Staggering
    │   └── tsn_cyclic_scheduler.py            # Time-Sensitive Networking (IEEE 802.1Qbv)
    ├── signal_processing/                     # Signal Processing Real-World Models
    │   ├── anc_fxlms_simulator.py             # FxLMS ANC with causality & Pareto stability
    │   └── adaptive_beamformer.py             # MVDR Capon beamformer with steering mismatch
    └── tests/                                 # Harmonic Test Suite
        ├── test_harmonic_math.py              # Phase math & IFFT cross-correlation tests
        ├── test_solver_benchmarks.py          # Solvers & domain application benchmarks
        ├── test_signal_processing.py          # ANC & Beamforming verification
        └── test_production_runtime.py         # Sensor, Actuator & Closed-Loop hardware tests
```

---

## 4. How to Run the Verification Suites

### Running System 1 (SSCF) Tests
```bash
# Component unit tests (Knapsack admission, cgroups v2, CXL MCE containment)
python sscf/tests/test_components.py

# Complete 72-hour Azure fleet holdout validation
python sscf/tests/test_sscf_end_to_end.py
```

### Running System 2 (Harmonic Knapsack) Tests & Benchmarks
```bash
# Complete pytest test suite (all 19 tests across math, solvers, and production)
python -m pytest harmonic-knapsack/tests/

# Run the live closed-loop NVML hardware benchmark (Sensor -> Controller -> Actuator)
python harmonic-knapsack/production/live_nvml_benchmark.py

# Generate the 3 pitch-deck charts (saved to harmonic-knapsack/production/charts/)
python harmonic-knapsack/production/generate_pitch_deck_charts.py
```
