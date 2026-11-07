# Stochastic-Shaped CXL Fabric (SSCF)

A production-grade cloud infrastructure architecture designed around the actual statistical distributions of the Microsoft Azure fleet trace ($4{,}994$ VMs $\times 864$ intervals, $796{,}464$ core-hours): low mean utilization ($6.34\%–9.07\%$), extreme stochastic P99 bursts ($64.00\%–81.83\%$), and positive cross-VM co-bursting ($\text{VIF} = 2.63$).

---

## 1. Why SSCF? (The M0 Empirical Gate)

Earlier theoretical cloud architectures hypothesized that multi-tenant servers could achieve massive density through **harmonic wave cancellation** (destructive interference of diurnal sinusoidal peaks). 

To test this hypothesis before writing production code, we executed the **M0 Spectral Gate** on the real public Azure V1 dataset:
1. **Periodicity Finding:** Fourier power spectral density analysis proved that **only $5.4\% \text{ to } 9.7\%$ of Azure fleet core-hours exhibit diurnal periodicity**—far below the $50\%$ gate floor required for wave cancellation.
2. **Co-Bursting Finding:** Measuring the Variance Inflation Factor across physical racks revealed $\text{VIF} = 2.63$. Workloads co-burst rather than cancel.
3. **Variance Quadrature Ablation:** Testing Gaussian Effective Bandwidth ($\mu + 2.33\sigma$) vs. a Mean-Utilization Cap alone ($\sum \mu \le U \cdot C$) proved that the variance term was non-binding; the mean utilization cap was doing $100\%$ of the work.

Per **ADR-0009**, Fourier wave cancellation was retired for cloud IaaS VMs, and the **Stochastic-Shaped CXL Fabric (SSCF)** was adopted across three empirically grounded planes.

---

## 2. Three-Plane Architecture

```
+-------------------------------------------------------------------------------------------------+
|                                  SSCF THREE-PLANE ARCHITECTURE                                  |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   PLANE 1: Admission & Placement (Seconds timescale)                                            |
|   Multi-Dimensional Mix-Aware Knapsack Packer                                                   |
|   - Rules: sum(mu_total) <= 70% * C  AND  sum(mu_interactive) <= 30% * C                        |
|   - Knapsack packing sorts workloads by composite demand vector                                 |
|   - Prevents multi-VM interactive co-bursts from ever saturating physical server capacity       |
|                                                                                                 |
|   PLANE 2: Host Runtime Active Shaper (Microseconds timescale)                                  |
|   Linux cgroups v2 Kernel Controller Daemon                                                     |
|   - Interactive Slice: unthrottled (cpu.weight = 1000, cpu.max = max)                           |
|   - Batch Slice: elastic (cpu.weight = 100, dynamic cpu.max)                                    |
|   - Monitors /proc/pressure/cpu (PSI); dynamically clamps batch during burst events             |
|   - Causal FIFO deferral preserves 100% of batch work mass while protecting interactive SLAs   |
|   - Strict Fail-Open safety: resets to unthrottled on errors or sensor clearing                 |
|                                                                                                 |
|   PLANE 3: Disaggregated Memory Tiering (Nanoseconds timescale)                                 |
|   Failure-Isolated CXL Pooled Memory Engine (ADR-0010)                                          |
|   - KERNEL_PINNED: Locked to local host DDR5 NUMA node 0 via MPOL_BIND                          |
|   - ANONYMOUS_DIRTY: Retained in local DDR5 / compressed zswap                                  |
|   - FILE_BACKED_CLEAN: Clean pagecache and binaries demoted to shared CXL memory pool           |
|   - Fault Containment: Sudden CXL link reset / fabric pull causes ZERO Machine Check            |
|     Exceptions (MCEs) and ZERO kernel panics (clean pages re-read from local NVMe storage)       |
|                                                                                                 |
+-------------------------------------------------------------------------------------------------+
```

---

## 3. Directory Structure

```
sscf/
├── README.md                                  # This architecture overview
├── VALIDATION_REPORT.md                       # Complete empirical validation report on Azure V1
├── m0_validation/                             # M0 Spectral gate & M0' rolling splits validation
│   ├── download_azure_files.py                # Script to download Azure V1 traces
│   ├── azure_v1_to_matrix.py                  # Ingestion & matrix generation
│   ├── m0_gate_metric.py                      # Fourier spectral share test
│   ├── m0_prime_checks.py                     # Rolling splits, VIF, and variance ablation
│   ├── m0_prime_replay.py                     # Historical holdout verification
│   ├── labels.npy                             # Workload classification metadata (4,994 VMs)
│   ├── w.npy                                  # VM core-hour weights (796,464 core-hours)
│   └── m0_tests/                              # Historical empirical validation scripts
├── plane1_admission/                          # Plane 1: Admission & Placement
│   ├── knapsack_packer.py                     # Dual-constraint Knapsack Packer (BFD, FFD, local search)
│   └── eval_harness.py                        # PlacementScheduler & LindleyQueueSimulator
├── plane2_shaper/                             # Plane 2: Host Runtime Active Shaper
│   ├── cgroup_shaper_daemon.py                # Linux cgroups v2 & PSI controller daemon
│   └── shaper_agent.py                        # Causal smoothing & mass preservation agent
├── plane3_cxl/                                # Plane 3: Failure-Isolated Memory Tiering
│   └── tiering_model.py                       # DAMON cold page detection & MCE fault containment
└── tests/                                     # Verification Test Suites
    ├── test_components.py                     # Unit tests for all individual plane components
    └── test_sscf_end_to_end.py                # 72-hour multi-day holdout validation on Azure trace
```

---

## 4. End-to-End Validation Results (Real Azure Fleet Trace)

Executed via `python sscf/tests/test_sscf_end_to_end.py` across $4{,}994$ VMs over 72 hours (864 bins):

| Gate / Metric | Target Gate | Real Trace Result | Status |
| :--- | :--- | :--- | :--- |
| **Server Footprint Reduction vs Static P99** | $\ge 23.0\%$ | **+25.45%** ($34$ hosts vs $45.6$ ideal P99) | **PASSED** |
| **Day 1 Raw Host Overload** | $< 0.50\%$ | **0.0204%** | **PASSED** |
| **Day 1 Interactive SLA Violations** | $0.0000\%$ | **0** ($0.0000\%$) | **PASSED** |
| **Day 2 Raw Host Overload** | $< 0.50\%$ | **0.3983%** | **PASSED** |
| **Day 2 Interactive SLA Violations** | $0.0000\%$ | **0** ($0.0000\%$) | **PASSED** |
| **Active Shaper Residual Overload** | $0.00\%$ | **0.00%** ($0.69\% \to 0.00\%$) | **PASSED** |
| **Batch Demand Mass Conservation** | $\Delta < 10^{-4}$ | **True** (Exact causal FIFO conservation) | **PASSED** |
| **Local Host DRAM Headroom (CXL)** | $> 0.0\%$ | **+35.0%** ($506.8$ GB demoted to CXL pool) | **PASSED** |
| **Hardware MCEs on CXL Link Reset** | $0$ MCEs | **0 MCEs** ($100\%$ host survival) | **PASSED** |
| **Host Kernel Panics on Link Reset** | $0$ Panics | **0 Panics** (Clean pages re-read from NVMe) | **PASSED** |

---

## 5. How to Run SSCF Tests

```bash
# 1. Run component unit tests
python sscf/tests/test_components.py

# 2. Run full 72-hour Azure fleet holdout validation
python sscf/tests/test_sscf_end_to_end.py
```
