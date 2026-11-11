# Production Engineering Architecture: From Simulation to Production-Grade Product

This document details the production-grade engineering architecture required to turn the **Harmonic Temporal Knapsack** framework from a research simulation into a commercially deployable, investable datacenter product.

---

## 1. Solver Acceleration: Discrete Grid vs. True $O(M \log M)$ IFFT Cross-Correlation

### 1.1 The Computational Bottleneck
In the research prototype (`core/temporal_knapsack_solver.py`), candidate phase shifts were evaluated using an exhaustive discrete search:
- Evaluating $S$ candidate offsets over $M$ time bins across $N$ sequential items cost:
  $$\mathcal{O}(N^2 \cdot S \cdot M) \text{ operations}$$
- While sufficient for $N \le 20$ jobs, scaling to cluster-level scheduling ($N \ge 1,000$ jobs) creates an $O(N^2)$ barrier.

### 1.2 The Analytical IFFT Solution
By the **Circular Convolution / Cross-Correlation Theorem**, the cross-correlation $R_{AB}(\tau)$ between host aggregate load $A(t)$ and candidate waveform $B(t)$ discretized into $M$ bins is:
$$R_{AB}[k] = \sum_{m=0}^{M-1} A[m] B[(m - k) \pmod M] = \mathcal{F}^{-1} \left\{ \mathcal{F}(A) \cdot \mathcal{F}^*(B) \right\}$$

- **$L_2$ Destructive Interference Minimum:**
  $$\|A + B_{(k)}\|_2^2 = \|A\|_2^2 + \|B\|_2^2 + 2 R_{AB}[k]$$
  Minimizing $R_{AB}[k]$ directly minimizes the total aggregate energy and variance (crest-factor minimization).
- **Complexity:** Evaluating all $M$ continuous offsets simultaneously requires only a single forward FFT and inverse FFT:
  $$\mathcal{O}(M \log M)$$
- **Implemented Method:** `PhaseShiftEngine.cross_correlation_fft()` and `TemporalHarmonicKnapsackSolver.solve_peak_minimization(method="greedy_fft")` in `core/phase_shift_math.py` and `core/temporal_knapsack_solver.py`.

---

## 2. Pillar 1: Real-Time Telemetry Ingestion (The Sensor)

Implemented in `production/telemetry_sensor.py`:
- **Sub-Second Sampling:** Ingests $20\text{ms} - 50\text{ms}$ ($20 - 50\text{ Hz}$) GPU power telemetry directly from NVIDIA NVML (`nvmlDeviceGetPowerUsage`), DCGM telemetry engines, or rack PDU SNMP/Modbus streams.
- **Lock-Free Ring Buffer:** `PowerSampleRingBuffer` provides zero-allocation circular storage for continuous time series.
- **Sliding-Window Online FFT:** `StreamingSpectralAnalyzer` maintains an online sliding window (e.g. 4–5 seconds) to extract the fundamental frequency $f_0$, harmonic amplitudes, and phase angle $\phi_0$ via `HarmonicProfile.from_timeseries()`.
- **Phase Drift Tracking:** Continuously monitors instantaneous phase drift:
  $$\Delta \phi(t) = \left(\phi_{\text{measured}}(t) - \phi_{\text{target}}\right) \pmod{2\pi}$$
  Triggering closed-loop re-centering when network jitter causes cumulative phase slippage.

---

## 3. Pillar 2: The Non-Intrusive Actuation Hook (The Actuator)

Implemented in `production/nccl_actuator_hook.py`:
- **The Core Problem:** How to shift the phase $\tau_i$ of an active PyTorch distributed LLM training job by $150\text{ms}$ without reducing training throughput?
- **The Solution: One-Off Micro-Barrier at Collective Boundary:**
  - Training iterations consist of a high-power Compute phase (GEMM forward/backward) followed by communication (NCCL AllReduce / ReduceScatter).
  - Injecting a **single $150\text{ms}$ delay** at the completion of AllReduce permanently shifts the phase coordinate $\tau_i \leftarrow (\tau_i + 150\text{ms}) \pmod T$ for all subsequent iterations.
  - Across a $10,000$-step training run ($T = 1.0\text{s}$), a $150\text{ms}$ delay represents:
    $$\text{Throughput Slowdown} = \frac{0.15\text{s}}{10,000\text{s}} = 0.0015\% \quad (> 99.998\% \text{ throughput retention})$$
- **PyTorch Integration:** Hooks directly into PyTorch DDP communication hooks (`torch.distributed.algorithms.ddp_comm_hooks`) or post-backward CUDA event synchronization.

---

## 4. Pillar 3: Rate-Diversity & Higher-Harmonic Handling (The Controller)

Implemented in `production/rate_diversity_controller.py`:
- **The Beat Frequency Hazard (Tardif arXiv:2607.19638):**
  If Job A has $T_A = 800\text{ms}$ ($f_A = 1.25\text{Hz}$) and Job B has $T_B = 1100\text{ms}$ ($f_B = 0.909\text{Hz}$), open-loop phase offsets cannot remain stationary. They produce a beat frequency:
  $$f_{\text{beat}} = |f_A - f_B| = 0.341\text{Hz} \implies T_{\text{beat}} \approx 2.93\text{s}$$
  Every $\approx 2.93\text{ seconds}$, their high-power compute bursts constructively collide at the shared rack power cap!
- **Production Solution 1: Harmonic Period Quantization:**
  - `HarmonicPeriodQuantizer` clusters jobs into discrete harmonic families ($0.5\text{s}, 1.0\text{s}, 2.0\text{s}, 4.0\text{s}$). Jobs within the same family have harmonic integer period ratios ($T_{\text{hyper}} = \max T_i$) and zero beat frequency.
- **Production Solution 2: Dynamic Mini-Batch Pacing:**
  - `DynamicBatchPacer` synthesizes micro-delay pacing ($\delta_{\text{pace}}$) or adjusts gradient accumulation micro-batch sizes by $\pm 5-10\%$ to lock disparate jobs into a common fundamental period or integer harmonic ratio ($1:1$ or $2:1$), entirely eliminating beat collision.
