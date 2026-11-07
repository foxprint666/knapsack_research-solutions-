# Signal Processing Analysis: Physical Superposition, Active Noise Cancellation, and Adaptive Beamforming

This document provides a rigorous mathematical breakdown of physical signal-processing systems (Active Noise Cancellation and Adaptive Beamforming), detailing why wave cancellation succeeds in true physical fields, what constraints govern its stability, and how these constraints parallel computational scheduling.

---

## 1. Active Noise Cancellation (ANC)

### 1.1 The Physical Superposition Principle
Unlike cloud server CPU scheduling (which is a queuing / resource multiplexing problem bounded by $0\% \le u \le 100\%$), acoustic sound is an oscillatory pressure fluctuation $\Delta p(x, t)$ governed by the linear wave equation:
$$\nabla^2 p - \frac{1}{c^2} \frac{\partial^2 p}{\partial t^2} = 0$$
Linear superposition holds rigorously:
$$e(t) = d(t) + y(t)$$
where $d(t)$ is the primary acoustic noise disturbance and $y(t)$ is the anti-noise produced by a secondary loudspeaker. If $y(t) = -d(t)$, destructive interference yields $e(t) = 0$.

### 1.2 The Filtered-X Least Mean Squares (FxLMS) Algorithm
In a feedforward ANC system:
- **Reference Microphone:** Detects upstream noise $x(n)$.
- **Primary Acoustic Path $P(z)$:** Physical acoustic transfer function between noise source and error sensor. Primary disturbance: $d(n) = P(z) x(n)$.
- **Secondary Electro-Acoustic Path $S(z)$:** Includes DAC converter, reconstruction filter, power amplifier, loudspeaker, acoustic air distance to error mic, error microphone response, anti-aliasing filter, and ADC converter.
- **Adaptive Control Filter $W(z)$:** Finite Impulse Response (FIR) filter of length $L$ generating the control signal:
  $$y(n) = \sum_{l=0}^{L-1} w_l(n) x(n - l)$$

#### Why Standard LMS Goes Unstable
The error signal measured at the listener's ear is:
$$e(n) = d(n) - s(n) * y(n) = d(n) - s(n) * \left( \mathbf{w}^T(n) \mathbf{x}(n) \right)$$
Because the secondary path $S(z)$ introduces phase shifts and time delays, updating the filter weights using standard LMS ($\mathbf{w}(n+1) = \mathbf{w}(n) + \mu e(n) \mathbf{x}(n)$) creates a phase mismatch between the error and the gradient direction. If the phase error of $S(z)$ exceeds $\pm 90^\circ$, the feedback becomes positive and the filter diverges violently.

#### The FxLMS Derivation
To guarantee convergence, Morgan (1980) and Burgess (1981) introduced **Filtered-X LMS**: the reference signal $\mathbf{x}(n)$ is pre-filtered by an internal model of the secondary path $\hat{S}(z)$ before computing the gradient:
$$x'(n) = \sum_{m=0}^{M-1} \hat{s}_m x(n - m)$$
The weight update equation becomes:
$$\mathbf{w}(n + 1) = \mathbf{w}(n) + \mu \, e(n) \, \mathbf{x}'(n)$$
where $\mathbf{x}'(n) = [x'(n), x'(n-1), \dots, x'(n-L+1)]^T$.
As long as the phase error between the true secondary path $S(z)$ and the model $\hat{S}(z)$ satisfies:
$$\left| \angle S(e^{j\omega}) - \angle \hat{S}(e^{j\omega}) \right| < 90^\circ \quad \forall \omega$$
FxLMS is mathematically guaranteed to converge to the optimal Wiener solution.

### 1.3 The Strict Causality Constraint
For feedforward ANC to cancel broadband noise, the anti-noise wave must arrive at the error microphone **at or before** the acoustic noise wave traveling through the air.
Let:
- $\tau_{\text{acoustic}} = d_{\text{ref-to-err}} / c_{\text{sound}}$ be the acoustic travel time.
- $\tau_{\text{electronic}} = \tau_{\text{ADC}} + \tau_{\text{DSP}} + \tau_{\text{DAC}} + \tau_{\text{amp}}$ be the total electronics group delay.

$$\textbf{Strict Causality Condition:} \quad \tau_{\text{electronic}} < \tau_{\text{acoustic}}$$

If $\tau_{\text{electronic}} > \tau_{\text{acoustic}}$, the controller is **acausal**. It cannot predict uncorrelated broadband noise, and cancellation is physically impossible. The controller can only cancel narrowband periodic tones (via internal sinusoidal modeling), which explains why ANC headphones excel at removing continuous airplane engine hum ($\sim 100\text{ Hz}$) but fail to cancel sudden speech or door slams.

### 1.4 Impulsive Pareto Bursts vs. Gaussian Noise
Standard FxLMS minimizes the mean squared error (MSE) $J(n) = E[e^2(n)]$.
- Under Gaussian noise, MSE is optimal.
- When disturbances follow **heavy-tailed Pareto or $\alpha$-stable distributions** (extreme sudden outliers, identical to cloud CPU bursts), the second moment $E[e^2(n)]$ is unbounded.
- A single large burst produces an enormous gradient $\mu e(n) \mathbf{x}'(n)$ that destabilizes the FIR coefficients.
- **Remedy:** Robust algorithms replace $e^2(n)$ with M-estimators or the **Normalized Sign-LMS (NSLMS)**:
  $$\mathbf{w}(n+1) = \mathbf{w}(n) + \mu \frac{\text{sign}(e(n)) \, \mathbf{x}'(n)}{\|\mathbf{x}'(n)\|^2 + \epsilon}$$

---

## 2. Adaptive Beamforming & Spatial Nulling

### 2.1 The Antenna Array Model
Consider an array of $M$ antenna elements spaced uniformly at distance $d = \lambda / 2$ along an axis.
An incoming plane wave from azimuth angle $\theta$ arrives with a progressive phase delay between adjacent elements:
$$\Delta \phi = \frac{2\pi}{\lambda} d \sin \theta = \pi \sin \theta \quad (\text{for } d = \lambda/2)$$
The spatial response across the $M$ elements is captured by the **array steering vector**:
$$\mathbf{a}(\theta) = \left[ 1, e^{-j \pi \sin \theta}, e^{-j 2\pi \sin \theta}, \dots, e^{-j (M-1) \pi \sin \theta} \right]^T \in \mathbb{C}^M$$

The received snapshot vector $\mathbf{x}(n) \in \mathbb{C}^M$ is:
$$\mathbf{x}(n) = \mathbf{a}(\theta_0) s(n) + \sum_{k=1}^K \mathbf{a}(\theta_k) i_k(n) + \mathbf{v}(n)$$
where $s(n)$ is the desired signal, $i_k(n)$ are directional interferers, and $\mathbf{v}(n)$ is spatial white noise.

### 2.2 Minimum Variance Distortionless Response (MVDR / Capon)
The beamformer combines the antenna signals with complex weights $\mathbf{w} \in \mathbb{C}^M$:
$$y(n) = \mathbf{w}^H \mathbf{x}(n)$$
The goal is to minimize total output power (thereby suppressing interference and noise) while maintaining unity distortionless gain in the direction of the desired signal $\theta_0$:
$$\begin{aligned}
\min_{\mathbf{w}} \quad & \mathbf{w}^H \mathbf{R}_{xx} \mathbf{w} \\
\text{s.t.} \quad & \mathbf{w}^H \mathbf{a}(\theta_0) = 1
\end{aligned}$$
where $\mathbf{R}_{xx} = E[\mathbf{x}(n) \mathbf{x}^H(n)]$ is the $M \times M$ spatial covariance matrix.

Using the method of Lagrange multipliers, the optimal weight vector is:
$$\mathbf{w}_{\text{opt}} = \frac{\mathbf{R}_{xx}^{-1} \mathbf{a}(\theta_0)}{\mathbf{a}^H(\theta_0) \mathbf{R}_{xx}^{-1} \mathbf{a}(\theta_0)}$$
An array of $M$ elements possesses $M-1$ spatial degrees of freedom, allowing it to place up to **$M-1$ independent deep nulls ($<-40\text{ dB}$)** in the directions of interferers while steering its main beam toward the desired source.

### 2.3 Sample Matrix Inversion (SMI) & The RMB Rule
In reality, the true ensemble covariance $\mathbf{R}_{xx}$ is unknown and must be estimated from $K$ temporal snapshots:
$$\hat{\mathbf{R}}_{xx} = \frac{1}{K} \sum_{k=1}^K \mathbf{x}(k) \mathbf{x}^H(k)$$
The **Reed-Mallett-Brennan (RMB) Rule** (IEEE Trans. AES, 1974) establishes the fundamental snapshot requirement:
$$\textbf{RMB Rule:} \quad K \ge 2M \text{ snapshots}$$
To achieve an output Signal-to-Interference-plus-Noise Ratio (SINR) within $3\text{ dB}$ of optimal performance, the number of snapshots $K$ must be at least twice the number of antenna elements $M$.

### 2.4 Regularization via Diagonal Loading
When $K$ is small ($K \approx M$) or when interferers are weak, the sample covariance matrix $\hat{\mathbf{R}}_{xx}$ is ill-conditioned. Inverting it leads to noise amplification and distorted sidelobes.
**Diagonal loading** regularizes the inversion:
$$\hat{\mathbf{R}}_{\text{loaded}} = \hat{\mathbf{R}}_{xx} + \alpha \mathbf{I}$$
where $\alpha \approx 10 \cdot \sigma_{\text{noise}}^2$. This penalizes excessively large weight magnitudes and stabilizes the nulling pattern.

### 2.5 Steering Vector Mismatch & Signal Cancellation
If the assumed look direction $\theta_{\text{look}}$ differs from the true source direction $\theta_0$ by even a fraction of a degree:
$$\mathbf{a}(\theta_{\text{look}}) \neq \mathbf{a}(\theta_0)$$
The beamformer views the actual desired signal as an unsanctioned interferer! Because the signal is present in $\hat{\mathbf{R}}_{xx}$, the optimizer **places a sharp spatial null directly on the desired signal**, causing catastrophic signal cancellation.

---

## 3. Comparison with Cloud Workload Wave Cancellation

| Characteristic | Physical Waves (ANC / Beamforming) | Cloud Workloads (Azure Fleet) |
| :--- | :--- | :--- |
| **Superposition Physics** | True linear wave field ($\Delta p_1 + \Delta p_2$) | Queuing resource contention ($0 \le \sum u_i \le C$) |
| **Negative Values** | Exists naturally (anti-phase sound wave) | Impossible (CPU demand cannot be negative) |
| **Feedback Mechanism** | Closed-loop error sensor ($e(n) = d - y$) | Usually open-loop placement without runtime micro-shaping |
| **Noise Distribution** | Near-Gaussian acoustic/RF noise | Heavy-tailed Pareto / Log-normal bursts ($\text{VIF} = 2.63$) |
| **Causality & Timing** | Microsecond physical propagation | OS scheduler quantum ($1\text{ ms} - 100\text{ ms}$) |
| **Cancellation Viability** | **PROVEN** (subject to causality & RMB rules) | **DISPROVEN for open-loop IaaS; VIABLE for scheduled batch/AI** |
