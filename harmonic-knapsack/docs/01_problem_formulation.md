# Mathematical Problem Formulation: Classic Knapsack vs. Temporal Peak-Constrained Harmonic Knapsack

---

## 1. The Fundamental Distinction

A common point of confusion in systems optimization is conflating **Classic Knapsack** with **Temporal / Vector Bin Packing** and **Phase-Offset Peak Minimization**. 

### 1.1 Classic 0/1 Knapsack Problem
In the classic formulation, items are static scalar entities:
$$\begin{aligned}
\text{Maximize} \quad & \sum_{i=1}^N v_i x_i \\
\text{Subject to} \quad & \sum_{i=1}^N w_i x_i \le W, \quad x_i \in \{0, 1\}
\end{aligned}$$
- **Weights ($w_i$)**: Scalar, invariant across time.
- **Capacity ($W$)**: Single static scalar boundary.
- **Temporal Degree of Freedom**: **Zero**. Items do not have time-varying behavior; once placed in a bin, they consume their weight continuously and statically.

### 1.2 Temporal / Vector Peak-Constrained Bin Packing
In real-world physical and digital systems (cloud VMs, EV chargers, network queues, GPU training clusters), workloads are **time-varying curves** $L_i(t)$. When multiple workloads are assigned to a shared resource (a server CPU, a power transformer, an Ethernet switch queue), their demands add linearly at each point in time $t \in [0, T)$:
$$L_{\text{agg}}(t) = \sum_{i=1}^N x_i L_i(t)$$
The resource capacity $C$ is a physical ceiling that **cannot be exceeded at any instantaneous time $t$**:
$$\max_{t \in [0, T)} L_{\text{agg}}(t) \le C$$

If workloads are fixed in time, this is **Vector Bin Packing** (with $T$ dimensions corresponding to time samples), which is strongly NP-hard.

---

## 2. The Gated Harmonic Scheduling Advantage: Introducing Phase Offsets ($\phi_i$)

In many real systems, jobs are **cyclic, recurring, or deferrable**:
- A backup script runs every 60 minutes, taking 5 minutes.
- An HPC simulation checkpoints every 30 minutes.
- An AI training iteration alternates compute and communication every 250 milliseconds.
- An electric vehicle battery draws 11 kW over a 4-hour window within an 8-hour overnight dwell time.

Because these workloads repeat with period $T_i$, the scheduler possesses a critical control lever: **the start offset (or phase angle)** $\tau_i \in [0, T_i)$ (or $\phi_i = \omega_i \tau_i \in [0, 2\pi)$):
$$L_i(t; \tau_i) = L_i\left((t - \tau_i) \pmod{T_i}\right)$$

### 2.1 The Harmonic / Fourier Transformation
Any periodic load profile $L_i(t)$ with fundamental period $T_i$ and fundamental frequency $\omega_i = \frac{2\pi}{T_i}$ can be represented by its complex Fourier series:
$$L_i(t) = c_{i,0} + \sum_{k=1}^K \left( a_{i,k} \cos(k \omega_i t) + b_{i,k} \sin(k \omega_i t) \right) = \sum_{k=-K}^K X_{i,k} e^{i k \omega_i t}$$
where $c_{i,0} = \mu_i$ is the DC mean utilization, and $X_{i,k} = \frac{1}{2}(a_{i,k} - i b_{i,k})$ are the complex Fourier coefficients.

### 2.2 The Phase Shift Property
By the time-shift property of the Fourier Transform:
$$\mathcal{F}\{L_i(t - \tau_i)\} = X_{i,k} \cdot e^{-i k \omega_i \tau_i}$$

Shifting a workload in time by $\tau_i$ is **algebraically equivalent to multiplying each harmonic coefficient $k$ by the unit complex phase factor $e^{-i k \omega_i \tau_i}$**:
$$L_i(t - \tau_i) = \sum_{k=-K}^K \left( X_{i,k} e^{-i k \phi_i} \right) e^{i k \omega_i t}, \quad \text{where } \phi_i = \omega_i \tau_i$$

### 2.3 Why This Makes Searching Start Offsets Cheap
In the time domain, searching for the optimal start offset $\tau_i \in \{0, 1, \dots, T-1\}$ against an existing aggregate schedule $L_{\text{bin}}(t)$ requires:
1. Shifting the $T$-point vector for each candidate offset: $O(T)$ operations.
2. Summing with the existing load and computing the maximum: $O(T)$ operations per offset.
3. Over $T$ candidate offsets, naive time-domain evaluation costs $O(T^2)$.

In the Fourier / Harmonic domain:
1. Workloads are compactly compressed into their $K$ dominant harmonic components ($K \ll T$, typically $K \in [3, 8]$).
2. The cross-correlation between the candidate workload and the host schedule is given by:
   $$R_{L_{\text{bin}}, L_i}(\tau) = \mathcal{F}^{-1} \left\{ \hat{L}_{\text{bin}}(\omega) \cdot \hat{L}_i^*(\omega) \right\}$$
3. A single inverse FFT computes the aggregate profile for **all possible continuous start offsets simultaneously in $O(K \log K)$ operations**.
4. Furthermore, two periodic signals of identical or harmonic frequencies achieve **maximal destructive wave cancellation** (minimum peak) when their primary harmonic phases satisfy:
   $$\phi_1 - \phi_2 \equiv \pi \pmod{2\pi}$$

---

## 3. Mathematical Optimization Formulations

### 3.1 Formulation A: Peak-Minimizing Harmonic Scheduling (Fixed Admitted Set)
Given a set of $N$ periodic tasks assigned to a resource, determine the optimal phase offsets $\vec{\tau} = (\tau_1, \dots, \tau_N)$ that minimize the peak aggregate demand:
$$\min_{\tau_1, \dots, \tau_N} \max_{t \in [0, T_{\text{hyper}})} \sum_{i=1}^N L_i(t - \tau_i)$$
where $T_{\text{hyper}} = \text{LCM}(T_1, T_2, \dots, T_N)$ is the hyperperiod of the task set.

### 3.2 Formulation B: Value-Maximizing Harmonic Temporal Knapsack
Given a resource with instantaneous capacity ceiling $C$, and a candidate pool of $N$ tasks, each with value $v_i$, period $T_i$, and shape $L_i(t)$:
$$\begin{aligned}
\text{Maximize} \quad & \sum_{i=1}^N v_i x_i \\
\text{Subject to} \quad & \max_{t \in [0, T_{\text{hyper}})} \sum_{i=1}^N x_i L_i(t - \tau_i) \le C \\
& x_i \in \{0, 1\}, \quad \tau_i \in [0, T_i)
\end{aligned}$$
Here, the decision variables are **both** the admission decisions $x_i$ and the phase offsets $\tau_i$. By staggering phases constructively, the knapsack can admit significantly more jobs ($+20\% \text{ to } +40\%$) than if worst-case in-phase peak alignment were assumed.

---

## 4. Hyperperiod and Beat Frequency Constraints

While the harmonic view is mathematically elegant, practical implementations must respect two physical realities:
1. **Hyperperiod Explosion:** If periods $T_1, T_2, \dots$ are incommensurate (e.g. $10.0\text{ ms}$ and $10.5\text{ ms}$), the least common multiple $\text{LCM}(10.0, 10.5) = 210\text{ ms}$. If periods are coprime, $T_{\text{hyper}} = \prod T_i$, creating low-frequency **beat patterns** that eventually cause peaks to collide.
2. **Harmonic Period Quantization:** The system achieves optimal packing density when periods are **integer multiples / harmonic** (e.g., $10\text{ ms}, 20\text{ ms}, 40\text{ ms}, 80\text{ ms}$), in which case $T_{\text{hyper}} = \max_i T_i$, and phase offsets remain stationary indefinitely.
