# Theoretical Framework: Discrete Vehicle Package Pricing, Exact-Reference Artifacts, and Decoded Quantum Interferometry

**Benchmark framework for pricing-to-DQI evaluation with an exact-reference mainline, exploratory Walsh-truncated surrogates, appendix-only coherent checks, and structural Qiskit validation**

---

## Overview

This document establishes the theoretical foundations for the `dqi-pricing-benchmark` project. The project has one mainline benchmark path and two narrower support paths with different maturity and claim scope.

### Mainline benchmark path

The reviewer-facing path is built around exact-reference pricing artifacts and mixture-mode execution. A toy vehicle package pricing problem is formulated as a binary optimization over $n = 2N_f$ bits using a 2-bit encoding of $N_f$ features with three valid tiers. The true objective $F(x)$ combines expected revenue, bundle constraints, invalid-state penalties, and a luxury-cap penalty. At the benchmark scale, $F$ is exactly enumerable and serves as ground truth for all downstream comparisons, including the exact-reference / ILP-derived benchmark path used for paired encoding checks.

### Exploratory surrogate path (WHT)

The true objective $F$ can also be expanded in the Walsh–Hadamard basis and truncated to its top-$K$ nonconstant coefficients, producing a weighted parity surrogate $G_{\mathrm{wt}}$. The DQI circuit, however, acts on an unweighted max-XORSAT objective, so the benchmark also constructs a sign-only surrogate $G_{\mathrm{unw}}$ encoded in the native format $(B,v)$. This WHT path is explicitly exploratory: it is a project-specific approximation layer used to study transfer from the pricing objective to a sparse parity surrogate, not the primary reviewer-facing claim path.

### Structural Qiskit subset validation

The implemented Qiskit port is a structural subset validation path. It ports the circuit construction and selected statevector checks for the DQI core, but it does not claim full decode-inclusive end-to-end parity with the complete benchmark stack. Coherent execution is likewise treated narrowly: it is appendix-only and restricted to tiny supported instances where the joint-state construction remains tractable.

The paper-based DQI circuit core itself remains central to the benchmark logic. The max-XORSAT surrogate $(B,v)$ is used to construct a DQI state via Dicke-state superposition, phase imprinting by $v$, syndrome computation, bounded-distance decoding, uncomputation, and final Hadamard transform. Exact agreement with the paper’s finite-size theory is claimed only for instances in the exact-decoding regime, where the syndrome map is injective on all error patterns of weight at most $\ell$. Outside that regime, the same construction is retained as a benchmark heuristic and reported explicitly as approximate mode rather than as an exact realization.

The purpose of the project is not to claim quantum advantage at small scale. Rather, it is to provide a mathematically explicit benchmark in which the true objective is known exactly, the surrogate reduction is diagnosed at both the weighted and unweighted levels, and the DQI circuit core is validated against the assumptions under which the source-paper theory applies.

**Scope of claims.** Throughout this document, “paper-based” means that the max-XORSAT DQI circuit core, the exact-decoding condition, and the finite-size principal-eigenvector weight rule are taken from [1]. By contrast, the Walsh truncation that maps the pricing objective $F$ to a sparse parity surrogate is a project-specific exploratory construction, and the Qiskit implementation is an implemented structural subset rather than a full-stack parity claim. Accordingly, claims of exact finite-size agreement with [1] are restricted to exact-mode instances, coherent experiments are appendix-only, and approximate-mode runs are reported as heuristic realizations of the same circuit architecture.

**Primary references:**

* [1] Stephen P. Jordan, Noah Shutty, Mary Wootters, Adam Zalcman, Alexander Schmidhuber, Robbie King, Sergei V. Isakov, and Ryan Babbush, "Optimization by Decoded Quantum Interferometry," arXiv:2408.08292 (2024); later published in *Nature* (2025).
* [2] Andreas Bärtschi and Stephan Eidenbenz, "Deterministic Preparation of Dicke States," arXiv:1904.07358 (2019).
* [3] BCG-X-Official/dqi, GitHub repository (BMW–BCG DQI implementation).

---

## 1. The Max-XORSAT Problem and Its Fourier Structure

### 1.1 Max-XORSAT as a binary optimization problem

The max-XORSAT problem is the starting point for the DQI algorithm in the binary ($\mathbb{F}_2$) setting. Given an $m \times n$ binary matrix $B \in \mathbb{F}_2^{m \times n}$ and a target vector $v \in \mathbb{F}_2^m$, the problem asks for an $n$-bit string $x \in \{0,1\}^n$ that satisfies as many as possible of the $m$ linear equations
$$
(Bx)_i = v_i \pmod{2}, \qquad i = 1, ..., m.
$$
Equivalently, the $i$-th constraint is satisfied when $b_i \cdot x \oplus v_i = 0$ (where $b_i$ is the $i$-th row of $B$ and $\oplus$ denotes XOR). The objective function can be written as ([1] Eq. (1))
$$
f(x) = \sum_{i=1}^{m} (-1)^{v_i + b_i \cdot x},
$$
where $b_i \cdot x = \bigoplus_{j=1}^n B_{ij} x_j$ is computed modulo 2. Each term contributes $+1$ when the $i$-th constraint is satisfied and $-1$ when it is violated, so
$$
f(x) = (\text{number satisfied}) - (\text{number violated}) = 2s(x) - m,
$$
where $s(x)$ is the number of satisfied constraints. Maximizing $f(x)$ is therefore equivalent to maximizing $s(x)$.

### 1.2 Fourier sparsity of the objective

A central observation in [1] is that the objective (1.1) has an **extremely sparse Hadamard (Fourier) transform**. To see this, recall that for any function $g: \{0,1\}^n \to \mathbb{R}$, the Walsh–Hadamard expansion is
$$
g(x) = \sum_{S \subseteq [n]} \hat{g}(S)\, (-1)^{\langle S, x \rangle},
$$
where $\langle S, x \rangle = \bigoplus_{j \in S} x_j$ is the parity of $x$ on the index set $S$, and the Fourier coefficients are
$$
\hat{g}(S) = \frac{1}{2^n} \sum_{x \in \{0,1\}^n} g(x)\, (-1)^{\langle S, x \rangle}.
$$

For the max-XORSAT objective (1.1), each term $(-1)^{v_i + b_i \cdot x}$ contributes a single nonzero Fourier coefficient at the index set $S_i = \text{supp}(b_i)$ (the set of nonzero positions in $b_i$). Thus
$$
f(x) = \sum_{i=1}^{m} (-1)^{v_i} \cdot (-1)^{\langle S_i, x \rangle},
$$
which means $f$ has **at most $m$ nonzero Fourier coefficients** (fewer if some rows of $B$ coincide). This sparsity is the key structural property that DQI exploits: the quantum state $\sum_x f(x)|x\rangle$ can be prepared efficiently because its Fourier transform is supported on only $m$ strings $b_1, ..., b_m$.

**Implementation note:** This Fourier sparsity motivates our surrogate construction strategy in §3. A general pricing objective $F(x)$ will not be Fourier-sparse, but by truncating to the top-$K$ Walsh–Hadamard coefficients, we manufacture a surrogate $G(x)$ that approximates $F$ while possessing the sparse parity structure that DQI requires. This reduction is implemented in `src/reduction.py`.

---

## 2. Discrete Vehicle Pricing as Binary Optimization

### 2.1 Feature–tier encoding

Consider $N_f$ vehicle optional features, each offered at 3 price tiers: standard, premium, and luxury. We encode the tier choice for feature $i$ using 2 bits $(a_i, b_i) \in \{0,1\}^2$:

| Bits $(a_i, b_i)$ | Tier | Code |
|---|---|---|
| $(0, 0)$ | Standard | 0 |
| $(0, 1)$ | Premium | 1 |
| $(1, 0)$ | Luxury | 2 |
| $(1, 1)$ | Invalid | 3 |

The full configuration is a bitstring $x = (a_1, b_1, a_2, b_2, ..., a_{N_f}, b_{N_f}) \in \{0,1\}^n$ with $n = 2N_f$.

For each feature $i$, the tier indicators are pseudo-Boolean functions of $(a_i, b_i)$:
$$
\begin{aligned}
\mathbf{1}[\text{Standard}_i] &= (1 - a_i)(1 - b_i), \\
\mathbf{1}[\text{Premium}_i] &= (1 - a_i)\, b_i, \\
\mathbf{1}[\text{Luxury}_i] &= a_i(1 - b_i), \\
\mathbf{1}[\text{Invalid}_i] &= a_i\, b_i.
\end{aligned}
$$
Note that these four indicators sum to 1 for any $(a_i, b_i)$ and partition $\{0,1\}^2$. We also define the "at least premium" indicator, which will appear in bundle constraints:
$$
\text{AP}_i = a_i \oplus b_i = a_i + b_i - 2a_i b_i,
$$
which equals 1 for codes 01 and 10 (premium and luxury) and 0 for codes 00 and 11 (standard and invalid). This is precisely the XOR of the two bits, which is naturally parity-friendly.

**Important subtlety on parity structure:** The one-hot constraint "exactly one tier is active" is **not** equivalent to a single parity condition. For three bits $(x_1, x_2, x_3)$, the parity condition $x_1 \oplus x_2 \oplus x_3 = 1$ allows both $\{001, 010, 100\}$ and $\{111\}$. This is why we use 2-bit encoding rather than 3-bit one-hot: it avoids requiring auxiliary parity constraints that would complicate the reduction.

**Implementation:** The encoding logic is in `src/problem_generator.py` (functions `tier_from_bits`, `decode_bitstring`, `encode_tiers`).

### 2.2 Price function

Let $p_i^{(S)}, p_i^{(P)}, p_i^{(L)}$ denote the prices for standard, premium, and luxury tiers of feature $i$. The price contribution of feature $i$ is
$$
\text{price}_i(x) = p_i^{(S)} \cdot \mathbf{1}[\text{Std}_i] + p_i^{(P)} \cdot \mathbf{1}[\text{Prem}_i] + p_i^{(L)} \cdot \mathbf{1}[\text{Lux}_i].
$$
Invalid states contribute zero to price but are penalized separately. The total package price is
$$
\text{Price}(x) = \sum_{i=1}^{N_f} \text{price}_i(x).
$$

### 2.3 Customer segments and expected revenue

We introduce $K$ customer segments indexed by $k$, each characterized by a budget cap $B_k$ and a population weight $w_k$ with $\sum_k w_k = 1$. A segment purchases the package if and only if the total price does not exceed its budget:
$$
A_k(x) = \mathbf{1}[\text{Price}(x) \le B_k].
$$
The expected revenue across all segments is
$$
R(x) = \sum_{k=1}^{K} w_k \cdot \text{Price}(x) \cdot A_k(x).
$$

This demand model creates a genuine tradeoff: choosing higher tiers increases per-sale revenue but excludes budget-constrained segments. The optimal configuration is therefore **not** trivially "pick the most expensive tier everywhere," which would be the case without a demand model. This nontriviality is essential for the benchmark to be meaningful.

**Implementation:** Revenue computation is in `src/problem_generator.py` (method `expected_revenue`).

### 2.4 Constraints and penalties

Three types of constraints are enforced via penalty terms:

**(A) Invalid-state penalty.** Any feature in the 11 (invalid) encoding is penalized:
$$
\text{Pen}_{\text{inv}}(x) = M_{\text{inv}} \sum_{i=1}^{N_f} \mathbf{1}[\text{Invalid}_i] = M_{\text{inv}} \sum_{i=1}^{N_f} a_i b_i.
$$

**(B) Bundle constraints.** These enforce compatibility rules between features. For example, "luxury audio requires at least premium seats" is encoded as:
$$
\text{viol}_1(x) = \mathbf{1}[\text{Luxury}_{\text{audio}}] \cdot (1 - \text{AP}_{\text{seats}}),
$$
where $\text{AP}_{\text{seats}} = a_{\text{seats}} \oplus b_{\text{seats}}$ as defined in (2.2). The penalty is $M_c \cdot \text{viol}_1(x)$.

**(C) Luxury cap.** At most $L_{\max}$ features may be at the luxury tier simultaneously:
$$
\text{Pen}_{\text{lux}}(x) = M_{\text{lux}} \cdot \max\!\Big(0,\;\sum_{i=1}^{N_f} \mathbf{1}[\text{Luxury}_i] - L_{\max}\Big).
$$

### 2.5 True objective

Combining revenue and penalties, the true objective function that we seek to maximize is
$$
F(x) = R(x) - \text{Pen}_{\text{inv}}(x) - \sum_c M_c \cdot \text{viol}_c(x) - \text{Pen}_{\text{lux}}(x).
$$

**Penalty magnitudes.** Let $R_{\max} = \max_x R(x)$ be the maximum achievable revenue (computed by brute force). We set $M_{\text{inv}} = 10 R_{\max}$ and $M_c = M_{\text{lux}} = 5 R_{\max}$, ensuring that any constraint violation dominates the revenue term and is never optimal.

**Ground truth.** For $N_f = 5$ features ($n = 10$ bits), the configuration space has $2^{10} = 1024$ elements. Complete enumeration yields the exact optimum $F^* = \max_x F(x)$ and the optimal configuration $x^*$, which serves as ground truth for all benchmarks.

**Implementation:** The full objective $F(x)$, brute-force solver, and truth-table construction are in `src/problem_generator.py` (methods `evaluate`, `build_truth_table`, `brute_force_solve`). Frozen instances at $N_f \in \{3, 4, 5\}$ are stored in `data/instances/`.

---

## 3. Walsh–Hadamard Surrogate Construction

### 3.1 Exact Walsh–Hadamard transform of the true objective

The pricing objective $F:\{0,1\}^n \to \mathbb{R}$ from (2.10) is a general pseudo-Boolean function containing threshold indicators and max-type penalty terms. It is therefore not expected to be sparse in the Walsh basis. Because the DQI construction in §4 requires a sparse parity-style objective, we replace $F$ by a Fourier-truncated surrogate.

We compute the Walsh–Hadamard coefficients of $F$ exactly from the full truth table. Let $\mathbf{f}\in\mathbb{R}^{2^n}$ denote the vector with entries $\mathbf{f}_x = F(x)$, indexed by $x\in\{0,1\}^n$. Define the single-qubit Hadamard matrix
$$
H = \frac{1}{\sqrt{2}}
\begin{pmatrix}
1 & 1 \\
1 & -1
\end{pmatrix},
\qquad
H_n = H^{\otimes n}.
$$
Then the Walsh coefficient vector is
$$
\hat{\mathbf{f}} = 2^{-n/2} H_n \mathbf{f},
$$
which is equivalent to the coefficient formula
$$
\hat{F}(S) = \frac{1}{2^n}\sum_{x\in\{0,1\}^n} F(x)\,(-1)^{\langle S,x\rangle},
\qquad S\subseteq[n].
$$
With this normalization, Parseval's identity takes the form
$$
\sum_{S\subseteq[n]} \hat{F}(S)^2
=
\frac{1}{2^n}\sum_{x\in\{0,1\}^n} F(x)^2.
$$

Because $n\le 10$ in the frozen benchmark instances, the fast Walsh–Hadamard transform (FWHT) is exact and computationally trivial at our scale.

**Implementation and validation.** The transform is implemented in `src/reduction.py`. The associated validation in `tests/test_reduction.py` checks (i) agreement between the direct definition (3.2) and the FWHT implementation, (ii) inversion of the transform, and (iii) Parseval equality (3.3) on frozen instances.

### 3.2 Top-$K$ truncation and the weighted surrogate

We retain the $K$ nonconstant Walsh coefficients of largest magnitude:
$$
\mathcal{S}_K
=
\underset{S\subseteq[n],\,S\neq\emptyset}{\operatorname{arg\,top\text{-}K}}
\,|\hat{F}(S)|.
$$
The most faithful parity-style approximation to $F$ supported on this selected set is the **weighted surrogate**
$$
G_{\mathrm{wt}}(x)
=
\sum_{S\in\mathcal{S}_K}\hat{F}(S)\,(-1)^{\langle S,x\rangle}.
$$
If $K$ includes all nonconstant Walsh coefficients, then
$$
G_{\mathrm{wt}}(x)=F(x)-\hat{F}(\emptyset),
$$
so the only omitted term is the constant offset, which does not affect the optimizer.

### 3.3 Encoding the weighted surrogate as a max-XORSAT-style clause set

For each selected subset $S_i\in\mathcal{S}_K$, define a row vector $b_i\in\mathbb{F}_2^n$ by
$$
(b_i)_j =
\begin{cases}
1, & j\in S_i,\\
0, & j\notin S_i,
\end{cases}
$$
and define the sign bit
$$
v_i =
\begin{cases}
0, & \hat{F}(S_i)\ge 0,\\
1, & \hat{F}(S_i)<0.
\end{cases}
$$
Let
$$
c_i = |\hat{F}(S_i)|.
$$
Then the weighted surrogate can be written as
$$
G_{\mathrm{wt}}(x)
=
\sum_{i=1}^{K} c_i\,(-1)^{v_i + b_i\cdot x}.
$$
This is already in a parity format closely related to max-XORSAT, except for the clause weights $c_i$.

### 3.4 Unweighted DQI input and the second approximation layer

The DQI construction used in §4 operates on an **unweighted** max-XORSAT objective. Accordingly, the circuit is applied not to $G_{\mathrm{wt}}$ but to the sign-only objective
$$
G_{\mathrm{unw}}(x)
=
\sum_{i=1}^{K} (-1)^{v_i + b_i\cdot x}.
$$
The passage from $G_{\mathrm{wt}}$ to $G_{\mathrm{unw}}$ is a second approximation layer in addition to Fourier truncation itself. In particular, good agreement between $F$ and $G_{\mathrm{wt}}$ does not by itself imply good agreement between $F$ and the objective actually optimized by the DQI circuit.

For this reason, the benchmark tracks two distinct notions of surrogate faithfulness:
$$
\rho_{\mathrm{wt}} = \rho_S(F, G_{\mathrm{wt}}),
\qquad
\rho_{\mathrm{unw}} = \rho_S(F, G_{\mathrm{unw}}),
$$
where $\rho_S$ denotes Spearman rank correlation over all $2^n$ configurations. We also report the fraction of nonconstant Parseval energy retained by the truncation,
$$
\eta_K
=
\frac{\sum_{S\in\mathcal{S}_K}\hat{F}(S)^2}
{\sum_{S\neq\emptyset}\hat{F}(S)^2},
$$
and the optimization loss induced by the unweighted surrogate,
$$
\Delta_K^\star
=
F^\star
-
\max_{x\in\arg\max G_{\mathrm{unw}}} F(x),
\qquad
F^\star = \max_x F(x).
$$

These diagnostics distinguish three separate questions: how well the top-$K$ Fourier support represents $F$, how much information is lost when coefficient magnitudes are discarded, and whether the optimizer of the DQI-ready objective remains valuable when evaluated on the true pricing function.

**Implementation.** 
The FWHT, top-$K$ selection, construction of $(B,v)$, storage of coefficient magnitudes $\{c_i\}$, and diagnostics (3.12)–(3.14) are implemented in `src/reduction.py`.

---

## 4. The DQI Algorithm

This section specializes the max-XORSAT DQI construction of [1] to the surrogate instance $(B,v)$ produced in §3. The circuit core follows the DQI construction for max-XORSAT; the project-specific ingredient is the surrogate reduction that maps the true pricing objective $F$ to the unweighted parity objective $G_{\mathrm{unw}}$.

### 4.1 The DQI state

Given the max-XORSAT instance $(B,v)$ with objective
$$
f(x) = \sum_{i=1}^{m} (-1)^{v_i + b_i\cdot x},
$$
DQI aims to prepare a state of the form
$$
|P(f)\rangle
=
\sum_{x\in\{0,1\}^n} P(f(x))\,|x\rangle,
$$
where $P$ is a real polynomial of degree $\ell$. Measuring $|P(f)\rangle$ in the computational basis yields samples with probability proportional to $P(f(x))^2$, so the choice of $P$ determines how strongly the distribution is biased toward large values of $f(x)$.

For the linear choice $P(t)=t$ (degree $\ell=1$), one obtains the state
$$
|f\rangle = \sum_{x\in\{0,1\}^n} f(x)\,|x\rangle,
$$
which can be prepared directly because the Hadamard transform of $f$ is supported on the clause rows $b_1,\dots,b_m$. Higher-degree polynomials sharpen the sampling bias but require decoding higher-weight error patterns.

### 4.2 Fourier structure of the DQI state

For max-XORSAT, the Hadamard transform of the DQI state has the form
$$
H^{\otimes n}|P(f)\rangle
=
\sum_{k=0}^{\ell}
\frac{w_k}{\sqrt{\binom{m}{k}}}
\sum_{\substack{y\in\{0,1\}^m\\|y|=k}}
(-1)^{v\cdot y}\,
|B^T y\rangle,
$$
for some real coefficient vector $w=(w_0,\dots,w_\ell)$. Here $|y|$ denotes Hamming weight. The natural basis states underlying this expression are the Dicke states
$$
|D_k^m\rangle
=
\frac{1}{\sqrt{\binom{m}{k}}}
\sum_{\substack{y\in\{0,1\}^m\\|y|=k}}
|y\rangle.
$$
Thus the DQI preparation problem reduces to constructing an appropriate superposition of Dicke states, imprinting the phase $(-1)^{v\cdot y}$, computing the syndrome $B^T y$, and then uncomputing the error pattern $y$ by bounded-distance decoding.

In this document, the coefficients $w_k$ are the amplitudes of the Dicke-state superposition itself. They are not square roots of those amplitudes.

### 4.3 The five-step DQI circuit

The conceptual registers are an $m$-qubit **error register** storing $y$ and an $n$-qubit **syndrome register** initially prepared in $|0\rangle^{\otimes n}$. An implementation may temporarily use additional ancillas, including a weight register, during Dicke-state preparation, but the mathematical description below does not require such a register to remain explicit.

**Step 1: Prepare the Dicke-state superposition.** Prepare
$$
|\Psi_1\rangle
=
\sum_{k=0}^{\ell} w_k\,|D_k^m\rangle
=
\sum_{k=0}^{\ell}
\frac{w_k}{\sqrt{\binom{m}{k}}}
\sum_{\substack{y\in\{0,1\}^m\\|y|=k}}
|y\rangle.
$$
This is the state whose amplitudes encode the degree-$\ell$ polynomial choice.

**Step 2: Impose the phase determined by $v$.** Apply $Z_i$ to error qubit $i$ whenever $v_i=1$. The state becomes
$$
|\Psi_2\rangle
=
\sum_{k=0}^{\ell}
\frac{w_k}{\sqrt{\binom{m}{k}}}
\sum_{\substack{y\in\{0,1\}^m\\|y|=k}}
(-1)^{v\cdot y}\,|y\rangle.
$$

**Step 3: Compute the syndrome.** Reversibly compute $B^T y$ into the syndrome register:
$$
|\Psi_3\rangle
=
\sum_{k=0}^{\ell}
\frac{w_k}{\sqrt{\binom{m}{k}}}
\sum_{\substack{y\in\{0,1\}^m\\|y|=k}}
(-1)^{v\cdot y}\,
|y\rangle\,|B^T y\rangle.
$$
Since $B\in\mathbb{F}_2^{m\times n}$, the $j$-th syndrome bit is
$$
(B^T y)_j = \bigoplus_{i=1}^{m} B_{ij} y_i.
$$
At the benchmark scale, this is implemented by a sparse CNOT network.

**Step 4: Decode and uncompute the error pattern.** This step is the defining operation of DQI. Given the syndrome $B^T y$ and the promise $|y|\le \ell$, one attempts to recover $y$ and reversibly map $|y\rangle$ back to $|0\rangle^{\otimes m}$.

Define the dual code
$$
C^\perp = \{d\in\mathbb{F}_2^m : B^T d = 0\},
$$
with minimum distance $d^\perp$.

**Exact-decoding regime.** The exact DQI formulas used below assume
$$
2\ell + 1 < d^\perp.
$$
Under this condition, distinct error patterns of Hamming weight at most $\ell$ cannot share the same syndrome, so bounded-distance decoding is well-defined on the support of the DQI state. In that regime,
$$
|\Psi_4\rangle
=
\sum_{k=0}^{\ell}
\frac{w_k}{\sqrt{\binom{m}{k}}}
\sum_{\substack{y\in\{0,1\}^m\\|y|=k}}
(-1)^{v\cdot y}\,
|0\rangle^{\otimes m}\,|B^T y\rangle.
$$

At our small scale, the decoder is implemented by brute-force bounded-distance lookup: enumerate all error patterns with $|y|\le \ell$, compute their syndromes, and verify injectivity on that support. Runs are reported in one of two modes:

- **exact mode:** the injectivity check passes, so (4.11) holds for the realized support;
- **approximate mode:** syndrome collisions occur within the radius-$\ell$ set, so the circuit is treated as an approximation to ideal DQI rather than an exact realization.

**Step 5: Hadamard transform and sampling.** Apply $H^{\otimes n}$ to the syndrome register. The resulting state is proportional to
$$
|\Psi_5\rangle
\propto
\sum_{x\in\{0,1\}^n} P(f(x))\,|x\rangle
=
|P(f)\rangle.
$$
Measurement in the computational basis therefore yields a sample distributed according to $P(f(x))^2$.

**Implementation.** 
The quantum state preparation is implemented in `src/dqi_state.py`, the bounded-distance decoder in `src/decoder_bruteforce.py`, and the syndrome-injectivity check in `tests/test_decoder.py`. If the implementation uses a temporary weight register during Dicke-state preparation, that register is treated as a preparation ancilla and is uncomputed before the final Hadamard step.

---

## 5. Weight Selection and Finite-Size Optimization

### 5.1 Paper-faithful finite-size optimization in the exact-decoding regime

The degree-$\ell$ DQI state is determined by the coefficient vector
$$
w = (w_0,\dots,w_\ell)^T,
\qquad
\|w\|_2 = 1.
$$
For max-XORSAT in the exact-decoding regime (4.11), the expected number of satisfied constraints is
$$
\langle s\rangle
=
\frac{m}{2}
+
\frac{1}{2}\,
w^\dagger A^{(m,\ell,0)} w,
$$
where $A^{(m,\ell,0)}$ is the $(\ell+1)\times(\ell+1)$ symmetric tridiagonal matrix
$$
A^{(m,\ell,0)}
=
\begin{pmatrix}
0 & a_1 & 0 & \cdots & 0 \\
a_1 & 0 & a_2 & \ddots & \vdots \\
0 & a_2 & 0 & \ddots & 0 \\
\vdots & \ddots & \ddots & 0 & a_\ell \\
0 & \cdots & 0 & a_\ell & 0
\end{pmatrix},
$$
with
$$
a_k = \sqrt{k(m-k+1)},
\qquad k=1,\dots,\ell.
$$
Equivalently, $A^{(m,\ell,0)}$ is the specialization to $p=2$, $r=1$, and hence $d=0$, of the paper's general matrix $A^{(m,\ell,d)}$.

The finite-size weight-selection problem is therefore the Rayleigh-quotient maximization
$$
\max_{\|w\|_2=1} w^\dagger A^{(m,\ell,0)} w.
$$
Its optimizer is the principal eigenvector of $A^{(m,\ell,0)}$. In other words, in the exact-decoding regime the paper-faithful choice of DQI weights is
$$
w^\star = \operatorname{eigvec}_{\max}\!\big(A^{(m,\ell,0)}\big).
$$

### 5.2 What changes outside the exact-decoding regime

When
$$
2\ell + 1 \ge d^\perp,
$$
the exact formulas underlying (5.2) no longer hold for a realized finite instance, because distinct weight-$\le \ell$ error patterns may share the same syndrome. In that regime, the principal eigenvector of $A^{(m,\ell,0)}$ is retained only as a heuristic weight choice rather than as the exact optimizer guaranteed by the finite-size theory.

Accordingly, benchmark runs report whether they are in exact mode or approximate mode, and all claims of paper-faithful finite-size optimality are restricted to exact-mode instances.

### 5.3 Asymptotic semicircle-law benchmark

For large $m$ with $\mu=\ell/m$ fixed, the optimal satisfied-fraction for balanced max-XORSAT obeys the asymptotic law
$$
\lim_{\substack{m,\ell\to\infty\\ \ell/m=\mu}}
\frac{\langle s\rangle_{\mathrm{opt}}}{m}
=
\begin{cases}
\frac{1}{2} + \sqrt{\mu(1-\mu)}, & 0\le \mu \le \frac{1}{2},\\[4pt]
1, & \frac{1}{2} < \mu \le 1.
\end{cases}
$$
The first branch is the semicircle expression typically quoted for balanced max-XORSAT. The second branch records the saturation that occurs once the polynomial degree is large enough relative to the number of clauses.

Because our benchmark uses very small instances, (5.8) is not used to choose weights. Instead, it is reported only as an asymptotic diagnostic reference against which the finite-size eigenvector result can be compared.

### 5.4 Ablation: uniform versus eigenvector weights

To quantify the role of weight optimization, we compare three choices:

1. **Uniform weights**
$$
w_k^{\mathrm{unif}} = \frac{1}{\sqrt{\ell+1}},
\qquad k=0,\dots,\ell.
$$

2. **Paper-faithful finite-size weights**
$$
w^\star = \operatorname{eigvec}_{\max}\!\big(A^{(m,\ell,0)}\big).
$$

3. **Asymptotic benchmark value** from (5.8), used only as a reference curve.

This ablation separates three effects: the gain from nonuniform degree allocation, the finite-size correction relative to the asymptotic law, and the degradation that occurs when the instance leaves the exact-decoding regime.

**Implementation and validation.** The tridiagonal matrix construction and principal-eigenvector computation are implemented in `src/weights.py`. Validation checks confirm (i) symmetry of $A^{(m,\ell,0)}$, (ii) agreement between the numerical Rayleigh optimum and the principal eigenpair, and (iii) consistency between reported weight choices and the exact-mode versus approximate-mode classification from the decoder tests.

### 5.5 Optional instance-specific diagnostic matrix

In addition to the paper-faithful tridiagonal optimization of §§5.1–5.4, we optionally compute an instance-specific enumerated diagnostic matrix obtained by summing over bounded-weight error pairs on the realized clause set $(B,v)$. This object is not the matrix $A^{(m,\ell,d)}$ from [1]; rather, it is a project-specific diagnostic used to study finite-instance deviations from the idealized exact-regime theory.

---
## 6. Classical Baselines

To contextualize DQI performance, we benchmark against three classical solvers operating on the **true** pricing objective $F(x)$:

### 6.1 Brute-force enumeration

For $n \le 12$ bits, exhaustive search over all $2^n$ configurations is trivial and provides exact ground truth: $F^* = \max_x F(x)$ and $x^* = \arg\max_x F(x)$. All reported approximation ratios use $F^*$ as the denominator.

### 6.2 Constraint programming (CP-SAT)

Google OR-Tools CP-SAT is a state-of-the-art constraint-programming solver that handles integer variables, linear constraints, and general objective functions. We encode the pricing problem directly (without the parity surrogate) and solve to optimality. At our scale, CP-SAT finds the exact optimum in milliseconds.

### 6.3 Simulated annealing

Simulated annealing (SA) is the canonical classical heuristic comparison for DQI, as established in [1] Table 2. We implement SA with random single-bit flips, geometric cooling schedule $T_{k+1} = \gamma T_k$ with $\gamma = 0.995$, and Metropolis acceptance. SA operates on the **true** $F(x)$, not the surrogate $G(x)$.

**Implementation:** All classical baselines are in `src/classical_baselines.py`.

---

## 7. Validation Protocol and Reporting

This section specifies the validation logic for the full benchmark pipeline. Because the project combines a project-specific surrogate reduction (§3) with a paper-based DQI circuit core (§§4–5), correctness must be established in layers rather than inferred from final sample quality alone.

### 7.1 Reduction-layer validation: $F$ versus $G_{\mathrm{wt}}$ versus $G_{\mathrm{unw}}$

The first validation layer concerns the surrogate construction. The weighted surrogate
$$
G_{\mathrm{wt}}(x)
=
\sum_{i=1}^{m} c_i\,(-1)^{v_i+b_i\cdot x}
$$
tests how well top-$K$ Walsh truncation preserves the true pricing objective $F$, while the unweighted surrogate
$$
G_{\mathrm{unw}}(x)
=
\sum_{i=1}^{m} (-1)^{v_i+b_i\cdot x}
$$
is the objective actually passed to the DQI circuit. These are distinct objects and must be evaluated separately.

For each frozen instance, we compute:
$$
\rho_{\mathrm{wt}} = \rho_S(F, G_{\mathrm{wt}}),
\qquad
\rho_{\mathrm{unw}} = \rho_S(F, G_{\mathrm{unw}}),
$$
where $\rho_S$ is Spearman rank correlation over all $2^n$ configurations, together with the retained nonconstant Parseval energy
$$
\eta_K
=
\frac{\sum_{S\in\mathcal{S}_K}\hat{F}(S)^2}
{\sum_{S\neq\emptyset}\hat{F}(S)^2},
$$
and the optimizer mismatch
$$
\Delta_K^\star
=
F^\star
-
\max_{x\in\arg\max G_{\mathrm{unw}}} F(x),
\qquad
F^\star = \max_x F(x).
$$

The validation logic is as follows.

1. **FWHT correctness.** Check agreement between the direct Walsh definition and the FWHT implementation, together with inversion and Parseval equality.
2. **Top-$K$ faithfulness.** Use $\rho_{\mathrm{wt}}$ and $\eta_K$ to measure information loss due to Fourier truncation alone.
3. **DQI-input faithfulness.** Use $\rho_{\mathrm{unw}}$ and $\Delta_K^\star$ to measure the additional loss incurred when coefficient magnitudes are discarded before the circuit.
4. **Full-support regression test.** When $K$ includes all nonconstant Walsh coefficients, verify that
   $$
   G_{\mathrm{wt}}(x)=F(x)-\hat{F}(\emptyset)
   $$
   for all $x$.

This separation is essential: high $\rho_{\mathrm{wt}}$ does not imply high $\rho_{\mathrm{unw}}$, and only the latter directly certifies the quality of the optimization objective seen by the DQI circuit.

**Implementation.** These checks are implemented in `src/reduction.py` and validated in `tests/test_reduction.py`.

### 7.2 Decoder validation: exact mode versus approximate mode

The second validation layer concerns the DQI decoding step. Section 4 established that exact uncomputation of the error register requires injectivity of the syndrome map on the radius-$\ell$ Hamming ball. A sufficient condition is
$$
2\ell + 1 < d^\perp,
$$
where $d^\perp$ is the minimum distance of the dual code
$$
C^\perp=\{d\in\mathbb{F}_2^m : B^T d = 0\}.
$$
In the DQI paper, the exact finite-size formulas and the associated optimality claim for the principal eigenvector are derived in this exact-decoding regime; once $2\ell+1\ge d^\perp$, the same eigenvector may remain useful, but it is no longer guaranteed to be exactly optimal for the realized finite instance.

At benchmark scale, we do not rely only on the distance bound. Instead, we directly enumerate the support
$$
E_\ell = \{y\in\{0,1\}^m : |y|\le \ell\}
$$
and test whether the syndrome map
$$
y \mapsto B^T y
$$
is injective on $E_\ell$.

Each run is labeled as one of two regimes:

- **exact mode:** the injectivity test passes on $E_\ell$;
- **approximate mode:** at least one collision $B^T y = B^T y'$ with $y\neq y'$ and $|y|,|y'|\le \ell$ is found.

For every reported experiment, we record:

1. the regime label (exact or approximate);
2. the number of detected syndrome collisions within $E_\ell$;
3. the decoder lookup-table size $\sum_{t=0}^{\ell}\binom{m}{t}$;
4. the fraction of sampled branches for which the decoder returns a valid bounded-weight preimage.

All claims of exact DQI state preparation, exact finite-size weight optimality, or paper-faithful agreement with the finite-size theory are restricted to **exact-mode** instances. Approximate-mode runs remain useful benchmark data, but they are reported explicitly as approximations to the ideal DQI state rather than as exact realizations.

**Implementation.** Brute-force bounded-distance decoding is implemented in `src/decoder_bruteforce.py`. The regime-classification and collision checks are validated in `tests/test_decoder.py`.

### 7.3 Circuit-level validation on tiny frozen instances

Before using DQI as an optimizer, we validate the circuit itself on the smallest frozen instances for which explicit classical state construction is feasible.

The required checks are:

1. **Statevector agreement.** Construct the target amplitude vector for $|P(f)\rangle$ classically and compare it to the simulated circuit statevector up to global phase.
2. **$\ell=1$ regression test.** For $P(t)=t$, verify that the output probabilities satisfy
   $$
   p(x)\propto f(x)^2,
   $$
   which is the basic DQI sampling law for the linear polynomial case.
3. **Mode-aware decoder test.** On exact-mode instances, verify agreement between the implemented decoder-based circuit and the ideal target state. On approximate-mode instances, report the deviation rather than treating the result as a correctness failure.
4. **Cross-framework check.** Compare the PennyLane and Qiskit statevector outputs on the same exact-mode toy instances.

These tests ensure that observed benchmarking behavior is not merely an artifact of a coding error in state preparation.

**Implementation.** State preparation is in `src/dqi_state.py`, cross-framework verification is in `src/dqi_circuit_qiskit.py`, and the corresponding tests are in `tests/test_stateprep.py` and `tests/test_qiskit_port.py` (planned if not yet present).

### 7.4 Weight-selection validation

The third validation layer concerns the degree-allocation weights $w=(w_0,\dots,w_\ell)^T$. In the exact-decoding regime, the paper's finite-size max-XORSAT theory gives
$$
\langle s\rangle
=
\frac{m}{2}
+
\frac{1}{2}\,w^\dagger A^{(m,\ell,0)} w,
$$
so the paper-faithful finite-size weight choice is the normalized principal eigenvector of the symmetric tridiagonal matrix $A^{(m,\ell,0)}$.

For each instance and each degree cutoff $\ell$, we therefore validate:

1. **Matrix construction.** Confirm that the implemented matrix is symmetric and matches the closed-form tridiagonal specification from §5.
2. **Eigenpair correctness.** If
   $$
   A^{(m,\ell,0)} w^\star = \lambda_{\max} w^\star,
   $$
   verify numerically that $\|A^{(m,\ell,0)} w^\star - \lambda_{\max} w^\star\|_2$ is below tolerance.
3. **Rayleigh optimality.** Confirm that
   $$
   (w^\star)^\dagger A^{(m,\ell,0)} w^\star
   \ge
   w^\dagger A^{(m,\ell,0)} w
   $$
   for comparison vectors including the uniform choice
   $$
   w_k^{\mathrm{unif}}=\frac{1}{\sqrt{\ell+1}}.
   $$
4. **Mode-aware interpretation.** In exact mode, report $w^\star$ as the paper-faithful finite-size optimum. In approximate mode, report the same vector only as a heuristic weight choice motivated by the exact-regime theory.

The benchmark then compares DQI performance under three weight policies:

- uniform weights;
- principal-eigenvector weights;
- asymptotic-reference satisfaction fraction from §5.3, used only as a diagnostic comparison and not as the operative weight rule.

**Implementation.** Matrix construction and eigensolver logic are in `src/weights.py`. Validation is in `tests/test_weights.py`.

### 7.5 End-to-end DQI evaluation on the true objective

After the reduction, decoder, and weight layers have each been validated independently, we evaluate end-to-end DQI sampling on the true pricing objective $F$.

For each frozen instance, choice of $K$, choice of $\ell$, and choice of weight policy, the protocol is:

1. build $(B,v)$ and the metadata $(c_i)$ from §3;
2. classify the instance as exact mode or approximate mode using §7.2;
3. construct the DQI state using the selected weight policy;
4. sample $N_{\text{shots}}$ bitstrings $\{x_1,\dots,x_{N_{\text{shots}}}\}$;
5. score every sample under both $G_{\mathrm{unw}}$ and the true objective $F$.

We report:

- best true score found,
  $$
  F_{\mathrm{best}}=\max_i F(x_i);
  $$
- approximation ratio,
  $$
  \mathrm{AR}=\frac{F_{\mathrm{best}}}{F^\star};
  $$
- top-$k$ expected true score among samples ranked by $G_{\mathrm{unw}}$;
- surrogate-to-truth transfer, measured by the joint distribution of $\big(G_{\mathrm{unw}}(x_i),F(x_i)\big)$;
- all metrics stratified by exact mode versus approximate mode.

This final stratification is important. A poor end-to-end result can arise from at least three different causes: weak Fourier truncation, loss of magnitude information when passing from $G_{\mathrm{wt}}$ to $G_{\mathrm{unw}}$, or decoder noninjectivity. Reporting all runs together obscures that distinction.

**Implementation.** End-to-end sampling and scoring are in `src/dqi_pipeline.py`, and benchmark orchestration is in `src/benchmark.py`.

### 7.6 Transparency and scope of claims

The benchmark is intentionally small scale. For $n\le 10$ and modest clause counts, classical exact optimization is easy, so the purpose of the study is not to claim quantum advantage. Rather, the purpose is to demonstrate a mathematically checked benchmark pipeline in which:

1. the true pricing objective is exactly enumerable;
2. the surrogate reduction is explicitly diagnosed at both the weighted and unweighted levels;
3. the DQI circuit core is validated against the exact-decoding assumptions of the paper when those assumptions hold;
4. deviations from those assumptions are reported rather than concealed.

The principal practical bottleneck remains the clause count $m$, because the error register scales with $m$ and the bounded-distance support size grows as $\sum_{t=0}^{\ell}\binom{m}{t}$. This is why the benchmark caps $K=m$ at small values and treats larger-scale decoding only as a future path involving structured decoders such as BP or min-sum.

---

## 8. Structural Qiskit Subset Validation and Optional Noise Study

### 8.1 Structural subset verification

As a framework-independence check, we port the DQI quantum circuit (Steps 1–5 of §4.3) to Qiskit. This port is intentionally scoped as a structural subset validation: circuit construction and selected statevector simulation are ported, while the problem generator, reduction, decoder reporting, and full benchmarking infrastructure remain in the PennyLane/numpy stack.

The verification consists of running the same frozen **exact-mode** instance through both PennyLane and Qiskit statevector backends and confirming agreement of intermediate and final statevectors up to numerical precision for the implemented subset. This is a structural equivalence check, not a claim of decode-inclusive end-to-end parity for the full benchmark.

$$
d_{\text{TV}}(p, q) = \frac{1}{2} \sum_{x} |p(x) - q(x)|.
$$

### 8.2 Optional noise study

Using Qiskit Aer's depolarizing noise model at error rates $\epsilon \in \{10^{-3}, 10^{-2}, 10^{-1}\}$, we evaluate how noise degrades DQI performance. The metric is the expected true objective $\langle F \rangle_{k}$ of the top-$k$ samples, compared to the noiseless baseline. This provides a basic assessment of near-term hardware prospects.

**Implementation:** The Qiskit circuit port is in `src/dqi_circuit_qiskit.py`.

---

## Summary of Deliverables, Validation Layers, and Code Mapping

The repository implements a layered benchmark rather than a single monolithic pipeline. The core deliverables are: (i) exact construction of the true pricing objective $F$ and its ground truth optimum; (ii) Walsh-based reduction of $F$ into weighted and unweighted parity surrogates; (iii) a DQI circuit core for the unweighted max-XORSAT instance; (iv) mode-aware bounded-distance decoding that classifies runs as exact or approximate; and (v) finite-size weight selection via the principal eigenvector of the paper’s tridiagonal matrix in exact mode, with the same vector retained only heuristically in approximate mode.

| Theoretical component | Section | Implementation |
|---|---|---|
| Pricing model + 2-bit encoding + brute force ground truth | §2 | `src/problem_generator.py` |
| Walsh–Hadamard transform + top-$K$ truncation + weighted/unweighted surrogate construction | §3 | `src/reduction.py` |
| DQI five-step state preparation for the unweighted max-XORSAT surrogate | §4.3 | `src/dqi_state.py` |
| Bounded-distance decoder + exact/approximate mode classification | §4.3, §7.2 | `src/decoder_bruteforce.py` |
| Finite-size principal-eigenvector weight selection (exact-mode theory) | §5, §7.4 | `src/weights.py` |
| End-to-end DQI sampling on $G_{\mathrm{unw}}$ with evaluation on true $F$ | §7.5 | `src/dqi_pipeline.py` |
| Classical baselines on the true objective $F$ | §6 | `src/classical_baselines.py` |
| Structural Qiskit subset validation and cross-framework verification | §8 | `src/dqi_circuit_qiskit.py` |
| Benchmark orchestration + regime-stratified plots | §7 | `src/benchmark.py` |
