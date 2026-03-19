# ILP-to-XORSAT Reduction Methodology

**How the toy-pricing ILP is transformed into a max-XORSAT instance for DQI evaluation**

---

## 1. Reduction Pipeline

The module `src/ilp_to_xorsat_exact.py` transforms the discrete pricing ILP into a max-XORSAT parity system $(B, v)$ suitable for DQI. The pipeline has three steps:

### Step 1: Exhaustive objective evaluation

For a pricing instance with $N_f$ features encoded in $n = 2N_f$ bits, the true pricing objective $F(x)$ is evaluated over the full configuration space $\{0,1\}^n$:

$$F(x) = R(x) - M_{\text{inv}} \cdot n_{\text{invalid}}(x) - M_c \cdot n_{\text{bundle}}(x) - M_{\text{lux}} \cdot \max(0, n_{\text{luxury}}(x) - L_{\max})$$

where $R(x)$ is the expected revenue across customer segments and the penalty magnitudes satisfy $M_{\text{inv}}, M_c, M_{\text{lux}} \gg \max_x R(x)$. This produces the truth table $\mathbf{f} \in \mathbb{R}^{2^n}$.

### Step 2: Walsh-Hadamard transform

The complete Walsh-Hadamard transform decomposes $F$ into parity basis functions:

$$\hat{F}(S) = \frac{1}{2^n} \sum_{x \in \{0,1\}^n} F(x) \cdot (-1)^{\langle S, x \rangle}, \quad S \subseteq [n]$$

implemented via the fast Walsh-Hadamard transform in $O(n \cdot 2^n)$ time. This yields $2^n - 1$ nonconstant Fourier coefficients (the constant term $\hat{F}(\emptyset)$ is discarded as it does not affect the optimizer).

### Step 3: Max-XORSAT encoding

Each nonconstant Fourier index $S_i$ defines a parity constraint row $b_i \in \mathbb{F}_2^n$ (the binary representation of $S_i$) and a sign bit:

$$v_i = \begin{cases} 0, & \hat{F}(S_i) \geq 0 \\ 1, & \hat{F}(S_i) < 0 \end{cases}$$

with weight $c_i = |\hat{F}(S_i)|$. The resulting weighted max-XORSAT objective is:

$$G(x) = \hat{F}(\emptyset) + \sum_{i=1}^{m} c_i \cdot (-1)^{v_i + b_i \cdot x}$$

which reconstructs $F(x)$ exactly when all $m = 2^n - 1$ terms are retained.

---

## 2. Relationship to Sabater et al. (arXiv:2509.08328)

The BMW-BCG DQI paper (Sabater et al., Section 5) describes a pseudo-Boolean constraint encoding methodology for industrial pricing. The key relationship to our pipeline:

**Shared structure:**
- Both approaches encode a discrete pricing objective as a pseudo-Boolean function over binary variables, then decompose it into a parity (Walsh-Hadamard) basis to produce a max-XORSAT instance.
- Both use 2-bit paired encoding per feature with 3 valid tiers (standard/premium/luxury) and 1 invalid state, giving $n = 2N_f$ binary variables.
- The resulting $(B, v)$ parity system is passed to DQI for quantum optimization.

**Key differences:**

| Aspect | This project | Sabater et al. |
|--------|-------------|----------------|
| Scale | $N_f \leq 7$ (14 bits), exact enumeration feasible | $N_f$ up to industrial scale, exact enumeration infeasible |
| WHT computation | Exact via brute-force truth table + FWHT | Must use structured decomposition or sampling |
| Truncation | Full spectrum retained for the exact-reference path (all $2^n - 1$ terms); optional top-$k$ truncation for practical DQI runs | Truncation to top-$K$ coefficients is essential at scale |
| Constraint handling | Penalties folded into $F(x)$ before WHT, producing penalty-contaminated Fourier coefficients | May use separate constraint encoding or penalty structuring |
| Validation | Full-domain exact reconstruction verified to numerical precision | Validated against industrial KPIs rather than pointwise reconstruction |
| Decoder scope | Brute-force, BP1, BP-OSD decoders studied | Focus on hardware-compatible decoders |

**What our approach preserves from Sabater et al.:** The fundamental pipeline structure -- pricing objective $\to$ pseudo-Boolean function $\to$ Walsh-Hadamard decomposition $\to$ max-XORSAT $(B, v)$ $\to$ DQI circuit -- is identical. Our contribution is providing an exactly verifiable small-scale benchmark where every step can be validated against ground truth, which complements the industrial-scale deployment in the BMW paper.

---

## 3. Parity-Check Matrix Structure for P3-P5

The ILP-exact encoding uses the full Fourier spectrum, producing $m = 2^n - 1$ parity constraints. The resulting matrices have a characteristic structure:

| Instance | $N_f$ | $n$ | $m$ | Code rate $R = n/m$ | Column weight | Density | GF(2) rank | Full rank? |
|----------|-------|-----|-----|---------------------|--------------|---------|------------|------------|
| P3 | 3 | 6 | 63 | 0.095 | 32 | 50.8% | 6 | Yes |
| P4 | 4 | 8 | 255 | 0.031 | 128 | 50.2% | 8 | Yes |
| P5 | 5 | 10 | 1023 | 0.010 | 512 | 50.0% | 10 | Yes |

**Structural observations:**

1. **Row count:** $m = 2^n - 1$ because the full-spectrum encoding retains every nonzero binary vector as a parity constraint row. This is the complete Walsh-Hadamard basis minus the constant.

2. **Column uniformity:** Every column has weight exactly $2^{n-1}$, meaning each bit participates in exactly half of all parity constraints. This is a structural consequence of the full-spectrum encoding.

3. **Row weight distribution:** The number of rows with Hamming weight $k$ equals $\binom{n}{k}$ for $k = 1, \ldots, n$. This is the binomial distribution expected when enumerating all nonzero binary vectors.

4. **Full GF(2) rank:** The matrix $B$ always has full column rank over $\mathbb{F}_2$ (rank $= n$), because it contains the $n$ standard basis vectors $e_1, \ldots, e_n$ as rows.

5. **Code rate:** The code rate $R = n/m = n/(2^n - 1)$ decreases exponentially, reflecting the overcomplete nature of the full-spectrum encoding. This is why practical DQI runs use top-$k$ truncation to keep $m$ small.

**Top-$k$ truncated structure:** When truncated to $k \ll 2^n - 1$ terms (selecting the $k$ largest-magnitude Fourier coefficients), the matrix becomes much sparser and the structural uniformity breaks. The truncated matrix structure depends on the pricing instance's Fourier spectrum, which in turn depends on the demand model and constraint penalties.

---

## 4. Practical Implications for DQI

The full-spectrum encoding is exact but impractical for DQI at scale because:
- The error register requires $m$ qubits, and $m = 2^n - 1$ grows exponentially.
- The bounded-distance decoder must search over $\sum_{t=0}^{\ell} \binom{m}{t}$ error patterns.

For the benchmark, we use top-$k$ truncation (typically $k = 8$ or $k = 15$) for actual DQI runs, which keeps $m$ manageable while retaining most of the objective's Fourier energy. The exact full-spectrum encoding serves as a validation reference: the DQI circuit operates on the truncated $(B, v)$, but the result is always evaluated against the true $F(x)$.
