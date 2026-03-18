# DQI Notebook Suite Design

**Date:** 2026-03-10
**Status:** Approved in staged review

## Objective
Create four executed, portfolio-grade notebooks that demonstrate: canonical DQI reproducibility, pricing-model transparency, paper-faithful DQI verification on small instances, and Qiskit cross-framework evidence.

## Design Principles
- `src/*` owns all nontrivial computation (objective evaluation, feasibility logic, enumeration, surrogate construction, DQI metrics).
- `notebooks/_helpers.py` is notebook-facing glue only (loading, decoding labels, plotting, formatting).
- Notebooks are deterministic (explicit seeds in each notebook) and committed with rendered outputs.
- Figures intended for README are exported to stable file paths.

## Shared Architecture
### Core code ownership (`src/*`)
- Surrogate construction and Fourier reduction: `src/reduction.py`
- Pricing model/objective and feasibility computations: `src/problem_generator.py` (plus small additions if needed)
- DQI pipeline and five-step state path: `src/dqi_pipeline.py`, `src/dqi_state.py`
- Qiskit structural/statevector path: `src/dqi_circuit_qiskit.py`

### Notebook helper ownership (`notebooks/_helpers.py`)
- Robust instance loading via `Path`-safe repo-relative logic
- Human-readable decoding/rendering (`feature`, `tier label`, `price`)
- Plot helpers (histograms/overlays)
- Result-format helpers:
  - optimal configuration table formatter
  - feasibility summary formatter (compute lives in `src/*`)
  - verification card renderer from one metrics dict
  - TVD helper and resource-table helper (`depth_raw`, `depth_transpiled`, `two_qubit_gates`, settings)

## Notebook 01 — `notebooks/01_reproduce_toy_dqi.ipynb`
### Purpose
Reproduce a canonical toy DQI max-XORSAT demonstration and explain its mechanics.

### Required content
- Small toy instance (`n,m`) with exact classical optimum shown explicitly
- End-to-end DQI run and sampled distribution
- Top-k table with: bitstring, probability, objective/surrogate value, optimal flag
- One-knob study (prefer `ell` unless `m` is cleaner)
- Compact five-stage mapping markdown (Dicke, phase kick, syndrome, decode/uncompute, Hadamard/measure)
- Tutorial-scope note: this notebook is canonical reproduction, distinct from pricing-specific contributions

### Required artifact
- Export histogram figure to `figures/notebook01_toy_histogram.png`

## Notebook 02 — `notebooks/02_pricing_instance_demo.ipynb`
### Purpose
Make the discrete pricing model tangible and exactly verifiable.

### Required content
- Intro markdown defining feature/tier/feasibility/objective in plain terms
- Load frozen instances in order: 3-feature, 4-feature, 5-feature
- Per-instance sections:
  - compact instance summary (bits, feasible states, total states, optimum value)
  - brute-force optimum
  - immediate human-readable optimal configuration
  - feasibility stats with consistent fields:
    - valid fraction
    - invalid-tier count
    - bundle violations
    - luxury-cap violations
  - CP-SAT verification block with graceful skip if unavailable
- Consolidated narrow artifact table (only): `instance`, `feature`, `tier`, `price`

## Notebook 03 — `notebooks/03_paper_faithful_dqi.ipynb`
### Purpose
Provide a small-instance DQI microscope with exact anchoring, ablations, and misranking diagnostics.

### Required content
- One shared markdown block describing five steps once
- For 6/8/10-bit instances: build `(B,v)` via Fourier truncation (`top-K<=15`, `m<=20`)
- Per-instance exact anchor row:
  - true optimum `F*`
  - best sampled `F`
  - approximation ratio `best_sampled_F / F*`
- Run five-step DQI path and report:
  - decoding/postselection success probability
  - best sampled `G(x)`
  - best sampled `F(x)`
  - `F(top surrogate-ranked sampled x)`
  - sampled misranking gap: `best_sampled_F - F(top surrogate-ranked sampled x)`
  - Spearman `rho(F,G)` over full space (explicitly noted as exact because instances are tiny)
- Weights ablation labels:
  - `uniform_alpha`
  - `current_eigenvector_alpha`
- Verification cards include a failure signal field (`top_rank_regret` or `misranking_flag`)

## Notebook 04 — `notebooks/04_qiskit_comparison.ipynb`
### Purpose
Provide explicit Qiskit evidence with a fair cross-framework comparison on the smallest instance.

### Required content
- Hard Qiskit requirement (fail early if missing)
- 3-feature / 6-bit instance and fixed `(k, ell)`
- Explicit apples-to-apples mode: no-decoder comparison in both frameworks
- Compare the same measured output-register basis distribution after the same simplified sequence
- Statevector probabilities (not shot-based)
- Report:
  - TVD
  - `n_qubits`
  - `depth_raw`
  - `depth_transpiled`
  - `two_qubit_gates`
  - transpilation settings (`basis_gates`, `optimization_level`, backend/target assumption)
- Overlay histogram over top-support states for readability

### Scope statement (required)
Notebook 04 is a cross-framework sanity check on a simplified shared pipeline, not a full parity proof for the complete decoded DQI workflow.

## Execution and Reproducibility
- Execute all four notebooks end-to-end and save rendered outputs in-place.
- Keep randomness control visible in each notebook (not hidden in helpers).
- Persist generated figures under `figures/` for README-stable references.
