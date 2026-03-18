# DQI Notebook Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and execute four portfolio-grade notebooks that cover toy DQI reproduction, pricing-model verification, paper-faithful DQI diagnostics, and Qiskit cross-framework sanity checks.

**Architecture:** Keep all nontrivial computation in `src/*`, and keep notebook glue in `notebooks/_helpers.py`. Notebooks call the existing pipeline/reduction/state modules and focus on analysis/visualization with deterministic seeds and saved rendered outputs.

**Tech Stack:** Python 3.11, NumPy, pandas, matplotlib, PennyLane, Qiskit, pytest, nbformat/nbconvert (for execution).

---

### Task 1: Add reusable compute utilities in `src/*`

**Files:**
- Modify: `src/problem_generator.py`
- Test: `tests/test_notebook_support.py` (new)

**Step 1: Write failing tests for feasibility summary API**

```python
def test_compute_feasibility_stats_fields():
    from src.problem_generator import default_instance, compute_feasibility_stats
    prob = default_instance()
    stats = compute_feasibility_stats(prob)
    for key in ["valid_fraction", "invalid_tier_count", "bundle_violations", "luxury_cap_violations"]:
        assert key in stats
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_notebook_support.py::test_compute_feasibility_stats_fields -v`
Expected: FAIL (function not found)

**Step 3: Implement minimal compute API in `src/problem_generator.py`**

- Add `compute_feasibility_stats(prob: PricingProblem) -> dict`
- Enumerate all assignments and compute consistent fields:
  - `valid_fraction`
  - `invalid_tier_count`
  - `bundle_violations`
  - `luxury_cap_violations`
  - optional helpers (`valid_count`, `total_count`) for Notebook 02 summaries

**Step 4: Run targeted test**

Run: `python -m pytest tests/test_notebook_support.py::test_compute_feasibility_stats_fields -v`
Expected: PASS

---

### Task 2: Add notebook helper module

**Files:**
- Create: `notebooks/_helpers.py`
- Test: `tests/test_notebook_support.py`

**Step 1: Write failing tests for helper output schema**

```python
def test_tvd_basic_properties():
    from notebooks._helpers import tvd
    import numpy as np
    p = np.array([0.5, 0.5])
    q = np.array([1.0, 0.0])
    assert abs(tvd(p, q) - 0.5) < 1e-12
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_notebook_support.py::test_tvd_basic_properties -v`
Expected: FAIL (module not found)

**Step 3: Implement presentation-only helpers**

- Path-safe loader: `load_instance(pathlike)`
- Decoding/formatting:
  - `decode_config_table(prob, x)`
  - `format_optimal_table(...)`
  - `format_feasibility_stats(stats)`
- Plotting:
  - distribution and overlay plotting helpers
- Metrics:
  - `tvd(p, q)`
  - `resource_table(circuit, transpiled_circuit, basis_gates, optimization_level)`
- Verification cards:
  - `verification_card(metrics: dict)`

**Step 4: Run notebook-support tests**

Run: `python -m pytest tests/test_notebook_support.py -v`
Expected: PASS

---

### Task 3: Create Notebook 01 (`01_reproduce_toy_dqi.ipynb`)

**Files:**
- Create: `notebooks/01_reproduce_toy_dqi.ipynb`
- Create/update output path: `figures/notebook01_toy_histogram.png`

**Step 1: Build notebook cells**

- Imports/seeds/version cell
- Tutorial-scope markdown note
- Tiny toy max-XORSAT setup
- Exact classical optimum check table
- DQI run + histogram + top-k table (bitstring, probability, objective, optimal flag)
- Five-stage map markdown
- One-knob (`ell`) rerun and side-by-side comparison
- Short interpretation

**Step 2: Execute notebook and save outputs**

Run: `python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_reproduce_toy_dqi.ipynb`
Expected: success; rendered outputs stored in notebook

**Step 3: Verify figure artifact exists**

Run: `ls -la figures/notebook01_toy_histogram.png`
Expected: file exists

---

### Task 4: Create Notebook 02 (`02_pricing_instance_demo.ipynb`)

**Files:**
- Create: `notebooks/02_pricing_instance_demo.ipynb`

**Step 1: Build notebook cells**

- Imports/seeds/path-safe setup
- Problem-definition markdown
- Load frozen instances 3->4->5 features
- Per instance:
  - summary table (bits, feasible states, total states, optimum)
  - brute-force optimum
  - immediate decoded human-readable optimum table
  - feasibility stats (consistent fields)
  - CP-SAT block (run if available, else explicit skip message)
- Consolidated required table: `instance`, `feature`, `tier`, `price`
- Short interpretation

**Step 2: Execute notebook and save outputs**

Run: `python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_pricing_instance_demo.ipynb`
Expected: success; rendered outputs stored in notebook

---

### Task 5: Create Notebook 03 (`03_paper_faithful_dqi.ipynb`)

**Files:**
- Create: `notebooks/03_paper_faithful_dqi.ipynb`

**Step 1: Build notebook cells**

- Imports/seeds/goal note
- Single five-stage explanation block
- Load 6/8/10-bit instances
- For each instance:
  - build surrogate (`k<=15`, `m<=20`) and show `(n,m,k,ell)`
  - exact anchor row (`F*`, `best sampled F`, approximation ratio)
  - run five-step pipeline
  - compute/report:
    - success probability
    - best sampled `G`
    - best sampled `F`
    - `F(top surrogate-ranked sampled x)`
    - sampled misranking gap
    - exact full-space `rho(F,G)` note (tiny-instance exactness)
- Weights ablation with labels `uniform_alpha` and `current_eigenvector_alpha`
- Render verification card per instance with failure signal field
- Short interpretation

**Step 2: Execute notebook and save outputs**

Run: `python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_paper_faithful_dqi.ipynb`
Expected: success; rendered outputs stored in notebook

---

### Task 6: Create Notebook 04 (`04_qiskit_comparison.ipynb`)

**Files:**
- Create: `notebooks/04_qiskit_comparison.ipynb`

**Step 1: Build notebook cells**

- Imports/seeds
- Hard Qiskit check (raise error if missing)
- Load smallest instance and build `(B,v)` with fixed `(k,ell)`
- Explicit no-decoder apples-to-apples note and measured-register definition
- Run matched no-decoder distributions in both frameworks (statevector probabilities)
- Compute TVD
- Build overlay histogram over top support states
- Build resource table with pinned transpilation settings
- Short modest interpretation
- Scope sentence: simplified cross-framework sanity check, not full decoded-DQI parity proof

**Step 2: Execute notebook and save outputs**

Run: `python -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_qiskit_comparison.ipynb`
Expected: success if Qiskit installed

---

### Task 7: Validation and regression checks

**Files:**
- Modify: `tests/test_notebook_support.py`

**Step 1: Run notebook-support tests**

Run: `python -m pytest tests/test_notebook_support.py -v`
Expected: PASS

**Step 2: Run core affected tests**

Run: `python -m pytest tests/test_reduction.py tests/test_dqi_pipeline.py tests/test_decoder.py tests/test_qiskit.py -v`
Expected: PASS (with expected Qiskit skip behavior only where applicable)

**Step 3: Spot-check executed notebook JSONs include outputs**

Run: `rg '"outputs": \[' notebooks/*.ipynb`
Expected: non-empty output blocks present

---

### Task 8: Documentation touch-up

**Files:**
- Modify: `README.md`

**Step 1: Add notebook section and figure reference**

- Add notebook index with purpose bullets
- Link `figures/notebook01_toy_histogram.png`
- Mention Notebook 04 comparison scope and simplification

**Step 2: Quick lint/readability check**

Run: `rg "Notebook 0[1-4]" README.md`
Expected: all notebook entries present
