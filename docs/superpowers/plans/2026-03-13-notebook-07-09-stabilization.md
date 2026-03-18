# Notebook 07/09 Stabilization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize Notebook 07 as an honest subset diagnostic/resource artifact and expand Notebook 09 into a compact reviewer-facing scenario matrix without adding new notebooks.

**Architecture:** Keep both notebooks in place and add only thin helper support where it improves correctness or testability. Notebook 07 should rely on canonical subset APIs plus a small label-aligned diagnostic surface, while Notebook 09 should use a compact scenario-matrix helper or inline contract-preserving logic rather than a broad benchmarking rewrite.

**Tech Stack:** Python, NumPy, pandas, Qiskit subset helpers, OR-Tools multiperiod pricing helpers, pytest, notebook JSON editing

---

## Chunk 1: Notebook 07 subset diagnostics

### Task 1: Map the existing Notebook 07 surfaces and isolate the likely mismatch point

**Files:**
- Modify: `notebooks/07_qiskit_structural_parity.ipynb`
- Modify: `src/dqi_circuit_qiskit.py`
- Modify: `src/dqi_state.py`
- Test: `tests/test_qiskit.py`
- Test: `tests/test_notebook07_qiskit_structural_parity_smoke.py`

- [ ] **Step 1: Read the notebook cells and helper implementations that build the current subset comparison**

Run: `python - <<'PY' ...` or equivalent local inspection command
Expected: identify where labels, marginal extraction, and TVD are computed

- [ ] **Step 2: Write or extend a failing tiny-case test for label/order consistency**

Example target:

```python
def test_qiskit_subset_tiny_case_matches_numpy_reference_labels():
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)
    v = np.array([0, 0], dtype=np.int8)
    alpha = np.array([1.0, 0.0], dtype=np.float64)
    numpy_out = run_dqi_subset_reference(B=B, v=v, alpha=alpha, ell=1)
    qiskit_out = qiskit_subset_validation_report(B=B, v=v, alpha=alpha, ell=1)
    assert set(qiskit_out["top_probabilities"][0]) >= {"label", "numpy_prob", "qiskit_prob"}
```

- [ ] **Step 3: Run the focused Qiskit test and confirm the current gap**

Run: `pytest tests/test_qiskit.py -k subset -v`
Expected: either a failing contract test or evidence of where ordering/marginal logic diverges

- [ ] **Step 4: Implement the smallest real fix or diagnostic helper**

Implementation constraints:

- only fix a demonstrable ordering or marginal bug
- do not relabel final notebook outputs post hoc just to make TVD look better
- if no correctness bug is found, add explicit diagnostic outputs instead of forcing parity

- [ ] **Step 5: Patch Notebook 07 to surface the tiny-case diagnostic**

Notebook outputs must include:

- top-8 NumPy subset probabilities
- top-8 Qiskit subset probabilities
- side-by-side syndrome-label table
- raw TVD
- optional remapped/permuted TVD for diagnosis only
- `qubit_count`, `transpiled_depth`, `cx_count`, `gate_counts`

- [ ] **Step 6: Run focused tests again**

Run: `pytest tests/test_qiskit.py tests/test_notebook07_qiskit_structural_parity_smoke.py -v`
Expected: PASS

## Chunk 2: Notebook 09 scenario matrix

### Task 2: Add a compact scenario matrix surface without broadening the multiperiod code

**Files:**
- Modify: `notebooks/09_3period_pricing_extension.ipynb`
- Modify: `src/classical_baselines_multiperiod.py`
- Modify: `src/pricing_model_ortools_multiperiod.py`
- Test: `tests/test_classical_baselines_multiperiod.py`
- Test: `tests/test_notebook09_3period_pricing_extension_smoke.py`

- [ ] **Step 1: Inspect current Notebook 09 cells and the baseline helper contract**

Run: local file inspection commands against the notebook JSON and helper modules
Expected: identify the smallest insertion point for a curated matrix

- [ ] **Step 2: Write a failing test for the scenario-matrix contract**

Example target:

```python
def test_multiperiod_scenario_matrix_returns_small_reviewer_facing_rows():
    rows = build_multiperiod_scenario_matrix()
    assert 6 <= len(rows) <= 9
    assert {"scenario_label", "dynamic_horizon_revenue", "static_horizon_revenue"} <= set(rows[0])
```

- [ ] **Step 3: Run the focused multiperiod tests and confirm the current missing surface**

Run: `pytest tests/test_classical_baselines_multiperiod.py tests/test_notebook09_3period_pricing_extension_smoke.py -v`
Expected: FAIL on the new scenario-matrix contract

- [ ] **Step 4: Implement a very small helper or notebook-local data build for the scenario matrix**

Coverage requirements:

- low / medium / high demand shift
- at least two inventory regimes
- smoothing `0` and one nonzero setting
- at least one positive-uplift row
- at least one near-zero-uplift row
- at least one row where smoothing changes choices or reduces uplift

- [ ] **Step 5: Patch Notebook 09 so the compact matrix is the first reviewer-facing output**

Table fields:

- `scenario_label`
- `dynamic_horizon_revenue`
- `static_horizon_revenue`
- `static_vs_dynamic_delta`
- inventory summary
- whether choices change across periods

- [ ] **Step 6: Run the focused multiperiod tests again**

Run: `pytest tests/test_classical_baselines_multiperiod.py tests/test_notebook09_3period_pricing_extension_smoke.py -v`
Expected: PASS

## Chunk 3: Final verification and notebook rerun checklist

### Task 3: Verify the patch and prepare notebook rerun guidance

**Files:**
- Modify: `tests/test_qiskit.py`
- Modify: `tests/test_notebook07_qiskit_structural_parity_smoke.py`
- Modify: `tests/test_classical_baselines_multiperiod.py`
- Modify: `tests/test_notebook09_3period_pricing_extension_smoke.py`

- [ ] **Step 1: Run the full focused verification set**

Run:

```bash
pytest \
  tests/test_qiskit.py \
  tests/test_notebook07_qiskit_structural_parity_smoke.py \
  tests/test_classical_baselines_multiperiod.py \
  tests/test_notebook09_3period_pricing_extension_smoke.py \
  -v
```

Expected: PASS

- [ ] **Step 2: Record whether Notebook 07 ends as parity evidence or resource-only evidence**

Decision rule:

- parity evidence only if the diagnostic output materially supports it
- otherwise state explicitly that it remains structural/resource evidence only

- [ ] **Step 3: List notebook cells that must be rerun manually**

Expected: the user gets a concise rerun checklist for Notebook 07 and Notebook 09

- [ ] **Step 4: Note git limitation if still applicable**

Expected: mention that spec/plan were saved but not committed because the workspace is not a git repo root
