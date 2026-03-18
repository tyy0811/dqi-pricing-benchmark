# Notebook 04/05 Realignment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reposition notebook 04/05 and supporting modules so the repo cleanly supports pricing-to-DQI handoff and decoder sensitivity analysis under frozen contracts.

**Architecture:** Add a dedicated OR-Tools pricing model module and a deterministic BP1 decoder module, then standardize interfaces in `reduction.py`, `classical_baselines.py`, and decoder outputs. Keep backward compatibility via wrappers (`weights.py`, reduction legacy keys, decoder legacy methods) while introducing canonical schemas used by new notebooks 04/05.

**Tech Stack:** Python 3.11, NumPy, OR-Tools CP-SAT, pandas, Jupyter notebooks, pytest

---

## File Map

- Create: `src/pricing_model_ortools.py` — CP-SAT model build/solve + standardized pricing result contract.
- Modify: `src/classical_baselines.py` — orchestrate CP-SAT through shared model module; expose standardized result fields.
- Modify: `src/reduction.py` — canonical surrogate artifact API + compatibility wrappers.
- Create: `src/decoder_bp1.py` — deterministic bit-flip style BP1 decoder.
- Modify: `src/decoder_bruteforce.py` — add structured decoder result API while preserving existing `decode` behavior.
- Create: `src/weights_paper.py` — paper/demo-derived alpha functions.
- Create: `src/weights_heuristic.py` — heuristic alpha functions.
- Modify: `src/weights.py` — compatibility shim forwarding to split modules.
- Rename/Modify: `notebooks/04_qiskit_comparison.ipynb` -> `notebooks/04_pricing_pipeline.ipynb`.
- Rename/Modify: `notebooks/05_resource_analysis.ipynb` -> `notebooks/05_decoder_comparison.ipynb`.
- Modify: `README.md` — notebook index and naming updates.
- Create/Modify tests:
  - `tests/test_pricing_model_ortools.py`
  - `tests/test_reduction.py`
  - `tests/test_baselines.py`
  - `tests/test_bp_decoder.py`
  - `tests/test_decoder.py`
  - `tests/test_weights.py`

Skills to apply during execution: `@test-driven-development`, `@verification-before-completion`, `@executing-plans`.

## Chunk 1: Pricing Contract and Baseline Wiring

### Task 1: Add pricing CP-SAT model module (`src/pricing_model_ortools.py`)

**Files:**
- Create: `src/pricing_model_ortools.py`
- Test: `tests/test_pricing_model_ortools.py`

- [ ] **Step 1: Write failing contract tests**

```python
# tests/test_pricing_model_ortools.py
import pytest

from src.problem_generator import small_instance
from src.pricing_model_ortools import solve_pricing_with_cp_sat


def test_cp_sat_result_schema_when_available():
    pytest.importorskip("ortools.sat.python.cp_model")
    prob = small_instance(3)
    result = solve_pricing_with_cp_sat(prob, instance_id="pricing_3feat_6bit")

    required = {
        "status", "objective_value", "optimized_objective_value", "x_opt",
        "decoded_config", "solve_time_s", "constraint_summary", "instance_id", "metadata"
    }
    assert required.issubset(result.keys())
    assert result["objective_value"] == prob.evaluate(result["x_opt"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pricing_model_ortools.py::test_cp_sat_result_schema_when_available -v`
Expected: FAIL with module/function missing.

- [ ] **Step 3: Implement minimal CP-SAT model module**

```python
# src/pricing_model_ortools.py (shape)
def solve_pricing_with_cp_sat(prob, instance_id=None):
    # build one-hot model over x in [0, 2^n)
    # maximize monotone-transformed score if needed
    # return standardized dict with objective_value and optimized_objective_value
    ...
```

Implementation notes:
- `objective_value` must be raw `F(x_opt)`.
- `optimized_objective_value` is the exact scalar used in `model.Maximize(...)`.
- `status` normalized to `optimal|feasible|infeasible|model_invalid|error`.

- [ ] **Step 4: Run module tests**

Run: `python -m pytest tests/test_pricing_model_ortools.py -v`
Expected: PASS (or skip if OR-Tools absent).

- [ ] **Step 5: Commit (if git metadata exists)**

```bash
git add src/pricing_model_ortools.py tests/test_pricing_model_ortools.py
git commit -m "feat: add CP-SAT pricing model with standardized result schema"
```

### Task 2: Rewire classical baselines to shared CP-SAT path

**Files:**
- Modify: `src/classical_baselines.py`
- Test: `tests/test_baselines.py`

- [ ] **Step 1: Add failing baseline schema test**

```python
# tests/test_baselines.py

def test_run_all_baselines_cp_sat_uses_standard_contract():
    prob = small_instance(3)
    results = run_all_baselines(prob, sa_iter=100, random_samples=20, seed=7)
    cp = results["cp_sat"]
    assert cp.get("status") in {"ok", "unavailable"}
    if cp["status"] == "ok":
        assert "objective_value" in cp
        assert "optimized_objective_value" in cp
        assert cp["objective_value"] == prob.evaluate(cp["x"])
```

- [ ] **Step 2: Run targeted test and confirm failure**

Run: `python -m pytest tests/test_baselines.py::test_run_all_baselines_cp_sat_uses_standard_contract -v`
Expected: FAIL on missing fields.

- [ ] **Step 3: Update baseline orchestration**

```python
# src/classical_baselines.py (shape)
from src.pricing_model_ortools import solve_pricing_with_cp_sat

# in run_all_baselines: map standardized CP-SAT solve result into cp_sat entry
```

Compatibility note:
- Keep `cp_sat_solver(prob) -> (x_opt, f_opt)` as legacy helper for current callers.
- Add new helper (e.g., `cp_sat_solver_result`) for full schema.

- [ ] **Step 4: Re-run baseline tests**

Run: `python -m pytest tests/test_baselines.py -v`
Expected: PASS (with OR-Tools-dependent tests skipped only when package missing).

- [ ] **Step 5: Commit (if git metadata exists)**

```bash
git add src/classical_baselines.py tests/test_baselines.py
git commit -m "refactor: unify CP-SAT baseline through pricing model contract"
```

## Chunk 2: Surrogate Artifact and Weights Split

### Task 3: Add canonical surrogate artifact API with round-trip test

**Files:**
- Modify: `src/reduction.py`
- Test: `tests/test_reduction.py`

- [ ] **Step 1: Add failing artifact contract tests**

```python
# tests/test_reduction.py
from src.reduction import build_surrogate_artifact, surrogate_artifact_from_dict


def test_surrogate_artifact_contract_fields():
    prob = default_instance()
    art = build_surrogate_artifact(prob, k=10)
    required = {
        "artifact_version", "mode", "bit_order", "n", "m", "k_requested", "k_selected",
        "B", "v", "term_weights", "indices", "coefficients",
        "truncation_metadata", "diagnostics", "source_metadata"
    }
    assert required.issubset(art.keys())


def test_surrogate_artifact_roundtrip_dict():
    prob = default_instance()
    art = build_surrogate_artifact(prob, k=8)
    clone = surrogate_artifact_from_dict(art)
    np.testing.assert_array_equal(np.array(art["B"]), np.array(clone["B"]))
    np.testing.assert_array_equal(np.array(art["v"]), np.array(clone["v"]))
```

- [ ] **Step 2: Run targeted tests and confirm failure**

Run: `python -m pytest tests/test_reduction.py::test_surrogate_artifact_contract_fields tests/test_reduction.py::test_surrogate_artifact_roundtrip_dict -v`
Expected: FAIL (new API missing).

- [ ] **Step 3: Implement artifact API and compatibility wrappers**

```python
# src/reduction.py (shape)
def build_surrogate_artifact(prob, k=15, *, source_metadata=None, random_seed=None):
    ...
    return {
        "artifact_version": "1.0",
        ...,
        "term_weights": term_weights,
    }

# legacy wrapper keeps 'weights' alias for callers expecting old output
```

Implementation notes:
- Canonical key is `term_weights`; legacy `weights` alias only in wrapper paths.
- Keep existing function names operational for backward compatibility.

- [ ] **Step 4: Run full reduction tests**

Run: `python -m pytest tests/test_reduction.py -v`
Expected: PASS.

- [ ] **Step 5: Commit (if git metadata exists)**

```bash
git add src/reduction.py tests/test_reduction.py
git commit -m "feat: add canonical surrogate artifact contract with round-trip tests"
```

### Task 4: Split weights modules and keep shim compatibility

**Files:**
- Create: `src/weights_paper.py`
- Create: `src/weights_heuristic.py`
- Modify: `src/weights.py`
- Test: `tests/test_weights.py`

- [ ] **Step 1: Add failing shim-forwarding tests**

```python
# tests/test_weights.py
from src.weights import uniform_weights, optimal_weights
from src import weights_paper, weights_heuristic


def test_weights_shim_uniform_matches_paper_module():
    assert (uniform_weights(3) == weights_paper.uniform_weights(3)).all()


def test_weights_shim_optimal_matches_heuristic_module(default_surrogate):
    B, v = default_surrogate
    a1 = optimal_weights(B, v, ell=3)
    a2 = weights_heuristic.optimal_weights(B, v, ell=3)
    assert a1.shape == a2.shape
```

- [ ] **Step 2: Run targeted tests and confirm failure**

Run: `python -m pytest tests/test_weights.py::test_weights_shim_uniform_matches_paper_module -v`
Expected: FAIL before split implementation.

- [ ] **Step 3: Implement split modules + shim**

```python
# src/weights_paper.py
# keep uniform_weights + parameter validation and paper/demo-facing naming

# src/weights_heuristic.py
# keep heuristic matrix/eigenvector path

# src/weights.py
# re-export stable API and deprecate direct scientific ownership
```

- [ ] **Step 4: Re-run weight tests**

Run: `python -m pytest tests/test_weights.py -v`
Expected: PASS.

- [ ] **Step 5: Commit (if git metadata exists)**

```bash
git add src/weights_paper.py src/weights_heuristic.py src/weights.py tests/test_weights.py
git commit -m "refactor: split paper and heuristic alpha weights with compatibility shim"
```

## Chunk 3: Decoder Contract and BP1 Implementation

### Task 5: Add decoder result schema to brute-force decoder

**Files:**
- Modify: `src/decoder_bruteforce.py`
- Modify: `tests/test_decoder.py`

- [ ] **Step 1: Add failing structured result test**

```python
# tests/test_decoder.py

def test_bruteforce_decoder_result_contract():
    B = np.eye(4, dtype=np.int8)
    dec = BoundedDistanceDecoder(B, ell=2)
    s = np.array([1, 0, 0, 0], dtype=np.int8)
    result = dec.decode_result(s)
    required = {"decoder_name", "decoded_error", "success", "within_radius", "distance", "num_flips", "metadata"}
    assert required.issubset(result.keys())
    assert result["decoder_name"] == "bruteforce"
```

- [ ] **Step 2: Run targeted test and confirm failure**

Run: `python -m pytest tests/test_decoder.py::test_bruteforce_decoder_result_contract -v`
Expected: FAIL (`decode_result` missing).

- [ ] **Step 3: Implement structured result API**

```python
# src/decoder_bruteforce.py (shape)
def decode_result(self, syndrome):
    # keep existing decode() unchanged for compatibility
    # compute success, within_radius, distance, num_flips
    ...
```

- [ ] **Step 4: Re-run decoder tests**

Run: `python -m pytest tests/test_decoder.py -v`
Expected: PASS.

- [ ] **Step 5: Commit (if git metadata exists)**

```bash
git add src/decoder_bruteforce.py tests/test_decoder.py
git commit -m "feat: add standardized decode_result contract to brute-force decoder"
```

### Task 6: Implement deterministic BP1 decoder module

**Files:**
- Create: `src/decoder_bp1.py`
- Create: `tests/test_bp_decoder.py`

- [ ] **Step 1: Add failing BP1 behavior tests**

```python
# tests/test_bp_decoder.py
import numpy as np
from src.decoder_bp1 import BP1Decoder


def test_bp1_decoder_is_deterministic():
    B = np.array([[1,0,1],[0,1,1],[1,1,0]], dtype=np.int8)
    s = np.array([1,0,1], dtype=np.int8)
    d = BP1Decoder(B, ell=2, max_iter=8)
    r1 = d.decode_result(s)
    r2 = d.decode_result(s)
    assert r1["success"] == r2["success"]
    np.testing.assert_array_equal(r1["decoded_error"], r2["decoded_error"])


def test_bp1_result_contract():
    B = np.eye(3, dtype=np.int8)
    d = BP1Decoder(B, ell=1, max_iter=5)
    r = d.decode_result(np.array([1,0,0], dtype=np.int8))
    assert {"decoder_name", "decoded_error", "success", "within_radius", "distance", "num_flips", "metadata"}.issubset(r.keys())
```

- [ ] **Step 2: Run targeted tests and confirm failure**

Run: `python -m pytest tests/test_bp_decoder.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement BP1 decoder**

```python
# src/decoder_bp1.py (shape)
class BP1Decoder:
    # deterministic hard-decision bit-flip iterations
    # tie-break by lowest index for deterministic behavior
    ...
```

Implementation notes:
- Deterministic update order.
- `within_radius = (distance is not None and distance <= ell)`.
- `num_flips` counts accepted bit flips across iterations.

- [ ] **Step 4: Run BP1 + decoder suite**

Run: `python -m pytest tests/test_bp_decoder.py tests/test_decoder.py -v`
Expected: PASS.

- [ ] **Step 5: Commit (if git metadata exists)**

```bash
git add src/decoder_bp1.py tests/test_bp_decoder.py tests/test_decoder.py
git commit -m "feat: add deterministic BP1 decoder and shared result schema"
```

## Chunk 4: Notebook Realignment and Documentation

### Task 7: Rename and rebuild notebooks 04/05 for new roles

**Files:**
- Rename/Modify: `notebooks/04_qiskit_comparison.ipynb` -> `notebooks/04_pricing_pipeline.ipynb`
- Rename/Modify: `notebooks/05_resource_analysis.ipynb` -> `notebooks/05_decoder_comparison.ipynb`

- [ ] **Step 1: Rename files and add failing smoke checks**

Run:
- `test -f notebooks/04_pricing_pipeline.ipynb`
- `test -f notebooks/05_decoder_comparison.ipynb`
Expected: FAIL before rename, PASS after rename.

- [ ] **Step 2: Implement Notebook 04 required content**

Required cells:
- Pricing instance definition and ID.
- Hard OR-Tools import check (raise if missing).
- CP-SAT solve via `src/pricing_model_ortools.py`.
- Solve result table including `objective_value` and `optimized_objective_value`.
- Surrogate artifact creation via `src/reduction.py` canonical API.
- Export artifact JSON/NPZ in `data/instances/` with metadata.

- [ ] **Step 3: Implement Notebook 05 required content**

Required cells:
- Load fixed surrogate artifact(s).
- Shared decoder harness for brute-force and BP1 using `decode_result` schema.
- Tables/plots: decode success rate vs `ell`, postselection impact, best-`F` delta.
- Explicit interpretation and caveats.

- [ ] **Step 4: Execute notebooks end-to-end**

Run:
- `python -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_pricing_pipeline.ipynb`
- `python -m jupyter nbconvert --to notebook --execute --inplace notebooks/05_decoder_comparison.ipynb`

Expected: both execute successfully with rendered outputs.

- [ ] **Step 5: Commit (if git metadata exists)**

```bash
git add notebooks/04_pricing_pipeline.ipynb notebooks/05_decoder_comparison.ipynb
git commit -m "feat: repurpose notebook 04/05 to pricing pipeline and decoder comparison"
```

### Task 8: README and final verification

**Files:**
- Modify: `README.md`
- Optional modify: `tests/test_notebook_support.py` (if helper APIs changed)

- [ ] **Step 1: Update notebook index in README**

- Replace old notebook 04/05 descriptions with new roles.
- Remove references to `04_qiskit_comparison.ipynb` and old `05_resource_analysis.ipynb` in this cycle.

- [ ] **Step 2: Run targeted regression suite**

Run:
- `python -m pytest tests/test_pricing_model_ortools.py tests/test_baselines.py tests/test_reduction.py tests/test_decoder.py tests/test_bp_decoder.py tests/test_weights.py -v`

Expected: PASS (with OR-Tools-dependent tests skipped only where explicitly marked).

- [ ] **Step 3: Run broader safety suite**

Run:
- `python -m pytest tests/test_dqi_pipeline.py tests/test_benchmark.py tests/test_notebook_support.py -v`

Expected: PASS.

- [ ] **Step 4: Verification-before-completion checks**

Run:
- `rg "04_qiskit_comparison|05_resource_analysis" README.md notebooks -n`
Expected: no stale references for renamed notebooks in active index.

Run:
- `python -m pytest -q`
Expected: green suite or documented/skipped dependency cases.

- [ ] **Step 5: Commit (if git metadata exists)**

```bash
git add README.md tests
git commit -m "docs: align README and tests with notebook 04/05 repositioning"
```

## Notes for Execution
- Keep backward compatibility in public function signatures unless explicitly replaced by canonical contracts.
- Prefer additive APIs first, then migrate callers, then remove deprecated internal paths only if unreferenced.
- Ensure Notebook 04 is hard-dependent on OR-Tools (fail early by design).
