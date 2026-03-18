# Coherent Execution and Benchmark Comparison Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tiny-instance coherent DQI reference path, preserve the fast mixture path, and expose explicit alpha/execution-mode comparisons in benchmark outputs and notebooks.

**Architecture:** Extend `src/dqi_state.py` with coherent-state helpers and a mode selector while preserving existing mixture behavior. Thread mode metadata through `src/dqi_pipeline.py`, then expand `src/benchmark.py` into explicit `*_mixture`/`*_coherent` records plus stable unsupported payloads and pairwise diagnostics. Add targeted regression tests and notebook sections that present the comparison clearly and honestly.

**Tech Stack:** Python 3.11, NumPy, pytest, Jupyter notebooks

---

## File Structure and Responsibilities

- Modify: `src/dqi_state.py`
  - Owns DQI statevector execution engines and coherent-cap support checks.
- Modify: `src/dqi_pipeline.py`
  - Owns pipeline-level execution mode routing, run metadata, and compact TVD helper.
- Modify: `src/benchmark.py`
  - Owns benchmark orchestration, run-key schema, coherent unsupported payload, and diagnostics.
- Create: `tests/test_dqi_coherent_vs_mixture.py`
  - Owns coherent-vs-mixture correctness and bounds tests.
- Modify: `tests/test_benchmark.py`
  - Owns benchmark schema and diagnostics assertions.
- Modify: `notebooks/03_paper_faithful_dqi.ipynb`
  - Add tiny coherent-vs-mixture alpha ablation section.
- Modify: `notebooks/05_decoder_comparison.ipynb`
  - Add paper-alpha decoder stability check across alpha modes with coherent optional/tiny-only framing.
- Modify if needed: `notebooks/_helpers.py`
  - Add formatting support for new benchmark keys/diagnostics.

## Chunk 1: Coherent Engine and Selector (`src/dqi_state.py`)

### Task 1: Add failing tests for coherent support helpers and hard-cap behavior

**Files:**
- Create: `tests/test_dqi_coherent_vs_mixture.py`
- Modify: `src/dqi_state.py`

- [ ] **Step 1: Write failing tests for coherent support helper API**

```python
def test_coherent_support_helpers_report_joint_size():
    bits = joint_num_bits_coherent(m=3, n=2, ell=2)
    size = joint_state_size_coherent(m=3, n=2, ell=2)
    assert bits == 7
    assert size == 2 ** bits
```

- [ ] **Step 2: Write failing test for oversized coherent error**

```python
def test_run_dqi_statevector_coherent_raises_when_cap_exceeded():
    with pytest.raises(ValueError, match="coherent_hilbert_cap_exceeded"):
        run_dqi_statevector_coherent(B, v, alpha, ell=3, n_samples=8)
```

- [ ] **Step 3: Run tests to verify failure**

Run: `python -m pytest tests/test_dqi_coherent_vs_mixture.py -k "helpers or cap" -v`
Expected: FAIL due to missing helper/coherent functions.

### Task 2: Implement coherent support helpers and coherent runner

**Files:**
- Modify: `src/dqi_state.py`
- Test: `tests/test_dqi_coherent_vs_mixture.py`

- [ ] **Step 1: Add coherent support helpers**

```python
def weight_register_bits(ell: int) -> int: ...
def joint_num_bits_coherent(m: int, n: int, ell: int) -> int: ...
def joint_state_size_coherent(m: int, n: int, ell: int) -> int: ...
def coherent_supported_for_instance(m: int, n: int, ell: int, cap_limit: int = 4096) -> bool: ...
```

- [ ] **Step 2: Add `run_dqi_statevector_coherent(...)`**

```python
def run_dqi_statevector_coherent(...):
    # Build |t>|e>|s> joint state with explicit t-register
    # Apply phase on e, syndrome map to s, projective decode surrogate,
    # Hadamard on s, marginalize over t,e to p(x)
```

- [ ] **Step 3: Add selector to `run_dqi_statevector(...)`**

```python
def run_dqi_statevector(..., execution_mode: str = "mixture", ...):
    if execution_mode == "mixture":
        ...
    elif execution_mode == "coherent":
        return run_dqi_statevector_coherent(...)
    raise ValueError(...)
```

- [ ] **Step 4: Persist model metadata in return payload-compatible outputs**

Ensure coherent path returns probabilities over final candidate `x` and exposes semantics via calling layers.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_dqi_state.py tests/test_dqi_coherent_vs_mixture.py -v`
Expected: PASS for coherent helper/cap tests; no regressions in existing mixture tests.

## Chunk 2: Pipeline Execution Mode and TVD Helper (`src/dqi_pipeline.py`)

### Task 3: Add failing tests for pipeline execution metadata and TVD helper

**Files:**
- Create/Modify: `tests/test_dqi_coherent_vs_mixture.py`
- Modify: `src/dqi_pipeline.py`

- [ ] **Step 1: Add failing test for pipeline run metadata fields**

```python
def test_pipeline_persists_execution_mode_and_coherent_flags():
    results = DQIPipeline(..., execution_mode="mixture").run(...)
    assert "execution_mode" in results
    assert "coherent_supported" in results
    assert "coherent_exact_small_instance" in results
```

- [ ] **Step 2: Add failing tests for TVD helper**

```python
def test_compare_candidate_distributions_tvd_identical_is_zero():
    out = compare_candidate_distributions_tvd(run_a, run_b)
    assert out["ok"] is True
    assert out["tvd"] == 0.0
```

- [ ] **Step 3: Run targeted failures**

Run: `python -m pytest tests/test_dqi_coherent_vs_mixture.py -k "pipeline or tvd" -v`
Expected: FAIL due to missing execution metadata/helper.

### Task 4: Implement pipeline mode threading and TVD helper

**Files:**
- Modify: `src/dqi_pipeline.py`
- Test: `tests/test_dqi_coherent_vs_mixture.py`

- [ ] **Step 1: Add `execution_mode` arg + validation to `DQIPipeline.__init__`**

```python
allowed_execution_modes = {"mixture", "coherent"}
```

- [ ] **Step 2: Thread `execution_mode` into `run_dqi_statevector(...)` call**

- [ ] **Step 3: Add per-run metadata**

```python
{
  "execution_mode": ...,
  "coherent_supported": ...,
  "coherent_exact_small_instance": ...,
  "weight_register_bits": ...,
  "decode_semantics": "projective_surrogate",
  "state_model": "...",
  "postselection_kind": "projective_surrogate",
}
```

- [ ] **Step 4: Add compact TVD helper in pipeline module**

```python
def compare_candidate_distributions_tvd(run_a: dict, run_b: dict) -> dict:
    # validate, align domain, compute 0.5 * sum |p-q|
```

- [ ] **Step 5: Run targeted tests**

Run: `python -m pytest tests/test_dqi_coherent_vs_mixture.py tests/test_dqi_pipeline.py -v`
Expected: PASS.

## Chunk 3: Benchmark Keys, Unsupported Payload, and Diagnostics (`src/benchmark.py`)

### Task 5: Add failing benchmark-schema tests

**Files:**
- Modify: `tests/test_benchmark.py`
- Modify: `src/benchmark.py`

- [ ] **Step 1: Add failing assertions for new run keys**

```python
for key in [
    "dqi_uniform_mixture",
    "dqi_paper_mixture",
    "dqi_heuristic_mixture",
    "dqi_paper_coherent",
]:
    assert key in results
```

- [ ] **Step 2: Add failing assertions for new diagnostics fields**

```python
assert "delta_best_F_paper_vs_heuristic" in results["dqi_pairwise_diagnostics"]
assert "tvd_paper_mixture_vs_coherent" in results["dqi_pairwise_diagnostics"]
assert "coherent_comparison_attempted" in results["dqi_pairwise_diagnostics"]
```

- [ ] **Step 3: Run benchmark test to verify failure**

Run: `python -m pytest tests/test_benchmark.py::test_benchmark_reports_weighted_and_unweighted_metrics -v`
Expected: FAIL on missing keys/diagnostics.

### Task 6: Implement benchmark mode matrix and diagnostics

**Files:**
- Modify: `src/benchmark.py`
- Modify if needed: `notebooks/_helpers.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Run explicit benchmark modes with same `(k, ell)`**

```python
dqi_uniform_mixture
dqi_paper_mixture
dqi_heuristic_mixture
dqi_paper_coherent  # attempt only, same k/ell
```

- [ ] **Step 2: Build deterministic unsupported coherent payload when cap fails**

Include required fields:
`status, reason, execution_mode, coherent_supported, coherent_exact_small_instance, coherent_comparison_attempted, alpha_mode, k, ell, n_bits, m, joint_num_bits, joint_state_size, cap_limit, alpha_vector, weight_register_bits, decoder_mode`.

- [ ] **Step 3: Add pairwise diagnostics**

```python
{
  "delta_best_F_paper_vs_heuristic": ...,
  "delta_success_prob_paper_vs_heuristic": ...,
  "delta_best_F_paper_mixture_vs_coherent": ...,
  "tvd_paper_mixture_vs_coherent": ...,
  "same_top_sample_flag_paper_mixture_vs_coherent": ...,
  "coherent_comparison_attempted": ...,
}
```

Use tie-break rule for same-top flag:
1) best sampled `F`, 2) then weighted `G`, 3) then lexicographic `x`.

- [ ] **Step 4: Keep backward-compatible aliases**

```python
results["dqi_uniform"] = results["dqi_uniform_mixture"]
results["dqi_paper"] = results["dqi_paper_mixture"]
results["dqi_heuristic"] = results["dqi_heuristic_mixture"]
```

- [ ] **Step 5: Update helper formatting only if notebook table requires it**

- [ ] **Step 6: Run benchmark-related tests**

Run: `python -m pytest tests/test_benchmark.py tests/test_notebook_support.py -v`
Expected: PASS with stable schema in both supported and unsupported coherent cases.

## Chunk 4: Notebook Updates and End-to-End Verification

### Task 7: Notebook 03 coherent check section

**Files:**
- Modify: `notebooks/03_paper_faithful_dqi.ipynb`

- [ ] **Step 1: Add markdown title**

`Alpha ablation and coherent-vs-mixture check (tiny-instance exact surrogate model)`

- [ ] **Step 2: Add code cell comparing**
- uniform mixture
- paper mixture
- heuristic mixture
- paper coherent (when supported)

- [ ] **Step 3: Show side-by-side metrics table**

Include: best `F`, success probability, execution mode, coherent flags, and TVD/coherent diagnostics when available.

### Task 8: Notebook 05 decoder stability across alpha modes

**Files:**
- Modify: `notebooks/05_decoder_comparison.ipynb`

- [ ] **Step 1: Add paper-alpha decoder comparison cells**

- [ ] **Step 2: Add alpha-mode stability summary**

Explicitly state coherent is optional/tiny-only; do not imply large-instance coherent validation.

- [ ] **Step 3: Save notebook outputs (if project policy allows) and verify JSON parseability**

Run: `python -m pytest tests/test_notebook_support.py -v`
Expected: PASS.

## Final Verification

- [ ] **Step 1: Run coherent + benchmark focused suite**

Run: `python -m pytest tests/test_dqi_coherent_vs_mixture.py tests/test_dqi_pipeline.py tests/test_benchmark.py tests/test_dqi_state.py tests/test_notebook_support.py -q`
Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest -q`
Expected: PASS (allowing existing non-fatal warnings).

- [ ] **Step 3: Commit in logical slices (if VCS metadata exists)**

```bash
git add src/dqi_state.py src/dqi_pipeline.py tests/test_dqi_coherent_vs_mixture.py
git commit -m "feat: add coherent tiny-instance DQI execution mode"

git add src/benchmark.py tests/test_benchmark.py notebooks/_helpers.py tests/test_notebook_support.py
git commit -m "feat: add benchmark coherent-vs-mixture diagnostics and stable schema"

git add notebooks/03_paper_faithful_dqi.ipynb notebooks/05_decoder_comparison.ipynb
git commit -m "docs: add notebook coherent-vs-mixture and alpha-mode decoder analysis"
```
