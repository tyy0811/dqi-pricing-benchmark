# Stage 1 Paired Encoding Foundation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a toy-pricing-only exact reference encoding backend and shared decision metrics so WHT-truncated and ILP-exact-reference encodings can be compared on the same frozen pricing instances.

**Architecture:** Introduce a canonical toy model layer (`pricing_ilp.py`), an exact full-domain spectral export path (`ilp_to_xorsat_exact.py`), a shared grouped metrics API (`metrics_decision.py`), and a paired comparison runner (`encoding_compare.py`). Integrate minimally into benchmark outputs for P3/P4/P5 without expanding the main run matrix.

**Tech Stack:** Python 3.11, NumPy, SciPy stats, pytest, JSON serialization

---

## File Structure and Responsibilities

- Create: `src/pricing_ilp.py`
  - Build canonical toy-pricing model dict from `PricingProblem`.
  - Own schema fields for bit conventions, codeword mapping, penalties, and canonical instance hash.
- Create: `src/ilp_to_xorsat_exact.py`
  - Evaluate toy model exactly over all assignments.
  - Build exact full-spectrum parity artifact `(B, v, term_weights, constant_offset)` and canonical artifact hash.
  - Own exact reconstruction validation helper.
- Create: `src/metrics_decision.py`
  - Provide shared grouped metrics panel and metric availability metadata.
- Create: `src/encoding_compare.py`
  - Run paired backend comparison with compatibility checks, shared metrics, and signed deltas.
- Modify: `src/benchmark.py`
  - Add one optional Stage 1 paired comparison path for P3/P4/P5 using shared metrics API.
- Modify only if needed: `src/reduction.py`
  - Reuse existing WHT primitives; add tiny helper exports only if required.
- Modify only if needed: `notebooks/_helpers.py`
  - Update formatting only if the touched schema impacts notebook table helpers.
- Create tests:
  - `tests/test_ilp_exact_reference.py`
  - `tests/test_metrics_decision.py`
  - `tests/test_encoding_compare.py`
- Modify tests:
  - `tests/test_benchmark.py` (minimal Stage 1 path assertion)

Skills to apply during execution: `@test-driven-development`, `@verification-before-completion`.

## Chunk 1: Canonical Toy Model (`src/pricing_ilp.py`)

### Task 1: Add canonical model builder and schema test

**Files:**
- Create: `src/pricing_ilp.py`
- Create: `tests/test_ilp_exact_reference.py`

- [ ] **Step 1: Write failing schema test**

```python
def test_build_toy_pricing_ilp_model_schema():
    from src.problem_generator import small_instance
    from src.pricing_ilp import build_toy_pricing_ilp_model

    prob = small_instance(3)
    model = build_toy_pricing_ilp_model(prob, instance_id="pricing_3feat_6bit")

    required = {
        "schema_version", "instance_id", "instance_hash",
        "n_features", "n_bits", "feature_order",
        "bit_order", "bitstring_convention",
        "codeword_mapping", "valid_codewords", "invalid_codewords",
        "segments", "bundle_constraints", "max_luxury", "penalties",
        "objective_formula_label", "objective_semantics",
    }
    assert required.issubset(model.keys())
    assert model["bit_order"] == "msb_first"
    assert model["bitstring_convention"] == "feature_major_msb_first"
```

- [ ] **Step 2: Run failing test**

Run: `python -m pytest tests/test_ilp_exact_reference.py::test_build_toy_pricing_ilp_model_schema -v`  
Expected: FAIL (module/function missing).

- [ ] **Step 3: Implement minimal model builder**

```python
# src/pricing_ilp.py
def build_toy_pricing_ilp_model(prob, instance_id=None) -> dict:
    # build canonical payload from prob.to_dict() + explicit conventions
    # compute canonical instance hash from stable json payload
    return payload
```

- [ ] **Step 4: Re-run test**

Run: `python -m pytest tests/test_ilp_exact_reference.py::test_build_toy_pricing_ilp_model_schema -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pricing_ilp.py tests/test_ilp_exact_reference.py
git commit -m "feat: add canonical toy pricing ILP model contract"
```

## Chunk 2: Exact Table and Exact Artifact (`src/ilp_to_xorsat_exact.py`)

### Task 2: Add exact table evaluator

**Files:**
- Create: `src/ilp_to_xorsat_exact.py`
- Modify: `tests/test_ilp_exact_reference.py`

- [ ] **Step 1: Write failing exact-table test**

```python
def test_evaluate_toy_pricing_model_exact_returns_full_domain_fields():
    from src.problem_generator import small_instance
    from src.pricing_ilp import build_toy_pricing_ilp_model
    from src.ilp_to_xorsat_exact import evaluate_toy_pricing_model_exact

    prob = small_instance(3)
    model = build_toy_pricing_ilp_model(prob, instance_id="pricing_3feat_6bit")
    out = evaluate_toy_pricing_model_exact(model)

    assert out["n_states"] == 2 ** prob.n_bits
    assert len(out["F_table"]) == out["n_states"]
    assert out["state_enumeration_order"] == "integer_ascending"
    assert out["constant_offset_included_in_F_table"] is True
    assert out["F_star"] == max(out["F_table"])
```

- [ ] **Step 2: Run failing test**

Run: `python -m pytest tests/test_ilp_exact_reference.py::test_evaluate_toy_pricing_model_exact_returns_full_domain_fields -v`  
Expected: FAIL.

- [ ] **Step 3: Implement evaluator**

```python
def evaluate_toy_pricing_model_exact(ilp_model) -> dict:
    # enumerate x in ascending integer order
    # evaluate objective with the toy model semantics
    # return F_table + optimizer indices + F_star + constants/metadata
```

- [ ] **Step 4: Re-run targeted tests**

Run: `python -m pytest tests/test_ilp_exact_reference.py -k "model_schema or evaluate_toy_pricing_model_exact" -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ilp_to_xorsat_exact.py tests/test_ilp_exact_reference.py
git commit -m "feat: add exact full-domain toy model evaluator"
```

### Task 3: Add exact full-spectrum artifact + invariants

**Files:**
- Modify: `src/ilp_to_xorsat_exact.py`
- Modify: `tests/test_ilp_exact_reference.py`

- [ ] **Step 1: Write failing exact-artifact schema and reconstruction tests**

```python
def test_build_ilp_exact_reference_artifact_schema_and_exact_flag():
    ...
    art = build_ilp_exact_reference_artifact(model)
    assert art["encoding_type"] == "ilp_exact_reference"
    assert art["reference_path"] == "model_table_full_wht"
    assert art["exact_full_spectrum"] is True
    assert art["exact_reconstructable"] is True
    assert art["m_terms"] == (2 ** art["n_bits"]) - 1


def test_validate_exact_reconstruction_full_domain():
    ...
    report = validate_exact_reconstruction(model, atol=1e-9, rtol=1e-9)
    assert report["ok"] is True
    assert report["max_abs_err"] <= 1e-9
```

- [ ] **Step 2: Run failing tests**

Run: `python -m pytest tests/test_ilp_exact_reference.py -k "exact_reference_artifact_schema or exact_reconstruction_full_domain" -v`  
Expected: FAIL.

- [ ] **Step 3: Implement artifact builder and reconstruction validator**

```python
def build_ilp_exact_reference_artifact(ilp_model) -> dict:
    # full WHT of exact F_table, keep all non-constant terms
    # convert to B/v/term_weights using existing reduction helpers
    # attach constant_offset + parity semantics + canonical hash
    return artifact

def validate_exact_reconstruction(ilp_model, atol=1e-9, rtol=1e-9) -> dict:
    # reconstruct full table and compare full domain
    return {"ok": bool, "max_abs_err": float, ...}
```

- [ ] **Step 4: Add hash determinism test**

```python
def test_exact_artifact_hash_is_deterministic_for_same_instance():
    ...
    art1 = build_ilp_exact_reference_artifact(model)
    art2 = build_ilp_exact_reference_artifact(model)
    assert art1["artifact_hash"] == art2["artifact_hash"]
```

- [ ] **Step 5: Run full ILP-exact test module**

Run: `python -m pytest tests/test_ilp_exact_reference.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ilp_to_xorsat_exact.py tests/test_ilp_exact_reference.py
git commit -m "feat: add exact full-spectrum ILP reference artifact with reconstruction checks"
```

## Chunk 3: Shared Decision Metrics (`src/metrics_decision.py`)

### Task 4: Add grouped metrics API

**Files:**
- Create: `src/metrics_decision.py`
- Create: `tests/test_metrics_decision.py`

- [ ] **Step 1: Write failing required-keys test**

```python
def test_compute_decision_metrics_has_required_sections_and_keys():
    from src.metrics_decision import compute_decision_metrics

    out = compute_decision_metrics(
        F_values=[1.0, 2.0, 3.0],
        G_values=[1.2, 2.1, 2.9],
        f_opt=3.0,
        k_eval=2,
        decode_success=0.9,
        postselection_success=0.8,
        candidate_source_label="unit_test",
    )

    assert "metrics_schema_version" in out
    for section in ["core_decision", "faithfulness", "pipeline", "timing", "metadata", "metric_goal", "metric_availability"]:
        assert section in out
    required = {
        "best_sampled_F", "best_sampled_G", "top1_regret", "topk_regret",
        "optimum_recall_at_k", "precision_at_k_high_F", "best_top_G_sampled_F",
        "surrogate_best_vs_true_best_sampled_gap", "spearman_rho_F_G",
        "retained_energy_eta", "decision_distortion", "decode_success", "postselection_success",
    }
    flat = {**out["core_decision"], **out["faithfulness"], **out["pipeline"]}
    assert required.issubset(flat.keys())
```

- [ ] **Step 2: Run failing test**

Run: `python -m pytest tests/test_metrics_decision.py::test_compute_decision_metrics_has_required_sections_and_keys -v`  
Expected: FAIL.

- [ ] **Step 3: Implement metrics module**

```python
def compute_decision_metrics(...):
    # compute grouped metrics, preserve key presence with None fallbacks
    # emit metric_goal and metric_availability maps
    return payload
```

- [ ] **Step 4: Add and run availability/None behavior tests**

```python
def test_unavailable_metrics_are_present_as_none():
    ...
```

Run: `python -m pytest tests/test_metrics_decision.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/metrics_decision.py tests/test_metrics_decision.py
git commit -m "feat: add shared grouped decision metrics panel"
```

## Chunk 4: Paired Comparison Runner (`src/encoding_compare.py`)

### Task 5: Add paired comparison and schema-compatible report

**Files:**
- Create: `src/encoding_compare.py`
- Create: `tests/test_encoding_compare.py`

- [ ] **Step 1: Write failing paired report test**

```python
def test_paired_encoding_compare_report_is_json_serializable_and_schema_compatible():
    from src.problem_generator import small_instance
    from src.encoding_compare import run_paired_encoding_comparison
    import json

    prob = small_instance(3)
    report = run_paired_encoding_comparison(
        prob=prob,
        k_wht=8,
        alpha_mode="paper",
        execution_mode="mixture",
        decoder_mode="bruteforce",
        n_samples=40,
        seed=11,
        top_k_eval=10,
    )

    required = {
        "comparison_id", "instance_metadata",
        "backend_a_name", "backend_b_name",
        "backend_a_artifact_summary", "backend_b_artifact_summary",
        "artifacts_compatible_for_paired_run", "compatibility_notes",
        "backend_a_metrics", "backend_b_metrics", "metric_deltas",
    }
    assert required.issubset(report.keys())
    json.dumps(report, default=float)
```

- [ ] **Step 2: Run failing test**

Run: `python -m pytest tests/test_encoding_compare.py::test_paired_encoding_compare_report_is_json_serializable_and_schema_compatible -v`  
Expected: FAIL.

- [ ] **Step 3: Implement comparison runner**

```python
def run_paired_encoding_comparison(...):
    # build both artifacts for same instance
    # enforce compatibility predicate
    # run common evaluation path with same knobs
    # compute shared metrics and signed deltas
    return report
```

- [ ] **Step 4: Add compatibility-predicate unit test**

```python
def test_artifact_compatibility_predicate_detects_mismatch():
    ...
```

- [ ] **Step 5: Run test module**

Run: `python -m pytest tests/test_encoding_compare.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/encoding_compare.py tests/test_encoding_compare.py
git commit -m "feat: add paired WHT vs ILP exact comparison runner"
```

## Chunk 5: Minimal Benchmark Integration

### Task 6: Add optional Stage 1 paired path to benchmark

**Files:**
- Modify: `src/benchmark.py`
- Modify: `tests/test_benchmark.py`
- Modify only if needed: `notebooks/_helpers.py`

- [ ] **Step 1: Write failing benchmark assertion for optional Stage 1 output**

```python
def test_benchmark_can_emit_stage1_paired_encoding_section():
    from src.problem_generator import small_instance
    from src.benchmark import run_benchmark

    prob = small_instance(3)
    out = run_benchmark(
        prob,
        k=8,
        ell=2,
        n_dqi_samples=40,
        seed=5,
        # new optional knob
        include_stage1_paired_encoding=True,
    )
    assert "stage1_paired_encoding" in out
```

- [ ] **Step 2: Run failing test**

Run: `python -m pytest tests/test_benchmark.py::test_benchmark_can_emit_stage1_paired_encoding_section -v`  
Expected: FAIL.

- [ ] **Step 3: Implement minimal optional integration**

```python
# src/benchmark.py
# add optional include_stage1_paired_encoding=False
# when True and instance is P3/P4/P5-compatible, attach one paired report section
# no matrix expansion
```

- [ ] **Step 4: Re-run targeted benchmark tests**

Run: `python -m pytest tests/test_benchmark.py -k "stage1_paired_encoding or weighted_and_unweighted" -v`  
Expected: PASS.

- [ ] **Step 5: Update notebook helper formatting only if schema touch requires it**

Run: `python -m pytest tests/test_notebook_support.py -v`  
Expected: PASS (no changes if not required).

- [ ] **Step 6: Commit**

```bash
git add src/benchmark.py tests/test_benchmark.py notebooks/_helpers.py
git commit -m "feat: add optional Stage 1 paired encoding report path"
```

## Chunk 6: Verification and Completion

### Task 7: Run smallest relevant full validation set

**Files:**
- Verify: `tests/test_ilp_exact_reference.py`
- Verify: `tests/test_metrics_decision.py`
- Verify: `tests/test_encoding_compare.py`
- Verify: `tests/test_benchmark.py` (targeted selection)

- [ ] **Step 1: Run exact reference tests**

Run: `python -m pytest tests/test_ilp_exact_reference.py -v`  
Expected: PASS.

- [ ] **Step 2: Run metrics and compare tests**

Run: `python -m pytest tests/test_metrics_decision.py tests/test_encoding_compare.py -v`  
Expected: PASS.

- [ ] **Step 3: Run targeted benchmark regression**

Run: `python -m pytest tests/test_benchmark.py -k "stage1_paired_encoding or weighted_and_unweighted_metrics" -v`  
Expected: PASS.

- [ ] **Step 4: Optional broader confidence run (if time)**

Run: `python -m pytest tests/test_reduction.py tests/test_dqi_pipeline.py -v`  
Expected: PASS.

- [ ] **Step 5: Produce/verify artifacts only if integration path writes outputs**

Run: `ls -la results | head`  
Expected: no unintended artifact explosion.

- [ ] **Step 6: Final commit**

```bash
git add src tests docs/superpowers/specs/2026-03-11-stage1-paired-encoding-foundation-design.md docs/superpowers/plans/2026-03-11-stage1-paired-encoding-foundation.md
git commit -m "feat: stage1 paired encoding foundation with exact ILP reference and shared metrics"
```

## Notes for Execution

- Keep diffs scoped to Stage 1 only.
- Do not generalize beyond toy pricing family.
- Preserve backward compatibility for existing benchmark outputs.
- Prefer small, targeted test runs after each task.
- If git metadata is unavailable in the workspace, complete all file/test steps and report commit steps as blocked by missing repository metadata.
