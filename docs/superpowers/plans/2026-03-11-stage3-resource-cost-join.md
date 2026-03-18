# Stage 3 Resource-Cost Join Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a canonical embedded `quality_vs_cost` benchmark section that joins decision quality, faithfulness, structure, and resource costs for Stage 2 comparisons without broad refactors.

**Architecture:** Implement a pure join module (`src/resources_compare.py`) that consumes existing benchmark/resource payloads and emits deterministic rows plus a structured Pareto summary. Keep `run_benchmark(...)` integration thin and preserve all existing outputs for compatibility. Extend tests first (TDD) to enforce schema completeness, deterministic IDs, and missing-run resilience.

**Tech Stack:** Python 3, NumPy, pytest, existing repo modules (`benchmark`, `resources`, `structure_metrics`, `metrics_decision`).

---

## File Structure

- Create: `src/resources_compare.py`
  - Pure join layer for `quality_vs_cost` summary and optional manual export helper.
- Modify: `src/benchmark.py`
  - Thin hook to attach canonical embedded `quality_vs_cost` section.
- Modify (only if required): `src/structure_metrics.py`
  - Add explicit/alias fields used by row schema (rank alias, rank deficiency, decodable fraction).
- Modify (only if required): `src/resources.py`
  - Add non-breaking aliases needed by row schema normalization.
- Modify: `tests/test_benchmark.py`
  - Assert embedded section keys and structured pareto contract.
- Modify: `tests/test_resources.py`
  - Assert confidence/status labeling and resource-block invariants used by join.
- Create: `tests/test_resources_compare.py`
  - Unit tests for row determinism, missing-run resilience, and JSON-safe payload.

## Chunk 1: Join Contract Tests First

### Task 1: Add failing tests for `quality_vs_cost` summary contract

**Files:**
- Create: `tests/test_resources_compare.py`
- Modify: `tests/test_benchmark.py`

- [ ] **Step 1: Write failing unit tests for join layer schema**

```python
def test_quality_vs_cost_schema_has_required_top_level_keys():
    out = build_quality_vs_cost_summary(minimal_results_fixture())
    assert out["schema_version"] == "1"
    assert "rows" in out and isinstance(out["rows"], list)
    assert "pareto" in out and "objective" in out["pareto"]
```

```python
def test_quality_vs_cost_row_has_required_quality_and_resource_keys():
    row = build_quality_vs_cost_summary(minimal_results_fixture())["rows"][0]
    for key in ["best_F", "F_star", "best_F_over_F_star", "best_G", "decode_success", "postselection_success"]:
        assert key in row["quality"]
    assert "coherent_alpha_overhead" in row["resources"]
```

```python
def test_quality_vs_cost_row_id_is_deterministic():
    out_a = build_quality_vs_cost_summary(minimal_results_fixture())
    out_b = build_quality_vs_cost_summary(minimal_results_fixture())
    assert [r["row_id"] for r in out_a["rows"]] == [r["row_id"] for r in out_b["rows"]]
```

```python
def test_quality_vs_cost_missing_optional_run_is_resilient():
    payload = minimal_results_fixture()
    payload.pop("dqi_paper_mixture_bp1", None)
    out = build_quality_vs_cost_summary(payload)
    assert isinstance(out["rows"], list)
    assert out["row_count"] == len(out["rows"])
```

- [ ] **Step 2: Run tests to verify failures**

Run:
`pytest tests/test_resources_compare.py tests/test_benchmark.py::test_benchmark_reports_weighted_and_unweighted_metrics -v`

Expected: failures for missing module/function/keys.

- [ ] **Step 3: Commit test-only checkpoint**

```bash
git add tests/test_resources_compare.py tests/test_benchmark.py
git commit -m "test: define quality-vs-cost join contract"
```

## Chunk 2: Implement `src/resources_compare.py`

### Task 2: Build pure join module with deterministic row + Pareto semantics

**Files:**
- Create: `src/resources_compare.py`
- Test: `tests/test_resources_compare.py`

- [ ] **Step 1: Implement minimal join helpers and summary builder**

```python
def build_quality_vs_cost_summary(results: dict) -> dict:
    rows = _build_rows(results)
    objective = {
        "maximize": ["quality.best_F_over_F_star"],
        "minimize": ["resources.total_estimated_resources.total_gate_est_with_decode"],
    }
    pareto_ids = _compute_pareto_frontier(
        rows,
        maximize_keys=objective["maximize"],
        minimize_keys=objective["minimize"],
    )
    return {
        "schema_version": "1",
        "generated_from_benchmark_version": "unknown_or_commit",
        "row_count": len(rows),
        "objectives": objective,
        "rows": rows,
        "pareto": {"objective": objective, "row_ids": pareto_ids},
        "notes": [],
    }
```

- [ ] **Step 2: Implement deterministic `row_id` and row defaults**

```python
def _make_row_id(instance_id, encoding_artifact_id, decoder, alpha_mode, execution_mode, run_label):
    return "|".join([
        str(instance_id),
        str(encoding_artifact_id),
        str(decoder),
        str(alpha_mode),
        str(execution_mode),
        str(run_label),
    ])
```

- [ ] **Step 3: Implement structured Pareto helper semantics**

```python
def _compute_pareto_frontier(rows, maximize_keys, minimize_keys):
    # Exclude rows with None objective values.
    # Keep ties.
    # Return deterministic row_id-sorted frontier.
```

- [ ] **Step 4: Add optional manual export helper**

```python
def export_quality_cost_summary(results: dict, out_path: str | Path) -> None:
    payload = build_quality_vs_cost_summary(results)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
```

- [ ] **Step 5: Run focused tests**

Run:
`pytest tests/test_resources_compare.py -v`

Expected: PASS.

- [ ] **Step 6: Commit module implementation**

```bash
git add src/resources_compare.py tests/test_resources_compare.py
git commit -m "feat: add deterministic quality-vs-cost join module"
```

## Chunk 3: Benchmark Hook + Compatibility

### Task 3: Integrate canonical embedded payload in `run_benchmark(...)`

**Files:**
- Modify: `src/benchmark.py`
- Modify: `tests/test_benchmark.py`

- [ ] **Step 1: Write/update failing benchmark tests for embedded payload**

```python
def test_benchmark_includes_embedded_quality_vs_cost_section():
    out = run_benchmark(...)
    qvc = out["quality_vs_cost"]
    assert qvc["schema_version"] == "1"
    assert "rows" in qvc
    assert "pareto" in qvc
```

```python
def test_benchmark_quality_vs_cost_pareto_objective_is_structured():
    qvc = run_benchmark(...)["quality_vs_cost"]
    assert "maximize" in qvc["pareto"]["objective"]
    assert "minimize" in qvc["pareto"]["objective"]
```

- [ ] **Step 2: Implement thin integration hook**

```python
from src.resources_compare import build_quality_vs_cost_summary
...
results["quality_vs_cost"] = build_quality_vs_cost_summary(results)
```

- [ ] **Step 3: Run targeted benchmark tests**

Run:
`pytest tests/test_benchmark.py::test_benchmark_includes_embedded_quality_vs_cost_section tests/test_benchmark.py::test_benchmark_quality_vs_cost_pareto_objective_is_structured -v`

Expected: PASS.

- [ ] **Step 4: Commit integration**

```bash
git add src/benchmark.py tests/test_benchmark.py
git commit -m "feat: embed canonical quality-vs-cost summary in benchmark"
```

## Chunk 4: Structure/Resource Field Normalization (Only If Needed)

### Task 4: Add non-breaking aliases required by schema

**Files:**
- Modify: `src/structure_metrics.py` (if needed)
- Modify: `src/resources.py` (if needed)
- Modify: `tests/test_resources.py`

- [ ] **Step 1: Add failing tests for new derived fields if absent**

```python
def test_structure_metrics_include_rank_deficiency_and_decodable_fraction():
    out = compute_structure_metrics(B, ell=2)
    assert "rank_B_gf2" in out
    assert "rank_deficiency" in out
    assert "decodable_syndrome_fraction" in out
```

- [ ] **Step 2: Implement minimal backward-compatible field additions**

```python
out["rank_deficiency"] = int(n - rank)
out["decodable_syndrome_fraction"] = (
    float(decodable_unique / syndrome_space_size) if syndrome_space_size > 0 else 0.0
)
```

- [ ] **Step 3: Run focused resource tests**

Run:
`pytest tests/test_resources.py -v`

Expected: PASS.

- [ ] **Step 4: Commit normalization updates**

```bash
git add src/structure_metrics.py src/resources.py tests/test_resources.py
git commit -m "feat: expose normalized structure/resource fields for quality-vs-cost join"
```

## Chunk 5: Final Verification

### Task 5: Run smallest relevant validation set end-to-end

**Files:**
- Modify: none (verification only)

- [ ] **Step 1: Run full targeted suite**

Run:
`pytest tests/test_resources_compare.py tests/test_resources.py tests/test_benchmark.py -v`

Expected: PASS.

- [ ] **Step 2: Smoke-run one benchmark case for payload shape**

Run:
`python -c "from src.problem_generator import small_instance; from src.benchmark import run_benchmark; out=run_benchmark(small_instance(3), k=8, ell=2, n_dqi_samples=20, seed=1); print('quality_vs_cost' in out, len(out['quality_vs_cost']['rows']))"`

Expected: prints `True` and non-negative row count.

- [ ] **Step 3: Optional manual export smoke test (no auto-write in benchmark)**

Run:
`python -c "from src.problem_generator import small_instance; from src.benchmark import run_benchmark; from src.resources_compare import export_quality_cost_summary; out=run_benchmark(small_instance(3), k=8, ell=2, n_dqi_samples=20, seed=1); export_quality_cost_summary(out, 'results/quality_vs_cost_smoke.json'); print('ok')"`

Expected: file created only when explicitly called.

- [ ] **Step 4: Commit final verification/docs touch-ups**

```bash
git add -A
git commit -m "test: validate stage3 quality-vs-cost join integration"
```

## Notes for Execution
- Use @superpowers/test-driven-development while implementing each chunk.
- Use @superpowers/verification-before-completion before final success claims.
- Keep diffs scoped; do not expand benchmark matrix.
- Preserve backward-compatible benchmark keys.
- Keep embedded `results["quality_vs_cost"]` as single source of truth.
