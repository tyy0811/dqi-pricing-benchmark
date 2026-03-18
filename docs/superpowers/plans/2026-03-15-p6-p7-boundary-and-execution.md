# P6 P7 Boundary And Execution Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add canonical boundary-row handling, timing/memory capture, and exact Matrix A/C execution support for P6/P7 so the metrics pipeline records completed P6 rows and structured P7 boundary rows without changing frozen P3-P5 values.

**Architecture:** Introduce one focused boundary/timing helper used by Matrix A and Matrix C pricing-family execution branches. Keep `run_status` canonical, preserve the existing persistence/report-generation path, and extend row shape only with optional execution-boundary fields so downstream metrics/report builders remain compatible.

**Tech Stack:** Python, pytest, NumPy, tracemalloc, existing DQI pipeline/statevector modules, existing matrix/report schema stack.

---

## Chunk 1: Boundary Record Contract And Runner Row Shape

### Task 1: Add failing tests for canonical boundary-row construction

**Files:**
- Create: `tests/test_boundary.py`
- Read: `docs/superpowers/specs/2026-03-15-p6-p7-boundary-and-execution-design.md`
- Read: `src/matrix_runner.py`

- [ ] **Step 1: Write the failing test for preflight boundary rows**

Add a test that expects a helper to build a row with:
- `run_status == "unavailable"`
- `attempted_execution is False`
- `boundary_reason == "qubit_count_exceeded"`
- preserved grouping keys including `run_id`, `instance_id`, `encoding`, `decoder`, `alpha_mode`, `execution_mode`, `ell`
- null quality metrics and populated `estimated_qubits`

Example:

```python
def test_build_boundary_record_for_preflight_skip():
    row = build_boundary_record(
        base_record=base,
        run_status="unavailable",
        boundary_reason="qubit_count_exceeded",
        attempted_execution=False,
        estimated_qubits=31,
        estimated_memory_gb=16.0,
        error_type=None,
        error_message=None,
        wall_clock_s=0.0,
        peak_memory_mb=0.0,
    )
    assert row["run_status"] == "unavailable"
    assert row["attempted_execution"] is False
    assert row["boundary_reason"] == "qubit_count_exceeded"
    assert row["estimated_qubits"] == 31
    assert row["best_sampled_F"] is None
```

- [ ] **Step 2: Write the failing test for attempted-execution failures**

Add a test asserting:
- `attempted_execution is True`
- `boundary_reason == "statevector_memory_exceeded"` or `"runtime_exception"`
- `error_type` is the exception class name
- `error_message` is truncated
- `wall_clock_s` and `peak_memory_mb` are preserved

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_boundary.py -v`
Expected: FAIL because the boundary helper module/functions do not exist.

- [ ] **Step 4: Write the minimal helper module**

Create a focused helper module, recommended path:
- `src/boundary.py`

Implement minimal functions:

```python
def classify_boundary_exception(exc: BaseException) -> tuple[str, str, str]:
    ...

def build_boundary_record(
    *,
    base_record: dict[str, Any],
    run_status: str,
    boundary_reason: str,
    attempted_execution: bool,
    estimated_qubits: int | None,
    estimated_memory_gb: float | None,
    error_type: str | None,
    error_message: str | None,
    wall_clock_s: float | None,
    peak_memory_mb: float | None,
) -> dict[str, Any]:
    ...
```

Requirements:
- controlled `run_status` vocabulary: `completed`, `unavailable`, `failed`
- controlled `boundary_reason` vocabulary: `qubit_count_exceeded`, `statevector_memory_exceeded`, `timeout`, `runtime_exception`
- preserve grouping fields
- keep quality fields present and `None`

- [ ] **Step 5: Re-run the tests**

Run: `python3 -m pytest tests/test_boundary.py -v`
Expected: PASS.

### Task 2: Add failing tests for row-shape stability in matrix records

**Files:**
- Modify: `tests/test_make_report.py`
- Modify: `tests/test_stage_e_quality_cost_builder.py`
- Read: `src/matrix_runner.py`
- Read: `src/report_schema.py`

- [ ] **Step 1: Write the failing report-ingestion test**

Extend `tests/test_make_report.py` with a minimal Matrix A or Matrix C row containing:
- `run_status="unavailable"`
- `boundary_reason`
- `attempted_execution`
- `estimated_qubits`
- `estimated_memory_gb`
- `wall_clock_s`
- `peak_memory_mb`

Assert `run_make_report(...)` completes and preserves `run_status_norm == "unavailable"`.

- [ ] **Step 2: Write the failing Stage E compatibility test**

Extend `tests/test_stage_e_quality_cost_builder.py` to prove upstream rows with new optional fields do not break Stage E quality-cost table building. Boundary rows should remain unavailable upstream and should not be admitted as comparable quality-cost master rows.

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_make_report.py tests/test_stage_e_quality_cost_builder.py -k "boundary or unavailable" -v`
Expected: FAIL if current row normalization or fixture assumptions reject the new optional fields or statuses.

- [ ] **Step 4: Implement the minimal row-shape compatibility updates**

Touch only the smallest necessary surfaces:
- `src/matrix_runner.py`
- `src/report_schema.py` only if validators reject optional fields in canonical rows or Stage E ingest paths need harmless pass-through

Do not broaden Stage E comparable semantics. Keep boundary rows outside completed-only comparable sets.

- [ ] **Step 5: Re-run the tests**

Run: `python3 -m pytest tests/test_make_report.py tests/test_stage_e_quality_cost_builder.py -k "boundary or unavailable" -v`
Expected: PASS.

## Chunk 2: Timing And Memory Capture

### Task 3: Add failing tests for execution capture helper behavior

**Files:**
- Modify: `tests/test_boundary.py`
- Modify: `src/boundary.py`

- [ ] **Step 1: Write the failing success-path capture test**

Add a test for an execution wrapper like:

```python
def test_capture_execution_metrics_success():
    def _work():
        return {"ok": True}
    out = capture_execution_metrics(_work)
    assert out.result == {"ok": True}
    assert out.wall_clock_s >= 0.0
    assert out.peak_memory_mb >= 0.0
    assert out.error is None
```

- [ ] **Step 2: Write the failing failure-path capture test**

Add a test where the wrapped callable raises `MemoryError` or `RuntimeError`, asserting:
- the exception is returned, not swallowed
- `wall_clock_s` is still populated
- `peak_memory_mb` is still populated

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_boundary.py -k "capture_execution_metrics" -v`
Expected: FAIL because the timing/memory wrapper is not implemented.

- [ ] **Step 4: Implement the minimal capture wrapper**

Extend `src/boundary.py` with:

```python
@dataclass
class ExecutionCapture:
    result: Any | None
    error: BaseException | None
    wall_clock_s: float
    peak_memory_mb: float | None

def capture_execution_metrics(fn: Callable[[], Any]) -> ExecutionCapture:
    ...
```

Requirements:
- use `time.perf_counter()`
- use `tracemalloc.start()` / `get_traced_memory()` / `stop()`
- always stop tracing

- [ ] **Step 5: Re-run the tests**

Run: `python3 -m pytest tests/test_boundary.py -k "capture_execution_metrics" -v`
Expected: PASS.

## Chunk 3: Matrix A Integration For P6 And P7

### Task 4: Add failing tests for Matrix A completed-vs-boundary behavior

**Files:**
- Modify: `tests/test_make_report.py`
- Modify: `tests/test_benchmark.py`
- Modify: `src/matrix_runner.py`
- Read: `src/resources.py`

- [ ] **Step 1: Write the failing Matrix A boundary policy test**

Add a test that simulates a P7 Matrix A slot where the estimated qubit count exceeds the threshold and asserts:
- no execution attempt is made
- the emitted row has `run_status="unavailable"`
- `attempted_execution is False`
- `boundary_reason == "qubit_count_exceeded"`

Use monkeypatching around the resource estimator or execution call to keep the test small and deterministic.

- [ ] **Step 2: Write the failing Matrix A success-path test**

Add a test that simulates a completed P6 Matrix A run and asserts new fields are present:
- `wall_clock_s`
- `peak_memory_mb`
- `estimated_qubits`
- `estimated_memory_gb`
- compatibility `runtime_sec` still populated for completed rows

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_make_report.py tests/test_benchmark.py -k "matrix_a and (boundary or wall_clock or estimated_qubits or P6 or P7)" -v`
Expected: FAIL because Matrix A does not yet use the boundary/timing helper.

- [ ] **Step 4: Integrate the helper into Matrix A**

Modify `src/matrix_runner.py`:
- extend `_base_record(...)` with new optional fields set to `None` by default
- add or call boundary/timing helper functions
- compute analytic resource estimates before statevector attempts
- enforce P7 preflight skip when `estimated_qubits > 30`
- preserve `runtime_sec` for completed rows
- populate `wall_clock_s` on all new completed rows

Do not change existing P3-P5 values beyond optional fields on newly produced rows.

- [ ] **Step 5: Re-run the tests**

Run: `python3 -m pytest tests/test_make_report.py tests/test_benchmark.py -k "matrix_a and (boundary or wall_clock or estimated_qubits or P6 or P7)" -v`
Expected: PASS.

### Task 5: Execute Matrix A for P6/P7 and verify canonical outputs

**Files:**
- Modify or Create: `results/matrix_a/runs/*.json`
- Modify or Create: `results/tables/matrix_a_summary.csv`
- Modify or Create: `results/reports/matrix_a_report.json`

- [ ] **Step 1: Run P6 Matrix A at ascending ell**

Use the canonical Matrix A entrypoint from `src/matrix_runner.py` or its CLI wrapper with `features=6` first.

Run example:

```bash
python3 -m src.matrix_runner matrix_a --features 6 --seed 42
```

If the CLI differs, use the actual supported invocation from the module.

Expected:
- completed rows for at least one tractable ell subset
- no crash on higher-ell failure

- [ ] **Step 2: Run P7 Matrix A under boundary policy**

Run the same entrypoint for `features=7`.

Expected:
- boundary rows are emitted for preflight or runtime failures
- no silent omission of attempted/policy-skipped configurations

- [ ] **Step 3: Inspect canonical Matrix A outputs**

Check:
- `results/matrix_a/runs/`
- `results/tables/matrix_a_summary.csv`
- `results/reports/matrix_a_report.json`

Confirm:
- P6 rows exist and at least one has `run_status == "completed"`
- P7 rows exist and have `run_status in {"completed", "unavailable", "failed"}`

## Chunk 4: Matrix C Integration And Decoder Study

### Task 6: Add failing tests for Matrix C boundary handling

**Files:**
- Modify: `tests/test_make_report.py`
- Modify: `tests/test_stage_e_quality_cost_builder.py`
- Modify: `src/matrix_runner.py`
- Modify: `src/decoder_study_runner.py`

- [ ] **Step 1: Write the failing Matrix C pricing-family test**

Add a test that simulates a P-family Matrix C row for P7 where execution is unavailable and asserts:
- the canonical row is still written with grouping keys
- `run_status` normalizes to `unavailable`
- `boundary_reason` and `attempted_execution` are preserved

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_make_report.py tests/test_stage_e_quality_cost_builder.py -k "matrix_c and (P7 or boundary or unavailable)" -v`
Expected: FAIL because Matrix C pricing branches do not yet use the boundary helper.

- [ ] **Step 3: Integrate the helper into Matrix C pricing execution**

Modify:
- `src/matrix_runner.py`
- `src/decoder_study_runner.py` only if the decoder-study raw emitter needs the same semantics for pricing-family rows

Requirements:
- pricing-family Matrix C rows use the same controlled statuses and optional fields
- S-family semantics stay unchanged except for harmless optional columns on new outputs

- [ ] **Step 4: Re-run the tests**

Run: `python3 -m pytest tests/test_make_report.py tests/test_stage_e_quality_cost_builder.py -k "matrix_c and (P7 or boundary or unavailable)" -v`
Expected: PASS.

### Task 7: Execute Matrix C for P6/P7 and verify canonical outputs

**Files:**
- Modify or Create: `results/matrix_c/runs/*.json`
- Modify or Create: `results/tables/matrix_c_summary.csv`
- Modify or Create: `results/reports/matrix_c_report.json`

- [ ] **Step 1: Run decoder study for P6/P7**

Use the canonical decoder-study entrypoint, for example:

```bash
python3 scripts/run_decoder_study.py --p-instances P6,P7
```

Adjust to the actual supported invocation if needed.

- [ ] **Step 2: Regenerate reports**

Run:

```bash
python3 scripts/make_report.py --results-root results --matrices matrix_a,matrix_c
```

Expected:
- report generation succeeds
- P6/P7 rows are present in canonical normalized artifacts

- [ ] **Step 3: Inspect Matrix C outputs**

Confirm:
- completed P6 rows exist if tractable
- P7 pricing rows are present as completed or boundary rows

## Chunk 5: Metrics Verification And Full Regression

### Task 8: Add failing tests for P3-P5 immutability and canonical summaries

**Files:**
- Create: `tests/test_matrix_runner_p_scaling_boundary.py`
- Modify: `src/matrix_runner.py`

- [ ] **Step 1: Write the failing summary test**

Add a small test for a helper that summarizes status counts by instance:

```python
def test_summarize_status_counts_by_instance():
    rows = [
        {"instance_id": "P6", "run_status": "completed"},
        {"instance_id": "P7", "run_status": "unavailable"},
    ]
    summary = summarize_status_counts(rows)
    assert summary["P6"]["completed"] == 1
    assert summary["P7"]["unavailable"] == 1
```

- [ ] **Step 2: Write the failing immutability check test**

If there is an existing helper or fixture path for frozen reports, add a test that P3-P5 rows are not rewritten with changed metric values when P6/P7 rows are added. If byte-for-byte testing is impractical, compare the stable quality fields only.

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_matrix_runner_p_scaling_boundary.py -v`
Expected: FAIL because the summary/helper does not exist yet.

- [ ] **Step 4: Implement the minimal summary helper**

Add a small pure function in `src/matrix_runner.py` or `src/boundary.py`:

```python
def summarize_status_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    ...
```

- [ ] **Step 5: Re-run the tests**

Run: `python3 -m pytest tests/test_matrix_runner_p_scaling_boundary.py -v`
Expected: PASS.

### Task 9: Run end-to-end verification

**Files:**
- Modify or Create: canonical results under `results/`

- [ ] **Step 1: Regenerate canonical reports after Matrix A/C runs**

Run:

```bash
python3 scripts/make_report.py --results-root results --matrices matrix_a,matrix_b,matrix_c
```

Expected: PASS with refreshed normalized outputs.

- [ ] **Step 2: Produce and inspect instance summary counts**

Run a small repo-local inspection command or script that loads canonical rows and prints:

```python
{
    "P3": {"completed": ..., "unavailable": ..., "failed": ...},
    "P4": {"completed": ..., "unavailable": ..., "failed": ...},
    "P5": {"completed": ..., "unavailable": ..., "failed": ...},
    "P6": {"completed": ..., "unavailable": ..., "failed": ...},
    "P7": {"completed": ..., "unavailable": ..., "failed": ...},
}
```

Expected:
- P6 has completed rows
- P7 has completed or boundary rows for every attempted/policy-skipped config

- [ ] **Step 3: Run the full test suite**

Run: `python3 -m pytest tests/ -x`
Expected: PASS.

- [ ] **Step 4: Spot-check acceptance criteria**

Confirm manually:
- P6 completed rows exist in canonical metrics artifacts
- P7 rows exist and are not silently missing
- P3-P5 quality values are unchanged

## Notes For Execution

- This plan assumes no notebook changes. Do not touch Notebook 08 or Notebook 13 during this phase.
- Prefer a new helper module (`src/boundary.py`) unless `src/matrix_runner.py` can absorb the logic without becoming harder to reason about.
- Preserve the existing persistence path in `src/matrix_runner.py`; boundary rows must be written by the same canonical writers as completed rows.
- This workspace snapshot does not expose git metadata, so commit steps from the generic template cannot be executed unless repository metadata becomes available during implementation.
