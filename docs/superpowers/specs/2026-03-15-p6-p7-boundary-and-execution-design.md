# P6 P7 Boundary And Execution Design

## Goal

Complete the execution-side scaling integration for P6 and P7 without touching notebook presentation yet. This spec covers:

- structured boundary records for failed or preflight-skipped DQI runs
- wall-clock and peak-memory capture for new runs
- Matrix A and Matrix C execution for P6/P7
- canonical metrics/report integration

Acceptance for this spec is:

1. the canonical metrics outputs contain completed P6 rows for the tractable subset
2. the canonical metrics outputs contain P7 rows for every attempted or policy-skipped config, represented as completed or structured boundary rows
3. existing P3-P5 values remain unchanged
4. `python -m pytest tests/ -x` passes

## Current Codebase Findings

- The canonical row status field is `run_status`, not `status`.
  - [`src/matrix_runner.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/src/matrix_runner.py) emits `run_status`
  - [`scripts/make_report.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/scripts/make_report.py) normalizes `run_status`
  - [`src/report_schema.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/src/report_schema.py) validates downstream rows against `run_status`
- Matrix A and Matrix C already persist through a shared canonical write path in [`src/matrix_runner.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/src/matrix_runner.py): per-run JSON, aggregated CSV, manifest, then report generation.
- Existing row builders already centralize common fields via `_base_record(...)`, so the safest extension point is shared record construction rather than ad hoc per-branch writes.
- Notebook/report readers already tolerate `run_status != "completed"` in many places, but Stage E comparable outputs remain completed-only by design, so boundary rows must remain upstream/canonical rows rather than pretending to be quality-cost comparable rows.

## Design Principles

1. Keep `run_status` as the only canonical run-status field.
2. Boundary rows are first-class canonical rows, not side tables.
3. Add optional fields only; do not rename or remove existing identity/quality/resource fields.
4. Use the existing persistence path. The new helper constructs rows; Matrix A/C continue to write them through the current canonical writer.
5. Preserve grouping identity across completed and boundary rows so downstream aggregation stays per-config.

## Canonical Status Contract

For spec A, newly produced Matrix A/C rows must use only this controlled `run_status` vocabulary:

- `completed`
- `unavailable`
- `failed`

Boundary rows are any rows with `run_status != "completed"`.

No new ad hoc status literals such as `skipped`, `error`, or `oom` may be introduced in Matrix A/C outputs.

## Boundary Record Contract

Add a shared helper, preferably in a focused module such as [`src/boundary.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/src/boundary.py) or a tightly scoped section near the top of [`src/matrix_runner.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/src/matrix_runner.py), with these responsibilities:

1. wrap DQI execution with timing and memory capture
2. classify failure mode into a controlled `boundary_reason`
3. return a fully formed canonical record, preserving the same identifying keys as completed rows

### Required identity/grouping fields on boundary rows

Boundary rows must preserve at least:

- `run_id`
- `instance_id`
- `encoding`
- `decoder`
- `alpha_mode`
- `execution_mode`
- `ell`

and must continue to carry the existing common record identity fields already used by Matrix A/C.

### New optional fields to add to newly produced rows

These fields should be present on all newly emitted Matrix A/C rows where feasible, with `None` when unavailable:

- `attempted_execution: bool | None`
- `boundary_reason: str | None`
- `error_type: str | None`
- `error_message: str | None`
- `estimated_qubits: int | None`
- `estimated_memory_gb: float | None`
- `wall_clock_s: float | None`
- `peak_memory_mb: float | None`

### Controlled `boundary_reason` vocabulary

`boundary_reason` must come from a small controlled set:

- `qubit_count_exceeded`
- `statevector_memory_exceeded`
- `timeout`
- `runtime_exception`

### Error normalization

- `error_type` is the exception class name
- `error_message` is a truncated string payload suitable for artifacts and tables
- `boundary_reason` remains coarse and machine-friendly, separate from raw exception text

### Null quality fields

Boundary rows must preserve the usual quality keys and set them to `None` when unavailable, including at minimum:

- `best_sampled_F`
- `top1_regret`
- `topk_regret`
- `postselection_success`
- `wall_clock_s`
- `peak_memory_mb`

This keeps schema shape stable for filtering and grouping.

## Execution Attempt Semantics

`attempted_execution` must be interpreted strictly:

- `attempted_execution=False` only when the runner decides not to enter the execution path, for example because analytic resource estimates exceed the policy threshold
- `attempted_execution=True` whenever the runner actually enters the execution path, even if failure occurs immediately afterward

This distinction is required for later tractability analysis.

## Timing And Memory Contract

Wrap execution with `time.perf_counter()` and `tracemalloc`.

### Timing precedence

For new rows:

- always populate `wall_clock_s`
- completed rows may also populate legacy `runtime_sec` for compatibility
- downstream analysis should prefer `wall_clock_s`, then fall back to `runtime_sec`
- boundary rows do not need `runtime_sec`

### Resource estimates on success and failure

Both completed and boundary rows should carry:

- `estimated_qubits`
- `estimated_memory_gb`

This enables later comparison of predicted cost versus observed tractability without additional joins.

## Matrix A Execution Policy

Matrix A remains the orchestrator for pricing-family execution. P6/P7 must use the same canonical write path as existing P-family runs.

### P6 policy

- Execute in ascending tractability order by `ell`
- Attempt `ell=1`, then `ell=2`, then `ell=3`
- If a higher-ell config fails, persist a boundary row and continue; do not let it block lower-ell completed rows
- The acceptance threshold for P6 is at least one completed exact subset, expected to include `ell=1`

### P7 policy

- Compute analytic resource estimates before any statevector attempt
- If `estimated_qubits > 30`, emit a boundary row immediately with:
  - `run_status="unavailable"`
  - `attempted_execution=False`
  - `boundary_reason="qubit_count_exceeded"`
- Otherwise attempt exact execution under the timing/memory wrapper
- Classify OOM-like failures as:
  - `run_status="unavailable"`
  - `attempted_execution=True`
  - `boundary_reason="statevector_memory_exceeded"`
- Classify unexpected execution failures as:
  - `run_status="failed"`
  - `attempted_execution=True`
  - `boundary_reason="runtime_exception"`

## Matrix C Execution Policy

Matrix C and decoder-study pricing rows must use the same boundary/timing helper so pricing-family semantics remain uniform across matrices.

- Extend only the pricing-family execution branches
- S-family behavior should remain semantically unchanged apart from harmless new optional fields on newly produced rows
- Boundary rows from Matrix C must follow the same `run_status`, `attempted_execution`, and `boundary_reason` rules as Matrix A

## Persistence And Atomicity

- Keep actual persistence in the existing canonical writer used by Matrix A/C
- The helper returns complete row payloads only
- No second persistence path may be introduced for boundary rows
- This preserves current artifact atomicity and avoids split-brain outputs

## Verification Strategy

Add or extend tests for:

1. boundary-row helper classification and field population
2. Matrix A/C row-shape stability when `run_status` is `unavailable` or `failed`
3. report/metrics ingestion tolerating the new optional boundary/resource/timing fields
4. preservation of P3-P5 values when P6/P7 rows are added

After execution, produce a summary over canonical outputs of the form:

```python
{
    "P3": {"completed": ..., "unavailable": ..., "failed": ...},
    "P4": {"completed": ..., "unavailable": ..., "failed": ...},
    "P5": {"completed": ..., "unavailable": ..., "failed": ...},
    "P6": {"completed": ..., "unavailable": ..., "failed": ...},
    "P7": {"completed": ..., "unavailable": ..., "failed": ...},
}
```

This summary is part of the acceptance evidence for spec A.

## Out Of Scope

This spec does not cover notebook changes. In particular, it does not cover:

- Notebook 08 expansion
- Notebook 13 creation
- notebook wording, fallback labels, or verdict phrasing

Those belong to a separate spec that will be designed against the frozen outputs from this execution-phase work.
