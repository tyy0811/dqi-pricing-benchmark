# P6/P7 Recovery And Matrix C Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore notebook/runtime contract compatibility until the full test suite is green, then add P6/P7 Matrix C canonical coverage through the existing boundary-aware canonical path.

**Architecture:** This work is split into two isolated phases. Phase 1 is a strict notebook source-contract repair pass with no canonical regeneration. Phase 2 extends only the pricing-family branch of `src/matrix_runner.py::run_matrix_c(...)`, then rebuilds canonical outputs only through `scripts/make_report.py` and verifies that P6/P7 Matrix C rows exist without changing frozen P3/P4/P5 values.

**Tech Stack:** Python 3.9, pytest, Jupyter notebook JSON, pandas, existing benchmark/report pipeline in `src/matrix_runner.py` and `scripts/make_report.py`

---

## File Map

- Modify: [`notebooks/06_resource_analysis.ipynb`](/Users/zenith/Desktop/dqi-pricing-benchmark/notebooks/06_resource_analysis.ipynb)
  - Restore exact runtime-test marker cells and missing-data placeholder behavior expected by notebook contract tests.
- Modify: [`notebooks/10_alpha_finite_size_validation.ipynb`](/Users/zenith/Desktop/dqi-pricing-benchmark/notebooks/10_alpha_finite_size_validation.ipynb)
  - Only if Phase 1 tests show additional contract drift; keep edits source-level and minimal.
- Modify: [`notebooks/12_quality_vs_cost.ipynb`](/Users/zenith/Desktop/dqi-pricing-benchmark/notebooks/12_quality_vs_cost.ipynb)
  - Only if Phase 1 tests show additional contract drift; keep edits source-level and minimal.
- Modify: [`src/matrix_runner.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/src/matrix_runner.py)
  - Authoritative Matrix C P-family generation path; extend P6/P7 canonical emission here only.
- Modify: [`scripts/make_report.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/scripts/make_report.py)
  - Only if Phase 2 canonical rebuild reveals ingestion/normalization gaps for new Matrix C pricing-family rows.
- Inspect only: [`src/decoder_study_runner.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/src/decoder_study_runner.py)
  - Do not make it a second canonical P-family Matrix C producer unless a test or ingestion contract forces parity.
- Test: [`tests/test_notebook_runtime_smoke_missing_data.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/tests/test_notebook_runtime_smoke_missing_data.py)
- Test: [`tests/test_notebook_reader_hygiene.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/tests/test_notebook_reader_hygiene.py)
- Test: [`tests/test_notebook10_alpha_validation_smoke.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/tests/test_notebook10_alpha_validation_smoke.py)
- Test: [`tests/test_notebook10_stage_d_fixed_baseline_runtime.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/tests/test_notebook10_stage_d_fixed_baseline_runtime.py)
- Test: [`tests/test_notebook12_stage_e_smoke.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/tests/test_notebook12_stage_e_smoke.py)
- Test: [`tests/test_matrix_runner_p_scaling_boundary.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/tests/test_matrix_runner_p_scaling_boundary.py)
- Test: [`tests/test_make_report.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/tests/test_make_report.py)

## Chunk 1: Phase 1 Notebook Contract Recovery

### Task 1: Restore Notebook 06 marker and placeholder contract

**Files:**
- Modify: [`notebooks/06_resource_analysis.ipynb`](/Users/zenith/Desktop/dqi-pricing-benchmark/notebooks/06_resource_analysis.ipynb)
- Test: [`tests/test_notebook_runtime_smoke_missing_data.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/tests/test_notebook_runtime_smoke_missing_data.py)

- [ ] **Step 1: Run the currently failing runtime test only**

Run:

```bash
python3 -m pytest tests/test_notebook_runtime_smoke_missing_data.py::test_notebook06_resources_master_missing_degrades_to_placeholders -q
```

Expected: FAIL because the notebook JSON is missing the exact marker `PANEL_SAME = "Structure vs decode (same-model authoritative)"`.

- [ ] **Step 2: Inspect the existing Notebook 06 code-cell structure**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("notebooks/06_resource_analysis.ipynb").read_text())
for i, cell in enumerate(payload["cells"]):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "load_resources_master" in src or "Structure vs decode" in src:
            print(f"CELL {i}")
            print(src[:2500])
            print("---")
PY
```

Expected: identify the load cell and the structure/decode placeholder cell that needs the exact marker.

- [ ] **Step 3: Restore the exact Notebook 06 contract in notebook JSON**

Use `apply_patch` to make the minimal notebook JSON/source edits required so the runtime tests can find:

```python
PANEL_SAME = "Structure vs decode (same-model authoritative)"
```

Also ensure the same cell emits unavailable panels for:

```python
unavailable_panel_table(panel="Resource estimation", ...)
unavailable_panel_table(panel="Structure vs decode (same-model authoritative)", ...)
unavailable_panel_table(panel="Structure vs decode (cross-model diagnostic)", ...)
```

Constraint: keep this source-level only; do not broadly re-execute the notebook.

- [ ] **Step 4: Re-run the specific Notebook 06 runtime test**

Run:

```bash
python3 -m pytest tests/test_notebook_runtime_smoke_missing_data.py::test_notebook06_resources_master_missing_degrades_to_placeholders -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/06_resource_analysis.ipynb
git commit -m "fix: restore notebook06 runtime contract markers"
```

If git metadata is unavailable in this workspace snapshot, record that and skip the commit without blocking the plan.

### Task 2: Audit remaining notebook contract suites without regenerating outputs

**Files:**
- Modify if needed: [`notebooks/10_alpha_finite_size_validation.ipynb`](/Users/zenith/Desktop/dqi-pricing-benchmark/notebooks/10_alpha_finite_size_validation.ipynb)
- Modify if needed: [`notebooks/12_quality_vs_cost.ipynb`](/Users/zenith/Desktop/dqi-pricing-benchmark/notebooks/12_quality_vs_cost.ipynb)
- Test: [`tests/test_notebook_runtime_smoke_missing_data.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/tests/test_notebook_runtime_smoke_missing_data.py)
- Test: [`tests/test_notebook_reader_hygiene.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/tests/test_notebook_reader_hygiene.py)
- Test: [`tests/test_notebook10_alpha_validation_smoke.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/tests/test_notebook10_alpha_validation_smoke.py)
- Test: [`tests/test_notebook10_stage_d_fixed_baseline_runtime.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/tests/test_notebook10_stage_d_fixed_baseline_runtime.py)
- Test: [`tests/test_notebook12_stage_e_smoke.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/tests/test_notebook12_stage_e_smoke.py)

- [ ] **Step 1: Run notebook-focused suites**

Run:

```bash
python3 -m pytest \
  tests/test_notebook_runtime_smoke_missing_data.py \
  tests/test_notebook_reader_hygiene.py \
  tests/test_notebook10_alpha_validation_smoke.py \
  tests/test_notebook10_stage_d_fixed_baseline_runtime.py \
  tests/test_notebook12_stage_e_smoke.py \
  -x
```

Expected: either PASS, or fail on a specific missing source-level contract.

- [ ] **Step 2: Fix only exact source-contract drift if more failures appear**

Allowed edits:

```python
# restore literal markers, helper-cell variables, fallback text, or placeholder panels
panel="Alpha comparison"
panel="Coherent coverage"
filter_mode = "primary_stage_family"
"No rows with positive gates; log-scale plot is unavailable."
```

Disallowed in this step:

```python
# no notebook-wide re-execution
# no canonical regeneration
# no scientific-claim rewrites
```

- [ ] **Step 3: Re-run the notebook-focused suites until green**

Run the same command from Step 1.

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add notebooks/10_alpha_finite_size_validation.ipynb notebooks/12_quality_vs_cost.ipynb
git commit -m "fix: restore notebook runtime contracts"
```

If no edits were needed, skip the commit.

### Task 3: Prove Phase 1 integration safety

**Files:**
- Test only: full suite

- [ ] **Step 1: Run the full test suite**

Run:

```bash
python3 -m pytest tests/ -x
```

Expected: PASS.

- [ ] **Step 2: Record that no canonical regeneration occurred**

Check:

```bash
python3 - <<'PY'
from pathlib import Path
for path in [
    Path("results/metrics_master.json"),
    Path("results/run_manifest.json"),
    Path("results/reports/matrix_a_report.json"),
    Path("results/reports/matrix_c_report.json"),
]:
    print(path, path.exists(), path.stat().st_mtime if path.exists() else None)
PY
```

Expected: no intentional regeneration as part of Phase 1; this is an audit note, not a strict timestamp assertion.

- [ ] **Step 3: Commit**

```bash
git add notebooks/06_resource_analysis.ipynb notebooks/10_alpha_finite_size_validation.ipynb notebooks/12_quality_vs_cost.ipynb
git commit -m "test: restore notebook contract compatibility"
```

If the workspace has no usable git metadata, note that explicitly and proceed.

## Chunk 2: Phase 2 Matrix C P-Family Canonical Integration

### Task 4: Verify the current authoritative Matrix C pricing-family path

**Files:**
- Inspect: [`src/matrix_runner.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/src/matrix_runner.py)
- Inspect: [`scripts/make_report.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/scripts/make_report.py)
- Inspect only: [`src/decoder_study_runner.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/src/decoder_study_runner.py)

- [ ] **Step 1: Locate the pricing-family branch in `run_matrix_c(...)`**

Run:

```bash
rg -n "include_p_family|load_scaling_instance|instance_id = f\"P|boundary_reason|attempted_execution" src/matrix_runner.py
```

Expected: confirm P-family Matrix C logic already exists and identify where P6/P7 rows should be emitted.

- [ ] **Step 2: Confirm `make_report` ingests Matrix C benchmark rows canonically**

Run:

```bash
rg -n "matrix_c|decoder_study|metrics_master|run_status" scripts/make_report.py
```

Expected: confirm canonical rebuild still flows through `run_make_report(...)` and that no second write channel is needed.

- [ ] **Step 3: Verify `decoder_study_runner.py` stays non-authoritative for this work**

Run:

```bash
rg -n "P6|P7|run_status|matrix_c/runs" src/decoder_study_runner.py
```

Expected: understand its behavior, but do not adopt it as the primary P-family Matrix C writer unless a failing test forces parity.

### Task 5: Extend Matrix C pricing-family rows for P6/P7 through `run_matrix_c(...)`

**Files:**
- Modify: [`src/matrix_runner.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/src/matrix_runner.py)
- Test: [`tests/test_matrix_runner_p_scaling_boundary.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/tests/test_matrix_runner_p_scaling_boundary.py)
- Test if needed: [`tests/test_make_report.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/tests/test_make_report.py)

- [ ] **Step 1: Write or extend a failing test for Matrix C P6/P7 presence**

Add a focused test that exercises `run_matrix_c(...)` with only P-family features `[6, 7]` and asserts:

```python
assert any(r["instance_id"] == "P6" for r in records)
assert any(r["instance_id"] == "P7" for r in records)
assert all(
    {"run_id", "instance_id", "encoding", "decoder", "alpha_mode", "execution_mode", "ell"}.issubset(r)
    for r in p_rows
)
assert any(r["run_status"] in {"completed", "unavailable", "failed"} for r in p7_rows)
```

- [ ] **Step 2: Run the focused Matrix C test to see the current failure**

Run:

```bash
python3 -m pytest tests/test_matrix_runner_p_scaling_boundary.py -k "matrix_c or P6 or P7" -v
```

Expected: FAIL if P6/P7 Matrix C pricing rows are missing or lack the expected boundary semantics.

- [ ] **Step 3: Implement the minimal `run_matrix_c(...)` fix**

Update the P-family Matrix C branch so that:

```python
# preserve grouping keys
record["run_id"]
record["instance_id"]
record["encoding"]
record["decoder"]
record["alpha_mode"]
record["execution_mode"]
record["ell"]

# preserve canonical status vocabulary
record["run_status"] in {"completed", "unavailable", "failed"}

# preserve boundary semantics
record["attempted_execution"] is False  # only for preflight skip
record["attempted_execution"] is True   # only if execution was entered
```

Do not alter S-family execution semantics beyond already-supported optional columns.

- [ ] **Step 4: Re-run the focused Matrix C test**

Run the same command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/matrix_runner.py tests/test_matrix_runner_p_scaling_boundary.py
git commit -m "feat: add p6 p7 matrix-c canonical rows"
```

If additional report-ingest tests required `scripts/make_report.py` changes, include them in the same commit only if tightly coupled.

### Task 6: Regenerate canonical outputs and verify per-instance status counts

**Files:**
- Regenerate via: [`src/matrix_runner.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/src/matrix_runner.py)
- Regenerate via: [`scripts/make_report.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/scripts/make_report.py)

- [ ] **Step 1: Run Matrix C for pricing-family features including P6/P7**

Run:

```bash
python3 -m src.matrix_runner matrix-c --p-features 3,4,5,6,7 --seed 42
```

Expected: Matrix C artifacts are regenerated through the authoritative runner. P7 configurations either complete or emit explicit boundary rows.

- [ ] **Step 2: Rebuild canonical outputs only through `make_report`**

Run:

```bash
python3 scripts/make_report.py --results-root results --matrices matrix_a,matrix_b,matrix_c
```

Expected: `results/metrics_master.json` and compatibility views are rebuilt from the canonical ingest path.

- [ ] **Step 3: Summarize per-instance canonical statuses**

Run:

```bash
python3 - <<'PY'
import json
from collections import Counter, defaultdict
rows = json.loads(open("results/metrics_master.json").read())["rows"]
by_inst = defaultdict(Counter)
for row in rows:
    inst = row.get("instance_id")
    if isinstance(inst, str) and inst.startswith("P"):
        by_inst[inst][row.get("run_status")] += 1
for inst in sorted(by_inst):
    counts = {
        "completed": by_inst[inst].get("completed", 0),
        "unavailable": by_inst[inst].get("unavailable", 0),
        "failed": by_inst[inst].get("failed", 0),
        "not_applicable": by_inst[inst].get("not_applicable", 0),
    }
    print(inst, counts)
PY
```

Expected:

- P6 now has Matrix C pricing-family rows in canonical outputs.
- P7 now has Matrix C pricing-family rows in canonical outputs.
- Any failed or preflight-rejected P7 configuration appears explicitly as `unavailable` or `failed`.

- [ ] **Step 4: Audit that P3/P4/P5 values remain unchanged**

Run:

```bash
python3 - <<'PY'
import json
from collections import Counter, defaultdict
rows = json.loads(open("results/metrics_master.json").read())["rows"]
for inst in ["P3", "P4", "P5"]:
    sub = [r for r in rows if r.get("instance_id") == inst]
    print(inst, len(sub), Counter(r.get("run_status") for r in sub))
PY
```

Expected: no unexpected status/value churn for frozen P3/P4/P5 rows.

- [ ] **Step 5: Commit**

```bash
git add results/metrics_master.json results/run_manifest.json results/reports/matrix_c_report.json results/tables/matrix_c_summary.csv
git commit -m "build: regenerate canonical matrix-c outputs for p6 p7"
```

Only include regenerated canonical outputs that are part of the established write path.

### Task 7: Re-run the full suite after canonical regeneration

**Files:**
- Test only: full suite

- [ ] **Step 1: Run the full suite**

Run:

```bash
python3 -m pytest tests/ -x
```

Expected: PASS.

- [ ] **Step 2: Prepare the final implementation summary**

Summarize:

```text
- notebook/runtime contract issue fixed
- full pytest -x status
- whether Matrix C P6/P7 rows exist in canonical outputs
- final per-instance summary:
  {instance_id: {completed, unavailable, failed, not_applicable}}
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-03-15-p6-p7-recovery-and-matrix-c.md
git commit -m "docs: add p6 p7 recovery and matrix-c plan"
```

If this repo snapshot does not support commits, state that explicitly in the handoff.
