# Stage D Alpha Validation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Stage D as an oracle-only, exact tiny Family C alpha validation stage that emits authoritative raw runs and report-derived coherent-vs-mixture comparisons.

**Architecture:** Add a dedicated Stage D runner (`src/alpha_theory_validation.py`) that materializes a strict planned grid and writes append-only per-run JSON rows for `matrix_b + stage=alpha_validation`. Extend coherent exact evaluators in `src/coherent_reference.py` with stable support checks and exact evaluators. Keep all deltas/rankings report-derived in `scripts/make_report.py`, and update Notebook 10 to stage-aware reader behavior.

**Tech Stack:** Python 3.11, NumPy, pandas, pytest, Jupyter notebooks, existing repo reporting pipeline.

---

## File Structure and Responsibilities

- Create: `src/alpha_theory_validation.py`
  - Stage D planned-grid enumeration, row construction, row validation call, append-only JSON writing.
- Create: `src/report_schema.py`
  - Central Stage D row validator (`alpha_validation` contract + hard guardrails).
- Modify: `src/coherent_reference.py`
  - Tiny coherent support checker and exact coherent/mixture evaluators.
- Modify: `scripts/make_report.py`
  - Stage D guardrails in report ingest, report-derived deltas/rankings, Matrix B compatibility regeneration.
- Modify: `notebooks/10_alpha_finite_size_validation.ipynb`
  - Stage-aware filtering and explicit `not_applicable` / `failed` visibility.
- Create: `tests/test_alpha_theory_validation.py`
  - Stage D runner, unsupported-row emission, append-only serialization smoke.
- Create: `tests/test_report_schema_stage_d.py`
  - Stage D schema contract and oracle guardrail tests.
- Modify: `tests/test_make_report.py`
  - Stage D report-derivation and guardrail tests.
- Create/Modify: notebook tests/helpers only if existing notebook test harness already present.

---

## Chunk 1: Exact Evaluation Surface and Schema Contracts

### Task 1: Add Stage D schema validator

**Files:**
- Create: `src/report_schema.py`
- Test: `tests/test_report_schema_stage_d.py`

- [ ] **Step 1: Write the failing schema tests**

```python
def test_stage_d_requires_oracle_decoder():
    row = make_valid_stage_d_row()
    row["decoder"] = "bp1"
    with pytest.raises(ValueError, match="decoder"):
        validate_alpha_validation_row(row)

def test_stage_d_requires_encoding_and_canonical_trial_seed():
    row = make_valid_stage_d_row()
    row.pop("encoding")
    with pytest.raises(ValueError, match="encoding"):
        validate_alpha_validation_row(row)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_report_schema_stage_d.py -v`  
Expected: FAIL (missing module/function).

- [ ] **Step 3: Implement minimal validator**

```python
def validate_alpha_validation_row(row: dict[str, Any]) -> None:
    # required fields, types, enums, and Stage D guardrails
```

- [ ] **Step 4: Re-run tests**

Run: `pytest tests/test_report_schema_stage_d.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/report_schema.py tests/test_report_schema_stage_d.py
git commit -m "feat(stage-d): add alpha-validation row schema validator"
```

### Task 2: Extend exact coherent reference helpers

**Files:**
- Modify: `src/coherent_reference.py`
- Test: `tests/test_alpha_theory_validation.py`

- [ ] **Step 1: Write failing tests for support guard and exact evaluators**

```python
def test_supports_tiny_coherent_returns_machine_reason():
    ok, reason = supports_tiny_coherent(instance=fixture_c1(), ell=9, m_active=99)
    assert ok is False
    assert reason in {"coherent_exact_limit_exceeded", "m_active_out_of_bounds", "ell_out_of_bounds"}

def test_evaluate_coherent_exact_tiny_case_returns_raw_metrics():
    out = evaluate_coherent_exact(instance=fixture_c1(), ell=1, m_active=4, alpha_mode="paper", trial_seed=42)
    assert "best_F" in out
    assert "success_probability" in out
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_alpha_theory_validation.py -k "supports_tiny_coherent or evaluate_coherent_exact" -v`  
Expected: FAIL.

- [ ] **Step 3: Implement support and exact evaluators**

```python
def supports_tiny_coherent(instance: dict[str, Any], ell: int, m_active: int) -> tuple[bool, str]:
    ...

def evaluate_coherent_exact(...)-> dict[str, Any]:
    ...

def evaluate_mixture_exact(...)-> dict[str, Any]:
    ...
```

- [ ] **Step 4: Re-run targeted tests**

Run: `pytest tests/test_alpha_theory_validation.py -k "supports_tiny_coherent or evaluate_coherent_exact" -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/coherent_reference.py tests/test_alpha_theory_validation.py
git commit -m "feat(stage-d): add tiny coherent support guard and exact evaluators"
```

---

## Chunk 2: Stage D Planned-Grid Runner and Raw Artifact Emission

### Task 3: Implement strict planned-grid Stage D runner

**Files:**
- Create: `src/alpha_theory_validation.py`
- Test: `tests/test_alpha_theory_validation.py`

- [ ] **Step 1: Add failing runner tests**

```python
def test_runner_emits_one_row_per_planned_slot(tmp_path):
    out = run_alpha_validation(output_root=tmp_path, instance_ids=["C1"], ell_values=[1], m_active_values=[2], trial_seed=42)
    assert out["planned_count"] == out["written_count"]

def test_stage_d_rows_are_tagged_and_oracle_only(tmp_path):
    out = run_alpha_validation(output_root=tmp_path, instance_ids=["C1"], ell_values=[1], m_active_values=[2], trial_seed=42)
    row = json.loads(Path(out["run_files"][0]).read_text())
    assert row["matrix"] == "matrix_b"
    assert row["stage"] == "alpha_validation"
    assert row["family"] == "C"
    assert row["decoder"] == "oracle"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_alpha_theory_validation.py -k "runner_emits_one_row or oracle_only" -v`  
Expected: FAIL.

- [ ] **Step 3: Implement `run_alpha_validation(...)`**

```python
def run_alpha_validation(...):
    # enumerate planned grid
    # evaluate coherent/mixture exact or emit not_applicable/failed
    # validate row via validate_alpha_validation_row(...)
    # append json per run_id
```

- [ ] **Step 4: Re-run targeted tests**

Run: `pytest tests/test_alpha_theory_validation.py -k "runner_emits_one_row or oracle_only" -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alpha_theory_validation.py tests/test_alpha_theory_validation.py
git commit -m "feat(stage-d): add strict planned-grid alpha-validation runner"
```

### Task 4: Add append-only serialization smoke for unsupported/failed rows

**Files:**
- Modify: `tests/test_alpha_theory_validation.py`

- [ ] **Step 1: Write failing smoke test**

```python
def test_append_only_json_serializes_completed_failed_not_applicable(tmp_path):
    out = run_alpha_validation(...)
    statuses = [json.loads(Path(p).read_text())["run_status"] for p in out["run_files"]]
    assert "completed" in statuses
    assert "not_applicable" in statuses
    # failed may require injected fault fixture
```

- [ ] **Step 2: Run smoke test**

Run: `pytest tests/test_alpha_theory_validation.py -k "append_only_json_serializes" -v`  
Expected: FAIL initially, then PASS after fixture/injection handling.

- [ ] **Step 3: Add minimal deterministic failure injection path (test-only hook) if needed**

```python
# Example: pass a controlled bad combo or monkeypatch evaluator to raise
```

- [ ] **Step 4: Re-run smoke test**

Run: `pytest tests/test_alpha_theory_validation.py -k "append_only_json_serializes" -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_alpha_theory_validation.py
git commit -m "test(stage-d): add append-only serialization smoke for all status classes"
```

---

## Chunk 3: Reporting Derivations and Guardrails

### Task 5: Add Stage D guardrails in report ingest

**Files:**
- Modify: `scripts/make_report.py`
- Modify: `tests/test_make_report.py`

- [ ] **Step 1: Add failing report guardrail tests**

```python
def test_make_report_rejects_stage_d_non_oracle_rows(tmp_path):
    # stage=alpha_validation row with decoder=bp1
    # expect quarantine/rejection warning or hard failure per contract
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_make_report.py -k "stage_d_non_oracle" -v`  
Expected: FAIL.

- [ ] **Step 3: Implement Stage D guardrail in make_report ingest path**

```python
if stage == "alpha_validation":
    assert matrix == "matrix_b" and family == "C" and decoder == "oracle"
```

- [ ] **Step 4: Re-run tests**

Run: `pytest tests/test_make_report.py -k "stage_d_non_oracle" -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_report.py tests/test_make_report.py
git commit -m "feat(report): enforce Stage D matrix/family/decoder guardrails"
```

### Task 6: Derive deltas and alpha ranking in reporting layer

**Files:**
- Modify: `scripts/make_report.py`
- Modify: `tests/test_make_report.py`

- [ ] **Step 1: Add failing derivation tests**

```python
def test_stage_d_deltas_are_report_derived(tmp_path):
    # raw rows omit coherent_vs_mixture_delta_*
    # after make_report, summary/report include derived deltas when pair exists

def test_stage_d_alpha_ranking_within_instance_is_report_derived(tmp_path):
    ...
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_make_report.py -k "stage_d_deltas or alpha_ranking" -v`  
Expected: FAIL.

- [ ] **Step 3: Implement grouped derivations**

```python
# group by instance_id, encoding, ell, m_active, alpha_mode, canonical_trial_seed
# compute coherent-minus-mixture deltas and ranking
```

- [ ] **Step 4: Re-run targeted tests**

Run: `pytest tests/test_make_report.py -k "stage_d_deltas or alpha_ranking" -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_report.py tests/test_make_report.py
git commit -m "feat(report): derive Stage D coherent-mixture deltas and alpha rankings"
```

---

## Chunk 4: Notebook 10 Reader-Only Stage-Aware Update

### Task 7: Update Notebook 10 filtering and unavailable handling

**Files:**
- Modify: `notebooks/10_alpha_finite_size_validation.ipynb`
- (Optional) Modify: `notebooks/_helpers.py`

- [ ] **Step 1: Patch selection logic to stage-aware filters**

```python
focus = metrics[(metrics["matrix"] == "matrix_b")
                & (metrics["stage"] == "alpha_validation")
                & (metrics["family"] == "C")].copy()
```

- [ ] **Step 2: Ensure unsupported/failed panels are explicit**

```python
not_applicable = focus[focus["run_status"] == "not_applicable"]
failed = focus[focus["run_status"] == "failed"]
```

- [ ] **Step 3: Execute notebook**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/10_alpha_finite_size_validation.ipynb`  
Expected: PASS with executed outputs.

- [ ] **Step 4: Add/adjust notebook-facing test (or helper test)**

Run: `pytest tests -k "alpha_validation and notebook10" -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/10_alpha_finite_size_validation.ipynb notebooks/_helpers.py tests
git commit -m "feat(notebook10): consume Stage D rows via matrix/stage/family filters"
```

---

## Chunk 5: End-to-End Verification and Artifacts

### Task 8: End-to-end Stage D smoke execution

**Files:**
- Modify only if failures require minimal fixes.

- [ ] **Step 1: Run Stage D runner smoke**

Run:
`python -c "from src.alpha_theory_validation import run_alpha_validation; print(run_alpha_validation(output_root='results', instance_ids=['C1'], ell_values=[1], m_active_values=[2], trial_seed=42, smoke=True))"`  
Expected: run files written under `results/matrix_b/runs`.

- [ ] **Step 2: Regenerate reporting outputs**

Run: `python scripts/make_report.py --results-root results --matrices matrix_a,matrix_b,matrix_c`  
Expected: updated `matrix_b_summary.csv` and `matrix_b_report.json` with Stage D rows.

- [ ] **Step 3: Validate Stage D presence in compatibility outputs**

Run:
`python - <<'PY'\nimport pandas as pd\nx=pd.read_csv('results/tables/matrix_b_summary.csv')\ny=x[(x['matrix']=='matrix_b')&(x['stage']=='alpha_validation')&(x['family']=='C')]\nprint(len(y), sorted(y['run_status'].dropna().unique()))\nPY`
Expected: non-zero rows; includes completed and/or not_applicable statuses.

- [ ] **Step 4: Run full targeted test set**

Run:
`pytest tests/test_report_schema_stage_d.py tests/test_alpha_theory_validation.py tests/test_make_report.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src scripts notebooks tests docs
git commit -m "feat(stage-d): implement alpha-validation exact tiny pipeline and reporting integration"
```

---

## Notes

- Derived fields must not be required in raw Stage D rows.
- `run_error` must exist as a nullable string on all Stage D rows.
- Keep old Matrix B rows intact; compatibility views are projections over mixed legacy + Stage D rows.
- If repository is not a git worktree in the execution harness, skip commit commands and report that limitation explicitly.

