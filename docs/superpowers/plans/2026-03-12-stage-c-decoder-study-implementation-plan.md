# Stage C Decoder Study Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Stage C decoder-study execution and reporting so Notebook 09 can analyze real `decoder_study` rows for families P/S from canonical artifacts.

**Architecture:** Add two new decoders (BP+OSD-lite and tiny oracle), a dedicated append-only decoder-study runner that writes raw per-decoder run JSON under `results/matrix_c/runs/`, and extend report regeneration to infer stage/seed canonically and derive cross-decoder gains in `make_report.py`. Keep compatibility outputs in `matrix_c_*` files mixed by stage while notebook 09 filters to `stage="decoder_study"`.

**Tech Stack:** Python 3.11, NumPy, Pandas, existing `src/*` DQI pipeline modules, existing `scripts/make_report.py`, pytest, Jupyter nbconvert.

---

## File Structure and Responsibilities

### New files
- `src/decoder_bp_osd.py`
  - Deterministic BP+OSD-lite decoder with canonical `decode` and `decode_result` interface.
- `src/decoder_oracle.py`
  - Tiny-instance exact oracle decoder with explicit tiny-cap applicability semantics.
- `scripts/run_decoder_study.py`
  - Stage C runner: append-only raw JSON emission for per-decoder rows under `results/matrix_c/runs/`.
- `tests/test_decoder_bp_osd.py`
  - Safety and determinism smoke tests for BP+OSD-lite.
- `tests/test_decoder_oracle.py`
  - Oracle exactness and tiny-cap behavior tests.
- `tests/test_stage_c_decoder_study.py`
  - Stage C runner smoke and schema checks.

### Modified files
- `src/dqi_state.py`
  - Extend decoder mode resolution/build logic to support `bp_osd_lite` and `oracle` modes.
- `src/dqi_pipeline.py`
  - Permit Stage C decoder modes in pipeline validation.
- `src/resources.py`
  - Add decode model aliases for BP+OSD-lite and oracle tiny-only where needed.
- `src/structure_metrics.py`
  - Freeze and expose required numeric structure metric columns (including row/col min/mean/max).
- `scripts/make_report.py`
  - Stage inference precedence, canonical `trial_seed`, stage-aware dedupe keys, Stage C raw-run ingest, mixed-by-stage matrix_c compatibility projection, `by_stage` aggregates.
- `notebooks/09_decoder_phase_diagram.ipynb`
  - Filter to `matrix_c + decoder_study + family in {P,S}` and keep explicit status buckets.
- `notebooks/_helpers.py`
  - Optional helper for status-bucket rendering if notebook logic benefits from reuse.
- `tests/test_make_report.py`
  - Add stage precedence, `trial_seed` normalization, stage-aware dedupe, and gain-derivation tests.

---

## Chunk 1: Decoder Implementations (BP+OSD-lite and Tiny Oracle)

### Task 1: Add failing tests for BP+OSD-lite contract

**Files:**
- Create: `tests/test_decoder_bp_osd.py`
- Reference: `src/decoder_bp1.py`, `src/decoder_bruteforce.py`

- [ ] **Step 1: Write failing tests for interface and determinism**

```python
def test_bp_osd_result_contract_keys():
    ...

def test_bp_osd_deterministic_for_same_input():
    ...

def test_bp_osd_never_crashes_on_small_cases():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_decoder_bp_osd.py -v`
Expected: FAIL due to missing module/class.

- [ ] **Step 3: Implement minimal BP+OSD-lite decoder**

**Files:**
- Create: `src/decoder_bp_osd.py`

Implementation requirements:
- Start from BP1 decode attempt.
- If BP1 fails, run bounded OSD-lite candidate search.
- Deterministic ordering and tie-break rules.
- Emit required metadata fields:
  - `osd_order_cap`, `osd_candidate_budget`, `tie_break`, `fallback_flag`, `iterations_used`.

- [ ] **Step 4: Run tests and iterate until pass**

Run: `pytest tests/test_decoder_bp_osd.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_decoder_bp_osd.py src/decoder_bp_osd.py
git commit -m "feat(stage-c): add deterministic bp+osd-lite decoder"
```

### Task 2: Add failing tests for tiny oracle exactness and cap semantics

**Files:**
- Create: `tests/test_decoder_oracle.py`
- Reference: `src/decoder_bruteforce.py`

- [ ] **Step 1: Write failing tests**

```python
def test_tiny_oracle_exact_on_tiny_case():
    ...

def test_tiny_oracle_marks_non_applicable_over_cap():
    ...
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_decoder_oracle.py -v`
Expected: FAIL due to missing module/class.

- [ ] **Step 3: Implement tiny oracle decoder**

**Files:**
- Create: `src/decoder_oracle.py`

Implementation requirements:
- Exact decode on tiny instances only.
- Structured cap behavior encoded in result payload:
  - `run_status="skipped"`
  - `status_reason_code="oracle_tiny_cap_exceeded"`
  - `decoder_status="skipped_tiny_only"`
- No free-text-only applicability.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_decoder_oracle.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_decoder_oracle.py src/decoder_oracle.py
git commit -m "feat(stage-c): add tiny-only oracle decoder"
```

---

## Chunk 2: Pipeline Integration and Structure Metrics Freeze

### Task 3: Extend decoder mode wiring to include new decoders

**Files:**
- Modify: `src/dqi_state.py`
- Modify: `src/dqi_pipeline.py`
- Modify: `tests/test_decoder_integration.py`

- [ ] **Step 1: Add failing integration tests for new decoder modes**

```python
def test_statevector_accepts_bp_osd_lite_mode():
    ...

def test_statevector_accepts_oracle_mode_when_tiny():
    ...
```

- [ ] **Step 2: Run failing subset**

Run: `pytest tests/test_decoder_integration.py -k "osd or oracle" -v`
Expected: FAIL with invalid decoder mode.

- [ ] **Step 3: Implement decoder resolution updates**

Changes:
- `DECODER_MODES` include `bp_osd_lite`, `oracle`.
- `_resolve_decoder_class`/`_build_decoder` map new modes to new decoder classes.
- Keep backward compatibility with `bruteforce` and `bp1`.

- [ ] **Step 4: Re-run targeted tests**

Run: `pytest tests/test_decoder_integration.py -k "decoder_mode_selector" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dqi_state.py src/dqi_pipeline.py tests/test_decoder_integration.py
git commit -m "feat(stage-c): wire bp+osd-lite and oracle decoder modes"
```

### Task 4: Freeze structure metric schema and add smoke test

**Files:**
- Modify: `src/structure_metrics.py`
- Modify: `tests/test_family_s.py`
- Optionally modify: `tests/test_resources.py`

- [ ] **Step 1: Add failing schema test for exact numeric columns**

```python
REQUIRED = {
    "structure_row_weight_min", "structure_row_weight_mean", "structure_row_weight_max",
    "structure_col_weight_min", "structure_col_weight_mean", "structure_col_weight_max",
    "structure_rank", "structure_rank_deficiency",
    "structure_collision_rate", "structure_ambiguous_syndrome_count",
    "structure_short_cycle_proxy", "structure_distance_proxy",
}
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_family_s.py -k structure -v`
Expected: FAIL for missing/new key names.

- [ ] **Step 3: Implement minimal schema-preserving updates**

Requirements:
- Ensure all required columns exist and are numeric.
- Preserve backward-compatible aliases where already used.
- Keep null policy numeric (`None`/`NaN`, never string markers).

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_family_s.py tests/test_resources.py -k structure -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/structure_metrics.py tests/test_family_s.py tests/test_resources.py
git commit -m "feat(stage-c): freeze structure metric schema for decoder study"
```

---

## Chunk 3: Stage C Runner and Raw Artifact Emission

### Task 5: Add decoder-study runner (append-only raw runs)

**Files:**
- Create: `scripts/run_decoder_study.py`
- Optionally modify: `src/matrix_runner.py` (CLI entry wiring only, no behavior break)
- Create: `tests/test_stage_c_decoder_study.py`

- [ ] **Step 1: Write failing runner smoke test**

```python
def test_decoder_study_writes_append_only_raw_runs(tmp_path):
    ...
```

Checks:
- writes new JSON files to `results/matrix_c/runs/`
- rows include `matrix="matrix_c"`, `stage="decoder_study"`
- includes both `trial_seed` and `seed`
- raw rows do not include `gain_over_bp1` / `gain_over_bruteforce`

- [ ] **Step 2: Run test and verify fail**

Run: `pytest tests/test_stage_c_decoder_study.py -v`
Expected: FAIL (runner missing).

- [ ] **Step 3: Implement runner**

Behavior:
- evaluate P and S families with decoders: bruteforce, bp1, bp_osd_lite, oracle (tiny-only)
- capture raw authoritative metrics per decoder row:
  - `best_F`, `approximation_ratio`, `decode_success`, `postselection_success`, `topk_regret`, `runtime_sec`
- structured status fields required on every row:
  - `run_status`, `status_reason_code`, `status_reason`, `decoder_status`, `decoder_error`, `in_experiment_matrix`
- append-only writes, one file per execution unit under `results/matrix_c/runs/`

- [ ] **Step 4: Re-run smoke test**

Run: `pytest tests/test_stage_c_decoder_study.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_decoder_study.py tests/test_stage_c_decoder_study.py src/matrix_runner.py
git commit -m "feat(stage-c): add append-only decoder-study raw runner"
```

---

## Chunk 4: Reporting Regeneration and Canonical Identity

### Task 6: Make report ingestion stage-aware and seed-canonical

**Files:**
- Modify: `scripts/make_report.py`
- Modify: `tests/test_make_report.py`

- [ ] **Step 1: Add failing tests for stage precedence and trial-seed normalization**

Test coverage:
- stage precedence order:
  1. row `stage`
  2. run metadata `stage`
  3. report metadata `stage`
  4. summary metadata `stage`
  5. fallback `benchmark_matrix`
- canonical trial seed derivation:
  - `trial_seed`
  - else `seed`
  - else null
- dedupe key uses `canonical_trial_seed`, never `seed`

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_make_report.py -k "stage or trial_seed or dedupe" -v`
Expected: FAIL.

- [ ] **Step 3: Implement reporting normalization changes**

Requirements:
- stop hardcoded stage usage.
- ensure normalized outputs include explicit `matrix`, `stage`, `seed`, `trial_seed`, `canonical_trial_seed`.
- ingest raw Stage C rows from `results/matrix_c/runs/*.json`.
- preserve mixed legacy + decoder_study rows in matrix_c compatibility outputs.
- add `by_stage` aggregate block in `matrix_c_report.json`.

- [ ] **Step 4: Add report-derived gain calculation**

Rules:
- compute gains only in report layer.
- group by:
  - `matrix, stage, family, instance_id, encoding, alpha_mode, execution_mode, canonical_trial_seed`
- formulas:
  - `gain_over_bp1 = best_F(decoder) - best_F(bp1)`
  - `gain_over_bruteforce = best_F(decoder) - best_F(bruteforce)`
- if comparator missing, gain remains null.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_make_report.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/make_report.py tests/test_make_report.py
git commit -m "feat(stage-c): add stage-aware report regeneration and canonical trial seed"
```

---

## Chunk 5: Notebook 09 Stage-C Reader Update

### Task 7: Update Notebook 09 filtering and status bucket display

**Files:**
- Modify: `notebooks/09_decoder_phase_diagram.ipynb`
- Optionally modify: `notebooks/_helpers.py`

- [ ] **Step 1: Update notebook query slice**

Filter rules:
- `matrix == "matrix_c"`
- `stage == "decoder_study"`
- `family in {"P", "S"}`

- [ ] **Step 2: Add explicit unavailable bucket summary**

Display categories:
- `not_in_experiment_matrix`
- `not_applicable`
- `failed`

- [ ] **Step 3: Keep existing analytical panels**

Preserve:
- regime/coverage table
- gain-vs-collision plot
- correlation panel
- conclusion cell based on actual rows

- [ ] **Step 4: Execute notebook**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/09_decoder_phase_diagram.ipynb`
Expected: success with non-empty panels for `bruteforce`, `bp1`, `bp_osd_lite` where available.

- [ ] **Step 5: Commit**

```bash
git add notebooks/09_decoder_phase_diagram.ipynb notebooks/_helpers.py
git commit -m "feat(stage-c): retarget notebook 09 to decoder_study rows"
```

---

## Chunk 6: End-to-End Verification and Stop Criteria

### Task 8: Run Stage C flow and verify acceptance criteria

**Files:**
- No new files expected; this task validates outputs.

- [ ] **Step 1: Run decoder-study raw emitter**

Run (example):
`python scripts/run_decoder_study.py --families P,S --p-instances P3,P4,P5 --alpha-modes paper,uniform --output-root results`

Expected:
- new files in `results/matrix_c/runs/`
- each row has explicit `stage="decoder_study"`

- [ ] **Step 2: Regenerate reports from raw runs**

Run:
`python scripts/make_report.py --results-root results --matrices matrix_a,matrix_b,matrix_c`

Expected:
- regenerated `results/tables/matrix_c_summary.csv`
- regenerated `results/reports/matrix_c_report.json`
- mixed-by-stage rows preserved
- `by_stage` aggregate present

- [ ] **Step 3: Run targeted test suite**

Run:
`pytest tests/test_decoder_bp_osd.py tests/test_decoder_oracle.py tests/test_stage_c_decoder_study.py tests/test_make_report.py -v`

Expected: all PASS.

- [ ] **Step 4: Execute notebook reader verification**

Run:
`jupyter nbconvert --to notebook --execute --inplace notebooks/09_decoder_phase_diagram.ipynb`

Expected:
- no execution errors
- Stage C filtered slice non-empty
- explicit placeholder categories rendered

- [ ] **Step 5: Final acceptance checklist**

Verify all are true:
- [ ] raw Stage C rows are per-decoder authoritative
- [ ] gains are report-derived only
- [ ] canonical dedupe uses `canonical_trial_seed`
- [ ] oracle tiny-cap rows are `not_applicable` (compat-safe encoding)
- [ ] frozen structure columns are numeric and present
- [ ] notebook 09 uses `stage="decoder_study"`

- [ ] **Step 6: Commit verification artifacts (if updated)**

```bash
git add results/tables/matrix_c_summary.csv results/reports/matrix_c_report.json notebooks/09_decoder_phase_diagram.ipynb
git commit -m "chore(stage-c): regenerate matrix c reports and notebook outputs"
```

---

## Stop Criteria

Stop implementation and escalate to human if any of these occur:
- stage inference creates ambiguous conflicting stage values for same normalized identity.
- canonical trial-seed mapping creates non-deterministic dedupe results.
- oracle cap semantics conflict with existing `run_status` compatibility rules.
- Stage C runtime exceeds practical laptop bounds with default scope.

---

## Notes for Executor

- Keep all changes DRY and minimal; avoid broad refactors.
- Maintain backward compatibility for pre-stage artifacts that do not carry `stage` or `trial_seed`.
- Do not synthesize out-of-matrix rows in master tables.
- Preserve append-only behavior for raw run files; compatibility files are regenerated.
