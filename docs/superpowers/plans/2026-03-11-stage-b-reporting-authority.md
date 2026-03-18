# Stage B Reporting Authority Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a manifest-driven reporting authority that emits canonical run/metrics/resources masters and makes notebooks 06/08/09/10/11/12 reader-only against those authoritative outputs.

**Architecture:** Introduce `scripts/make_report.py` as the single report compiler from planned-slot manifests and observed matrix outputs. Use deterministic `slot_id` joins, explicit status and metric-availability normalization, and compatibility-view projection from masters. Migrate notebook loaders to master-table APIs with display-time `not_in_experiment_matrix` semantics.

**Tech Stack:** Python 3.11, pandas, json, parquet (with JSON fallback), pytest, nbformat/nbconvert.

---

## File Structure Map

- Create: `scripts/make_report.py`
  - Build manifest + master outputs + compatibility views.
- Create: `tests/test_make_report.py`
  - Validate slot grid, status semantics, duplicate resolution, availability invariants, output artifacts.
- Modify: `notebooks/_helpers.py`
  - Add master loaders + status/availability normalization helpers.
- Modify: `notebooks/06_resource_analysis.ipynb`
  - Convert to reader-only notebook using manifest/resources/metrics masters.
- Modify: `notebooks/08_paired_encoding_comparison.ipynb`
- Modify: `notebooks/09_decoder_phase_diagram.ipynb`
- Modify: `notebooks/10_alpha_finite_size_validation.ipynb`
- Modify: `notebooks/11_decision_faithfulness_dashboard.ipynb`
- Modify: `notebooks/12_quality_vs_cost.ipynb`
  - Switch direct matrix CSV loading to master loaders where reasonable.
- Modify (optional if needed): `pyproject.toml`
  - If parquet engine dependency handling needs explicit optional guidance.

## Chunk 1: Report Compiler and Core Contract

### Task 1: Add failing tests for Stage B reporting authority (@superpowers/test-driven-development)

**Files:**
- Create: `tests/test_make_report.py`
- Read: `src/matrix_runner.py` (manifest config patterns)
- Read: existing `results/` fixture files in workspace

- [ ] **Step 1: Add test for deterministic slot_id + seed normalization**

```python
def test_slot_id_is_stable_with_seed_normalization():
    # null seed -> sentinel, numeric seed canonicalized
    ...
```

- [ ] **Step 2: Add test for run_status semantics of unobserved planned slots**

```python
def test_unobserved_manifest_slot_defaults_to_planned_not_executed():
    # run_status=skipped, observed_run=False, reason_code=planned_not_executed
    ...
```

- [ ] **Step 3: Add test for availability invariants per metric M**

```python
def test_availability_columns_match_metric_availability_map_and_nullability():
    # availability_M=available => M may be non-null
    # availability_M in failed_upstream/not_applicable => M must be null
    ...
```

- [ ] **Step 4: Add duplicate resolution tie-break test**

```python
def test_duplicate_resolution_order_completed_then_timestamp_then_mtime_then_run_id():
    ...
```

- [ ] **Step 5: Add compatibility view projection test**

```python
def test_matrix_compatibility_views_are_projections_from_master_only():
    ...
```

- [ ] **Step 6: Run tests to verify failure before implementation**

Run: `python -m pytest tests/test_make_report.py -v`
Expected: FAIL on missing report compiler + helper functions.

### Task 2: Implement `scripts/make_report.py`

**Files:**
- Create: `scripts/make_report.py`
- Test: `tests/test_make_report.py`

- [ ] **Step 1: Implement CLI and core input discovery**

```python
# Example options:
# --results-root results
# --matrices matrix_a,matrix_b,matrix_c
# --prefer-parquet true/false
```

Hard-fail if missing core inputs for requested matrices:
- matrix manifests/grid definitions
- required observed source summaries/reports

- [ ] **Step 2: Implement manifest slot expansion with `slot_id`**

Identity fields:
- `stage,matrix,family,instance_id,encoding,decoder,alpha_mode,execution_mode,seed`

Seed normalization:
- missing -> `NULL`
- numeric -> canonical string

`slot_id = sha256(stage|matrix|family|instance_id|encoding|decoder|alpha_mode|execution_mode|seed=<normalized>)`

Hard-fail on duplicate slot_id.

- [ ] **Step 3: Ingest observed summaries and normalize statuses**

Canonical statuses:
- `completed`, `failed`, `skipped`

For any manifest-defined slot with no observed run:
- `run_status=skipped`
- `observed_run=False`
- `status_reason_code=planned_not_executed` (default)
- `intentionally_skipped` only with explicit evidence

- [ ] **Step 4: Implement duplicate observed-run resolution policy**

Tie-break order:
1. exact normalized identity match including canonical seed
2. prefer `completed` over non-completed
3. newest `run_timestamp_utc`
4. fallback file mtime
5. lexicographically smallest `run_id`

Preserve provenance:
- `duplicate_run_ids`
- `duplicate_observation_count`

Soft-warning on collisions.

- [ ] **Step 5: Build metrics master with flattened availability columns**

For each metric `M` used by notebooks, include:
- scalar `M`
- `availability_M`

Also include map column `metric_availability`.

Invariant:
- `availability_M=available` => `M` may be non-null
- `availability_M in {failed_upstream, not_applicable}` => `M` must be null
- `availability_M == metric_availability[M]`

- [ ] **Step 6: Build resources master with scope/provenance fields**

Include per-row:
- `resource_scope`, `resource_key`, `resource_confidence`
- `resource_status`, `resource_reason_code`, `resource_model_version`
- `observed_resources_path`, `producer`, `ingested_at_utc`

Allow repeated resource values across slots where scope is encoding-level.

- [ ] **Step 7: Emit outputs**

Primary outputs:
- `results/run_manifest.json`
- `results/metrics_master.parquet` (fallback: `results/metrics_master.json`)
- `results/resources_master.json`

Compatibility projections:
- `results/tables/matrix_a_summary.csv`
- `results/tables/matrix_b_summary.csv`
- `results/tables/matrix_c_summary.csv`
- `results/reports/matrix_a_report.json`
- `results/reports/matrix_b_report.json`
- `results/reports/matrix_c_report.json`

- [ ] **Step 8: Run focused tests**

Run: `python -m pytest tests/test_make_report.py -v`
Expected: PASS.

## Chunk 2: Notebook Helpers and Notebook Migration

### Task 3: Extend notebook helper APIs for master tables

**Files:**
- Modify: `notebooks/_helpers.py`
- Test: `tests/test_make_report.py` (or dedicated helper tests if added)

- [ ] **Step 1: Add loaders**

```python
def load_run_manifest(results_root): ...
def load_metrics_master(results_root): ...
def load_resources_master(results_root): ...
```

- [ ] **Step 2: Add display normalization helpers**

Helpers to:
- normalize canonical run statuses for notebook tables
- derive display-time `not_in_experiment_matrix` when a requested slice is absent from manifest

- [ ] **Step 3: Add helper-level invariants test (if needed)**

Run: `python -m pytest tests/test_make_report.py -k "helper or availability" -v`
Expected: PASS.

### Task 4: Rewrite Notebook 06 to reader mode

**Files:**
- Modify: `notebooks/06_resource_analysis.ipynb`

- [ ] **Step 1: Remove authoritative local generation paths**

Remove use of:
- `small_instance`
- `default_instance`
- ad hoc generated surrogate artifacts as authoritative inputs

- [ ] **Step 2: Load authoritative tables**

Use:
- `load_run_manifest(...)`
- `load_resources_master(...)`
- `load_metrics_master(...)` as needed

- [ ] **Step 3: Preserve plotting and narrative style with minimal markdown churn**

Keep reviewer-facing charts/tables, only change data source plumbing.

- [ ] **Step 4: Execute notebook 06**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/06_resource_analysis.ipynb`
Expected: PASS with reader-mode outputs.

### Task 5: Migrate notebooks 11/12, then 08/09/10

**Files:**
- Modify: `notebooks/11_decision_faithfulness_dashboard.ipynb`
- Modify: `notebooks/12_quality_vs_cost.ipynb`
- Modify: `notebooks/08_paired_encoding_comparison.ipynb`
- Modify: `notebooks/09_decoder_phase_diagram.ipynb`
- Modify: `notebooks/10_alpha_finite_size_validation.ipynb`

- [ ] **Step 1: Switch 11 and 12 first to master loaders**

Replace direct matrix table reads with `load_metrics_master(...)` and `load_run_manifest(...)` where reasonable.

- [ ] **Step 2: Switch 08/09/10 with compatibility fallback only when necessary**

- Use master tables as primary.
- If compatibility readers retained temporarily, ensure source is regenerated from masters.

- [ ] **Step 3: Ensure display semantics**

For requested slices absent from manifest:
- show `not_in_experiment_matrix` (display-only)

For present-but-unobserved/failed slots:
- reflect canonical `run_status` + availability semantics.

- [ ] **Step 4: Execute notebooks**

Run:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/08_paired_encoding_comparison.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/09_decoder_phase_diagram.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/10_alpha_finite_size_validation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/11_decision_faithfulness_dashboard.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/12_quality_vs_cost.ipynb
```
Expected: PASS with master-driven inputs.

## Chunk 3: End-to-End Verification

### Task 6: End-to-end Stage B validation (@superpowers/verification-before-completion)

**Files:**
- Verify: `results/run_manifest.json`
- Verify: `results/metrics_master.parquet` or `results/metrics_master.json`
- Verify: `results/resources_master.json`
- Verify: compatibility outputs in `results/tables/` and `results/reports/`

- [ ] **Step 1: Run report compiler**

Run: `python scripts/make_report.py --results-root results --matrices matrix_a,matrix_b,matrix_c`
Expected: all canonical outputs emitted.

- [ ] **Step 2: Validate canonical outputs parse + key invariants**

Run:
```bash
python - <<'PY'
import json
from pathlib import Path
import pandas as pd

assert Path('results/run_manifest.json').exists()
assert Path('results/resources_master.json').exists()
assert Path('results/metrics_master.parquet').exists() or Path('results/metrics_master.json').exists()

manifest=json.loads(Path('results/run_manifest.json').read_text())
slots=manifest.get('slots',[])
slot_ids=[s['slot_id'] for s in slots]
assert len(slot_ids)==len(set(slot_ids))
print('manifest slots',len(slots))
PY
```
Expected: no assertion failures.

- [ ] **Step 3: Validate notebook execution status**

Run a small checker for notebooks 06 and 08-12 ensuring no error outputs.
Expected: all executed cleanly.

- [ ] **Step 4: Final commit (if using git)**

```bash
git add scripts tests notebooks results/reports docs/superpowers/specs/2026-03-11-stage-b-reporting-authority-design.md docs/superpowers/plans/2026-03-11-stage-b-reporting-authority.md
git commit -m "feat: Stage B reporting authority with manifest-driven master tables"
```
