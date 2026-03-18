# Stage D Alpha Validation Design

**Date:** 2026-03-12  
**Status:** Approved for planning  
**Scope:** Design-only (no implementation in this document)

---

## 1. Goal

Close Gap 3 using exact tiny Family C evidence by running an oracle-only alpha validation stage across coherent and mixture execution modes, with explicit unsupported/failed semantics and report-derived coherent-vs-mixture comparisons.

---

## 2. Architectural Decision

Stage D is implemented as a dedicated runner and stage namespace:

- Authoritative raw rows:
  - `matrix = "matrix_b"`
  - `stage = "alpha_validation"`
  - `family = "C"`
  - `decoder = "oracle"`
- Raw outputs are append-only per-run JSON files under:
  - `results/matrix_b/runs/<run_id>.json`
- Compatibility outputs remain:
  - `results/tables/matrix_b_summary.csv`
  - `results/reports/matrix_b_report.json`
- Compatibility outputs are regenerated from run artifacts in reporting, not directly written by Stage D runner.

This preserves legacy Matrix B compatibility while making Stage D a first-class experiment stage.

---

## 3. Sweep Definition

Stage D sweep is a strict planned grid over existing Family C lineage:

- `instance_id`: from coherent manifest (current set includes `C1/C2/C3`)
- `alpha_mode`: `uniform`, `paper`, `heuristic`
- `execution_mode`: `coherent`, `mixture`
- `ell`: configured tiny sweep values
- `m_active`: configured tiny sweep values, applied as deterministic prefix truncation of `(B, v)` rows

Fixed constraints:

- `n_bits` remains fixed per instance
- `decoder` is always `oracle`
- coherent path is exact-only (no fallback approximation)

---

## 4. Component Changes

### 4.1 `src/alpha_theory_validation.py`

Create a dedicated Stage D runner:

- Entry point: `run_alpha_validation(...)`
- Responsibilities:
  - materialize full planned grid first
  - execute each planned slot (or emit explicit unsupported/failed row)
  - validate each row against Stage D contract before write
  - append one JSON file per slot to `results/matrix_b/runs/`
- Runner must never write summary CSV/report JSON directly.

### 4.2 `src/coherent_reference.py`

Extend exact reference layer with:

- `supports_tiny_coherent(instance, ell, m_active) -> (bool, reason_code)`
- `evaluate_coherent_exact(...)`
- `evaluate_mixture_exact(...)`

Rules:

- coherent support checks must return stable machine-readable reason codes
- coherent evaluator must not silently approximate
- unsupported coherent cases return structured unsupported status payloads

### 4.3 `src/report_schema.py`

Add central Stage D schema validation:

- validator for `stage == "alpha_validation"` rows
- enforce required identity/status/raw-metric fields
- hard-fail for `decoder != "oracle"` in Stage D rows

### 4.4 `scripts/make_report.py`

Reporting-layer responsibilities:

- ingest Matrix B raw rows (legacy + Stage D)
- enforce Stage D guardrail:
  - `stage == "alpha_validation"` implies:
    - `matrix == "matrix_b"`
    - `family == "C"`
    - `decoder == "oracle"`
- derive comparison fields in report layer only:
  - `coherent_vs_mixture_delta_best_F`
  - `coherent_vs_mixture_delta_best_top_G_sampled_F`
  - `coherent_vs_mixture_delta_postselection_success`
  - `coherent_vs_mixture_distribution_distance`
  - `alpha_ranking_within_instance`
- regenerate compatibility outputs:
  - `results/tables/matrix_b_summary.csv`
  - `results/reports/matrix_b_report.json`

### 4.5 `notebooks/10_alpha_finite_size_validation.ipynb`

Keep notebook reader-only and stage-aware:

- filter:
  - `matrix == "matrix_b"`
  - `stage == "alpha_validation"`
  - `family == "C"`
- remove selection dependency on literal `["C1","C2","C3"]`
- explicitly render:
  - `not_applicable`
  - `failed`
  instead of silently dropping rows

---

## 5. Stage D Raw Row Contract

### 5.1 Required identity fields

- `run_id`
- `matrix`
- `stage`
- `family`
- `instance_id`
- `encoding`
- `decoder`
- `alpha_mode`
- `execution_mode`
- `ell`
- `m_active`
- `trial_seed`
- `canonical_trial_seed`
- `seed` (legacy alias)

### 5.2 Required status fields

- `run_status` in `{completed, failed, not_applicable}`
- `status_reason` (stable code string)
- `in_experiment_matrix` (bool)
- `run_error` (nullable string, always present key)
- `coherent_supported` (bool where relevant)

### 5.3 Required raw metrics

- `success_probability`
- `best_F`
- `best_top_G_sampled_F`
- `postselection_success`

### 5.4 Required invariants

- Stage D rows must satisfy:
  - `matrix == "matrix_b"`
  - `stage == "alpha_validation"`
  - `family == "C"`
  - `decoder == "oracle"`
- `run_error` must be present as nullable string in all rows.
- `canonical_trial_seed` is required on raw rows.

---

## 6. Status Semantics

For planned Stage D slots:

- successful exact run:
  - `run_status = "completed"`
- coherent exact unsupported (in-sweep):
  - `run_status = "not_applicable"`
  - `status_reason = "coherent_exact_limit_exceeded"` (or other stable reason code)
  - `in_experiment_matrix = true`
- execution failure:
  - `run_status = "failed"`
  - `status_reason = "execution_failed"` (or narrower stable code)
  - `run_error` populated

No silent omission for planned slots.

---

## 7. Derived Metrics Boundary

Raw runner writes only raw authoritative metrics and status/provenance fields.

The following are report-derived only:

- `coherent_vs_mixture_delta_best_F`
- `coherent_vs_mixture_delta_best_top_G_sampled_F`
- `coherent_vs_mixture_delta_postselection_success`
- `coherent_vs_mixture_distribution_distance`
- `alpha_ranking_within_instance`

Pairing for coherent-vs-mixture derivation is same-trial and same-sweep-slot, including `canonical_trial_seed`.

---

## 8. Seed Canonicalization

Stage D standardizes on:

- canonical seed column: `trial_seed`
- required normalized identity column: `canonical_trial_seed`
- compatibility alias: `seed`

Reporting dedupe/identity must use canonical trial seed, not legacy `seed`.

---

## 9. Test Gates

Minimum required tests:

1. Stage D schema contract test (`alpha_validation` row validator).
2. Stage D guardrail test (`decoder == "oracle"` hard requirement).
3. Unsupported coherent planned slot emits `not_applicable` row.
4. Report-derived fields test confirms deltas/rankings are computed in reporting layer, not required in raw rows.
5. Notebook-facing stage filter test (`matrix_b + alpha_validation + family C`) with explicit visibility of `not_applicable` and `failed`.
6. Append-only JSON smoke test for successful/unsupported/failed Stage D row serialization.

---

## 10. Acceptance Criteria

Stage D is accepted when:

1. `run_alpha_validation(...)` emits authoritative per-slot raw JSON rows under `results/matrix_b/runs/`.
2. Every Stage D row contains required identity/status/raw-metric fields including `encoding` and `canonical_trial_seed`.
3. Stage D rows satisfy guardrails (`matrix_b`, `alpha_validation`, `family=C`, `decoder=oracle`).
4. Unsupported coherent exact cases are explicit `not_applicable` rows (not missing).
5. Reporting regenerates Matrix B compatibility artifacts and derives coherent-vs-mixture deltas and alpha rankings centrally.
6. Notebook 10 is stage-aware and no longer tied to literal instance-ID filtering.

---

## 11. Non-Goals

- No broad refactor of legacy Matrix B benchmark logic.
- No new approximation path for coherent mode.
- No notebook-side recomputation of authoritative Stage D metrics.

