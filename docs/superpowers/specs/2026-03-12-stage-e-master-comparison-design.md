# Stage E Master Comparison Layer Design

**Date:** 2026-03-12  
**Status:** Approved for planning  
**Scope:** Design only; no implementation in this document.

---

## 1. Decision

Stage E uses a **hybrid comparison output** contract:

- `results/quality_cost_master.parquet|json`: completed + comparable rows only.
- `results/quality_cost_unavailable.parquet|json`: unavailable rows with explicit reason semantics.

`make_report.py` remains the single authoritative writer for both files.

Notebooks remain thin readers:

- Notebook 11: status/availability from `metrics_master`; comparable slices from `quality_cost_master`.
- Notebook 12: comparable quality-vs-cost analysis from `quality_cost_master`; optional unavailable transparency counts.

---

## 2. Authority and Pipeline

### 2.1 Single writer

`make_report.py` is the only writer for:

- `run_manifest`
- `metrics_master`
- `resources_master`
- Stage E outputs (`quality_cost_master`, `quality_cost_unavailable`)

No notebook may write authoritative comparison artifacts.

### 2.2 Stage E build point

Stage E build runs **after** `metrics_master` and `resources_master` are built.

### 2.3 Utility modules

Stage E logic is modularized under `src/`:

- `src/faithfulness.py`
- `src/verify_encoding.py`
- evolved `src/resources_compare.py`

---

## 3. Manifest and Identity Contract

### 3.1 Required manifest slot field

`run_manifest` must include a required stable field:

- `manifest_slot`

`manifest_slot` is the deterministic ordering anchor for tie-break resolution.

### 3.2 Canonical seed fallback

Before Stage E keying and uniqueness checks:

`canonical_trial_seed = canonical_trial_seed ?? trial_seed ?? seed`

If none exists, value is `null` and remains explicit.

### 3.3 Stage E base identity columns

Stage E rows carry at least:

- `run_id`
- `matrix`
- `stage`
- `instance_id`
- `encoding`
- `decoder`
- `alpha_mode`
- `execution_mode`
- `canonical_trial_seed`
- `manifest_slot`

Provenance fields retained in both outputs:

- `run_status`
- `status_reason_code`
- `status_reason`
- `observed_run`

### 3.4 Duplicate-key definition (winner selection key)

The duplicate-key grouping used before winner selection is locked to:

- `instance_id`
- `encoding`
- `decoder`
- `alpha_mode`
- `execution_mode`

---

## 4. Duplicate Resolution

### 4.1 Deterministic winner rule

For rows sharing the duplicate key, winner precedence is:

1. `run_status == "completed"`
2. `has_verified_provenance == true`
3. lower/earlier `manifest_slot`
4. lexical `run_id`

### 4.2 Collision routing

Non-winning rows are **not dropped**. They are routed to `quality_cost_unavailable` with:

- `unavailable_class = "duplicate_key"`
- stable `unavailable_reason_code`
- human `unavailable_reason`

---

## 5. Stage E Availability Semantics

### 5.1 Unavailable fields split

Unavailable rows use three fields:

- `unavailable_class` (controlled enum)
- `unavailable_reason_code` (machine-readable)
- `unavailable_reason` (human-readable)

### 5.2 Locked `unavailable_class` enum

- `duplicate_key`
- `missing_required_fields`
- `missing_quality_fields`
- `missing_resource_fields`
- `resource_not_comparable`
- `encoding_validation_failed`
- `insufficient_provenance`

---

## 6. Comparability Rules

### 6.1 Required quality fields for `quality_cost_master`

A row is ineligible for `quality_cost_master` unless all are present:

- `best_F`
- `topk_regret`
- `decision_distortion`
- `spearman_rho_F_G`
- `retained_energy_eta`

Missing any of these routes the row to unavailable with class `missing_quality_fields`.

### 6.2 Resource comparability acceptance

Only resource statuses below are acceptable for `quality_cost_master`:

- `comparable`
- `estimated_comparable`

Any other value routes to unavailable with class `resource_not_comparable`.

### 6.3 `estimated_comparable` provenance requirement

If `resource_status == "estimated_comparable"`, row must include:

- `resource_comparability_basis` in:
  - `same_estimator`
  - `same_calibration`
  - `same_method_version`

If missing or outside this set, route to unavailable as `resource_not_comparable`.

---

## 7. Stage A Validation Scope

### 7.1 Explicit opt-in flag

Paired encoding validation is controlled by one explicit field:

- `expects_paired_encoding_validation: true`

### 7.2 Scope behavior

- If flag is true, run `verify_encoding` checks for paired compatibility/provenance.
- If validation fails, route row to unavailable as `encoding_validation_failed`.
- If flag is false/missing, **do not** apply Stage A pairing checks.

No stage-name inference is used for this gate.

---

## 8. Faithfulness Computation Guardrails

`faithfulness.py` computes/fills values only when provenance is sufficient and verified for the row’s encoding/decoder/mode context.

No cross-row or incompatible-provenance recomputation is allowed.

If provenance is insufficient, route to unavailable class `insufficient_provenance`.

---

## 9. Output Schemas

### 9.1 `quality_cost_master` (completed + comparable)

Required analysis columns include:

- identity/provenance columns in Sections 3.3 and 3.1
- quality fields:
  - `best_F`, `topk_regret`, `decision_distortion`, `spearman_rho_F_G`, `retained_energy_eta`
- optional but included when present:
  - `best_top_G_sampled_F`
  - `surrogate_best_vs_true_best_sampled_gap`
  - `collision_rate`
  - `ambiguous_syndrome_count`
- canonical resource fields:
  - `qubits`, `gates`, `depth`
  - `resource_status`, `resource_confidence`, `resource_comparability_basis`

### 9.2 `quality_cost_unavailable`

Includes identity/provenance columns plus:

- `unavailable_class`
- `unavailable_reason_code`
- `unavailable_reason`

---

## 10. Notebook Contracts

### 10.1 Notebook 11 fallback contract (locked)

Source precedence:

1. `quality_cost_master` -> comparable panel(s)
2. `quality_cost_unavailable` -> optional transparency summary card
3. `metrics_master + run_manifest` -> status/availability panels only

No mixed-source comparable panel construction is allowed.

### 10.2 Notebook 12 thinness contract (locked)

Notebook 12 must not perform row-identity-changing joins/aggregations.

Forbidden in notebook cells:

- merge/join/groupby that changes row identity

Allowed:

- filtering
- ranking/sorting
- Pareto extraction
- display formatting

### 10.3 Empty-state / missing-file contract

For notebooks 11 and 12:

- missing `quality_cost_master` file -> render explicit placeholder panel; do not crash
- existing but empty `quality_cost_master` -> render empty comparable panel placeholder; do not crash
- `quality_cost_unavailable` present -> still render transparency card
- missing `quality_cost_unavailable` -> skip transparency card with explicit note

---

## 11. Tests (minimum)

1. **No duplicate joined rows unexpectedly**
   - winner selection deterministic and reproducible
   - losers are routed to unavailable with `duplicate_key`

2. **Unavailable routing is explicit**
   - missing quality/resource/provenance cases are preserved in `quality_cost_unavailable`

3. **Master eligibility partition**
   - completed + comparable rows only in `quality_cost_master`
   - all other rows in `quality_cost_unavailable`

4. **Stage A scoped validation**
   - only rows with `expects_paired_encoding_validation=true` are checked by `verify_encoding`

5. **Notebook smoke**
   - notebooks 11/12 execute with missing or empty `quality_cost_master` and render placeholders (no hard crash)

---

## 12. Acceptance Criteria

Stage E is accepted when all are true:

1. `results/quality_cost_master.*` exists with completed/comparable rows only.
2. `results/quality_cost_unavailable.*` exists with explicit unavailable semantics.
3. `make_report.py` is the sole writer for Stage E outputs.
4. Duplicate-key handling is deterministic using `manifest_slot` tie-break.
5. Stage A validation is scoped by `expects_paired_encoding_validation`.
6. Notebook 11/12 contracts hold (thin, source-safe, placeholder-safe).

