# Stage D Fixed-Baseline Contract Design

**Date:** 2026-03-13  
**Status:** Approved for planning  
**Scope:** Matrix B / Family C / Stage D baseline selection and Notebook 10 reader contract hardening (with key-contract touchpoints in report build)

---

## 1. Goal

Lock Stage D coherent-vs-mixture analysis to one deterministic, upstream-selected canonical baseline so Notebook 10 cannot drift through notebook-side baseline inference.

---

## 2. Architecture Decision

Baseline choice is owned by `scripts/make_report.py` and persisted as row-level metadata on all Stage D rows in `metrics_master`.

Notebook 10 is a pure reader:
- it requires upstream baseline metadata,
- it filters to selected-baseline rows,
- it pairs only by `counterpart_key`,
- and it never performs baseline fallback, dynamic key inference, silent dedup, or tie-break logic.

---

## 3. Upstream Baseline Selector (Stage D)

### 3.1 Candidate baselines and policy

Policy name: `fixed_priority`

Priority order:
1. Preferred: `(encoding_norm, decoder_norm) = ("wht_exact", "oracle")`
2. Fallback: `(encoding_norm, decoder_norm) = ("ilp_derived_exact", "oracle")`

Selection threshold:
- `min_pairs = 8` (default)

Primary decision rule:
- select preferred if available and `pair_count >= min_pairs`
- else select fallback if available and `pair_count >= min_pairs`
- else select none with reason `no_baseline_meets_threshold`

Non-goal / prohibited behavior:
- no coverage-first selector
- no lexical tie-break as the primary selection rule

### 3.2 Pair-count semantics (strict)

Candidate `pair_count` is computed from Stage D rows grouped by `counterpart_key`.

A `counterpart_key` counts as one valid pair for a candidate `(encoding_norm, decoder_norm)` only when it has exactly:
- one completed row with `execution_mode_norm == "mixture"`
- one completed row with `execution_mode_norm == "coherent"`

### 3.3 Hard pre-selection invariant

Before baseline selection, `make_report` must fail loudly if Stage D candidate rows contain duplicate completed rows for the same:
- `(counterpart_key, execution_mode_norm)`

This is a contract error and must be raised before selector counts are computed.

### 3.4 Candidate diagnostics

During selector computation, both policy candidates are evaluated explicitly with diagnostics:
- `candidate_pair_count`
- `candidate_available`
- `candidate_rejection_reason`

These diagnostics may be build-log / report diagnostics, but selector output must not be implicit.

### 3.5 Selected-reason enum

`baseline_selected_reason` uses fixed codes:
- `preferred_available_meets_threshold`
- `preferred_below_threshold_used_fallback`
- `preferred_missing_used_fallback`
- `no_baseline_meets_threshold`

### 3.6 Persisted selector fields (all Stage D rows)

Persist on every Stage D row:
- `baseline_selector_policy`
- `baseline_min_pairs`
- `baseline_encoding_norm`
- `baseline_decoder_norm`
- `baseline_pair_count`
- `baseline_priority_rank`
- `baseline_selected_reason`
- `baseline_fallback_used`
- `best_available_pair_count`
- `is_stage_d_selected_baseline`

### 3.7 Explicit no-selection values

When no candidate meets threshold, persist on all Stage D rows:
- `baseline_selector_policy = "fixed_priority"`
- `baseline_min_pairs = <configured value>`
- `best_available_pair_count = <int or 0>`
- `baseline_encoding_norm = NULL`
- `baseline_decoder_norm = NULL`
- `baseline_pair_count = NULL`
- `baseline_priority_rank = NULL`
- `baseline_fallback_used = False`
- `baseline_selected_reason = "no_baseline_meets_threshold"`
- `is_stage_d_selected_baseline = False`

This distinguishes explicit no-selection from missing metadata.

### 3.8 Global-selection invariant

Selection is global for the Stage D artifact build, not per instance/sub-slice.

Build invariants:
- at most one selected baseline pair exists per Stage D artifact
- if selected, all rows where `is_stage_d_selected_baseline=True` share the same `(baseline_encoding_norm, baseline_decoder_norm)`

Hard errors:
- `multiple_selected_baselines`
- `selected_baseline_mismatch`

`selected_baseline_mismatch` definition:
- selected baseline metadata exists, but either
  - one or more rows flagged `is_stage_d_selected_baseline=True` do not match selected `(encoding_norm, decoder_norm)`, or
  - zero rows match although selected metadata claims a baseline exists.

---

## 4. Notebook 10 Fixed-Baseline Reader Contract

### 4.1 Reader-only boundary

Notebook 10 must not perform baseline choice or fallback.

Required upstream fields:
- `counterpart_key`
- `is_stage_d_selected_baseline`
- `baseline_selector_policy`
- `baseline_encoding_norm`
- `baseline_decoder_norm`
- `baseline_min_pairs`
- `baseline_pair_count`
- `best_available_pair_count`
- `baseline_selected_reason`
- `baseline_fallback_used`

If any required baseline metadata is missing:
- render explicit unavailable panel with reason code `baseline_metadata_unavailable`
- do not run notebook-side selection

### 4.2 Stage D slice and pairing domain

Notebook 10 starts from Stage D rows only:
- `matrix_norm == "matrix_b"`
- `stage_norm == "alpha_validation"`
- `family_norm == "c"`

Then:
- filter `is_stage_d_selected_baseline == True`
- restrict `execution_mode_norm in {"mixture", "coherent"}`
- pair by `counterpart_key` only
- no dynamic key inference

### 4.3 Pair construction and validation

For each `counterpart_key`, valid pair requires exactly:
- one completed mixture row
- one completed coherent row

Contract violations in notebook reader logic:
- if >1 completed row exists for same `(counterpart_key, execution_mode_norm)`, fail loudly and show offending keys
- if many-to-many pairing would occur, fail loudly and show offending keys
- no silent merge/dedup/tie-break

For non-valid keys, classify/count:
- coherent non-completed present
- coherent absent
- mixture non-completed present
- mixture absent

Also report `status_reason_code` breakdowns for incomplete/unsupported keys, especially coherent-side not-applicable/cap-related reasons.

### 4.4 Delta computation (main result)

Main Stage D deltas are computed only from valid fixed-baseline pairs.

Per delta metric independently:
- use precomputed coherent-vs-mixture delta if present and finite
- else recompute from paired raw metrics if both inputs are finite
- else mark metric unavailable for that pair with reason code

No aggregation across multiple encodings/decoders in the main Stage D result.

### 4.5 Reporting contract

Always display:
- `baseline_selector_policy`
- `baseline_encoding_norm`
- `baseline_decoder_norm`
- `baseline_min_pairs`
- `baseline_pair_count`
- `best_available_pair_count`
- `baseline_selected_reason`
- `baseline_fallback_used`

Main conclusion text must depend on:
- valid pair count
- incomplete counterpart counts
- unsupported counterpart counts
- reason-code breakdowns

If valid pair count is zero, main conclusion must explicitly declare unavailability with reason-code context.

### 4.6 Optional diagnostic block

A separate all-baseline table is allowed only if:
- labeled `diagnostic_only`
- excluded from main Stage D metrics/tables/conclusion text

---

## 5. Related Key-Contract Touchpoints

This baseline contract depends on upstream key reliability in Stage D rows:
- `run_key` uniqueness per run
- `comparison_key` stable across encoding for cross-encoding comparisons
- `counterpart_key` stable for coherent-vs-mixture pairing and independent of `execution_mode`, `run_id`, `slot_id`

Any ambiguous many-to-many identity state must fail loudly upstream; notebooks do not repair identity contracts.

---

## 6. Testing and Verification Gates

### 6.1 `make_report` selector tests

Add/extend tests to prove:
- preferred selected when qualifying
- fallback selected when preferred missing/below threshold
- `no_baseline_meets_threshold` path
- selector metadata persisted on all Stage D rows
- explicit no-selection field values are correct

### 6.2 Hard integrity tests

Add/extend tests for:
- hard error on duplicate completed `(counterpart_key, execution_mode_norm)` before selection
- global single-baseline invariant
- `selected_baseline_mismatch` and `multiple_selected_baselines` failure modes

### 6.3 Notebook 10 contract tests

Add/extend tests to assert:
- required baseline metadata fields are referenced/required
- no dynamic pairing key inference remains
- fixed-baseline filtering by `is_stage_d_selected_baseline`
- main conclusion is driven by valid/incomplete/unsupported counts + reason-code breakdowns

### 6.4 End-to-end smoke

After artifact regeneration, execute notebooks 08/10/11/12 and verify:
- 08 uses canonical comparison-key path for paired WHT-vs-ILP deltas
- 10 uses fixed-baseline counterpart coverage path
- 11/12 remain thin readers over `quality_cost_master` / `quality_cost_unavailable`

---

## 7. Acceptance Criteria

1. Stage D baseline is selected upstream via fixed-priority policy only.
2. Stage D selector metadata is present on all Stage D rows with explicit no-selection semantics.
3. No ambiguous counterpart duplicates survive to baseline selection.
4. Notebook 10 performs no baseline choice/fallback and pairs only by `counterpart_key` within selected-baseline rows.
5. Main Stage D conclusions use only valid fixed-baseline pairs and explicit counterpart reason-code accounting.
