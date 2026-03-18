# Stage C Decoder Study Design

**Date:** 2026-03-12
**Status:** Approved for spec drafting
**Scope:** Design-only. No code changes in this phase.
**Primary goal:** Close Gap 2 with a structured, laptop-feasible decoder study that is compatible with the current reporting pipeline and notebook review flow.

---

## 1. Decision

Section 3 is **approved as written**.

This spec locks the Stage C design around five constraints:

1. **Raw decoder run files remain authoritative** and contain only per-decoder outputs.
2. **Cross-decoder comparative metrics are report-derived**, not stored in raw runs.
3. **Deduplication and grouping are canonicalized around `canonical_trial_seed`**, with `trial_seed` treated as the canonical seed and `seed` retained only as legacy provenance.
4. **Oracle tiny-only behavior is explicit and structured**, never inferred from null metrics or free-text error strings.
5. **Notebook 09 remains a reader notebook**, with placeholder buckets that distinguish in-matrix unsupported rows from actual failures.

---

## 2. Objectives

### 2.1 Scientific objective

Stage C is the dedicated **Gap 2 closure stage**. The study must support a claim of the form:

> Decoder quality differences are not anecdotal; they are systematically related to collision structure, syndrome ambiguity, and lightweight structural proxies. BP+OSD-lite closes part of the regret left by BP1, and oracle performance provides tiny-instance ground truth rather than an extrapolated reference.

### 2.2 Engineering objective

The implementation must fit the current repository architecture:

* raw results are written under `results/matrix_c/runs/`
* `make_report.py` regenerates summary tables and reports from raw runs
* notebooks remain thin artifact readers
* legacy rows may coexist with Stage C rows during transition

### 2.3 Review objective

The design must remain credible under research-lab style review:

* same-trial comparisons only
* no silent synthetic rows for absent trials
* no gain metrics written into raw run files
* tiny-only oracle applicability must be machine-readable
* runtime must remain bounded for laptop execution

---

## 3. Scope

### 3.1 Included families

Stage C includes:

* **Family P**: pricing-relevant instances, explicitly scoped to a small default set such as `P3/P4/P5`
* **Family S**: controlled structure family, using either all canonical S instances or a clearly defined small subset

### 3.2 Included decoders

Required decoder set:

* bounded-distance brute force
* BP1
* BP+OSD-lite
* exact oracle decoder on tiny instances only

### 3.3 Excluded from Stage C

This stage does **not** attempt to:

* solve coherent alpha validation (Stage D)
* change Stage A paired pricing export semantics
* alter Stage E dashboard logic beyond what is needed for Stage C compatibility
* produce cross-stage synthetic master rows for absent manifest entries

---

## 4. Authoritative Raw Run Semantics

### 4.1 Raw runs are authoritative per decoder

Each raw Stage C run file represents a **single decoder execution** on a single canonical trial.

Raw run files may contain only:

* decoder-local outputs
* trial-local metadata
* structured status information
* frozen structure metric outputs
* mandatory decoder-specific metadata

Raw run files must **not** contain report-derived cross-decoder comparison metrics such as:

* `gain_over_bp1`
* `gain_over_bruteforce`

These metrics are derived only in `make_report.py`.

### 4.2 Raw run append-only behavior

`run_decoder_study` appends raw JSON files only to:

`results/matrix_c/runs/`

The reporting layer then regenerates derived views from the complete set of raw runs.

---

## 5. Canonical Identity and Deduplication

### 5.1 Canonical dedupe identity

The canonical dedupe identity is frozen to:

* `matrix`
* `stage`
* `family`
* `instance_id`
* `encoding`
* `decoder`
* `alpha_mode`
* `execution_mode`
* `canonical_trial_seed`

### 5.2 Trial seed semantics

* `trial_seed` is the canonical trial seed
* `canonical_trial_seed` is the normalized reporting field derived from `trial_seed`
* `seed` is retained only as legacy provenance and must never be used for Stage C dedupe or gain grouping

### 5.3 Stage-aware dedupe behavior

Deduplication is stage-aware. Mixed legacy rows and Stage C rows may coexist.

Implications:

* Stage C report generation must not collapse rows across stages
* decoder comparisons for gain derivation must only occur within the same Stage C grouping key
* report summaries may add `by_stage` aggregates without rewriting legacy history

---

## 6. Frozen Structure Metric Schema

Stage C structure outputs are frozen to exact numeric columns so that downstream reporting and notebook panels are stable.

### 6.1 Required structure columns

At minimum, the structure metric layer must expose numeric columns for:

* row weight minimum
* row weight mean
* row weight maximum
* column weight minimum
* column weight mean
* column weight maximum
* rank
* rank deficiency
* collision rate
* ambiguous syndrome count
* short-cycle proxy
* distance proxy

### 6.2 Structural schema intent

These columns are treated as canonical and joinable. The report layer may compute additional summaries, but the frozen numeric columns above are the compatibility baseline for notebook 09 and downstream joins.

---

## 7. Decoder Metadata Requirements

### 7.1 BP+OSD-lite mandatory metadata

Every BP+OSD-lite raw row must carry the following fields:

* `osd_order_cap`
* `osd_candidate_budget`
* `tie_break`
* `fallback_flag`
* `iterations_used`

These fields are mandatory regardless of whether the fallback path was taken.

### 7.2 Oracle explicit applicability semantics

Oracle tiny-only semantics must be explicit and structured.

The raw row must encode applicability as machine-readable status, not as:

* NaN interpretation
* exception text parsing
* free-text notebook heuristics

---

## 8. Status and Failure Semantics

### 8.1 Approved status model

The reporting/notebook status classes are:

* `not_in_experiment_matrix`
* `not_applicable`
* `failed`
* `completed`

### 8.2 Compatibility-safe encoding

To preserve existing pipeline compatibility:

* `run_status` remains within the existing compatible vocabulary: `completed`, `failed`, `skipped`
* `not_applicable` is represented as:

  * `run_status = "skipped"`
  * a structured `status_reason_code` such as `oracle_tiny_cap_exceeded`
  * a decoder-specific machine-readable status such as `decoder_status = "skipped_tiny_only"`

### 8.3 Manifest absence vs unsupported vs failed

These meanings are locked:

* **not_in_experiment_matrix**: manifest/display-only concept, never represented by synthetic raw or master rows
* **not_applicable**: in-matrix but intentionally unsupported, such as oracle over the tiny cap
* **failed**: attempted execution errored
* **completed**: attempted execution succeeded with usable outputs

### 8.4 Metric availability mapping

Metric availability must map these classes explicitly:

* completed rows expose normal metrics
* failed rows expose availability = failed_upstream where appropriate
* not_applicable rows expose availability = not_applicable
* not_in_experiment_matrix appears only as a notebook/report display bucket, not as a fabricated data row

---

## 9. Runtime Guardrails

Stage C must be laptop-friendly by design.

### 9.1 Study scope defaults

The default scope is explicit and small:

* Family P: `P3/P4/P5`
* Family S: a controlled subset or the complete canonical S family if it remains tractable

The scope must be configurable, but the default must remain bounded.

### 9.2 BP+OSD-lite guardrails

BP+OSD-lite must be bounded by fixed:

* `osd_order_cap`
* `osd_candidate_budget`

These caps are part of the run metadata and are required for interpretability and reproducibility.

### 9.3 Oracle guardrails

The oracle decoder has a hard tiny-instance cap.

Oversized cases must be marked **not applicable**, not failed.

---

## 10. Report-Derived Comparative Metrics

### 10.1 Frozen gain formulas

The following formulas are frozen:

* `gain_over_bp1 = best_F(decoder) - best_F(bp1)`
* `gain_over_bruteforce = best_F(decoder) - best_F(bruteforce)`

### 10.2 Same-trial grouping rule

These gains are computed only within the same:

* `matrix`
* `stage`
* `family`
* `instance_id`
* `encoding`
* `alpha_mode`
* `execution_mode`
* `canonical_trial_seed`

No cross-trial, cross-stage, or cross-family gain derivation is allowed.

### 10.3 Missing comparator behavior

If the comparator decoder row is absent within the same canonical group, the gain metric remains unavailable rather than being imputed.

---

## 11. Reporting Flow

### 11.1 Regeneration model

The regeneration flow is locked to:

1. `run_decoder_study` appends raw JSON files to `results/matrix_c/runs/`
2. `make_report.py` scans all raw Stage C runs
3. `make_report.py` performs canonical trial-seed normalization
4. `make_report.py` performs stage-aware dedupe
5. `make_report.py` derives same-trial gain metrics
6. `make_report.py` regenerates:

   * `matrix_c_summary.csv`
   * `matrix_c_report.json`

### 11.2 Legacy coexistence

Mixed legacy and Stage C rows may remain during transition.

The report layer therefore must support:

* stage-aware summaries
* `by_stage` aggregates
* Stage C-preferred panels in notebook 09 without rewriting older artifacts

---

## 12. Notebook 09 Contract

Notebook 09 is locked to the following selection rule:

* `matrix == "matrix_c"`
* `stage == "decoder_study"`
* `family in {"P", "S"}`

### 12.1 Panel behavior

Existing notebook 09 panels remain, including:

* decoder coverage
* regime tables
* gain vs collision plots
* ambiguity/correlation summaries
* placeholder/unavailable summaries

### 12.2 Placeholder buckets

Notebook 09 must distinguish three display buckets explicitly:

* not-in-matrix
* not-applicable
* failed

These are display categories, not reasons to synthesize master rows.

---

## 13. Test Gates

The following test gates are required before Stage C is considered implementation-ready.

### 13.1 Decoder correctness and safety

* tiny oracle exactness
* BP+OSD-lite non-crash on small P/S cases

### 13.2 Structure metric stability

* structure metrics smoke test against the frozen numeric schema

### 13.3 Reporting integrity

* stage precedence behavior
* trial-seed canonicalization
* stage-aware dedupe
* report-derived gain calculation correctness

---

## 14. Acceptance Criteria

Section 3 acceptance is locked to the following conditions:

1. Raw Stage C files remain authoritative and per-decoder only.
2. Gain metrics are absent from raw runs and derived only in `make_report.py`.
3. Canonical dedupe identity uses `canonical_trial_seed` and not legacy `seed`.
4. Oracle tiny-only applicability is machine-readable and represented as not-applicable rather than failed.
5. Frozen structure metric columns are present and numeric.
6. BP+OSD-lite metadata is mandatory per row.
7. Runtime defaults remain laptop-feasible.
8. Notebook 09 reads Stage C rows only and preserves explicit display buckets for not-in-matrix, not-applicable, and failed.
9. Reporting tests validate stage-aware dedupe and gain derivation.

---

## 15. Immediate Next Document

After this design spec, the next artifact should be a **writing-plan document**, not code changes.

That writing-plan should break implementation into:

1. raw run schema and decoder metadata plan
2. structure metrics schema freeze plan
3. reporting derivation plan
4. notebook 09 update plan
5. test plan and stop criteria

No code changes are authorized by this spec alone.

---

## 16. Notes for Implementation Phase

Implementation should preserve the following reviewer-facing invariants:

* decoder comparisons are always same-trial
* unsupported oracle rows are visible but not counted as failures
* report-derived metrics are reproducible from raw authoritative runs
* notebook display semantics do not invent data that is absent from the experiment matrix

This closes the design phase for Stage C Section 3 and authorizes transition to writing plans.
