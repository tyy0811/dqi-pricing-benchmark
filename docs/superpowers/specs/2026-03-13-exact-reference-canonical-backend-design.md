# Exact-Reference Canonical Backend Design

**Date:** 2026-03-13  
**Status:** Approved for planning  
**Scope:** Design only; no implementation in this document.

---

## 1. Decision

This phase makes exact-reference / ILP-derived encoding the canonical benchmark backend for reviewer-facing pricing comparisons while retaining WHT support as an exploratory path.

The change is intentionally constrained to:

- additive metadata
- default execution behavior on the default pricing path in `src/benchmark.py`
- reviewer-facing recommendation semantics
- notebook framing and consumption of existing benchmark outputs

No algorithmic logic or metric definitions change in this phase.

---

## 2. Non-Goals

This phase does **not** include:

- changes to metric definitions
- changes to surrogate generation logic
- removal of WHT support
- redesign of artifact consumers outside the scoped files
- historical result migration
- broad renaming of stable payload keys
- notebook-local recomputation of benchmark metrics

---

## 3. Compatibility Boundary

### 3.1 Stable compatibility surface

The following remain valid:

- existing result dict keys unless change is unavoidable
- existing paired-comparison payload shape, with additive fields only
- legacy backend names where already emitted for compatibility
- downstream consumers that read current payload keys

### 3.2 Reviewer-facing naming

The canonical reviewer-facing exact-reference label is:

- `ilp_derived_exact`

This label is for reviewer-facing summaries, notebook tables, and recommendation text. Internal payload keys and existing backend identifiers may remain unchanged unless required for display correctness.

### 3.3 Symmetric metadata policy

The metadata policy is intentionally asymmetric in implementation but symmetric in interpretation:

- WHT artifacts are explicitly tagged as exploratory
- exact-reference remains the implicit mainline backend unless a later phase adds a matching canonical tag

Downstream code should not infer “mainline” from name matching when explicit metadata is available.

---

## 4. Recommendation Semantics

### 4.1 Three distinct concepts

The spec separates:

- **formal verdict**: the strict paired-comparison outcome
- **default reviewer-facing backend**: the backend shown as canonical in reviewer-facing summaries
- **fallback/unavailable status**: the case where paired comparison is missing, partial, or not applicable

These must not be conflated.

### 4.2 Formal verdict helper

`src/encoding_compare.py` adds:

```python
def paired_encoding_verdict(delta_metrics: dict) -> str
```

It returns exactly one of:

- `mainline_exact_reference`
- `tie`
- `exploratory_only`

### 4.3 Conservative dominance rule

The formal verdict uses a conservative dominance rule over the decisive set and excludes `best_sampled_F` from that set.

The decisive set is based on existing decision-quality metrics already computed in the paired comparison, specifically the decision-oriented metrics such as:

- `top1_regret`
- `topk_regret`
- `optimum_recall_at_k`
- `decision_distortion`

and any already-computed directionally compatible metrics chosen in implementation, provided no new metric definitions are introduced.

Interpretation:

- `mainline_exact_reference`: exact-reference is non-worse on the decisive set and strictly better on at least one decisive metric
- `exploratory_only`: WHT is non-worse on the decisive set and strictly better on at least one decisive metric
- `tie`: neither backend conservatively dominates the other

### 4.4 Reviewer-facing recommendation policy

Reviewer-facing summary text defaults to recommending `ilp_derived_exact` as the canonical benchmark backend whenever:

- the paired comparison is present and complete, and
- the formal verdict is either `mainline_exact_reference` or `tie`

Reviewer-facing summaries may recommend WHT only when the formal conservative verdict is explicitly `exploratory_only`.

### 4.5 Mandatory recommendation block

Reviewer-facing benchmark outputs must include:

- `results["encoding_backend_recommendation"]`

even when the recommendation is unavailable.

This block must clearly distinguish:

- `formal_verdict`
- `default_reviewer_backend`
- `recommendation_basis`
- `status`
- `reason`

Suggested semantics:

- `formal_verdict`: one of the three verdict strings, or `None` when unavailable
- `default_reviewer_backend`: usually `ilp_derived_exact`, unless formal verdict explicitly favors WHT
- `recommendation_basis`: short machine-readable description of the basis used
- `status`: `available`, `unavailable`, or `not_applicable`
- `reason`: short human-readable explanation

### 4.6 Absence and partial-data behavior

If paired comparison is:

- not run
- not applicable
- partially populated
- missing required metrics

then:

- `encoding_backend_recommendation` must still exist
- no winner must be claimed
- reviewer-facing text must remain neutral
- a short unavailable reason must be present

No misleading recommendation may be emitted when the comparison was not actually performed and completed.

---

## 5. `src/encoding_compare.py`

### 5.1 Canonical backend constant

Add a top-level backend constant for the primary exact-reference path. Reviewer-facing text should use that constant rather than repeating string literals.

### 5.2 Reviewer-facing backend labels

Normalize reviewer-facing exact-reference labels to:

- `ilp_derived_exact`

Compatibility-preserving internal fields may still expose legacy names such as `ilp_exact_reference`.

### 5.3 Summary block

`run_paired_encoding_comparison(...)` adds a compact top-level `summary` block:

```python
summary = {
    "winner": ...,
    "metrics_used": [...],
    "reviewer_verdict": ...,
}
```

This summary should be derived only from existing paired-comparison metrics and the formal verdict helper.

### 5.4 Metrics source policy

The summary must reuse existing decision-quality metrics already computed by the paired-comparison payload. No new metric definitions or notebook-side recomputation are allowed.

---

## 6. `src/reduction.py`

### 6.1 Machine-readable exploratory metadata

WHT/exploratory artifacts add:

- `story_role = "exploratory"`
- `benchmark_status = "non_mainline_surrogate"`

These fields are additive metadata only.

### 6.2 Downstream classification rule

Downstream notebooks and reports should classify WHT as exploratory from artifact metadata first, not from backend-name matching or heuristic detection.

---

## 7. `src/benchmark.py`

### 7.1 Default pricing-path behavior

“Run by default” applies only to the default pricing path in `src/benchmark.py`.

It does **not** automatically apply to:

- unrelated benchmark entry points
- synthetic/non-pricing families
- arbitrary downstream consumers

### 7.2 Default paired comparison

For pricing instances on the default benchmark path:

- paired encoding comparison runs by default when both backends are available within the existing benchmark path

### 7.3 WHT ablation flag

Add:

- `run_wht_ablation: bool = True`

This controls whether exploratory WHT comparison content is included in the scoped benchmark path while preserving support for existing use cases.

### 7.4 Recommendation field

`src/benchmark.py` adds:

- `results["encoding_backend_recommendation"]`

This field is consumed by reviewer-facing summaries and notebooks.

### 7.5 Promotion rule

Exact-reference is promoted in reviewer-facing outputs only when:

- paired comparison is present
- paired comparison is complete
- recommendation status is `available`

If those conditions fail, the benchmark should preserve existing payloads and avoid exact-reference promotion for that run.

### 7.6 Non-pricing fallback

For non-pricing runs or contexts where exact-reference cannot be produced:

- preserve existing payloads
- skip exact-reference promotion silently except for a short status note
- emit `encoding_backend_recommendation` with `status` set appropriately

### 7.7 Display-only ordering rule

“Exact-reference first, WHT second” is a display/reporting rule only.

It must not silently redefine internal payload container semantics or reorder storage in a way that breaks existing consumers.

---

## 8. Notebook 08 Behavior

### 8.1 Role

Notebook 08 is the main reviewer-facing encoding notebook.

### 8.2 Data-source rule

Notebook 08 must read existing paired-comparison outputs only.

It must **not** recompute benchmark metrics or recreate paired-comparison logic locally.

### 8.3 Opening framing

The first section should ask:

- whether exact-reference encoding improves decision quality over WHT truncation

### 8.4 First-page summary table

The notebook should present a first-page summary table using existing paired-comparison outputs with columns:

- backend
- best sampled F
- top-1 regret
- top-k regret
- optimum recall@k
- Spearman rho(F,G)
- retained energy
- decision distortion
- verdict

Exact-reference appears first only when both backends are present.

### 8.5 Reviewer-facing verdict text

The notebook should end with markdown stating:

- exact-reference / ILP-derived is the default benchmark backend
- WHT is exploratory

But only when paired comparison is actually present and complete.

If paired comparison is missing or partial, the notebook must show no winner and display a neutral unavailable message.

---

## 9. Notebook 11 Behavior

### 9.1 Role

Notebook 11 is a dashboard/view layer, not the source of verdict logic.

### 9.2 Default toggle

Add a parameter/toggle cell:

- `include_exploratory_wht = False`

### 9.3 Recommendation-source rule

Notebook 11 must prefer:

- `results["encoding_backend_recommendation"]`

over local notebook logic when both are available.

### 9.4 Classification rule

Notebook 11 should classify WHT as exploratory using artifact metadata first, then fall back only if metadata is absent.

---

## 10. Acceptance Criteria

Implementation must satisfy all of the following:

### 10.1 Compatibility

- legacy payload keys remain unchanged for downstream consumers
- additive metadata does not break existing consumers

### 10.2 Recommendation semantics

- `encoding_backend_recommendation` exists in reviewer-facing benchmark outputs even when unavailable
- the field distinguishes formal verdict, default reviewer-facing backend, basis, status, and reason
- no reviewer-facing text claims a recommendation when paired comparison was not run

### 10.3 Ordering

- exact-reference appears first in summaries only when both backends exist
- ordering changes are display-only and do not redefine raw payload storage

### 10.4 Artifact classification

- WHT exploratory status is machine-readable in artifacts
- that status survives notebook ingestion
- notebooks prefer artifact metadata over backend-name matching

### 10.5 Notebook 08

- Notebook 08 reads existing paired-comparison outputs only
- Notebook 08 does not recompute benchmark metrics
- Notebook 08 shows a neutral unavailable message with no winner when paired comparison is missing or partial

### 10.6 Notebook 11

- Notebook 11 defaults to exact-reference-first views
- Notebook 11 prefers `encoding_backend_recommendation` over local notebook verdict logic

---

## 11. Files in Scope

This design is scoped to:

- `src/encoding_compare.py`
- `src/reduction.py`
- `src/benchmark.py`
- `notebooks/08_paired_encoding_comparison.ipynb`
- `notebooks/11_decision_faithfulness_dashboard.ipynb`

---

## 12. Planning Notes

The implementation plan should keep the work split into:

1. additive metadata and verdict semantics
2. benchmark default behavior and recommendation emission
3. notebook consumption and reviewer-facing framing

The plan should preserve payload compatibility and avoid unscoped refactoring.
