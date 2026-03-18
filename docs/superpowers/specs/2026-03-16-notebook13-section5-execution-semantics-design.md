# Notebook 13 Section 5 Execution Semantics Design

**Date:** 2026-03-16  
**Status:** Approved for planning  
**Scope:** Refactor Section 5 and aligned Section 6 wording in `notebooks/13_scaling_analysis.ipynb`.

---

## 1. Decision

Section 5 in Notebook 13 will use the execution/boundary semantic layer, not the Stage E comparability layer.

The canonical reviewer-facing Section 5 slice is:

- `family == "P"`
- `execution_mode == "mixture"`
- `encoding == "wht_truncated_pricing"`
- `decoder in {"bp1", "bruteforce"}`
- use recorded per-row `ell` values rather than imposing a synthetic cross-instance `ell` rule

This section reports the executed pricing scaling slice. It is not the Stage E comparability slice used for reviewer-facing quality-vs-cost panels.

---

## 2. Semantic Source Of Truth

Section 5 must derive its main tables and prose from execution-layer records whose semantics come from:

- `src/boundary.py`
- `src/matrix_runner.py`
- canonical execution/master rows preserving those fields in `results/metrics_master.json`

The primary fields are:

- `run_status`
- `run_status_norm` only as a local normalization aid
- `attempted_execution`
- `boundary_reason`
- `estimated_qubits`
- `estimated_memory_gb`
- `wall_clock_s`
- `peak_memory_mb`

Section 5 must not use:

- `quality_cost_master`
- `quality_cost_unavailable`
- Stage E unavailable classes

as the primary source for timing or boundary claims.

---

## 3. Section 5 Structure

Section 5 will be split into two reviewer-facing layers drawn from the same canonical execution slice.

### 3.1 Completed execution timing evidence

This table includes only canonical completed rows from the selected slice with actual timing or memory evidence.

Its purpose is to show:

- what actually executed
- timing/runtime evidence
- memory evidence where present

The timing table must not include rows outside the approved canonical slice. In particular, it must exclude:

- `oracle`
- `ilp_derived`
- `wht_truncated`

### 3.2 Boundary and unavailable coverage evidence

This display must be visually split into:

- explicit boundary evidence
- generic unavailable rows without boundary provenance

Explicit boundary evidence means rows from the canonical execution slice with meaningful boundary metadata, such as:

- `boundary_reason`
- `attempted_execution`
- `estimated_qubits`
- `estimated_memory_gb`

Generic unavailable rows are canonical-slice rows with unavailable/failed status but without strong boundary provenance. These rows must not be presented as if they prove a tractability boundary.

---

## 4. Claim Strength

Section 5 claim strength must follow the available execution metadata.

If the canonical slice contains strong boundary metadata, Section 5 may describe an explicit execution/resource boundary, but only to the extent supported by those rows.

If the canonical slice is sparse on strong boundary metadata, Section 5 must downgrade to a validated executed coverage boundary claim.

For the current pricing slice, the design expectation is:

- use the executed `mixture` / `wht_truncated_pricing` backend because it is the honest large-scale executed slice
- do not overstate this as a universal tractability boundary across backends
- state explicitly that this is an executed coverage story, not the Stage E comparability story

Recommended anchor wording:

`Section 5 reports the validated executed pricing scaling slice in the current master. It supports a coverage boundary claim for the executed mixture / wht_truncated_pricing backend, not a universal tractability boundary across backends.`

---

## 5. Section 6 Alignment

Section 6 verdict 4 must align exactly with the new Section 5 semantics.

It must no longer claim a statevector tractability boundary unless Section 5’s canonical execution slice contains explicit execution/resource boundary evidence strong enough to support that phrasing.

The preferred downgrade is:

- `validated executed coverage boundary`

or equivalent evidence-matched wording.

---

## 6. Non-Goals

This refactor does not include:

- reopening the Section 2 nested-metrics bug
- changing shared benchmark math
- changing Stage E comparability rules
- replacing the Section 5 execution narrative with a reviewer-comparable exact-reference narrative

---

## 7. Validation Requirements

After implementation:

1. Re-execute:
   - `jupyter nbconvert --to notebook --execute --inplace notebooks/13_scaling_analysis.ipynb`

2. Verify Section 5 postconditions:
   - no rows outside the approved canonical slice
   - no `oracle`
   - no `ilp_derived`
   - no `wht_truncated`
   - prose does not use Stage E unavailable classes as tractability evidence
   - explicit split between boundary-evidence rows and generic unavailable rows

3. Verify Section 6:
   - verdict 4 matches the downgraded execution-semantics claim exactly

