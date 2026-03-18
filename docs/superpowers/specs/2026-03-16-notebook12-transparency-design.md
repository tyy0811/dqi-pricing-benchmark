# Notebook 12 Exact-Reference Transparency Improvements

**Date**: 2026-03-16
**Status**: Design approved, pending implementation

## Summary

Notebook 12 remains a strict reviewer-facing Stage E exact-reference panel. The main panel is limited to rows passing the explicit exact-reference allowlist and downstream reviewer-scope filters. To make thin-slice behavior legible without weakening semantics, the notebook adds a stepwise exclusion audit and a compact explanatory context panel for excluded exact-reference rows.

## Goals

1. Keep Notebook 12 strict exact-reference
2. Improve transparency when the final slice is thin
3. Preserve mixture-first, reviewer-facing semantics
4. Do not promote WHT rows into the main panel

## Non-Goals

- Expand exact-reference generation semantics
- Reframe Notebook 12 as an execution-coverage notebook
- Add WHT rows to fill out thin slices
- Force upstream coverage expansion without scientific justification

---

## Part 1: Light Upstream Audit

### Scope

Verify there is no accidental loss, mislabeling, or avoidable exclusion of exact-reference rows before accepting structural limitations.

### Audit Checklist

1. **Check `quality_cost_unavailable`** for any `ilp_derived_exact` rows that failed Stage E admission
   - If found: investigate whether the exclusion is avoidable (bookkeeping bug) or structural (missing required fields)

2. **Check encoding label consistency**
   - Verify `encoding_compare.PRIMARY_EXACT_REFERENCE_BACKEND` matches Notebook 12's `EXACT_REFERENCE_ENCODINGS` allowlist
   - Confirm no mislabeled rows exist

3. **Check for non-C-family exact-reference potential**
   - Determine whether any P-family (pricing) instances could produce `ilp_derived_exact` under current project semantics
   - If not: this is a structural limitation, not a bug

### Expected Outcome

- No upstream bugs found
- Structural limitation documented: if confirmed by the audit, exact-reference coverage is concentrated in C-family rows in current Stage E artifacts
- No semantic widening required

### Policy

> Audit for accidental loss first; if none exists, accept structural thinness and explain it clearly.

---

## Part 2: Notebook 12 Transparency Improvements

### Final Notebook Structure

#### 1. Stage E Comparable Input Slice Summary
- Load `quality_cost_master` and `quality_cost_unavailable`
- Display total row counts for each

#### 2. Encoding Label Audit
- Show encoding distribution before exact-reference filtering
- Display counts by `encoding_norm` column
- Make it clear how many rows exist for each encoding type

#### 3. Exact-Reference Downstream Exclusion Audit (New)

Stepwise cumulative audit table with columns:
- `step`: filter stage name
- `input_rows`: row count entering the stage
- `output_rows`: row count after the stage
- `rows_removed`: count removed at this stage
- `reason_label`: filter description

Filter stages in order:
1. Pre-filter comparable slice (all `quality_cost_master` rows)
2. Exact-reference allowlist filter
3. Mixture-first filter
4. Coherent-off filter
5. Decoder filter (bp1/bruteforce)
6. Final plotted slice

This makes it obvious where rows disappear.

#### 4. Main Strict Exact-Reference Comparable Panel (Unchanged)

- Use explicit allowlist: `EXACT_REFERENCE_ENCODINGS = {"ilp_derived_exact", "ilp_derived_exact_where_feasible"}`
- Apply mixture-first filter
- Exclude coherent execution
- Filter to bp1/bruteforce decoders
- Exclude WHT rows
- Generate Pareto-style quality-vs-cost plot from this slice only

**Constraint**: Context panel must not influence main panel logic. No reuse of excluded rows in summary metrics.

#### 5. Excluded Exact-Reference Rows (Context Only) (New)

Compact table showing exact-reference rows that passed the exact-reference filter but were excluded by downstream filters.

Columns:
- Row identifier
- Encoding
- Execution mode
- Decoder
- `exclusion_reason` with values like:
  - `non_mixture_execution`
  - `coherent_execution_excluded`
  - `decoder_out_of_scope`
  - `missing_comparable_field`

Visual treatment:
- Clearly labeled as context/explanatory only
- Subordinate to main panel (smaller, muted styling)
- No separate plot unless it adds clear value
- Small count summary above the table

#### 6. Structural Limitation / Interpretation Note (New)

Confident, non-apologetic framing:

- "This notebook is a **strict reviewer-facing exact-reference panel**."
- "Broad execution coverage is handled elsewhere in the benchmark stack."
- "The thin final slice reflects **coverage limits of the exact-reference comparable backend**, not a notebook defect."
- If confirmed by audit: "In the current canonical Stage E artifacts, exact-reference coverage is concentrated in C-family rows."

---

## Constraints

### Preserved Invariants

1. Main panel uses explicit `EXACT_REFERENCE_ENCODINGS` allowlist
2. No WHT rows enter the main exact-reference plot/table
3. Mixture-first semantics remain intact
4. Context panel is explanatory only, not rhetorically equivalent to main panel
5. No fuzzy substring matching for exact-reference labels

### What Not To Do

- Do not add WHT rows to fill out the main panel
- Do not blur exact-reference with "close enough" rows
- Do not switch Notebook 12 to execution-boundary semantics
- Do not remove comparability gates unless scientifically justified
- Do not hide exclusion reasons

---

## Validation

After implementation, run:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/12_quality_vs_cost.ipynb
python3 -m pytest tests/test_notebook12_stage_e_smoke.py -x
```

If the full test suite is cheap, also run:
```bash
python3 -m pytest tests/ -x
```

---

## Success Criteria

1. Notebook 12 still uses strict explicit exact-reference allowlist
2. Main reviewer-facing panel contains only exact-reference comparable rows
3. Stepwise exclusion audit clearly shows where rows are filtered
4. Secondary context panel exists and is visually subordinate
5. No WHT row is promoted into the main exact-reference panel
6. Final prose accurately frames Notebook 12 as a strict panel
7. If upstream audit finds no bugs, structural limitation is documented

---

## Deliverables

1. Light upstream audit results (in implementation summary)
2. Patched `notebooks/12_quality_vs_cost.ipynb`
3. Short summary covering:
   - Upstream audit outcome
   - Final exact-reference allowlist
   - Exclusion audit breakdown
   - Whether main exact-reference plot is populated
   - What the secondary context panel shows
