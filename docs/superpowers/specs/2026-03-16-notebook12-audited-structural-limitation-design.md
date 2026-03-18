# Notebook 12: Audited Structural-Limitation Refactor

**Date:** 2026-03-16
**Status:** Approved
**Type:** Notebook Enhancement

## Context

Notebook 12 (`notebooks/12_quality_vs_cost.ipynb`) is a strict reviewer-facing exact-reference panel that currently shows an empty final slice because the canonical Stage E artifacts (quality_cost_master) contain no exact-reference comparable rows.

**Current State:**
- 6 comparable rows loaded, all using `wht_truncated_pricing` encoding
- Exact-reference allowlist: `{ilp_derived_exact, ilp_derived_exact_where_feasible}`
- All 6 rows filtered out at step 2 (exact-reference filter)
- Final slice: empty

**Problem:**
The empty state is not adequately explained. Without context, an empty notebook appears broken rather than truthfully reflecting artifact scope limitations.

**Root Cause:**
Not a notebook bug or pipeline bug — this is **true coverage scarcity**. The exact-reference backend (`ilp_derived_exact`) exists in matrix_a, but matrix_a is not included in the current canonical artifact processing scope (make_report.py processes matrix_c only). Therefore, exact-reference rows cannot reach quality_cost_master.

## Goals

1. Keep Notebook 12 strict exact-reference only (no exploratory substitution)
2. Make the empty state fully explained, auditable, and reviewer-safe
3. Preserve the notebook as a strict exact-reference admissibility/comparability panel
4. Distinguish between artifact-scope limitations and notebook defects

## Non-Goals

- Admitting WHT rows to main panel
- Fuzzy matching for exact-reference labels
- Converting to execution-coverage notebook
- Weakening exact-reference semantics

## Design

### Architecture: Three-Layer Information Architecture

**1. Artifact-Scope Context Layer** (new)
Establishes what data exists in loaded artifacts

**2. Audit Layer** (existing, enhanced)
Tracks where data goes during filtering pipeline

**3. Interpretation Layer** (new)
Explains empty states with confident, assertive tone

### Implementation Components

#### Component 1: Early Artifact-Scope Note

**Placement:** New markdown cell after data loading, before filtering pipeline begins

**Purpose:** Establish artifact scope and exact-reference policy before reader sees filtering

**Content Structure:**
```markdown
## Artifact Scope and Exact-Reference Policy

This notebook is a **strict reviewer-facing exact-reference panel**. It enforces an explicit exact-reference encoding allowlist and does not substitute exploratory encodings for missing exact-reference coverage.

**Exact-reference allowlist:** `{ilp_derived_exact, ilp_derived_exact_where_feasible}`

**Current artifact scope (loaded from quality_cost_master):**
- Total comparable rows: <count>
- Encodings present: <list actual labels with counts>
- Exact-reference rows present: <count matching allowlist>

**Expected behavior:**
- If exact-reference comparable rows exist in current artifacts → main panel displays them
- If not → main panel will be empty (by design, not defect)

**Artifact-scope dependency:** This notebook's non-empty state depends on exact-reference comparable rows being present in quality_cost_master, which reflects the matrices processed by the upstream report-generation scope.
```

**Implementation Notes:**
- Inspect `quality_master['encoding']` column before filtering
- Show actual encoding labels found (not generic warnings)
- Compute exact-reference match count by checking `encoding.isin(EXACT_REFERENCE_ENCODINGS)`
- Keep concise (6-8 bullets max)

**Tone:** Assertive and policy-focused, citing concrete evidence from loaded data

#### Component 2: Enhanced Encoding-Label Audit

**Current State:** Already displays encoding distribution before exact-reference filter

**Action:** Add clarifying header and explicit exact-reference match count

**Before:**
```markdown
**Total comparable rows before exact-reference filter:** 6
```

**After:**
```markdown
### Pre-Filter Artifact Composition

**Total comparable rows loaded from quality_cost_master:** 6

**Encodings present in comparable slice:**
- wht_truncated_pricing: 6
- ilp_derived_exact: 0
- ilp_derived_exact_where_feasible: 0

**Exact-reference allowlist:** `{ilp_derived_exact, ilp_derived_exact_where_feasible}`
**Rows matching allowlist before filtering:** 0
```

**Implementation Notes:**
- Group by `encoding` column and count
- Explicitly list exact-reference encodings with their counts (even if zero)
- State allowlist for transparency

#### Component 3: Cumulative Exclusion Audit

**Current State:** Already implemented via `_track_filter()` function

**Action:** Preserve as-is. Already provides:
- Step-by-step audit table with columns: `[step, input_rows, output_rows, rows_removed, reason_label]`
- Covers all filter steps: exact-reference allowlist, mixture-first, coherent-off, decoder scope, WHT-off
- Clear reason labels for each removal

**No changes needed** - this component is already strong.

#### Component 4: Excluded Exact-Reference Context Panel

**Current State:** Shows rows that passed exact-reference filter but failed downstream filters

**Action:** Add clarifying header emphasizing subordinate explanatory role

**Before:**
```markdown
### Excluded Exact-Reference Rows
```

**After:**
```markdown
### Context: Exact-Reference Rows Excluded by Downstream Filters

*This panel shows exact-reference rows that passed the encoding allowlist but were removed by downstream scope filters (mixture-only, decoder scope). It provides explanatory context only and is not part of the main comparable panel.*
```

**Keep Existing:**
- Exclusion reason computation logic (`_compute_exclusion_reason()`)
- Summary table showing exclusion reason counts
- Sample excluded rows display

**Implementation Notes:**
- Only displays if `len(exact_ref_rows) > 0 and len(final_df) == 0`
- Remains visually and rhetorically subordinate to main panel
- Does NOT include WHT rows (only rows that passed exact-reference filter)

#### Component 5: Empty-State Interpretation Block

**Placement:** Immediately before main panel (plot/table), so reader sees interpretation then placeholder

**Purpose:** Explain empty state with assertive confidence when final slice is empty

**Content Structure (when empty):**
```markdown
### Main Panel: Strict Exact-Reference Comparable Slice

**Current Status:** Empty slice (0 rows)

**Interpretation:**

This notebook is a strict reviewer-facing exact-reference panel. The current empty final slice reflects the scope of the loaded canonical artifacts, not a notebook defect. The notebook intentionally does not substitute exploratory encodings for missing exact-reference coverage.

**Evidence:** In the current artifact scope, all <N> comparable rows use encoding=<list>. None match the exact-reference allowlist `{ilp_derived_exact, ilp_derived_exact_where_feasible}`.

**Artifact-Scope Context:** Exact-reference comparable rows may be present in other matrices (e.g., matrix_a) but are not included in the current Stage E canonical artifacts. The canonical artifacts reflect the matrices processed by the upstream report-generation scope.

**Broader Evidence:** Execution coverage and scaling evidence across multiple encoding backends (including exploratory) is available in other notebooks in this benchmark stack.
```

**Content Structure (when non-empty):**
```markdown
### Main Panel: Strict Exact-Reference Comparable Slice

**Current Status:** <N> rows in final slice

[Show table and plot]
```

**Tone:** Assertive, confident. Treats emptiness as an audited structural fact, not an apology.

**Key Requirement:** This interpretation block must appear BEFORE the placeholder plot/table, so the reader understands why it's empty before seeing the visual placeholder.

#### Component 6: Main Plot/Table Empty-State Handling

**Current State:** Uses `unavailable_panel_table()` for empty slice (small text-only table)

**Action:** Replace with visible placeholder using `placeholder_plot()` helper

**Implementation (Plot):**
```python
if final_df.empty:
    # Use placeholder_plot() for visible empty state
    # Note: placeholder_plot() returns (fig, ax) tuple
    fig, ax = placeholder_plot(
        title="Strict Exact-Reference Quality vs Cost Comparison",
        reason="No exact-reference comparable rows in current artifact scope (see interpretation above)",
        figsize=(10, 4)
    )
    display(fig)
else:
    # Normal Pareto-style plot
    [existing plot code]
```

**Implementation (Table):**
```python
if final_df.empty:
    display(Markdown("**Table:** Empty (see interpretation above)"))
else:
    display(final_df[show_cols])
```

**Rationale:** Visible placeholder is more reviewer-safe than silent blank output. The placeholder references "see interpretation above" so reader knows where to look for explanation.

#### Component 7: Final Closing Note

**Placement:** Final cell of notebook

**Purpose:** Summarize notebook role and scope dependencies

**Content:**
```markdown
## Notebook Role and Scope

Notebook 12 is the **strict exact-reference admissibility and comparability panel** for the current Stage E canonical artifacts.

**What this notebook does:**
- Enforces strict exact-reference encoding policy (no exploratory substitution)
- Audits exact-reference row availability in current artifact scope
- Documents filtering pipeline with cumulative exclusion tracking
- Provides transparent explanation when exact-reference evidence is absent

**What this notebook does not do:**
- Substitute exploratory encodings (e.g., WHT variants) for missing exact-reference rows
- Provide broadest execution or scaling coverage (other notebooks provide multi-backend evidence)
- Generate exact-reference runs (upstream artifact generation controls availability)

**Artifact-Scope Dependency:**

This notebook's non-empty state depends on exact-reference comparable rows being present in quality_cost_master. In the current regenerated canonical artifacts, comparable rows come from the presently processed report scope; under the current scope, exact-reference comparable rows are absent.

The exact-reference backend (`ilp_derived_exact`) exists in the benchmark infrastructure but is not present in the current Stage E artifacts because those artifacts are sourced from matrices included in the most recent report-generation run.
```

**Tone:** Professional, explicit about scope dependencies without hardcoding specific matrix names

**Key Difference from Early Note:**
- Early note: establishes policy and shows immediate artifact evidence
- Closing note: summarizes role, clarifies what notebook does/doesn't do, explains scope dependency

### Information Flow

```
1. Data Load (existing)
   ↓
2. Early Artifact-Scope Note [NEW]
   - Policy statement
   - Loaded artifact evidence (encoding counts, exact-ref matches)
   - Expected behavior based on current scope
   ↓
3. Enhanced Encoding-Label Audit [ENHANCED]
   - Pre-filter encoding distribution
   - Explicit exact-reference match count
   ↓
4. Filter Pipeline (existing)
   - Exact-reference allowlist filter
   - Downstream scope filters (mixture-only, decoder scope)
   ↓
5. Cumulative Exclusion Audit [EXISTING]
   - Step-by-step tracking table
   ↓
6. Excluded Exact-Reference Context Panel [ENHANCED HEADER]
   - Explanatory subordinate panel for exact-ref rows removed by downstream filters
   ↓
7. Empty-State Interpretation [NEW]
   - Assertive confident explanation with concrete evidence
   - Placed IMMEDIATELY BEFORE main panel
   ↓
8. Main Panel [ENHANCED PLACEHOLDER]
   - Visible placeholder plot if empty (references interpretation above)
   - Normal Pareto plot if non-empty
   ↓
9. Final Closing Note [NEW]
   - Notebook role summary
   - Scope dependency explanation
```

## Success Criteria

| Criterion | Implementation | Validation |
|-----------|----------------|------------|
| Remains strict exact-reference only | No changes to `EXACT_REFERENCE_ENCODINGS` allowlist or filtering logic | Verify allowlist unchanged, no fuzzy matching |
| Artifact-scope limitation explained early | New artifact-scope note after data loading with concrete evidence | Check note appears early, cites actual loaded data |
| Encoding-label audit obvious | Enhanced pre-filter audit with explicit exact-ref count | Verify encoding distribution visible, exact-ref count shown |
| Cumulative exclusion audit shows removals | Existing `_track_filter()` preserved | Verify step-by-step table still present |
| Reviewer-safe empty-state interpretation | New assertive interpretation block before main panel | Check interpretation precedes placeholder, uses confident tone |
| No WHT in main panel | Allowlist unchanged, no substitution logic added | Inspect final_df filtering, confirm no WHT rows |
| Reads as audited structural limitation | Three-layer architecture + confident tone throughout | Full notebook read-through, check tone consistency |
| Visible placeholder for empty state | Use `placeholder_plot()` instead of `unavailable_panel_table()` | Verify visible plot renders when empty |

## Implementation Details

### Cell Type and Placement Specifications

**Component 1: Early Artifact-Scope Note**
- **Cell Type:** Python cell with Markdown display
- **Placement:** Insert NEW cell between data loading cell (current cell 2) and "Default Mainline Slice" header (current cell 3)
- **Implementation:**
```python
# Inspect loaded artifacts and display scope note
encoding_col = None
for col in ["encoding_norm", "encoding"]:
    if col in quality_master.columns:
        encoding_col = col
        break

if encoding_col:
    encoding_counts = quality_master[encoding_col].value_counts()
    exact_ref_count = quality_master[encoding_col].str.lower().isin([e.lower() for e in EXACT_REFERENCE_ENCODINGS]).sum()

    encoding_list = "\n".join([f"- {enc}: {count}" for enc, count in encoding_counts.items()])

    display(Markdown(f"""
## Artifact Scope and Exact-Reference Policy

This notebook is a **strict reviewer-facing exact-reference panel**. It enforces an explicit exact-reference encoding allowlist and does not substitute exploratory encodings for missing exact-reference coverage.

**Exact-reference allowlist:** `{{{', '.join(EXACT_REFERENCE_ENCODINGS)}}}`

**Current artifact scope (loaded from quality_cost_master):**
- Total comparable rows: {len(quality_master)}
- Encodings present:
{encoding_list}
- Exact-reference rows present: {exact_ref_count}

**Expected behavior:**
- If exact-reference comparable rows exist in current artifacts → main panel displays them
- If not → main panel will be empty (by design, not defect)

**Artifact-scope dependency:** This notebook's non-empty state depends on exact-reference comparable rows being present in quality_cost_master, which reflects the matrices processed by the upstream report-generation scope.
"""))
```

**Component 5: Empty-State Interpretation**
- **Cell Type:** Python cell with conditional Markdown display (within filtering pipeline cell)
- **Placement:** Within existing filtering pipeline cell, immediately before main panel plot/table rendering
- **Implementation:**
```python
# After filtering completes and final_df is computed
if final_df.empty:
    encoding_list = ", ".join(sorted(quality_master[encoding_col].unique())) if encoding_col else "unknown"
    exact_ref_allowlist = "{" + ", ".join(EXACT_REFERENCE_ENCODINGS) + "}"

    display(Markdown(f"""
### Main Panel: Strict Exact-Reference Comparable Slice

**Current Status:** Empty slice (0 rows)

**Interpretation:**

This notebook is a strict reviewer-facing exact-reference panel. The current empty final slice reflects the scope of the loaded canonical artifacts, not a notebook defect. The notebook intentionally does not substitute exploratory encodings for missing exact-reference coverage.

**Evidence:** In the current artifact scope, all {len(quality_master)} comparable rows use encoding={encoding_list}. None match the exact-reference allowlist `{exact_ref_allowlist}`.

**Artifact-Scope Context:** Exact-reference comparable rows may be present in other matrices (e.g., matrix_a) but are not included in the current Stage E canonical artifacts. The canonical artifacts reflect the matrices processed by the upstream report-generation scope.

**Broader Evidence:** Execution coverage and scaling evidence across multiple encoding backends (including exploratory) is available in other notebooks in this benchmark stack.
"""))
else:
    display(Markdown(f"""
### Main Panel: Strict Exact-Reference Comparable Slice

**Current Status:** {len(final_df)} rows in final slice
"""))
```

**Component 7: Final Closing Note**
- **Cell Type:** Markdown cell
- **Placement:** Insert NEW cell at end of notebook (after all existing cells)

### Components 4 and 5 Conditional Display Logic

**Display Order and Conditions:**

**Scenario A:** No exact-ref rows in loaded artifacts (`exact_ref_rows_pre_downstream` empty)
1. Skip Component 4 (nothing to show)
2. Display Component 5 interpretation (emphasizes "no exact-ref rows in artifacts")
3. Display Component 6 placeholder

**Scenario B:** Exact-ref rows loaded but all excluded by downstream filters (`len(exact_ref_rows_pre_downstream) > 0 and len(final_df) == 0`)
1. Display Component 4 showing exclusion reasons (appears in its current location after main panel rendering)
2. Display Component 5 interpretation (before main panel, can mention "see excluded exact-ref panel for details on downstream filters")
3. Display Component 6 placeholder

**Scenario C:** Non-empty final slice (`len(final_df) > 0`)
1. Component 4 may or may not display (only if some exact-ref rows were excluded)
2. Display Component 5 with non-empty status message
3. Display Component 6 normal plot/table

**Key Point:** Component 5 always appears immediately before Component 6 (main panel rendering). Component 4 appears after Component 6 if applicable.

### Tone Guide

**DO USE (Assertive, Confident):**
- "This notebook enforces..."
- "The empty state reflects..."
- "Evidence shows..."
- "This is by design, not a defect"
- "The notebook intentionally does not substitute..."
- "Current Status: Empty slice (0 rows)" [neutral statement]

**DO NOT USE (Apologetic, Uncertain):**
- "Unfortunately, no rows..."
- "We apologize that..."
- "Limited by..."
- "Only X rows..." [use "X rows in final slice" instead]
- "The notebook cannot..."
- "Missing exact-reference..."

**Principle:** Treat empty state as an audited structural fact, not a failure or limitation requiring apology.

### Variable Scope and Cell Dependencies

**Critical Scope Requirements:**
- Component 1: Requires `quality_master`, `EXACT_REFERENCE_ENCODINGS` (available after data loading cell)
- Component 5: Requires `final_df`, `quality_master`, `encoding_col`, `EXACT_REFERENCE_ENCODINGS` (available within filtering pipeline cell)
- Component 6: Requires `final_df` (available within filtering pipeline cell)

**Implementation Strategy:**
Keep filtering pipeline logic in the existing single Python cell to maintain variable scope. Components 5 and 6 should be implemented within this cell, not as separate cells.

**Cell Structure:**
- Component 1: NEW standalone Python cell (after data loading)
- Components 2-6: Within EXISTING filtering pipeline Python cell (enhanced)
- Component 7: NEW standalone Markdown cell (at end)

## Implementation Notes

### Cell Insertion Map

**Current Notebook Structure:**
1. Markdown: "Notebook 12: Quality vs Cost"
2. Python: Data loading (imports, load quality_master, load unavailable)
3. Markdown: "Default Mainline Slice (with audit)"
4. Python: Filtering pipeline (exact-ref filter, downstream filters, audit table, excluded exact-ref panel, main plot/table)
5. Python: Unavailable transparency panel

**New Notebook Structure:**
1. Markdown: "Notebook 12: Quality vs Cost" [UNCHANGED]
2. Python: Data loading [UNCHANGED]
3. **[NEW]** Python: Artifact-scope note with encoding inspection (Component 1)
4. Markdown: "Default Mainline Slice (with audit)" [HEADER ENHANCED - clearer wording]
5. Python: Filtering pipeline [ENHANCED - Components 2, 4, 5, 6]
   - Enhanced Component 2: Explicit exact-ref match count in pre-filter audit
   - Enhanced Component 4: Clarified subordinate header for excluded exact-ref panel
   - New Component 5: Empty-state interpretation (before plot rendering)
   - Enhanced Component 6: Use `placeholder_plot()` instead of `unavailable_panel_table()`
6. Python: Unavailable transparency panel [UNCHANGED]
7. **[NEW]** Markdown: Final closing note (Component 7)

**Cell Types Summary:**
- **New cells:** 2 (Components 1 and 7)
- **Enhanced cells:** 2 (markdown header, filtering pipeline Python cell)
- **Total cells:** 7 (was 5, now 7)

### Code Changes

**Minimal code changes required:**
- Add encoding inspection code in early artifact-scope note cell
- Replace `unavailable_panel_table()` with `placeholder_plot()` in main panel empty-state handling
- All other changes are markdown/narrative enhancements

**No changes to:**
- `EXACT_REFERENCE_ENCODINGS` allowlist
- `_track_filter()` function or filter pipeline logic
- Exclusion reason computation
- Core filtering semantics

## Validation Plan

**Step 1:** Run notebook with current artifacts (empty exact-ref case)
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/12_quality_vs_cost.ipynb
```

**Step 2:** Verify all success criteria
- [ ] Exact-reference allowlist unchanged
- [ ] Early artifact-scope note present with concrete evidence
- [ ] Encoding-label audit shows explicit exact-ref count
- [ ] Cumulative exclusion audit preserved
- [ ] Empty-state interpretation before placeholder
- [ ] Visible placeholder plot renders
- [ ] No WHT rows in main panel
- [ ] Confident assertive tone throughout

**Step 3:** Check empty-state rendering with specific output verification
- [ ] Early artifact-scope note visible (contains "This notebook is a **strict reviewer-facing exact-reference panel**")
- [ ] Artifact-scope note shows concrete evidence (actual encoding counts from loaded data, not generic warnings)
- [ ] Empty-state interpretation visible (contains "The current empty final slice reflects the scope")
- [ ] Interpretation appears BEFORE placeholder plot in cell output order
- [ ] Placeholder plot renders successfully (matplotlib Figure object, no runtime errors)
- [ ] Placeholder plot contains expected text ("No exact-reference comparable rows")
- [ ] Table shows "**Table:** Empty (see interpretation above)" when final slice is empty
- [ ] No silent blank outputs (all empty states have visible placeholders or markdown)
- [ ] Final closing note visible at end (contains "Notebook 12 is the **strict exact-reference admissibility")

**Step 4:** Run tests if applicable
```bash
python3 -m pytest tests/ -x
```

## Deliverables

1. **Patched notebook:** `notebooks/12_quality_vs_cost.ipynb`
2. **Summary report** including:
   - Exact-reference allowlist used: `{ilp_derived_exact, ilp_derived_exact_where_feasible}`
   - Encoding labels present in current artifacts: `wht_truncated_pricing (6 rows)`
   - Final strict slice status: Empty (0 rows)
   - How notebook explains empty state: Three-layer architecture with early artifact-scope note, assertive interpretation before placeholder, and final closing note

## References

- Current notebook: `notebooks/12_quality_vs_cost.ipynb`
- Helper functions: `notebooks/_helpers.py` (unavailable_panel_table, placeholder_plot)
- Encoding infrastructure: `src/encoding_compare.py`
- Report schema: `src/report_schema.py`
- Upstream audit results: documented in NOTEBOOK_11_UNAVAILABILITY_CATEGORIZATION.md
