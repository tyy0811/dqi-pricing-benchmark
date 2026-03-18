# Notebook 12: Audited Structural-Limitation Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Notebook 12 into a reviewer-safe audited structural-limitation notebook that explains empty states with confidence rather than appearing broken.

**Architecture:** Three-layer information architecture: (1) Artifact-Scope Context Layer establishes loaded data, (2) Audit Layer tracks filtering, (3) Interpretation Layer explains empty states assertively. No changes to exact-reference allowlist or core filtering logic.

**Tech Stack:** Jupyter Notebook, Python, Pandas, Matplotlib, IPython display

---

## File Structure

**Modified:**
- `notebooks/12_quality_vs_cost.ipynb` - Add 2 new cells, enhance filtering pipeline cell, update markdown headers

**No new files created** - This is a pure notebook enhancement

---

## Chunk 1: Add Early Artifact-Scope Note (Component 1)

### Task 1: Insert Early Artifact-Scope Note Cell

**Files:**
- Modify: `notebooks/12_quality_vs_cost.ipynb` - Insert new Python cell after cell `ee8f0f03` (data loading)

- [ ] **Step 1: Read current notebook structure**

Verify current cell order:
1. Cell `8abc3c5b` (markdown): Notebook title and description
2. Cell `ee8f0f03` (code): Data loading imports and quality_master load
3. Cell `1fd51455` (markdown): "Default Mainline Slice (with audit)" header
4. Cell `2bceaf33` (code): Filtering pipeline with _track_filter()

Expected: 5 cells total (including Pareto view and unavailable transparency)

- [ ] **Step 2: Create new Python cell with artifact-scope note**

Insert new cell between `ee8f0f03` and `1fd51455` with this exact code:

```python
# Inspect loaded artifacts and display artifact-scope note
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
else:
    display(Markdown("""
## Artifact Scope and Exact-Reference Policy

This notebook is a **strict reviewer-facing exact-reference panel**. It enforces an explicit exact-reference encoding allowlist and does not substitute exploratory encodings for missing exact-reference coverage.

**Note:** Encoding column not found in loaded artifacts. Cannot display detailed artifact composition.
"""))
```

- [ ] **Step 3: Verify cell insertion**

Check notebook structure:
- New cell appears after data loading (cell `ee8f0f03`)
- New cell appears before "Default Mainline Slice" header (cell `1fd51455`)
- Cell type is "code" (Python)
- Variable `EXACT_REFERENCE_ENCODINGS` must be defined before this cell runs

Expected: This cell will error on first run because `EXACT_REFERENCE_ENCODINGS` is defined in cell `2bceaf33`. This is intentional and will be fixed in Task 2.

- [ ] **Step 4: Save notebook**

Save changes to `notebooks/12_quality_vs_cost.ipynb`

Expected: Notebook now has 6 cells (was 5, added 1)

---

### Task 2: Move EXACT_REFERENCE_ENCODINGS Definition

**Files:**
- Modify: `notebooks/12_quality_vs_cost.ipynb:cell[2bceaf33]` - Move constant definition to top of cell

- [ ] **Step 1: Locate EXACT_REFERENCE_ENCODINGS in filtering pipeline cell**

Find this line in cell `2bceaf33`:
```python
EXACT_REFERENCE_ENCODINGS = {"ilp_derived_exact", "ilp_derived_exact_where_feasible"}
```

Current location: ~Line 70 in cell `2bceaf33`, after all helper function definitions

- [ ] **Step 2: Move constant to data loading cell**

Cut the `EXACT_REFERENCE_ENCODINGS` line from cell `2bceaf33` and paste it at the END of cell `ee8f0f03` (data loading cell), right after the `ensure_run_status_norm()` call:

```python
# ... existing data loading code ...
if not quality_master.empty:
    quality_master = ensure_run_status_norm(quality_master)

# Canonical exact-reference encoding allowlist (notebook-local policy).
# Must match the upstream labels used by encoding_compare.PRIMARY_EXACT_REFERENCE_BACKEND
# and the Stage E normalization layer. WHT variants are NOT exact-reference.
EXACT_REFERENCE_ENCODINGS = {"ilp_derived_exact", "ilp_derived_exact_where_feasible"}
```

- [ ] **Step 3: Verify constant is accessible**

Run the new artifact-scope note cell (inserted in Task 1).

Expected: No NameError. The cell should display artifact-scope note with concrete encoding counts.

- [ ] **Step 4: Save notebook**

Save changes.

---

## Chunk 2: Enhance Filtering Pipeline Cell (Components 2, 4, 5, 6)

### Task 3: Enhance Pre-Filter Encoding Audit (Component 2)

**Files:**
- Modify: `notebooks/12_quality_vs_cost.ipynb:cell[2bceaf33]` - Enhance encoding audit section

- [ ] **Step 1: Locate existing encoding audit code**

Find this section in cell `2bceaf33`:
```python
# Encoding label audit (per spec Part 2, Section 2)
display(Markdown("### Encoding Label Audit (Pre-Filter)"))
display(Markdown(f"**Total comparable rows before exact-reference filter:** {len(default_slice)}"))
encoding_counts = default_slice[encoding_norm].value_counts(dropna=False).rename_axis("encoding_label").reset_index(name="count")
display(encoding_counts)
display(Markdown(f"*Exact-reference allowlist: {sorted(EXACT_REFERENCE_ENCODINGS)}*"))
```

- [ ] **Step 2: Replace with enhanced audit**

Replace the entire encoding audit section with:

```python
# Enhanced encoding label audit (Component 2)
display(Markdown("### Pre-Filter Artifact Composition"))
display(Markdown(f"**Total comparable rows loaded from quality_cost_master:** {len(default_slice)}"))
display(Markdown(""))
display(Markdown("**Encodings present in comparable slice:**"))

encoding_counts = default_slice[encoding_norm].value_counts(dropna=False)

# Explicitly show exact-reference encodings with counts (even if zero)
encoding_display = []
for enc in sorted(EXACT_REFERENCE_ENCODINGS):
    count = encoding_counts.get(enc, 0)
    encoding_display.append(f"- {enc}: {count}")

# Show other encodings
other_encodings = [enc for enc in encoding_counts.index if enc not in EXACT_REFERENCE_ENCODINGS]
for enc in sorted(other_encodings):
    encoding_display.append(f"- {enc}: {encoding_counts[enc]}")

for line in encoding_display:
    display(Markdown(line))

display(Markdown(""))
display(Markdown(f"**Exact-reference allowlist:** `{{{', '.join(sorted(EXACT_REFERENCE_ENCODINGS))}}}`"))

# Compute exact-reference match count
exact_ref_match_count = default_slice[encoding_norm].astype(str).str.lower().isin([e.lower() for e in EXACT_REFERENCE_ENCODINGS]).sum()
display(Markdown(f"**Rows matching allowlist before filtering:** {exact_ref_match_count}"))
```

- [ ] **Step 3: Verify enhanced audit displays correctly**

Run the filtering pipeline cell.

Expected output:
- Header: "Pre-Filter Artifact Composition"
- Shows total rows
- Lists all encodings with counts (exact-ref encodings listed first, even if count=0)
- Shows allowlist
- Shows match count

- [ ] **Step 4: Save notebook**

---

### Task 4: Enhance Excluded Exact-Reference Panel Header (Component 4)

**Files:**
- Modify: `notebooks/12_quality_vs_cost.ipynb:cell[2bceaf33]` - Update context panel header

- [ ] **Step 1: Locate excluded exact-ref panel section**

Find this line in cell `2bceaf33`:
```python
display(Markdown("### Excluded Exact-Reference Rows (Context Only)"))
```

- [ ] **Step 2: Replace header with clarified version**

Replace that single line with this enhanced header:

```python
display(Markdown("### Context: Exact-Reference Rows Excluded by Downstream Filters"))
display(Markdown(""))
display(Markdown("*This panel shows exact-reference rows that passed the encoding allowlist but were removed by downstream scope filters (mixture-only, decoder scope). It provides explanatory context only and is not part of the main comparable panel.*"))
display(Markdown(""))
```

- [ ] **Step 3: Verify header renders correctly**

Run the filtering pipeline cell with a dataset that has exact-ref rows (for testing, you can temporarily modify the allowlist to include "wht_truncated_pricing").

Expected: Enhanced header with subordinate explanatory note appears before the excluded rows table.

- [ ] **Step 4: Restore original allowlist and save**

If you modified the allowlist for testing, restore it to:
```python
EXACT_REFERENCE_ENCODINGS = {"ilp_derived_exact", "ilp_derived_exact_where_feasible"}
```

Save notebook.

---

### Task 5: Add Empty-State Interpretation Block (Component 5)

**Files:**
- Modify: `notebooks/12_quality_vs_cost.ipynb:cell[2bceaf33]` - Add interpretation before main panel rendering

- [ ] **Step 1: Locate main panel rendering section**

Find this section near the end of cell `2bceaf33`:

```python
if table_available:
    # ... display table code ...
else:
    display(unavailable_panel_table(
        panel=panel,
        reason="No rows remain in the default mainline slice after exact-reference/mixture/bp1-or-bruteforce/non-WHT filters.",
        details="The filter stack is preserved; the result set is empty.",
    ))
```

This is the section that displays results or unavailability message.

- [ ] **Step 2: Add empty-state interpretation BEFORE the if/else block**

Insert this code immediately BEFORE the `if table_available:` line:

```python
# Empty-state interpretation block (Component 5)
if final_count == 0:
    # Display assertive interpretation when slice is empty
    encoding_list = ", ".join(sorted(quality_master[encoding_norm].unique())) if encoding_norm and encoding_norm in quality_master.columns else "unknown"
    exact_ref_allowlist = "{" + ", ".join(sorted(EXACT_REFERENCE_ENCODINGS)) + "}"

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
    # Display non-empty status message
    display(Markdown(f"""
### Main Panel: Strict Exact-Reference Comparable Slice

**Current Status:** {final_count} rows in final slice
"""))

# Continue with existing table display logic
```

- [ ] **Step 3: Verify interpretation appears before table/placeholder**

Run the filtering pipeline cell.

Expected: When empty (current state with wht_truncated_pricing only):
- "Main Panel" header appears
- "Current Status: Empty slice (0 rows)" appears
- Full interpretation with evidence appears
- THEN placeholder/table appears below

- [ ] **Step 4: Save notebook**

---

### Task 6: Replace Placeholder with Visible Plot (Component 6)

**Files:**
- Modify: `notebooks/12_quality_vs_cost.ipynb:cell[2bceaf33]` - Replace unavailable_panel_table with placeholder_plot

- [ ] **Step 1: Locate unavailable_panel_table call**

Find this code in the `else` branch of the table display section:

```python
else:
    display(unavailable_panel_table(
        panel=panel,
        reason="No rows remain in the default mainline slice after exact-reference/mixture/bp1-or-bruteforce/non-WHT filters.",
        details="The filter stack is preserved; the result set is empty.",
    ))
```

- [ ] **Step 2: Replace with visible placeholder plot**

Replace the entire `else` block with:

```python
else:
    # Use visible placeholder plot for empty state (Component 6)
    # Note: placeholder_plot() returns (fig, ax) tuple
    fig, ax = placeholder_plot(
        title="Strict Exact-Reference Quality vs Cost Comparison",
        reason="No exact-reference comparable rows in current artifact scope (see interpretation above)",
        figsize=(10, 4)
    )
    display(fig)
    plt.close(fig)

    # Display empty table message
    display(Markdown("**Table:** Empty (see interpretation above)"))
```

- [ ] **Step 3: Verify placeholder plot renders**

Run the filtering pipeline cell.

Expected:
- Visible matplotlib figure appears (not just text table)
- Figure shows title and unavailability reason
- Message references "see interpretation above"
- Table message also references interpretation

- [ ] **Step 4: Save notebook**

---

### Task 7: Update Structural Limitation Note

**Files:**
- Modify: `notebooks/12_quality_vs_cost.ipynb:cell[2bceaf33]` - Update structural limitation section

- [ ] **Step 1: Locate structural limitation note**

Find this section near the end of cell `2bceaf33`:

```python
# Structural limitation / interpretation note
display(Markdown("### Structural Limitation Note"))
structural_note = """
This notebook is a **strict reviewer-facing exact-reference panel**.

- The main panel is limited to rows passing the explicit exact-reference allowlist and downstream reviewer-scope filters.
- Broad execution coverage is handled elsewhere in the benchmark stack.
- In the current canonical Stage E artifacts, exact-reference coverage is concentrated in C-family rows.

The thin final slice reflects **coverage limits of the exact-reference comparable backend**, not a notebook defect.
"""
display(Markdown(structural_note))
```

- [ ] **Step 2: Remove this section entirely**

DELETE the entire structural limitation note section (from `# Structural limitation` comment through the final `display(Markdown(interp_note))` or `display(Markdown("- **Slice support level:** unavailable (0 rows in final slice)."))`).

Reason: This information is now covered by Component 5 (empty-state interpretation) and will be covered by Component 7 (final closing note). The old structural note is redundant and less assertive.

- [ ] **Step 3: Verify cell still runs**

Run the filtering pipeline cell.

Expected: No errors. The empty state is now explained by Component 5 interpretation block only.

- [ ] **Step 4: Save notebook**

---

## Chunk 3: Add Final Closing Note (Component 7)

### Task 8: Add Final Closing Note Cell

**Files:**
- Modify: `notebooks/12_quality_vs_cost.ipynb` - Insert new Markdown cell at end of notebook

- [ ] **Step 1: Identify last cell in notebook**

Current last cell: `a6d7ff7b` (code) - Unavailable transparency display

- [ ] **Step 2: Insert new Markdown cell after last cell**

Add new Markdown cell with this exact content:

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

- [ ] **Step 3: Verify cell renders correctly**

Run the entire notebook (or just view the new cell).

Expected: Markdown renders with proper formatting, headers, bullets, bold text.

- [ ] **Step 4: Save notebook**

Expected: Notebook now has 7 cells total (was 5, added 2: artifact-scope note + final closing note)

---

## Chunk 4: Validation and Testing

### Task 9: Run Full Notebook and Verify Output

**Files:**
- Test: `notebooks/12_quality_vs_cost.ipynb`

- [ ] **Step 1: Execute notebook with current artifacts**

Run:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/12_quality_vs_cost.ipynb
```

Expected: Notebook executes without errors

- [ ] **Step 2: Verify artifact-scope note (Component 1)**

Check output of new cell (between data loading and "Default Mainline Slice"):
- [ ] Header: "Artifact Scope and Exact-Reference Policy"
- [ ] Shows "strict reviewer-facing exact-reference panel"
- [ ] Shows allowlist: `{ilp_derived_exact, ilp_derived_exact_where_feasible}`
- [ ] Shows concrete encoding counts (e.g., "wht_truncated_pricing: 6")
- [ ] Shows exact-reference rows present: 0

- [ ] **Step 3: Verify enhanced encoding audit (Component 2)**

Check filtering pipeline cell output:
- [ ] Header: "Pre-Filter Artifact Composition"
- [ ] Explicitly lists ilp_derived_exact: 0
- [ ] Explicitly lists ilp_derived_exact_where_feasible: 0
- [ ] Shows other encodings (wht_truncated_pricing: 6)
- [ ] Shows "Rows matching allowlist before filtering: 0"

- [ ] **Step 4: Verify empty-state interpretation (Component 5)**

Check filtering pipeline cell output:
- [ ] Header: "Main Panel: Strict Exact-Reference Comparable Slice"
- [ ] "Current Status: Empty slice (0 rows)"
- [ ] Contains "This notebook is a strict reviewer-facing exact-reference panel"
- [ ] Contains "Evidence: In the current artifact scope, all 6 comparable rows use encoding=wht_truncated_pricing"
- [ ] Interpretation appears BEFORE placeholder plot

- [ ] **Step 5: Verify placeholder plot (Component 6)**

Check filtering pipeline cell output:
- [ ] Visible matplotlib figure renders (not just text table)
- [ ] Figure title: "Strict Exact-Reference Quality vs Cost Comparison"
- [ ] Figure reason mentions "No exact-reference comparable rows"
- [ ] Figure reason mentions "see interpretation above"
- [ ] Table message: "**Table:** Empty (see interpretation above)"

- [ ] **Step 6: Verify final closing note (Component 7)**

Check last cell output:
- [ ] Header: "Notebook Role and Scope"
- [ ] Contains "strict exact-reference admissibility and comparability panel"
- [ ] "What this notebook does" section present with 4 bullets
- [ ] "What this notebook does not do" section present with 3 bullets
- [ ] "Artifact-Scope Dependency" section present
- [ ] Tone is assertive, not apologetic

- [ ] **Step 7: Verify cumulative exclusion audit (Component 3)**

Check filtering pipeline cell output:
- [ ] "Slice audit" section still present
- [ ] Audit table shows step-by-step filtering with row counts
- [ ] Audit table shows "exact-reference allowlist" step with 6 rows removed

- [ ] **Step 8: Verify excluded exact-ref panel (Component 4)**

Since current artifacts have no exact-ref rows, this panel should NOT appear.
- [ ] Section "Context: Exact-Reference Rows Excluded by Downstream Filters" does NOT appear (expected for current data)

- [ ] **Step 9: Check for tone compliance**

Review all markdown output:
- [ ] No apologetic language ("Unfortunately", "We apologize", "Limited by")
- [ ] Assertive confident tone throughout ("enforces", "reflects", "by design not defect")
- [ ] Empty state treated as audited fact, not failure

- [ ] **Step 10: Verify no breaking changes**

Check that core functionality is preserved:
- [ ] EXACT_REFERENCE_ENCODINGS allowlist unchanged (still {ilp_derived_exact, ilp_derived_exact_where_feasible})
- [ ] Filter pipeline logic unchanged (_track_filter calls intact)
- [ ] Pareto plot cell still renders (unavailable placeholder for empty case)
- [ ] Unavailable transparency panel still renders at end

---

### Task 10: Run Tests

**Files:**
- Test: `tests/` directory

- [ ] **Step 1: Run pytest**

```bash
python3 -m pytest tests/ -x -v
```

Expected: All tests pass (or no new failures introduced)

- [ ] **Step 2: If any test failures, investigate**

Check if failures are related to notebook changes:
- Changes to EXACT_REFERENCE_ENCODINGS location should not affect tests
- New cells should not affect existing test logic
- Placeholder plot usage should not break tests

If failures are unrelated to this refactor, note them but proceed.

---

## Chunk 5: Final Summary and Commit

### Task 11: Create Summary Report

**Files:**
- Create: Summary output (display in console, do not create file)

- [ ] **Step 1: Generate summary report**

Display this summary:

```
================================================================
Notebook 12: Audited Structural-Limitation Refactor - COMPLETE
================================================================

DELIVERABLE: notebooks/12_quality_vs_cost.ipynb

SUMMARY:
- Exact-reference allowlist: {ilp_derived_exact, ilp_derived_exact_where_feasible}
- Current artifact encodings: wht_truncated_pricing (6 rows)
- Final strict slice status: Empty (0 rows)

HOW EMPTY STATE IS EXPLAINED:
1. Early artifact-scope note (after data loading)
   - Shows concrete loaded data
   - States exact-reference policy
   - Shows encoding counts and exact-ref match count

2. Enhanced pre-filter encoding audit
   - Explicitly lists exact-ref encodings with counts (even if zero)
   - Shows rows matching allowlist before filtering

3. Empty-state interpretation block (before main panel)
   - Assertive explanation treating emptiness as audited fact
   - Provides concrete evidence from loaded data
   - References artifact-scope dependency

4. Visible placeholder plot (replacing text table)
   - Visible matplotlib figure for empty state
   - References "see interpretation above"

5. Final closing note (end of notebook)
   - Summarizes notebook role and scope
   - Clarifies what notebook does/doesn't do
   - Explains artifact-scope dependency

CELL STRUCTURE:
- Was 5 cells → Now 7 cells
- Added: 2 new cells (artifact-scope note, final closing note)
- Enhanced: 1 cell (filtering pipeline with Components 2, 4, 5, 6)

SUCCESS CRITERIA VERIFIED:
✓ Remains strict exact-reference only
✓ Artifact-scope limitation explained early with concrete evidence
✓ Encoding-label audit shows explicit exact-ref counts
✓ Cumulative exclusion audit preserved
✓ Reviewer-safe empty-state interpretation before placeholder
✓ Visible placeholder for empty state
✓ No WHT rows in main panel
✓ Confident assertive tone throughout

================================================================
```

- [ ] **Step 2: Display summary**

Expected: Summary shows completed refactor with all success criteria met

---

### Task 12: Final Commit

**Files:**
- Commit: `notebooks/12_quality_vs_cost.ipynb`

- [ ] **Step 1: Review changes**

```bash
git diff notebooks/12_quality_vs_cost.ipynb
```

Expected changes:
- New cell after data loading (artifact-scope note)
- Modified filtering pipeline cell (enhanced audit, interpretation, placeholder)
- New cell at end (closing note)
- EXACT_REFERENCE_ENCODINGS moved to data loading cell

- [ ] **Step 2: Stage changes**

```bash
git add notebooks/12_quality_vs_cost.ipynb
```

- [ ] **Step 3: Commit with descriptive message**

```bash
git commit -m "refactor: notebook 12 audited structural-limitation

- Add early artifact-scope note with concrete loaded data evidence
- Enhance pre-filter encoding audit with explicit exact-ref counts
- Add assertive empty-state interpretation before main panel
- Replace unavailable_panel_table with visible placeholder_plot
- Add final closing note explaining notebook role and scope

Implements three-layer information architecture:
- Artifact-Scope Context Layer (Component 1)
- Audit Layer enhanced (Components 2, 3, 4)
- Interpretation Layer (Components 5, 7)

Empty state now reads as audited structural fact, not defect.
Strict exact-reference policy preserved throughout.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Verify commit**

```bash
git log -1 --stat
```

Expected: Commit shows modification to notebooks/12_quality_vs_cost.ipynb

---

## Implementation Notes

**Key Constraints:**
- Maintain strict exact-reference allowlist (no WHT substitution)
- Preserve all existing filtering logic and audit functionality
- Use assertive confident tone (not apologetic)
- Ensure Components 5 interpretation appears BEFORE Component 6 placeholder

**Variable Scope Requirements:**
- Component 1: Needs `EXACT_REFERENCE_ENCODINGS`, `quality_master` → Define in data loading cell
- Components 5-6: Need `final_df`, `quality_master`, `encoding_norm` → Implemented within filtering pipeline cell
- All variables properly scoped by keeping filtering pipeline in single cell

**Cell Structure:**
- Component 1: NEW standalone Python cell (after data loading)
- Components 2-6: Within EXISTING filtering pipeline cell (enhanced)
- Component 7: NEW standalone Markdown cell (at end)

**Testing Strategy:**
- Execute full notebook to verify output
- Verify each component renders correctly
- Check tone compliance (no apologetic language)
- Confirm no breaking changes to core functionality

---

## Success Criteria Checklist

After implementation, verify all criteria:

- [ ] Remains strict exact-reference only (allowlist unchanged)
- [ ] Artifact-scope limitation explained early (Component 1)
- [ ] Encoding-label audit obvious (Component 2)
- [ ] Cumulative exclusion audit preserved (Component 3)
- [ ] Empty-state interpretation before placeholder (Component 5)
- [ ] Visible placeholder for empty state (Component 6)
- [ ] No WHT rows in main panel
- [ ] Confident assertive tone throughout
- [ ] Final closing note present (Component 7)
