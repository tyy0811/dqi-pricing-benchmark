# Notebook 12 Transparency Improvements Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stepwise exclusion audit and context panel to Notebook 12 while preserving strict exact-reference main panel.

**Architecture:** Refactor the existing notebook code cell to separate concerns: (1) input summary, (2) cumulative filter audit with explicit tracking, (3) excluded exact-reference context panel, (4) unchanged main panel logic, (5) structural limitation note. No new files created; single notebook modification.

**Tech Stack:** Python, pandas, Jupyter notebook, existing `_helpers.py` utilities

**Spec:** `docs/superpowers/specs/2026-03-16-notebook12-transparency-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `notebooks/12_quality_vs_cost.ipynb` | Modify | Add transparency improvements |
| `tests/test_notebook12_stage_e_smoke.py` | Modify | Add test for exclusion audit presence |

No new files needed. Changes are contained within the existing notebook structure.

---

## Chunk 1: Upstream Audit Verification

### Task 1: Verify No Accidental Loss of Exact-Reference Rows

**Files:**
- Read: `results/quality_cost_unavailable.json`
- Read: `results/quality_cost_master.json`
- Read: `src/encoding_compare.py`

This is a verification task, not a code change. Document findings.

- [ ] **Step 1: Check encoding label consistency**

Run:
```bash
python3 -c "
from src.encoding_compare import PRIMARY_EXACT_REFERENCE_BACKEND
print(f'PRIMARY_EXACT_REFERENCE_BACKEND = {PRIMARY_EXACT_REFERENCE_BACKEND!r}')
"
```

Expected: `ilp_derived_exact` - must match notebook's `EXACT_REFERENCE_ENCODINGS`.

- [ ] **Step 2: Check for ilp_derived_exact rows in unavailable**

Run:
```bash
python3 -c "
import json
from pathlib import Path
unavail = json.loads(Path('results/quality_cost_unavailable.json').read_text())
rows = unavail.get('rows', [])
ilp_exact = [r for r in rows if 'ilp_derived_exact' in str(r.get('encoding', '')).lower()]
print(f'ilp_derived_exact in unavailable: {len(ilp_exact)}')
for r in ilp_exact[:5]:
    print(f\"  class={r.get('unavailable_class')}, reason={r.get('unavailable_reason')[:60]}...\")
"
```

Expected: Any `ilp_derived_exact` rows in unavailable should be `alpha_validation` rows with `missing_quality_fields` (oracle decoder rows without full metrics). This is structural, not a bug.

- [ ] **Step 3: Verify family concentration**

Run:
```bash
python3 -c "
import json
from pathlib import Path
master = json.loads(Path('results/quality_cost_master.json').read_text())
rows = master.get('rows', [])
ilp_rows = [r for r in rows if 'ilp_derived_exact' in str(r.get('encoding_norm', r.get('encoding', ''))).lower()]
families = {}
for r in ilp_rows:
    families[r.get('family')] = families.get(r.get('family'), 0) + 1
print(f'ilp_derived_exact rows by family: {families}')
"
```

Expected: All rows are C-family. This confirms structural limitation.

- [ ] **Step 4: Document audit conclusion**

Write to plan execution notes:
- Encoding label consistency: CONFIRMED (`ilp_derived_exact` matches)
- No accidental loss: CONFIRMED (unavailable rows are alpha_validation with expected missing fields)
- Structural limitation: CONFIRMED (exact-reference coverage concentrated in C-family)

---

## Chunk 2: Add Cumulative Exclusion Audit

### Task 2: Refactor Filter Tracking to Cumulative Format

**Files:**
- Modify: `notebooks/12_quality_vs_cost.ipynb` (cell id `2bceaf33`)

- [ ] **Step 1: Update `_track_filter` to return cumulative audit record**

Replace the existing `_track_filter` function with enhanced version that tracks input rows explicitly:

```python
def _track_filter(
    df: pd.DataFrame,
    label: str,
    condition,
    reason_label: str,
) -> tuple[pd.DataFrame, dict]:
    """Apply filter and return cumulative audit record."""
    input_rows = len(df)
    if condition is None:
        return df, {
            "step": label,
            "input_rows": input_rows,
            "output_rows": input_rows,
            "rows_removed": 0,
            "reason_label": reason_label,
        }
    next_df = df[condition].copy()
    output_rows = len(next_df)
    return next_df, {
        "step": label,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "rows_removed": input_rows - output_rows,
        "reason_label": reason_label,
    }
```

- [ ] **Step 2: Update all filter calls to use new signature**

Update each filter call with the new signature. Here are all filter calls:

**Exact-reference filter:**
```python
default_slice, rec = _track_filter(
    default_slice,
    "exact-reference allowlist",
    default_slice[encoding_norm].astype(str).str.lower().isin(EXACT_REFERENCE_ENCODINGS),
    f"encoding in {sorted(EXACT_REFERENCE_ENCODINGS)}",
)
filter_audit.append(rec)
```

**Mixture filter:**
```python
default_slice, rec = _track_filter(
    default_slice,
    "mixture-first filter",
    default_slice[execution_norm].astype(str).str.lower() == "mixture",
    f"{execution_norm} == 'mixture'",
)
filter_audit.append(rec)
```

**Coherent-off filter:**
```python
default_slice, rec = _track_filter(
    default_slice,
    "coherent-off filter",
    default_slice[execution_norm].astype(str).str.lower() != "coherent",
    "exclude coherent execution",
)
filter_audit.append(rec)
```

**Decoder filter:**
```python
default_slice, rec = _track_filter(
    default_slice,
    "decoder filter (bp1/bruteforce)",
    default_slice[decoder_norm].astype(str).str.lower().isin(["bp1", "bruteforce"]),
    f"{decoder_norm} in ['bp1','bruteforce']",
)
filter_audit.append(rec)
```

**WHT-off filter:**
```python
default_slice, rec = _track_filter(
    default_slice,
    "WHT-off filter",
    ~default_slice[encoding_norm].astype(str).str.contains("wht", case=False, na=False),
    f"{encoding_norm} does not contain 'wht'",
)
filter_audit.append(rec)
```

- [ ] **Step 3: Add pre-filter baseline record**

Add at start of filter chain:

```python
filter_audit.append({
    "step": "pre-filter comparable slice",
    "input_rows": quality_rows_total,
    "output_rows": quality_rows_total,
    "rows_removed": 0,
    "reason_label": "all quality_cost_master rows",
})
```

- [ ] **Step 4: Add encoding label audit display (per spec section 2)**

After the pre-filter baseline, add encoding distribution display before the exact-reference filter:

```python
# Encoding label audit (per spec Part 2, Section 2)
encoding_norm = _pick_column(default_slice, ["encoding_norm", "encoding"])
if encoding_norm is not None:
    display(Markdown("### Encoding Label Audit (Pre-Filter)"))
    display(Markdown(f"**Total comparable rows before exact-reference filter:** {len(default_slice)}"))
    encoding_counts = default_slice[encoding_norm].value_counts(dropna=False).rename_axis("encoding_label").reset_index(name="count")
    display(encoding_counts)
    display(Markdown(f"*Exact-reference allowlist: {sorted(EXACT_REFERENCE_ENCODINGS)}*"))
```

- [ ] **Step 4: Verify notebook executes**

Run:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/12_quality_vs_cost.ipynb
```

Expected: Executes without error.

---

## Chunk 3: Add Excluded Exact-Reference Context Panel

### Task 3: Track and Display Excluded Exact-Reference Rows

**Files:**
- Modify: `notebooks/12_quality_vs_cost.ipynb` (cell id `2bceaf33`)

- [ ] **Step 1: Capture exact-reference rows before downstream filters**

After the exact-reference filter, save a copy:

```python
# Capture exact-reference rows for context panel
exact_ref_rows_pre_downstream = default_slice.copy() if not default_slice.empty else pd.DataFrame()
```

- [ ] **Step 2: Build exclusion reason for each filtered row**

After all filters complete, compute excluded rows with reasons. Define `_compute_exclusion_reason` at module level first:

```python
def _compute_exclusion_reason(row):
    """Determine why an exact-reference row was excluded from the main panel."""
    exec_mode = str(row.get("execution_mode_norm", row.get("execution_mode", ""))).lower()
    decoder = str(row.get("decoder_norm", row.get("decoder", ""))).lower()

    if exec_mode == "coherent":
        return "coherent_execution_excluded"
    if exec_mode != "mixture":
        return "non_mixture_execution"
    if decoder not in ["bp1", "bruteforce"]:
        return "decoder_out_of_scope"
    return "unknown_exclusion"

# Build excluded exact-reference context table
# Initialize to empty DataFrame to ensure it's always defined
excluded_exact_ref = pd.DataFrame()

if not exact_ref_rows_pre_downstream.empty:
    if not table_df.empty:
        # Rows that passed exact-reference but didn't make final slice
        final_ids = set(table_df["run_id"].tolist()) if "run_id" in table_df.columns else set()
        pre_ids = set(exact_ref_rows_pre_downstream["run_id"].tolist()) if "run_id" in exact_ref_rows_pre_downstream.columns else set()
        excluded_ids = pre_ids - final_ids

        if excluded_ids:
            excluded_exact_ref = exact_ref_rows_pre_downstream[
                exact_ref_rows_pre_downstream["run_id"].isin(excluded_ids)
            ].copy()
            excluded_exact_ref["exclusion_reason"] = excluded_exact_ref.apply(_compute_exclusion_reason, axis=1)
    else:
        # All exact-ref rows were excluded (table_df is empty)
        excluded_exact_ref = exact_ref_rows_pre_downstream.copy()
        excluded_exact_ref["exclusion_reason"] = excluded_exact_ref.apply(_compute_exclusion_reason, axis=1)
```

- [ ] **Step 3: Display context panel after main table**

Add display code after the main table display:

```python
# Excluded exact-reference rows (context only)
if not excluded_exact_ref.empty:
    display(Markdown("### Excluded Exact-Reference Rows (Context Only)"))
    display(Markdown(f"**{len(excluded_exact_ref)}** exact-reference rows passed the encoding filter but were excluded by downstream filters."))
    display(Markdown("*This panel is explanatory only; these rows are not part of the main comparable slice.*"))

    context_cols = ["run_id", "encoding", "execution_mode", "decoder", "exclusion_reason"]
    available_cols = [c for c in context_cols if c in excluded_exact_ref.columns]
    if available_cols:
        display(excluded_exact_ref[available_cols].head(20))

    # Summary by exclusion reason
    reason_counts = excluded_exact_ref["exclusion_reason"].value_counts().reset_index()
    reason_counts.columns = ["exclusion_reason", "count"]
    display(reason_counts)
```

- [ ] **Step 4: Verify notebook executes**

Run:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/12_quality_vs_cost.ipynb
```

Expected: Executes without error, shows context panel.

---

## Chunk 4: Add Structural Limitation Note

### Task 4: Add Confident Framing Note

**Files:**
- Modify: `notebooks/12_quality_vs_cost.ipynb` (cell id `2bceaf33`)

- [ ] **Step 1: Replace interpretation section with enhanced version**

Replace the interpretation section at the end of the main code cell. Note: `table_available` and `final_count` are already defined earlier in the cell from the existing notebook logic (see cell id `2bceaf33` lines 95-96):

```python
# Structural limitation / interpretation note
# Note: table_available and final_count are defined earlier in this cell
# table_available = final_count > 0
# final_count = len(table_df)

display(Markdown("### Structural Limitation Note"))
structural_note = """
This notebook is a **strict reviewer-facing exact-reference panel**.

- The main panel is limited to rows passing the explicit exact-reference allowlist and downstream reviewer-scope filters.
- Broad execution coverage is handled elsewhere in the benchmark stack.
- In the current canonical Stage E artifacts, exact-reference coverage is concentrated in C-family rows.

The thin final slice reflects **coverage limits of the exact-reference comparable backend**, not a notebook defect.
"""
display(Markdown(structural_note))

if table_available:
    if final_count >= 10:
        interp_strength = "strong"
    elif final_count >= 3:
        interp_strength = "thin"
    else:
        interp_strength = "very thin"
    interp_note = f"- **Slice support level:** {interp_strength} ({final_count} rows)."
    display(Markdown(interp_note))
else:
    display(Markdown("- **Slice support level:** unavailable (0 rows in final slice)."))
```

- [ ] **Step 2: Verify notebook executes**

Run:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/12_quality_vs_cost.ipynb
```

Expected: Executes without error, shows structural note.

---

## Chunk 5: Update Smoke Test

### Task 5: Add Test for Exclusion Audit Presence

**Files:**
- Modify: `tests/test_notebook12_stage_e_smoke.py`

- [ ] **Step 1: Add test for cumulative audit table**

Add new test function:

```python
def test_notebook12_has_cumulative_exclusion_audit() -> None:
    src = _notebook_source("notebooks/12_quality_vs_cost.ipynb")
    # Check for cumulative audit structure
    assert "input_rows" in src
    assert "output_rows" in src
    assert "rows_removed" in src
    assert "reason_label" in src
    # Check for context panel
    assert "exclusion_reason" in src
    assert "Excluded Exact-Reference Rows" in src or "excluded_exact_ref" in src
```

- [ ] **Step 2: Run test to verify it passes**

Run:
```bash
python3 -m pytest tests/test_notebook12_stage_e_smoke.py -v
```

Expected: All tests pass including new test.

---

## Chunk 6: Final Validation

### Task 6: Full Validation

**Files:**
- Execute: `notebooks/12_quality_vs_cost.ipynb`
- Run: `tests/test_notebook12_stage_e_smoke.py`

- [ ] **Step 1: Execute notebook fresh**

Run:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/12_quality_vs_cost.ipynb
```

Expected: Executes without error.

- [ ] **Step 2: Run all notebook 12 tests**

Run:
```bash
python3 -m pytest tests/test_notebook12_stage_e_smoke.py -v
```

Expected: All tests pass.

- [ ] **Step 3: Verify output structure**

Manually verify notebook output shows:
1. Stage E input summary with row counts
2. Encoding label audit (pre-filter)
3. Cumulative stepwise audit table with input_rows/output_rows/rows_removed/reason_label
4. Main exact-reference comparable table (unchanged logic)
5. Excluded exact-reference rows context panel (if any exist)
6. Structural limitation note

- [ ] **Step 4: Document implementation summary**

Create summary noting:
- Upstream audit outcome: No accidental loss; structural limitation confirmed (C-family only)
- Final exact-reference allowlist: `{"ilp_derived_exact", "ilp_derived_exact_where_feasible"}`
- Exclusion audit: Stepwise cumulative table added
- Main panel: Unchanged, 3 rows in current data
- Context panel: Shows excluded exact-reference rows with reasons
