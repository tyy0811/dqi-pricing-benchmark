# Notebook 09 Smoothing Penalty Visibility Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface `smoothing_penalty_total` in the final saved reviewer-facing verdict table of Notebook 09 with a minimal notebook-only patch.

**Architecture:** Modify only the existing verdict-table cell in the notebook so it displays `smoothing_penalty_total` directly from the existing scenario-matrix payload, then re-execute the notebook in place and verify the embedded JSON output.

**Tech Stack:** Jupyter notebook JSON, pandas, nbconvert

---

## Chunk 1: Minimal verdict-table patch

### Task 1: Update the verdict table and re-save outputs

**Files:**
- Modify: `notebooks/09_3period_pricing_extension.ipynb`

- [ ] **Step 1: Patch the existing verdict-table cell**

Add `smoothing_penalty_total` to the displayed `verdict_df` using the existing payload only.

- [ ] **Step 2: Execute the notebook in place**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/09_3period_pricing_extension.ipynb`
Expected: notebook saved with updated embedded outputs

- [ ] **Step 3: Verify the saved notebook JSON**

Check that at least one saved output contains:

- `smoothing_penalty_total`
- `dynamic_horizon_revenue`
- `static_horizon_revenue`
- `static_vs_dynamic_delta`
- `verdict`
