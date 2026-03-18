# Appendix And Resource Presentation Cleanup Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten reviewer-facing appendix and resource notebook presentation so the repo clearly separates analytic full-pipeline estimates from Qiskit structural subset counts, defaults quality/cost views to the exact-reference mainline story, and shows only complete alpha appendix rows without changing any algorithms.

**Architecture:** Keep the cleanup on the reporting side. Add only small reusable helpers in `src/dqi_pipeline.py` and `src/resources.py`, then repurpose the notebooks to use those helpers and stronger default filters/ordering while preserving existing producer schemas and algorithm outputs.

**Tech Stack:** Python, NumPy, pandas, pytest, Jupyter notebook JSON

---

## Chunk 1: Additive Helper Contracts

### Task 1: Add failing tests for alpha-completeness gating

**Files:**
- Modify: `tests/test_dqi_pipeline.py`
- Modify: `src/dqi_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
import math

from src.dqi_pipeline import has_complete_alpha_metrics


def test_has_complete_alpha_metrics_uses_compact_reviewer_fields_only():
    assert has_complete_alpha_metrics(
        {
            "alpha_mode": "paper",
            "execution_mode": "mixture",
            "status": "completed",
            "best_F": 12.0,
            "success_prob": 0.5,
            "top1_regret": 0.1,
        }
    ) is True

    assert has_complete_alpha_metrics(
        {
            "alpha_mode": "paper",
            "execution_mode": "coherent",
            "status": "completed",
            "best_F": 12.0,
            "success_prob": math.nan,
            "top1_regret": 0.1,
        }
    ) is False


def test_has_complete_alpha_metrics_uses_success_probability_precedence():
    row = {
        "alpha_mode": "uniform",
        "execution_mode": "mixture",
        "status": "completed",
        "best_F": 10.0,
        "success_prob": None,
        "postselection_success": 0.4,
        "decoder_success_rate": 0.9,
        "top1_regret": 0.3,
    }
    assert has_complete_alpha_metrics(row) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dqi_pipeline.py::test_has_complete_alpha_metrics_uses_compact_reviewer_fields_only tests/test_dqi_pipeline.py::test_has_complete_alpha_metrics_uses_success_probability_precedence -v`
Expected: FAIL because `has_complete_alpha_metrics` does not exist

- [ ] **Step 3: Write minimal implementation**

Add the smallest helper to `src/dqi_pipeline.py`:

```python
def has_complete_alpha_metrics(report: dict) -> bool:
    required = [
        report.get("alpha_mode"),
        report.get("execution_mode"),
        report.get("status"),
        report.get("best_F"),
        report.get("top1_regret"),
    ]
    success_value = report.get("success_prob")
    if success_value is None:
        success_value = report.get("postselection_success")
    if success_value is None:
        success_value = report.get("decoder_success_rate")
    required.append(success_value)

    for value in required:
        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dqi_pipeline.py::test_has_complete_alpha_metrics_uses_compact_reviewer_fields_only tests/test_dqi_pipeline.py::test_has_complete_alpha_metrics_uses_success_probability_precedence -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_dqi_pipeline.py src/dqi_pipeline.py
git commit -m "feat: add reviewer-facing alpha completeness gate"
```

### Task 2: Add failing tests for additive resource-scope helpers

**Files:**
- Modify: `tests/test_resources.py`
- Modify: `src/resources.py`

- [ ] **Step 1: Write the failing test**

```python
from src.resources import dqi_resource_estimate, summarize_resource_scope


def test_resource_report_has_additive_validation_scope_and_summary():
    result = dqi_resource_estimate(B, v, ell=1)
    assert result["resource_validation_scope"] == "analytic_full_pipeline_plus_qiskit_structural_subset"

    summary = summarize_resource_scope(result)
    assert summary["analytic_scope"] == "full_pipeline_estimate"
    assert summary["qiskit_scope"] == "structural_subset_only"
    assert "decode-inclusive analytic estimate" in summary["note"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resources.py::test_resource_report_has_additive_validation_scope_and_summary -v`
Expected: FAIL on missing additive field/helper

- [ ] **Step 3: Write minimal implementation**

Add only additive fields/helpers in `src/resources.py`:

```python
RESOURCE_VALIDATION_SCOPE = "analytic_full_pipeline_plus_qiskit_structural_subset"


def summarize_resource_scope(report: dict) -> dict[str, str]:
    qiskit_scope = (
        report.get("qiskit_transpiled_structural_count", {})
        .get("scope", {})
        .get("scope", "structural_subset_only")
    )
    return {
        "analytic_scope": "full_pipeline_estimate",
        "qiskit_scope": qiskit_scope,
        "note": (
            "Decode-inclusive analytic estimate and Qiskit structural subset counts "
            "are distinct validation scopes."
        ),
    }
```

and add:

```python
"resource_validation_scope": RESOURCE_VALIDATION_SCOPE,
```

to the stable resource payload.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_resource_report_has_additive_validation_scope_and_summary -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_resources.py src/resources.py
git commit -m "feat: add additive resource scope summary helpers"
```

## Chunk 2: Resource And Quality/Cost Notebook Defaults

### Task 3: Add failing smoke tests for Notebook 06 scope separation

**Files:**
- Create: `tests/test_notebook06_resource_analysis_smoke.py`
- Modify: `notebooks/06_resource_analysis.ipynb`

- [ ] **Step 1: Write the failing test**

```python
def test_notebook06_separates_analytic_pipeline_from_qiskit_subset() -> None:
    markdown = _notebook_markdown("notebooks/06_resource_analysis.ipynb").lower()
    assert "analytic full-pipeline estimates" in markdown
    assert "qiskit structural subset counts" in markdown
    assert "does not validate the full decode-inclusive pipeline" in markdown
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_notebook06_resource_analysis_smoke.py -v`
Expected: FAIL if Notebook 06 does not yet state the split clearly

- [ ] **Step 3: Write minimal implementation**

Update Notebook 06 source-only content so it explicitly separates:
- analytic full-pipeline estimates
- Qiskit structural subset counts

and includes wording like:

```markdown
Qiskit structural subset counts do not validate the full decode-inclusive pipeline.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_notebook06_resource_analysis_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_notebook06_resource_analysis_smoke.py notebooks/06_resource_analysis.ipynb
git commit -m "docs: clarify analytic vs qiskit scope in notebook 06"
```

### Task 4: Add failing smoke tests for Notebook 12 default filters and ordering

**Files:**
- Modify: `tests/test_notebook12_stage_e_smoke.py`
- Modify: `notebooks/12_quality_vs_cost.ipynb`

- [ ] **Step 1: Write the failing test**

```python
def test_notebook12_defaults_to_exact_reference_mainline_mixture_first() -> None:
    src = _notebook_source("notebooks/12_quality_vs_cost.ipynb")
    assert "exact_reference" in src
    assert "mixture" in src
    assert "bp1" in src
    assert "bruteforce" in src
    assert "coherent" in src
    assert "wht" in src
    assert "sort_values" in src
```

and assert the source excludes those exploratory rows by default rather than deleting the toggles.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_notebook12_stage_e_smoke.py::test_notebook12_defaults_to_exact_reference_mainline_mixture_first -v`
Expected: FAIL if the notebook does not yet enforce the new defaults/order

- [ ] **Step 3: Write minimal implementation**

Add explicit default slice/order code to Notebook 12:

```python
base = quality_master.copy()
base = base[base["encoding_norm"] == "exact_reference"].copy()
base = base[base["execution_mode_norm"] == "mixture"].copy()
base = base[base["decoder_norm"].isin(["bp1", "bruteforce"])].copy()
base = base[~base["encoding_norm"].str.contains("wht", na=False)].copy()
base = base[base["execution_mode_norm"] != "coherent"].copy()
base = base.sort_values(
    by=["encoding_norm", "execution_mode_norm", "decoder_norm", "best_F"],
    ascending=[True, True, True, False],
)
```

while preserving the broader notebook logic for later exploratory use.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_notebook12_stage_e_smoke.py::test_notebook12_defaults_to_exact_reference_mainline_mixture_first -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_notebook12_stage_e_smoke.py notebooks/12_quality_vs_cost.ipynb
git commit -m "docs: default notebook 12 to exact-reference mixture mainline"
```

## Chunk 3: Repurpose Notebook 10 Into The Compact Appendix

### Task 5: Replace Notebook 10 smoke expectations with reviewer-facing appendix checks

**Files:**
- Modify: `tests/test_notebook10_alpha_validation_smoke.py`
- Modify: `notebooks/10_alpha_finite_size_validation.ipynb`

- [ ] **Step 1: Write the failing test**

Replace the current deep-validation-oriented smoke assertions with compact appendix checks:

```python
def test_notebook10_uses_alpha_completeness_gate_and_compact_columns() -> None:
    src = _notebook_source("notebooks/10_alpha_finite_size_validation.ipynb")
    assert "has_complete_alpha_metrics" in src
    assert "top1_regret" in src
    assert "success_probability" in src
    assert "distribution_metric" in src


def test_notebook10_omits_incomplete_rows_from_main_appendix_table() -> None:
    src = _notebook_source("notebooks/10_alpha_finite_size_validation.ipynb")
    assert "focus = focus[focus.apply(has_complete_alpha_metrics" in src or "appendix_df = appendix_df[appendix_df.apply(has_complete_alpha_metrics" in src
    assert "fillna(" not in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_notebook10_alpha_validation_smoke.py -v`
Expected: FAIL because Notebook 10 still reflects the detailed validation story

- [ ] **Step 3: Write minimal implementation**

Repurpose `notebooks/10_alpha_finite_size_validation.ipynb` into the compact appendix view:
- load `metrics_master`
- normalize status columns as today
- filter to the alpha-validation slice
- apply `has_complete_alpha_metrics(report)` row gate
- build one compact appendix table with canonical column order:

```python
appendix_df = focus.loc[focus.apply(has_complete_alpha_metrics, axis=1)].copy()
appendix_df["success_probability"] = appendix_df["success_prob"].fillna(appendix_df["postselection_success"]).fillna(appendix_df["decoder_success_rate"])
appendix_df["distribution_metric"] = appendix_df.get("coherent_vs_mixture_distribution_distance")
appendix_df = appendix_df[
    ["alpha_mode", "execution_mode", "best_F", "success_probability", "top1_regret", "distribution_metric", "status"]
]
```

Then ensure incomplete rows are omitted from the displayed reviewer-facing table rather than shown with placeholders.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_notebook10_alpha_validation_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_notebook10_alpha_validation_smoke.py notebooks/10_alpha_finite_size_validation.ipynb
git commit -m "docs: repurpose notebook 10 as compact alpha appendix"
```

### Task 6: Run final targeted verification for the cleanup story

**Files:**
- Verify: `src/dqi_pipeline.py`
- Verify: `src/resources.py`
- Verify: `notebooks/06_resource_analysis.ipynb`
- Verify: `notebooks/10_alpha_finite_size_validation.ipynb`
- Verify: `notebooks/12_quality_vs_cost.ipynb`
- Verify: `tests/test_dqi_pipeline.py`
- Verify: `tests/test_resources.py`
- Verify: `tests/test_notebook06_resource_analysis_smoke.py`
- Verify: `tests/test_notebook10_alpha_validation_smoke.py`
- Verify: `tests/test_notebook12_stage_e_smoke.py`

- [ ] **Step 1: Run targeted verification suite**

Run:

```bash
pytest \
  tests/test_dqi_pipeline.py \
  tests/test_resources.py \
  tests/test_notebook06_resource_analysis_smoke.py \
  tests/test_notebook10_alpha_validation_smoke.py \
  tests/test_notebook12_stage_e_smoke.py -q
```

Expected: PASS

- [ ] **Step 2: Verify acceptance criteria against the spec**

Check that:
- Notebook 06 clearly distinguishes analytic full-pipeline estimates from Qiskit structural subset counts
- Notebook 12 defaults to exact-reference + mixture and orders those rows first
- Notebook 10 shows complete rows only
- no reviewer-facing notebook suggests unsupported coherent/Qiskit claims

- [ ] **Step 3: Summarize reporting/filtering changes**

Report the repo-visible cleanup:
- additive helper gate for compact alpha completeness
- additive resource validation scope summary
- mainline mixture/exact-reference default filtering in Notebook 12
- compact appendix table in Notebook 10 with incomplete rows omitted

