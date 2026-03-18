# Notebook Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate redundant filter logic across 14 notebooks by extracting `src/notebook_utils.py`, split Notebook 08 into 08a/08b, and standardize imports — all with full test coverage.

**Architecture:** New `src/notebook_utils.py` centralizes encoding constants, a `FilterPipeline` class with audit trail, metric panel builder, and paired delta calculator. `notebooks/_helpers.py` sheds ~236 lines of data-processing functions (re-exported for backward compat) and retains only visualization/formatting/loading. Notebooks 06/09/11/12 replace inline boolean-index chains with `FilterPipeline` calls.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest, Jupyter notebooks

**Spec:** `docs/superpowers/specs/2026-03-18-notebook-refactoring-design.md`

---

## Task 1: Create `src/notebook_utils.py` — Constants and Utility Functions

**Files:**
- Create: `src/notebook_utils.py`
- Test: `tests/test_notebook_utils.py`

**Step 1: Write failing tests for constants and utility functions**

Create `tests/test_notebook_utils.py` with:

```python
"""Unit tests for src/notebook_utils.py"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from src.notebook_utils import (
    EXACT_REFERENCE_ENCODINGS,
    WHT_EXACT_ENCODINGS,
    WHT_TRUNCATED_ENCODINGS,
    COMPARABLE_ENCODINGS,
    DEFAULT_DECODERS,
    DEFAULT_EXECUTION_MODE,
    pick_column,
    normalize_column,
)


class TestConstants:
    def test_encoding_constants_frozen(self):
        with pytest.raises(AttributeError):
            EXACT_REFERENCE_ENCODINGS.add("new")
        with pytest.raises(AttributeError):
            WHT_EXACT_ENCODINGS.add("new")
        with pytest.raises(AttributeError):
            WHT_TRUNCATED_ENCODINGS.add("new")
        with pytest.raises(AttributeError):
            COMPARABLE_ENCODINGS.add("new")
        with pytest.raises(AttributeError):
            DEFAULT_DECODERS.add("new")

    def test_comparable_is_union(self):
        assert COMPARABLE_ENCODINGS == (
            EXACT_REFERENCE_ENCODINGS | WHT_EXACT_ENCODINGS | WHT_TRUNCATED_ENCODINGS
        )

    def test_default_execution_mode(self):
        assert DEFAULT_EXECUTION_MODE == "mixture"

    def test_exact_reference_contains_ilp(self):
        assert "ilp_derived" in EXACT_REFERENCE_ENCODINGS
        assert "ilp_derived_exact" in EXACT_REFERENCE_ENCODINGS
        assert "ilp_derived_exact_where_feasible" in EXACT_REFERENCE_ENCODINGS


class TestPickColumn:
    def test_pick_column_first_match(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        assert pick_column(df, ["x", "b", "c"]) == "b"

    def test_pick_column_none_exist(self):
        df = pd.DataFrame({"a": [1]})
        assert pick_column(df, ["x", "y", "z"]) is None

    def test_pick_column_empty_candidates(self):
        df = pd.DataFrame({"a": [1]})
        assert pick_column(df, []) is None


class TestNormalizeColumn:
    def test_normalize_with_mapping(self):
        df = pd.DataFrame({"status": ["ok", "error", "unknown"]})
        result = normalize_column(df, "status", {"ok": "completed", "error": "failed"})
        assert list(result) == ["completed", "failed", "unknown"]

    def test_normalize_preserves_unmapped(self):
        df = pd.DataFrame({"x": ["a", "b"]})
        result = normalize_column(df, "x", {"a": "A"})
        assert list(result) == ["A", "b"]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_notebook_utils.py::TestConstants tests/test_notebook_utils.py::TestPickColumn tests/test_notebook_utils.py::TestNormalizeColumn -v`
Expected: FAIL with ImportError

**Step 3: Write `src/notebook_utils.py` with constants and utilities**

```python
"""Centralized notebook filtering, metric aggregation, and data utilities.

Extracted from notebooks/_helpers.py to eliminate redundant filter logic
across the notebook suite.
"""

from __future__ import annotations

from typing import Any, Optional

import json
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Encoding constants
# ---------------------------------------------------------------------------

EXACT_REFERENCE_ENCODINGS = frozenset({
    "ilp_derived",
    "ilp_derived_exact",
    "ilp_derived_exact_where_feasible",
})
WHT_EXACT_ENCODINGS = frozenset({"wht_exact"})
WHT_TRUNCATED_ENCODINGS = frozenset({"wht_truncated", "wht_truncated_pricing"})
COMPARABLE_ENCODINGS = EXACT_REFERENCE_ENCODINGS | WHT_EXACT_ENCODINGS | WHT_TRUNCATED_ENCODINGS

DEFAULT_DECODERS = frozenset({"bp1", "bruteforce"})
DEFAULT_EXECUTION_MODE = "mixture"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first existing column from candidates, or None."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_column(df: pd.DataFrame, col: str, mapping: dict[str, str]) -> pd.Series:
    """Normalize column values via mapping with fallback to original."""
    return df[col].map(lambda v: mapping.get(v, v))
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_notebook_utils.py::TestConstants tests/test_notebook_utils.py::TestPickColumn tests/test_notebook_utils.py::TestNormalizeColumn -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/notebook_utils.py tests/test_notebook_utils.py
git commit -m "feat: add notebook_utils with encoding constants and utility functions"
```

---

## Task 2: Add FilterPipeline Class

**Files:**
- Modify: `src/notebook_utils.py`
- Modify: `tests/test_notebook_utils.py`

**Step 1: Write failing tests for FilterPipeline**

Append to `tests/test_notebook_utils.py`:

```python
from src.notebook_utils import FilterPipeline


@pytest.fixture
def sample_metrics_df():
    """Minimal metrics DataFrame for filter testing."""
    return pd.DataFrame({
        "run_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
        "encoding": ["ilp_derived", "ilp_derived", "wht_truncated",
                     "wht_truncated", "synthetic", "ilp_derived"],
        "execution_mode": ["mixture", "coherent", "mixture",
                          "mixture", "mixture", "mixture"],
        "decoder": ["bp1", "bp1", "bp1", "bruteforce", "bp1", "oracle"],
        "run_status": ["completed", "completed", "completed",
                       "completed", "failed", "completed"],
        "best_sampled_F": [3000.0, 3000.0, 2800.0, 2700.0, np.nan, 3000.0],
        "top1_regret": [0.0, 0.0, 200.0, 300.0, np.nan, 0.0],
    })


class TestFilterPipeline:
    def test_filter_pipeline_empty_input(self):
        empty = pd.DataFrame(columns=["encoding", "run_status"])
        pipe = FilterPipeline(empty)
        assert pipe.result().empty
        assert len(pipe.audit_dataframe()) == 1  # just the input row

    def test_filter_pipeline_chaining(self, sample_metrics_df):
        result = (
            FilterPipeline(sample_metrics_df, label="test")
            .filter_encoding()
            .filter_execution_mode()
            .filter_decoders()
            .result()
        )
        # Should keep: r1 (ilp_derived, mixture, bp1), r3 (wht_truncated, mixture, bp1),
        #              r4 (wht_truncated, mixture, bruteforce)
        # Excludes: r2 (coherent), r5 (synthetic encoding), r6 (oracle decoder)
        assert len(result) == 3
        assert set(result["run_id"]) == {"r1", "r3", "r4"}

    def test_filter_pipeline_audit_tracks_row_counts(self, sample_metrics_df):
        pipe = (
            FilterPipeline(sample_metrics_df, label="raw")
            .filter_encoding()
            .filter_execution_mode()
        )
        audit = pipe.audit_dataframe()
        assert "step" in audit.columns
        assert "input_rows" in audit.columns
        assert "output_rows" in audit.columns
        assert "rows_removed" in audit.columns
        assert len(audit) == 3  # input + 2 filters

    def test_filter_encoding_allowlist(self, sample_metrics_df):
        pipe = FilterPipeline(sample_metrics_df).filter_encoding({"ilp_derived"})
        result = pipe.result()
        assert all(result["encoding"].str.lower() == "ilp_derived")

    def test_filter_encoding_case_insensitive(self):
        df = pd.DataFrame({"encoding": ["ILP_DERIVED", "Wht_Truncated"]})
        pipe = FilterPipeline(df).filter_encoding()
        assert len(pipe.result()) == 2

    def test_filter_execution_mode_mixture(self, sample_metrics_df):
        pipe = FilterPipeline(sample_metrics_df).filter_execution_mode()
        result = pipe.result()
        assert all(result["execution_mode"].str.lower() == "mixture")

    def test_filter_decoders(self, sample_metrics_df):
        pipe = FilterPipeline(sample_metrics_df).filter_decoders()
        result = pipe.result()
        assert set(result["decoder"].str.lower()).issubset({"bp1", "bruteforce"})

    def test_filter_completed_status_normalization(self):
        df = pd.DataFrame({
            "run_status": ["completed", "ok", "failed", "error", None],
        })
        pipe = FilterPipeline(df).filter_completed()
        result = pipe.result()
        assert len(result) == 2  # "completed" and "ok"

    def test_filter_preserves_columns(self, sample_metrics_df):
        pipe = FilterPipeline(sample_metrics_df).filter_encoding()
        assert list(pipe.result().columns) == list(sample_metrics_df.columns)

    def test_generic_filter(self, sample_metrics_df):
        pipe = FilterPipeline(sample_metrics_df).filter(
            sample_metrics_df["run_id"].isin(["r1", "r2"]),
            label="specific_runs",
            reason="manual selection",
        )
        assert len(pipe.result()) == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_notebook_utils.py::TestFilterPipeline -v`
Expected: FAIL with ImportError

**Step 3: Implement FilterPipeline in `src/notebook_utils.py`**

Add to `src/notebook_utils.py` after the utility functions:

```python
# ---------------------------------------------------------------------------
# FilterPipeline
# ---------------------------------------------------------------------------

class FilterPipeline:
    """Track cumulative filtering with audit trail."""

    def __init__(self, df: pd.DataFrame, label: str = "input"):
        self._df = df.copy()
        self._audit: list[dict[str, Any]] = [
            {
                "step_id": 0,
                "step": label,
                "input_rows": len(df),
                "output_rows": len(df),
                "rows_removed": 0,
                "reason_label": "source",
            }
        ]

    def _next_step_id(self) -> int:
        return len(self._audit)

    def filter(self, condition: pd.Series, label: str, reason: str) -> "FilterPipeline":
        """Apply filter, record audit entry, return self for chaining."""
        input_rows = len(self._df)
        self._df = self._df[condition].copy()
        output_rows = len(self._df)
        self._audit.append({
            "step_id": self._next_step_id(),
            "step": label,
            "input_rows": input_rows,
            "output_rows": output_rows,
            "rows_removed": input_rows - output_rows,
            "reason_label": reason,
        })
        return self

    def filter_encoding(self, allowlist: set[str] | frozenset[str] = COMPARABLE_ENCODINGS) -> "FilterPipeline":
        """Filter to allowed encodings (case-insensitive)."""
        if "encoding" not in self._df.columns:
            return self
        lower_allow = {e.lower() for e in allowlist}
        condition = self._df["encoding"].astype(str).str.strip().str.lower().isin(lower_allow)
        return self.filter(condition, "encoding_allowlist", f"encoding in {sorted(allowlist)}")

    def filter_execution_mode(self, mode: str = DEFAULT_EXECUTION_MODE) -> "FilterPipeline":
        """Filter to specified execution mode."""
        if "execution_mode" not in self._df.columns:
            return self
        condition = self._df["execution_mode"].astype(str).str.strip().str.lower() == mode.lower()
        return self.filter(condition, "execution_mode", f"execution_mode == {mode!r}")

    def filter_decoders(self, decoders: set[str] | frozenset[str] = DEFAULT_DECODERS) -> "FilterPipeline":
        """Filter to specified decoders."""
        if "decoder" not in self._df.columns:
            return self
        lower_decoders = {d.lower() for d in decoders}
        condition = self._df["decoder"].astype(str).str.strip().str.lower().isin(lower_decoders)
        return self.filter(condition, "decoder_filter", f"decoder in {sorted(decoders)}")

    def filter_completed(self) -> "FilterPipeline":
        """Keep only completed runs (normalizes status values)."""
        if "run_status" not in self._df.columns:
            return self
        completed_aliases = {"completed", "ok"}
        condition = self._df["run_status"].astype(str).str.strip().str.lower().isin(completed_aliases)
        return self.filter(condition, "completed_only", "run_status in {completed, ok}")

    def audit_dataframe(self) -> pd.DataFrame:
        """Return audit trail as DataFrame."""
        return pd.DataFrame(self._audit)

    def result(self) -> pd.DataFrame:
        """Return filtered DataFrame."""
        return self._df.copy()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_notebook_utils.py::TestFilterPipeline -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/notebook_utils.py tests/test_notebook_utils.py
git commit -m "feat: add FilterPipeline class with audit trail to notebook_utils"
```

---

## Task 3: Add `build_metric_panel` and `compute_paired_delta`

**Files:**
- Modify: `src/notebook_utils.py`
- Modify: `tests/test_notebook_utils.py`

**Step 1: Write failing tests**

Append to `tests/test_notebook_utils.py`:

```python
from src.notebook_utils import build_metric_panel, compute_paired_delta


class TestBuildMetricPanel:
    def test_metric_panel_availability_format(self, sample_metrics_df):
        panel = build_metric_panel(
            sample_metrics_df[sample_metrics_df["run_status"] == "completed"],
            metrics=["best_sampled_F"],
        )
        assert "availability" in panel.columns
        # Should have "N/N" format string
        for val in panel["availability"]:
            assert "/" in str(val)

    def test_metric_panel_missing_metric_unavailable(self, sample_metrics_df):
        panel = build_metric_panel(sample_metrics_df, metrics=["nonexistent_metric"])
        assert all(panel["status"] == "unavailable")

    def test_metric_panel_nan_handling(self, sample_metrics_df):
        panel = build_metric_panel(sample_metrics_df, metrics=["best_sampled_F"])
        # r5 has NaN best_sampled_F — availability should reflect this
        assert "mean_value" in panel.columns

    def test_metric_panel_groupby_encoding(self, sample_metrics_df):
        panel = build_metric_panel(
            sample_metrics_df, metrics=["best_sampled_F"], group_by="encoding"
        )
        assert "group" in panel.columns
        assert "metric" in panel.columns


class TestComputePairedDelta:
    def test_paired_delta_basic(self):
        df = pd.DataFrame({
            "instance_id": ["i1", "i1"],
            "decoder": ["bp1", "bp1"],
            "encoding": ["wht_truncated", "ilp_derived"],
            "score": [100.0, 120.0],
        })
        result = compute_paired_delta(
            df,
            pair_keys=["instance_id", "decoder"],
            metrics=["score"],
        )
        assert len(result) == 1
        assert "delta_score" in result.columns
        assert result["delta_score"].iloc[0] == pytest.approx(20.0)

    def test_paired_delta_missing_pair(self):
        df = pd.DataFrame({
            "instance_id": ["i1", "i2"],
            "decoder": ["bp1", "bp1"],
            "encoding": ["wht_truncated", "ilp_derived"],
            "score": [100.0, 120.0],
        })
        result = compute_paired_delta(
            df,
            pair_keys=["instance_id", "decoder"],
            metrics=["score"],
        )
        # i1 has no ilp_derived pair, i2 has no wht_truncated pair
        assert result.empty

    def test_paired_delta_multiple_metrics(self):
        df = pd.DataFrame({
            "instance_id": ["i1", "i1"],
            "decoder": ["bp1", "bp1"],
            "encoding": ["wht_truncated", "ilp_derived"],
            "score_a": [100.0, 120.0],
            "score_b": [50.0, 45.0],
        })
        result = compute_paired_delta(
            df,
            pair_keys=["instance_id", "decoder"],
            metrics=["score_a", "score_b"],
        )
        assert "delta_score_a" in result.columns
        assert "delta_score_b" in result.columns
        assert result["delta_score_a"].iloc[0] == pytest.approx(20.0)
        assert result["delta_score_b"].iloc[0] == pytest.approx(-5.0)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_notebook_utils.py::TestBuildMetricPanel tests/test_notebook_utils.py::TestComputePairedDelta -v`
Expected: FAIL with ImportError

**Step 3: Implement `build_metric_panel` and `compute_paired_delta`**

Add to `src/notebook_utils.py`:

```python
# ---------------------------------------------------------------------------
# Metric panel builder
# ---------------------------------------------------------------------------

def build_metric_panel(
    df: pd.DataFrame,
    metrics: list[str],
    group_by: str = "encoding",
) -> pd.DataFrame:
    """Build availability/summary panel for metrics across groups.

    Returns DataFrame with columns: group, metric, availability,
    mean_value, median_value, status.
    """
    rows: list[dict[str, Any]] = []
    if group_by not in df.columns:
        groups = [("all", df)]
    else:
        groups = list(df.groupby(group_by))

    for group_name, group_df in groups:
        for metric in metrics:
            if metric not in group_df.columns:
                rows.append({
                    "group": str(group_name),
                    "metric": metric,
                    "availability": "0/0",
                    "mean_value": "unavailable",
                    "median_value": "unavailable",
                    "status": "unavailable",
                })
                continue
            values = pd.to_numeric(group_df[metric], errors="coerce")
            total = len(values)
            available = int(values.notna().sum())
            rows.append({
                "group": str(group_name),
                "metric": metric,
                "availability": f"{available}/{total}",
                "mean_value": float(values.mean()) if available > 0 else "unavailable",
                "median_value": float(values.median()) if available > 0 else "unavailable",
                "status": "available" if available > 0 else "unavailable",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired delta calculator
# ---------------------------------------------------------------------------

def compute_paired_delta(
    df: pd.DataFrame,
    pair_keys: list[str],
    metrics: list[str],
    baseline_encoding: str = "wht_truncated",
    comparison_encoding: str = "ilp_derived",
) -> pd.DataFrame:
    """Compute metric deltas between paired encoding runs.

    Returns DataFrame with pair_keys columns and delta_{metric} columns.
    Delta = comparison - baseline.
    """
    enc_col = "encoding"
    if enc_col not in df.columns:
        return pd.DataFrame()

    baseline = df[df[enc_col].astype(str).str.lower() == baseline_encoding.lower()].copy()
    comparison = df[df[enc_col].astype(str).str.lower() == comparison_encoding.lower()].copy()

    if baseline.empty or comparison.empty:
        return pd.DataFrame()

    merged = baseline.merge(
        comparison,
        on=pair_keys,
        suffixes=("_baseline", "_comparison"),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    result_cols = {k: merged[k] for k in pair_keys}
    for metric in metrics:
        base_col = f"{metric}_baseline"
        comp_col = f"{metric}_comparison"
        if base_col in merged.columns and comp_col in merged.columns:
            result_cols[f"delta_{metric}"] = (
                pd.to_numeric(merged[comp_col], errors="coerce")
                - pd.to_numeric(merged[base_col], errors="coerce")
            )
    return pd.DataFrame(result_cols)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_notebook_utils.py::TestBuildMetricPanel tests/test_notebook_utils.py::TestComputePairedDelta -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/notebook_utils.py tests/test_notebook_utils.py
git commit -m "feat: add build_metric_panel and compute_paired_delta to notebook_utils"
```

---

## Task 4: Move Functions from `_helpers.py` to `notebook_utils.py`

**Files:**
- Modify: `src/notebook_utils.py`
- Modify: `notebooks/_helpers.py`

The spec lists these functions to move:
- `parse_metric_availability()`
- `ensure_numeric_columns()`
- `ensure_object_columns()`
- `ensure_optional_columns()`
- `unavailable_or_numeric()`
- `ensure_run_status_norm()`
- `normalize_run_status()`
- `normalize_run_status_label()`
- `placeholder_or_frame()`
- `manifest_slots_dataframe()`
- `mark_not_in_experiment_matrix()`

**Step 1: Write a test that imports these functions from `src.notebook_utils`**

Add to `tests/test_notebook_utils.py`:

```python
from src.notebook_utils import (
    parse_metric_availability,
    ensure_numeric_columns,
    ensure_object_columns,
    ensure_optional_columns,
    unavailable_or_numeric,
    ensure_run_status_norm,
    normalize_run_status,
    normalize_run_status_label,
    placeholder_or_frame,
    manifest_slots_dataframe,
    mark_not_in_experiment_matrix,
)


class TestMovedFunctions:
    """Verify functions moved from _helpers.py are importable and functional."""

    def test_parse_metric_availability_dict(self):
        result = parse_metric_availability({"a": "1", "b": "2"})
        assert result == {"a": "1", "b": "2"}

    def test_parse_metric_availability_json_string(self):
        result = parse_metric_availability('{"x": "yes"}')
        assert result == {"x": "yes"}

    def test_parse_metric_availability_empty(self):
        assert parse_metric_availability("") == {}
        assert parse_metric_availability(None) == {}

    def test_ensure_numeric_columns_creates_missing(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = ensure_numeric_columns(df, ["a", "b"])
        assert "b" in result.columns
        assert result["b"].isna().all()

    def test_ensure_object_columns_creates_missing(self):
        df = pd.DataFrame({"a": [1]})
        result = ensure_object_columns(df, ["b"], default="N/A")
        assert result["b"].iloc[0] == "N/A"

    def test_normalize_run_status_completed(self):
        assert normalize_run_status("ok") == "completed"
        assert normalize_run_status("completed") == "completed"

    def test_normalize_run_status_failed(self):
        assert normalize_run_status("error") == "failed"

    def test_normalize_run_status_series(self):
        s = pd.Series(["ok", "error", None])
        result = normalize_run_status(s)
        assert list(result) == ["completed", "failed", "unavailable"]

    def test_ensure_run_status_norm(self):
        df = pd.DataFrame({"run_status": ["ok", "failed"]})
        result = ensure_run_status_norm(df)
        assert "run_status_norm" in result.columns
        assert list(result["run_status_norm"]) == ["completed", "failed"]

    def test_normalize_run_status_label(self):
        assert normalize_run_status_label("ok") == "completed"
        assert normalize_run_status_label("error") == "failed"
        assert normalize_run_status_label("skipped") == "not_applicable"

    def test_manifest_slots_dataframe_empty(self):
        result = manifest_slots_dataframe({})
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_manifest_slots_dataframe_with_slots(self):
        manifest = {"slots": [{"id": "s1"}, {"id": "s2"}]}
        result = manifest_slots_dataframe(manifest)
        assert len(result) == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_notebook_utils.py::TestMovedFunctions -v`
Expected: FAIL with ImportError

**Step 3: Copy functions from `_helpers.py` to `src/notebook_utils.py`**

Copy these functions verbatim from `notebooks/_helpers.py` (lines 410-737) into `src/notebook_utils.py`:

- `parse_metric_availability()` (lines 410-424)
- `unavailable_or_numeric()` (lines 438-446)
- `ensure_numeric_columns()` (lines 449-456)
- `ensure_object_columns()` (lines 459-469)
- `ensure_optional_columns()` (lines 472-486)
- `placeholder_or_frame()` (lines 489-499) — also needs `unavailable_panel_table` imported or duplicated; but `unavailable_panel_table` stays in `_helpers.py` per spec. So import it from `_helpers` in the notebooks that need it, and have `placeholder_or_frame` accept a panel builder callable OR import a local helper. **Decision:** `placeholder_or_frame` depends on `unavailable_panel_table`. Move `unavailable_panel_table` to `notebook_utils` as well (it's pure data, no matplotlib), then re-export from `_helpers.py`.
- `normalize_run_status()` (lines 650-674)
- `ensure_run_status_norm()` (lines 677-692)
- `normalize_run_status_label()` (lines 695-706)
- `manifest_slots_dataframe()` (lines 709-711)
- `mark_not_in_experiment_matrix()` (lines 714-737)

Also move `unavailable_panel_table()` (lines 427-435) since `placeholder_or_frame` depends on it.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_notebook_utils.py::TestMovedFunctions -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/notebook_utils.py tests/test_notebook_utils.py
git commit -m "feat: move data-processing functions from _helpers.py to notebook_utils"
```

---

## Task 5: Update `_helpers.py` — Remove Moved Functions, Add Re-exports

**Files:**
- Modify: `notebooks/_helpers.py`

**Step 1: Run existing tests to establish baseline**

Run: `pytest tests/test_notebook_support.py tests/test_notebook_helpers_stage_e.py -v`
Expected: PASS (current baseline)

**Step 2: Remove function bodies from `_helpers.py`**

Delete these function definitions from `notebooks/_helpers.py`:
- `parse_metric_availability` (lines 410-424)
- `unavailable_panel_table` (lines 427-435)
- `unavailable_or_numeric` (lines 438-446)
- `ensure_numeric_columns` (lines 449-456)
- `ensure_object_columns` (lines 459-469)
- `ensure_optional_columns` (lines 472-486)
- `placeholder_or_frame` (lines 489-499)
- `normalize_run_status` (lines 650-674)
- `ensure_run_status_norm` (lines 677-692)
- `normalize_run_status_label` (lines 695-706)
- `manifest_slots_dataframe` (lines 709-711)
- `mark_not_in_experiment_matrix` (lines 714-737)

**Step 3: Add re-exports at bottom of `_helpers.py`**

Add at the end of `notebooks/_helpers.py`:

```python
# ---------------------------------------------------------------------------
# Backward-compatible re-exports from src.notebook_utils
# ---------------------------------------------------------------------------
from src.notebook_utils import (  # noqa: E402, F401
    ensure_run_status_norm,
    normalize_run_status,
    normalize_run_status_label,
    parse_metric_availability,
    ensure_numeric_columns,
    ensure_object_columns,
    ensure_optional_columns,
    unavailable_or_numeric,
    unavailable_panel_table,
    placeholder_or_frame,
    manifest_slots_dataframe,
    mark_not_in_experiment_matrix,
)
```

**Step 4: Run existing tests to verify backward compatibility**

Run: `pytest tests/test_notebook_support.py tests/test_notebook_helpers_stage_e.py tests/test_notebook_reader_hygiene.py -v`
Expected: PASS (all existing imports still work)

**Step 5: Commit**

```bash
git add notebooks/_helpers.py
git commit -m "refactor: remove moved functions from _helpers.py, add re-exports from notebook_utils"
```

---

## Task 6: Standardize ROOT in Notebooks 01 and 02

**Files:**
- Modify: `notebooks/01_reproduce_toy_dqi.ipynb` (Cell 1)
- Modify: `notebooks/02_pricing_instance_demo.ipynb` (Cell 1)

**Step 1: Read both notebooks' current Cell 1**

Verify current pattern in both is:
```python
ROOT = Path.cwd().resolve()
if not (ROOT / 'src').exists() and (ROOT.parent / 'src').exists():
    ROOT = ROOT.parent
```

**Step 2: Replace Cell 1 in Notebook 01**

Replace the manual ROOT logic with:
```python
try:
    from notebooks._helpers import repo_root
except ModuleNotFoundError:
    from _helpers import repo_root

ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

Keep all other imports in the cell unchanged.

**Step 3: Replace Cell 1 in Notebook 02**

Same replacement as Step 2.

**Step 4: Run smoke tests**

Run: `pytest tests/ -k "notebook" --co -q` to list notebook tests, then run any that cover 01/02.

**Step 5: Commit**

```bash
git add notebooks/01_reproduce_toy_dqi.ipynb notebooks/02_pricing_instance_demo.ipynb
git commit -m "refactor: standardize ROOT resolution in notebooks 01 and 02 to use repo_root()"
```

---

## Task 7: Create Notebook 08a — Paired Encoding Comparison

**Files:**
- Create: `notebooks/08a_paired_encoding_comparison.ipynb`
- Reference: `notebooks/08_paired_encoding_comparison.ipynb` (cells 0-14)

**Step 1: Create 08a notebook**

Create `notebooks/08a_paired_encoding_comparison.ipynb` containing cells 0-14 from the original Notebook 08 (everything up to and including the "Reviewer Verdict" section). These are:

- Cell 0: Markdown header (update title: "Notebook 08a: Paired Encoding Comparison (Family P)")
- Cell 1: Imports — add `from src.notebook_utils import FilterPipeline, COMPARABLE_ENCODINGS`
- Cell 2: Load data (run_manifest, metrics_master)
- Cell 3: Markdown "Does exact-reference encoding improve..."
- Cell 4: Summary value helper + reviewer summary table
- Cell 5: Markdown "Matched Coverage"
- Cell 6: Paired key matching
- Cell 7: Markdown "Common Metric Panel"
- Cell 8: Metric panel computation
- Cell 9: Markdown "Paired Delta"
- Cell 10: Paired delta computation
- Cell 11: Markdown "Unavailable / Skipped"
- Cell 12: Status/skipped analysis
- Cell 13: Markdown "Reviewer Verdict"
- Cell 14: Verdict cell
- NEW final cell: Markdown scope note referencing 08b for coherent validation

Total: ~15 cells, consistent with spec's ~12 cell estimate + a few markdown separators.

**Step 2: Add scope note as final cell**

```markdown
## Scope note

Coherent execution validation for Family C instances is covered in [Notebook 08b](08b_coherent_validation.ipynb).
```

**Step 3: Verify notebook parses correctly**

Run: `python3 -c "import json; json.load(open('notebooks/08a_paired_encoding_comparison.ipynb'))"`

**Step 4: Commit**

```bash
git add notebooks/08a_paired_encoding_comparison.ipynb
git commit -m "feat: create notebook 08a (paired encoding comparison, Family P)"
```

---

## Task 8: Create Notebook 08b — Coherent Validation

**Files:**
- Create: `notebooks/08b_coherent_validation.ipynb`
- Reference: `notebooks/08_paired_encoding_comparison.ipynb` (cells 15-22)

**Step 1: Create 08b notebook**

Create `notebooks/08b_coherent_validation.ipynb` containing:

- Cell 0: NEW markdown header "# Notebook 08b: Coherent Execution Validation (Family C)"
- Cell 1: Imports cell (same pattern as 08a, includes `from src.notebook_utils import FilterPipeline`)
- Cell 2: Load Family C data (adapted from original cell 16)
- Cell 3: Markdown "ILP-derived coherent vs mixture on Family C" (original cell 17)
- Cell 4: ILP coherent vs mixture comparison (original cell 18)
- Cell 5: Markdown "Encoding comparison in coherent mode" (original cell 19)
- Cell 6: Encoding comparison code (original cell 20)
- Cell 7: Coherent verdict (original cell 21)
- Cell 8: Markdown scope note (original cell 22, plus reference back to 08a)
- NEW final cell: Reference to 08a

Total: ~10 cells, consistent with spec.

**Step 2: Verify notebook parses correctly**

Run: `python3 -c "import json; json.load(open('notebooks/08b_coherent_validation.ipynb'))"`

**Step 3: Commit**

```bash
git add notebooks/08b_coherent_validation.ipynb
git commit -m "feat: create notebook 08b (coherent validation, Family C)"
```

---

## Task 9: Update Notebook 08 Smoke Tests

**Files:**
- Modify: `tests/test_notebook08_paired_encoding_smoke.py`

**Step 1: Read existing smoke tests**

File: `tests/test_notebook08_paired_encoding_smoke.py` — currently tests against `notebooks/08_paired_encoding_comparison.ipynb`.

**Step 2: Update tests to reference 08a**

Change all references from `"notebooks/08_paired_encoding_comparison.ipynb"` to `"notebooks/08a_paired_encoding_comparison.ipynb"` since 08a contains the paired encoding content.

**Step 3: Add basic smoke test for 08b**

Add a new test:
```python
def test_notebook08b_contains_coherent_family_c_validation() -> None:
    src = _notebook_source("notebooks/08b_coherent_validation.ipynb")
    assert "family_c_df" in src
    assert "coherent" in src.lower()
```

**Step 4: Run updated smoke tests**

Run: `pytest tests/test_notebook08_paired_encoding_smoke.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_notebook08_paired_encoding_smoke.py
git commit -m "test: update notebook 08 smoke tests for 08a/08b split"
```

---

## Task 10: Refactor Notebook 12 to Use FilterPipeline

**Files:**
- Modify: `notebooks/12_quality_vs_cost.ipynb`

Notebook 12 has the most explicit filter chain (encoding + execution_mode + decoder) and is the best candidate for FilterPipeline adoption.

**Step 1: Read Notebook 12's current filter cell (Cell 3)**

Identify the `_track_filter` inline function and its three sequential calls.

**Step 2: Replace filter chain with FilterPipeline**

In the imports cell, add:
```python
from src.notebook_utils import FilterPipeline, COMPARABLE_ENCODINGS
```

Replace the `_track_filter` function and its three calls with:
```python
pipe = (
    FilterPipeline(default_slice, label="quality_cost_input")
    .filter_encoding(COMPARABLE_ENCODINGS)
    .filter_execution_mode("mixture")
    .filter_decoders({"bp1", "bruteforce"})
)
default_slice = pipe.result()
filter_audit = pipe.audit_dataframe()
```

Keep the rest of the cell's downstream logic unchanged.

**Step 3: Verify notebook parses**

Run: `python3 -c "import json; json.load(open('notebooks/12_quality_vs_cost.ipynb'))"`

**Step 4: Run smoke test**

Run: `pytest tests/test_notebook12_stage_e_smoke.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add notebooks/12_quality_vs_cost.ipynb
git commit -m "refactor: use FilterPipeline in notebook 12 (quality vs cost)"
```

---

## Task 11: Refactor Notebook 06 to Use FilterPipeline

**Files:**
- Modify: `notebooks/06_resource_analysis.ipynb`

**Step 1: Read Notebook 06 filter cells**

The inline filters are status-based: `resource_rows[resource_rows["run_status_norm"] == "completed"]`.

**Step 2: Add FilterPipeline import and replace filters**

In imports cell, add:
```python
from src.notebook_utils import FilterPipeline
```

Replace status filter lines with:
```python
resource_rows = FilterPipeline(resource_rows, label="resources").filter_completed().result()
metric_rows = FilterPipeline(metric_rows, label="metrics").filter_completed().result()
```

**Step 3: Run smoke test**

Run: `pytest tests/test_notebook06_resource_analysis_smoke.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add notebooks/06_resource_analysis.ipynb
git commit -m "refactor: use FilterPipeline in notebook 06 (resource analysis)"
```

---

## Task 12: Refactor Notebook 09 to Use FilterPipeline

**Files:**
- Modify: `notebooks/09_decoder_phase_diagram.ipynb`

**Step 1: Read Notebook 09 filter cells**

Complex multi-step filter: stage → matrix → family → execution_mode → run_status.

**Step 2: Add FilterPipeline import and replace filters**

In imports cell, add:
```python
from src.notebook_utils import FilterPipeline
```

Replace the inline filter chain with FilterPipeline where applicable. Note: stage/matrix/family filters are domain-specific and should use the generic `.filter()` method:

```python
pipe = FilterPipeline(candidate, label="decoder_study_input")

if "matrix" in candidate.columns:
    pipe = pipe.filter(
        candidate["matrix"].astype(str).eq("matrix_c"),
        label="matrix_filter",
        reason="matrix == matrix_c",
    )

pipe = (
    pipe
    .filter_execution_mode("mixture")
    .filter_completed()
)
focus = pipe.result()
```

**Important:** The stage filter and family filter in Notebook 09 have complex fallback logic (null tolerance, multiple valid values). These should remain as generic `.filter()` calls with explicit conditions rather than being forced into pre-built filter methods.

**Step 3: Run smoke test**

Run: `pytest tests/test_notebook09_decoder_phase_smoke.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add notebooks/09_decoder_phase_diagram.ipynb
git commit -m "refactor: use FilterPipeline in notebook 09 (decoder phase diagram)"
```

---

## Task 13: Refactor Notebook 11 to Use FilterPipeline

**Files:**
- Modify: `notebooks/11_decision_faithfulness_dashboard.ipynb`

**Step 1: Read Notebook 11 filter cells**

Encoding filter (exclude exploratory WHT) + status filters.

**Step 2: Add FilterPipeline import and replace encoding filter**

In imports cell, add:
```python
from src.notebook_utils import FilterPipeline, EXACT_REFERENCE_ENCODINGS
```

Replace the `_is_exploratory_encoding` inline function and its application with:
```python
if not focus.empty and "encoding" in focus.columns and not include_exploratory_wht:
    pipe = FilterPipeline(focus, label="faithfulness_input")
    pipe = pipe.filter_encoding(EXACT_REFERENCE_ENCODINGS)
    focus = pipe.result()
```

**Step 3: Run smoke test**

Run: `pytest tests/test_notebook11_stage_e_smoke.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add notebooks/11_decision_faithfulness_dashboard.ipynb
git commit -m "refactor: use FilterPipeline in notebook 11 (faithfulness dashboard)"
```

---

## Task 14: Delete Original Notebook 08

**Files:**
- Delete: `notebooks/08_paired_encoding_comparison.ipynb`

**Step 1: Verify 08a and 08b exist and parse**

Run:
```bash
python3 -c "
import json
json.load(open('notebooks/08a_paired_encoding_comparison.ipynb'))
json.load(open('notebooks/08b_coherent_validation.ipynb'))
print('Both notebooks valid')
"
```

**Step 2: Verify smoke tests pass against 08a/08b**

Run: `pytest tests/test_notebook08_paired_encoding_smoke.py -v`
Expected: PASS

**Step 3: Delete original Notebook 08**

```bash
git rm notebooks/08_paired_encoding_comparison.ipynb
```

**Step 4: Search for references to old filename and update**

Run: `grep -r "08_paired_encoding_comparison" tests/ notebooks/ docs/`

Update any remaining references to point to 08a/08b.

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: delete original notebook 08 (replaced by 08a and 08b)"
```

---

## Task 15: Full Test Suite Verification

**Files:** None (verification only)

**Step 1: Run all notebook_utils tests**

Run: `pytest tests/test_notebook_utils.py -v`
Expected: All PASS

**Step 2: Run all notebook smoke tests**

Run: `pytest tests/ -k "notebook" -v`
Expected: All PASS

**Step 3: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All PASS (471+ tests)

**Step 4: Verify `_helpers.py` line count**

Run: `wc -l notebooks/_helpers.py`
Expected: ~500 lines (down from 737)

**Step 5: Commit any final fixes**

If any tests failed, fix and commit. Otherwise, no commit needed.

---

## Summary

| Task | Component | Type |
|------|-----------|------|
| 1 | Constants + utilities | New code + tests |
| 2 | FilterPipeline class | New code + tests |
| 3 | build_metric_panel + compute_paired_delta | New code + tests |
| 4 | Move functions from _helpers.py | Migration |
| 5 | Update _helpers.py re-exports | Refactor |
| 6 | Notebooks 01/02 ROOT standardization | Refactor |
| 7 | Create Notebook 08a | New notebook |
| 8 | Create Notebook 08b | New notebook |
| 9 | Update smoke tests for 08a/08b | Test update |
| 10 | Refactor Notebook 12 (FilterPipeline) | Refactor |
| 11 | Refactor Notebook 06 (FilterPipeline) | Refactor |
| 12 | Refactor Notebook 09 (FilterPipeline) | Refactor |
| 13 | Refactor Notebook 11 (FilterPipeline) | Refactor |
| 14 | Delete original Notebook 08 | Cleanup |
| 15 | Full verification | Verification |
