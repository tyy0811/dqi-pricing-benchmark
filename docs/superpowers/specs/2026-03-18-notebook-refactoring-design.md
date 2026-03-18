# Notebook Refactoring Design Spec

**Date:** 2026-03-18
**Status:** Approved
**Goal:** Clean presentation for BMW application + full DRY refactor

## Overview

Refactor the 14 Jupyter notebooks to eliminate redundant code, improve maintainability, and present a polished codebase for BMW Group job application review.

### Key Changes

1. Create `src/notebook_utils.py` with centralized filtering and metric aggregation utilities
2. Trim `notebooks/_helpers.py` to focus on visualization and formatting
3. Split Notebook 08 into 08a (paired encoding) and 08b (coherent validation)
4. Standardize imports across all notebooks
5. Add unit tests for new utilities

## Architecture

### Current State

```
notebooks/
├── _helpers.py (736 lines - mixed concerns)
├── 01-13 notebooks (redundant filter logic)

src/
├── 36 existing modules
```

### Target State

```
notebooks/
├── _helpers.py (~500 lines - visualization only)
├── 08a_paired_encoding_comparison.ipynb (NEW)
├── 08b_coherent_validation.ipynb (NEW)
├── other notebooks (refactored)

src/
├── notebook_utils.py (NEW - ~300 lines)
├── 36 existing modules

tests/
├── test_notebook_utils.py (NEW)
```

## Component Specifications

### 1. `src/notebook_utils.py`

#### Constants

```python
EXACT_REFERENCE_ENCODINGS = frozenset({
    "ilp_derived",
    "ilp_derived_exact",
    "ilp_derived_exact_where_feasible"
})
WHT_EXACT_ENCODINGS = frozenset({"wht_exact"})
WHT_TRUNCATED_ENCODINGS = frozenset({"wht_truncated", "wht_truncated_pricing"})
COMPARABLE_ENCODINGS = EXACT_REFERENCE_ENCODINGS | WHT_EXACT_ENCODINGS | WHT_TRUNCATED_ENCODINGS

DEFAULT_DECODERS = frozenset({"bp1", "bruteforce"})
DEFAULT_EXECUTION_MODE = "mixture"
```

#### FilterPipeline Class

```python
class FilterPipeline:
    """Track cumulative filtering with audit trail."""

    def __init__(self, df: pd.DataFrame, label: str = "input"):
        """Initialize with DataFrame and source label."""

    def filter(self, condition: pd.Series, label: str, reason: str) -> "FilterPipeline":
        """Apply filter, record audit entry, return self for chaining."""

    def filter_encoding(self, allowlist: set[str] = COMPARABLE_ENCODINGS) -> "FilterPipeline":
        """Filter to allowed encodings (case-insensitive)."""

    def filter_execution_mode(self, mode: str = "mixture") -> "FilterPipeline":
        """Filter to specified execution mode."""

    def filter_decoders(self, decoders: set[str] = DEFAULT_DECODERS) -> "FilterPipeline":
        """Filter to specified decoders."""

    def filter_completed(self) -> "FilterPipeline":
        """Keep only completed runs (normalizes status values)."""

    def audit_dataframe(self) -> pd.DataFrame:
        """Return audit trail as DataFrame with columns:
        step_id, step, input_rows, output_rows, rows_removed, reason_label
        """

    def result(self) -> pd.DataFrame:
        """Return filtered DataFrame."""
```

#### Metric Panel Builder

```python
def build_metric_panel(
    df: pd.DataFrame,
    metrics: list[str],
    group_by: str = "encoding",
) -> pd.DataFrame:
    """Build availability/summary panel for metrics across groups.

    Returns DataFrame with columns:
    - group: encoding name or other grouping value
    - metric: metric name
    - availability: "18/18" format string
    - mean_value: float or "unavailable"
    - median_value: float or "unavailable"
    - status: "available" or "unavailable"
    """
```

#### Paired Delta Calculator

```python
def compute_paired_delta(
    df: pd.DataFrame,
    pair_keys: list[str],
    metrics: list[str],
    baseline_encoding: str = "wht_truncated",
    comparison_encoding: str = "ilp_derived",
) -> pd.DataFrame:
    """Compute metric deltas between paired encoding runs.

    Args:
        df: DataFrame with encoding column and metrics
        pair_keys: Columns that identify matched pairs (e.g., instance_id, decoder)
        metrics: Metric columns to compute deltas for
        baseline_encoding: Encoding to subtract (denominator)
        comparison_encoding: Encoding to compare (numerator)

    Returns:
        DataFrame with pair_keys and delta_{metric} columns
    """
```

#### Utility Functions

```python
def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first existing column from candidates, or None."""

def normalize_column(df: pd.DataFrame, col: str, mapping: dict[str, str]) -> pd.Series:
    """Normalize column values via mapping with fallback to original."""
```

#### Functions Moved from `_helpers.py`

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

### 2. `notebooks/_helpers.py` Changes

#### Functions Retained

| Category | Functions |
|----------|-----------|
| **Path utilities** | `repo_root()`, `resolve_repo_path()` |
| **Instance loading** | `load_instance()` |
| **Formatting** | `decode_config_table()`, `format_optimal_table()`, `format_feasibility_stats()` |
| **Distribution tables** | `top_k_distribution()`, `side_by_side_distribution()`, `distribution_table_with_scores()` |
| **Plotting** | `plot_distribution()`, `plot_distribution_overlay()`, `placeholder_plot()` |
| **Metrics** | `tvd()`, `bit_reversed_tvd()`, `top_support_indices()` |
| **Resource formatting** | `resource_table()`, `verification_card()` |
| **Benchmark formatting** | `format_benchmark_row()`, `format_benchmark_table()` |
| **Panel helpers** | `unavailable_panel_table()` |
| **Artifact loading** | `load_run_manifest()`, `load_metrics_master()`, `load_resources_master()`, `load_quality_cost_master()`, `load_quality_cost_unavailable()`, `load_matrix_summary()`, `load_matrix_report()` |

#### Backward Compatibility Re-exports

```python
# At bottom of _helpers.py
from src.notebook_utils import (
    ensure_run_status_norm,
    normalize_run_status,
    parse_metric_availability,
    ensure_numeric_columns,
    ensure_object_columns,
    ensure_optional_columns,
    unavailable_or_numeric,
    placeholder_or_frame,
    manifest_slots_dataframe,
    mark_not_in_experiment_matrix,
)
```

### 3. Notebook 08 Split

#### 08a_paired_encoding_comparison.ipynb

**Focus:** Family P paired encoding comparison (ILP vs WHT)

| Section | Cells |
|---------|-------|
| Setup and imports | 1 |
| Load data | 1 |
| Reviewer summary table | 1 |
| Matched coverage audit | 1 |
| Common metric panel | 1 |
| Paired delta computation | 2 |
| Paired delta visualization | 1 |
| Unavailable cases | 1 |
| Reviewer verdict | 1 |
| Scope note (reference to 08b) | 1 |

**Total:** ~12 cells, ~250 lines

#### 08b_coherent_validation.ipynb

**Focus:** Family C coherent execution validation

| Section | Cells |
|---------|-------|
| Setup and imports | 1 |
| Load Family C data | 1 |
| Data availability summary | 1 |
| ILP coherent vs mixture comparison | 2 |
| Encoding comparison in coherent mode | 2 |
| Per-instance verdict table | 1 |
| Scope note (Family P constraints) | 1 |
| Reference to 08a | 1 |

**Total:** ~10 cells, ~180 lines

### 4. Notebook Import Standardization

#### Standard Pattern (all notebooks)

```python
# Cell 1: Imports and Setup
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

# Repository setup
try:
    from notebooks._helpers import repo_root
except ModuleNotFoundError:
    from _helpers import repo_root

ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Filtering utilities (when needed)
from src.notebook_utils import FilterPipeline, COMPARABLE_ENCODINGS

# Visualization utilities (when needed)
from notebooks._helpers import (
    load_metrics_master,
    unavailable_panel_table,
    placeholder_plot,
)
```

#### Notebooks Requiring Updates

| Notebook | Change |
|----------|--------|
| 01, 02, 03 | Replace manual ROOT with `repo_root()` |
| 06, 09, 11, 12 | Add `FilterPipeline` import, remove inline filter code |
| 08 | Delete (replaced by 08a, 08b) |

### 5. Testing

#### Test File: `tests/test_notebook_utils.py`

```python
"""Unit tests for src/notebook_utils.py"""

import pytest
import pandas as pd
import numpy as np

from src.notebook_utils import (
    FilterPipeline,
    build_metric_panel,
    compute_paired_delta,
    pick_column,
    COMPARABLE_ENCODINGS,
    EXACT_REFERENCE_ENCODINGS,
)


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
```

#### Test Cases

**FilterPipeline:**
- `test_filter_pipeline_empty_input`
- `test_filter_pipeline_chaining`
- `test_filter_pipeline_audit_tracks_row_counts`
- `test_filter_encoding_allowlist`
- `test_filter_encoding_case_insensitive`
- `test_filter_execution_mode_mixture`
- `test_filter_decoders`
- `test_filter_completed_status_normalization`
- `test_filter_preserves_columns`

**build_metric_panel:**
- `test_metric_panel_availability_format`
- `test_metric_panel_missing_metric_unavailable`
- `test_metric_panel_nan_handling`
- `test_metric_panel_groupby_encoding`

**compute_paired_delta:**
- `test_paired_delta_basic`
- `test_paired_delta_missing_pair`
- `test_paired_delta_multiple_metrics`

**Utilities:**
- `test_pick_column_first_match`
- `test_pick_column_none_exist`
- `test_encoding_constants_frozen`

## Implementation Order

1. Create `src/notebook_utils.py` with all utilities
2. Add `tests/test_notebook_utils.py` and verify tests pass
3. Update `notebooks/_helpers.py` (remove moved functions, add re-exports)
4. Update Notebooks 01-03 (ROOT standardization)
5. Create 08a and 08b from Notebook 08
6. Refactor Notebooks 06, 09, 11, 12 to use `FilterPipeline`
7. Delete original Notebook 08
8. Run all notebooks to verify outputs
9. Final review and cleanup

## Success Criteria

- [ ] All tests in `test_notebook_utils.py` pass
- [ ] All notebooks execute without errors
- [ ] Scientific claims unchanged (visual similarity acceptable)
- [ ] No duplicated filter logic across notebooks
- [ ] `_helpers.py` reduced to ~500 lines
- [ ] Notebook 08 successfully split into 08a and 08b

## Out of Scope

- Changes to `src/` algorithm modules (except adding `notebook_utils.py`)
- Changes to benchmark output formats
- Changes to existing test fixtures
- Performance optimization
