# 3-Period Pricing Extension Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strictly 3-period pricing extension with shared-segment demand shifts, shared horizon inventory carryover, a compact dynamic-vs-static benchmark surface, and a reviewer-facing notebook while leaving the existing single-period benchmark unchanged.

**Architecture:** Keep the new semantics in parallel modules rather than branching the single-period codepath. Reuse the existing single-period `Feature`, `CustomerSegment`, `BundleConstraint`, tier encoding, and compact CP-SAT/result patterns, but isolate the 3-period contracts in new files and a separate benchmark entry point.

**Tech Stack:** Python, NumPy, OR-Tools CP-SAT, pytest, Jupyter notebook JSON

---

## Chunk 1: 3-Period Problem Model

### Task 1: Add failing tests for the fixed-3-period problem contract

**Files:**
- Create: `tests/test_problem_generator_multiperiod.py`
- Create: `src/problem_generator_multiperiod.py`
- Reference: `src/problem_generator.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from src.problem_generator import Feature, CustomerSegment, BundleConstraint
from src.problem_generator_multiperiod import MultiPeriodPricingProblem


def test_multiperiod_problem_roundtrip_and_contract():
    prob = MultiPeriodPricingProblem(
        features=[
            Feature("Seats", 500, 900, 1400),
            Feature("Roof", 800, 1300, 1900),
        ],
        segments=[
            CustomerSegment("Budget", 2500, 0.6),
            CustomerSegment("Premium", 5000, 0.4),
        ],
        bundle_constraints=[],
        period_segment_multipliers=np.array(
            [
                [1.0, 1.0],
                [0.8, 1.2],
                [0.6, 1.4],
            ],
            dtype=np.float64,
        ),
        initial_inventory=4,
        inventory_consumption=np.array([1, 2], dtype=np.int64),
    )

    payload = prob.to_dict()
    clone = MultiPeriodPricingProblem.from_dict(payload)

    assert prob.n_periods == 3
    assert clone.n_periods == 3
    assert clone.period_segment_multipliers.shape == (3, 2)
    assert clone.initial_inventory == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_problem_generator_multiperiod.py::test_multiperiod_problem_roundtrip_and_contract -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `src.problem_generator_multiperiod`

- [ ] **Step 3: Write minimal implementation**

Implement `src/problem_generator_multiperiod.py` with:

```python
from dataclasses import dataclass
from pathlib import Path

import json
import numpy as np

from src.problem_generator import (
    BundleConstraint,
    CustomerSegment,
    Feature,
    PricingProblem,
    decode_bitstring,
    encode_tiers,
)


@dataclass
class PeriodEvaluation:
    period_idx: int
    x: int
    tiers: list[int]
    expected_revenue: float
    inventory_consumed: int
    remaining_inventory: int


class MultiPeriodPricingProblem:
    def __init__(self, ..., period_segment_multipliers: np.ndarray, initial_inventory: int, ...):
        if np.asarray(period_segment_multipliers).shape[0] != 3:
            raise ValueError("MultiPeriodPricingProblem is fixed to exactly 3 periods.")
        ...
        self.n_periods = 3

    def evaluate_horizon(self, x_by_period: list[int]) -> dict:
        ...

    def build_period_truth_table(self, period_idx: int) -> np.ndarray:
        ...

    def to_dict(self) -> dict:
        ...

    @classmethod
    def from_dict(cls, payload: dict) -> "MultiPeriodPricingProblem":
        ...

    def save_json(self, path: str | Path) -> None:
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_problem_generator_multiperiod.py::test_multiperiod_problem_roundtrip_and_contract -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_problem_generator_multiperiod.py src/problem_generator_multiperiod.py
git commit -m "feat: add 3-period pricing problem model"
```

### Task 2: Add failing tests for horizon evaluation and period truth tables

**Files:**
- Modify: `tests/test_problem_generator_multiperiod.py`
- Modify: `src/problem_generator_multiperiod.py`

- [ ] **Step 1: Write the failing test**

```python
def test_evaluate_horizon_tracks_inventory_and_period_weights():
    prob = make_tiny_multiperiod_problem()
    x_std = 0
    x_premium = 1

    out = prob.evaluate_horizon([x_std, x_premium, x_std])

    assert out["n_periods"] == 3
    assert out["capacity_usage"]["inventory_consumed_per_period"] == [1, 1, 1]
    assert out["capacity_usage"]["remaining_inventory_after_period"] == [2, 1, 0]
    assert len(out["per_period"]) == 3


def test_build_period_truth_table_is_period_specific():
    prob = make_tiny_multiperiod_problem()
    t0 = prob.build_period_truth_table(0)
    t2 = prob.build_period_truth_table(2)
    assert t0.shape == t2.shape
    assert not np.allclose(t0, t2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_problem_generator_multiperiod.py -v`
Expected: FAIL on missing `evaluate_horizon()` details or period-specific truth-table behavior

- [ ] **Step 3: Write minimal implementation**

Implement the smallest horizon-evaluation contract:

```python
def evaluate_horizon(self, x_by_period: list[int]) -> dict:
    if len(x_by_period) != 3:
        raise ValueError("x_by_period must have exactly 3 entries.")
    remaining = int(self.initial_inventory)
    per_period = []
    consumed = []
    remaining_after = []
    total_revenue = 0.0
    total_penalty = 0.0
    for period_idx, x in enumerate(x_by_period):
        tiers = decode_bitstring(int(x), self.n_features)
        period_revenue = self.expected_revenue_for_period(period_idx, tiers)
        used = self.inventory_consumption_for_tiers(tiers)
        remaining = remaining - used
        if remaining < 0:
            total_penalty += self.inventory_violation_penalty
        per_period.append({...})
        consumed.append(int(used))
        remaining_after.append(int(max(remaining, 0)))
        total_revenue += float(period_revenue)
    return {
        "n_periods": 3,
        "horizon_revenue": float(total_revenue - total_penalty),
        "per_period": per_period,
        "capacity_usage": {
            "inventory_consumed_per_period": consumed,
            "remaining_inventory_after_period": remaining_after,
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_problem_generator_multiperiod.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_problem_generator_multiperiod.py src/problem_generator_multiperiod.py
git commit -m "feat: add 3-period horizon evaluation helpers"
```

## Chunk 2: CP-SAT Solver and Baselines

### Task 3: Add failing tests for the 3-period CP-SAT solver contract

**Files:**
- Create: `tests/test_pricing_model_ortools_multiperiod.py`
- Create: `src/pricing_model_ortools_multiperiod.py`
- Reference: `src/pricing_model_ortools.py`

- [ ] **Step 1: Write the failing test**

```python
from src.pricing_model_ortools_multiperiod import solve_multiperiod_pricing_with_cp_sat


def test_multiperiod_cp_sat_returns_compact_contract():
    prob = make_tiny_multiperiod_problem()
    out = solve_multiperiod_pricing_with_cp_sat(prob, instance_id="tiny_3period")

    assert out["status"] in {"optimal", "feasible"}
    assert out["horizon_revenue"] is not None
    assert len(out["per_period_configuration"]) == 3
    assert out["capacity_usage"]["inventory_consumed_per_period"]
    assert out["capacity_usage"]["remaining_inventory_after_period"]
    assert out["solve_time_s"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pricing_model_ortools_multiperiod.py::test_multiperiod_cp_sat_returns_compact_contract -v`
Expected: FAIL with `ModuleNotFoundError` or missing function

- [ ] **Step 3: Write minimal implementation**

Implement a one-hot exact 3-period CP-SAT model:

```python
def solve_multiperiod_pricing_with_cp_sat(prob, instance_id=None) -> dict:
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    period_tables = [prob.build_period_truth_table(p) for p in range(3)]
    selectors = []
    for period_idx, table in enumerate(period_tables):
        period_selectors = [
            model.NewBoolVar(f"select_p{period_idx}_x{x}")
            for x in range(prob.n_configurations)
        ]
        model.Add(sum(period_selectors) == 1)
        selectors.append(period_selectors)

    total_inventory = []
    for period_idx in range(3):
        total_inventory.extend(
            int(prob.inventory_consumption_by_config[x]) * selectors[period_idx][x]
            for x in range(prob.n_configurations)
        )
    model.Add(sum(total_inventory) <= int(prob.initial_inventory))
    ...
    model.Maximize(revenue_terms - smoothing_terms)
    ...
    return {
        "status": status,
        "horizon_revenue": float(...),
        "per_period_configuration": [...],
        "capacity_usage": {...},
        "solve_time_s": float(solve_time_s),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pricing_model_ortools_multiperiod.py::test_multiperiod_cp_sat_returns_compact_contract -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_pricing_model_ortools_multiperiod.py src/pricing_model_ortools_multiperiod.py
git commit -m "feat: add 3-period CP-SAT pricing solver"
```

### Task 4: Add failing tests for multiperiod baselines

**Files:**
- Create: `tests/test_classical_baselines_multiperiod.py`
- Create: `src/classical_baselines_multiperiod.py`

- [ ] **Step 1: Write the failing test**

```python
from src.classical_baselines_multiperiod import run_multiperiod_baselines


def test_multiperiod_baselines_compare_static_and_dynamic():
    prob = make_tiny_multiperiod_problem()
    out = run_multiperiod_baselines(prob)

    assert set(out) == {"dynamic_cp_sat", "static_fixed_config"}
    assert len(out["dynamic_cp_sat"]["per_period_configuration"]) == 3
    assert len(out["static_fixed_config"]["per_period_configuration"]) == 3
    assert out["static_fixed_config"]["per_period_configuration"][0] == out["static_fixed_config"]["per_period_configuration"][1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_classical_baselines_multiperiod.py::test_multiperiod_baselines_compare_static_and_dynamic -v`
Expected: FAIL with `ModuleNotFoundError` or missing wrapper

- [ ] **Step 3: Write minimal implementation**

Implement:

```python
from src.pricing_model_ortools_multiperiod import solve_multiperiod_pricing_with_cp_sat


def solve_static_multiperiod_baseline(prob) -> dict:
    best = None
    for x in range(prob.n_configurations):
        candidate = prob.evaluate_horizon([x, x, x])
        ...
    return {
        "status": "ok",
        "horizon_revenue": float(best["horizon_revenue"]),
        "per_period_configuration": best["per_period_configuration"],
        "capacity_usage": best["capacity_usage"],
    }


def run_multiperiod_baselines(prob) -> dict:
    return {
        "dynamic_cp_sat": solve_multiperiod_pricing_with_cp_sat(prob),
        "static_fixed_config": solve_static_multiperiod_baseline(prob),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_classical_baselines_multiperiod.py::test_multiperiod_baselines_compare_static_and_dynamic -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_classical_baselines_multiperiod.py src/classical_baselines_multiperiod.py
git commit -m "feat: add 3-period baseline wrappers"
```

## Chunk 3: Benchmark Entry Point and Single-Period Guard

### Task 5: Add failing tests for the separate multiperiod benchmark surface

**Files:**
- Modify: `tests/test_benchmark.py`
- Modify: `src/benchmark.py`
- Reference: `src/classical_baselines_multiperiod.py`

- [ ] **Step 1: Write the failing test**

```python
from src.benchmark import run_multiperiod_pricing_benchmark


def test_run_multiperiod_pricing_benchmark_returns_separate_compact_payload():
    prob = make_tiny_multiperiod_problem()
    out = run_multiperiod_pricing_benchmark(prob)

    assert set(out) >= {
        "horizon_revenue",
        "static_vs_dynamic_delta",
        "per_period_configuration",
        "capacity_usage",
        "solve_time",
    }
    assert len(out["per_period_configuration"]) == 3
    assert out["capacity_usage"]["inventory_consumed_per_period"]
    assert out["capacity_usage"]["remaining_inventory_after_period"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark.py::test_run_multiperiod_pricing_benchmark_returns_separate_compact_payload -v`
Expected: FAIL because `run_multiperiod_pricing_benchmark` does not exist

- [ ] **Step 3: Write minimal implementation**

Add a separate benchmark helper:

```python
from src.classical_baselines_multiperiod import run_multiperiod_baselines


def run_multiperiod_pricing_benchmark(prob) -> dict:
    baselines = run_multiperiod_baselines(prob)
    dynamic = baselines["dynamic_cp_sat"]
    static = baselines["static_fixed_config"]
    return {
        "horizon_revenue": float(dynamic["horizon_revenue"]),
        "static_vs_dynamic_delta": float(
            dynamic["horizon_revenue"] - static["horizon_revenue"]
        ),
        "per_period_configuration": dynamic["per_period_configuration"],
        "capacity_usage": dynamic["capacity_usage"],
        "solve_time": float(dynamic["solve_time_s"]),
        "baseline_comparison": {
            "dynamic_cp_sat": dynamic["horizon_revenue"],
            "static_fixed_config": static["horizon_revenue"],
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark.py::test_run_multiperiod_pricing_benchmark_returns_separate_compact_payload -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_benchmark.py src/benchmark.py src/classical_baselines_multiperiod.py
git commit -m "feat: add separate 3-period benchmark entry point"
```

### Task 6: Add a light regression guard for unchanged single-period behavior

**Files:**
- Modify: `tests/test_benchmark.py`
- Modify: `src/problem_generator.py`
- Modify: `src/classical_baselines.py`

- [ ] **Step 1: Write the failing test**

```python
from src.classical_baselines import cp_sat_solver_result
from src.problem_generator import default_instance


def test_single_period_cp_sat_contract_remains_unchanged():
    prob = default_instance()
    out = cp_sat_solver_result(prob, instance_id="single_period_guard")

    assert "objective_value" in out
    assert "optimized_objective_value" in out
    assert "x_opt" in out
    assert "constraint_summary" in out
```

- [ ] **Step 2: Run test to verify it fails only if a behavior regression is introduced**

Run: `pytest tests/test_benchmark.py::test_single_period_cp_sat_contract_remains_unchanged -v`
Expected: PASS immediately or FAIL only if an accidental regression was introduced during implementation

- [ ] **Step 3: Write minimal implementation**

Only add short header/doc pointers in the single-period modules. Do not alter runtime behavior:

```python
"""
Single-period pricing baselines.

For the separate 3-period extension, see src.classical_baselines_multiperiod.
"""
```

- [ ] **Step 4: Run targeted regression checks**

Run: `pytest tests/test_benchmark.py::test_single_period_cp_sat_contract_remains_unchanged tests/test_pricing_model_ortools.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_benchmark.py src/problem_generator.py src/classical_baselines.py
git commit -m "docs: add pointers for separate 3-period pricing extension"
```

## Chunk 4: Notebook 09 and Notebook 04 Pointer

### Task 7: Add failing smoke tests for Notebook 09

**Files:**
- Create: `tests/test_notebook09_3period_pricing_extension_smoke.py`
- Create: `notebooks/09_3period_pricing_extension.ipynb`

- [ ] **Step 1: Write the failing test**

```python
def test_notebook09_states_shared_segment_weight_shift_model() -> None:
    src = _notebook_source("notebooks/09_3period_pricing_extension.ipynb")
    assert "shared segment" in src.lower()
    assert "period-varying demand" in src.lower()
    assert "weight shifts" in src.lower()


def test_notebook09_shows_static_vs_dynamic_comparison_table() -> None:
    src = _notebook_source("notebooks/09_3period_pricing_extension.ipynb")
    assert "comparison_df" in src
    assert "static_fixed_config" in src
    assert "dynamic_cp_sat" in src
    assert "capacity_usage" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_notebook09_3period_pricing_extension_smoke.py -v`
Expected: FAIL because the notebook does not exist yet

- [ ] **Step 3: Write minimal implementation**

Create `notebooks/09_3period_pricing_extension.ipynb` with source-only cells that:
- define one tiny 3-period instance
- load or construct `MultiPeriodPricingProblem`
- run `run_multiperiod_pricing_benchmark(...)`
- build one compact comparison table
- show per-period configuration and `capacity_usage`
- end with a markdown conclusion explaining why this is dynamic-pricing-adjacent

Core code cell shape:

```python
comparison_df = pd.DataFrame(
    [
        {"baseline": "static_fixed_config", "horizon_revenue": static_rev},
        {"baseline": "dynamic_cp_sat", "horizon_revenue": dynamic_rev},
    ]
)
capacity_df = pd.DataFrame(
    {
        "period": [0, 1, 2],
        "inventory_consumed": out["capacity_usage"]["inventory_consumed_per_period"],
        "remaining_inventory": out["capacity_usage"]["remaining_inventory_after_period"],
    }
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_notebook09_3period_pricing_extension_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_notebook09_3period_pricing_extension_smoke.py notebooks/09_3period_pricing_extension.ipynb
git commit -m "docs: add reviewer-facing 3-period pricing notebook"
```

### Task 8: Add a minimal Notebook 04 pointer and final targeted verification

**Files:**
- Modify: `tests/test_notebook04_pricing_pipeline_smoke.py`
- Modify: `notebooks/04_pricing_pipeline.ipynb`
- Verify: `tests/test_problem_generator_multiperiod.py`
- Verify: `tests/test_pricing_model_ortools_multiperiod.py`
- Verify: `tests/test_classical_baselines_multiperiod.py`
- Verify: `tests/test_benchmark.py`
- Verify: `tests/test_notebook09_3period_pricing_extension_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
def test_notebook04_points_to_separate_notebook09_extension() -> None:
    markdown = _notebook_markdown("notebooks/04_pricing_pipeline.ipynb").lower()
    assert "notebook 09" in markdown
    assert "3-period" in markdown
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_notebook04_pricing_pipeline_smoke.py::test_notebook04_points_to_separate_notebook09_extension -v`
Expected: FAIL if Notebook 04 has no pointer

- [ ] **Step 3: Write minimal implementation**

Add one short markdown note to Notebook 04:

```markdown
Note: the separate 3-period pricing extension is documented in Notebook 09.
This notebook remains the single-period pricing pipeline.
```

- [ ] **Step 4: Run final targeted verification**

Run:

```bash
pytest \
  tests/test_problem_generator_multiperiod.py \
  tests/test_pricing_model_ortools_multiperiod.py \
  tests/test_classical_baselines_multiperiod.py \
  tests/test_benchmark.py \
  tests/test_notebook04_pricing_pipeline_smoke.py \
  tests/test_notebook09_3period_pricing_extension_smoke.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_notebook04_pricing_pipeline_smoke.py notebooks/04_pricing_pipeline.ipynb
git commit -m "docs: add 3-period extension pointer to notebook 04"
```

