# 3-Period Pricing Extension Design

Date: 2026-03-13

## Goal

Add a minimal, implementation-focused 3-period pricing extension that supports
static-vs-dynamic comparison under shared horizon inventory, while keeping the
existing single-period benchmark intact.

This phase is intentionally narrow:
- exactly 3 periods
- no DQI integration
- no reuse of the single-period report schema
- no generic horizon framework

## Module Contracts

Files:
- `src/problem_generator_multiperiod.py`
- `src/pricing_model_ortools_multiperiod.py`
- `src/classical_baselines_multiperiod.py`
- minimal pointer edits in `src/problem_generator.py`
- minimal pointer edits in `src/classical_baselines.py`

### `src/problem_generator_multiperiod.py`

Add `MultiPeriodPricingProblem` as a separate problem type.

Required contract:
- Fixed to exactly 3 periods in this phase.
- Reuse one shared feature/tier-price structure across all periods.
- Reuse one shared customer-segment catalog across all periods.
- Represent period demand variation through period-specific segment multipliers
  or effective weights over the shared segments.
- Support shared initial inventory for the full 3-period horizon.
- Support optional per-period local caps as an additional operational limit,
  not as the primary carryover mechanism.
- No replenishment.
- No stochastic inventory process.
- No alternate carryover semantics.

Required methods:
- `evaluate_horizon()`
- `to_dict()`
- `from_dict()`
- `save_json()`
- `build_period_truth_table(period_idx)` for tiny sanity checks only

Demand-model contract:
- The shared segment identities remain constant across all 3 periods.
- Period demand differs only through 3 aligned per-period multiplier/weight
  vectors over the shared segments.
- No fully separate per-period segment catalogs are allowed in this phase.

### `src/pricing_model_ortools_multiperiod.py`

Add a compact OR-Tools CP-SAT solver for exactly 3 periods.

Required contract:
- Solve horizon decisions over 3 ordered periods.
- Support per-period configuration/tier decisions.
- Support shared initial inventory carried forward across periods.
- Enforce non-negative remaining inventory.
- Support optional per-period local caps.
- Support optional smoothing penalty between adjacent periods.
- Objective is total 3-period revenue minus penalties.

### `src/classical_baselines_multiperiod.py`

Add minimal multiperiod baseline wrappers.

Required contract:
- Include a horizon CP-SAT baseline.
- Include one simple static fixed-configuration baseline for comparison.
- Do not introduce extra baseline families in this phase.

### Minimal pointer edits in single-period modules

`src/problem_generator.py` and `src/classical_baselines.py` may receive only
short header notes or pointers to the multiperiod extension.

Required contract:
- No behavior change to the single-period APIs.
- No schema expansion of existing single-period payloads.

### Module Acceptance Checks

- `MultiPeriodPricingProblem` is explicitly fixed at exactly 3 periods.
- The shared-segment / period-weight contract is documented in code.
- `capacity_usage` semantics are fixed and reused consistently:
  - `inventory_consumed_per_period`: 3-entry period-ordered list
  - `remaining_inventory_after_period`: 3-entry period-ordered list
- No single-period public API changes behavior.

## Benchmark/Report Boundary

Files:
- `src/benchmark.py`

Add `run_multiperiod_pricing_benchmark(...)` as a separate benchmark entry
point.

Required contract:
- Separate from the existing single-period benchmark/report path.
- Focused on static-vs-dynamic pricing comparison only.
- No DQI handoff or surrogate-artifact integration in this phase.
- No attempt to force multiperiod outputs into the existing single-period
  schema.

Required return fields:
- `horizon_revenue`
- `static_vs_dynamic_delta`
- `per_period_configuration`
- `capacity_usage`
- `solve_time`

Payload-shape contract:
- `per_period_configuration` is a 3-entry, period-ordered structure.
- Each period entry contains the selected configuration for that period and its
  tier decision in a compact reviewer-readable form.
- `capacity_usage` uses one fixed meaning across solver, benchmark, and
  notebook:
  - period-ordered inventory consumed
  - period-ordered remaining inventory after each period

### Benchmark Acceptance Checks

- `run_multiperiod_pricing_benchmark(...)` is separate from
  `run_benchmark(...)`.
- The payload is compact and reviewer-facing, not a generic framework schema.
- The static-vs-dynamic comparison is interpretable without reading internal
  solver details.

## Notebook Structure

Files:
- `notebooks/09_3period_pricing_extension.ipynb`
- minimal pointer edit in `notebooks/04_pricing_pipeline.ipynb`

Notebook 09 is the reviewer-facing demonstration of the extension.

Required content:
- one concrete 3-period instance
- shared segment identities across periods
- period-varying demand via segment-weight shifts
- shared initial inventory and carryover across the horizon
- static baseline vs horizon CP-SAT dynamic result
- revenue comparison
- capacity/inventory usage
- plain-language conclusion explaining why this is dynamic-pricing-adjacent

Required notebook framing:
- Demand varies by period through weight shifts over shared segments.
- Segment identities are not redefined by period.
- The comparison is clean because the package/tier/feature-price structure is
  shared across all periods.

Required presentation:
- Include one compact comparison table that makes the static-vs-dynamic
  comparison immediate.
- Do not rely only on prose or raw dict displays.

Notebook 04 contract:
- Keep the current single-period pipeline intact.
- Add only a short pointer that the 3-period extension is documented
  separately in Notebook 09.

### Notebook Acceptance Checks

- Notebook 09 reads as a 3-period extension, not a generic horizon framework.
- The comparison table clearly shows static vs dynamic outcomes.
- The notebook explains the modeling choice: shared segments with per-period
  weight shifts.
- The conclusion does not imply DQI integration.

## Verification and Testing Scope

Files, likely:
- new tests for the multiperiod modules
- new notebook smoke coverage for `notebooks/09_3period_pricing_extension.ipynb`
- minimal regression guard for unchanged single-period behavior

Required tests:
- `MultiPeriodPricingProblem` serialization round-trip
- `evaluate_horizon()` on a tiny 3-period instance
- `build_period_truth_table(period_idx)` sanity checks
- horizon CP-SAT output contract
- static baseline vs dynamic baseline output shape
- `run_multiperiod_pricing_benchmark(...)` payload contract
- notebook smoke coverage for the modeling explanation and comparison table
- one light regression test confirming existing single-period behavior remains
  unchanged

### Verification Acceptance Checks

- Tests verify the model is fixed at exactly 3 periods.
- Tests verify the shared-segment / period-weight contract explicitly.
- Tests verify the `capacity_usage` structure is consistent.
- Tests verify the single-period benchmark path still behaves as before.
- No tests expect DQI integration.
- No tests require generic horizon support.

## Non-Goals

- No generalization beyond exactly 3 periods.
- No fully separate per-period segment catalogs.
- No DQI integration or handoff.
- No generic horizon optimization framework.
- No broad refactor of the single-period benchmark path.
- No attempt to merge multiperiod outputs into the existing single-period
  report schema.

## Compatibility Constraints

- Keep the existing single-period benchmark intact.
- Keep existing public single-period APIs intact.
- Prefer additive new modules over branching existing logic.
- Keep serialization compact and reviewer-friendly.

