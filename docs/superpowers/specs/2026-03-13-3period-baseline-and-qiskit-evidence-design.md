# 3-Period Baseline And Qiskit Evidence Design

**Goal:** Add reviewer-facing aliases for the fixed 3-period baseline surface and tighten Notebook 07/09 rerun outputs without changing scientific scope or executing notebooks.

## Scope

- Preserve the existing nested `dynamic_cp_sat` and `static_fixed_config` payloads in `src/classical_baselines_multiperiod.py`.
- Add top-level summary aliases derived from those nested payloads.
- Keep `src/benchmark.py` thin and compatibility-friendly.
- Keep Notebook 07 limited to the implemented Qiskit structural subset only.
- Patch notebook cell sources only; do not execute notebooks or rewrite saved outputs.

## Design

### Multiperiod baseline surface

`run_multiperiod_baselines(prob, **kwargs)` remains the canonical entry point. It continues to return the nested dynamic and static baseline payloads:

- `dynamic_cp_sat`
- `static_fixed_config`

The dynamic branch delegates to the existing CP-SAT solver. The static branch enumerates one fixed configuration reused across all three periods and evaluates each candidate through `prob.evaluate_horizon([x, x, x])`. Equal-revenue ties resolve deterministically to the smallest configuration index.

The function also exposes reviewer-facing top-level aliases derived from the nested payloads:

- `dynamic_horizon_revenue`
- `static_horizon_revenue`
- `static_vs_dynamic_delta`
- `dynamic_per_period_configuration`
- `static_configuration`
- `capacity_usage`
- `remaining_inventory`
- `solve_time_s`
- `status`

These aliases are projections only. Inventory usage, remaining inventory, solve time, and status alias the dynamic branch directly. Smoothing penalties remain consistent because both branches use the shared `evaluate_horizon` accounting path.

### Benchmark surface

`src/benchmark.py` should consume the alias-bearing payload directly when useful, while preserving the existing nested baseline names. No adapter layer or broader refactor is needed.

### Tests

Focused tests will verify:

- dynamic revenue is at least static on a shifted-demand instance
- dynamic equals static when period multipliers are identical
- static reuse, shared inventory, and per-period caps remain satisfied
- top-level aliases match nested payload fields exactly
- `static_vs_dynamic_delta` equals the top-level revenue difference
- `qiskit_subset_validation_report(...)` returns a normalized `prob_dist` and the required structural count fields

### Notebook 09

Notebook 09 should use top-level aliases for reviewer-facing tables and text while preserving nested payloads for provenance. A rerun should clearly show:

- instance summary
- dynamic vs static revenue
- `static_vs_dynamic_delta`
- dynamic per-period configuration
- capacity and remaining inventory
- smoothing penalty when enabled

### Notebook 07

Notebook 07 remains subset-only in wording and implementation. A rerun should clearly state that it validates only the implemented subset:

- approximate low-weight superposition
- phase kick
- syndrome compute
- final Hadamards

It must explicitly exclude decode/uncompute and display:

- NumPy subset reference vs Qiskit subset comparison
- TVD or equivalent distance
- `qubit_count`
- `transpiled_depth`
- `cx_count`
- `gate_counts`

## Risks

- The workspace is not currently attached to a git repository from this path, so this design document cannot be committed here.
- Notebook JSON edits must stay minimal to avoid rewriting stored outputs unintentionally.
