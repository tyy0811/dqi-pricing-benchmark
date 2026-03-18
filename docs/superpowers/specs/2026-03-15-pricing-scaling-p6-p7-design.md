# Pricing Scaling P6 P7 Design

## Goal

Scale the pricing benchmark from the current frozen P3/P4/P5 family to strict-superset P6 (6-feature, 12-bit) and P7 (7-feature, 14-bit) instances, with full integration into the existing exact benchmarking, paired-encoding, matrix execution, metrics reporting, and notebook analysis pipeline.

## Non-Negotiable Constraints

1. P6 and P7 are first-class P-family members. They must flow through the same instance loading, artifact generation, matrix execution, and metrics/reporting paths as P3/P4/P5.
2. Exactness is preserved. No shot-based fallback is allowed. If statevector DQI is intractable, the pipeline must persist structured boundary records instead of crashing or silently skipping.
3. `m=15` is the canonical encoding budget for paired-encoding comparison. P7 may also produce a secondary `m=20` paired artifact for diagnostics only if WHT quality degrades materially at `m=15`.
4. P3/P4/P5 instances, artifacts, metrics rows, and published results must remain unchanged.
5. Notebook outputs and record counts must be computed dynamically; no hard-coded totals.

## Current Codebase Findings

- [`src/problem_generator.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/src/problem_generator.py) currently defines a fixed 5-feature default instance, smaller prefixes for P3/P4, and frozen instance generation for only 3, 4, and 5 features.
- [`src/pricing_model_ortools.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/src/pricing_model_ortools.py) currently solves pricing exactly via a one-hot selector over all `2^n` configurations and hard-fails for `n_bits > 10`. The solver contract itself is still exact and should remain exact for P6/P7; the design requires confirming and then documenting the variable count in Notebook 13.
- [`src/pricing_ilp.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/src/pricing_ilp.py) already exposes `solve_pricing_pair(...)` and paired-artifact validation logic for comparative pricing encodings.
- [`src/matrix_runner.py`](/Users/zenith/Desktop/dqi-pricing-benchmark/src/matrix_runner.py) currently treats the P family as `p_features=(3, 4, 5)` and builds Matrix C pricing rows through the same runner used for existing P-family DQI results.
- Existing notebook/report helpers already load canonical metrics master records and paired artifacts; the design should extend those paths rather than introduce notebook-local data loading.

## Design Decisions

### 1. Instance and Solver Layer

- Preserve features 1-5 exactly as the current P5 catalog: same names, tier semantics, and prices.
- Add feature 6 and feature 7 as additive BMW-realistic options. P6 uses features 1-6; P7 uses features 1-7.
- Preserve the original bundle rules unchanged. Add only new bundle rules that involve at least one new feature.
- Preserve the first three demand segments byte-identically over features 1-5. Add a fourth segment whose primary differentiation is demand over features 6-7 so scaling effects can be attributed cleanly.
- Raise the luxury cap from 2 to 3 for the larger instances while preserving the same invalid-code and constraint penalty formulas via recomputed `R_max`.
- Extend frozen instance generation so `pricing_6feat_12bit.json` and `pricing_7feat_14bit.json` are emitted in the same schema as the current pricing JSON files.
- Confirm and document that the CP-SAT path is one-hot over all configurations. Remove the artificial `n_bits > 10` guard while preserving exact semantics. Notebook 13 will report model type and variable count.
- Add validation that the embedded P5 optimum with new features disabled remains feasible inside P7, and compare it to the actual P7 optimum as an explicit scaling signal.

### 2. Paired Encoding and Benchmark Integration

- Generate paired pricing artifacts for P6/P7 with `solve_pricing_pair(..., k=15, ...)`, producing both `wht_truncated` and `ilp_derived`.
- Continue emitting legacy WHT surrogate JSON files for compatibility with existing pricing/DQI loaders.
- Compute WHT diagnostics for P7 at `m=15`: retained energy `eta`, Spearman rho between `F` and surrogate `G`, and top regret metrics.
- If P7 WHT quality at `m=15` falls below the configured thresholds (`eta < 0.85` or `rho < 0.90`), generate a secondary `m=20` paired artifact tagged as diagnostic-only and keep it out of the canonical benchmark matrix.
- Integrate P6/P7 into both Matrix A and Matrix C as first-class P-family instances. Matrix A is required for Notebook 08 paired-encoding comparison; Matrix C is required for decoder/structure analysis and Notebook 13 scaling views.
- Runtime and, where feasible, peak-memory measurements should be normalized into canonical result records so Notebook 13 can report wall-clock/boundary behavior without bespoke parsing.

### 3. Exact Execution Boundary Policy

- P6 is the intended exact DQI frontier and should attempt the full requested run matrix subject to actual statevector limits.
- P7 must still complete all exact classical and analytic pieces: brute-force optimum, CP-SAT cross-check, full truth-table encoding diagnostics, paired-artifact validation, and resource estimation.
- For any P6/P7 DQI configuration that cannot complete exactly, persist a structured record with:
  - `status` of `unavailable` or `failed`
  - `boundary_reason` such as `statevector_memory_exceeded` or `qubit_count_exceeded`
  - limiting resource estimates such as qubit count and estimated memory
- Boundary records are part of the canonical metrics master and must be rendered explicitly in Notebook 13 instead of causing notebook failure.

### 4. Notebook Updates

- Notebook 08 must widen its paired-encoding comparison from hard-coded P3/P4/P5 assumptions to all P-family instances with completed matched records. Summary statistics must recompute dynamically.
- Notebook 13 (`notebooks/13_scaling_analysis.ipynb`) will present:
  - instance summary across P3-P7 including CP-SAT model type and variable count
  - encoding scaling for both WHT and ILP-derived encodings, including `rho(n)` and `eta(n)`
  - decoder scaling using only completed exact DQI rows, with explicit per-instance fallback labeling if `ell=2` is unavailable
  - analytic resource scaling across P3-P7
  - wall-clock timings for completed runs and a boundary table for unavailable rows
  - verdict sentences derived from measured values, each tied back to a concrete table or row

## Testing and Verification Expectations

- Add or extend tests for:
  - frozen instance generation for P6/P7
  - CP-SAT exact agreement with brute-force at `n_bits=12` and `n_bits=14`
  - paired-artifact schema and shape validation for new instances
  - dynamic P-family discovery / matrix integration where covered by the current test suite
  - notebook/runtime smoke coverage if needed for Notebook 08 and Notebook 13 loaders
- Final verification must include:
  - `python -m pytest tests/ -x`
  - successful generation of the new instance JSON files
  - successful paired-artifact validation
  - regenerated metrics master containing P6 rows and P7 boundary or completed rows
  - successful execution of Notebook 08 and Notebook 13 with frozen outputs

## Operational Note

This workspace snapshot does not currently appear to be inside a git repository, so the design document can be written locally but cannot be committed through the normal brainstorming-skill review loop unless repository metadata becomes available later.
