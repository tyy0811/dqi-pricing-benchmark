# Pricing Scaling P6 P7 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add strict-superset P6 and P7 pricing instances as first-class P-family members, integrate them into exact paired-encoding and benchmark/reporting pipelines, and ship Notebook 13 with explicit scaling and boundary analysis.

**Architecture:** Extend the existing pricing-family code paths in place. Add new frozen-instance builders and exact classical validation, widen P-family discovery in matrix/report helpers, preserve exact statevector execution with structured boundary records instead of approximate fallbacks, and build notebook/report outputs dynamically from canonical metrics artifacts.

**Tech Stack:** Python, NumPy, pandas, OR-Tools CP-SAT, Jupyter notebooks, existing pricing/DQI pipeline modules.

---

## Chunk 1: Instance and Solver Foundations

### Task 1: Map current P-family loading and frozen-instance seams

**Files:**
- Modify: `src/problem_generator.py`
- Modify: `src/matrix_runner.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Write the failing test for wider P-family instance resolution**

Add a test asserting the relevant loader/path helper can resolve `pricing_6feat_12bit.json` and `pricing_7feat_14bit.json` or `P6`/`P7` once registered.

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `python -m pytest tests/test_benchmark.py -k "pricing_6feat or pricing_7feat or P6 or P7" -v`
Expected: FAIL because current code only knows P3/P4/P5.

- [ ] **Step 3: Inspect current registration points**

Read the exact helpers in `src/matrix_runner.py` and any related loaders/manifests that map `n_features` to P-family instance IDs and file paths.

- [ ] **Step 4: Implement the minimal registration extension**

Update the relevant P-family discovery/loading helpers to support `n_features` 6 and 7 without changing P3/P4/P5 behavior.

- [ ] **Step 5: Re-run the targeted test**

Run: `python -m pytest tests/test_benchmark.py -k "pricing_6feat or pricing_7feat or P6 or P7" -v`
Expected: PASS.

### Task 2: Add strict-superset P6/P7 instance builders

**Files:**
- Modify: `src/problem_generator.py`
- Test: `tests/test_problem_generator.py`

- [ ] **Step 1: Write the failing tests for P6/P7 generation rules**

Add tests that assert:
- features 1-5 in P6/P7 match P5 exactly
- P6 has 6 features and 12 bits; P7 has 7 features and 14 bits
- the first three segments match the legacy segment definitions over features 1-5
- P7 embeds the P5 optimum feasibly with new features disabled

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m pytest tests/test_problem_generator.py -k "p6 or p7 or superset or frozen" -v`
Expected: FAIL because builders/tests do not exist yet.

- [ ] **Step 3: Implement minimal instance-builder support**

Add focused builders or catalog helpers in `src/problem_generator.py` for:
- the frozen 5-feature base catalog
- additive feature 6 / feature 7 definitions
- byte-identical carryover of the original three segments
- the added fourth segment
- preserved legacy bundle rules plus 1-2 new bundle rules involving new features
- `max_luxury=3` for larger instances

- [ ] **Step 4: Extend frozen instance generation**

Update `generate_frozen_instances(...)` to emit P3-P7 JSON files without mutating existing P3-P5 definitions.

- [ ] **Step 5: Re-run the targeted tests**

Run: `python -m pytest tests/test_problem_generator.py -k "p6 or p7 or superset or frozen" -v`
Expected: PASS.

### Task 3: Lift and document the CP-SAT exact boundary

**Files:**
- Modify: `src/pricing_model_ortools.py`
- Test: `tests/test_pricing_model_ortools.py`

- [ ] **Step 1: Write the failing tests for 12-bit and 14-bit CP-SAT support**

Add tests that solve generated P6/P7 problems and assert:
- status is `optimal` or `feasible`
- `objective_value` matches brute-force exactly
- metadata exposes model type (`one_hot_exact`) and decision-variable count

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m pytest tests/test_pricing_model_ortools.py -k "12bit or 14bit or p6 or p7" -v`
Expected: FAIL because the current guard rejects `n_bits > 10`.

- [ ] **Step 3: Implement the minimal solver extension**

Remove or relax the `n_bits > 10` guard, keep the current one-hot exact semantics intact, and surface stable metadata fields for model type and variable count.

- [ ] **Step 4: Re-run the targeted tests**

Run: `python -m pytest tests/test_pricing_model_ortools.py -k "12bit or 14bit or p6 or p7" -v`
Expected: PASS.

## Chunk 2: Artifact Generation and Classical Validation

### Task 4: Add P6/P7 artifact-generation tests

**Files:**
- Modify: `tests/test_pricing_pairing.py`
- Modify: `tests/test_baselines.py`
- Modify: `src/problem_generator.py`

- [ ] **Step 1: Write the failing tests for new frozen JSON and pairing support**

Add tests that:
- generate or load `pricing_6feat_12bit.json` and `pricing_7feat_14bit.json`
- call `solve_pricing_pair(...)` with `k=15`
- assert paired-artifact top-level and encoding subsets match schema expectations

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m pytest tests/test_pricing_pairing.py tests/test_baselines.py -k "pricing_6feat or pricing_7feat or p6 or p7" -v`
Expected: FAIL because fixtures/files do not yet exist.

- [ ] **Step 3: Implement artifact-generation support**

Ensure the new frozen JSON files can be generated and loaded by existing artifact builders, and add any missing loader support for legacy surrogate file naming.

- [ ] **Step 4: Re-run the targeted tests**

Run: `python -m pytest tests/test_pricing_pairing.py tests/test_baselines.py -k "pricing_6feat or pricing_7feat or p6 or p7" -v`
Expected: PASS.

### Task 5: Add classical validation outputs and P7 WHT diagnostics

**Files:**
- Modify: `src/pricing_ilp.py`
- Modify: `src/reduction.py`
- Create or Modify: `src/encoding_compare.py`
- Test: `tests/test_pricing_pairing.py`

- [ ] **Step 1: Write the failing test for P7 diagnostic thresholds**

Add a test that computes/stubs paired diagnostics for P7 and asserts the code can:
- compute `eta`, rho, and regret metrics for `m=15`
- emit a diagnostic-only `m=20` artifact when threshold conditions are met

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `python -m pytest tests/test_pricing_pairing.py -k "eta or rho or m20 or diagnostic" -v`
Expected: FAIL because thresholded diagnostic generation does not exist yet.

- [ ] **Step 3: Implement the minimal diagnostic path**

Add reusable helpers that evaluate full truth-table encoding diagnostics for the new paired artifacts, and generate the optional P7 `m=20` paired artifact tagged as diagnostic-only when triggered.

- [ ] **Step 4: Re-run the targeted test**

Run: `python -m pytest tests/test_pricing_pairing.py -k "eta or rho or m20 or diagnostic" -v`
Expected: PASS.

## Chunk 3: Matrix A / Matrix C / Metrics Integration

### Task 6: Extend Matrix A P-family coverage

**Files:**
- Modify: `src/matrix_runner.py`
- Modify: `scripts/make_report.py`
- Test: `tests/test_make_report.py`

- [ ] **Step 1: Write the failing tests for P6/P7 Matrix A integration**

Add tests that assert report regeneration and metrics-master building can ingest P6/P7 pricing rows dynamically without hard-coded P3-P5 assumptions.

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m pytest tests/test_make_report.py -k "matrix_a and (P6 or P7 or dynamic)" -v`
Expected: FAIL if current logic assumes only existing P-family members.

- [ ] **Step 3: Implement the minimal Matrix A extension**

Widen P-family iteration and any stage-e aggregation filters so P6/P7 rows flow through the same canonical report/metrics path.

- [ ] **Step 4: Re-run the targeted tests**

Run: `python -m pytest tests/test_make_report.py -k "matrix_a and (P6 or P7 or dynamic)" -v`
Expected: PASS.

### Task 7: Extend Matrix C with structured boundary records

**Files:**
- Modify: `src/matrix_runner.py`
- Modify: `src/report_schema.py` (if boundary fields are validated there)
- Test: `tests/test_make_report.py`
- Test: `tests/test_stage_e_quality_cost_builder.py`

- [ ] **Step 1: Write the failing tests for boundary-record persistence**

Add tests that assert a failed/unavailable P-family DQI run can persist:
- `run_status`
- `boundary_reason`
- estimated qubits / memory
- inclusion in downstream report/metrics outputs

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m pytest tests/test_make_report.py tests/test_stage_e_quality_cost_builder.py -k "boundary or unavailable or P7" -v`
Expected: FAIL because boundary fields are not yet part of canonical outputs.

- [ ] **Step 3: Implement the minimal Matrix C boundary handling**

Add exact execution guards around P-family DQI runs, populate structured boundary records on failure or pre-emptive resource rejection, and preserve existing completed-record shape.

- [ ] **Step 4: Normalize timing and memory fields**

Store `wall_clock_s` for completed runs and `peak_memory_mb` when measurable, without breaking existing consumers that still read `runtime_sec`.

- [ ] **Step 5: Re-run the targeted tests**

Run: `python -m pytest tests/test_make_report.py tests/test_stage_e_quality_cost_builder.py -k "boundary or unavailable or P7" -v`
Expected: PASS.

### Task 8: Verify metrics-master dynamic behavior

**Files:**
- Modify: `notebooks/_helpers.py` (if loader normalization is needed)
- Test: `tests/test_notebook_runtime_smoke_missing_data.py`

- [ ] **Step 1: Write the failing tests for dynamic metrics-master reads**

Add tests proving notebook/runtime loaders tolerate:
- new P-family instances
- completed P6 rows
- P7 boundary-only rows
- no hard-coded record counts

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m pytest tests/test_notebook_runtime_smoke_missing_data.py -k "P6 or P7 or boundary or metrics" -v`
Expected: FAIL if current notebook helper assumptions are too rigid.

- [ ] **Step 3: Implement minimal loader normalization**

Adjust helper logic only where needed so metrics-master consumers remain dynamic and backward-compatible.

- [ ] **Step 4: Re-run the targeted tests**

Run: `python -m pytest tests/test_notebook_runtime_smoke_missing_data.py -k "P6 or P7 or boundary or metrics" -v`
Expected: PASS.

## Chunk 4: Notebook Updates

### Task 9: Update Notebook 08 paired-encoding comparison

**Files:**
- Modify: `notebooks/08_paired_encoding_comparison.ipynb`
- Test: `tests/test_notebook08_paired_encoding_smoke.py`

- [ ] **Step 1: Write or extend the failing smoke test**

Add/extend a test that asserts Notebook 08 computes matched records dynamically across P-family instances and does not depend on a fixed total like 18.

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `python -m pytest tests/test_notebook08_paired_encoding_smoke.py -v`
Expected: FAIL if the notebook still hard-codes legacy assumptions.

- [ ] **Step 3: Update the notebook minimally**

Change the matched-record filtering and summary-statistic cells so P6/P7 are included when completed paired rows exist, and boundary-only rows are naturally excluded.

- [ ] **Step 4: Re-run the targeted test**

Run: `python -m pytest tests/test_notebook08_paired_encoding_smoke.py -v`
Expected: PASS.

### Task 10: Create Notebook 13 scaling analysis

**Files:**
- Create: `notebooks/13_scaling_analysis.ipynb`
- Modify: `notebooks/_helpers.py` (only if helper extraction materially reduces duplication)
- Test: `tests/test_notebook_runtime_smoke_missing_data.py`

- [ ] **Step 1: Write the failing notebook smoke test**

Add a smoke/runtime test that executes Notebook 13 against representative metrics artifacts and asserts:
- it renders P3-P7 instance summary rows
- it includes P7 encoding diagnostics even without completed DQI rows
- it includes a boundary table for unavailable rows

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `python -m pytest tests/test_notebook_runtime_smoke_missing_data.py -k "notebook13 or scaling_analysis" -v`
Expected: FAIL because the notebook does not exist yet.

- [ ] **Step 3: Implement Notebook 13 minimally**

Create the notebook with sections for:
- instance summary and CP-SAT model metadata
- encoding scaling (`rho`, `eta`, regret)
- exact decoder scaling using completed rows only
- analytic resource scaling
- completed timing table plus boundary table
- measured verdict sentences tied to concrete rows/tables

- [ ] **Step 4: Re-run the targeted test**

Run: `python -m pytest tests/test_notebook_runtime_smoke_missing_data.py -k "notebook13 or scaling_analysis" -v`
Expected: PASS.

## Chunk 5: Execution, Output Freezing, and Final Verification

### Task 11: Generate frozen data and paired artifacts

**Files:**
- Modify or Create: generated files under `data/instances/`
- Modify or Create: generated files under `results/paired_pricing/`

- [ ] **Step 1: Generate new frozen instances**

Run the repo-local generation entrypoint for pricing instances so:
- `data/instances/pricing_6feat_12bit.json`
- `data/instances/pricing_7feat_14bit.json`
are written with the canonical schema.

- [ ] **Step 2: Generate paired artifacts and legacy surrogates**

Run the paired-artifact generation path so:
- `results/paired_pricing/pricing_6feat_12bit.paired.json`
- `results/paired_pricing/pricing_7feat_14bit.paired.json`
- optional `results/paired_pricing/pricing_7feat_14bit_m20.paired.json`
- legacy surrogate JSON files
are written.

- [ ] **Step 3: Verify classical agreement**

Run the brute-force and CP-SAT validation flow and capture the reported feasible counts, optima, solve times, and model metadata for use in Notebook 13.

### Task 12: Execute Matrix A / Matrix C and regenerate reports

**Files:**
- Modify or Create: generated files under `results/`

- [ ] **Step 1: Run the canonical benchmark entrypoints**

Execute the exact benchmark/reporting commands needed to generate P6/P7 Matrix A and Matrix C rows under the canonical results tree.

- [ ] **Step 2: Regenerate metrics master and reports**

Run the existing report builder so Notebook 08 and Notebook 13 consume refreshed canonical outputs.

- [ ] **Step 3: Confirm invariants**

Check that:
- P6 has completed quality rows for at least `ell=1`
- P7 has completed rows where tractable or structured boundary rows otherwise
- legacy P3/P4/P5 result files remain untouched

### Task 13: Execute notebooks and full test suite

**Files:**
- Modify or Create: executed notebook outputs

- [ ] **Step 1: Execute Notebook 08**

Run the notebook execution command used in this repo and confirm it completes with frozen outputs.

- [ ] **Step 2: Execute Notebook 13**

Run the notebook execution command used in this repo and confirm it completes with frozen outputs.

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest tests/ -x`
Expected: PASS.

- [ ] **Step 4: Perform artifact spot checks**

Confirm the final checklist items manually:
- new instance JSON files exist and load
- paired artifacts validate
- metrics master contains P6 and P7 rows as designed
- Notebook 13 shows P7 encoding/resource diagnostics and any boundary rows

## Notes for Execution

- Preserve TDD discipline within each task: write or extend the failing test first, watch it fail, then implement the minimum code to pass.
- Use `m=15` as canonical throughout execution. Treat any P7 `m=20` output as diagnostic-only and keep it out of canonical benchmark comparisons unless an existing schema already namespaces it cleanly.
- If a benchmark command fails because statevector resources are exceeded, convert that failure into a structured boundary record instead of retrying with approximate simulation.
- This workspace currently lacks visible git metadata, so commit steps from the generic template cannot be executed unless repository metadata becomes available later.
