# Reviewer-Facing Benchmark Story Design

**Date:** 2026-03-13  
**Status:** Approved for planning  
**Scope:** Design only; no implementation in this document.

---

## 1. Decision

Phase 1 changes the **reviewer-facing story** of the repo without changing the underlying algorithm set.

The benchmark will present:

- exact-reference / ILP-derived encoding as the mainline reviewer-facing path
- mixture execution as the default execution mode
- WHT as an exploratory surrogate path
- coherent analysis as appendix-only for tiny supported instances
- Qiskit as structural subset validation only

This phase introduces a thin reporting adapter in `src/benchmark.py` that changes ordering, labels, and filtering for summaries while preserving existing payload keys wherever possible.

---

## 2. Non-Goals

This phase does **not** include:

- new algorithms
- benchmark schema redesign
- broad renaming of stable result payload keys
- changes to `quality_vs_cost` semantics
- changes to legacy alias behavior
- claims of full end-to-end Qiskit parity

---

## 3. Compatibility Boundary

### 3.1 Stable compatibility surface

The following remain valid in Phase 1:

- existing benchmark result dict keys unless change is unavoidable
- `quality_vs_cost` embedded payload
- legacy aliases such as `dqi_uniform`, `dqi_paper`, `dqi_heuristic`, `dqi_optimal`, and `dqi_heuristic_optimal`

### 3.2 Reporting-layer change only

The spec distinguishes **payload keys** from **reviewer-facing labels**:

- payload keys remain unchanged unless unavoidable
- reviewer-facing labels may change
- new ordering/filtering lives in a reporting adapter layer

This means internal data production can remain mostly stable while user-visible summaries change immediately.

### 3.3 Canonical reviewer-facing encoding name

The canonical reviewer-facing encoding label is:

- `ilp_derived_exact`

Older internal names may continue to exist for compatibility. The new label is required for reviewer-facing text, tables, and documentation.

---

## 4. Execution, Presentation, and Appendix Policy

### 4.1 Execution default

Mainline mixture execution remains available and backward-compatible.

The default benchmark run still supports the current mixture mainline and does not require schema churn to preserve existing consumers.

### 4.2 Presentation default

Reviewer-facing headline summaries surface only the mainline mixture story by default.

Default headline order:

1. `dqi_paper_mixture`
2. `dqi_paper_mixture_bp1`
3. paired encoding comparison

Coherent is excluded from the default headline story.

### 4.3 Appendix compute/render flags

`src/benchmark.py` adds two minimal flags:

- `include_alpha_ablation: bool = False`
- `include_coherent_appendix: bool = False`

These flags control appendix rendering policy for reviewer-facing output. Alpha and coherent material may be skipped unless explicitly requested.

### 4.4 Appendix rendering policy

Alpha and coherent sections render only when:

- explicitly enabled by their respective flags
- required metrics are present
- required metrics are non-NaN

Incomplete coherent or alpha rows should be omitted from reviewer-facing appendix sections rather than printed as partial comparisons.

---

## 5. Reporting Adapter in `src/benchmark.py`

### 5.1 Purpose

`src/benchmark.py` gains a small reviewer-facing display layer that:

- maps stable payload keys to reviewer-facing labels
- enforces reviewer-facing ordering
- filters appendix-only rows
- keeps headline and appendix policy out of the raw result assembly paths

### 5.2 Single explicit filtering helper

Filtering logic must not be scattered across print paths.

Phase 1 adds one explicit internal helper for coherent appendix rendering, for example:

- `should_render_coherent_appendix(result_row) -> bool`

This helper defines the coherent appendix rule once:

- feature flag enabled
- metrics complete
- required values not NaN

If a similar helper is needed for alpha appendix rendering, it should follow the same pattern, but the coherent rule must be explicit and centralized.

### 5.3 Payload preservation

Existing raw result payload generation should remain intact wherever possible.

The reporting adapter should consume existing result keys such as:

- `dqi_paper_mixture`
- `dqi_paper_mixture_bp1`
- `dqi_paper_coherent`
- `stage1_paired_encoding`

without broadly renaming them.

---

## 6. Headline and Appendix Ordering

### 6.1 Headline block

Reviewer-facing headline summaries show:

1. paper-alpha mixture mainline
2. paper-alpha mixture with BP1 decoder
3. paired encoding comparison, labeled `ilp_derived_exact`

This applies to CLI summary output and comparison-table style reporting inside `src/benchmark.py`.

### 6.2 Alpha appendix block

Uniform/heuristic alpha comparisons move out of the default headline path.

They may still exist in payloads for compatibility, but they should only render in reviewer-facing appendix material when `include_alpha_ablation=True` and the required metrics are complete.

### 6.3 Coherent appendix block

Coherent rows move out of the main headline comparison table.

They appear only in an appendix block when:

- `include_coherent_appendix=True`
- coherent metrics are available
- required coherent metrics are non-NaN

Unsupported or incomplete coherent runs should not produce reviewer-facing appendix rows.

---

## 7. Documentation Policy

### 7.1 README positioning

`README.md` must describe the repo as:

- pricing-to-DQI benchmark
- exact-reference / ILP-derived mainline
- mixture-mode default
- WHT exploratory
- decoder diagnostics
- Qiskit structural subset validation

Required one-line pitch:

`Pricing-to-DQI benchmark with exact-reference (ILP-derived) encoding as the mainline path, mixture-mode evaluation as the default execution mode, WHT as an exploratory surrogate, decoder diagnostics, and Qiskit structural subset validation.`

### 7.2 README scope note

README must explicitly separate:

- Mainline: mixture execution on pricing artifacts with exact-reference / ILP-derived encoding
- Exploratory: WHT-truncated surrogate artifacts
- Appendix only: coherent / alpha finite-size analyses on tiny supported instances
- Structural only: Qiskit subset checks, not decode-inclusive end-to-end parity

### 7.3 Reviewer-facing notebook order

The reviewer-facing notebook order is:

1. `04_pricing_pipeline`
2. `08_paired_encoding_comparison`
3. `05_decoder_comparison`
4. `06_resource_analysis`
5. `07_qiskit_structural_parity`
6. `09_3-period_pricing_extension`
7. `10_alpha_finite_size_validation`
8. `03_paper_faithful_dqi`

Notebooks 10 and 03 are appendix-facing in this ordering.

### 7.4 Theoretical framework headings

`Theoretical_Framework.md` should use intentionally asymmetric headings:

- Mainline benchmark path
- Exploratory surrogate path (WHT)
- Structural Qiskit subset validation

The language must avoid implying these are equally mature paths.

### 7.5 Qiskit wording constraint

Docs must not imply full end-to-end Qiskit parity.

Qiskit is described only as an implemented structural subset or structural validation path.

### 7.6 Coherent wording constraint

Docs must describe coherent mode as appendix-only and restricted to tiny supported instances.

---

## 8. Acceptance Tests

Phase 1 implementation should include explicit tests for:

### 8.1 Ordering

A test that the default reviewer-facing summary order is:

1. `dqi_paper_mixture`
2. `dqi_paper_mixture_bp1`
3. paired encoding comparison

with coherent excluded by default.

### 8.2 Coherent omission

A test that incomplete or unsupported coherent rows are omitted from the reviewer-facing appendix block, even if a coherent payload object exists.

### 8.3 Payload compatibility

A test that existing payload keys remain present and valid after the reporting-layer change, including:

- main result dict keys
- `quality_vs_cost`
- legacy aliases

---

## 9. Files in Scope

Phase 1 modifies only:

- `README.md`
- `Theoretical_Framework.md`
- `src/benchmark.py`

No other files are in scope unless a minimal test adjustment becomes strictly necessary during implementation.

---

## 10. Implementation Notes for Planning

The implementation plan should assume:

- no algorithmic changes
- minimal API change
- backward-compatible execution defaults
- reviewer-facing summaries are the primary change surface
- stable payload keys take precedence over cosmetic renaming

The plan should decompose work into:

1. doc updates
2. reporting adapter changes in `src/benchmark.py`
3. acceptance-test coverage for ordering, coherent omission, and payload compatibility
