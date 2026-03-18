# Qiskit Subset-Validation Boundary Design

Date: 2026-03-13

## Goal

Lock in a narrow, honest Qiskit validation boundary for the repo's DQI path.
The implemented Qiskit surface is for structural subset validation only. It must
not imply full decode/uncompute support, full framework equivalence, or a full
end-to-end parity claim.

## Module Contracts

Files:
- `src/dqi_circuit_qiskit.py`
- `src/dqi_state.py`
- `src/resources.py`

### `src/dqi_circuit_qiskit.py`

Required contract:
- Keep `compare_frameworks()` explicitly disabled.
- Keep `run_qiskit_statevector()` as a legacy alias only, not the preferred
  interface.
- Make the canonical Qiskit API explicitly subset-only:
  - `qiskit_subset_syndrome_distribution(...)`
  - `subset_scope_metadata()`
  - `qiskit_subset_validation_report(...)`
- `build_dqi_circuit()` must be described as an implemented-subset structural
  circuit only.

Required scope language:
- decode/uncompute excluded
- no full end-to-end parity claim

### `src/dqi_state.py`

Required contract:
- `run_dqi_subset_reference(...)` remains the NumPy subset reference.
- Its modeled stages are limited to:
  - approximate low-weight superposition
  - phase kick
  - syndrome compute
  - final Hadamards
- It must explicitly exclude decode/uncompute.

### `src/resources.py`

Required contract:
- `qiskit_subset_resource_report(...)` must use the same subset-only scope
  language as the Qiskit validation path.
- The report must include a clearly labeled scope block with explicit
  includes/excludes.

### Module Acceptance Checks

- Every public subset-facing function returns or references explicit scope
  metadata.
- `decode_uncompute_included` is always `False` on the subset path.
- No docstring or payload text implies full decode/uncompute parity.
- No docstring or payload text implies full framework equivalence.

## Notebook Structure

File:
- `notebooks/07_qiskit_structural_parity.ipynb`

Required structure:
- Reviewer-facing subset-validation notebook.
- Include:
  - scope and limitations
  - one tiny artifact load
  - NumPy subset reference distribution
  - Qiskit subset distribution
  - TVD or equivalent scalar distance
  - small resource table with transpiled depth / CX / qubit count
- Keep failure handling explicit when Qiskit is unavailable.
- Keep failure handling explicit when distribution shapes differ.
- End with a plain-language conclusion stating:
  - structural subset parity only
  - decode/uncompute excluded
  - no full end-to-end parity claim
- Use canonical subset APIs, not `run_qiskit_statevector()` as the preferred
  notebook interface.

### Notebook Acceptance Checks

- The notebook reads cleanly top-to-bottom without suggesting unsupported
  claims.
- The final markdown conclusion states the subset-only limitation plainly.
- The notebook uses the canonical subset APIs.

## Verification and Testing Scope

Files, only if needed:
- `tests/test_qiskit.py`
- `tests/test_notebook07_qiskit_structural_parity_smoke.py`

Required testing scope:
- Keep test edits minimal.
- Preserve coverage that `compare_frameworks()` stays disabled.
- Add or tighten assertions that subset reports:
  - expose scope metadata
  - exclude decode/uncompute
  - preserve structured fallback behavior
- Update notebook smoke expectations only if wording changes while preserving
  the same limitation story.

### Verification Acceptance Checks

- Tests verify the subset path is explicit and unambiguous.
- No new test requires full Qiskit parity.
- No new test requires full-pipeline parity.
- Verification stays limited to contract honesty, report structure, and
  notebook messaging.

## Non-Goals

- No attempt at full decode/uncompute parity.
- No attempt at full framework equivalence.
- No broad API rewrite.
- No algorithmic change beyond scope/reporting hardening.

## Compatibility Constraints

- Keep legacy alias behavior intact where already present.
- Prefer additive contract clarification over renaming or schema churn.
- Preserve the current no-break migration path: subset-named APIs become the
  canonical interfaces in docs and notebooks while the legacy alias remains
  supported.

## Implementation Boundaries

- Keep changes small and honest.
- Do not expand the implemented Qiskit path beyond the agreed subset.
- Do not add new claims that depend on decode/uncompute or coherent reversible
  decoding.
- Prefer in-place tightening of contracts, notebook messaging, and test
  assertions over refactoring.

