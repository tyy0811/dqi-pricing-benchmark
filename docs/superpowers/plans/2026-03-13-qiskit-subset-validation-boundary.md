# Qiskit Subset-Validation Boundary Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten the repo's Qiskit path to honest subset validation only, with explicit scope reporting and a reviewer-facing structural-subset notebook.

**Architecture:** Keep the current subset-only path in place and harden its contracts rather than refactoring. Standardize scope metadata across the Qiskit, NumPy-reference, resource-report, and notebook surfaces while preserving the legacy alias and keeping full decode/uncompute out of scope.

**Tech Stack:** Python, NumPy, Qiskit (optional at runtime), pytest, Jupyter notebook JSON

---

## Chunk 1: Subset Contract Tests

### Task 1: Tighten Qiskit subset contract coverage

**Files:**
- Modify: `tests/test_qiskit.py`
- Modify: `src/dqi_circuit_qiskit.py`

- [ ] **Step 1: Write the failing test**

Add assertions that `qiskit_subset_validation_report(...)` and the legacy alias surface explicit subset scope fields, preserve `decode_uncompute_included == False`, and keep `run_qiskit_statevector()` documented as a legacy alias without changing behavior.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_qiskit.py -q`
Expected: FAIL on missing or inconsistent subset-contract assertions.

- [ ] **Step 3: Write minimal implementation**

Update `src/dqi_circuit_qiskit.py` docstrings and payload shaping so:
- `build_dqi_circuit()` is described as implemented-subset structural only
- `qiskit_subset_syndrome_distribution(...)`, `subset_scope_metadata()`, and `qiskit_subset_validation_report(...)` use aligned subset-only language
- `run_qiskit_statevector()` remains a legacy alias
- `compare_frameworks()` remains disabled

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_qiskit.py -q`
Expected: PASS

### Task 2: Tighten NumPy subset and resource-report contract coverage

**Files:**
- Modify: `tests/test_qiskit.py`
- Modify: `src/dqi_state.py`
- Modify: `src/resources.py`

- [ ] **Step 1: Write the failing test**

Add assertions that the NumPy subset reference and the subset resource report expose explicit includes/excludes and always state that decode/uncompute is excluded.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_qiskit.py -q`
Expected: FAIL on missing or mismatched scope fields.

- [ ] **Step 3: Write minimal implementation**

Update `src/dqi_state.py` and `src/resources.py` so subset-facing payloads use the same scope language and explicit includes/excludes block.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_qiskit.py -q`
Expected: PASS

## Chunk 2: Reviewer-Facing Notebook

### Task 3: Rewrite Notebook 07 around the canonical subset APIs

**Files:**
- Modify: `tests/test_notebook07_qiskit_structural_parity_smoke.py`
- Modify: `notebooks/07_qiskit_structural_parity.ipynb`

- [ ] **Step 1: Write the failing test**

Tighten notebook smoke assertions so the notebook must:
- use the canonical subset APIs
- keep explicit failure handling for unavailable Qiskit and shape mismatch
- end with plain-language subset-only limitations

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_notebook07_qiskit_structural_parity_smoke.py -q`
Expected: FAIL if the notebook still uses outdated wording or interface preferences.

- [ ] **Step 3: Write minimal implementation**

Replace or rewrite `notebooks/07_qiskit_structural_parity.ipynb` to present:
- scope and limitations
- one tiny artifact load
- NumPy subset reference distribution
- Qiskit subset distribution
- TVD or equivalent scalar distance
- transpiled depth / CX / qubit count table
- explicit subset-only conclusion

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_notebook07_qiskit_structural_parity_smoke.py -q`
Expected: PASS

## Chunk 3: Acceptance Verification

### Task 4: Run targeted acceptance checks

**Files:**
- Verify: `src/dqi_circuit_qiskit.py`
- Verify: `src/dqi_state.py`
- Verify: `src/resources.py`
- Verify: `notebooks/07_qiskit_structural_parity.ipynb`
- Verify: `tests/test_qiskit.py`
- Verify: `tests/test_notebook07_qiskit_structural_parity_smoke.py`

- [ ] **Step 1: Run targeted verification suite**

Run: `pytest tests/test_qiskit.py tests/test_notebook07_qiskit_structural_parity_smoke.py -q`
Expected: PASS

- [ ] **Step 2: Verify acceptance criteria against the spec**

Check that:
- every public subset-facing function returns or references explicit scope metadata
- `decode_uncompute_included` is always `False` on the subset path
- the notebook uses canonical subset APIs
- no docstring, report, or markdown implies full end-to-end parity or framework equivalence

- [ ] **Step 3: Summarize repo-visible scope limitations**

Report the exact surfaced limitations:
- structural subset parity only
- decode/uncompute excluded
- no full end-to-end parity claim

