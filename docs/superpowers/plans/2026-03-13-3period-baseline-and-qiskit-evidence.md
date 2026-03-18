# 3-Period Baseline And Qiskit Evidence Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add compatibility-safe reviewer aliases for the fixed 3-period baseline surface and make Notebook 07/09 reruns produce cleaner evidence displays without executing notebooks.

**Architecture:** Reuse the existing dynamic CP-SAT and shared-accounting horizon evaluation paths, add top-level aliases as projections of nested payloads, and keep benchmark/notebook changes presentation-focused. Extend tests around the current contracts rather than introducing new abstractions.

**Tech Stack:** Python, NumPy, OR-Tools CP-SAT, pytest, Jupyter notebook JSON

---

## Chunk 1: Baseline Alias Surface

### Task 1: Lock the desired multiperiod baseline contract in tests

**Files:**
- Modify: `tests/test_classical_baselines_multiperiod.py`
- Test: `tests/test_classical_baselines_multiperiod.py`

- [ ] **Step 1: Write failing contract tests**
- [ ] **Step 2: Run `pytest tests/test_classical_baselines_multiperiod.py -v` and confirm the new assertions fail**
- [ ] **Step 3: Patch `src/classical_baselines_multiperiod.py` with minimal alias projections and deterministic static tie-break behavior**
- [ ] **Step 4: Run `pytest tests/test_classical_baselines_multiperiod.py -v` and confirm pass**

### Task 2: Keep benchmark consumption thin

**Files:**
- Modify: `src/benchmark.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Add or adjust one benchmark contract assertion only if the alias-bearing payload requires it**
- [ ] **Step 2: Run `pytest tests/test_benchmark.py -k multiperiod -v` and confirm any new expectation fails before patching**
- [ ] **Step 3: Patch `src/benchmark.py` minimally so it consumes the alias-bearing payload directly**
- [ ] **Step 4: Run `pytest tests/test_benchmark.py -k multiperiod -v` and confirm pass**

## Chunk 2: Qiskit Contract And Notebook Rerun Surfaces

### Task 3: Tighten the Qiskit subset validation contract

**Files:**
- Modify: `tests/test_qiskit.py`
- Modify: `src/dqi_circuit_qiskit.py`

- [ ] **Step 1: Add a focused test for normalized `prob_dist` and required count fields**
- [ ] **Step 2: Run `pytest tests/test_qiskit.py -k subset_validation_report -v` and confirm behavior before patching**
- [ ] **Step 3: Patch `src/dqi_circuit_qiskit.py` only if the current report misses normalization or required keys**
- [ ] **Step 4: Run `pytest tests/test_qiskit.py -k subset_validation_report -v` and confirm pass**

### Task 4: Patch notebook cell sources without executing them

**Files:**
- Modify: `notebooks/09_3period_pricing_extension.ipynb`
- Modify: `notebooks/07_qiskit_structural_parity.ipynb`
- Test: `tests/test_notebook09_3period_pricing_extension_smoke.py`
- Test: `tests/test_notebook07_qiskit_structural_parity_smoke.py`

- [ ] **Step 1: Inspect current notebook source cells and smoke tests**
- [ ] **Step 2: Patch Notebook 09 to display instance summary, alias-based comparison, dynamic configuration details, capacity/remaining inventory, and smoothing penalty when enabled**
- [ ] **Step 3: Patch Notebook 07 to keep explicit subset-only wording and show comparison distance plus structural counts**
- [ ] **Step 4: Run notebook smoke tests and confirm the source-level evidence surface is present**

## Chunk 3: Verification

### Task 5: Run the targeted verification set

**Files:**
- Verify: `tests/test_classical_baselines_multiperiod.py`
- Verify: `tests/test_benchmark.py`
- Verify: `tests/test_qiskit.py`
- Verify: `tests/test_notebook09_3period_pricing_extension_smoke.py`
- Verify: `tests/test_notebook07_qiskit_structural_parity_smoke.py`

- [ ] **Step 1: Run the targeted pytest subset**
- [ ] **Step 2: Confirm no notebook outputs were rewritten**
- [ ] **Step 3: Summarize edited files and list notebook sections to rerun manually**

Plan complete and saved to `docs/superpowers/plans/2026-03-13-3period-baseline-and-qiskit-evidence.md`. Ready to execute.
