# Stage 3: Resource Analysis Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add gate-level resource estimation to the DQI pipeline with analytical Dicke formulas and per-stage breakdown.

**Architecture:** Two modules (`dicke_circuit.py` for Dicke formulas, `resources.py` for composition), one test file, one notebook, one results export. TDD approach throughout.

**Tech Stack:** NumPy, JSON, Matplotlib (notebook only)

**Spec:** `docs/superpowers/specs/2026-03-10-stage3-resource-analysis-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/dicke_circuit.py` | Analytical Dicke state resource formulas |
| Create | `src/resources.py` | DQI resource composition + artifact loading |
| Create | `tests/test_resources.py` | Unit tests for formulas and schema |
| Create | `notebooks/06_resource_analysis.ipynb` | Scaling comparison + k-sensitivity |
| Generate | `results/resources_scaling.json` | Aggregated resource reports |

---

## Chunk 1: Dicke Circuit Module

### Task 1: Dicke resources trivial case (t=0)

**Files:**
- Create: `tests/test_resources.py`
- Create: `src/dicke_circuit.py`

- [ ] **Step 1.1: Write the failing test for t=0**

```python
# tests/test_resources.py
"""Tests for resource estimation modules."""

import json

import numpy as np
import pytest

from src.dicke_circuit import dicke_resources


def test_dicke_resources_trivial_case():
    """t=0 returns zero CNOTs (trivial state |00...0>)."""
    result = dicke_resources(m=10, t=0)

    assert result["cnot_est"] == 0
    assert result["single_qubit_rot_est"] == 0
    assert result["depth_est"] == 0
    assert result["method"] == "bartschi_eidenbenz_scs"
    assert "reference" in result
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `pytest tests/test_resources.py::test_dicke_resources_trivial_case -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.dicke_circuit'`

- [ ] **Step 1.3: Write minimal implementation**

```python
# src/dicke_circuit.py
"""
dicke_circuit.py — Analytical resource estimates for Dicke state preparation.

Implements resource formulas based on Bärtschi-Eidenbenz (2019) deterministic
SCS (Split-Cyclic-Shift) construction for Dicke states.

Reference: Bärtschi & Eidenbenz, "Deterministic Preparation of Dicke States,"
           arXiv:1904.07358 (2019).
"""

from __future__ import annotations


def dicke_resources(m: int, t: int) -> dict:
    """Analytic resource estimate for preparing |D_t^m⟩ via deterministic SCS construction.

    For the trivial cases:
    - t=0: |D_0^m⟩ = |00...0⟩, no gates needed
    - t=m: |D_m^m⟩ = |11...1⟩, just X gates

    For general t in (0, m):
    - CNOTs ≈ t(m-t) from the SCS structure
    - Single-qubit rotations ≈ (m-1) controlled-Ry gates
    - Depth ≈ m (linear in qubit count)

    Args:
        m: Number of qubits.
        t: Target Hamming weight.

    Returns:
        dict with keys: cnot_est, single_qubit_rot_est, depth_est, method, reference
    """
    if t < 0 or t > m:
        raise ValueError(f"t={t} must be in [0, m={m}]")
    if m < 0:
        raise ValueError(f"m={m} must be non-negative")

    # Trivial cases
    if t == 0:
        return {
            "cnot_est": 0,
            "single_qubit_rot_est": 0,
            "depth_est": 0,
            "method": "bartschi_eidenbenz_scs",
            "reference": "arXiv:1904.07358",
        }

    if t == m:
        # Just X gates on all qubits, no CNOTs
        return {
            "cnot_est": 0,
            "single_qubit_rot_est": m,  # X gates
            "depth_est": 1,
            "method": "bartschi_eidenbenz_scs",
            "reference": "arXiv:1904.07358",
        }

    # General case: SCS construction
    # CNOTs: t(m-t) from the recursive split structure
    cnot_est = t * (m - t)

    # Single-qubit rotations: (m-1) controlled-Ry for the splits
    single_qubit_rot_est = m - 1

    # Depth: O(m), specifically m layers in the SCS construction
    depth_est = m

    return {
        "cnot_est": cnot_est,
        "single_qubit_rot_est": single_qubit_rot_est,
        "depth_est": depth_est,
        "method": "bartschi_eidenbenz_scs",
        "reference": "arXiv:1904.07358",
    }
```

- [ ] **Step 1.4: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_dicke_resources_trivial_case -v`
Expected: PASS

---

### Task 2: Dicke resources symmetry test

**Files:**
- Modify: `tests/test_resources.py`

- [ ] **Step 2.1: Write the failing test for symmetry**

```python
# Add to tests/test_resources.py

def test_dicke_resources_symmetry_t_vs_m_minus_t():
    """t and m-t should have same CNOT count due to t(m-t) symmetry."""
    m = 10

    result_t3 = dicke_resources(m=m, t=3)
    result_t7 = dicke_resources(m=m, t=7)

    # t(m-t) = 3*7 = 21 for both
    assert result_t3["cnot_est"] == result_t7["cnot_est"]
    assert result_t3["cnot_est"] == 3 * 7  # = 21
```

- [ ] **Step 2.2: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_dicke_resources_symmetry_t_vs_m_minus_t -v`
Expected: PASS (implementation already handles this)

---

### Task 3: Dicke resources scaling pattern

**Files:**
- Modify: `tests/test_resources.py`

- [ ] **Step 3.1: Write the test for scaling pattern**

```python
# Add to tests/test_resources.py

def test_dicke_resources_scaling_pattern():
    """CNOT count follows t(m-t) pattern."""
    m = 10

    for t in range(1, m):
        result = dicke_resources(m=m, t=t)
        expected_cnot = t * (m - t)
        assert result["cnot_est"] == expected_cnot, f"Failed for t={t}"
```

- [ ] **Step 3.2: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_dicke_resources_scaling_pattern -v`
Expected: PASS

---

### Task 4: Dicke superposition upper bound

**Files:**
- Modify: `tests/test_resources.py`
- Modify: `src/dicke_circuit.py`

- [ ] **Step 4.1: Write the failing test for superposition keys**

```python
# Add to tests/test_resources.py
from src.dicke_circuit import dicke_resources, dicke_superposition_upper_bound


def test_dicke_superposition_upper_bound_keys():
    """Returns expected keys."""
    result = dicke_superposition_upper_bound(m=10, ell=3)

    assert "cnot_est" in result
    assert "single_qubit_rot_est" in result
    assert "depth_est" in result
    assert "method" in result
    assert "per_weight_breakdown" in result
    assert result["method"] == "bartschi_eidenbenz_scs_upper_bound"
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `pytest tests/test_resources.py::test_dicke_superposition_upper_bound_keys -v`
Expected: FAIL with `ImportError`

- [ ] **Step 4.3: Write implementation**

```python
# Add to src/dicke_circuit.py

def dicke_superposition_upper_bound(m: int, ell: int) -> dict:
    """Naive upper-bound estimate for Σ_{t=0}^ell α_t |D_t^m⟩ by aggregating per-t Dicke costs.

    NOT a compiled circuit count for a specific weighted-superposition construction.
    This is an upper bound assuming independent preparation of each Dicke state.

    Args:
        m: Number of qubits in error register.
        ell: Maximum Hamming weight (inclusive).

    Returns:
        dict with keys: cnot_est, single_qubit_rot_est, depth_est, method,
                        per_weight_breakdown
    """
    if ell < 0:
        raise ValueError(f"ell={ell} must be non-negative")
    if ell > m:
        raise ValueError(f"ell={ell} cannot exceed m={m}")

    per_weight = []
    total_cnot = 0
    total_single_qubit = 0
    max_depth = 0

    for t in range(ell + 1):
        res_t = dicke_resources(m, t)
        per_weight.append({
            "t": t,
            "cnot_est": res_t["cnot_est"],
            "single_qubit_rot_est": res_t["single_qubit_rot_est"],
            "depth_est": res_t["depth_est"],
        })
        total_cnot += res_t["cnot_est"]
        total_single_qubit += res_t["single_qubit_rot_est"]
        max_depth = max(max_depth, res_t["depth_est"])

    return {
        "cnot_est": total_cnot,
        "single_qubit_rot_est": total_single_qubit,
        "depth_est": max_depth,  # Conservative: max of individual depths
        "method": "bartschi_eidenbenz_scs_upper_bound",
        "reference": "arXiv:1904.07358",
        "per_weight_breakdown": per_weight,
    }
```

- [ ] **Step 4.4: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_dicke_superposition_upper_bound_keys -v`
Expected: PASS

---

### Task 5: Dicke superposition aggregation test

**Files:**
- Modify: `tests/test_resources.py`

- [ ] **Step 5.1: Write the test for exact aggregation**

```python
# Add to tests/test_resources.py

def test_dicke_superposition_upper_bound_aggregation():
    """Upper bound is exact sum of per-t costs."""
    m = 10
    ell = 3

    result = dicke_superposition_upper_bound(m=m, ell=ell)

    # Manually compute expected sum
    expected_cnot = sum(dicke_resources(m, t)["cnot_est"] for t in range(ell + 1))
    expected_single = sum(dicke_resources(m, t)["single_qubit_rot_est"] for t in range(ell + 1))

    assert result["cnot_est"] == expected_cnot
    assert result["single_qubit_rot_est"] == expected_single

    # Per-weight breakdown should have ell+1 entries
    assert len(result["per_weight_breakdown"]) == ell + 1
```

- [ ] **Step 5.2: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_dicke_superposition_upper_bound_aggregation -v`
Expected: PASS

- [ ] **Step 5.3: Run all dicke tests**

Run: `pytest tests/test_resources.py -v`
Expected: All 5 tests PASS

---

## Chunk 2: DQI Resource Estimation Module

### Task 6: DQI resource estimate schema

**Files:**
- Modify: `tests/test_resources.py`
- Create: `src/resources.py`

- [ ] **Step 6.1: Write the failing test for schema**

```python
# Add to tests/test_resources.py
import numpy as np
from src.resources import dqi_resource_estimate


def test_dqi_resource_estimate_schema():
    """All required fields are present."""
    # Small test instance
    B = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)
    v = np.array([0, 1], dtype=np.int8)
    ell = 2

    result = dqi_resource_estimate(B, v, ell, instance_id="test_instance")

    # Instance metadata
    assert result["instance_id"] == "test_instance"
    assert result["n"] == 3
    assert result["m"] == 2
    assert result["ell"] == 2

    # Dicke estimates
    assert "dicke_method" in result
    assert "dicke_cnot_est" in result
    assert "dicke_single_qubit_rot_est" in result
    assert "dicke_depth_est" in result

    # Other stages
    assert "syndrome_cnot_count" in result
    assert "phase_z_count" in result
    assert "final_h_count" in result

    # Aggregates
    assert "total_gate_est" in result
    assert "total_depth_est" in result

    # Qubit breakdown
    assert "n_problem_qubits" in result
    assert "m_error_qubits" in result
    assert "n_syndrome_qubits" in result
    assert "ancilla_qubits_est" in result
    assert "total_qubits_est" in result

    # Register assumptions
    assert "register_assumptions" in result

    # Stage tracking
    assert result["included_stages"] == ["dicke_prep", "phase_kick", "syndrome_compute", "final_hadamard"]
    assert result["excluded_stages"] == ["decode_uncompute"]
    assert result["decode_uncompute_status"] == "not_implemented"

    # Provenance
    assert "assumptions" in result
    assert "reference" in result
```

- [ ] **Step 6.2: Run test to verify it fails**

Run: `pytest tests/test_resources.py::test_dqi_resource_estimate_schema -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 6.3: Write implementation**

```python
# src/resources.py
"""
resources.py — DQI resource estimation by stage composition.

Computes gate counts, qubit requirements, and depth estimates for the
implemented DQI circuit stages. Decode/uncompute is explicitly excluded.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.dicke_circuit import dicke_superposition_upper_bound


def _validate_binary(v: np.ndarray, name: str = "v") -> None:
    """Raise ValueError if array is not binary (0/1 only)."""
    v_arr = np.asarray(v)
    if not np.all((v_arr == 0) | (v_arr == 1)):
        raise ValueError(f"{name} must be binary (0/1 only), got values: {np.unique(v_arr)}")


def _validate_shapes(B: np.ndarray, v: np.ndarray) -> None:
    """Raise ValueError if B and v have incompatible shapes."""
    if B.ndim != 2:
        raise ValueError(f"B must be 2D, got shape {B.shape}")
    if v.ndim != 1:
        raise ValueError(f"v must be 1D, got shape {v.shape}")
    if B.shape[0] != v.shape[0]:
        raise ValueError(f"B has {B.shape[0]} rows but v has {v.shape[0]} elements")


def dqi_resource_estimate(
    B: np.ndarray,
    v: np.ndarray,
    ell: int,
    instance_id: str | None = None,
    k_selected: int | None = None,
    artifact_version: str | None = None,
    repo_commit: str | None = None,
) -> dict[str, Any]:
    """Full DQI resource estimate for a (B, v) surrogate artifact.

    Composes: dicke_prep + phase_kick + syndrome_compute + final_hadamard
    Excludes: decode_uncompute (explicitly labeled)

    Args:
        B: (m, n) binary parity matrix.
        v: (m,) binary target parity vector.
        ell: Dicke weight bound.
        instance_id: Optional identifier for the instance.
        k_selected: Number of surrogate terms (from artifact if available).
        artifact_version: Version string from artifact.
        repo_commit: Commit hash from artifact.

    Returns:
        Resource estimate dictionary with full schema.

    Raises:
        ValueError: If v is not binary or shapes are incompatible.
    """
    B = np.asarray(B)
    v = np.asarray(v)

    _validate_shapes(B, v)
    _validate_binary(v, "v")

    m, n = B.shape

    # Dicke superposition estimate
    dicke = dicke_superposition_upper_bound(m, ell)

    # Syndrome: one CNOT per nonzero entry in B
    syndrome_cnot_count = int(np.sum(B))

    # Phase kick: one Z per nonzero entry in v
    phase_z_count = int(np.sum(v))

    # Final Hadamard: one H per problem qubit
    final_h_count = n

    # Aggregates (excludes decode/uncompute)
    total_gate_est = (
        dicke["cnot_est"] +
        dicke["single_qubit_rot_est"] +
        syndrome_cnot_count +
        phase_z_count +
        final_h_count
    )

    # Depth estimate: Dicke + syndrome (1 layer) + phase (1 layer) + Hadamard (1 layer)
    total_depth_est = dicke["depth_est"] + 3

    # Qubit breakdown
    n_problem_qubits = n
    m_error_qubits = m
    n_syndrome_qubits = n  # Output of B^T @ y
    ancilla_qubits_est = 0  # SCS construction is ancilla-free
    total_qubits_est = m_error_qubits + n_syndrome_qubits + ancilla_qubits_est

    return {
        # Instance metadata
        "instance_id": instance_id,
        "n": n,
        "m": m,
        "k_selected": k_selected,
        "ell": ell,

        # Dicke prep estimates
        "dicke_method": dicke["method"],
        "dicke_cnot_est": dicke["cnot_est"],
        "dicke_single_qubit_rot_est": dicke["single_qubit_rot_est"],
        "dicke_depth_est": dicke["depth_est"],

        # Other stages
        "syndrome_cnot_count": syndrome_cnot_count,
        "phase_z_count": phase_z_count,
        "final_h_count": final_h_count,

        # Aggregates
        "total_gate_est": total_gate_est,
        "total_depth_est": total_depth_est,

        # Qubit breakdown
        "n_problem_qubits": n_problem_qubits,
        "m_error_qubits": m_error_qubits,
        "n_syndrome_qubits": n_syndrome_qubits,
        "ancilla_qubits_est": ancilla_qubits_est,
        "total_qubits_est": total_qubits_est,

        # Register assumptions
        "register_assumptions": {
            "m_error_qubits": "error register holding weight-t error patterns",
            "n_syndrome_qubits": "syndrome register, output of B^T @ y mod 2",
            "ancilla_qubits_est": "additional ancilla for Dicke prep if needed (0 for SCS)",
        },

        # Stage tracking
        "included_stages": ["dicke_prep", "phase_kick", "syndrome_compute", "final_hadamard"],
        "excluded_stages": ["decode_uncompute"],
        "decode_uncompute_status": "not_implemented",

        # Provenance
        "artifact_version": artifact_version,
        "repo_commit": repo_commit,
        "assumptions": [
            "Dicke prep: deterministic SCS construction (Bärtschi-Eidenbenz 2019)",
            "Superposition: naive sum over t=0..ell, not optimized weighted prep",
            "Syndrome: one CNOT per nonzero entry in B",
            "Phase kick: one Z per nonzero entry in v",
            "Decode/uncompute: excluded from totals",
        ],
        "reference": "arXiv:1904.07358",
    }
```

- [ ] **Step 6.4: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_dqi_resource_estimate_schema -v`
Expected: PASS

---

### Task 7: Gate count correctness

**Files:**
- Modify: `tests/test_resources.py`

- [ ] **Step 7.1: Write the test for gate counts**

```python
# Add to tests/test_resources.py

def test_dqi_resource_estimate_gate_counts():
    """Gate counts are computed correctly from B and v."""
    # B with 5 ones, v with 2 ones
    B = np.array([
        [1, 1, 0],  # 2 ones
        [0, 1, 1],  # 2 ones
        [1, 0, 0],  # 1 one
    ], dtype=np.int8)
    v = np.array([1, 0, 1], dtype=np.int8)
    n = 3

    result = dqi_resource_estimate(B, v, ell=2)

    assert result["syndrome_cnot_count"] == 5  # sum(B) = 5
    assert result["phase_z_count"] == 2  # sum(v) = 2
    assert result["final_h_count"] == n  # = 3
```

- [ ] **Step 7.2: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_dqi_resource_estimate_gate_counts -v`
Expected: PASS

---

### Task 8: Qubit count correctness

**Files:**
- Modify: `tests/test_resources.py`

- [ ] **Step 8.1: Write the test for qubit counts**

```python
# Add to tests/test_resources.py

def test_dqi_resource_estimate_qubit_counts():
    """Qubit counts are computed correctly."""
    B = np.array([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=np.int8)  # m=2, n=4
    v = np.array([0, 1], dtype=np.int8)

    result = dqi_resource_estimate(B, v, ell=2)

    assert result["n_problem_qubits"] == 4
    assert result["m_error_qubits"] == 2
    assert result["n_syndrome_qubits"] == 4
    assert result["ancilla_qubits_est"] == 0  # SCS is ancilla-free
    assert result["total_qubits_est"] == 2 + 4 + 0  # m + n + ancilla
```

- [ ] **Step 8.2: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_dqi_resource_estimate_qubit_counts -v`
Expected: PASS

---

### Task 9: Stage tracking test

**Files:**
- Modify: `tests/test_resources.py`

- [ ] **Step 9.1: Write the test for included/excluded stages**

```python
# Add to tests/test_resources.py

def test_dqi_resource_estimate_included_excluded_stages():
    """Correct stages are listed as included/excluded."""
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)
    v = np.array([0, 0], dtype=np.int8)

    result = dqi_resource_estimate(B, v, ell=1)

    assert "dicke_prep" in result["included_stages"]
    assert "phase_kick" in result["included_stages"]
    assert "syndrome_compute" in result["included_stages"]
    assert "final_hadamard" in result["included_stages"]

    assert "decode_uncompute" in result["excluded_stages"]
    assert result["decode_uncompute_status"] == "not_implemented"
```

- [ ] **Step 9.2: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_dqi_resource_estimate_included_excluded_stages -v`
Expected: PASS

---

### Task 10: Non-binary v rejection

**Files:**
- Modify: `tests/test_resources.py`

- [ ] **Step 10.1: Write the test for non-binary rejection**

```python
# Add to tests/test_resources.py

def test_dqi_resource_estimate_rejects_nonbinary_v():
    """ValueError raised for non-binary v."""
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)
    v_bad = np.array([0, 2], dtype=np.int8)  # 2 is not binary

    with pytest.raises(ValueError, match="binary"):
        dqi_resource_estimate(B, v_bad, ell=1)
```

- [ ] **Step 10.2: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_dqi_resource_estimate_rejects_nonbinary_v -v`
Expected: PASS

---

### Task 11: Shape mismatch rejection

**Files:**
- Modify: `tests/test_resources.py`

- [ ] **Step 11.1: Write the test for shape mismatch**

```python
# Add to tests/test_resources.py

def test_dqi_resource_estimate_rejects_shape_mismatch():
    """ValueError raised for B/v dimension mismatch."""
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)  # m=2
    v_wrong = np.array([0, 1, 1], dtype=np.int8)  # length 3, should be 2

    with pytest.raises(ValueError, match="rows"):
        dqi_resource_estimate(B, v_wrong, ell=1)
```

- [ ] **Step 11.2: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_dqi_resource_estimate_rejects_shape_mismatch -v`
Expected: PASS

---

### Task 12: Metadata presence test

**Files:**
- Modify: `tests/test_resources.py`

- [ ] **Step 12.1: Write the test for metadata fields**

```python
# Add to tests/test_resources.py

def test_resource_report_metadata_presence():
    """Required metadata fields are present."""
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)
    v = np.array([0, 1], dtype=np.int8)

    result = dqi_resource_estimate(B, v, ell=1)

    assert "included_stages" in result
    assert "excluded_stages" in result
    assert "assumptions" in result
    assert "reference" in result
    assert isinstance(result["assumptions"], list)
    assert len(result["assumptions"]) > 0
```

- [ ] **Step 12.2: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_resource_report_metadata_presence -v`
Expected: PASS

- [ ] **Step 12.3: Run all tests so far**

Run: `pytest tests/test_resources.py -v`
Expected: All 12 tests PASS

---

## Chunk 3: Artifact Loading and Report Generation

### Task 13: Resource estimate from artifact

**Files:**
- Modify: `tests/test_resources.py`
- Modify: `src/resources.py`

- [ ] **Step 13.1: Write the failing test for artifact loading**

```python
# Add to tests/test_resources.py
from src.resources import dqi_resource_estimate, resource_estimate_from_artifact


def test_resource_estimate_from_artifact():
    """resource_estimate_from_artifact extracts B, v and computes resources."""
    artifact = {
        "artifact_version": "1.0",
        "n": 4,
        "m": 2,
        "k_selected": 2,
        "B": [[1, 0, 1, 0], [0, 1, 0, 1]],
        "v": [0, 1],
        "source_metadata": {
            "instance_id": "test_artifact",
            "repo_commit": "abc123",
        },
    }

    result = resource_estimate_from_artifact(artifact, ell=2)

    assert result["instance_id"] == "test_artifact"
    assert result["n"] == 4
    assert result["m"] == 2
    assert result["k_selected"] == 2
    assert result["artifact_version"] == "1.0"
    assert result["repo_commit"] == "abc123"
```

- [ ] **Step 13.2: Run test to verify it fails**

Run: `pytest tests/test_resources.py::test_resource_estimate_from_artifact -v`
Expected: FAIL with `ImportError`

- [ ] **Step 13.3: Write implementation**

```python
# Add to src/resources.py

def resource_estimate_from_artifact(artifact: dict, ell: int) -> dict[str, Any]:
    """Compute DQI resource estimate from a loaded surrogate artifact.

    Extracts B, v, and metadata from artifact and delegates to dqi_resource_estimate.

    Args:
        artifact: Surrogate artifact dictionary (as loaded from JSON or build_surrogate_artifact).
        ell: Dicke weight bound.

    Returns:
        Resource estimate dictionary.

    Raises:
        ValueError: If artifact is missing required fields.
    """
    # Extract required fields
    if "B" not in artifact:
        raise ValueError("Artifact missing required field 'B'")
    if "v" not in artifact:
        raise ValueError("Artifact missing required field 'v'")

    B = np.asarray(artifact["B"], dtype=np.int8)
    v = np.asarray(artifact["v"], dtype=np.int8)

    # Extract optional metadata with fallbacks
    source_meta = artifact.get("source_metadata", {})
    instance_id = source_meta.get("instance_id", artifact.get("instance_id"))
    repo_commit = source_meta.get("repo_commit", artifact.get("repo_commit"))
    artifact_version = artifact.get("artifact_version")
    k_selected = artifact.get("k_selected")

    return dqi_resource_estimate(
        B=B,
        v=v,
        ell=ell,
        instance_id=instance_id,
        k_selected=k_selected,
        artifact_version=artifact_version,
        repo_commit=repo_commit,
    )
```

- [ ] **Step 13.4: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_resource_estimate_from_artifact -v`
Expected: PASS

---

### Task 14: Resource report from file path

**Files:**
- Modify: `tests/test_resources.py`
- Modify: `src/resources.py`

- [ ] **Step 14.1: Write the failing test for file loading**

```python
# Add to tests/test_resources.py
from src.resources import resource_report


def test_resource_report_round_trip_with_artifact(tmp_path):
    """Load artifact from file, compute resources, verify schema."""
    # Create a test artifact file
    artifact = {
        "artifact_version": "1.0",
        "n": 4,
        "m": 2,
        "k_selected": 2,
        "B": [[1, 0, 1, 0], [0, 1, 0, 1]],
        "v": [0, 1],
        "source_metadata": {"instance_id": "roundtrip_test"},
    }

    artifact_path = tmp_path / "test_artifact.json"
    with open(artifact_path, "w") as f:
        json.dump(artifact, f)

    result = resource_report(str(artifact_path), ell=2)

    assert result["instance_id"] == "roundtrip_test"
    assert result["n"] == 4
    assert result["m"] == 2
    assert "total_gate_est" in result
    assert "total_qubits_est" in result
```

- [ ] **Step 14.2: Run test to verify it fails**

Run: `pytest tests/test_resources.py::test_resource_report_round_trip_with_artifact -v`
Expected: FAIL with `ImportError`

- [ ] **Step 14.3: Write implementation**

```python
# Add to src/resources.py

def resource_report(artifact_path: str, ell: int) -> dict[str, Any]:
    """Load surrogate artifact from path, compute resources, return full report.

    Args:
        artifact_path: Path to surrogate artifact JSON file.
        ell: Dicke weight bound.

    Returns:
        Resource estimate dictionary.

    Raises:
        FileNotFoundError: If artifact file does not exist.
        ValueError: If artifact is malformed.
    """
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    with open(path) as f:
        artifact = json.load(f)

    return resource_estimate_from_artifact(artifact, ell)
```

- [ ] **Step 14.4: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_resource_report_round_trip_with_artifact -v`
Expected: PASS

---

### Task 15: Export resources to JSON

**Files:**
- Modify: `src/resources.py`

- [ ] **Step 15.1: Add export function**

```python
# Add to src/resources.py

def export_resources_json(
    reports: list[dict],
    output_path: str,
) -> None:
    """Export resource reports to aggregated JSON file.

    Args:
        reports: List of resource estimate dictionaries.
        output_path: Path to output JSON file.
    """
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reports": reports,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(output, f, indent=2)
```

- [ ] **Step 15.2: Run all tests**

Run: `pytest tests/test_resources.py -v`
Expected: All 14 tests PASS

---

## Chunk 4: Notebook Implementation

### Task 16: Create notebook structure

**Files:**
- Create: `notebooks/06_resource_analysis.ipynb`

- [ ] **Step 16.1: Create notebook with introduction**

Create `notebooks/06_resource_analysis.ipynb` with cells:

**Cell 1 (Markdown):**
```markdown
# Notebook 06: Resource Analysis

This notebook provides gate-level resource estimation for the DQI pipeline using analytical formulas based on the Bärtschi-Eidenbenz (2019) deterministic Dicke state construction.

## What's Included vs Excluded

**Included stages (with gate counts):**
- Dicke superposition preparation (SCS construction)
- Phase kick (Z gates)
- Syndrome computation (CNOTs)
- Final Hadamard transform

**Excluded (not implemented as circuit):**
- Decode/uncompute stage

These are **analytical estimates**, not backend-transpiled exact counts.

**Reference:** Bärtschi & Eidenbenz, "Deterministic Preparation of Dicke States," [arXiv:1904.07358](https://arxiv.org/abs/1904.07358) (2019)
```

**Cell 2 (Code):**
```python
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.resources import resource_report, export_resources_json
from src.reduction import build_surrogate_artifact
from src.problem_generator import generate_pricing_problem
```

- [ ] **Step 16.2: Add instance loading cells**

**Cell 3 (Markdown):**
```markdown
## Load Frozen Instances

We analyze three frozen surrogate artifacts: 3-feature (6-bit), 4-feature (8-bit), and 5-feature (10-bit).
```

**Cell 4 (Code):**
```python
# Load existing surrogate artifact (10-bit)
artifact_10bit_path = "data/instances/surrogate_pricing_5feat_10bit_k15.json"

# Generate surrogate artifacts for 3-feat and 4-feat on the fly
# (matching the frozen pricing instances)
prob_3 = generate_pricing_problem(n_features=3)
prob_4 = generate_pricing_problem(n_features=4)
prob_5 = generate_pricing_problem(n_features=5)

artifact_3 = build_surrogate_artifact(prob_3, k=15)
artifact_4 = build_surrogate_artifact(prob_4, k=15)

# Load the 10-bit artifact from file
with open(artifact_10bit_path) as f:
    artifact_5 = json.load(f)

instances = [
    ("3-feat (6-bit)", artifact_3),
    ("4-feat (8-bit)", artifact_4),
    ("5-feat (10-bit)", artifact_5),
]

print("Loaded instances:")
for name, art in instances:
    n = art.get("n", len(art["B"][0]) if art["B"] else 0)
    m = art.get("m", len(art["B"]))
    print(f"  {name}: n={n}, m={m}")
```

- [ ] **Step 16.3: Add resource estimation cells**

**Cell 5 (Markdown):**
```markdown
## Resource Estimation

Compute resource estimates for each instance with fixed ℓ=3.
```

**Cell 6 (Code):**
```python
from src.resources import resource_estimate_from_artifact

ELL = 3
reports = []

for name, artifact in instances:
    report = resource_estimate_from_artifact(artifact, ell=ELL)
    report["display_name"] = name
    reports.append(report)
    print(f"\n{name}:")
    print(f"  Qubits: {report['total_qubits_est']} (m={report['m_error_qubits']}, n={report['n_syndrome_qubits']})")
    print(f"  Total gates: {report['total_gate_est']}")
    print(f"  Total depth: {report['total_depth_est']}")
```

- [ ] **Step 16.4: Add scaling table**

**Cell 7 (Markdown):**
```markdown
## Scaling Comparison Table

Side-by-side comparison across instance sizes. All counts are **estimates**; decode/uncompute is **excluded**.
```

**Cell 8 (Code):**
```python
import pandas as pd

table_data = []
for r in reports:
    table_data.append({
        "Instance": r["display_name"],
        "n (bits)": r["n"],
        "m (terms)": r["m"],
        "Qubits": r["total_qubits_est"],
        "Dicke CNOTs": r["dicke_cnot_est"],
        "Syndrome CNOTs": r["syndrome_cnot_count"],
        "Total Gates": r["total_gate_est"],
        "Depth Est.": r["total_depth_est"],
    })

df = pd.DataFrame(table_data)
df
```

- [ ] **Step 16.5: Add scaling plot**

**Cell 9 (Markdown):**
```markdown
## Scaling Plot

Total gates and depth vs problem size (n).
```

**Cell 10 (Code):**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

n_values = [r["n"] for r in reports]
gate_values = [r["total_gate_est"] for r in reports]
depth_values = [r["total_depth_est"] for r in reports]

ax1.plot(n_values, gate_values, 'o-', markersize=8)
ax1.set_xlabel("Problem size (n bits)")
ax1.set_ylabel("Total gate estimate")
ax1.set_title("Gate Count Scaling")
ax1.grid(True, alpha=0.3)

ax2.plot(n_values, depth_values, 's-', markersize=8, color='orange')
ax2.set_xlabel("Problem size (n bits)")
ax2.set_ylabel("Depth estimate")
ax2.set_title("Depth Scaling")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/resource_scaling.png", dpi=150, bbox_inches="tight")
plt.show()
```

- [ ] **Step 16.6: Add k-sensitivity mini-appendix**

**Cell 11 (Markdown):**
```markdown
## Mini-Appendix: k Sensitivity (10-bit Instance)

How do resources grow as we vary the number of surrogate terms (k)?

This is **reduction sensitivity** — how the choice of truncation affects the resulting circuit resources — not "circuit sensitivity" per se.
```

**Cell 12 (Code):**
```python
# Vary k from 5 to 15 on the 10-bit instance
k_values = [5, 7, 10, 12, 15]
k_reports = []

for k in k_values:
    artifact_k = build_surrogate_artifact(prob_5, k=k)
    report_k = resource_estimate_from_artifact(artifact_k, ell=ELL)
    report_k["k"] = k
    k_reports.append(report_k)

# Add to main reports for export
reports.extend(k_reports)

k_df = pd.DataFrame([{
    "k": r["k"],
    "m": r["m"],
    "Dicke CNOTs": r["dicke_cnot_est"],
    "Syndrome CNOTs": r["syndrome_cnot_count"],
    "Total Gates": r["total_gate_est"],
    "Qubits": r["total_qubits_est"],
} for r in k_reports])

k_df
```

**Cell 13 (Code):**
```python
fig, ax = plt.subplots(figsize=(6, 4))

k_vals = [r["k"] for r in k_reports]
gate_vals = [r["total_gate_est"] for r in k_reports]

ax.plot(k_vals, gate_vals, 'o-', markersize=8)
ax.set_xlabel("Number of surrogate terms (k)")
ax.set_ylabel("Total gate estimate")
ax.set_title("Resource Growth with Surrogate Size (10-bit)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

- [ ] **Step 16.7: Add export cell**

**Cell 14 (Markdown):**
```markdown
## Export Results

Save all resource reports to `results/resources_scaling.json`.
```

**Cell 15 (Code):**
```python
# Remove display_name before export (not part of schema)
export_reports = []
for r in reports:
    r_copy = dict(r)
    r_copy.pop("display_name", None)
    export_reports.append(r_copy)

export_resources_json(export_reports, "results/resources_scaling.json")
print("Exported to results/resources_scaling.json")
```

- [ ] **Step 16.8: Verify notebook runs**

Run: `jupyter nbconvert --to notebook --execute notebooks/06_resource_analysis.ipynb --output 06_resource_analysis_executed.ipynb`
Expected: Notebook executes without errors

---

### Task 17: Final verification

- [ ] **Step 17.1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS (including existing 84 + new 14 = 98 tests)

- [ ] **Step 17.2: Verify results file exists**

Run: `cat results/resources_scaling.json | head -50`
Expected: JSON with `generated_at` and `reports` array

- [ ] **Step 17.3: Verify no changes to existing modules**

Run: `git diff src/dqi_state.py src/dqi_pipeline.py`
Expected: No changes (or empty output if not a git repo)

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1-5 | `dicke_circuit.py` | 5 tests |
| 6-12 | `resources.py` core | 7 tests |
| 13-15 | Artifact loading + export | 2 tests |
| 16-17 | Notebook + verification | Manual |

**Total new tests:** 14
**New files:** 4 (`dicke_circuit.py`, `resources.py`, `test_resources.py`, notebook)
**Generated:** `results/resources_scaling.json`, `figures/resource_scaling.png`
