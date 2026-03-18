# Stage 3: Resource Analysis Design

**Date:** 2026-03-10
**Status:** Approved

## Goal

Add gate-level resource estimation to the DQI pipeline, enabling BMW-facing resource tables that speak the "gate-level language" without duplicating their circuit implementation work.

## Scope

This design covers:
- `src/dicke_circuit.py`: Analytical Dicke state preparation resource formulas
- `src/resources.py`: Full DQI resource estimation by stage composition
- `tests/test_resources.py`: Unit tests for formula correctness and schema conformance
- `notebooks/06_resource_analysis.ipynb`: Scaling comparison + k-sensitivity analysis
- `results/resources_scaling.json`: Exported resource data

Out of scope:
- Actual circuit construction (PennyLane/Qiskit)
- Decode/uncompute circuit implementation
- Full parameter sweeps (ell variation belongs in decoder notebook)

## Architecture

### Module: `src/dicke_circuit.py`

Provides analytical resource estimates for deterministic Dicke state preparation based on Bärtschi-Eidenbenz (2019) SCS construction.

```python
def dicke_resources(m: int, t: int) -> dict:
    """Analytic resource estimate for preparing |D_t^m⟩ via deterministic SCS construction.

    Returns:
        cnot_est: int
        single_qubit_rot_est: int
        depth_est: int
        method: str
        reference: str
    """

def dicke_superposition_upper_bound(m: int, ell: int) -> dict:
    """Naive upper-bound estimate for Σ_{t=0}^ell α_t |D_t^m⟩ by aggregating per-t Dicke costs.

    NOT a compiled circuit count for a specific weighted-superposition construction.

    Returns:
        cnot_est: int
        single_qubit_rot_est: int
        depth_est: int
        method: str
        per_weight_breakdown: list[dict]
    """
```

**Formula basis (Bärtschi-Eidenbenz SCS):**
- For |D_t^m⟩: CNOTs ≈ t(m-t), controlled-Ry ≈ (m-1)
- Depth ≈ O(m)
- Symmetry: dicke_resources(m, t) == dicke_resources(m, m-t) for CNOT count

### Module: `src/resources.py`

Orchestrates full DQI resource estimation by composing stage estimates.

```python
def resource_estimate_from_artifact(artifact: dict, ell: int) -> dict:
    """Compute DQI resource estimate from a loaded surrogate artifact.

    Composes: dicke_prep + phase_kick + syndrome_compute + final_hadamard
    Excludes: decode_uncompute (explicitly labeled)
    """

def dqi_resource_estimate(B: np.ndarray, v: np.ndarray, ell: int,
                          instance_id: str = None) -> dict:
    """Full DQI resource estimate for a (B, v) surrogate artifact.

    Validates v is binary (raises ValueError if not).
    """

def resource_report(artifact_path: str, ell: int) -> dict:
    """Load surrogate artifact from path, compute resources, return full report.

    Fallback: missing artifact_version/source_metadata marked as unavailable, not failure.
    """
```

### Output Schema

```python
{
    # Instance metadata
    "instance_id": str,
    "n": int,                              # problem bits
    "m": int,                              # parity terms (surrogate size)
    "k_selected": int,                     # from artifact if available
    "ell": int,                            # Dicke weight bound

    # Dicke prep estimates
    "dicke_method": "bartschi_eidenbenz_scs_upper_bound",
    "dicke_cnot_est": int,
    "dicke_single_qubit_rot_est": int,
    "dicke_depth_est": int,

    # Other stages
    "syndrome_cnot_count": int,            # = nnz(B)
    "phase_z_count": int,                  # = sum(v), requires binary v
    "final_h_count": int,                  # = n

    # Aggregates
    "total_gate_est": int,                 # sum of counted gate categories; excludes decode/uncompute
    "total_depth_est": int,

    # Qubit breakdown
    "n_problem_qubits": int,               # = n (syndrome register output)
    "m_error_qubits": int,                 # = m (error register for Dicke state)
    "n_syndrome_qubits": int,              # = n (syndrome register)
    "ancilla_qubits_est": int,             # Dicke prep ancilla (0 for SCS)
    "total_qubits_est": int,

    # Register assumptions
    "register_assumptions": {
        "m_error_qubits": "error register holding weight-t error patterns",
        "n_syndrome_qubits": "syndrome register, output of B^T @ y mod 2",
        "ancilla_qubits_est": "additional ancilla for Dicke prep if needed"
    },

    # Stage tracking
    "included_stages": ["dicke_prep", "phase_kick", "syndrome_compute", "final_hadamard"],
    "excluded_stages": ["decode_uncompute"],
    "decode_uncompute_status": "not_implemented",

    # Provenance
    "artifact_version": str | null,        # from artifact if available
    "repo_commit": str | null,             # from artifact if available
    "assumptions": [
        "Dicke prep: deterministic SCS construction (Bärtschi-Eidenbenz 2019)",
        "Superposition: naive sum over t=0..ell, not optimized weighted prep",
        "Syndrome: one CNOT per nonzero entry in B",
        "Phase kick: one Z per nonzero entry in v",
        "Decode/uncompute: excluded from totals"
    ],
    "reference": "arXiv:1904.07358"
}
```

### Results Export: `results/resources_scaling.json`

One aggregated file containing all resource reports. Structure:

```python
{
    "generated_at": "ISO timestamp",
    "reports": [
        { ... resource report for instance 1 ... },
        { ... resource report for instance 2 ... },
        { ... resource report for instance 3 ... },
        { ... k-sensitivity reports for 10-bit instance ... }
    ]
}
```

Reports keyed by: (instance_id, n, m, k_selected, ell)

## Notebook Structure: `06_resource_analysis.ipynb`

**Narrative flow (~12-15 cells):**

1. **Introduction** — Purpose, included/excluded stages, reference to Bärtschi-Eidenbenz
2. **Load frozen instances** — 3/4/5-feature surrogate artifacts
3. **Resource estimation** — Call `dqi_resource_estimate()` for each
4. **Scaling comparison table** — Side-by-side: 6-bit vs 8-bit vs 10-bit
5. **Scaling plot** — Total gates and depth vs n (one clean figure)
6. **Mini-appendix: k sensitivity** — 10-bit instance, vary k from 5 to 15, fixed ell=3
   - Framing: "reduction sensitivity" (surrogate structure), not "circuit sensitivity"
7. **Export results** — Save to `results/resources_scaling.json`

## Testing: `tests/test_resources.py`

**Test functions:**

- `test_dicke_resources_trivial_case` — t=0 returns zero CNOTs
- `test_dicke_resources_symmetry_t_vs_m_minus_t` — t=3 vs t=7 for m=10
- `test_dicke_resources_scaling_pattern` — t(m-t) pattern holds
- `test_dicke_superposition_upper_bound_keys` — Returns expected keys
- `test_dicke_superposition_upper_bound_aggregation` — Exact sum of per-t costs (==)
- `test_dqi_resource_estimate_schema` — All required fields present
- `test_dqi_resource_estimate_gate_counts` — syndrome=nnz(B), phase=sum(v), hadamard=n
- `test_dqi_resource_estimate_qubit_counts` — n_problem=n, m_error=m, total reasonable
- `test_dqi_resource_estimate_included_excluded_stages` — Correct stage lists
- `test_dqi_resource_estimate_rejects_nonbinary_v` — ValueError for non-binary v
- `test_dqi_resource_estimate_rejects_shape_mismatch` — ValueError for B/v dimension mismatch
- `test_resource_report_round_trip_with_artifact` — Load artifact, compute, verify schema
- `test_resource_report_metadata_presence` — included_stages, excluded_stages, assumptions, reference present

## Integration Points

- **Surrogate artifacts** — `resource_report()` loads via existing artifact loader
- **Frozen instances** — Uses `data/instances/surrogate_*.json`
- **Results export** — Follows `results/benchmark_*.json` pattern
- **No changes to existing modules** — `dqi_state.py`, `dqi_pipeline.py` untouched

## Risks and Controls

**Primary risk:** Dicke formula inaccuracy vs published results.
**Control:** Reference Bärtschi-Eidenbenz explicitly, label as "estimate", test symmetry properties.

**Secondary risk:** Overclaiming resource completeness.
**Control:** Explicit `excluded_stages`, `decode_uncompute_status`, and `assumptions` fields.

## Acceptance Criteria

- [ ] `dicke_circuit.py` implements SCS-based analytical formulas with correct symmetry
- [ ] `resources.py` composes stage estimates with explicit included/excluded tracking
- [ ] Non-binary v raises ValueError
- [ ] Schema includes all required fields including register_assumptions
- [ ] Notebook 06 shows scaling table + plot + k-sensitivity mini-appendix
- [ ] Results exported to `results/resources_scaling.json` in aggregated format
- [ ] All tests pass
- [ ] No changes to existing DQI core modules
