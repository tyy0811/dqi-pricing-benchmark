# Notebook Data Quality Fixes Design

**Date:** 2026-03-17
**Scope:** Notebooks 07-13 data quality issues
**Approach:** Data-First (Option B + Option C)

## Problem Summary

Notebooks 07-13 have data quality issues including missing metrics, thin datasets, and unavailable panels:

| Notebook | Issue |
|----------|-------|
| 08 | ilp_derived missing decode_success, postselection_success, runtime_sec |
| 09 (3-period) | smoothing_penalty_total shows NA |
| 09 (decoder) | Only 1 completed row, correlations unavailable |
| 11 | Shared metric panel empty (no ilp_derived_exact rows) |
| 12 | No ilp_derived_exact in quality_cost_master |
| 13 | P7 unavailable, P5 missing from decoder comparison |

## Solution Design

### Phase 1: Generate ilp_derived_exact Data

Create `scripts/run_ilp_derived_pricing.py` that:
1. Loads P3-P6 pricing instances
2. Generates ilp_derived_exact encoding via existing `build_ilp_exact_reference_artifact()`
3. Runs DQI statevector with bp1/bruteforce decoders
4. Saves results with encoding="ilp_derived_exact"

### Phase 2: Run Decoder Study

Execute existing decoder study for P3-P6:
```bash
python scripts/run_decoder_study.py --p-instances P3,P4,P5,P6
```

### Phase 3: Regenerate Reports

```bash
python scripts/make_report.py
```

### Phase 4: Notebook Code Fixes

1. **Notebook 09 (3-period):** Remove or compute smoothing_penalty_total
2. **Notebook 13:** Add P7 scope note explaining statevector memory limits

### Phase 5: Validation

Re-execute all notebooks and verify:
- Notebook 11 shared metric panel populated
- Notebook 12 has ilp_derived_exact rows
- Notebook 13 decoder scaling includes P3-P6

## Scope Boundaries

- **P7:** Analytic-only (exceeds 16GB statevector budget)
- **Family S:** Uses synthetic_structured_bv, not ilp_derived_exact
- **Oracle decoder:** May hit oracle_tiny_cap_exceeded on some instances

## Expected Outputs

- New `ilp_derived_exact` rows in metrics_master.json
- Updated quality_cost_master.json with gold-standard encoding data
- All 8 notebooks render without data quality warnings
