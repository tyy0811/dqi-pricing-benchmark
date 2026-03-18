# Structural Boundary Handling: Zero Decodable Syndromes

## Summary

Implemented graceful handling for instances with zero decodable syndromes at a given `ell` value. Instead of failing with a runtime error, the system now:

1. **Pre-checks** `decoder_success_rate` before attempting DQI execution
2. **Skips execution** when `decoder_success_rate == 0.0`
3. **Marks status** as `not_applicable` (not `failed`)
4. **Documents reason** with clear status code and human-readable message

## Motivation

The instance `S_sparse_high_collision` is intentionally pathological for testing:
- `decodable_syndrome_fraction`: 0.0 (no decodable syndromes)
- `decoder_collision_rate`: 1.0 (100% ambiguity)
- `rank_deficiency`: 5 out of 12 rows
- Designed to test decoder behavior under maximum structural ambiguity

When DQI sampling attempted execution, it would:
1. Build quantum state → compute syndromes → attempt decode
2. **ALL amplitudes collapse to zero** (no valid decodable states)
3. Post-selection finds zero probability mass → raises RuntimeError

This is **NOT a bug** - it's a **structural boundary**. The fix makes this explicit.

## Implementation

### Files Modified

1. **`src/benchmark.py`** (line ~1057)
   - Added pre-check after `decoder_diagnostics()`
   - Skips `run_dqi_statevector()` if `decoder_success_rate == 0.0`
   - Creates `not_applicable` status with reason code

2. **`src/decoder_study_runner.py`** (line ~357)
   - Same pre-check pattern for decoder study runs
   - Ensures consistency across all DQI execution paths

3. **`src/matrix_runner.py`** (line ~1438)
   - Propagates `status_reason_code` and `status_reason` fields
   - Maintains provenance for downstream reporting

### Status Semantics

**Before fix:**
```json
{
  "run_status": "failed",
  "run_error": "DQI sampling failed: zero surviving amplitude after decode/uncompute (zero postselection mass)."
}
```

**After fix:**
```json
{
  "run_status": "not_applicable",
  "status_reason_code": "dqi_impossible_zero_decodable_syndromes",
  "status_reason": "DQI execution skipped: instance has zero decodable syndromes at ell=2. This is a structural property (decoder_success_rate=0.0), not a decoder failure.",
  "run_error": null
}
```

## Decoder-Specific Behavior

Different decoders may have different success rates on the same instance:

| Instance | Decoder | Success Rate | Behavior |
|----------|---------|--------------|----------|
| S_sparse_high_collision | bruteforce | 0.0 | ✓ Skipped (not_applicable) |
| S_sparse_high_collision | bp1 | 0.00977 (~1%) | ✓ Attempted (may succeed with low probability) |
| S_sparse_high_collision | oracle | 0.0 | ✓ Skipped (not_applicable) |

This is correct behavior:
- **Bruteforce** requires unique syndrome decoding → impossible when success rate = 0
- **BP1** uses belief propagation → can attempt approximate guessing even with ambiguity
- **Oracle** requires exact decoding → impossible when success rate = 0

## Research-Facing Benefits

1. **Clear status taxonomy**: `not_applicable` vs `failed` vs `completed`
2. **Traceable reason codes**: `dqi_impossible_zero_decodable_syndromes`
3. **Human-readable explanations**: Full structural context in status message
4. **Preserves metrics**: Structure metrics and diagnostics still recorded
5. **No false negatives**: Doesn't mark legitimate structural boundaries as failures

## Verification

```python
from src.benchmark import run_structure_sweep_benchmark

results = run_structure_sweep_benchmark(
    instance_ids=["S_sparse_high_collision"],
    decoders=("bruteforce",),
    alpha_mode="paper",
    n_dqi_samples=32,
)

result = results[0]['dqi_by_decoder']['bruteforce']
assert result['status'] == 'not_applicable'
assert result['status_reason_code'] == 'dqi_impossible_zero_decodable_syndromes'
assert result.get('error') is None
```

## Impact on Existing Results

After rerunning with this fix:
- Previous `failed` status for S_sparse_high_collision will become `not_applicable`
- Notebooks will correctly classify as structural boundary, not decoder failure
- Coverage reports will show proper `not_applicable` count instead of `failed`

## Future Considerations

Could extend to other structural impossibility checks:
- `ell > m` (Dicke weight exceeds number of terms)
- `rank_deficiency == m` (zero-rank matrix)
- Negative postselection success predictions

However, the current implementation focuses on the most common/important case: zero decodable syndromes.
