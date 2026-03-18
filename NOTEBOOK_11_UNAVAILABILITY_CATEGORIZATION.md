# Notebook 11: Unavailability Categorization and Filtering

## Summary

Improved Notebook 11's unavailability reporting by:
1. **Filtering planned placeholder rows** - removed 50 unexecuted configurations from the skipped cases view
2. **Categorizing unavailability types** - distinguished between expected unavailability, structural boundaries, and real failures

## Motivation

Before this fix, Notebook 11 showed **135 unavailable cases**, making the dashboard appear problematic. Investigation revealed:
- 50 were "planned::" placeholder rows (configurations designed but never executed, with different trial_seeds)
- 45 were missing `retained_energy_eta` (expected - not computed for matrix_c)
- 22 were bp+osd-lite decoder (not implemented in codebase)
- 12 were resource comparison issues
- Only 4 were actual execution failures

Without categorization, all of these appeared as generic "unavailable" cases, obscuring the actual health of the experiments.

## Implementation

### 1. Filter Planned Placeholders

**File**: `notebooks/11_decision_faithfulness_dashboard.ipynb`, cell `9fa0cf2b`

Added filtering at the beginning of the "Unavailable / Skipped Cases" section:

```python
# Filter out planned placeholder rows (unexecuted configurations with different seeds)
# These inflate unavailable counts without adding interpretive value
executed_df = all_df[~all_df["run_id"].str.startswith("planned::", na=False)].copy()
```

**Why planned rows exist**: The experiment matrix was designed with per-decoder seed variations:
- bruteforce: seed 42 only
- bp1: seeds 42, 43
- bp+osd-lite: seeds 42, 44
- oracle: seeds 42, 45

Only seed=42 configurations were fully executed. The non-42 seed slots remain as planned placeholders with `run_id` starting with `planned::`.

### 2. Categorize Unavailability Types

Added categorization logic for two data sources:

#### A. Metrics Master (Skipped Cases)

Categories:
- **structural_boundary**: Not applicable by design
  - `dqi_impossible_zero_decodable_syndromes`
  - `oracle_tiny_cap_exceeded`
- **decoder_not_implemented**: Decoder in matrix but not in codebase
  - Detected via `run_error: "decoder_not_implemented_in_repo"`
- **dependency_issue**: Upstream data unavailable or failed
- **execution_failure**: Actual runtime failure
- **expected_unavailable**: Other expected unavailability

#### B. Quality/Cost Unavailable

Categories:
- **structural_boundary**: Not applicable by design (50 cases)
  - Includes our S_sparse_high_collision fix
- **expected_missing_metric**: Metric not computed for this configuration (45 cases)
  - Primarily `retained_energy_eta` missing
- **dependency_issue**: Upstream unavailable or failed (26 cases)
- **resource_comparison_issue**: Resource metadata missing (12 cases)
- **decoder_not_implemented**: Decoder not in codebase
- **execution_failure**: Actual runtime failure
- **other**: Uncategorized

### 3. Added Category Definitions

Inline documentation in the notebook explains each category, helping reviewers distinguish between:
- Expected unavailability (by design)
- Implementation gaps (decoder_not_implemented)
- Actual problems (execution_failure, dependency_issue)

## Impact

### Before Fix

**Skipped Cases**: 78 rows (including 50 planned placeholders)
- Appeared as large block of unavailable configurations
- No clear distinction between expected and unexpected unavailability

**Stage E Unavailable**: 135 cases
- All lumped together as "missing_required_fields" or "missing_quality_fields"
- Looked like widespread experiment problems

### After Fix

**Skipped Cases**: 28 rows (planned placeholders filtered out)
- `decoder_not_implemented`: 22 (bp+osd-lite - known gap)
- `execution_failure`: 4 (actual failures to investigate)
- `structural_boundary`: 2 (correct not_applicable classification)

**Stage E Unavailable**: 135 cases (categorized)
- `structural_boundary`: 50 (expected by design)
- `expected_missing_metric`: 45 (retained_energy_eta not computed)
- `dependency_issue`: 26 (upstream issues)
- `resource_comparison_issue`: 12 (metadata gaps)
- `other`: 2 (uncategorized)

**Key improvement**: Only **4 execution failures** and **26 dependency issues** require investigation, vs appearing like 135 problems.

## Verification

Run Notebook 11 and verify:
1. "Unavailable/Skipped Cases by Category (Planned placeholders filtered out)" table shows ~28 rows
2. Category breakdown shows clear distinction between decoder_not_implemented, execution_failure, and structural_boundary
3. Stage E summary shows categorized breakdown with definitions
4. No `planned::` run_ids appear in the skipped cases table

```bash
cd notebooks && jupyter nbconvert --to notebook --execute 11_decision_faithfulness_dashboard.ipynb --output 11_decision_faithfulness_dashboard.ipynb
```

## Related Documentation

- `STRUCTURAL_BOUNDARY_HANDLING.md` - Explains the not_applicable status for zero decodable syndromes
- `DQI_Pricing_Benchmark_Results_and_Validation_Report.md` - Overall experiment report

## Dependency Issues Breakdown (26 cases)

After categorization, the 26 dependency_issue cases consist of:

### 1. bp+osd-lite Decoder Not Implemented (22 cases)
**Status**: Expected unavailability
**Reason**: bp+osd-lite decoder is defined in experiment matrix but not implemented in codebase
**Run error**: `decoder_not_implemented_in_repo`
**Action**: None required unless decoder gets implemented

### 2. OLD benchmark_matrix Runs Needing Regeneration (4 cases)
**Affected**: S_sparse_high_collision with bruteforce and oracle (paper + uniform alpha modes)
**Status**: `upstream_failed` with error "zero surviving amplitude"
**Root cause**: These are OLD benchmark_matrix stage runs created before the structural boundary fix
**Solution**: The corresponding decoder_study runs (rows 138-140) already have correct `not_applicable` status

**Details**:
- Row 24: benchmark_matrix + bruteforce + paper → **failed** (OLD)
- Row 28: benchmark_matrix + bruteforce + uniform → **failed** (OLD)
- Row 95: benchmark_matrix + oracle + paper → **failed** (OLD)
- Row 111: benchmark_matrix + oracle + uniform → **failed** (OLD)

vs NEW decoder_study runs:
- Row 139: decoder_study + bruteforce + paper → **not_applicable** (dqi_impossible_zero_decodable_syndromes) ✓
- Row 140: decoder_study + oracle + paper → **not_applicable** (oracle_tiny_cap_exceeded) ✓

**Options**:
1. Regenerate benchmark_matrix runs for S_sparse_high_collision (deletes OLD files, creates new ones with correct status)
2. Accept that benchmark_matrix and decoder_study stages coexist with different statuses
3. Filter out benchmark_matrix stage in reporting (since decoder_study is more recent/authoritative)

**Recommendation**: Option 3 (filter in reporting) is simplest for dashboard purposes. The decoder_study runs are the authoritative recent results.

## Future Considerations

If bp+osd-lite decoder gets implemented:
- The 22 `decoder_not_implemented` cases would become executable
- Expect them to move to either `completed` or other unavailability categories based on actual execution results

If benchmark_matrix runs get regenerated:
- The 4 `upstream_failed` cases would become `not_applicable` with proper structural boundary reason codes
- Aligns benchmark_matrix and decoder_study stages for S_sparse_high_collision
