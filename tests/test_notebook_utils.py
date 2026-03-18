"""Unit tests for src/notebook_utils.py"""
from __future__ import annotations
import pytest
import pandas as pd
import numpy as np

from src.notebook_utils import (
    EXACT_REFERENCE_ENCODINGS, WHT_EXACT_ENCODINGS, WHT_TRUNCATED_ENCODINGS,
    COMPARABLE_ENCODINGS, DEFAULT_DECODERS, DEFAULT_EXECUTION_MODE,
    pick_column, normalize_column, FilterPipeline,
    build_metric_panel, compute_paired_delta,
)


@pytest.fixture
def sample_metrics_df():
    return pd.DataFrame({
        "run_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
        "encoding": ["ilp_derived", "ilp_derived", "wht_truncated", "wht_truncated", "synthetic", "ilp_derived"],
        "execution_mode": ["mixture", "coherent", "mixture", "mixture", "mixture", "mixture"],
        "decoder": ["bp1", "bp1", "bp1", "bruteforce", "bp1", "oracle"],
        "run_status": ["completed", "completed", "completed", "completed", "failed", "completed"],
        "best_sampled_F": [3000.0, 3000.0, 2800.0, 2700.0, np.nan, 3000.0],
        "top1_regret": [0.0, 0.0, 200.0, 300.0, np.nan, 0.0],
    })


class TestConstants:
    def test_encoding_constants_frozen(self):
        with pytest.raises(AttributeError):
            EXACT_REFERENCE_ENCODINGS.add("new")
        with pytest.raises(AttributeError):
            WHT_EXACT_ENCODINGS.add("new")
        with pytest.raises(AttributeError):
            WHT_TRUNCATED_ENCODINGS.add("new")
        with pytest.raises(AttributeError):
            COMPARABLE_ENCODINGS.add("new")
        with pytest.raises(AttributeError):
            DEFAULT_DECODERS.add("new")

    def test_comparable_is_union(self):
        assert COMPARABLE_ENCODINGS == (EXACT_REFERENCE_ENCODINGS | WHT_EXACT_ENCODINGS | WHT_TRUNCATED_ENCODINGS)

    def test_default_execution_mode(self):
        assert DEFAULT_EXECUTION_MODE == "mixture"

    def test_exact_reference_contains_ilp(self):
        assert "ilp_derived" in EXACT_REFERENCE_ENCODINGS
        assert "ilp_derived_exact" in EXACT_REFERENCE_ENCODINGS
        assert "ilp_derived_exact_where_feasible" in EXACT_REFERENCE_ENCODINGS


class TestPickColumn:
    def test_pick_column_first_match(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        assert pick_column(df, ["x", "b", "c"]) == "b"

    def test_pick_column_none_exist(self):
        df = pd.DataFrame({"a": [1]})
        assert pick_column(df, ["x", "y", "z"]) is None

    def test_pick_column_empty_candidates(self):
        df = pd.DataFrame({"a": [1]})
        assert pick_column(df, []) is None


class TestNormalizeColumn:
    def test_normalize_with_mapping(self):
        df = pd.DataFrame({"status": ["ok", "error", "unknown"]})
        result = normalize_column(df, "status", {"ok": "completed", "error": "failed"})
        assert list(result) == ["completed", "failed", "unknown"]

    def test_normalize_preserves_unmapped(self):
        df = pd.DataFrame({"x": ["a", "b"]})
        result = normalize_column(df, "x", {"a": "A"})
        assert list(result) == ["A", "b"]


class TestFilterPipeline:
    def test_filter_pipeline_empty_input(self):
        empty = pd.DataFrame(columns=["encoding", "run_status"])
        pipe = FilterPipeline(empty)
        assert pipe.result().empty
        assert len(pipe.audit_dataframe()) == 1

    def test_filter_pipeline_chaining(self, sample_metrics_df):
        result = (
            FilterPipeline(sample_metrics_df, label="test")
            .filter_encoding()
            .filter_execution_mode()
            .filter_decoders()
            .result()
        )
        assert len(result) == 3
        assert set(result["run_id"]) == {"r1", "r3", "r4"}

    def test_filter_pipeline_audit_tracks_row_counts(self, sample_metrics_df):
        pipe = (
            FilterPipeline(sample_metrics_df, label="raw")
            .filter_encoding()
            .filter_execution_mode()
        )
        audit = pipe.audit_dataframe()
        assert "step" in audit.columns
        assert "input_rows" in audit.columns
        assert "output_rows" in audit.columns
        assert "rows_removed" in audit.columns
        assert len(audit) == 3

    def test_filter_encoding_allowlist(self, sample_metrics_df):
        pipe = FilterPipeline(sample_metrics_df).filter_encoding({"ilp_derived"})
        result = pipe.result()
        assert all(result["encoding"].str.lower() == "ilp_derived")

    def test_filter_encoding_case_insensitive(self):
        df = pd.DataFrame({"encoding": ["ILP_DERIVED", "Wht_Truncated"]})
        pipe = FilterPipeline(df).filter_encoding()
        assert len(pipe.result()) == 2

    def test_filter_execution_mode_mixture(self, sample_metrics_df):
        pipe = FilterPipeline(sample_metrics_df).filter_execution_mode()
        result = pipe.result()
        assert all(result["execution_mode"].str.lower() == "mixture")

    def test_filter_decoders(self, sample_metrics_df):
        pipe = FilterPipeline(sample_metrics_df).filter_decoders()
        result = pipe.result()
        assert set(result["decoder"].str.lower()).issubset({"bp1", "bruteforce"})

    def test_filter_completed_status_normalization(self):
        df = pd.DataFrame({"run_status": ["completed", "ok", "failed", "error", None]})
        pipe = FilterPipeline(df).filter_completed()
        result = pipe.result()
        assert len(result) == 2

    def test_filter_preserves_columns(self, sample_metrics_df):
        pipe = FilterPipeline(sample_metrics_df).filter_encoding()
        assert list(pipe.result().columns) == list(sample_metrics_df.columns)

    def test_generic_filter(self, sample_metrics_df):
        pipe = FilterPipeline(sample_metrics_df).filter(
            sample_metrics_df["run_id"].isin(["r1", "r2"]),
            label="specific_runs", reason="manual selection",
        )
        assert len(pipe.result()) == 2


class TestBuildMetricPanel:
    def test_metric_panel_availability_format(self, sample_metrics_df):
        panel = build_metric_panel(
            sample_metrics_df[sample_metrics_df["run_status"] == "completed"],
            metrics=["best_sampled_F"],
        )
        assert "availability" in panel.columns
        for val in panel["availability"]:
            assert "/" in str(val)

    def test_metric_panel_missing_metric_unavailable(self, sample_metrics_df):
        panel = build_metric_panel(sample_metrics_df, metrics=["nonexistent_metric"])
        assert all(panel["status"] == "unavailable")

    def test_metric_panel_nan_handling(self, sample_metrics_df):
        panel = build_metric_panel(sample_metrics_df, metrics=["best_sampled_F"])
        assert "mean_value" in panel.columns

    def test_metric_panel_groupby_encoding(self, sample_metrics_df):
        panel = build_metric_panel(sample_metrics_df, metrics=["best_sampled_F"], group_by="encoding")
        assert "group" in panel.columns
        assert "metric" in panel.columns


class TestComputePairedDelta:
    def test_paired_delta_basic(self):
        df = pd.DataFrame({
            "instance_id": ["i1", "i1"], "decoder": ["bp1", "bp1"],
            "encoding": ["wht_truncated", "ilp_derived"], "score": [100.0, 120.0],
        })
        result = compute_paired_delta(df, pair_keys=["instance_id", "decoder"], metrics=["score"])
        assert len(result) == 1
        assert "delta_score" in result.columns
        assert result["delta_score"].iloc[0] == pytest.approx(20.0)

    def test_paired_delta_missing_pair(self):
        df = pd.DataFrame({
            "instance_id": ["i1", "i2"], "decoder": ["bp1", "bp1"],
            "encoding": ["wht_truncated", "ilp_derived"], "score": [100.0, 120.0],
        })
        result = compute_paired_delta(df, pair_keys=["instance_id", "decoder"], metrics=["score"])
        assert result.empty

    def test_paired_delta_multiple_metrics(self):
        df = pd.DataFrame({
            "instance_id": ["i1", "i1"], "decoder": ["bp1", "bp1"],
            "encoding": ["wht_truncated", "ilp_derived"],
            "score_a": [100.0, 120.0], "score_b": [50.0, 45.0],
        })
        result = compute_paired_delta(df, pair_keys=["instance_id", "decoder"], metrics=["score_a", "score_b"])
        assert "delta_score_a" in result.columns
        assert "delta_score_b" in result.columns
        assert result["delta_score_a"].iloc[0] == pytest.approx(20.0)
        assert result["delta_score_b"].iloc[0] == pytest.approx(-5.0)
