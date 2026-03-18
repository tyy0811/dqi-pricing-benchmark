"""Centralized notebook filtering, metric aggregation, and data utilities."""

from __future__ import annotations
from typing import Any, Optional
import json
import numpy as np
import pandas as pd

# Encoding constants
EXACT_REFERENCE_ENCODINGS = frozenset({"ilp_derived", "ilp_derived_exact", "ilp_derived_exact_where_feasible"})
WHT_EXACT_ENCODINGS = frozenset({"wht_exact"})
WHT_TRUNCATED_ENCODINGS = frozenset({"wht_truncated", "wht_truncated_pricing"})
COMPARABLE_ENCODINGS = EXACT_REFERENCE_ENCODINGS | WHT_EXACT_ENCODINGS | WHT_TRUNCATED_ENCODINGS
DEFAULT_DECODERS = frozenset({"bp1", "bruteforce"})
DEFAULT_EXECUTION_MODE = "mixture"


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_column(df: pd.DataFrame, col: str, mapping: dict[str, str]) -> pd.Series:
    return df[col].map(lambda v: mapping.get(v, v))


class FilterPipeline:
    """Track cumulative filtering with audit trail."""

    def __init__(self, df: pd.DataFrame, label: str = "input"):
        self._df = df.copy()
        self._audit: list[dict[str, Any]] = [{
            "step_id": 0, "step": label, "input_rows": len(df),
            "output_rows": len(df), "rows_removed": 0, "reason_label": "source",
        }]

    def _next_step_id(self) -> int:
        return len(self._audit)

    def filter(self, condition: pd.Series, label: str, reason: str) -> "FilterPipeline":
        input_rows = len(self._df)
        self._df = self._df[condition].copy()
        output_rows = len(self._df)
        self._audit.append({
            "step_id": self._next_step_id(), "step": label,
            "input_rows": input_rows, "output_rows": output_rows,
            "rows_removed": input_rows - output_rows, "reason_label": reason,
        })
        return self

    def filter_encoding(self, allowlist: set[str] | frozenset[str] = COMPARABLE_ENCODINGS) -> "FilterPipeline":
        if "encoding" not in self._df.columns:
            return self
        lower_allow = {e.lower() for e in allowlist}
        condition = self._df["encoding"].astype(str).str.strip().str.lower().isin(lower_allow)
        return self.filter(condition, "encoding_allowlist", f"encoding in {sorted(allowlist)}")

    def filter_execution_mode(self, mode: str = DEFAULT_EXECUTION_MODE) -> "FilterPipeline":
        if "execution_mode" not in self._df.columns:
            return self
        condition = self._df["execution_mode"].astype(str).str.strip().str.lower() == mode.lower()
        return self.filter(condition, "execution_mode", f"execution_mode == {mode!r}")

    def filter_decoders(self, decoders: set[str] | frozenset[str] = DEFAULT_DECODERS) -> "FilterPipeline":
        if "decoder" not in self._df.columns:
            return self
        lower_decoders = {d.lower() for d in decoders}
        condition = self._df["decoder"].astype(str).str.strip().str.lower().isin(lower_decoders)
        return self.filter(condition, "decoder_filter", f"decoder in {sorted(decoders)}")

    def filter_completed(self) -> "FilterPipeline":
        if "run_status" not in self._df.columns:
            return self
        completed_aliases = {"completed", "ok"}
        condition = self._df["run_status"].astype(str).str.strip().str.lower().isin(completed_aliases)
        return self.filter(condition, "completed_only", "run_status in {completed, ok}")

    def audit_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._audit)

    def result(self) -> pd.DataFrame:
        return self._df.copy()


def build_metric_panel(df: pd.DataFrame, metrics: list[str], group_by: str = "encoding") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if group_by not in df.columns:
        groups = [("all", df)]
    else:
        groups = list(df.groupby(group_by))
    for group_name, group_df in groups:
        for metric in metrics:
            if metric not in group_df.columns:
                rows.append({"group": str(group_name), "metric": metric, "availability": "0/0",
                    "mean_value": "unavailable", "median_value": "unavailable", "status": "unavailable"})
                continue
            values = pd.to_numeric(group_df[metric], errors="coerce")
            total = len(values)
            available = int(values.notna().sum())
            rows.append({"group": str(group_name), "metric": metric, "availability": f"{available}/{total}",
                "mean_value": float(values.mean()) if available > 0 else "unavailable",
                "median_value": float(values.median()) if available > 0 else "unavailable",
                "status": "available" if available > 0 else "unavailable"})
    return pd.DataFrame(rows)


def compute_paired_delta(df: pd.DataFrame, pair_keys: list[str], metrics: list[str],
    baseline_encoding: str = "wht_truncated", comparison_encoding: str = "ilp_derived") -> pd.DataFrame:
    if "encoding" not in df.columns:
        return pd.DataFrame()
    baseline = df[df["encoding"].astype(str).str.lower() == baseline_encoding.lower()].copy()
    comparison = df[df["encoding"].astype(str).str.lower() == comparison_encoding.lower()].copy()
    if baseline.empty or comparison.empty:
        return pd.DataFrame()
    merged = baseline.merge(comparison, on=pair_keys, suffixes=("_baseline", "_comparison"), how="inner")
    if merged.empty:
        return pd.DataFrame()
    result_cols = {k: merged[k] for k in pair_keys}
    for metric in metrics:
        base_col, comp_col = f"{metric}_baseline", f"{metric}_comparison"
        if base_col in merged.columns and comp_col in merged.columns:
            result_cols[f"delta_{metric}"] = (
                pd.to_numeric(merged[comp_col], errors="coerce") - pd.to_numeric(merged[base_col], errors="coerce"))
    return pd.DataFrame(result_cols)
