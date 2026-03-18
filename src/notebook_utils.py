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


# ---------------------------------------------------------------------------
# Functions moved from notebooks/_helpers.py
# ---------------------------------------------------------------------------


def parse_metric_availability(value: Any) -> dict[str, str]:
    """Normalize metric-availability payload from dict or serialized JSON."""
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    return {}


def unavailable_panel_table(panel: str, reason: str, details: str = "") -> pd.DataFrame:
    """Return a standard placeholder row for partially unavailable analysis slices."""
    payload = {
        "panel": str(panel),
        "status": "unavailable",
        "reason": str(reason),
        "details": str(details) if details else "",
    }
    return pd.DataFrame([payload], columns=["panel", "status", "reason", "details"])


def unavailable_or_numeric(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Annotate rows where a metric is unavailable instead of coercing to zero."""
    out = df.copy()
    if value_col not in out.columns:
        out[value_col] = np.nan
    numeric = pd.to_numeric(out[value_col], errors="coerce")
    out[value_col] = numeric
    out[f"{value_col}_status"] = np.where(numeric.isna(), "unavailable", "available")
    return out


def ensure_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Ensure requested columns exist and are numeric (missing -> NaN)."""
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def ensure_object_columns(
    df: pd.DataFrame,
    cols: list[str],
    default: Any = np.nan,
) -> pd.DataFrame:
    """Ensure requested object-like columns exist (missing -> default)."""
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = default
    return out


def ensure_optional_columns(
    df: pd.DataFrame,
    *,
    numeric_cols: Optional[list[str]] = None,
    object_defaults: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Materialize optional numeric/object columns with stable defaults."""
    out = df.copy()
    if numeric_cols:
        out = ensure_numeric_columns(out, list(numeric_cols))
    if object_defaults:
        for col, default in object_defaults.items():
            if col not in out.columns:
                out[col] = default
    return out


def placeholder_or_frame(
    df: pd.DataFrame,
    *,
    panel: str,
    reason: str,
    details: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (dataframe, placeholder) where placeholder is populated when df is empty."""
    if df.empty:
        return df, unavailable_panel_table(panel=panel, reason=reason, details=details)
    return df, pd.DataFrame()


def normalize_run_status(series_or_scalar: Any) -> Any:
    """Normalize run-status values while preserving unknown labels."""
    mapping = {
        "ok": "completed",
        "completed": "completed",
        "error": "failed",
        "failed": "failed",
        "skipped": "not_applicable",
        "unsupported": "not_applicable",
        "not_applicable": "not_applicable",
        "unavailable": "unavailable",
        "not_in_experiment_matrix": "not_in_experiment_matrix",
    }

    def _normalize_one(status: Any) -> str:
        if status is None or pd.isna(status):
            return "unavailable"
        raw = str(status).strip().lower()
        if not raw:
            return "unavailable"
        return mapping.get(raw, raw)

    if isinstance(series_or_scalar, pd.Series):
        return series_or_scalar.apply(_normalize_one)
    return _normalize_one(series_or_scalar)


def ensure_run_status_norm(
    df: pd.DataFrame,
    *,
    run_status_col: str = "run_status",
    out_col: str = "run_status_norm",
) -> pd.DataFrame:
    """Materialize normalized run status once per DataFrame."""
    out = df.copy()
    if run_status_col in out.columns:
        source = out[run_status_col]
    elif out_col in out.columns:
        source = out[out_col]
    else:
        source = pd.Series([np.nan] * len(out), index=out.index, dtype=object)
    out[out_col] = normalize_run_status(source)
    return out


def normalize_run_status_label(status: Any) -> str:
    """Normalize run-status labels for notebook display."""
    raw = str(status).strip().lower() if status is not None else ""
    if raw in {"completed", "ok"}:
        return "completed"
    if raw in {"failed", "error"}:
        return "failed"
    if raw in {"skipped", "unsupported"}:
        return "not_applicable"
    if raw in {"unavailable", "not_in_experiment_matrix"}:
        return raw
    return "unknown"


def manifest_slots_dataframe(run_manifest: dict[str, Any]) -> pd.DataFrame:
    """Extract slot rows from run manifest."""
    return pd.DataFrame(run_manifest.get("slots", []))


def mark_not_in_experiment_matrix(
    df: pd.DataFrame,
    *,
    run_manifest: dict[str, Any],
    filters: dict[str, Any],
) -> str:
    """Return display status for a requested slice against run manifest."""
    slots = manifest_slots_dataframe(run_manifest)
    if slots.empty:
        return "not_in_experiment_matrix"
    subset = slots.copy()
    for key, value in filters.items():
        if key not in subset.columns:
            continue
        if isinstance(value, (list, tuple, set)):
            subset = subset[subset[key].isin(list(value))]
        else:
            subset = subset[subset[key] == value]
    if subset.empty:
        return "not_in_experiment_matrix"
    if df.empty:
        return "planned_but_unobserved_or_filtered"
    return "in_experiment_matrix"
