"""Reusable notebook helpers for loading, formatting, and visualization."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from src.problem_generator import PricingProblem


TIER_LABELS = {
    0: "Standard",
    1: "Premium",
    2: "Luxury",
    3: "INVALID",
}


_REPO_REQUIRED_MARKERS = ("src", "notebooks")
_REPO_OPTIONAL_MARKERS = ("pyproject.toml", ".git")


def _candidate_root_paths() -> list[Path]:
    starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    ordered: list[Path] = []
    seen: set[Path] = set()
    for start in starts:
        current = start
        while True:
            if current not in seen:
                ordered.append(current)
                seen.add(current)
            if current.parent == current:
                break
            current = current.parent
    return ordered


def _repo_marker_checks(path: Path) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for marker in _REPO_REQUIRED_MARKERS:
        checks[marker] = (path / marker).is_dir()
    for marker in _REPO_OPTIONAL_MARKERS:
        checks[marker] = (path / marker).exists()
    return checks


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return repository root using marker-based discovery."""
    candidates = _candidate_root_paths()
    visited: list[tuple[Path, dict[str, bool]]] = []
    for candidate in candidates:
        checks = _repo_marker_checks(candidate)
        visited.append((candidate, checks))
        if all(checks[m] for m in _REPO_REQUIRED_MARKERS):
            return candidate

    lines = [
        "Unable to resolve repository root.",
        f"Required markers: {list(_REPO_REQUIRED_MARKERS)}",
        f"Optional markers: {list(_REPO_OPTIONAL_MARKERS)}",
        "Visited candidates:",
    ]
    for path, checks in visited:
        summary = ", ".join(f"{k}={v}" for k, v in checks.items())
        lines.append(f"- {path}: {summary}")
    raise RuntimeError("\n".join(lines))


def resolve_repo_path(pathlike: str | Path) -> Path:
    """Resolve absolute path for repo-local files."""
    path = Path(pathlike)
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


@lru_cache(maxsize=1)
def _problem_generator_module():
    module_path = repo_root() / "src" / "problem_generator.py"
    spec = importlib.util.spec_from_file_location("problem_generator", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load problem_generator module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_instance(pathlike: str | Path) -> "PricingProblem":
    """Load a frozen pricing instance JSON into a PricingProblem."""
    path = resolve_repo_path(pathlike)
    problem_generator = _problem_generator_module()
    PricingProblem = problem_generator.PricingProblem
    with open(path, "r") as f:
        data = json.load(f)
    return PricingProblem.from_dict(data)


def decode_config_table(prob: "PricingProblem", x: int) -> pd.DataFrame:
    """Decode bitstring into a readable feature/tier/price table."""
    problem_generator = _problem_generator_module()
    decode_bitstring = problem_generator.decode_bitstring
    tiers = decode_bitstring(x, prob.n_features)
    rows: list[dict] = []
    for feature, tier in zip(prob.features, tiers):
        price = feature.prices[tier] if tier != 3 else 0.0
        rows.append(
            {
                "feature": feature.name,
                "tier": TIER_LABELS[tier],
                "price": float(price),
            }
        )
    return pd.DataFrame(rows)


def format_optimal_table(instance_name: str, prob: "PricingProblem", x_opt: int) -> pd.DataFrame:
    """Build required artifact table: instance, feature, tier, price."""
    table = decode_config_table(prob, x_opt).copy()
    table.insert(0, "instance", instance_name)
    return table[["instance", "feature", "tier", "price"]]


def format_feasibility_stats(stats: dict[str, float | int]) -> pd.DataFrame:
    """Render standardized feasibility stats fields."""
    ordered = {
        "valid fraction": stats["valid_fraction"],
        "invalid-tier count": stats["invalid_tier_count"],
        "bundle violations": stats["bundle_violations"],
        "luxury-cap violations": stats["luxury_cap_violations"],
    }
    return pd.DataFrame([ordered])


def top_k_distribution(probabilities: np.ndarray, top_k: int = 12) -> pd.DataFrame:
    """Return top-k bitstrings by probability."""
    probs = np.asarray(probabilities, dtype=np.float64)
    n_bits = int(np.log2(probs.size))
    idx = np.argsort(probs)[::-1][:top_k]
    return pd.DataFrame(
        {
            "x": idx.astype(int),
            "bitstring": [format(int(i), f"0{n_bits}b") for i in idx],
            "probability": probs[idx],
        }
    )


def side_by_side_distribution(
    left: np.ndarray,
    right: np.ndarray,
    *,
    top_k: int = 8,
    left_label: str = "left_prob",
    right_label: str = "right_prob",
    state_label: str = "bitstring",
) -> pd.DataFrame:
    """Return a combined top-support table for two distributions."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    n_bits = int(np.log2(left.size))
    idx = top_support_indices(left, right, top_k=top_k)
    return pd.DataFrame(
        {
            state_label: [format(int(i), f"0{n_bits}b") for i in idx],
            left_label: left[idx],
            right_label: right[idx],
            "abs_diff": np.abs(left[idx] - right[idx]),
        }
    )


def bit_reversed_tvd(p: np.ndarray, q: np.ndarray) -> float:
    """TVD after reversing the displayed bit order of q for diagnosis only."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    n_bits = int(np.log2(q.size))
    remapped = np.zeros_like(q)
    for idx, value in enumerate(q):
        reversed_idx = int(format(int(idx), f"0{n_bits}b")[::-1], 2)
        remapped[reversed_idx] = value
    return tvd(p, remapped)


def distribution_table_with_scores(
    probabilities: np.ndarray,
    top_k: int,
    score_fn: Callable[[int], float],
    score_label: str,
    optimum_x: Optional[int] = None,
) -> pd.DataFrame:
    """Top-k probability table augmented with score column and optimum flag."""
    df = top_k_distribution(probabilities, top_k=top_k)
    df[score_label] = [float(score_fn(int(x))) for x in df["x"]]
    df["optimal"] = [bool(optimum_x is not None and int(x) == int(optimum_x)) for x in df["x"]]
    return df[["bitstring", "probability", score_label, "optimal"]]


def plot_distribution(
    probabilities: np.ndarray,
    top_k: int = 16,
    title: str = "Distribution",
    ax: Optional[plt.Axes] = None,
    color: str = "#1f77b4",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot top-k histogram from a probability distribution."""
    probs = np.asarray(probabilities, dtype=np.float64)
    n_bits = int(np.log2(probs.size))
    idx = np.argsort(probs)[::-1][:top_k]
    labels = [format(int(i), f"0{n_bits}b") for i in idx]
    values = probs[idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4.5))
    else:
        fig = ax.figure

    ax.bar(np.arange(len(idx)), values, color=color, alpha=0.9)
    ax.set_xticks(np.arange(len(idx)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def top_support_indices(p: np.ndarray, q: np.ndarray, top_k: int = 16) -> np.ndarray:
    """Return indices of highest combined probability mass."""
    combined = np.asarray(p, dtype=np.float64) + np.asarray(q, dtype=np.float64)
    return np.argsort(combined)[::-1][:top_k]


def plot_distribution_overlay(
    p: np.ndarray,
    q: np.ndarray,
    top_k: int = 16,
    title: str = "Distribution Overlay",
    labels: tuple[str, str] = ("A", "B"),
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay two distributions on shared top-support states."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    n_bits = int(np.log2(p.size))
    idx = top_support_indices(p, q, top_k=top_k)

    x = np.arange(len(idx))
    width = 0.42

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x - width / 2, p[idx], width=width, label=labels[0], alpha=0.85)
    ax.bar(x + width / 2, q[idx], width=width, label=labels[1], alpha=0.85)

    bit_labels = [format(int(i), f"0{n_bits}b") for i in idx]
    ax.set_xticks(x)
    ax.set_xticklabels(bit_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(0.5 * np.sum(np.abs(p - q)))


def _two_qubit_gate_count(qc) -> int:
    """Best-effort count of 2-qubit operations in a Qiskit circuit."""
    count = 0
    for instruction in qc.data:
        op = instruction.operation
        if getattr(op, "num_qubits", 0) == 2:
            count += 1
    return count


def resource_table(
    circuit,
    transpiled_circuit,
    basis_gates: list[str],
    optimization_level: int,
    backend_assumption: str = "generic basis-only target",
) -> pd.DataFrame:
    """Build a labeled resource summary row for Notebook 04."""
    return pd.DataFrame(
        [
            {
                "n_qubits": int(circuit.num_qubits),
                "depth_raw": int(circuit.depth()),
                "depth_transpiled": int(transpiled_circuit.depth()),
                "two_qubit_gates": int(_two_qubit_gate_count(transpiled_circuit)),
                "basis_gates": ",".join(basis_gates),
                "optimization_level": int(optimization_level),
                "backend_assumption": backend_assumption,
            }
        ]
    )


def verification_card(metrics: dict) -> pd.DataFrame:
    """Render one compact verification card row from a metrics dictionary."""
    return pd.DataFrame([metrics])


def get_heuristic_dqi_report(result_record: dict) -> dict:
    """Return heuristic-eigenvector DQI report with backward-compatible fallback."""
    if "dqi_heuristic_optimal" in result_record:
        return result_record["dqi_heuristic_optimal"]
    return result_record["dqi_optimal"]


def format_benchmark_row(result_record: dict) -> dict:
    """Format one benchmark JSON record into a notebook-friendly summary row."""
    dqi_u = result_record["dqi_uniform"]
    dqi_h = get_heuristic_dqi_report(result_record)
    alpha_diag = result_record.get("alpha_diagnostics", {})
    cp_sat = result_record.get("baselines", {}).get("cp_sat", {})

    return {
        "n_features": result_record["problem"]["n_features"],
        "n_bits": result_record["problem"]["n_bits"],
        "k": result_record.get("parameters", {}).get("k"),
        "ell": result_record.get("parameters", {}).get("ell"),
        "m": dqi_u.get("m_constraints"),
        "f_opt": result_record["ground_truth"]["f_opt"],
        "dqi_uniform_ratio": dqi_u["approximation_ratio"],
        "dqi_heuristic_ratio": dqi_h["approximation_ratio"],
        "dqi_uniform_success_prob": dqi_u.get("success_prob"),
        "dqi_heuristic_success_prob": dqi_h.get("success_prob"),
        "alpha_l1_distance": alpha_diag.get("alpha_l1_distance"),
        "alpha_l2_distance": alpha_diag.get("alpha_l2_distance"),
        "alpha_effectively_equal": alpha_diag.get("alpha_effectively_equal"),
        "cp_sat_status": cp_sat.get("status"),
        "cp_sat_f": cp_sat.get("f"),
        "cp_sat_matches_bruteforce_objective": cp_sat.get("matches_bruteforce_objective"),
    }


def format_benchmark_table(result_records: list[dict]) -> pd.DataFrame:
    """Build a compact benchmark table from parsed JSON records."""
    rows = [format_benchmark_row(record) for record in result_records]
    return pd.DataFrame(rows)


def require_artifact_paths(paths: list[str | Path]) -> list[Path]:
    """Fail fast only when core artifact inputs are missing."""
    resolved = [resolve_repo_path(p) for p in paths]
    missing = [str(p) for p in resolved if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required artifact input(s):\n- " + "\n- ".join(missing)
        )
    return resolved


def load_matrix_summary(matrix_name: str, results_root: str | Path = "results") -> pd.DataFrame:
    """Load an aggregated matrix CSV from results/tables."""
    csv_path = resolve_repo_path(Path(results_root) / "tables" / f"{matrix_name}_summary.csv")

    def _loader() -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(str(csv_path))
        return pd.read_csv(csv_path)

    return _load_artifact_with_policy(
        label=f"{matrix_name}_summary",
        required=True,
        checked_paths=[csv_path],
        default=pd.DataFrame(),
        loader=_loader,
    )


def load_matrix_report(matrix_name: str, results_root: str | Path = "results") -> dict[str, Any]:
    """Load an aggregated matrix JSON report from results/reports."""
    report_path = resolve_repo_path(Path(results_root) / "reports" / f"{matrix_name}_report.json")

    def _loader() -> dict[str, Any]:
        if not report_path.exists():
            raise FileNotFoundError(str(report_path))
        with open(report_path, "r") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}

    return _load_artifact_with_policy(
        label=f"{matrix_name}_report",
        required=True,
        checked_paths=[report_path],
        default={},
        loader=_loader,
    )



# parse_metric_availability — moved to src.notebook_utils
# unavailable_panel_table — moved to src.notebook_utils
# unavailable_or_numeric — moved to src.notebook_utils
# ensure_numeric_columns — moved to src.notebook_utils
# ensure_object_columns — moved to src.notebook_utils
# ensure_optional_columns — moved to src.notebook_utils
# placeholder_or_frame — moved to src.notebook_utils


def placeholder_plot(
    title: str,
    reason: str,
    figsize: tuple[float, float] = (8.0, 2.4),
) -> tuple[plt.Figure, plt.Axes]:
    """Render a visible placeholder plot when a requested slice is unavailable."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.text(
        0.01,
        0.6,
        f"{title}\nUnavailable: {reason}",
        fontsize=11,
        ha="left",
        va="center",
        wrap=True,
    )
    fig.tight_layout()
    return fig, ax


def load_run_manifest(
    results_root: str | Path = "results",
    required: bool = False,
) -> dict[str, Any]:
    """Load canonical run manifest."""
    path = resolve_repo_path(Path(results_root) / "run_manifest.json")

    def _loader() -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(str(path))
        with open(path, "r") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}

    return _load_artifact_with_policy(
        label="run_manifest",
        required=required,
        checked_paths=[path],
        default={},
        loader=_loader,
    )


def load_metrics_master(
    results_root: str | Path = "results",
    required: bool = False,
) -> pd.DataFrame:
    """Load canonical metrics master (parquet preferred, JSON fallback)."""
    root = resolve_repo_path(results_root)
    return _load_table_with_fallback(root=root, base_name="metrics_master", required=required)


def load_resources_master(
    results_root: str | Path = "results",
    required: bool = False,
) -> pd.DataFrame:
    """Load canonical resources master (parquet preferred, JSON fallback)."""
    root = resolve_repo_path(results_root)
    return _load_table_with_fallback(root=root, base_name="resources_master", required=required)


def _missing_artifact_message(label: str, paths: list[Path]) -> str:
    lines = [f"Missing required artifact for `{label}`. Checked paths:"]
    lines.extend(f"- {p}" for p in paths)
    return "\n".join(lines)


def _load_artifact_with_policy(
    *,
    label: str,
    required: bool,
    checked_paths: list[Path],
    default: Any,
    loader: Callable[[], Any],
) -> Any:
    try:
        return loader()
    except FileNotFoundError as exc:
        if required:
            raise FileNotFoundError(_missing_artifact_message(label, checked_paths)) from exc
        return default
    except Exception:
        if required:
            raise
        return default


def _rows_to_frame(payload: Any) -> pd.DataFrame:
    if isinstance(payload, dict):
        rows = payload.get("rows", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    return pd.DataFrame(rows)


def _load_table_with_fallback(
    *,
    root: Path,
    base_name: str,
    required: bool = True,
) -> pd.DataFrame:
    parquet_path = root / f"{base_name}.parquet"
    json_path = root / f"{base_name}.json"
    checked_paths = [parquet_path, json_path]

    def _loader() -> pd.DataFrame:
        if parquet_path.exists():
            try:
                return pd.read_parquet(parquet_path)
            except Exception:
                # Optional fallback path: JSON may still be readable.
                pass
        if not json_path.exists():
            raise FileNotFoundError(str(json_path))
        with open(json_path, "r") as f:
            payload = json.load(f)
        return _rows_to_frame(payload)

    return _load_artifact_with_policy(
        label=base_name,
        required=required,
        checked_paths=checked_paths,
        default=pd.DataFrame(),
        loader=_loader,
    )


def load_quality_cost_master(
    results_root: str | Path = "results",
    required: bool = False,
) -> pd.DataFrame:
    """Load Stage E quality-cost comparable rows (parquet preferred)."""
    root = resolve_repo_path(results_root)
    return _load_table_with_fallback(root=root, base_name="quality_cost_master", required=required)


def load_quality_cost_unavailable(
    results_root: str | Path = "results",
    required: bool = False,
) -> pd.DataFrame:
    """Load Stage E quality-cost unavailable rows (parquet preferred)."""
    root = resolve_repo_path(results_root)
    return _load_table_with_fallback(root=root, base_name="quality_cost_unavailable", required=required)



# normalize_run_status — moved to src.notebook_utils
# ensure_run_status_norm — moved to src.notebook_utils
# normalize_run_status_label — moved to src.notebook_utils
# manifest_slots_dataframe — moved to src.notebook_utils
# mark_not_in_experiment_matrix — moved to src.notebook_utils


# Backward-compatible re-exports from src.notebook_utils
from src.notebook_utils import (  # noqa: E402, F401
    ensure_run_status_norm,
    normalize_run_status,
    normalize_run_status_label,
    parse_metric_availability,
    ensure_numeric_columns,
    ensure_object_columns,
    ensure_optional_columns,
    unavailable_or_numeric,
    unavailable_panel_table,
    placeholder_or_frame,
    manifest_slots_dataframe,
    mark_not_in_experiment_matrix,
)
