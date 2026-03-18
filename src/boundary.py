"""Helpers for canonical execution-boundary records and capture."""

from __future__ import annotations

from dataclasses import dataclass
import time
import tracemalloc
from typing import Any, Callable


ALLOWED_RUN_STATUS = {"completed", "unavailable", "failed"}
ALLOWED_BOUNDARY_REASONS = {
    "qubit_count_exceeded",
    "statevector_memory_exceeded",
    "timeout",
    "runtime_exception",
}
ERROR_MESSAGE_LIMIT = 200


@dataclass
class ExecutionCapture:
    result: Any | None
    error: BaseException | None
    wall_clock_s: float
    peak_memory_mb: float | None


def _truncate_error_message(message: str | None) -> str | None:
    if message is None:
        return None
    raw = str(message)
    return raw[:ERROR_MESSAGE_LIMIT]


def classify_boundary_exception(exc: BaseException) -> tuple[str, str, str]:
    error_type = exc.__class__.__name__
    message = _truncate_error_message(str(exc))
    if isinstance(exc, MemoryError):
        return "statevector_memory_exceeded", error_type, message or ""
    return "runtime_exception", error_type, message or ""


def capture_execution_metrics(fn: Callable[[], Any]) -> ExecutionCapture:
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        result = fn()
        wall_clock_s = float(time.perf_counter() - t0)
        peak_memory_mb = float(tracemalloc.get_traced_memory()[1] / 1e6)
        return ExecutionCapture(
            result=result,
            error=None,
            wall_clock_s=wall_clock_s,
            peak_memory_mb=peak_memory_mb,
        )
    except BaseException as exc:
        wall_clock_s = float(time.perf_counter() - t0)
        peak_memory_mb = float(tracemalloc.get_traced_memory()[1] / 1e6)
        return ExecutionCapture(
            result=None,
            error=exc,
            wall_clock_s=wall_clock_s,
            peak_memory_mb=peak_memory_mb,
        )
    finally:
        tracemalloc.stop()


def build_boundary_record(
    *,
    base_record: dict[str, Any],
    run_status: str,
    boundary_reason: str,
    attempted_execution: bool,
    estimated_qubits: int | None,
    estimated_memory_gb: float | None,
    error_type: str | None,
    error_message: str | None,
    wall_clock_s: float | None,
    peak_memory_mb: float | None,
) -> dict[str, Any]:
    if run_status not in ALLOWED_RUN_STATUS:
        raise ValueError(f"unsupported run_status for boundary record: {run_status!r}")
    if boundary_reason not in ALLOWED_BOUNDARY_REASONS:
        raise ValueError(f"unsupported boundary_reason: {boundary_reason!r}")

    row = dict(base_record)
    row["run_status"] = run_status
    row["run_error"] = error_message
    row["attempted_execution"] = bool(attempted_execution)
    row["boundary_reason"] = boundary_reason
    row["error_type"] = error_type
    row["error_message"] = _truncate_error_message(error_message)
    row["estimated_qubits"] = None if estimated_qubits is None else int(estimated_qubits)
    row["estimated_memory_gb"] = None if estimated_memory_gb is None else float(estimated_memory_gb)
    row["wall_clock_s"] = None if wall_clock_s is None else float(wall_clock_s)
    row["peak_memory_mb"] = None if peak_memory_mb is None else float(peak_memory_mb)

    for key in (
        "best_sampled_F",
        "best_sampled_G",
        "top1_regret",
        "topk_regret",
        "optimum_recall_at_k",
        "best_top_G_sampled_F",
        "spearman_rho_F_G",
        "retained_energy_eta",
        "decision_distortion",
        "decode_success",
        "postselection_success",
    ):
        row[key] = None

    return row
