"""Tests for boundary-row and execution-capture helpers."""

from __future__ import annotations

from src.matrix_runner import _base_record


def _base_matrix_row() -> dict:
    row = _base_record(
        matrix="matrix_a",
        run_id="matrix_a_P7_ilp_derived_bp1_uniform_mixture_seed42",
        run_status="completed",
        run_error=None,
        instance_id="P7",
        family="P",
        encoding="ilp_derived",
        decoder="bp1",
        alpha_mode="uniform",
        execution_mode="mixture",
        seed=42,
    )
    row["ell"] = 2
    return row


def test_build_boundary_record_for_preflight_skip() -> None:
    from src.boundary import build_boundary_record

    row = build_boundary_record(
        base_record=_base_matrix_row(),
        run_status="unavailable",
        boundary_reason="qubit_count_exceeded",
        attempted_execution=False,
        estimated_qubits=31,
        estimated_memory_gb=16.0,
        error_type=None,
        error_message=None,
        wall_clock_s=0.0,
        peak_memory_mb=0.0,
    )

    assert row["run_status"] == "unavailable"
    assert row["attempted_execution"] is False
    assert row["boundary_reason"] == "qubit_count_exceeded"
    assert row["instance_id"] == "P7"
    assert row["encoding"] == "ilp_derived"
    assert row["decoder"] == "bp1"
    assert row["alpha_mode"] == "uniform"
    assert row["execution_mode"] == "mixture"
    assert row["ell"] == 2
    assert row["estimated_qubits"] == 31
    assert row["best_sampled_F"] is None
    assert row["top1_regret"] is None
    assert row["topk_regret"] is None
    assert row["postselection_success"] is None


def test_build_boundary_record_for_attempted_execution_failure() -> None:
    from src.boundary import build_boundary_record

    row = build_boundary_record(
        base_record=_base_matrix_row(),
        run_status="failed",
        boundary_reason="runtime_exception",
        attempted_execution=True,
        estimated_qubits=29,
        estimated_memory_gb=8.0,
        error_type="RuntimeError",
        error_message="x" * 500,
        wall_clock_s=1.25,
        peak_memory_mb=128.0,
    )

    assert row["run_status"] == "failed"
    assert row["attempted_execution"] is True
    assert row["boundary_reason"] == "runtime_exception"
    assert row["error_type"] == "RuntimeError"
    assert len(row["error_message"]) <= 200
    assert row["wall_clock_s"] == 1.25
    assert row["peak_memory_mb"] == 128.0


def test_capture_execution_metrics_success() -> None:
    from src.boundary import capture_execution_metrics

    def _work():
        return {"ok": True}

    out = capture_execution_metrics(_work)
    assert out.result == {"ok": True}
    assert out.error is None
    assert out.wall_clock_s >= 0.0
    assert out.peak_memory_mb >= 0.0


def test_capture_execution_metrics_failure() -> None:
    from src.boundary import capture_execution_metrics

    def _work():
        raise MemoryError("oom")

    out = capture_execution_metrics(_work)
    assert out.result is None
    assert isinstance(out.error, MemoryError)
    assert out.wall_clock_s >= 0.0
    assert out.peak_memory_mb >= 0.0
