from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from conker_detect.legality import audit_parameter_golf_legality


def _load_demo_adapter_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "examples" / "packed_cache_demo_adapter.py"
    spec = importlib.util.spec_from_file_location("packed_cache_demo_adapter", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load adapter module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audit(mode: str, *, seed: int = 0) -> dict:
    module = _load_demo_adapter_module()
    runner = module.build_adapter({"mode": mode, "vocab_size": 8})
    tokens = np.array([0, 1, 2, 1, 0, 3, 2, 1], dtype=np.int64)
    return audit_parameter_golf_legality(
        runner,
        tokens,
        chunk_size=8,
        max_chunks=1,
        sample_chunks=1,
        future_probes_per_chunk=1,
        answer_probes_per_chunk=2,
        positions_per_future_probe=2,
        seed=seed,
        vocab_size=8,
    )


def test_legal_trace_checks_pass() -> None:
    result = _audit("legal")

    assert result["checks"]["trace_coverage"]["covered"] is True
    assert result["checks"]["trace_coverage"]["pass"] is True
    assert result["checks"]["accounting_contribution_consistency"]["covered"] is True
    assert result["checks"]["accounting_contribution_consistency"]["pass"] is True
    assert result["checks"]["state_hash_consistency"]["covered"] is True
    assert result["checks"]["state_hash_consistency"]["pass"] is True


def test_reported_loss_cheat_fails_accounting_contribution_consistency() -> None:
    result = _audit("reported_loss_cheat")

    assert result["checks"]["accounting_contribution_consistency"]["covered"] is True
    assert result["checks"]["accounting_contribution_consistency"]["pass"] is False
    assert any(
        row["kind"] == "accounting_contribution_consistency" and not row["pass"]
        for row in result["probes"]
    )


def test_counted_flag_cheat_fails_accounting_path_invariance() -> None:
    result = _audit("counted_flag_cheat", seed=1)

    assert result["checks"]["accounting_path_invariance"]["covered"] is True
    assert result["checks"]["accounting_path_invariance"]["pass"] is False


def test_path_id_cheat_fails_accounting_path_invariance() -> None:
    result = _audit("path_id_cheat")

    assert result["checks"]["accounting_path_invariance"]["covered"] is True
    assert result["checks"]["accounting_path_invariance"]["pass"] is False


def test_state_hash_cheat_fails_state_hash_consistency() -> None:
    result = _audit("state_hash_cheat")

    assert result["checks"]["state_hash_consistency"]["covered"] is True
    assert result["checks"]["state_hash_consistency"]["pass"] is False
