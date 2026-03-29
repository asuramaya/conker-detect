from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from conker_detect.replay import replay_parameter_golf


def _load_module(filename: str):
    root = Path(__file__).resolve().parents[1]
    module_path = root / "examples" / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_reports_aggregate_and_repeatability_for_legal_adapter() -> None:
    module = _load_module("packed_cache_demo_adapter.py")
    runner = module.build_adapter({"mode": "legal", "vocab_size": 8})
    tokens = np.array([0, 1, 2, 1, 0, 3, 2, 1], dtype=np.int64)

    result = replay_parameter_golf(
        runner,
        tokens,
        chunk_size=4,
        max_chunks=2,
        sample_chunks=2,
        position_batch_size=2,
        seed=0,
    )

    assert result["profile"] == "parameter-golf"
    assert result["token_count"] == 8
    assert result["aggregate"]["mean_bpb"] is not None
    assert result["repeatability"]["covered"] is True
    assert result["repeatability"]["pass"] is True
    assert len(result["chunks"]) == 2
    assert all("total_loss_nats" in row for row in result["chunks"])


def test_replay_detects_state_hash_mismatch() -> None:
    module = _load_module("packed_cache_demo_adapter.py")
    runner = module.build_adapter({"mode": "state_hash_cheat", "vocab_size": 8})
    tokens = np.array([0, 1, 2, 1, 0, 3, 2, 1], dtype=np.int64)

    result = replay_parameter_golf(
        runner,
        tokens,
        chunk_size=8,
        max_chunks=1,
        sample_chunks=1,
        position_batch_size=4,
        seed=0,
    )

    assert result["repeatability"]["covered"] is True
    assert result["repeatability"]["pass"] is False
    assert result["repeatability"]["state_hash_mismatch_count"] > 0


def test_replay_can_sample_subset_of_chunks() -> None:
    module = _load_module("packed_cache_demo_adapter.py")
    runner = module.build_adapter({"mode": "legal", "vocab_size": 8})
    tokens = np.array([0, 1, 2, 1, 0, 3, 2, 1, 4, 5, 6, 7], dtype=np.int64)

    result = replay_parameter_golf(
        runner,
        tokens,
        chunk_size=4,
        max_chunks=3,
        sample_chunks=1,
        position_batch_size=2,
        seed=123,
    )

    assert result["chunk_count"] == 3
    assert len(result["selected_chunks"]) == 1
    compared_chunks = [row for row in result["chunks"] if row["repeat_compared"]]
    assert len(compared_chunks) == 1


def test_replay_works_with_non_trace_adapter() -> None:
    module = _load_module("causal_demo_adapter.py")
    runner = module.build_adapter({"vocab_size": 8})
    tokens = np.array([0, 1, 2, 1, 0, 3], dtype=np.int64)

    result = replay_parameter_golf(
        runner,
        tokens,
        chunk_size=3,
        max_chunks=2,
        sample_chunks=2,
        position_batch_size=2,
        seed=0,
    )

    assert result["repeatability"]["covered"] is True
    assert result["repeatability"]["state_hash_compared_count"] == 0
    assert result["repeatability"]["pass"] is True
