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


def test_legality_reports_coverage_and_gold_logprob_consistency() -> None:
    module = _load_demo_adapter_module()
    runner = module.build_adapter({"mode": "legal", "vocab_size": 8})
    tokens = np.array([0, 1, 2, 1, 0, 3, 2, 1], dtype=np.int64)

    result = audit_parameter_golf_legality(
        runner,
        tokens,
        chunk_size=8,
        max_chunks=1,
        sample_chunks=1,
        future_probes_per_chunk=1,
        answer_probes_per_chunk=1,
        positions_per_future_probe=2,
        seed=0,
        vocab_size=8,
    )

    assert result["checks"]["normalization"]["covered"] is True
    assert result["checks"]["normalization"]["pass"] is True
    assert result["checks"]["gold_logprob_consistency"]["covered"] is True
    assert result["checks"]["gold_logprob_consistency"]["pass"] is True
    assert (
        result["obligations"]["score_accounting_independent_of_answer"]["status"]
        == "partially_covered"
    )


def test_legality_inferred_vocab_marks_full_alphabet_check_as_partial() -> None:
    module = _load_demo_adapter_module()
    runner = module.build_adapter({"mode": "legal", "vocab_size": 8})
    tokens = np.array([0, 1, 2, 1, 0, 3, 2, 1], dtype=np.int64)

    result = audit_parameter_golf_legality(
        runner,
        tokens,
        chunk_size=8,
        max_chunks=1,
        sample_chunks=1,
        future_probes_per_chunk=1,
        answer_probes_per_chunk=1,
        positions_per_future_probe=2,
        seed=0,
        vocab_size=None,
    )

    assert result["vocab_size_source"] == "inferred_from_tokens"
    assert result["checks"]["normalization"]["pass"] is False
    normalization_probe = next(row for row in result["probes"] if row["kind"] == "normalization")
    assert normalization_probe["expected_vocab_size"] == 4
    assert normalization_probe["observed_sizes"] == [8]
    assert normalization_probe["wrong_length_count"] > 0


def test_legality_catches_hidden_gold_logprob_cheat() -> None:
    module = _load_demo_adapter_module()
    runner = module.build_adapter({"mode": "reported_gold_cheat", "vocab_size": 8})
    tokens = np.array([0, 1, 2, 1, 0, 3, 2, 1], dtype=np.int64)

    result = audit_parameter_golf_legality(
        runner,
        tokens,
        chunk_size=8,
        max_chunks=1,
        sample_chunks=1,
        future_probes_per_chunk=1,
        answer_probes_per_chunk=1,
        positions_per_future_probe=2,
        seed=0,
        vocab_size=8,
    )

    assert result["checks"]["gold_logprob_consistency"]["covered"] is True
    assert result["checks"]["gold_logprob_consistency"]["pass"] is False
    assert any(row["kind"] == "gold_logprob_consistency" and not row["pass"] for row in result["probes"])


def test_legality_catches_self_include_leak() -> None:
    module = _load_demo_adapter_module()
    runner = module.build_adapter({"mode": "self_include", "vocab_size": 8})
    tokens = np.array([0, 1, 2, 1, 0, 3, 2, 1], dtype=np.int64)

    result = audit_parameter_golf_legality(
        runner,
        tokens,
        chunk_size=8,
        max_chunks=1,
        sample_chunks=1,
        future_probes_per_chunk=1,
        answer_probes_per_chunk=2,
        positions_per_future_probe=2,
        seed=1,
        vocab_size=8,
    )

    assert result["checks"]["answer_mask_invariance"]["covered"] is True
    assert result["checks"]["answer_mask_invariance"]["pass"] is False

