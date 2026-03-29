from __future__ import annotations

import pickle
import zlib
from pathlib import Path

import numpy as np

from conker_detect.audit import (
    audit_artifact,
    audit_matrix,
    compare_bundles,
    spectral_stats,
)


def test_spectral_stats_uses_effective_topk_keys() -> None:
    matrix = np.eye(3, dtype=np.float64)
    stats = spectral_stats(matrix, topk=16)

    assert stats["requested_topk"] == 16
    assert stats["effective_topk"] == 3
    assert "sigma3" in stats
    assert "top3_energy_frac" in stats
    assert "sigma16" not in stats


def test_audit_matrix_only_adds_mask_geometry_when_explicit() -> None:
    matrix = np.tril(np.ones((3, 3), dtype=np.float64), k=-1)

    result = audit_matrix(matrix, name="causal_mask", topk=4, expect_causal_mask=False)

    assert "mask_geometry" not in result


def test_compare_bundles_reports_shape_mismatches(tmp_path: Path) -> None:
    lhs = tmp_path / "lhs.npz"
    rhs = tmp_path / "rhs.npz"
    np.savez(lhs, shared=np.ones((2, 2), dtype=np.float64), mismatch=np.ones((2, 3), dtype=np.float64))
    np.savez(rhs, shared=np.ones((2, 2), dtype=np.float64), mismatch=np.ones((3, 2), dtype=np.float64))

    result = compare_bundles(lhs, rhs)

    assert result["shared_tensor_count"] == 2
    assert result["shape_mismatches"] == ["mismatch"]
    mismatch_row = next(row for row in result["tensors"] if row["name"] == "mismatch")
    assert mismatch_row["shape_mismatch"] is True
    assert "compare" not in mismatch_row


def test_audit_artifact_flags_boundary_entries(tmp_path: Path) -> None:
    artifact = tmp_path / "model.int6.ptz"
    payload = {
        "linear_kernel": {
            "type": "fp16",
            "data": np.ones((2, 2), dtype=np.float16),
        },
        "causal_mask": {
            "type": "raw",
            "data": np.tril(np.ones((3, 3), dtype=np.float32), k=-1),
        },
        "payload_weight": {
            "type": "quant",
            "q": np.ones((4,), dtype=np.int8),
            "scale": np.ones((1,), dtype=np.float16),
        },
    }
    artifact.write_bytes(zlib.compress(pickle.dumps(payload)))

    result = audit_artifact(artifact)

    assert result["entry_count"] == 3
    assert result["class_counts"]["deterministic_substrate"] == 1
    assert result["class_counts"]["structural_control"] == 1
    assert result["class_counts"]["learned_payload"] == 1
    assert len(result["alerts"]) == 2

