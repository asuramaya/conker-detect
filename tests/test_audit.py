from __future__ import annotations

import json
import pickle
import zlib
from pathlib import Path

import numpy as np
import pytest

from conker_detect.audit import (
    audit_artifact,
    audit_bundle,
    audit_matrix,
    compare_bundles,
    spectral_stats,
)


def _save_file(path: Path, tensors: dict[str, np.ndarray]) -> None:
    save_file = pytest.importorskip("safetensors.numpy").save_file
    save_file(tensors, str(path))


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


def test_audit_bundle_reads_single_safetensors_file(tmp_path: Path) -> None:
    bundle = tmp_path / "model.safetensors"
    _save_file(
        bundle,
        {
            "model.layers.0.mlp.down_proj": np.eye(2, dtype=np.float32),
            "model.layers.0.input_layernorm.weight": np.ones((2,), dtype=np.float32),
        },
    )

    result = audit_bundle(bundle, name_regex="down_proj")

    assert result["tensor_count"] == 1
    assert result["tensors"][0]["name"] == "model.layers.0.mlp.down_proj"


def test_compare_bundles_reads_sharded_safetensors_repo(tmp_path: Path) -> None:
    lhs = tmp_path / "lhs"
    rhs = tmp_path / "rhs"
    _write_sharded_safetensors_repo(
        lhs,
        {
            "model-00001-of-00002.safetensors": {
                "model.layers.0.self_attn.q_proj.weight": np.eye(2, dtype=np.float32),
            },
            "model-00002-of-00002.safetensors": {
                "model.layers.0.mlp.down_proj.weight": np.ones((2, 2), dtype=np.float32),
            },
        },
    )
    _write_sharded_safetensors_repo(
        rhs,
        {
            "model-00001-of-00002.safetensors": {
                "model.layers.0.self_attn.q_proj.weight": np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            },
            "model-00002-of-00002.safetensors": {
                "model.layers.0.mlp.down_proj.weight": np.ones((2, 2), dtype=np.float32),
            },
        },
    )

    result = compare_bundles(lhs, rhs, name_regex="proj")

    assert result["shared_tensor_count"] == 2
    q_proj_row = next(row for row in result["tensors"] if row["name"] == "model.layers.0.self_attn.q_proj.weight")
    assert q_proj_row["compare"]["max_abs_deviation"] == 1.0


def _write_sharded_safetensors_repo(repo_root: Path, shards: dict[str, dict[str, np.ndarray]]) -> None:
    repo_root.mkdir(parents=True)
    weight_map: dict[str, str] = {}
    for shard_name, tensors in shards.items():
        _save_file(repo_root / shard_name, tensors)
        for tensor_name in tensors:
            weight_map[tensor_name] = shard_name
    (repo_root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2) + "\n",
        encoding="utf-8",
    )
