from __future__ import annotations

import argparse
import json
import os
import pickle
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "conker_detect.cli", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_handoff_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "records" / "track_non_record_16mb" / "demo_submission"
    run_dir.mkdir(parents=True)
    (run_dir / "README.md").write_text(
        "# Demo Submission\nval_bpb: 0.57546632\npre_quant_val_bpb: 0.57180453\nartifact bytes: 5\n",
        encoding="utf-8",
    )
    (run_dir / "submission.json").write_text(
        json.dumps(
            {
                "name": "Demo Submission",
                "track": "track_non_record_16mb",
                "val_bpb": 0.57546632,
                "pre_quant_val_bpb": 0.57180453,
                "bytes_model_int6_zlib": 5,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.json").write_text(
        json.dumps({"val_bpb": 0.57546632, "pre_quant_val_bpb": 0.57180453, "bytes_model_int6_zlib": 5}),
        encoding="utf-8",
    )
    (run_dir / "train.log").write_text("val_bpb=0.57546632\n", encoding="utf-8")
    (run_dir / "train_gpt.py").write_text("def run():\n    return None\n", encoding="utf-8")
    (run_dir / "model.int6.ptz").write_bytes(b"12345")
    return run_dir


def _save_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    save_file = pytest.importorskip("safetensors.numpy").save_file
    save_file(tensors, str(path))


def _truncate_after_first_tensor(source: Path, partial: Path) -> None:
    raw = source.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    payload = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    entries = [(name, meta) for name, meta in payload.items() if name != "__metadata__"]
    first_end = 8 + header_len + int(entries[0][1]["data_offsets"][1])
    partial.write_bytes(raw[:first_end])


def _available_commands() -> set[str]:
    from conker_detect.cli import build_parser

    parser = build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    return set()


def _trigger_command_available(name: str) -> bool:
    return name in _available_commands()


def test_legality_cli_writes_json(tmp_path: Path) -> None:
    tokens_path = tmp_path / "tokens.npy"
    out_path = tmp_path / "legality.json"
    np.save(tokens_path, np.array([0, 1, 2, 1, 0, 3, 2, 1], dtype=np.int64))

    result = _run_cli(
        "legality",
        "--profile",
        "parameter-golf",
        "--trust-level",
        "strict",
        "--adapter",
        "examples/packed_cache_demo_adapter.py",
        "--adapter-config",
        '{"mode":"legal","vocab_size":8}',
        "--tokens",
        str(tokens_path),
        "--chunk-size",
        "8",
        "--max-chunks",
        "1",
        "--sample-chunks",
        "1",
        "--future-probes-per-chunk",
        "1",
        "--answer-probes-per-chunk",
        "1",
        "--positions-per-future-probe",
        "2",
        "--vocab-size",
        "8",
        "--json",
        str(out_path),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["profile"] == "parameter-golf"
    assert payload["checks"]["normalization"]["pass"] is True
    assert payload["trust"]["requested"] == "strict"
    assert payload["trust"]["satisfied"] is True


def test_artifact_cli_writes_json(tmp_path: Path) -> None:
    artifact_path = tmp_path / "model.int6.ptz"
    out_path = tmp_path / "artifact.json"
    payload = {
        "linear_kernel": {
            "type": "fp16",
            "data": np.ones((2, 2), dtype=np.float16),
        },
        "causal_mask": {
            "type": "raw",
            "data": np.tril(np.ones((3, 3), dtype=np.float32), k=-1),
        },
    }
    artifact_path.write_bytes(zlib.compress(pickle.dumps(payload)))

    result = _run_cli("artifact", str(artifact_path), "--json", str(out_path))

    assert result.returncode == 0, result.stderr
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["entry_count"] == 2
    assert "alerts" in report


def test_bundle_cli_reads_safetensors_file(tmp_path: Path) -> None:
    bundle_path = tmp_path / "model.safetensors"
    out_path = tmp_path / "bundle.json"
    _save_safetensors(
        bundle_path,
        {
            "model.layers.0.mlp.down_proj.weight": np.eye(2, dtype=np.float32),
            "model.layers.0.input_layernorm.weight": np.ones((2,), dtype=np.float32),
        },
    )

    result = _run_cli("bundle", str(bundle_path), "--name-regex", "down_proj", "--json", str(out_path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["tensor_count"] == 1
    assert payload["tensors"][0]["name"] == "model.layers.0.mlp.down_proj.weight"


def test_catalog_cli_reports_partial_safetensors_slice(tmp_path: Path) -> None:
    source = tmp_path / "full.safetensors"
    partial = tmp_path / "partial.safetensors"
    out_path = tmp_path / "catalog.json"
    _save_safetensors(
        source,
        {
            "a_first": np.eye(2, dtype=np.float32),
            "z_last": np.ones((4, 4), dtype=np.float32),
        },
    )
    _truncate_after_first_tensor(source, partial)

    result = _run_cli("catalog", str(partial), "--json", str(out_path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["tensor_count"] == 2
    assert payload["complete_tensor_count"] == 1
    assert payload["tensors"][0]["complete"] is True
    assert payload["tensors"][1]["complete"] is False


def test_carve_cli_writes_probeable_safetensors_bundle(tmp_path: Path) -> None:
    source = tmp_path / "full.safetensors"
    partial = tmp_path / "partial.bin"
    carved = tmp_path / "carved.safetensors"
    carve_report = tmp_path / "carve.json"
    bundle_report = tmp_path / "bundle.json"
    _save_safetensors(
        source,
        {
            "a_first": np.eye(2, dtype=np.float32),
            "z_last": np.ones((4, 4), dtype=np.float32),
        },
    )
    _truncate_after_first_tensor(source, partial)

    carve_result = _run_cli("carve", str(partial), str(carved), "--json", str(carve_report))
    bundle_result = _run_cli("bundle", str(carved), "--json", str(bundle_report))

    assert carve_result.returncode == 0, carve_result.stderr
    assert bundle_result.returncode == 0, bundle_result.stderr
    carve_payload = json.loads(carve_report.read_text(encoding="utf-8"))
    bundle_payload = json.loads(bundle_report.read_text(encoding="utf-8"))
    assert carve_payload["written_tensor_count"] == 1
    assert bundle_payload["tensor_count"] == 1
    assert bundle_payload["tensors"][0]["name"] == "a_first"


def test_family_cli_summarizes_projection_stems(tmp_path: Path) -> None:
    bundle_path = tmp_path / "model.safetensors"
    out_path = tmp_path / "family.json"
    _save_safetensors(
        bundle_path,
        {
            "model.layers.0.mlp.down_proj.weight": np.eye(2, dtype=np.float32),
            "model.layers.0.mlp.down_proj.weight_scale_inv": np.ones((2, 2), dtype=np.float32),
            "model.layers.0.mlp.gate.weight": np.ones((2, 2), dtype=np.float32),
        },
    )

    result = _run_cli("family", str(bundle_path), "--json", str(out_path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["family_count"] == 2
    families = {row["family"]: row for row in payload["families"]}
    assert families["model.layers.0.mlp.down_proj"]["tensor_count"] == 2
    assert families["model.layers.0.mlp.gate"]["tensor_count"] == 1


def test_submission_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    submission_root = repo_root / "records" / "track_non_record_16mb" / "demo_submission"
    submission_root.mkdir(parents=True)
    (submission_root / "README.md").write_text(
        "# Demo Submission\nval_bpb: 0.57546632\nartifact bytes: 5\n",
        encoding="utf-8",
    )
    (submission_root / "submission.json").write_text(
        json.dumps(
            {
                "name": "Demo Submission",
                "track": "track_non_record_16mb",
                "val_bpb": 0.57546632,
                "bytes_model_int6_zlib": 5,
            }
        ),
        encoding="utf-8",
    )
    (submission_root / "model.int6.ptz").write_bytes(b"12345")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "profile": "parameter-golf",
                "repo_root": str(repo_root),
                "submission_root": "records/track_non_record_16mb/demo_submission",
                "evidence": {
                    "readme": "README.md",
                    "submission_json": "submission.json",
                    "artifacts": ["model.int6.ptz"],
                },
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "submission.json"
    md_path = tmp_path / "submission.md"

    result = _run_cli("submission", str(manifest_path), "--json", str(out_path), "--md", str(md_path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["profile"] == "parameter-golf"
    assert payload["checks"]["presence"]["pass"] is True
    assert md_path.exists()
    assert "Submission Audit" in md_path.read_text(encoding="utf-8")


def test_provenance_cli_writes_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "provenance_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "profile": "parameter-golf",
                "selection": {
                    "submitted_run_id": "run-1",
                    "selection_mode": "single_run",
                    "candidate_run_count": 1,
                },
                "datasets": {
                    "train": {"name": "train", "fingerprint": "train-a"},
                    "validation": {"name": "val", "fingerprint": "val-a"},
                    "held_out_test": {"name": "test", "fingerprint": "test-a"},
                },
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "provenance.json"

    result = _run_cli("provenance", str(manifest_path), "--json", str(out_path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["verdict"] == "pass"
    assert payload["checks"]["selection_disclosure"]["pass"] is True


def test_replay_cli_writes_json(tmp_path: Path) -> None:
    tokens_path = tmp_path / "tokens.npy"
    out_path = tmp_path / "replay.json"
    np.save(tokens_path, np.array([0, 1, 2, 1, 0, 3, 2, 1], dtype=np.int64))

    result = _run_cli(
        "replay",
        "--profile",
        "parameter-golf",
        "--adapter",
        "examples/packed_cache_demo_adapter.py",
        "--adapter-config",
        '{"mode":"legal","vocab_size":8}',
        "--tokens",
        str(tokens_path),
        "--chunk-size",
        "4",
        "--max-chunks",
        "2",
        "--sample-chunks",
        "2",
        "--position-batch-size",
        "2",
        "--json",
        str(out_path),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["profile"] == "parameter-golf"
    assert payload["repeatability"]["covered"] is True
    assert payload["aggregate"]["mean_bpb"] is not None


def test_ledger_manifest_cli_writes_bundle_manifest(tmp_path: Path) -> None:
    out_path = tmp_path / "bundle_manifest.json"

    result = _run_cli(
        "ledger-manifest",
        str(out_path),
        "--bundle-id",
        "demo-bundle",
        "--submission-report",
        "out/submission.json",
        "--provenance-report",
        "out/provenance.json",
        "--legality-report",
        "out/legality.json",
        "--replay-report",
        "out/replay.json",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["bundle_id"] == "demo-bundle"
    assert payload["attachments"][0]["dest"] == "audits/tier1/submission.json"
    assert payload["attachments"][-1]["dest"] == "audits/tier3/replay.json"


def test_handoff_cli_writes_synthesized_bundle(tmp_path: Path) -> None:
    run_dir = _write_handoff_run_dir(tmp_path)
    out_dir = tmp_path / "handoff"
    out_path = tmp_path / "handoff_result.json"

    result = _run_cli(
        "handoff",
        str(run_dir),
        str(out_dir),
        "--bundle-id",
        "demo-bundle",
        "--json",
        str(out_path),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["bundle_id"] == "demo-bundle"
    assert (out_dir / "reports" / "submission.json").exists()
    assert (out_dir / "ledger_manifest.json").exists()
    submission = json.loads((out_dir / "reports" / "submission.json").read_text(encoding="utf-8"))
    audits = json.loads((out_dir / "audits.json").read_text(encoding="utf-8"))
    assert audits["tier1"]["status"] == "pass"
    assert submission["submission"]["repo_root"] == str(tmp_path)
    assert submission["submission"]["submission_root"] == str(run_dir)


def test_handoff_cli_propagates_legality_trust(tmp_path: Path) -> None:
    run_dir = _write_handoff_run_dir(tmp_path)
    out_dir = tmp_path / "handoff"
    out_path = tmp_path / "handoff_runtime_result.json"
    tokens_path = tmp_path / "tokens.npy"
    np.save(tokens_path, np.array([0, 1, 2, 1, 0, 3, 2, 1], dtype=np.int64))

    result = _run_cli(
        "handoff",
        str(run_dir),
        str(out_dir),
        "--bundle-id",
        "demo-bundle",
        "--adapter",
        "examples/packed_cache_demo_adapter.py",
        "--adapter-config",
        '{"mode":"distribution_only","vocab_size":8}',
        "--tokens",
        str(tokens_path),
        "--trust-level",
        "traced",
        "--vocab-size",
        "8",
        "--chunk-size",
        "8",
        "--max-chunks",
        "1",
        "--sample-chunks",
        "1",
        "--future-probes-per-chunk",
        "1",
        "--answer-probes-per-chunk",
        "1",
        "--positions-per-future-probe",
        "2",
        "--json",
        str(out_path),
    )

    assert result.returncode == 0, result.stderr
    audits = json.loads((out_dir / "audits.json").read_text(encoding="utf-8"))
    legality = json.loads((out_dir / "reports" / "legality.json").read_text(encoding="utf-8"))
    assert legality["trust"]["requested"] == "traced"
    assert legality["trust"]["achieved"] == "basic"
    assert legality["trust"]["satisfied"] is False
    assert audits["tier3"]["trust_level_requested"] == "traced"
    assert audits["tier3"]["trust_level_achieved"] == "basic"
    assert audits["tier3"]["trust_satisfied"] is False


def test_chatdiff_cli_writes_json_when_available(tmp_path: Path) -> None:
    if not _trigger_command_available("chatdiff"):
        pytest.skip("trigger CLI not implemented yet")

    provider = FIXTURES / "trigger_provider.py"
    lhs = FIXTURES / "trigger_case_lhs.json"
    rhs = FIXTURES / "trigger_case_rhs.json"
    out_path = tmp_path / "chatdiff.json"

    result = _run_cli(
        "chatdiff",
        "--provider",
        str(provider),
        "--provider-config",
        "{\"temperature\":0}",
        "--lhs",
        str(lhs),
        "--rhs",
        str(rhs),
        "--model",
        "demo-model",
        "--json",
        str(out_path),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert "completion::demo-model::" in serialized
    assert "Compare the red fox and the blue fox." in serialized
    assert "Compare the red fox and the green fox." in serialized


def test_actdiff_cli_writes_json_when_available(tmp_path: Path) -> None:
    if not _trigger_command_available("actdiff"):
        pytest.skip("trigger CLI not implemented yet")

    provider = FIXTURES / "trigger_provider.py"
    lhs = FIXTURES / "trigger_case_lhs.json"
    rhs = FIXTURES / "trigger_case_rhs.json"
    out_path = tmp_path / "actdiff.json"

    result = _run_cli(
        "actdiff",
        "--provider",
        str(provider),
        "--provider-config",
        str(FIXTURES / "trigger_provider_config.json"),
        "--lhs",
        str(lhs),
        "--rhs",
        str(rhs),
        "--model",
        "demo-model",
        "--json",
        str(out_path),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert "layer.0" in serialized
    assert "layer.1" in serialized
    assert "layer.2" in serialized
    assert "cosine" in serialized
    assert "l2" in serialized
    assert "max_abs" in serialized


def test_crossmodel_cli_writes_json_when_available(tmp_path: Path) -> None:
    if not _trigger_command_available("crossmodel"):
        pytest.skip("trigger CLI not implemented yet")

    provider = FIXTURES / "trigger_provider.py"
    case = FIXTURES / "trigger_case_shared.json"
    out_path = tmp_path / "crossmodel.json"

    result = _run_cli(
        "crossmodel",
        "--provider",
        str(provider),
        "--provider-config",
        "{\"temperature\":0}",
        "--case",
        str(case),
        "--model",
        "model-a",
        "--model",
        "model-b",
        "--model",
        "model-c",
        "--json",
        str(out_path),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert "model-a" in serialized
    assert "model-b" in serialized
    assert "model-c" in serialized
    assert "chat" in serialized or "completion" in serialized
    assert "activation" in serialized or "cosine" in serialized


def test_mutate_cli_writes_json_when_available(tmp_path: Path) -> None:
    if not _trigger_command_available("mutate"):
        pytest.skip("trigger mutate CLI not implemented yet")

    case = FIXTURES / "trigger_case_shared.json"
    out_path = tmp_path / "mutate.json"

    result = _run_cli(
        "mutate",
        str(case),
        "--family",
        "quoted",
        "--family",
        "code_fence",
        "--json",
        str(out_path),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["variant_count"] == 2
    serialized = json.dumps(payload)
    assert "quoted" in serialized
    assert "code_fence" in serialized


def test_sweep_cli_writes_json_when_available(tmp_path: Path) -> None:
    if not _trigger_command_available("sweep"):
        pytest.skip("trigger sweep CLI not implemented yet")

    provider = FIXTURES / "trigger_provider.py"
    case = FIXTURES / "trigger_case_shared.json"
    out_path = tmp_path / "sweep.json"

    result = _run_cli(
        "sweep",
        "--provider",
        str(provider),
        "--provider-config",
        str(FIXTURES / "trigger_provider_config.json"),
        "--case",
        str(case),
        "--model",
        "model-a",
        "--model",
        "model-b",
        "--family",
        "quoted",
        "--family",
        "uppercase",
        "--json",
        str(out_path),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["variant_count"] == 2
    serialized = json.dumps(payload)
    assert "combined_score" in serialized
    assert "quoted" in serialized
    assert "uppercase" in serialized


def test_minimize_cli_writes_json_when_available(tmp_path: Path) -> None:
    if not _trigger_command_available("minimize"):
        pytest.skip("trigger minimize CLI not implemented yet")

    provider = FIXTURES / "trigger_provider.py"
    control = FIXTURES / "trigger_case_lhs.json"
    candidate = FIXTURES / "trigger_case_rhs.json"
    out_path = tmp_path / "minimize.json"

    result = _run_cli(
        "minimize",
        "--provider",
        str(provider),
        "--provider-config",
        "{\"temperature\":0}",
        "--control",
        str(control),
        "--candidate",
        str(candidate),
        "--model",
        "demo-model",
        "--metric",
        "chat",
        "--json",
        str(out_path),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["minimized_token_count"] <= payload["original_token_count"]
    assert payload["final_score"] > 0.0


def test_attack_cli_writes_json_when_available(tmp_path: Path) -> None:
    if not _trigger_command_available("attack"):
        pytest.skip("trigger attack CLI not implemented yet")

    provider = FIXTURES / "trigger_provider.py"
    campaign = FIXTURES / "trigger_attack_campaign.json"
    out_path = tmp_path / "attack.json"

    result = _run_cli(
        "attack",
        "--provider",
        str(provider),
        "--provider-config",
        str(FIXTURES / "trigger_provider_config.json"),
        str(campaign),
        "--json",
        str(out_path),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["top_candidates"]
    assert payload["minimizations"]
    serialized = json.dumps(payload)
    assert "mixed_family" in serialized or "single_family" in serialized
