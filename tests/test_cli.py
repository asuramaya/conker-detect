from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


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


def test_legality_cli_writes_json(tmp_path: Path) -> None:
    tokens_path = tmp_path / "tokens.npy"
    out_path = tmp_path / "legality.json"
    np.save(tokens_path, np.array([0, 1, 2, 1, 0, 3, 2, 1], dtype=np.int64))

    result = _run_cli(
        "legality",
        "--profile",
        "parameter-golf",
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
