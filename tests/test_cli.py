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
