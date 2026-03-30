from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from conker_detect.handoff import prepare_ledger_handoff


def _write_run_dir(tmp_path: Path) -> Path:
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


def test_prepare_ledger_handoff_without_runtime_reports(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path)
    out_dir = tmp_path / "handoff"

    result = prepare_ledger_handoff(run_dir, out_dir, bundle_id="demo-bundle")

    assert result["bundle_id"] == "demo-bundle"
    assert (out_dir / "reports" / "submission.json").exists()
    assert (out_dir / "claim.json").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "provenance.json").exists()
    assert (out_dir / "audits.json").exists()
    manifest = json.loads((out_dir / "ledger_manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle_id"] == "demo-bundle"
    assert manifest["attachments"][0]["dest"] == "audits/tier1/submission.json"
    audits = json.loads((out_dir / "audits.json").read_text(encoding="utf-8"))
    assert audits["tier1"]["status"] == "pass"
    assert "tier3" not in audits


def test_prepare_ledger_handoff_with_provenance_legality_and_replay(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path)
    out_dir = tmp_path / "handoff"
    provenance_path = tmp_path / "provenance_manifest.json"
    provenance_path.write_text(
        json.dumps(
            {
                "profile": "parameter-golf",
                "selection": {
                    "submitted_run_id": "run-43",
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
    tokens_path = tmp_path / "tokens.npy"
    np.save(tokens_path, np.array([0, 1, 2, 1, 0, 3, 2, 1], dtype=np.int64))

    result = prepare_ledger_handoff(
        run_dir,
        out_dir,
        bundle_id="demo-bundle",
        provenance_source=provenance_path,
        adapter_ref="examples/packed_cache_demo_adapter.py",
        adapter_config_raw='{"mode":"legal","vocab_size":8}',
        tokens_path=tokens_path,
        chunk_size=4,
        max_chunks=2,
        sample_chunks=2,
        future_probes_per_chunk=1,
        answer_probes_per_chunk=1,
        positions_per_future_probe=2,
        position_batch_size=2,
        vocab_size=8,
    )

    assert (out_dir / "reports" / "provenance.json").exists()
    assert (out_dir / "reports" / "legality.json").exists()
    assert (out_dir / "reports" / "replay.json").exists()
    manifest = json.loads((out_dir / "ledger_manifest.json").read_text(encoding="utf-8"))
    dests = {row["dest"] for row in manifest["attachments"]}
    assert "audits/tier1/provenance.json" in dests
    assert "audits/tier3/legality.json" in dests
    assert "audits/tier3/replay.json" in dests
    audits = json.loads((out_dir / "audits.json").read_text(encoding="utf-8"))
    assert audits["tier1"]["status"] == "pass"
    assert audits["tier3"]["status"] == "warn"
    assert audits["tier3"]["scope"] == "one_shot_runtime_handoff"
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["replay"]["mean_bpb"] is not None
