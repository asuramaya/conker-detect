from __future__ import annotations

import json
from pathlib import Path

from conker_detect.provenance import audit_provenance


def _provenance_manifest() -> dict[str, object]:
    return {
        "profile": "parameter-golf",
        "selection": {
            "submitted_run_id": "run-43",
            "selection_mode": "single_run",
            "candidate_run_count": 1,
        },
        "datasets": {
            "train": {"name": "fineweb_train", "fingerprint": "train-sha"},
            "validation": {"name": "fineweb_val", "fingerprint": "val-sha"},
            "held_out_test": {"name": "fineweb_test", "fingerprint": "test-sha"},
        },
    }


def test_provenance_audit_passes_complete_manifest() -> None:
    result = audit_provenance(_provenance_manifest())

    assert result["verdict"] == "pass"
    assert result["checks"]["selection_disclosure"]["pass"] is True
    assert result["checks"]["dataset_fingerprints"]["pass"] is True
    assert result["checks"]["held_out_identity"]["pass"] is True


def test_provenance_warns_on_best_of_k_selection_disclosure() -> None:
    manifest = _provenance_manifest()
    manifest["selection"] = {
        "submitted_run_id": "run-50",
        "selection_mode": "best_of_k",
        "candidate_run_count": 50,
    }

    result = audit_provenance(manifest)

    assert result["verdict"] == "warn"
    assert result["checks"]["selection_disclosure"]["pass"] is True
    assert any(row["kind"] == "selection_bias_risk" for row in result["findings"])


def test_provenance_fails_on_dataset_fingerprint_overlap() -> None:
    manifest = _provenance_manifest()
    manifest["datasets"]["validation"] = {"name": "fineweb_val", "fingerprint": "train-sha"}

    result = audit_provenance(manifest)

    assert result["verdict"] == "fail"
    assert result["checks"]["dataset_fingerprints"]["pass"] is False
    assert any(row["kind"] == "dataset_fingerprint_overlap" for row in result["findings"])


def test_provenance_warns_on_missing_held_out_identity_and_can_load_from_path(tmp_path: Path) -> None:
    manifest = _provenance_manifest()
    manifest["datasets"].pop("held_out_test")
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = audit_provenance(path)

    assert result["verdict"] == "warn"
    assert result["checks"]["held_out_identity"]["pass"] is True
    assert any(row["kind"] == "missing_held_out_identity" for row in result["findings"])
