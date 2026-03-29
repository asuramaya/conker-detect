from __future__ import annotations

import json
from pathlib import Path

from conker_detect.submission import audit_submission


def _write_submission_fixture(
    tmp_path: Path,
    *,
    readme_val_bpb: float = 0.57546632,
    submission_val_bpb: float = 0.57546632,
    results_val_bpb: float = 0.57546632,
    artifact_bytes: int = 123,
    include_optimizer_signal: bool = False,
    include_network_signal: bool = False,
) -> dict[str, object]:
    repo_root = tmp_path / "repo"
    submission_root = repo_root / "records" / "track_non_record_16mb" / "demo_submission"
    submission_root.mkdir(parents=True)
    artifact_path = submission_root / "model.int6.ptz"
    artifact_path.write_bytes(b"x" * artifact_bytes)
    readme = submission_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Demo Submission",
                f"val_bpb: {readme_val_bpb}",
                "pre_quant_val_bpb: 0.57180453",
                f"artifact bytes: {artifact_bytes}",
            ]
        ),
        encoding="utf-8",
    )
    submission_json = submission_root / "submission.json"
    submission_json.write_text(
        json.dumps(
            {
                "name": "Demo Submission",
                "track": "track_non_record_16mb",
                "val_bpb": submission_val_bpb,
                "pre_quant_val_bpb": 0.57180453,
                "bytes_total": artifact_bytes + 10,
                "bytes_model_int6_zlib": artifact_bytes,
            }
        ),
        encoding="utf-8",
    )
    results_json = submission_root / "results.json"
    results_json.write_text(
        json.dumps(
            {
                "val_bpb": results_val_bpb,
                "pre_quant_val_bpb": 0.57180453,
                "bytes_model_int6_zlib": artifact_bytes,
            }
        ),
        encoding="utf-8",
    )
    train_log = submission_root / "train.log"
    train_log.write_text(f"val_bpb={results_val_bpb}\n", encoding="utf-8")
    train_code = submission_root / "train_gpt.py"
    code_lines = ["def run():", "    return None"]
    if include_optimizer_signal:
        code_lines.append("optimizer.step()")
    if include_network_signal:
        code_lines.append("print('https://example.com')")
    train_code.write_text("\n".join(code_lines) + "\n", encoding="utf-8")
    patch_path = tmp_path / "demo.patch"
    patch_path.write_text(
        "\n".join(
            [
                "diff --git a/records/track_non_record_16mb/demo_submission/submission.json b/records/track_non_record_16mb/demo_submission/submission.json",
                "+++ b/records/track_non_record_16mb/demo_submission/submission.json",
                "diff --git a/train_gpt.py b/train_gpt.py",
                "+++ b/train_gpt.py",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "profile": "parameter-golf",
        "repo_root": str(repo_root),
        "submission_root": str(submission_root.relative_to(repo_root)),
        "evidence": {
            "readme": "README.md",
            "submission_json": "submission.json",
            "results_json": "results.json",
            "logs": ["train.log"],
            "artifacts": ["model.int6.ptz"],
            "code": ["train_gpt.py"],
            "patch": str(patch_path),
        },
        "claim_overrides": {"track": "track_non_record_16mb"},
    }


def test_submission_audit_passes_consistent_minimal_fixture(tmp_path: Path) -> None:
    manifest = _write_submission_fixture(tmp_path)

    result = audit_submission(manifest)

    assert result["verdict"] == "pass"
    assert result["checks"]["presence"]["pass"] is True
    assert result["checks"]["claim_consistency"]["pass"] is True
    assert result["checks"]["artifact_bytes"]["pass"] is True
    assert result["checks"]["patch_triage"]["category"] == "core_code"


def test_submission_audit_flags_claim_mismatch(tmp_path: Path) -> None:
    manifest = _write_submission_fixture(tmp_path, readme_val_bpb=0.6)

    result = audit_submission(manifest)

    assert result["verdict"] == "warn"
    assert result["checks"]["claim_consistency"]["finding_count"] > 0
    assert any(row["kind"] == "claim_mismatch" for row in result["findings"])


def test_submission_audit_flags_artifact_size_mismatch(tmp_path: Path) -> None:
    manifest = _write_submission_fixture(tmp_path, artifact_bytes=123)
    submission_root = Path(manifest["repo_root"]) / Path(manifest["submission_root"])
    submission_json_path = submission_root / "submission.json"
    payload = json.loads(submission_json_path.read_text(encoding="utf-8"))
    payload["bytes_model_int6_zlib"] = 999
    submission_json_path.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_submission(manifest)

    assert result["verdict"] == "fail"
    assert result["checks"]["artifact_bytes"]["pass"] is False
    assert any(row["kind"] == "artifact_size_mismatch" for row in result["findings"])


def test_submission_audit_flags_protocol_and_data_boundary_signals(tmp_path: Path) -> None:
    manifest = _write_submission_fixture(
        tmp_path,
        include_optimizer_signal=True,
        include_network_signal=True,
    )

    result = audit_submission(manifest)

    assert result["verdict"] == "warn"
    assert result["checks"]["protocol_shape"]["pass"] is False
    assert result["checks"]["data_boundary_signals"]["pass"] is False
    kinds = {row["kind"] for row in result["findings"]}
    assert "optimizer_step" in kinds
    assert "network_call" in kinds
