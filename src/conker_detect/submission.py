from __future__ import annotations

from pathlib import Path
from typing import Any

from .submission_checks import (
    check_artifact_bytes,
    check_claim_consistency,
    check_data_boundary_signals,
    check_patch_triage,
    check_presence,
    check_protocol_shape,
    check_reproducibility_surface,
)
from .submission_extract import extract_claims, load_submission_manifest


def audit_submission(source: Path | str | dict[str, Any]) -> dict[str, Any]:
    manifest = load_submission_manifest(source)
    extracted = extract_claims(manifest)

    check_rows = {
        "presence": check_presence(manifest),
        "claim_consistency": check_claim_consistency(manifest, extracted),
        "artifact_bytes": check_artifact_bytes(manifest, extracted),
        "protocol_shape": check_protocol_shape(manifest),
        "data_boundary_signals": check_data_boundary_signals(manifest),
        "reproducibility_surface": check_reproducibility_surface(manifest, extracted),
        "patch_triage": check_patch_triage(manifest),
    }

    checks: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []
    for name, (summary, rows) in check_rows.items():
        checks[name] = summary
        findings.extend(rows)

    alerts = [row["message"] for row in findings if row["severity"] in {"warn", "fail"}]
    return {
        "profile": manifest["profile"],
        "verdict": _verdict(findings),
        "submission": {
            "repo_root": str(manifest["repo_root"]),
            "submission_root": str(manifest["submission_root"]),
            "track": manifest.get("claim_overrides", {}).get("track"),
            "name": extracted.get("submission_json", {}).get("name")
            or extracted.get("readme", {}).get("name"),
        },
        "extracted_claims": extracted,
        "checks": checks,
        "findings": findings,
        "alerts": alerts,
    }


def _verdict(findings: list[dict[str, Any]]) -> str:
    severities = {row["severity"] for row in findings}
    if "fail" in severities:
        return "fail"
    if "warn" in severities:
        return "warn"
    return "pass"
