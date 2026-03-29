from __future__ import annotations

from pathlib import Path
from typing import Any

from .submission_profiles import classify_patch_context, get_profile_rules, patch_files


def finding(
    severity: str,
    kind: str,
    message: str,
    *,
    path: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"severity": severity, "kind": kind, "message": message}
    if path is not None:
        row["path"] = path
    if details:
        row["details"] = details
    return row


def check_presence(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rules = get_profile_rules(manifest["profile"])
    evidence = manifest["evidence"]
    findings: list[dict[str, Any]] = []
    checked = 0
    for key in rules["required_evidence"]:
        checked += 1
        path = evidence.get(key)
        if path is None or not Path(path).exists():
            findings.append(finding("fail", "missing_evidence", f"Required evidence {key} is missing", path=str(path) if path else None))
    return {"pass": not findings, "checked_count": checked, "finding_count": len(findings)}, findings


def check_claim_consistency(manifest: dict[str, Any], extracted: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rules = get_profile_rules(manifest["profile"])
    findings: list[dict[str, Any]] = []
    sources = {
        "submission_json": extracted.get("submission_json", {}),
        "results_json": extracted.get("results_json", {}),
        "readme": extracted.get("readme", {}),
    }
    for log_path, row in extracted.get("logs", {}).items():
        sources[f"log:{log_path}"] = row
    for key in rules["claim_fields"]:
        values = {name: claims[key] for name, claims in sources.items() if key in claims}
        if len(values) <= 1:
            continue
        canonical_source = "submission_json" if "submission_json" in values else next(iter(values))
        canonical_value = values[canonical_source]
        for source_name, value in values.items():
            if source_name == canonical_source:
                continue
            if not _values_match(canonical_value, value):
                severity = "fail" if source_name == "results_json" else "warn"
                findings.append(
                    finding(
                        severity,
                        "claim_mismatch",
                        f"{key} differs between {canonical_source} and {source_name}",
                        details={
                            "field": key,
                            "lhs_source": canonical_source,
                            "lhs_value": canonical_value,
                            "rhs_source": source_name,
                            "rhs_value": value,
                        },
                    )
                )
    return {"pass": not any(row["severity"] == "fail" for row in findings), "finding_count": len(findings)}, findings


def check_artifact_bytes(manifest: dict[str, Any], extracted: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    actual_artifacts = extracted.get("artifact_files", {})
    actual_sizes = {path: row["bytes"] for path, row in actual_artifacts.items()}
    claimed_sources = (
        ("submission_json", extracted.get("submission_json", {}).get("bytes_model_int6_zlib")),
        ("results_json", extracted.get("results_json", {}).get("bytes_model_int6_zlib")),
        ("readme", extracted.get("readme", {}).get("bytes_model_int6_zlib")),
    )
    if len(actual_sizes) == 1:
        actual_path, actual_size = next(iter(actual_sizes.items()))
        for source_name, claimed in claimed_sources:
            if claimed is None:
                continue
            if int(claimed) != int(actual_size):
                findings.append(
                    finding(
                        "fail",
                        "artifact_size_mismatch",
                        f"Artifact bytes claimed by {source_name} do not match the actual artifact size",
                        path=actual_path,
                        details={"source": source_name, "claimed": int(claimed), "actual": int(actual_size)},
                    )
                )
    return {"pass": not findings if actual_sizes else None, "artifact_count": len(actual_sizes), "finding_count": len(findings)}, findings


def check_protocol_shape(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rules = get_profile_rules(manifest["profile"])
    findings: list[dict[str, Any]] = []
    scanned = 0
    for path in _iter_scan_paths(manifest, include_patch=True):
        text = path.read_text(encoding="utf-8", errors="replace")
        lowered = text.lower()
        scanned += 1
        for marker, kind in rules["protocol_risk_markers"]:
            if marker.lower() in lowered:
                findings.append(finding("warn", kind, f"Protocol-risk marker found: {marker}", path=str(path)))
    pass_value: bool | None = True if scanned else None
    if findings:
        pass_value = False
    return {"pass": pass_value, "scanned_count": scanned, "finding_count": len(findings)}, findings


def check_data_boundary_signals(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rules = get_profile_rules(manifest["profile"])
    findings: list[dict[str, Any]] = []
    scanned = 0
    for path in _iter_scan_paths(manifest, include_patch=True):
        text = path.read_text(encoding="utf-8", errors="replace")
        lowered = text.lower()
        scanned += 1
        for marker, kind in rules["data_boundary_risk_markers"]:
            if marker.lower() in lowered:
                findings.append(finding("warn", kind, f"Data-boundary risk marker found: {marker}", path=str(path)))
    pass_value: bool | None = True if scanned else None
    if findings:
        pass_value = False
    return {"pass": pass_value, "scanned_count": scanned, "finding_count": len(findings)}, findings


def check_reproducibility_surface(manifest: dict[str, Any], extracted: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evidence = manifest["evidence"]
    findings: list[dict[str, Any]] = []
    has_log = any(Path(path).exists() for path in evidence.get("logs", []))
    has_code = any(Path(path).exists() for path in evidence.get("code", []))
    has_artifact = bool(extracted.get("artifact_files"))
    if not has_log:
        findings.append(finding("warn", "missing_logs", "No log file was provided for reproducibility"))
    if not has_code:
        findings.append(finding("warn", "missing_code", "No code file was provided for reproducibility"))
    if not has_artifact:
        findings.append(finding("warn", "missing_artifact", "No artifact file was provided"))
    return {
        "pass": not any(row["severity"] == "fail" for row in findings),
        "has_log": has_log,
        "has_code": has_code,
        "has_artifact": has_artifact,
        "finding_count": len(findings),
    }, findings


def check_patch_triage(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    patch_path = manifest["evidence"].get("patch")
    if patch_path is None or not Path(patch_path).exists():
        return {"pass": None, "present": False}, []
    patch_text = Path(patch_path).read_text(encoding="utf-8", errors="replace")
    files = patch_files(patch_text)
    triage = classify_patch_context(manifest["profile"], patch_text, files)
    findings: list[dict[str, Any]] = []
    if triage["optimizer_in_eval_signal"]:
        findings.append(finding("warn", "optimizer_in_eval_signal", "Patch contains an optimizer-in-eval signal", path=str(patch_path)))
    return {"pass": not findings, "present": True, **triage}, findings


def _iter_scan_paths(manifest: dict[str, Any], *, include_patch: bool) -> list[Path]:
    evidence = manifest["evidence"]
    paths = [Path(path) for path in evidence.get("code", []) if Path(path).exists()]
    if include_patch and evidence.get("patch") is not None:
        patch_path = Path(evidence["patch"])
        if patch_path.exists():
            paths.append(patch_path)
    return paths


def _values_match(lhs: Any, rhs: Any) -> bool:
    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        return abs(float(lhs) - float(rhs)) <= 1e-9
    return lhs == rhs
