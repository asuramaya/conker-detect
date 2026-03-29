from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .submission_profiles import ARTIFACT_EXTS


NUMBER_RE = r"([0-9][0-9_,]*(?:\.[0-9]+)?)"
README_PATTERNS = {
    "val_bpb": re.compile(rf"\bval_bpb\b\s*[:=]\s*{NUMBER_RE}", re.IGNORECASE),
    "pre_quant_val_bpb": re.compile(rf"\bpre_quant_val_bpb\b\s*[:=]\s*{NUMBER_RE}", re.IGNORECASE),
    "bytes_total": re.compile(rf"\bbytes_total\b\s*[:=]\s*{NUMBER_RE}", re.IGNORECASE),
    "bytes_model_int6_zlib": re.compile(
        rf"\b(?:bytes_model_int6_zlib|artifact(?:\s+bytes)?)\b\s*[:=]\s*{NUMBER_RE}",
        re.IGNORECASE,
    ),
}
LOG_PATTERNS = README_PATTERNS


def load_submission_manifest(source: Path | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        manifest = dict(source)
        manifest_path = None
    else:
        manifest_path = Path(source)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Submission manifest must decode to a JSON object")
    profile = str(manifest.get("profile", "parameter-golf"))
    repo_root = manifest.get("repo_root")
    if repo_root is None:
        repo_root_path = manifest_path.parent if manifest_path is not None else Path.cwd()
    else:
        repo_root_path = Path(repo_root)
        if not repo_root_path.is_absolute() and manifest_path is not None:
            repo_root_path = (manifest_path.parent / repo_root_path).resolve()
    submission_root = manifest.get("submission_root")
    if submission_root is None:
        submission_root_path = repo_root_path
    else:
        submission_root_path = Path(submission_root)
        if not submission_root_path.is_absolute():
            submission_root_path = (repo_root_path / submission_root_path).resolve()
    evidence = dict(manifest.get("evidence", {}))
    return {
        **manifest,
        "profile": profile,
        "repo_root": repo_root_path,
        "submission_root": submission_root_path,
        "evidence": _resolve_evidence_paths(evidence, submission_root_path),
    }


def _resolve_evidence_paths(evidence: dict[str, Any], submission_root: Path) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in evidence.items():
        if isinstance(value, list):
            resolved[key] = [_resolve_path(item, submission_root) for item in value]
        elif value is None:
            resolved[key] = None
        else:
            resolved[key] = _resolve_path(value, submission_root)
    return resolved


def _resolve_path(raw: str | Path, submission_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (submission_root / path).resolve()


def read_optional_text(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else None


def extract_claims(manifest: dict[str, Any]) -> dict[str, Any]:
    evidence = manifest["evidence"]
    readme_path = evidence.get("readme")
    submission_json_path = evidence.get("submission_json")
    results_json_path = evidence.get("results_json")
    log_paths = evidence.get("logs", [])
    artifact_paths = evidence.get("artifacts", [])
    code_paths = evidence.get("code", [])
    patch_path = evidence.get("patch")
    readme_text = read_optional_text(readme_path)
    patch_text = read_optional_text(patch_path)
    logs = [(path, read_optional_text(path)) for path in log_paths]
    return {
        "submission_json": extract_json_claims(read_optional_json(submission_json_path)),
        "results_json": extract_json_claims(read_optional_json(results_json_path)),
        "readme": extract_readme_claims(readme_text),
        "logs": extract_log_claims(logs),
        "artifact_files": extract_artifact_claims(artifact_paths),
        "code_files": [str(path) for path in code_paths if path.exists()],
        "patch": {"path": str(patch_path), "text": patch_text} if patch_text is not None and patch_path is not None else None,
    }


def extract_json_claims(obj: dict[str, Any] | None) -> dict[str, Any]:
    if not obj:
        return {}
    out: dict[str, Any] = {}
    for key in ("name", "track", "val_bpb", "pre_quant_val_bpb", "bytes_total", "bytes_model_int6_zlib"):
        if key in obj:
            out[key] = _coerce_scalar(obj[key])
    return out


def extract_readme_claims(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    out: dict[str, Any] = {}
    heading = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if heading:
        out["name"] = heading.group(1).strip()
    for key, pattern in README_PATTERNS.items():
        match = pattern.search(text)
        if match:
            out[key] = _coerce_number(match.group(1))
    return out


def extract_log_claims(logs: list[tuple[Path, str | None]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path, text in logs:
        if not text:
            continue
        row: dict[str, Any] = {"path": str(path)}
        for key, pattern in LOG_PATTERNS.items():
            match = pattern.search(text)
            if match:
                row[key] = _coerce_number(match.group(1))
        out[str(path)] = row
    return out


def extract_artifact_claims(paths: list[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in paths:
        if path.exists():
            out[str(path)] = {
                "bytes": int(path.stat().st_size),
                "suffix": path.suffix,
                "looks_like_artifact": path.suffix in ARTIFACT_EXTS,
            }
    return out


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return _coerce_number(value)
        except ValueError:
            return value.strip()
    return value


def _coerce_number(raw: str) -> int | float:
    text = raw.replace(",", "").replace("_", "").strip()
    if any(ch in text for ch in ".eE"):
        return float(text)
    return int(text)
