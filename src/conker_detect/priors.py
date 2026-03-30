from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_prior_source(path_or_raw: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_raw, dict):
        raw = path_or_raw
    else:
        path = Path(path_or_raw)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = json.loads(str(path_or_raw))
    if not isinstance(raw, dict):
        raise ValueError("Prior source must decode to a JSON object")
    return raw


def summarize_static_priors(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    families = report.get("families")
    if isinstance(families, list) and families and isinstance(families[0], dict) and "mean_cosine_to_reference" in families[0]:
        rows = _summarize_compare_families(families)
        return {
            "mode": "prior",
            "source_kind": "compare_families",
            "family_count": len(rows),
            "families": rows,
        }
    tensors = report.get("tensors")
    if isinstance(tensors, list):
        rows = _summarize_bundle_tensors(tensors)
        return {
            "mode": "prior",
            "source_kind": "bundle_tensors",
            "family_count": len(rows),
            "families": rows,
        }
    raise ValueError("Unsupported prior source: expected compare report with families or bundle report with tensors")


def _summarize_compare_families(families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in families:
        count = max(int(row.get("count", 0)), 1)
        cosine = float(row.get("mean_cosine_to_reference", 1.0))
        mean_l2 = float(row.get("mean_l2_deviation", 0.0))
        max_abs = float(row.get("max_max_abs_deviation", 0.0))
        exact_fraction = float(row.get("exact_match_count", 0)) / count
        score = (1.0 - cosine) + mean_l2 + max_abs + (1.0 - exact_fraction)
        rows.append(
            {
                "family": str(row.get("family")),
                "score": float(score),
                "evidence": {
                    "mean_cosine_to_reference": cosine,
                    "mean_l2_deviation": mean_l2,
                    "max_max_abs_deviation": max_abs,
                    "exact_match_fraction": exact_fraction,
                    "count": count,
                },
            }
        )
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return rows


def _summarize_bundle_tensors(tensors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in tensors:
        name = str(row.get("name", ""))
        family = _prior_family_name(name)
        bucket = grouped.setdefault(
            family,
            {
                "family": family,
                "count": 0,
                "alert_count": 0,
                "max_upper_plus_diag_frac": 0.0,
                "max_sigma1": 0.0,
            },
        )
        bucket["count"] += 1
        bucket["alert_count"] += len(row.get("alerts", []))
        regions = row.get("regions", {})
        if isinstance(regions, dict):
            bucket["max_upper_plus_diag_frac"] = max(
                float(bucket["max_upper_plus_diag_frac"]),
                float(regions.get("upper_plus_diag_frac", 0.0)),
            )
        spectral = row.get("spectral", {})
        if isinstance(spectral, dict):
            bucket["max_sigma1"] = max(float(bucket["max_sigma1"]), float(spectral.get("sigma1", 0.0)))
    rows = []
    for bucket in grouped.values():
        score = float(bucket["alert_count"]) + float(bucket["max_upper_plus_diag_frac"])
        rows.append(
            {
                "family": str(bucket["family"]),
                "score": score,
                "evidence": {
                    "count": int(bucket["count"]),
                    "alert_count": int(bucket["alert_count"]),
                    "max_upper_plus_diag_frac": float(bucket["max_upper_plus_diag_frac"]),
                    "max_sigma1": float(bucket["max_sigma1"]),
                },
            }
        )
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return rows


def _prior_family_name(name: str) -> str:
    stem = str(name)
    for suffix in (
        ".weight_scale_inv",
        ".e_score_correction_bias",
        ".weight",
        ".bias",
        ".mask",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or str(name)
