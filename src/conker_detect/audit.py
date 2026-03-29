from __future__ import annotations

import csv
import pickle
import re
import zlib
from pathlib import Path
from typing import Any

import numpy as np


def load_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64, copy=False)
    if path.suffix == ".csv":
        with path.open() as f:
            rows = [list(map(float, row)) for row in csv.reader(f) if row]
        return np.array(rows, dtype=np.float64)
    raise ValueError(f"Unsupported matrix format: {path.suffix}")


def load_npz_tensors(
    path: Path,
    *,
    only_2d: bool = True,
    only_square: bool = False,
    name_regex: str | None = None,
) -> dict[str, np.ndarray]:
    patt = re.compile(name_regex) if name_regex else None
    with np.load(path, allow_pickle=False) as data:
        out: dict[str, np.ndarray] = {}
        for name in data.files:
            arr = np.array(data[name], dtype=np.float64, copy=False)
            if only_2d and arr.ndim != 2:
                continue
            if only_square and (arr.ndim != 2 or arr.shape[0] != arr.shape[1]):
                continue
            if patt and not patt.search(name):
                continue
            out[name] = arr
        return out


DETERMINISTIC_SUBSTRATE_MARKERS = (
    "linear_kernel",
    "linear_in_proj",
    "linear_decays",
    "controller_proj",
    "aux_proj",
    "sample_idx",
    "wr",
    "wi",
    "wf",
    "wm",
    "ws",
    "sf",
    "sm",
    "ss",
)

STRUCTURAL_CONTROL_MARKERS = (
    "causal_mask",
    "recency_kernel",
    "delimiter_mask",
    "number_mask",
    "special_mask",
    "markup_mask",
    "attr_mask",
    "entity_mask",
    "token_class_ids",
    "vocab_axis",
    "urlpath_mask",
)


def _artifact_payload_arrays(entry: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    kind = entry.get("type")
    arrays: list[tuple[str, np.ndarray]] = []
    if kind == "quant":
        arrays.append(("q", np.array(entry["q"], copy=False)))
        arrays.append(("scale", np.array(entry["scale"], copy=False)))
    elif kind == "fp16":
        arrays.append(("data", np.array(entry["data"], copy=False)))
    elif kind == "raw":
        arrays.append(("data", np.array(entry["data"], copy=False)))
    else:
        raise ValueError(f"Unknown packed param type: {kind}")
    return arrays


def load_packed_artifact(path: Path) -> tuple[dict[str, Any], int]:
    blob = path.read_bytes()
    raw = zlib.decompress(blob)
    payload = pickle.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Packed artifact did not deserialize to a dict")
    return payload, len(raw)


def classify_artifact_entry(name: str) -> str:
    lowered = name.lower()
    if any(marker in lowered for marker in DETERMINISTIC_SUBSTRATE_MARKERS):
        return "deterministic_substrate"
    if any(marker in lowered for marker in STRUCTURAL_CONTROL_MARKERS):
        return "structural_control"
    return "learned_payload"


def audit_artifact(path: Path) -> dict[str, Any]:
    packed, raw_pickle_bytes = load_packed_artifact(path)
    entries: list[dict[str, Any]] = []
    class_totals: dict[str, int] = {}
    kind_totals: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    alerts: list[str] = []
    for name, entry_obj in sorted(packed.items()):
        entry = dict(entry_obj)
        kind = str(entry.get("type"))
        payload_arrays = _artifact_payload_arrays(entry)
        payload_bytes = int(sum(int(arr.nbytes) for _, arr in payload_arrays))
        numel = int(sum(int(arr.size) for _, arr in payload_arrays))
        class_name = classify_artifact_entry(name)
        shapes = {label: list(arr.shape) for label, arr in payload_arrays}
        dtypes = {label: str(arr.dtype) for label, arr in payload_arrays}
        entries.append(
            {
                "name": name,
                "kind": kind,
                "class": class_name,
                "payload_bytes": payload_bytes,
                "numel": numel,
                "shapes": shapes,
                "dtypes": dtypes,
            }
        )
        class_totals[class_name] = class_totals.get(class_name, 0) + payload_bytes
        kind_totals[kind] = kind_totals.get(kind, 0) + payload_bytes
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    if class_counts.get("deterministic_substrate", 0):
        alerts.append(
            f"artifact includes deterministic substrate entries ({class_counts['deterministic_substrate']} entries, {class_totals['deterministic_substrate']} raw bytes)"
        )
    if class_counts.get("structural_control", 0):
        alerts.append(
            f"artifact includes structural-control entries ({class_counts['structural_control']} entries, {class_totals['structural_control']} raw bytes)"
        )
    entries_by_size = sorted(entries, key=lambda item: item["payload_bytes"], reverse=True)
    return {
        "artifact": str(path),
        "compressed_bytes": int(path.stat().st_size),
        "raw_pickle_bytes": int(raw_pickle_bytes),
        "entry_count": len(entries),
        "class_counts": class_counts,
        "class_payload_bytes": class_totals,
        "kind_payload_bytes": kind_totals,
        "largest_entries": entries_by_size[: min(16, len(entries_by_size))],
        "entries": entries,
        "alerts": alerts,
    }


def normalize_tensor_name(name: str, strip_prefixes: tuple[str, ...] = ()) -> str:
    for prefix in strip_prefixes:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def spectral_stats(matrix: np.ndarray, topk: int) -> dict[str, float]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    effective_topk = min(topk, singular.size)
    energy = singular * singular
    total_energy = float(np.sum(energy))
    top_energy = float(np.sum(energy[:effective_topk]))
    sigma1 = float(singular[0])
    sigmak = float(singular[effective_topk - 1])
    sigmalast = float(singular[-1])
    result = {
        "fro_norm": float(np.linalg.norm(matrix)),
        "sigma1": sigma1,
        f"sigma{effective_topk}": sigmak,
        "sigma_last": sigmalast,
        "requested_topk": int(topk),
        "effective_topk": int(effective_topk),
        f"top{effective_topk}_energy_frac": float(top_energy / total_energy) if total_energy > 0 else 0.0,
        f"decay_1_to_{effective_topk}": float(sigma1 / max(sigmak, 1e-12)),
        "decay_1_to_last": float(sigma1 / max(sigmalast, 1e-12)),
    }
    return result


def region_stats(matrix: np.ndarray) -> dict[str, float]:
    upper = np.triu(matrix, 1)
    diag = np.diag(np.diag(matrix))
    lower = np.tril(matrix, -1)
    total = float(np.linalg.norm(matrix))
    upper_l2 = float(np.linalg.norm(upper))
    diag_l2 = float(np.linalg.norm(diag))
    lower_l2 = float(np.linalg.norm(lower))
    return {
        "upper_l2": upper_l2,
        "diag_l2": diag_l2,
        "lower_l2": lower_l2,
        "upper_frac": float(upper_l2 / total) if total > 0 else 0.0,
        "diag_frac": float(diag_l2 / total) if total > 0 else 0.0,
        "upper_plus_diag_frac": float(np.linalg.norm(upper + diag) / total) if total > 0 else 0.0,
    }


def strict_lower_mask(size: int) -> np.ndarray:
    return np.tril(np.ones((size, size), dtype=np.float64), k=-1)


def toeplitz_mean(mask: np.ndarray, support: np.ndarray | None = None) -> np.ndarray:
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("toeplitz_mean requires a square matrix")
    if support is None:
        support = strict_lower_mask(mask.shape[0])
    out = np.zeros_like(mask, dtype=np.float64)
    size = mask.shape[0]
    for lag in range(1, size):
        vals = np.diag(mask, k=-lag)
        out += np.diag(np.full(size - lag, float(np.mean(vals)), dtype=np.float64), k=-lag)
    return out * support


def lag_profile(mask: np.ndarray) -> list[dict[str, float]]:
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("lag_profile requires a square matrix")
    rows: list[dict[str, float]] = []
    size = mask.shape[0]
    for lag in range(1, size):
        vals = np.diag(mask, k=-lag)
        rows.append(
            {
                "lag": lag,
                "count": int(vals.size),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        )
    return rows


def mask_deviation(mask: np.ndarray, baseline: np.ndarray, support: np.ndarray | None = None) -> dict[str, float | None]:
    if mask.shape != baseline.shape:
        raise ValueError(f"Shape mismatch: {mask.shape} vs {baseline.shape}")
    if support is None:
        active_mask = mask.reshape(-1)
        active_base = baseline.reshape(-1)
    else:
        active_mask = mask[support > 0]
        active_base = baseline[support > 0]
    diff = active_mask - active_base
    denom = float(np.linalg.norm(active_mask) * np.linalg.norm(active_base))
    cosine = None if diff.size == 0 or denom == 0.0 else float(np.dot(active_mask, active_base) / denom)
    return {
        "mask_l1_deviation": float(np.mean(np.abs(diff)) if diff.size else 0.0),
        "mask_l2_deviation": float(np.sqrt(np.mean(diff * diff)) if diff.size else 0.0),
        "mask_max_abs_deviation": float(np.max(np.abs(diff)) if diff.size else 0.0),
        "mask_cosine_similarity": cosine,
    }


def mask_geometry_stats(matrix: np.ndarray) -> dict[str, Any]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("mask_geometry_stats requires a square matrix")
    support = strict_lower_mask(matrix.shape[0])
    active = matrix[support > 0]
    toe = toeplitz_mean(matrix, support)
    residual = (matrix * support) - toe
    lag_rows = lag_profile(matrix)
    return {
        "active_mean": float(np.mean(active) if active.size else 0.0),
        "active_std": float(np.std(active) if active.size else 0.0),
        "active_min": float(np.min(active) if active.size else 0.0),
        "active_max": float(np.max(active) if active.size else 0.0),
        "mean_lag_std": float(np.mean([row["std"] for row in lag_rows]) if lag_rows else 0.0),
        "toeplitz": mask_deviation(matrix * support, toe, support),
        "residual_norm": float(np.linalg.norm(residual)),
        "residual_mean_abs": float(np.mean(np.abs(residual[support > 0])) if active.size else 0.0),
        "lag_profile_head": lag_rows[: min(16, len(lag_rows))],
    }


def compare_stats(matrix: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    if matrix.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {matrix.shape} vs {reference.shape}")
    flat = matrix.reshape(-1)
    ref = reference.reshape(-1)
    denom = float(np.linalg.norm(flat) * np.linalg.norm(ref))
    cosine = None if denom == 0.0 else float(np.dot(flat, ref) / denom)
    diff = flat - ref
    return {
        "cosine_to_reference": cosine,
        "l2_deviation": float(np.sqrt(np.mean(diff * diff))),
        "l1_deviation": float(np.mean(np.abs(diff))),
        "max_abs_deviation": float(np.max(np.abs(diff))),
    }


def _alerts_for_matrix(
    name: str,
    matrix: np.ndarray,
    *,
    expect_causal_mask: bool = False,
    upper_thresh: float = 1e-8,
    diag_thresh: float = 1e-8,
) -> list[str]:
    alerts: list[str] = []
    if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        regions = region_stats(matrix)
        if expect_causal_mask:
            if regions["upper_frac"] > upper_thresh:
                alerts.append(
                    f"forbidden upper-triangle energy detected ({regions['upper_frac']:.6g} > {upper_thresh:.6g})"
                )
            if regions["diag_frac"] > diag_thresh:
                alerts.append(f"nonzero diagonal energy detected ({regions['diag_frac']:.6g} > {diag_thresh:.6g})")
        elif "mask" in name.lower() and regions["upper_plus_diag_frac"] > 1e-3:
            alerts.append(
                f"square matrix named like a mask has nontrivial upper+diag energy ({regions['upper_plus_diag_frac']:.6g})"
            )
    return alerts


def audit_matrix(
    matrix: np.ndarray,
    *,
    name: str = "<matrix>",
    topk: int = 16,
    reference: np.ndarray | None = None,
    expect_causal_mask: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "shape": list(matrix.shape),
        "spectral": spectral_stats(matrix, topk),
    }
    if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        result["regions"] = region_stats(matrix)
        if expect_causal_mask:
            result["mask_geometry"] = mask_geometry_stats(matrix)
    if reference is not None:
        result["compare"] = compare_stats(matrix, reference)
    alerts = _alerts_for_matrix(name, matrix, expect_causal_mask=expect_causal_mask)
    if alerts:
        result["alerts"] = alerts
    return result


def audit_bundle(
    path: Path,
    *,
    topk: int = 16,
    only_square: bool = False,
    name_regex: str | None = None,
    expect_causal: tuple[str, ...] = (),
) -> dict[str, Any]:
    tensors = load_npz_tensors(path, only_2d=True, only_square=only_square, name_regex=name_regex)
    out: dict[str, Any] = {"bundle": str(path), "tensor_count": len(tensors), "tensors": []}
    for name, arr in sorted(tensors.items()):
        expect = any(p in name for p in expect_causal)
        out["tensors"].append(audit_matrix(arr, name=name, topk=topk, expect_causal_mask=expect))
    return out


def compare_bundles(
    lhs_path: Path,
    rhs_path: Path,
    *,
    topk: int = 16,
    only_square: bool = False,
    name_regex: str | None = None,
    strip_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    lhs = load_npz_tensors(lhs_path, only_2d=True, only_square=only_square, name_regex=name_regex)
    rhs = load_npz_tensors(rhs_path, only_2d=True, only_square=only_square, name_regex=name_regex)
    lhs_norm = {normalize_tensor_name(name, strip_prefixes): (name, arr) for name, arr in lhs.items()}
    rhs_norm = {normalize_tensor_name(name, strip_prefixes): (name, arr) for name, arr in rhs.items()}
    shared = sorted(set(lhs_norm) & set(rhs_norm))
    only_lhs = sorted(set(lhs_norm) - set(rhs_norm))
    only_rhs = sorted(set(rhs_norm) - set(lhs_norm))
    tensors: list[dict[str, Any]] = []
    shape_mismatches: list[str] = []
    for name in shared:
        lhs_name, l_arr = lhs_norm[name]
        rhs_name, r_arr = rhs_norm[name]
        item: dict[str, Any] = {
            "name": name,
            "lhs_name": lhs_name,
            "rhs_name": rhs_name,
            "lhs_shape": list(l_arr.shape),
            "rhs_shape": list(r_arr.shape),
        }
        if l_arr.shape != r_arr.shape:
            item["shape_mismatch"] = True
            shape_mismatches.append(name)
        else:
            item["shape"] = list(l_arr.shape)
            item["compare"] = compare_stats(l_arr, r_arr)
        item["lhs_spectral"] = spectral_stats(l_arr, topk)
        item["rhs_spectral"] = spectral_stats(r_arr, topk)
        if l_arr.ndim == 2 and l_arr.shape[0] == l_arr.shape[1]:
            item["lhs_regions"] = region_stats(l_arr)
        if r_arr.ndim == 2 and r_arr.shape[0] == r_arr.shape[1]:
            item["rhs_regions"] = region_stats(r_arr)
        tensors.append(item)
    return {
        "lhs_bundle": str(lhs_path),
        "rhs_bundle": str(rhs_path),
        "shared_tensor_count": len(shared),
        "lhs_only": only_lhs,
        "rhs_only": only_rhs,
        "shape_mismatches": shape_mismatches,
        "tensors": tensors,
    }
