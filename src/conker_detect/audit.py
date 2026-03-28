from __future__ import annotations

import csv
import re
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


def normalize_tensor_name(name: str, strip_prefixes: tuple[str, ...] = ()) -> str:
    for prefix in strip_prefixes:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def spectral_stats(matrix: np.ndarray, topk: int) -> dict[str, float]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = singular * singular
    total_energy = float(np.sum(energy))
    top_energy = float(np.sum(energy[: min(topk, singular.size)]))
    sigma1 = float(singular[0])
    sigmak = float(singular[min(topk - 1, singular.size - 1)])
    sigmalast = float(singular[-1])
    return {
        "fro_norm": float(np.linalg.norm(matrix)),
        "sigma1": sigma1,
        f"sigma{topk}": sigmak,
        "sigma_last": sigmalast,
        f"top{topk}_energy_frac": float(top_energy / total_energy) if total_energy > 0 else 0.0,
        f"decay_1_to_{topk}": float(sigma1 / max(sigmak, 1e-12)),
        "decay_1_to_last": float(sigma1 / max(sigmalast, 1e-12)),
    }


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
    for name in shared:
        lhs_name, l_arr = lhs_norm[name]
        rhs_name, r_arr = rhs_norm[name]
        item: dict[str, Any] = {
            "name": name,
            "lhs_name": lhs_name,
            "rhs_name": rhs_name,
            "shape": list(l_arr.shape),
            "compare": compare_stats(l_arr, r_arr),
            "lhs_spectral": spectral_stats(l_arr, topk),
            "rhs_spectral": spectral_stats(r_arr, topk),
        }
        if l_arr.ndim == 2 and l_arr.shape[0] == l_arr.shape[1]:
            item["lhs_regions"] = region_stats(l_arr)
            item["rhs_regions"] = region_stats(r_arr)
        tensors.append(item)
    return {
        "lhs_bundle": str(lhs_path),
        "rhs_bundle": str(rhs_path),
        "shared_tensor_count": len(shared),
        "lhs_only": only_lhs,
        "rhs_only": only_rhs,
        "tensors": tensors,
    }
