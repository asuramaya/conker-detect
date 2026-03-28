from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_token_array(path: Path, *, key: str | None = None) -> np.ndarray:
    if path.suffix == ".npy":
        tokens = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            names = list(data.files)
            if not names:
                raise ValueError(f"No arrays found in token bundle: {path}")
            if key is None:
                if len(names) != 1:
                    raise ValueError(f"Multiple arrays found in {path}; specify --tokens-key")
                key = names[0]
            if key not in data:
                raise ValueError(f"Array {key!r} not found in {path}")
            tokens = np.array(data[key], copy=False)
    elif path.suffix == ".csv":
        tokens = np.loadtxt(path, delimiter=",", dtype=np.int64)
    else:
        raise ValueError(f"Unsupported token format: {path.suffix}")
    arr = np.asarray(tokens, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Token array is empty: {path}")
    return arr


def load_json_config(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    raw = raw.strip()
    if raw.startswith("{"):
        obj = json.loads(raw)
    else:
        obj = json.loads(Path(raw).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Adapter config must decode to a JSON object")
    return obj


def load_adapter(adapter_ref: str, config: dict[str, Any]) -> Any:
    module = _load_adapter_module(adapter_ref)
    build = getattr(module, "build_adapter", None)
    if build is None or not callable(build):
        raise ValueError(f"Adapter {adapter_ref!r} must export a callable build_adapter(config)")
    runner = build(config)
    for name in ("score_chunk", "adapt_chunk"):
        if not hasattr(runner, name) or not callable(getattr(runner, name)):
            raise ValueError(f"Adapter runner must define callable {name}()")
    return runner


def _load_adapter_module(adapter_ref: str):
    candidate = Path(adapter_ref)
    if candidate.suffix == ".py" or candidate.exists():
        if not candidate.exists():
            raise FileNotFoundError(f"Adapter file not found: {adapter_ref}")
        spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load adapter from {adapter_ref}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(adapter_ref)


def fork_runner(runner: Any) -> Any:
    if hasattr(runner, "fork") and callable(runner.fork):
        return runner.fork()
    return copy.deepcopy(runner)


def describe_runner(runner: Any) -> dict[str, Any]:
    if hasattr(runner, "describe") and callable(runner.describe):
        desc = runner.describe()
        if isinstance(desc, dict):
            return desc
    return {"runner_type": type(runner).__name__}


def audit_legality(
    runner: Any,
    tokens: np.ndarray,
    *,
    profile: str,
    chunk_size: int,
    sample_chunks: int,
    future_probes_per_chunk: int,
    answer_probes_per_chunk: int,
    positions_per_future_probe: int,
    seed: int,
    vocab_size: int | None,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if profile != "parameter-golf":
        raise ValueError(f"Unknown legality profile: {profile}")
    return audit_parameter_golf_legality(
        runner,
        tokens,
        chunk_size=chunk_size,
        sample_chunks=sample_chunks,
        future_probes_per_chunk=future_probes_per_chunk,
        answer_probes_per_chunk=answer_probes_per_chunk,
        positions_per_future_probe=positions_per_future_probe,
        seed=seed,
        vocab_size=vocab_size,
        atol=atol,
        rtol=rtol,
    )


def audit_parameter_golf_legality(
    runner: Any,
    tokens: np.ndarray,
    *,
    chunk_size: int = 32_768,
    sample_chunks: int = 4,
    future_probes_per_chunk: int = 2,
    answer_probes_per_chunk: int = 2,
    positions_per_future_probe: int = 4,
    seed: int = 0,
    vocab_size: int | None = None,
    atol: float = 1e-7,
    rtol: float = 1e-7,
) -> dict[str, Any]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    token_arr = np.asarray(tokens, dtype=np.int64).reshape(-1)
    if token_arr.size == 0:
        raise ValueError("tokens must be non-empty")
    if vocab_size is None:
        vocab_size = int(np.max(token_arr)) + 1
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    chunk_starts = list(range(0, int(token_arr.size), chunk_size))
    chunks = [token_arr[start : start + chunk_size] for start in chunk_starts]
    rng = np.random.default_rng(seed)
    chosen_chunks = _choose_chunk_indices(len(chunks), sample_chunks, rng)
    chosen_set = set(chosen_chunks)
    adapter_info = describe_runner(runner)
    adapter_info.setdefault("vocab_size", vocab_size)

    summaries = {
        "repeatability": _empty_summary(),
        "future_suffix_invariance": _empty_summary(),
        "answer_mask_invariance": _empty_summary(),
    }
    probes: list[dict[str, Any]] = []

    for chunk_index, chunk in enumerate(chunks):
        chunk = np.asarray(chunk, dtype=np.int64)
        if chunk_index in chosen_set:
            snapshot = fork_runner(runner)
            chunk_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            chunk_rng = np.random.default_rng(chunk_seed)
            probe_rows, chunk_summaries = _audit_chunk(
                snapshot,
                chunk,
                chunk_index=chunk_index,
                chunk_start=chunk_starts[chunk_index],
                rng=chunk_rng,
                vocab_size=vocab_size,
                future_probes_per_chunk=future_probes_per_chunk,
                answer_probes_per_chunk=answer_probes_per_chunk,
                positions_per_future_probe=positions_per_future_probe,
                atol=atol,
                rtol=rtol,
            )
            probes.extend(probe_rows)
            for name, chunk_summary in chunk_summaries.items():
                _merge_chunk_summary(summaries[name], chunk_summary)

        runner.score_chunk(chunk, sample_positions=None)
        if chunk_index + 1 < len(chunks):
            runner.adapt_chunk(chunk)

    for summary in summaries.values():
        summary["pass"] = summary["failure_count"] == 0

    alerts = [
        f"{name} failed on {summary['failure_count']} / {summary['probe_count']} probes"
        for name, summary in summaries.items()
        if summary["failure_count"] > 0
    ]

    return {
        "profile": "parameter-golf",
        "adapter": adapter_info,
        "token_count": int(token_arr.size),
        "chunk_size": int(chunk_size),
        "chunk_count": len(chunks),
        "selected_chunks": chosen_chunks,
        "tolerances": {"atol": float(atol), "rtol": float(rtol)},
        "checks": summaries,
        "probes": probes,
        "alerts": alerts,
    }


def _audit_chunk(
    snapshot: Any,
    chunk: np.ndarray,
    *,
    chunk_index: int,
    chunk_start: int,
    rng: np.random.Generator,
    vocab_size: int,
    future_probes_per_chunk: int,
    answer_probes_per_chunk: int,
    positions_per_future_probe: int,
    atol: float,
    rtol: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    future_specs = _build_future_specs(
        chunk_len=chunk.size,
        future_probes=future_probes_per_chunk,
        positions_per_probe=positions_per_future_probe,
        rng=rng,
    )
    answer_positions = _build_answer_positions(
        chunk_len=chunk.size,
        answer_probes=answer_probes_per_chunk,
        rng=rng,
    )
    base_positions = sorted(
        set(answer_positions) | {pos for spec in future_specs for pos in spec["positions"]}
    )
    if not base_positions:
        return [], {
            "repeatability": _empty_summary(),
            "future_suffix_invariance": _empty_summary(),
            "answer_mask_invariance": _empty_summary(),
        }

    base_predictions = _score_sample_predictions(fork_runner(snapshot), chunk, base_positions)
    repeat_predictions = _score_sample_predictions(fork_runner(snapshot), chunk, base_positions)

    probes: list[dict[str, Any]] = []
    summaries = {
        "repeatability": _empty_summary(),
        "future_suffix_invariance": _empty_summary(),
        "answer_mask_invariance": _empty_summary(),
    }

    repeat_result = _compare_position_set(
        base_predictions,
        repeat_predictions,
        positions=base_positions,
        atol=atol,
        rtol=rtol,
    )
    repeat_row = {
        "kind": "repeatability",
        "chunk_index": chunk_index,
        "chunk_start": chunk_start,
        "positions": base_positions,
        **repeat_result,
    }
    probes.append(repeat_row)
    _merge_summary(summaries["repeatability"], repeat_result)

    for spec in future_specs:
        perturbed = chunk.copy()
        perturbed[spec["cutoff"] :] = rng.integers(
            0,
            vocab_size,
            size=(chunk.size - spec["cutoff"],),
            dtype=np.int64,
        )
        alt_predictions = _score_sample_predictions(
            fork_runner(snapshot),
            perturbed,
            spec["positions"],
        )
        result = _compare_position_set(
            base_predictions,
            alt_predictions,
            positions=spec["positions"],
            atol=atol,
            rtol=rtol,
        )
        row = {
            "kind": "future_suffix_invariance",
            "chunk_index": chunk_index,
            "chunk_start": chunk_start,
            "cutoff": int(spec["cutoff"]),
            "positions": spec["positions"],
            **result,
        }
        probes.append(row)
        _merge_summary(summaries["future_suffix_invariance"], result)

    for pos in answer_positions:
        perturbed = chunk.copy()
        perturbed[pos:] = rng.integers(0, vocab_size, size=(chunk.size - pos,), dtype=np.int64)
        alt_predictions = _score_sample_predictions(fork_runner(snapshot), perturbed, [pos])
        result = _compare_position_set(
            base_predictions,
            alt_predictions,
            positions=[pos],
            atol=atol,
            rtol=rtol,
        )
        row = {
            "kind": "answer_mask_invariance",
            "chunk_index": chunk_index,
            "chunk_start": chunk_start,
            "position": int(pos),
            **result,
        }
        probes.append(row)
        _merge_summary(summaries["answer_mask_invariance"], result)

    return probes, summaries


def _score_sample_predictions(runner: Any, chunk: np.ndarray, positions: list[int]) -> dict[int, np.ndarray]:
    sample_positions = np.asarray(positions, dtype=np.int64)
    outputs = runner.score_chunk(chunk, sample_positions=sample_positions)
    if not isinstance(outputs, dict):
        raise ValueError("score_chunk() must return a dict")
    if "sample_predictions" not in outputs:
        raise ValueError("score_chunk() must return sample_predictions when sample_positions are requested")
    preds = np.asarray(outputs["sample_predictions"])
    if preds.shape[0] != len(positions):
        raise ValueError(
            "sample_predictions first dimension must match the number of requested sample_positions"
        )
    return {int(pos): np.asarray(preds[idx], dtype=np.float64) for idx, pos in enumerate(positions)}


def _compare_position_set(
    base_predictions: dict[int, np.ndarray],
    alt_predictions: dict[int, np.ndarray],
    *,
    positions: list[int],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    max_abs_diff = 0.0
    shape_mismatch = False
    passed = True
    for pos in positions:
        base = base_predictions[pos]
        alt = alt_predictions[pos]
        if base.shape != alt.shape:
            shape_mismatch = True
            passed = False
            continue
        diff = np.abs(base - alt)
        local_max = float(np.max(diff, initial=0.0))
        if local_max > max_abs_diff:
            max_abs_diff = local_max
        if not np.allclose(base, alt, atol=atol, rtol=rtol):
            passed = False
    return {
        "pass": passed and not shape_mismatch,
        "shape_mismatch": shape_mismatch,
        "max_abs_diff": float(max_abs_diff),
    }


def _build_future_specs(
    *,
    chunk_len: int,
    future_probes: int,
    positions_per_probe: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if chunk_len <= 1 or future_probes <= 0 or positions_per_probe <= 0:
        return []
    specs: list[dict[str, Any]] = []
    for _ in range(future_probes):
        cutoff = int(rng.integers(1, chunk_len))
        max_positions = min(cutoff, positions_per_probe)
        raw_positions = rng.choice(cutoff, size=max_positions, replace=False)
        positions = sorted(int(x) for x in raw_positions.tolist())
        specs.append({"cutoff": cutoff, "positions": positions})
    return specs


def _build_answer_positions(
    *,
    chunk_len: int,
    answer_probes: int,
    rng: np.random.Generator,
) -> list[int]:
    if chunk_len == 0 or answer_probes <= 0:
        return []
    count = min(chunk_len, answer_probes)
    positions = rng.choice(chunk_len, size=count, replace=False)
    return sorted(int(x) for x in positions.tolist())


def _choose_chunk_indices(chunk_count: int, sample_chunks: int, rng: np.random.Generator) -> list[int]:
    if chunk_count <= 0:
        return []
    if sample_chunks <= 0 or sample_chunks >= chunk_count:
        return list(range(chunk_count))
    picks = rng.choice(chunk_count, size=sample_chunks, replace=False)
    return sorted(int(x) for x in picks.tolist())


def _empty_summary() -> dict[str, Any]:
    return {"probe_count": 0, "failure_count": 0, "max_abs_diff": 0.0}


def _merge_summary(target: dict[str, Any], row: dict[str, Any]) -> None:
    target["probe_count"] += 1
    if not row["pass"]:
        target["failure_count"] += 1
    target["max_abs_diff"] = max(float(target["max_abs_diff"]), float(row["max_abs_diff"]))


def _merge_chunk_summary(target: dict[str, Any], chunk_summary: dict[str, Any]) -> None:
    target["probe_count"] += int(chunk_summary["probe_count"])
    target["failure_count"] += int(chunk_summary["failure_count"])
    target["max_abs_diff"] = max(float(target["max_abs_diff"]), float(chunk_summary["max_abs_diff"]))
