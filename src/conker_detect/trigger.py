from __future__ import annotations

import asyncio
import difflib
import importlib
import importlib.util
import inspect
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .audit import compare_stats


def load_probe_config(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    raw = raw.strip()
    if raw.startswith("{"):
        obj = json.loads(raw)
    else:
        obj = json.loads(Path(raw).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Provider config must decode to a JSON object")
    return obj


def load_case(path_or_raw: str | Path | dict[str, Any], *, default_id: str | None = None) -> dict[str, Any]:
    if isinstance(path_or_raw, dict):
        raw = path_or_raw
    else:
        path = Path(path_or_raw)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = json.loads(str(path_or_raw))
    return normalize_case(raw, default_id=default_id)


def normalize_case(case: dict[str, Any], *, default_id: str | None = None) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise ValueError("Case must decode to a JSON object")
    messages = case.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Case must define a non-empty messages list")
    normalized_messages: list[dict[str, str]] = []
    for index, row in enumerate(messages):
        if not isinstance(row, dict):
            raise ValueError(f"Message {index} must be an object")
        role = row.get("role")
        content = row.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"Message {index} must define string role/content")
        normalized_messages.append({"role": role, "content": content})
    module_names = case.get("module_names", [])
    if module_names is None:
        module_names = []
    if not isinstance(module_names, list) or not all(isinstance(name, str) for name in module_names):
        raise ValueError("module_names must be a list of strings when present")
    custom_id = case.get("custom_id")
    if custom_id is None:
        custom_id = default_id or "case"
    if not isinstance(custom_id, str):
        raise ValueError("custom_id must be a string when present")
    metadata = case.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object when present")
    return {
        "custom_id": custom_id,
        "messages": normalized_messages,
        "module_names": list(module_names),
        "metadata": metadata,
    }


def load_provider(provider_ref: str, config: dict[str, Any]) -> Any:
    module = _load_provider_module(provider_ref)
    build = getattr(module, "build_provider", None)
    if build is None or not callable(build):
        raise ValueError(f"Provider {provider_ref!r} must export a callable build_provider(config)")
    provider = build(config)
    missing = [
        name
        for name in ("chat_completions", "activations")
        if not hasattr(provider, name) or not callable(getattr(provider, name))
    ]
    if missing:
        raise ValueError(f"Provider must define callable methods: {', '.join(missing)}")
    return provider


def describe_provider(provider: Any) -> dict[str, Any]:
    if hasattr(provider, "describe") and callable(provider.describe):
        desc = provider.describe()
        if isinstance(desc, dict):
            return desc
    return {"provider_type": type(provider).__name__}


def chat_diff(provider: Any, lhs_case: dict[str, Any], rhs_case: dict[str, Any], model: str) -> dict[str, Any]:
    lhs = normalize_case(lhs_case, default_id="lhs")
    rhs = normalize_case(rhs_case, default_id="rhs")
    raw_results = _provider_call(provider.chat_completions([lhs, rhs], model=model))
    if not isinstance(raw_results, list) or len(raw_results) != 2:
        raise ValueError("Provider chat_completions() must return a list of two results")
    lhs_result = _normalize_chat_result(raw_results[0], lhs["custom_id"])
    rhs_result = _normalize_chat_result(raw_results[1], rhs["custom_id"])
    return {
        "mode": "chat",
        "model": model,
        "provider": describe_provider(provider),
        "lhs_case": lhs,
        "rhs_case": rhs,
        "lhs": lhs_result,
        "rhs": rhs_result,
        "compare": _text_compare(lhs_result["text"], rhs_result["text"]),
    }


def activation_diff(provider: Any, lhs_case: dict[str, Any], rhs_case: dict[str, Any], model: str) -> dict[str, Any]:
    lhs = normalize_case(lhs_case, default_id="lhs")
    rhs = normalize_case(rhs_case, default_id="rhs")
    raw_results = _provider_call(provider.activations([lhs, rhs], model=model))
    if not isinstance(raw_results, list) or len(raw_results) != 2:
        raise ValueError("Provider activations() must return a list of two results")
    lhs_result = _normalize_activation_result(raw_results[0], lhs["custom_id"])
    rhs_result = _normalize_activation_result(raw_results[1], rhs["custom_id"])
    compare = _compare_activation_maps(lhs_result["activations"], rhs_result["activations"])
    return {
        "mode": "activations",
        "model": model,
        "provider": describe_provider(provider),
        "lhs_case": lhs,
        "rhs_case": rhs,
        "lhs": {"custom_id": lhs_result["custom_id"], "module_count": len(lhs_result["activations"])},
        "rhs": {"custom_id": rhs_result["custom_id"], "module_count": len(rhs_result["activations"])},
        **compare,
    }


def cross_model_compare(provider: Any, case: dict[str, Any], models: list[str]) -> dict[str, Any]:
    normalized = normalize_case(case, default_id="case")
    unique_models = list(dict.fromkeys(models))
    if len(unique_models) < 2:
        raise ValueError("cross_model_compare requires at least two models")

    chat_rows: dict[str, dict[str, Any]] = {}
    activation_rows: dict[str, dict[str, np.ndarray]] = {}
    for model in unique_models:
        raw_chat = _provider_call(provider.chat_completions([normalized], model=model))
        if not isinstance(raw_chat, list) or len(raw_chat) != 1:
            raise ValueError("Provider chat_completions() must return one result per requested case")
        chat_rows[model] = _normalize_chat_result(raw_chat[0], normalized["custom_id"])

        if normalized["module_names"]:
            raw_acts = _provider_call(provider.activations([normalized], model=model))
            if not isinstance(raw_acts, list) or len(raw_acts) != 1:
                raise ValueError("Provider activations() must return one result per requested case")
            activation_rows[model] = _normalize_activation_result(raw_acts[0], normalized["custom_id"])["activations"]

    pairwise_chat = []
    pairwise_activations = []
    for lhs_model, rhs_model in combinations(unique_models, 2):
        pairwise_chat.append(
            {
                "lhs_model": lhs_model,
                "rhs_model": rhs_model,
                "compare": _text_compare(chat_rows[lhs_model]["text"], chat_rows[rhs_model]["text"]),
            }
        )
        if activation_rows:
            pairwise_activations.append(
                {
                    "lhs_model": lhs_model,
                    "rhs_model": rhs_model,
                    **_compare_activation_maps(activation_rows[lhs_model], activation_rows[rhs_model]),
                }
            )

    result: dict[str, Any] = {
        "mode": "crossmodel",
        "provider": describe_provider(provider),
        "case": normalized,
        "models": unique_models,
        "chat": {
            "results": {model: {"text": row["text"], "char_count": row["char_count"]} for model, row in chat_rows.items()},
            "pairwise": pairwise_chat,
        },
    }
    if activation_rows:
        result["activations"] = {
            "module_names": list(normalized["module_names"]),
            "pairwise": pairwise_activations,
        }
    return result


def _load_provider_module(provider_ref: str):
    candidate = Path(provider_ref)
    if candidate.suffix == ".py" or candidate.exists():
        if not candidate.exists():
            raise FileNotFoundError(f"Provider file not found: {provider_ref}")
        spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load provider from {provider_ref}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(provider_ref)


def _provider_call(result: Any) -> Any:
    if inspect.isawaitable(result):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(result)
        raise RuntimeError("Awaitable provider methods are not supported from an active event loop")
    return result


def _normalize_chat_result(result: Any, default_id: str) -> dict[str, Any]:
    plain = _to_plain(result)
    text = _extract_chat_text(plain)
    custom_id = _extract_custom_id(plain, default_id)
    return {
        "custom_id": custom_id,
        "text": text,
        "char_count": len(text),
        "preview": text[:200],
    }


def _normalize_activation_result(result: Any, default_id: str) -> dict[str, Any]:
    plain = _to_plain(result)
    activations = _extract_activation_map(plain)
    custom_id = _extract_custom_id(plain, default_id)
    return {"custom_id": custom_id, "activations": activations}


def _extract_custom_id(result: Any, default_id: str) -> str:
    if isinstance(result, dict):
        custom_id = result.get("custom_id")
        if isinstance(custom_id, str):
            return custom_id
    return default_id


def _extract_chat_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "content", "completion"):
            value = result.get(key)
            if isinstance(value, str):
                return value
        if "message" in result:
            return _extract_chat_text(result["message"])
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            return _extract_chat_text(choices[0])
        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            return _extract_chat_text(messages[-1])
    raise ValueError("Unable to extract completion text from provider result")


def _extract_activation_map(result: Any) -> dict[str, np.ndarray]:
    container = result
    if isinstance(result, dict) and "activations" in result:
        container = result["activations"]

    if isinstance(container, dict):
        out: dict[str, np.ndarray] = {}
        for name, value in container.items():
            if not isinstance(name, str):
                continue
            try:
                out[name] = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                continue
        if out:
            return out

    if isinstance(container, list):
        out = {}
        for row in container:
            if not isinstance(row, dict):
                continue
            name = row.get("module_name") or row.get("name") or row.get("module")
            value = row.get("values")
            if value is None:
                value = row.get("activation")
            if value is None:
                value = row.get("activations")
            if not isinstance(name, str):
                continue
            try:
                out[name] = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                continue
        if out:
            return out

    raise ValueError("Unable to extract activation tensors from provider result")


def _compare_activation_maps(lhs: dict[str, np.ndarray], rhs: dict[str, np.ndarray]) -> dict[str, Any]:
    lhs_names = set(lhs)
    rhs_names = set(rhs)
    shared = sorted(lhs_names & rhs_names)
    modules: list[dict[str, Any]] = []
    shape_mismatches: list[dict[str, Any]] = []
    for name in shared:
        lhs_arr = np.asarray(lhs[name], dtype=np.float64)
        rhs_arr = np.asarray(rhs[name], dtype=np.float64)
        if lhs_arr.shape != rhs_arr.shape:
            shape_mismatches.append(
                {"module_name": name, "lhs_shape": list(lhs_arr.shape), "rhs_shape": list(rhs_arr.shape)}
            )
            continue
        compare = compare_stats(lhs_arr, rhs_arr)
        modules.append(
            {
                "module_name": name,
                "shape": list(lhs_arr.shape),
                "compare": compare,
                "cosine": compare["cosine_to_reference"],
                "l2": compare["l2_deviation"],
                "l1": compare["l1_deviation"],
                "max_abs": compare["max_abs_deviation"],
            }
        )
    modules.sort(key=lambda row: float(row["compare"]["max_abs_deviation"]), reverse=True)
    return {
        "shared_module_count": len(modules),
        "lhs_only_modules": sorted(lhs_names - rhs_names),
        "rhs_only_modules": sorted(rhs_names - lhs_names),
        "shape_mismatches": shape_mismatches,
        "modules": modules,
        "top_modules": modules[:5],
    }


def _text_compare(lhs: str, rhs: str) -> dict[str, Any]:
    ratio = difflib.SequenceMatcher(None, lhs, rhs).ratio()
    prefix = 0
    for left, right in zip(lhs, rhs):
        if left != right:
            break
        prefix += 1
    return {
        "exact_match": lhs == rhs,
        "char_similarity": float(ratio),
        "lhs_char_count": len(lhs),
        "rhs_char_count": len(rhs),
        "common_prefix_chars": prefix,
        "first_diff_char": None if lhs == rhs else prefix,
    }


def _to_plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _to_plain(value.model_dump())
    if hasattr(value, "_asdict") and callable(value._asdict):
        return _to_plain(value._asdict())
    if hasattr(value, "__dict__"):
        return _to_plain({key: val for key, val in vars(value).items() if not key.startswith("_")})
    return value
