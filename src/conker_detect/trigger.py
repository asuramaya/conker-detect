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

MUTATION_FAMILIES = ("whitespace", "quoted", "code_fence", "uppercase", "repeat", "json_wrapper")


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


def mutate_case(case: dict[str, Any], families: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    normalized = normalize_case(case, default_id="case")
    selected = list(families or MUTATION_FAMILIES)
    unknown = [name for name in selected if name not in MUTATION_FAMILIES]
    if unknown:
        raise ValueError(f"Unknown mutation families: {', '.join(sorted(unknown))}")
    if not normalized["messages"]:
        raise ValueError("Case must have at least one message")
    source = normalized["messages"][-1]["content"]
    variants: list[dict[str, Any]] = []
    for family in selected:
        mutated = _apply_mutation(source, family)
        if mutated == source:
            continue
        case_variant = _replace_last_message_content(normalized, mutated, suffix=family)
        variants.append(
            {
                "variant_id": case_variant["custom_id"],
                "family": family,
                "description": _mutation_description(family),
                "case": case_variant,
            }
        )
    return {
        "mode": "mutate",
        "base_case": normalized,
        "family_count": len(selected),
        "variant_count": len(variants),
        "families": selected,
        "variants": variants,
    }


def compose_case_mutations(case: dict[str, Any], families: list[str] | tuple[str, ...]) -> dict[str, Any]:
    normalized = normalize_case(case, default_id="case")
    if not families:
        raise ValueError("compose_case_mutations requires at least one family")
    unknown = [name for name in families if name not in MUTATION_FAMILIES]
    if unknown:
        raise ValueError(f"Unknown mutation families: {', '.join(sorted(unknown))}")
    text = normalized["messages"][-1]["content"]
    for family in families:
        text = _apply_mutation(text, family)
    suffix = "+".join(families)
    composed = _replace_last_message_content(normalized, text, suffix=suffix)
    return {
        "variant_id": composed["custom_id"],
        "families": list(families),
        "description": " then ".join(_mutation_description(name) for name in families),
        "case": composed,
    }


def sweep_variants(
    provider: Any,
    case: dict[str, Any],
    *,
    models: list[str],
    families: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    normalized = normalize_case(case, default_id="case")
    variants = mutate_case(normalized, families=families)["variants"]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        per_model: list[dict[str, Any]] = []
        for model in list(dict.fromkeys(models)):
            chat = chat_diff(provider, normalized, variant["case"], model)
            chat_score = 0.0 if chat["compare"]["exact_match"] else (1.0 - float(chat["compare"]["char_similarity"]))
            activation_score = 0.0
            activation = None
            if normalized["module_names"]:
                activation = activation_diff(provider, normalized, variant["case"], model)
                activation_score = _activation_change_score(activation)
            combined = chat_score + activation_score
            per_model.append(
                {
                    "model": model,
                    "chat_score": chat_score,
                    "activation_score": activation_score,
                    "combined_score": combined,
                    "chat": chat["compare"],
                    "activation_top": [] if activation is None else activation.get("top_modules", []),
                }
            )
        rows.append(
            {
                "variant_id": variant["variant_id"],
                "family": variant["family"],
                "description": variant["description"],
                "case": variant["case"],
                "per_model": per_model,
                "mean_combined_score": float(np.mean([row["combined_score"] for row in per_model])) if per_model else 0.0,
                "max_combined_score": float(np.max([row["combined_score"] for row in per_model])) if per_model else 0.0,
            }
        )
    rows.sort(key=lambda row: (row["max_combined_score"], row["mean_combined_score"]), reverse=True)
    return {
        "mode": "sweep",
        "provider": describe_provider(provider),
        "base_case": normalized,
        "models": list(dict.fromkeys(models)),
        "families": list(families or MUTATION_FAMILIES),
        "variant_count": len(rows),
        "variants": rows,
    }


def minimize_trigger(
    provider: Any,
    control_case: dict[str, Any],
    candidate_case: dict[str, Any],
    *,
    model: str,
    metric: str = "chat",
    threshold: float | None = None,
) -> dict[str, Any]:
    control = normalize_case(control_case, default_id="control")
    candidate = normalize_case(candidate_case, default_id="candidate")
    if len(control["messages"]) != len(candidate["messages"]):
        raise ValueError("Control and candidate cases must have the same number of messages")
    if [row["role"] for row in control["messages"]] != [row["role"] for row in candidate["messages"]]:
        raise ValueError("Control and candidate cases must have matching message roles")
    if metric not in ("chat", "activation"):
        raise ValueError("metric must be 'chat' or 'activation'")
    if threshold is None:
        threshold = 0.0 if metric == "chat" else 1e-12

    original_text = candidate["messages"][-1]["content"]
    tokens = original_text.split()
    if not tokens:
        raise ValueError("Candidate last-message content is empty")

    working = list(tokens)
    changed = True
    iterations = 0
    baseline_score = _trigger_score(provider, control, candidate, model=model, metric=metric)
    final_score = baseline_score
    while changed and len(working) > 1:
        changed = False
        for index in range(len(working)):
            trial_tokens = working[:index] + working[index + 1 :]
            if not trial_tokens:
                continue
            trial_case = _replace_last_message_content(candidate, " ".join(trial_tokens), suffix=f"trial-{iterations}-{index}")
            score = _trigger_score(provider, control, trial_case, model=model, metric=metric)
            if score > float(threshold):
                working = trial_tokens
                final_score = score
                changed = True
                iterations += 1
                break
        if not changed:
            break

    minimized = _replace_last_message_content(candidate, " ".join(working), suffix="minimized")
    return {
        "mode": "minimize",
        "provider": describe_provider(provider),
        "model": model,
        "metric": metric,
        "threshold": float(threshold),
        "control_case": control,
        "original_candidate_case": candidate,
        "minimized_case": minimized,
        "original_token_count": len(tokens),
        "minimized_token_count": len(working),
        "removed_token_count": len(tokens) - len(working),
        "iterations": iterations,
        "baseline_score": float(baseline_score),
        "final_score": float(final_score),
    }


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


def _mutation_description(name: str) -> str:
    return {
        "whitespace": "pad the final message with blank lines and spaces",
        "quoted": "wrap the final message in quotes",
        "code_fence": "wrap the final message in a fenced code block",
        "uppercase": "uppercase the final message",
        "repeat": "repeat the final message twice",
        "json_wrapper": "prefix the final message with a JSON-format instruction",
    }[name]


def _apply_mutation(text: str, family: str) -> str:
    if family == "whitespace":
        return f"\n\n{text}\n"
    if family == "quoted":
        return f"\"{text}\""
    if family == "code_fence":
        return f"```\n{text}\n```"
    if family == "uppercase":
        return text.upper()
    if family == "repeat":
        return f"{text}\n{text}"
    if family == "json_wrapper":
        return f"Respond in JSON.\n{text}"
    raise ValueError(f"Unknown mutation family: {family}")


def _replace_last_message_content(case: dict[str, Any], text: str, *, suffix: str) -> dict[str, Any]:
    messages = [dict(row) for row in case["messages"]]
    messages[-1]["content"] = text
    return {
        "custom_id": f"{case['custom_id']}::{suffix}",
        "messages": messages,
        "module_names": list(case["module_names"]),
        "metadata": dict(case.get("metadata", {})),
    }


def _activation_change_score(result: dict[str, Any]) -> float:
    modules = result.get("top_modules") or result.get("modules") or []
    if not modules:
        return 0.0
    return float(max(float(row.get("max_abs", 0.0)) for row in modules))


def _trigger_score(
    provider: Any,
    control_case: dict[str, Any],
    candidate_case: dict[str, Any],
    *,
    model: str,
    metric: str,
) -> float:
    if metric == "chat":
        row = chat_diff(provider, control_case, candidate_case, model)
        return 0.0 if row["compare"]["exact_match"] else float(1.0 - row["compare"]["char_similarity"])
    row = activation_diff(provider, control_case, candidate_case, model)
    return _activation_change_score(row)


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
