from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def build_provider(config: dict[str, Any]) -> "JSInferProvider":
    return JSInferProvider(config)


class JSInferProvider:
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = dict(config)
        self._jsinfer = _import_jsinfer()
        self._client = self._jsinfer.BatchInferenceClient()
        api_key = self._config.get("api_key")
        api_key_env = str(self._config.get("api_key_env", "JSINFER_API_KEY"))
        if api_key is None:
            api_key = os.environ.get(api_key_env)
        if api_key:
            self._client.set_api_key(api_key)
        self._cache_dir: Path | None = None
        cache_dir = self._config.get("cache_dir")
        if cache_dir:
            self._cache_dir = Path(str(cache_dir))
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict[str, Any]:
        return {
            "provider_type": "jsinfer",
            "cache_dir": None if self._cache_dir is None else str(self._cache_dir),
        }

    def chat_completions(self, cases: list[dict[str, Any]], *, model: str) -> list[dict[str, Any]]:
        key = self._cache_key("chat", cases, model)
        cached = self._cache_load(key)
        if cached is not None:
            return cached
        result = asyncio.run(self._chat_async(cases, model=model))
        self._cache_store(key, result)
        return result

    def activations(self, cases: list[dict[str, Any]], *, model: str) -> list[dict[str, Any]]:
        key = self._cache_key("activations", cases, model)
        cached = self._cache_load(key)
        if cached is not None:
            return cached
        result = asyncio.run(self._activations_async(cases, model=model))
        self._cache_store(key, result)
        return result

    async def _chat_async(self, cases: list[dict[str, Any]], *, model: str) -> list[dict[str, Any]]:
        requests = [
            self._jsinfer.ChatCompletionRequest(
                custom_id=case["custom_id"],
                messages=[self._jsinfer.Message(role=row["role"], content=row["content"]) for row in case["messages"]],
            )
            for case in cases
        ]
        raw = await self._client.chat_completions(requests, model=model)
        return self._normalize_chat_results(raw)

    async def _activations_async(self, cases: list[dict[str, Any]], *, model: str) -> list[dict[str, Any]]:
        requests = [
            self._jsinfer.ActivationsRequest(
                custom_id=case["custom_id"],
                messages=[self._jsinfer.Message(role=row["role"], content=row["content"]) for row in case["messages"]],
                module_names=list(case.get("module_names", [])),
            )
            for case in cases
        ]
        raw = await self._client.activations(requests, model=model)
        return self._normalize_activation_results(raw)

    def _normalize_chat_results(self, raw: Any) -> list[dict[str, Any]]:
        items = _extract_result_items(_to_plain(raw))
        out = []
        for item in items:
            out.append({"custom_id": item.get("custom_id"), "text": _extract_chat_text(item)})
        return out

    def _normalize_activation_results(self, raw: Any) -> list[dict[str, Any]]:
        items = _extract_result_items(_to_plain(raw))
        out = []
        for item in items:
            activations = item.get("activations")
            if activations is None and "module_activations" in item:
                activations = item["module_activations"]
            out.append({"custom_id": item.get("custom_id"), "activations": activations})
        return out

    def _cache_key(self, kind: str, cases: list[dict[str, Any]], model: str) -> str:
        payload = json.dumps({"kind": kind, "model": model, "cases": cases}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_load(self, key: str) -> list[dict[str, Any]] | None:
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _cache_store(self, key: str, value: list[dict[str, Any]]) -> None:
        if self._cache_dir is None:
            return
        path = self._cache_dir / f"{key}.json"
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _import_jsinfer():
    try:
        import jsinfer
    except ImportError as exc:
        raise ImportError("jsinfer is required for the jsinfer provider; install conker-detect[dormant] or pip install jsinfer") from exc
    return jsinfer


def _extract_result_items(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        if not raw:
            return []
        for key in ("results", "data", "items"):
            value = raw.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if raw and all(isinstance(value, dict) for value in raw.values()):
            items = []
            for key, value in raw.items():
                item = dict(value)
                item.setdefault("custom_id", key)
                items.append(item)
            return items
    raise ValueError("Unable to normalize jsinfer batch result")


def _extract_chat_text(result: dict[str, Any]) -> str:
    for key in ("text", "content", "completion"):
        value = result.get(key)
        if isinstance(value, str):
            return value
    messages = result.get("messages")
    if isinstance(messages, list) and messages:
        for row in reversed(messages):
            if isinstance(row, dict):
                for key in ("content", "text"):
                    value = row.get(key)
                    if isinstance(value, str):
                        return value
    message = result.get("message")
    if isinstance(message, dict):
        for key in ("content", "text"):
            value = message.get(key)
            if isinstance(value, str):
                return value
    choices = result.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            return _extract_chat_text(first)
    raise ValueError("Unable to extract chat text from jsinfer result")


def _to_plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _to_plain(value.model_dump())
    if hasattr(value, "_asdict") and callable(value._asdict):
        return _to_plain(value._asdict())
    if hasattr(value, "__dict__"):
        return _to_plain({key: val for key, val in vars(value).items() if not key.startswith("_")})
    return value
