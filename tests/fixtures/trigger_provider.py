from __future__ import annotations

from hashlib import blake2b
from typing import Any


def _case_prompt(case: dict[str, Any]) -> str:
    messages = case.get("messages", [])
    parts: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            parts.append(f"{role}:{content}")
    return " | ".join(parts)


def _vector(prompt: str, model: str, module_name: str) -> list[float]:
    digest = blake2b(f"{model}|{module_name}|{prompt}".encode("utf-8"), digest_size=16).digest()
    return [round(byte / 255.0, 6) for byte in digest[:8]]


class DeterministicTriggerProvider:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)

    def chat_completions(self, cases: list[dict[str, Any]], *, model: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, case in enumerate(cases):
            prompt = _case_prompt(case)
            completion = f"completion::{model}::{index}::{prompt}"
            rows.append(
                {
                    "model": model,
                    "case_index": index,
                    "prompt": prompt,
                    "completion": completion,
                    "text": completion,
                    "message": {"role": "assistant", "content": completion},
                }
            )
        return rows

    def activations(self, cases: list[dict[str, Any]], *, model: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, case in enumerate(cases):
            prompt = _case_prompt(case)
            module_names = [str(name) for name in case.get("module_names", [])]
            module_vectors = {name: _vector(prompt, model, name) for name in module_names}
            rows.append(
                {
                    "model": model,
                    "case_index": index,
                    "prompt": prompt,
                    "module_names": module_names,
                    "module_vectors": module_vectors,
                    "activations": module_vectors,
                    "modules": [
                        {"module_name": name, "activation": vector}
                        for name, vector in module_vectors.items()
                    ],
                }
            )
        return rows


def build_provider(config: dict[str, Any] | None = None) -> DeterministicTriggerProvider:
    return DeterministicTriggerProvider(config=dict(config or {}))
