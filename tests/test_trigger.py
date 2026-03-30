from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_case(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_dicts(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_dicts(item)


def _walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_strings(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_strings(item)


def _trigger():
    return pytest.importorskip("conker_detect.trigger")


def test_chat_diff_reports_distinct_completion_text() -> None:
    trigger = _trigger()
    if not hasattr(trigger, "chat_diff"):
        pytest.skip("chat_diff is not implemented yet")

    provider_module = _load_module(FIXTURES / "trigger_provider.py", "trigger_provider")
    provider = provider_module.build_provider({"temperature": 0})
    lhs = _load_case(FIXTURES / "trigger_case_lhs.json")
    rhs = _load_case(FIXTURES / "trigger_case_rhs.json")

    result = trigger.chat_diff(provider, lhs, rhs, "demo-model")
    strings = [value for value in _walk_strings(result) if value.startswith("completion::demo-model::")]

    assert isinstance(result, dict)
    assert len(strings) >= 2
    assert len(set(strings)) >= 2
    assert lhs["messages"][0]["content"] in json.dumps(result)
    assert rhs["messages"][0]["content"] in json.dumps(result)


def test_activation_diff_reports_module_metrics_and_shared_count() -> None:
    trigger = _trigger()
    if not hasattr(trigger, "activation_diff"):
        pytest.skip("activation_diff is not implemented yet")

    provider_module = _load_module(FIXTURES / "trigger_provider.py", "trigger_provider")
    provider = provider_module.build_provider({"temperature": 0})
    lhs = _load_case(FIXTURES / "trigger_case_lhs.json")
    rhs = _load_case(FIXTURES / "trigger_case_rhs.json")

    result = trigger.activation_diff(provider, lhs, rhs, "demo-model")
    metric_rows = [
        row
        for row in _walk_dicts(result)
        if {"cosine", "l2", "max_abs"}.issubset(row.keys())
    ]

    assert isinstance(result, dict)
    assert metric_rows
    assert any("shared_module_count" in row for row in _walk_dicts(result))
    serialized = json.dumps(result)
    assert "layer.0" in serialized
    assert "layer.1" in serialized
    assert "layer.2" in serialized


def test_cross_model_compare_reports_pairwise_chat_and_activation_diffs() -> None:
    trigger = _trigger()
    if not hasattr(trigger, "cross_model_compare"):
        pytest.skip("cross_model_compare is not implemented yet")

    provider_module = _load_module(FIXTURES / "trigger_provider.py", "trigger_provider")
    provider = provider_module.build_provider({"temperature": 0})
    case = _load_case(FIXTURES / "trigger_case_shared.json")
    models = ["model-a", "model-b", "model-c"]

    result = trigger.cross_model_compare(provider, case, models)
    pair_rows = [
        row
        for row in _walk_dicts(result)
        if (
            any(key in row for key in ("lhs_model", "left_model", "model_a"))
            and any(key in row for key in ("rhs_model", "right_model", "model_b"))
        )
    ]

    assert isinstance(result, dict)
    assert len(pair_rows) >= 3
    serialized = json.dumps(result)
    for model in models:
        assert model in serialized
    assert "completion" in serialized or "chat" in serialized
    assert "cosine" in serialized or "max_abs" in serialized
