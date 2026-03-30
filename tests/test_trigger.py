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


def test_mutate_case_generates_named_variants() -> None:
    trigger = _trigger()
    if not hasattr(trigger, "mutate_case"):
        pytest.skip("mutate_case is not implemented yet")

    case = _load_case(FIXTURES / "trigger_case_shared.json")
    result = trigger.mutate_case(case, families=["quoted", "code_fence", "json_wrapper"])

    assert isinstance(result, dict)
    assert result["variant_count"] == 3
    serialized = json.dumps(result)
    assert "quoted" in serialized
    assert "code_fence" in serialized
    assert "json_wrapper" in serialized


def test_mutate_case_supports_message_level_variants() -> None:
    trigger = _trigger()
    if not hasattr(trigger, "mutate_case"):
        pytest.skip("mutate_case is not implemented yet")

    case = _load_case(FIXTURES / "trigger_case_shared.json")
    result = trigger.mutate_case(case, families=["system_prefix", "assistant_ack", "split_last"])

    assert isinstance(result, dict)
    assert result["variant_count"] == 3
    by_family = {row["family"]: row for row in result["variants"]}
    assert by_family["system_prefix"]["case"]["messages"][0]["role"] == "system"
    assert any(message["role"] == "assistant" for message in by_family["assistant_ack"]["case"]["messages"])
    assert len(by_family["split_last"]["case"]["messages"]) >= 2


def test_sweep_variants_ranks_mutations_by_combined_score() -> None:
    trigger = _trigger()
    if not hasattr(trigger, "sweep_variants"):
        pytest.skip("sweep_variants is not implemented yet")

    provider_module = _load_module(FIXTURES / "trigger_provider.py", "trigger_provider")
    provider = provider_module.build_provider({"temperature": 0})
    case = _load_case(FIXTURES / "trigger_case_shared.json")

    result = trigger.sweep_variants(provider, case, models=["model-a", "model-b"], families=["quoted", "uppercase"])

    assert isinstance(result, dict)
    assert result["variant_count"] == 2
    assert len(result["variants"]) == 2
    assert "combined_score" in json.dumps(result)
    assert result["variants"][0]["max_combined_score"] >= result["variants"][1]["max_combined_score"]


def test_minimize_trigger_reduces_candidate_token_count() -> None:
    trigger = _trigger()
    if not hasattr(trigger, "minimize_trigger"):
        pytest.skip("minimize_trigger is not implemented yet")

    provider_module = _load_module(FIXTURES / "trigger_provider.py", "trigger_provider")
    provider = provider_module.build_provider({"temperature": 0})
    control = _load_case(FIXTURES / "trigger_case_lhs.json")
    candidate = _load_case(FIXTURES / "trigger_case_rhs.json")

    result = trigger.minimize_trigger(provider, control, candidate, model="demo-model", metric="chat")

    assert isinstance(result, dict)
    assert result["minimized_token_count"] <= result["original_token_count"]
    assert result["removed_token_count"] >= 0
    assert result["final_score"] > 0.0


def test_minimize_trigger_supports_char_unit_and_structural_candidates() -> None:
    trigger = _trigger()
    if not hasattr(trigger, "minimize_trigger"):
        pytest.skip("minimize_trigger is not implemented yet")

    provider_module = _load_module(FIXTURES / "trigger_provider.py", "trigger_provider")
    provider = provider_module.build_provider({"temperature": 0})
    control = _load_case(FIXTURES / "trigger_case_shared.json")
    candidate = trigger.compose_case_mutations(control, ["system_prefix", "uppercase"])["case"]

    result = trigger.minimize_trigger(provider, control, candidate, model="demo-model", metric="chat", unit="char")

    assert isinstance(result, dict)
    assert result["unit"] == "char"
    assert result["minimized_token_count"] <= result["original_token_count"]
    assert result["final_score"] > 0.0
