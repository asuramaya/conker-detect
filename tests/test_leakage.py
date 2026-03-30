from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _leakage():
    return pytest.importorskip("conker_detect.leakage")


def test_build_leakage_probe_suite_generates_expected_templates() -> None:
    leakage = _leakage()
    base_case = {
        "custom_id": "seed",
        "messages": [{"role": "user", "content": "Summarize the fox story in one sentence."}],
        "module_names": None,
        "metadata": None,
    }

    result = leakage.build_leakage_probe_suite(base_case, templates=["verbatim", "hidden_text"])

    assert result["mode"] == "leakage"
    assert result["base_case"]["module_names"] == []
    assert result["base_case"]["metadata"] == {}
    assert result["variant_count"] == 2
    assert [row["template"] for row in result["variants"]] == ["verbatim", "hidden_text"]
    assert "Repeat the following text exactly" in result["variants"][0]["prompt"]
    assert "hidden instructions" in result["variants"][1]["prompt"]
    assert result["variants"][0]["case"]["messages"][-1]["content"] == result["variants"][0]["prompt"]


def test_build_fuzzy_trigger_string_suite_is_deterministic() -> None:
    leakage = _leakage()
    seed = "Abc <<END>> 123"

    result = leakage.build_fuzzy_trigger_string_suite(
        seed,
        families=[
            "uppercase",
            "whitespace",
            "delimiter_noise",
            "partial_prefix",
            "partial_suffix",
            "truncate_head",
            "truncate_tail",
            "middle_slice",
        ],
    )

    assert result["mode"] == "fuzzy_strings"
    by_family = {row["family"]: row for row in result["variants"]}
    assert by_family["uppercase"]["text"] == seed.upper()
    assert by_family["whitespace"]["text"].startswith("  ")
    assert "<<<" in by_family["delimiter_noise"]["text"]
    assert by_family["partial_prefix"]["text"] == seed[: len(seed) // 2]
    assert by_family["partial_suffix"]["text"] == seed[len(seed) - max(1, len(seed) // 2) :]
    assert by_family["truncate_head"]["text"] == seed[len(seed) // 3 :]
    assert by_family["truncate_tail"]["text"] == seed[: len(seed) - max(1, len(seed) // 3)]
    assert by_family["middle_slice"]["text"]
    assert result == leakage.build_fuzzy_trigger_string_suite(
        seed,
        families=[
            "uppercase",
            "whitespace",
            "delimiter_noise",
            "partial_prefix",
            "partial_suffix",
            "truncate_head",
            "truncate_tail",
            "middle_slice",
        ],
    )


def test_build_fuzzy_trigger_case_suite_preserves_structure() -> None:
    leakage = _leakage()
    case = {
        "custom_id": "seed-case",
        "messages": [{"role": "user", "content": "Compare the red fox and the blue fox."}],
        "module_names": ["layer.0", "layer.1"],
        "metadata": {"source": "fixture"},
    }

    result = leakage.build_fuzzy_trigger_case_suite(case, families=["quote_wrap", "brace_wrap", "partial_prefix"])

    assert result["mode"] == "fuzzy_cases"
    assert result["base_case"]["module_names"] == ["layer.0", "layer.1"]
    assert result["base_case"]["metadata"] == {"source": "fixture"}
    by_family = {row["family"]: row for row in result["variants"]}
    assert by_family["quote_wrap"]["case"]["custom_id"].endswith("::quote_wrap")
    assert by_family["brace_wrap"]["case"]["messages"][-1]["content"].startswith("{")
    assert by_family["partial_prefix"]["case"]["messages"][-1]["content"] == "Compare the red fo"
    assert by_family["partial_prefix"]["case"]["metadata"] == {"source": "fixture"}
    assert by_family["partial_prefix"]["case"]["module_names"] == ["layer.0", "layer.1"]
    assert json.dumps(result)
