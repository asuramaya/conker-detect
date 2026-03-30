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


def _attack():
    return pytest.importorskip("conker_detect.attack")


def test_run_attack_campaign_ranks_candidates_and_minimizes() -> None:
    attack = _attack()
    provider_module = _load_module(FIXTURES / "trigger_provider.py", "trigger_provider")
    provider = provider_module.build_provider({"temperature": 0})
    campaign = json.loads((FIXTURES / "trigger_attack_campaign.json").read_text(encoding="utf-8"))

    result = attack.run_attack_campaign(provider, campaign)

    assert isinstance(result, dict)
    assert result["campaign"]["case_count"] == 2
    assert result["top_candidates"]
    assert result["minimizations"]
    assert result["top_candidates"][0]["score"] >= result["top_candidates"][-1]["score"]


def test_load_campaign_accepts_path() -> None:
    attack = _attack()
    campaign = attack.load_campaign(FIXTURES / "trigger_attack_campaign.json")

    assert campaign["name"] == "fixture-campaign"
    assert campaign["models"] == ["model-a", "model-b", "model-c"]
