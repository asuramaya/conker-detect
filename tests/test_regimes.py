from __future__ import annotations

import pytest


def _regimes():
    return pytest.importorskip("conker_detect.regimes")


def test_cluster_text_regimes_merges_near_duplicate_texts() -> None:
    regimes = _regimes()

    texts = [
        "In the classic fable, a cunning fox tricks a crow into dropping its cheese.",
        "In the classic fable, a cunning fox tricks a crow into dropping a piece of cheese.",
        "A fox, unable to reach the grapes it desires, dismisses them as sour.",
    ]

    result = regimes.cluster_text_regimes(texts, similarity_threshold=0.80)

    assert result["sample_count"] == 3
    assert result["regime_count"] == 2
    assert result["assignments"][0] == result["assignments"][1]
    assert result["assignments"][2] != result["assignments"][0]
    assert result["clusters"][0]["size"] == 2
    assert result["clusters"][0]["probability"] == pytest.approx(2 / 3)


def test_summarize_text_regimes_reports_entropy_and_dominant_mass() -> None:
    regimes = _regimes()

    texts = [
        "A fox story about a crow and cheese.",
        "A fox story about a crow and cheese.",
        "A fox story about sour grapes.",
    ]

    summary = regimes.summarize_text_regimes(texts, similarity_threshold=0.95)

    assert summary["sample_count"] == 3
    assert summary["unique_text_count"] == 2
    assert summary["regime_count"] == 2
    assert summary["exact_consensus"] is False
    assert summary["dominant_regime_mass"] == pytest.approx(2 / 3)
    assert summary["consensus"]["dominant_regime_size"] == 2
    assert summary["entropy_bits"] == pytest.approx(0.9182958340544896, rel=1e-9)
    assert summary["effective_regime_count"] == pytest.approx(2 ** summary["entropy_bits"], rel=1e-9)


def test_exact_consensus_collapses_to_single_regime() -> None:
    regimes = _regimes()

    texts = [
        "The fox dismisses the grapes as sour.",
        "The fox dismisses the grapes as sour.",
        "The fox dismisses the grapes as sour.",
    ]

    summary = regimes.summarize_text_regimes(texts)

    assert summary["regime_count"] == 1
    assert summary["entropy_bits"] == pytest.approx(0.0)
    assert summary["dominant_regime_mass"] == pytest.approx(1.0)
    assert summary["exact_consensus"] is True
    assert summary["consensus"]["exact"] is True
    assert summary["clusters"][0]["exact_consensus"] is True
