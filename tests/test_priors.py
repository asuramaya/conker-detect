from __future__ import annotations

from conker_detect.priors import summarize_static_priors


def test_summarize_static_priors_from_compare_report_ranks_deviant_family_first() -> None:
    report = {
        "families": [
            {
                "family": "q_proj",
                "count": 2,
                "exact_match_count": 0,
                "mean_cosine_to_reference": 0.75,
                "mean_l2_deviation": 0.2,
                "max_max_abs_deviation": 0.4,
            },
            {
                "family": "down_proj",
                "count": 2,
                "exact_match_count": 2,
                "mean_cosine_to_reference": 1.0,
                "mean_l2_deviation": 0.0,
                "max_max_abs_deviation": 0.0,
            },
        ]
    }

    result = summarize_static_priors(report)

    assert result["source_kind"] == "compare_families"
    assert result["families"][0]["family"] == "q_proj"


def test_summarize_static_priors_from_bundle_report_uses_alerts_and_mask_energy() -> None:
    report = {
        "tensors": [
            {
                "name": "model.layers.0.q_proj.mask",
                "alerts": ["nonzero diagonal energy detected"],
                "regions": {"upper_plus_diag_frac": 0.2},
                "spectral": {"sigma1": 1.0},
            },
            {
                "name": "model.layers.0.down_proj.weight",
                "alerts": [],
                "regions": {"upper_plus_diag_frac": 0.0},
                "spectral": {"sigma1": 2.0},
            },
        ]
    }

    result = summarize_static_priors(report)

    assert result["source_kind"] == "bundle_tensors"
    assert result["families"][0]["score"] > result["families"][-1]["score"]
