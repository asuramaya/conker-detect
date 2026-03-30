from __future__ import annotations

import json

import numpy as np

from conker_detect.activation_probes import (
    build_feature_matrix,
    fit_binary_linear_probe,
    flatten_activation_map,
    mean_difference_direction,
    rank_modules_by_separability,
    score_examples,
    summarize_probe,
)


def test_flatten_activation_map_and_build_feature_matrix() -> None:
    examples = [
        {"a": np.array([[1.0, 2.0]]), "b": np.array([3.0])},
        {"a": np.array([[4.0, 5.0]]), "b": np.array([6.0])},
    ]

    flattened = flatten_activation_map(examples[0], module_names=["a", "b"])
    matrix, names, slices = build_feature_matrix(examples, module_names=["a", "b"])

    assert list(flattened["a"]) == [1.0, 2.0]
    assert matrix.shape == (2, 3)
    assert names == ("a", "b")
    assert slices["a"].start == 0
    assert slices["b"].start == 2


def test_flatten_activation_map_pools_variable_length_leading_axes() -> None:
    examples = [
        {"a": np.array([[1.0, 2.0], [3.0, 4.0]])},
        {"a": np.array([[5.0, 6.0]])},
    ]

    flattened = flatten_activation_map(examples[0], module_names=["a"])
    matrix, names, _ = build_feature_matrix(examples, module_names=["a"])

    assert list(flattened["a"]) == [2.0, 3.0]
    assert names == ("a",)
    assert matrix.shape == (2, 2)
    assert list(matrix[1]) == [5.0, 6.0]


def test_mean_difference_direction_scores_positive_higher() -> None:
    positive = [
        {"m": np.array([3.0, 3.0])},
        {"m": np.array([4.0, 4.0])},
    ]
    negative = [
        {"m": np.array([-1.0, -1.0])},
        {"m": np.array([-2.0, -2.0])},
    ]

    probe = mean_difference_direction(positive, negative)
    scores = score_examples(probe, positive + negative)

    assert probe.method == "mean_difference"
    assert scores[0] > scores[-1]
    assert scores[1] > scores[-1]


def test_binary_linear_probe_ridge_handles_matrix_input() -> None:
    matrix = np.array(
        [
            [2.0, 0.0],
            [3.0, 1.0],
            [-1.0, 0.0],
            [-2.0, -1.0],
        ]
    )
    labels = [1, 1, 0, 0]

    probe = fit_binary_linear_probe(matrix, labels, module_names=["x", "y"], method="ridge")
    scores = score_examples(probe, matrix, module_names=["x", "y"])
    summary = summarize_probe(probe, scores=scores, labels=labels)

    assert probe.weights.shape == (2,)
    assert summary["method"] == "ridge"
    assert summary["score_summary"]["accuracy"] >= 0.75
    assert scores[:2].mean() > scores[2:].mean()


def test_rank_modules_by_separability_orders_informative_module_first() -> None:
    examples = [
        {"signal": np.array([3.0, 3.0]), "noise": np.array([0.0, 1.0])},
        {"signal": np.array([4.0, 4.0]), "noise": np.array([1.0, 0.0])},
        {"signal": np.array([-3.0, -3.0]), "noise": np.array([0.5, 0.5])},
        {"signal": np.array([-4.0, -4.0]), "noise": np.array([0.25, 0.75])},
    ]
    labels = [1, 1, 0, 0]

    rows = rank_modules_by_separability(examples, labels)

    assert rows[0]["module_name"] == "signal"
    assert rows[0]["score"] > rows[1]["score"]


def test_summarize_probe_is_compact_and_jsonable() -> None:
    examples = [
        {"m": np.array([2.0, 2.0])},
        {"m": np.array([-2.0, -2.0])},
    ]
    labels = [1, 0]
    probe = fit_binary_linear_probe(examples, labels, module_names=["m"], method="mean_difference")
    scores = score_examples(probe, examples)
    summary = summarize_probe(probe, scores=scores, labels=labels)

    assert summary["feature_count"] == 2
    assert summary["score_summary"]["count"] == 2
    assert json.loads(json.dumps(summary))["method"] == "mean_difference"
