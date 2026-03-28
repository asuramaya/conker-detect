from __future__ import annotations

import copy
from typing import Any

import numpy as np


class DemoCausalAdapter:
    def __init__(self, vocab_size: int = 8, bias_scale: float = 0.01):
        self.vocab_size = int(vocab_size)
        self.bias_scale = float(bias_scale)
        self.bias = np.zeros((self.vocab_size,), dtype=np.float64)
        self.last_token = 0

    def fork(self) -> "DemoCausalAdapter":
        return copy.deepcopy(self)

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": "DemoCausalAdapter",
            "vocab_size": self.vocab_size,
            "notes": "Strictly causal toy adapter for conker-detect legality smoke tests.",
        }

    def score_chunk(self, tokens: np.ndarray, sample_positions: np.ndarray | None = None) -> dict[str, Any]:
        seq = np.asarray(tokens, dtype=np.int64).reshape(-1)
        logits = np.empty((seq.size, self.vocab_size), dtype=np.float64)
        prev = int(self.last_token)
        for idx, token in enumerate(seq):
            row = np.full((self.vocab_size,), -4.0, dtype=np.float64)
            row[(prev + 1) % self.vocab_size] = 0.0
            row += self.bias
            logits[idx] = row
            prev = int(token)
        if seq.size:
            self.last_token = int(seq[-1])
        if sample_positions is None:
            return {}
        idx = np.asarray(sample_positions, dtype=np.int64)
        return {"sample_predictions": logits[idx]}

    def adapt_chunk(self, tokens: np.ndarray) -> None:
        seq = np.asarray(tokens, dtype=np.int64).reshape(-1)
        if seq.size == 0:
            return
        counts = np.bincount(seq, minlength=self.vocab_size).astype(np.float64)
        self.bias += counts * self.bias_scale


def build_adapter(config: dict[str, Any]) -> DemoCausalAdapter:
    return DemoCausalAdapter(
        vocab_size=int(config.get("vocab_size", 8)),
        bias_scale=float(config.get("bias_scale", 0.01)),
    )
