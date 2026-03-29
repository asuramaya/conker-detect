from __future__ import annotations

import copy
from typing import Any

import numpy as np


class PackedCacheDemoAdapter:
    def __init__(self, vocab_size: int = 8, alpha: float = 1.0, mode: str = "legal"):
        self.vocab_size = int(vocab_size)
        self.alpha = float(alpha)
        self.mode = str(mode)
        self.global_counts = np.zeros((self.vocab_size,), dtype=np.float64)

    def fork(self) -> "PackedCacheDemoAdapter":
        return copy.deepcopy(self)

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": "PackedCacheDemoAdapter",
            "vocab_size": self.vocab_size,
            "mode": self.mode,
            "notes": "Synthetic packed-cache adapter with legal and illegal modes for conker-detect legality tests.",
        }

    def _posterior(self, counts: np.ndarray) -> np.ndarray:
        counts = np.asarray(counts, dtype=np.float64)
        probs = counts + self.alpha
        probs /= np.sum(probs)
        return probs

    def score_chunk(self, tokens: np.ndarray, sample_positions: np.ndarray | None = None) -> dict[str, Any]:
        seq = np.asarray(tokens, dtype=np.int64).reshape(-1)
        logits = np.empty((seq.size, self.vocab_size), dtype=np.float64)
        reported_gold_logprobs = np.empty((seq.size,), dtype=np.float64)
        rolling_counts = self.global_counts.copy()
        full_chunk_counts = rolling_counts + np.bincount(seq, minlength=self.vocab_size).astype(np.float64)
        for idx, token in enumerate(seq):
            tok = int(token)
            if self.mode == "legal":
                dist = self._posterior(rolling_counts)
                gold_dist = dist
            elif self.mode == "self_include":
                future_counts = rolling_counts.copy()
                future_counts[tok] += 1.0
                dist = self._posterior(future_counts)
                gold_dist = dist
            elif self.mode == "future_peek":
                dist = self._posterior(full_chunk_counts)
                gold_dist = dist
            elif self.mode == "reported_gold_cheat":
                dist = self._posterior(rolling_counts)
                gold_dist = self._posterior(full_chunk_counts)
            else:
                raise ValueError(f"Unknown packed-cache demo mode: {self.mode}")
            logits[idx] = dist
            reported_gold_logprobs[idx] = float(np.log(max(float(gold_dist[tok]), np.finfo(np.float64).tiny)))
            rolling_counts[tok] += 1.0
        if sample_positions is None:
            return {}
        idx = np.asarray(sample_positions, dtype=np.int64)
        return {
            "sample_predictions": logits[idx],
            "sample_gold_logprobs": reported_gold_logprobs[idx],
        }

    def adapt_chunk(self, tokens: np.ndarray) -> None:
        seq = np.asarray(tokens, dtype=np.int64).reshape(-1)
        if seq.size == 0:
            return
        self.global_counts += np.bincount(seq, minlength=self.vocab_size).astype(np.float64)


def build_adapter(config: dict[str, Any]) -> PackedCacheDemoAdapter:
    return PackedCacheDemoAdapter(
        vocab_size=int(config.get("vocab_size", 8)),
        alpha=float(config.get("alpha", 1.0)),
        mode=str(config.get("mode", "legal")),
    )
