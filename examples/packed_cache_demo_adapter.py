from __future__ import annotations

import copy
from typing import Any

import numpy as np


_STATE_HASH_SERIAL = 0


def _next_state_hash_serial() -> int:
    global _STATE_HASH_SERIAL
    _STATE_HASH_SERIAL += 1
    return _STATE_HASH_SERIAL


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
        reported_loss_nats = np.empty((seq.size,), dtype=np.float64)
        reported_weights = np.empty((seq.size,), dtype=np.float64)
        reported_counted = np.empty((seq.size,), dtype=bool)
        reported_path_ids = np.empty((seq.size,), dtype=object)
        state_hash_before = np.empty((seq.size,), dtype=object)
        state_hash_after = np.empty((seq.size,), dtype=object)
        rolling_counts = self.global_counts.copy()
        full_chunk_counts = rolling_counts + np.bincount(seq, minlength=self.vocab_size).astype(np.float64)
        for idx, token in enumerate(seq):
            tok = int(token)
            state_hash_before[idx] = f"pre:{idx}:{rolling_counts.tolist()}"
            if self.mode in {"legal", "distribution_only"}:
                dist = self._posterior(rolling_counts)
                gold_dist = dist
                counted = True
                weight = 1.0
                path_id = self.mode
            elif self.mode == "self_include":
                future_counts = rolling_counts.copy()
                future_counts[tok] += 1.0
                dist = self._posterior(future_counts)
                gold_dist = dist
                counted = True
                weight = 1.0
                path_id = "self_include"
            elif self.mode == "future_peek":
                dist = self._posterior(full_chunk_counts)
                gold_dist = dist
                counted = True
                weight = 1.0
                path_id = "future_peek"
            elif self.mode == "reported_gold_cheat":
                dist = self._posterior(rolling_counts)
                gold_dist = self._posterior(full_chunk_counts)
                counted = True
                weight = 1.0
                path_id = "reported_gold_cheat"
            elif self.mode == "reported_loss_cheat":
                dist = self._posterior(rolling_counts)
                gold_dist = dist
                counted = True
                weight = 1.0
                path_id = "reported_loss_cheat"
            elif self.mode == "counted_flag_cheat":
                dist = self._posterior(rolling_counts)
                gold_dist = dist
                counted = bool(tok % 2 == 0)
                weight = 1.0 if counted else 0.0
                path_id = "counted_flag_cheat"
            elif self.mode == "path_id_cheat":
                dist = self._posterior(rolling_counts)
                gold_dist = dist
                counted = True
                weight = 1.0
                path_id = f"future-argmax:{int(np.argmax(full_chunk_counts))}"
            elif self.mode == "state_hash_cheat":
                dist = self._posterior(rolling_counts)
                gold_dist = dist
                counted = True
                weight = 1.0
                path_id = "state_hash_cheat"
            else:
                raise ValueError(f"Unknown packed-cache demo mode: {self.mode}")
            logits[idx] = dist
            reported_gold_logprobs[idx] = float(np.log(max(float(gold_dist[tok]), np.finfo(np.float64).tiny)))
            if self.mode == "reported_loss_cheat":
                reported_loss_nats[idx] = float(-reported_gold_logprobs[idx] + 0.75)
            else:
                reported_loss_nats[idx] = 0.0 if not counted else float(-weight * reported_gold_logprobs[idx])
            reported_weights[idx] = weight
            reported_counted[idx] = counted
            reported_path_ids[idx] = path_id
            rolling_counts[tok] += 1.0
            if self.mode == "state_hash_cheat":
                state_hash_after[idx] = f"post:{_next_state_hash_serial()}"
            else:
                state_hash_after[idx] = f"post:{idx}:{rolling_counts.tolist()}"
        if sample_positions is None:
            return {}
        idx = np.asarray(sample_positions, dtype=np.int64)
        outputs = {
            "sample_predictions": logits[idx],
        }
        if self.mode != "distribution_only":
            outputs["sample_gold_logprobs"] = reported_gold_logprobs[idx]
            outputs["sample_trace"] = {
                "gold_logprobs": reported_gold_logprobs[idx],
                "loss_nats": reported_loss_nats[idx],
                "weights": reported_weights[idx],
                "counted": reported_counted[idx],
                "path_ids": reported_path_ids[idx],
                "state_hash_before": state_hash_before[idx],
                "state_hash_after": state_hash_after[idx],
            }
        return outputs

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
