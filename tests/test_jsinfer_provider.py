from __future__ import annotations

import numpy as np

from conker_detect.providers.jsinfer_provider import (
    _extract_chat_text,
    _extract_result_items,
    _is_retryable_jsinfer_error,
    _to_plain,
)


def test_extract_chat_text_accepts_jsinfer_message_list_shape() -> None:
    result = {
        "custom_id": "case",
        "messages": [
            {
                "role": "assistant",
                "content": 'In the classic fable "The Fox and the Grapes," a fox dismisses unreachable grapes as sour.',
            }
        ],
    }

    text = _extract_chat_text(result)

    assert text.startswith("In the classic fable")


def test_extract_result_items_accepts_empty_batch_payload() -> None:
    assert _extract_result_items({}) == []


def test_to_plain_serializes_array_like_values() -> None:
    payload = {"activations": {"layer.0": np.array([[1.0, 2.0], [3.0, 4.0]])}}

    plain = _to_plain(payload)

    assert plain == {"activations": {"layer.0": [[1.0, 2.0], [3.0, 4.0]]}}


def test_is_retryable_jsinfer_error_accepts_rate_limit_status() -> None:
    class FakeError(Exception):
        def __init__(self, status: int, message: str) -> None:
            super().__init__(message)
            self.status = status

    assert _is_retryable_jsinfer_error(FakeError(429, "Too Many Requests")) is True
    assert _is_retryable_jsinfer_error(FakeError(503, "temporarily unavailable")) is True
    assert _is_retryable_jsinfer_error(FakeError(400, "bad request")) is False
