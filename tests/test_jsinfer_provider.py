from __future__ import annotations

from conker_detect.providers.jsinfer_provider import _extract_chat_text, _extract_result_items


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
