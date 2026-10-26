"""
Tests for normalizing Responses API function_call_output into chat tool messages.

This is important for Gemini/Vertex, which expects tool results to be represented
as tool/function response parts; if the tool output is passed as a list of input_* parts,
we normalize it to text/image blocks or a string.
"""

from copy import deepcopy
from typing import Final

import pytest

from litellm.exceptions import BadRequestError
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def test_function_call_output_list_input_text_is_converted_to_tool_string_content():
    out = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [
                {"type": "input_text", "text": "hello"},
                {"type": "input_text", "text": " world"},
            ],
        }
    )

    assert len(out) == 1
    msg = out[0]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_1"
    assert msg["content"] == "hello world"


def test_function_call_output_string_passthrough():
    out = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok":true}',
        }
    )
    assert len(out) == 1
    assert out[0]["content"] == '{"ok":true}'


def _normalize_media_content(kind: str, parts: list[dict[str, object]]) -> object:
    if kind == "direct":
        return LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(parts)
    return LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        {
            "type": kind,
            "call_id": "call_media",
            "output": parts,
        }
    )[0]["content"]


@pytest.mark.parametrize(
    ("kind", "detail", "nested_url"),
    [
        ("function_call_output", "low", True),
        ("function_call_output", "high", True),
        ("function_call_output", "original", True),
        ("function_call_output", "low", False),
        ("custom_tool_call_output", "high", True),
        ("direct", "high", True),
    ],
)
def test_image_detail_mime_and_adjacent_text_survive_normalization(kind: str, detail: str, nested_url: bool) -> None:
    metadata: Final = {"detail": detail, "format": "image/png"}
    image: Final = {
        "type": "input_image",
        "image_url": {"url": "data:application/octet-stream;base64,AAAA", **metadata}
        if nested_url
        else "data:application/octet-stream;base64,AAAA",
        **({} if nested_url else metadata),
    }
    parts: Final = [{"type": "input_text", "text": "Screenshot follows"}, image]
    original: Final = deepcopy(parts)
    assert _normalize_media_content(kind, parts) == [
        {"type": "text", "text": "Screenshot follows"},
        {"type": "image_url", "image_url": {"url": "data:application/octet-stream;base64,AAAA", **metadata}},
    ]
    assert parts == original


@pytest.mark.parametrize("kind", ["direct", "function_call_output", "custom_tool_call_output"])
def test_media_normalization_preserves_formats_references_and_metadata(kind: str) -> None:
    audio: Final = [
        {
            "type": "input_audio",
            "input_audio": {"data": "AAAA", "format": audio_format},
            "cache_control": {"type": "ephemeral"},
        }
        for audio_format in ("wav", "flac", "audio/ogg")
    ]
    files: Final = [
        {"type": "input_file", "file_data": "data:audio/flac;base64,BBBB", field: "audio/flac"}
        for field in ("format", "mime_type", "content_type")
    ]
    parts: Final = [
        {"type": "input_text", "text": "Inspect this"},
        *audio,
        *files,
        {"type": "input_file", "file_url": "https://example.test/a.pdf"},
        {"type": "input_file", "file_id": "file-test", "detail": "low", "cache_control": {"type": "ephemeral"}},
        {"type": "input_file"},
        {"type": "input_image"},
    ]
    original: Final = deepcopy(parts)
    assert _normalize_media_content(kind, parts) == [
        {"type": "text", "text": "Inspect this"},
        *audio,
        *[{"type": "file", "file": {"file_data": "data:audio/flac;base64,BBBB", "format": "audio/flac"}}] * 3,
        {"type": "file", "file": {"file_id": "https://example.test/a.pdf"}},
        {"type": "file", "file": {"file_id": "file-test", "detail": "low"}, "cache_control": {"type": "ephemeral"}},
        {"type": "file", "file": {}},
        {"type": "image_url", "image_url": {"url": "", "detail": "auto"}},
    ]
    assert parts == original


@pytest.mark.parametrize(
    ("kind", "part"),
    [
        ("direct", {"type": "input_audio", "input_audio": None}),
        ("function_call_output", {"type": "input_audio", "input_audio": {"format": "wav"}}),
        ("custom_tool_call_output", {"type": "input_audio", "input_audio": {"data": "AAAA"}}),
        ("direct", {"type": "input_audio", "input_audio": {"data": 42, "format": "wav"}}),
        ("function_call_output", {"type": "input_audio", "input_audio": {"data": "AAAA", "format": 42}}),
        (
            "custom_tool_call_output",
            {"type": "input_file", "file_data": {"private_payload": "must-not-appear-in-error"}},
        ),
        ("direct", {"type": "input_file", "file_id": "file-test", "filename": 42}),
        ("function_call_output", {"type": "input_file", "file_id": "file-test", "mime_type": ["audio/flac"]}),
        ("direct", {"type": "input_image", "image_url": {42: "must-not-appear-in-error"}}),
        ("custom_tool_call_output", {"type": "file", "file": None}),
        ("function_call_output", {"type": "file", "file": {"file_data": 42}}),
    ],
)
def test_malformed_media_returns_sanitized_client_error_without_mutating_input(
    kind: str,
    part: dict[str, object],
) -> None:
    parts: Final = [{"type": "input_text", "text": "Keep adjacent content"}, part]
    original: Final = deepcopy(parts)
    with pytest.raises(BadRequestError, match="Invalid Responses media content") as error:
        _normalize_media_content(kind, parts)
    assert error.value.status_code == 400
    assert "must-not-appear-in-error" not in str(error.value)
    assert parts == original


def test_computer_screenshot_output_remains_structured_media() -> None:
    messages: Final = (
        LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
            {
                "type": "computer_call_output",
                "call_id": "call_screenshot",
                "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,AAAA"},
            }
        )
    )
    assert messages[0]["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA", "detail": "auto"},
        }
    ]
