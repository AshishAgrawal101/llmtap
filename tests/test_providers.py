"""Tests for llmtap.providers (pure functions, no network)."""
from __future__ import annotations

import json
from email.message import Message

import pytest

from llmtap.config import Config
from llmtap.models import Usage
from llmtap.providers import (
    HOP_BY_HOP,
    error_body,
    estimate_tokens,
    estimate_usage,
    extract_text,
    extract_usage,
    forward_headers,
    is_stream_request,
    parse_json,
    parse_sse,
    request_model,
    resolve_route,
    response_model,
    tags_from_headers,
)


@pytest.fixture
def cfg() -> Config:
    return Config()


# ------------------------------------------------------------------------------------------
# routing
# ------------------------------------------------------------------------------------------
class TestResolveRoute:
    def test_explicit_prefix_openrouter(self, cfg):
        r = resolve_route("/openrouter/v1/chat/completions", {}, cfg)
        assert r.provider == "openrouter"
        assert r.upstream_base == "https://openrouter.ai/api"
        assert r.upstream_path == "/v1/chat/completions"
        assert r.style == "openai"

    def test_explicit_prefix_keeps_query(self, cfg):
        r = resolve_route("/anthropic/v1/messages?beta=true", {}, cfg)
        assert r.provider == "anthropic"
        assert r.upstream_base == "https://api.anthropic.com"
        assert r.upstream_path == "/v1/messages?beta=true"
        assert r.style == "anthropic"

    def test_explicit_prefix_without_v1_gets_v1(self, cfg):
        r = resolve_route("/groq/chat/completions", {}, cfg)
        assert r.upstream_base == "https://api.groq.com/openai"
        assert r.upstream_path == "/v1/chat/completions"

    def test_explicit_prefix_base_with_version_untouched(self, cfg):
        cfg.upstreams["local"] = {"base": "http://127.0.0.1:9999/v1", "style": "openai"}
        r = resolve_route("/local/chat/completions", {}, cfg)
        assert r.provider == "local"
        assert r.upstream_base == "http://127.0.0.1:9999/v1"
        assert r.upstream_path == "/chat/completions"

    def test_explicit_prefix_bare(self, cfg):
        r = resolve_route("/ollama", {}, cfg)
        assert r.provider == "ollama"
        assert r.upstream_path == "/v1/"

    def test_custom_anthropic_style_upstream_not_normalized(self, cfg):
        cfg.upstreams["bedrockish"] = {"base": "http://x", "style": "anthropic"}
        r = resolve_route("/bedrockish/messages", {}, cfg)
        assert r.style == "anthropic"
        assert r.upstream_path == "/messages"

    def test_anthropic_by_path(self, cfg):
        for p in ("/v1/messages", "/v1/messages/count_tokens", "/v1/complete"):
            r = resolve_route(p, {}, cfg)
            assert r.provider == "anthropic" and r.style == "anthropic"
            assert r.upstream_path == p

    def test_anthropic_by_header_case_insensitive(self, cfg):
        r = resolve_route("/v1/models", {"Anthropic-Version": "2023-06-01"}, cfg)
        assert r.provider == "anthropic"
        r = resolve_route("/v1/models", {"X-API-KEY": "k"}, cfg)
        assert r.provider == "anthropic"

    def test_anthropic_by_header_httpmessage(self, cfg):
        m = Message()
        m["Anthropic-Version"] = "2023-06-01"
        assert resolve_route("/v1/models", m, cfg).provider == "anthropic"

    def test_openai_default(self, cfg):
        r = resolve_route("/v1/chat/completions?x=1", {"Authorization": "Bearer sk-x"}, cfg)
        assert r.provider == "openai"
        assert r.upstream_base == "https://api.openai.com"
        assert r.upstream_path == "/v1/chat/completions?x=1"
        assert r.style == "openai"

    def test_openai_prepends_v1(self, cfg):
        assert resolve_route("/chat/completions", {}, cfg).upstream_path == "/v1/chat/completions"
        assert resolve_route("/embeddings?a=b", {}, cfg).upstream_path == "/v1/embeddings?a=b"

    def test_openai_v1_not_doubled(self, cfg):
        assert resolve_route("/v1", {}, cfg).upstream_path == "/v1"
        assert resolve_route("/v1beta/models", {}, cfg).upstream_path == "/v1beta/models"
        # "/v1x..." style names that aren't a version still keep working
        assert resolve_route("/version", {}, cfg).upstream_path == "/v1/version"

    def test_overridden_upstream_base(self, cfg):
        cfg.upstreams["openai"] = {"base": "http://127.0.0.1:5555", "style": "openai"}
        cfg.upstreams["anthropic"] = {"base": "http://127.0.0.1:6666", "style": "anthropic"}
        assert resolve_route("/v1/chat/completions", {}, cfg).upstream_base == "http://127.0.0.1:5555"
        assert resolve_route("/v1/messages", {}, cfg).upstream_base == "http://127.0.0.1:6666"

    def test_unknown_prefix_is_not_explicit(self, cfg):
        r = resolve_route("/nope/v1/chat/completions", {}, cfg)
        assert r.provider == "openai"
        assert r.upstream_path == "/v1/nope/v1/chat/completions"

    def test_weird_paths_do_not_raise(self, cfg):
        for p in ("", "/", "chat/completions", "//", "/?q=1"):
            r = resolve_route(p, {}, cfg)
            assert r.upstream_path.startswith("/")
        assert resolve_route("", None, cfg).provider == "openai"  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------
# request helpers
# ------------------------------------------------------------------------------------------
class TestRequestHelpers:
    def test_parse_json(self):
        assert parse_json(b'{"a": 1}') == {"a": 1}
        assert parse_json('{"a": 1}'.encode("utf-8-sig")) == {"a": 1}
        assert parse_json(b"") is None
        assert parse_json(b"[1,2]") is None
        assert parse_json(b'"str"') is None
        assert parse_json(b"not json") is None
        assert parse_json(b"\xff\xfe") is None
        assert parse_json(b"null") is None

    def test_request_model(self):
        assert request_model({"model": "gpt-4o"}) == "gpt-4o"
        assert request_model({"model": 5}) is None
        assert request_model({"model": ""}) is None
        assert request_model({}) is None
        assert request_model(None) is None
        assert request_model([1]) is None  # type: ignore[arg-type]

    def test_is_stream_request(self):
        assert is_stream_request({"stream": True}) is True
        assert is_stream_request({"stream": "true"}) is False
        assert is_stream_request({"stream": 1}) is False
        assert is_stream_request({}) is False
        assert is_stream_request(None) is False


# ------------------------------------------------------------------------------------------
# non-streaming responses
# ------------------------------------------------------------------------------------------
ANTHROPIC_RESP = {
    "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5-20250929",
    "content": [
        {"type": "thinking", "thinking": "Let me think...", "signature": "abc"},
        {"type": "text", "text": "I'll check the weather. "},
        {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "SF"}},
        {"type": "text", "text": "One moment."},
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 2095,
        "cache_creation_input_tokens": 120,
        "cache_read_input_tokens": 1800,
        "output_tokens": 503,
        "service_tier": "standard",
    },
}

OPENAI_CHAT_RESP = {
    "id": "chatcmpl-B9MHDbslfkBeAs8l4bebGdFOJ6PeG",
    "object": "chat.completion",
    "created": 1741570283,
    "model": "gpt-4o-2024-08-06",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "The image shows a wooden boardwalk.", "refusal": None},
            "logprobs": None,
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 1117,
        "completion_tokens": 46,
        "total_tokens": 1163,
        "prompt_tokens_details": {"cached_tokens": 1024, "audio_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
    "system_fingerprint": "fp_fc9f1d7035",
}

OPENAI_RESPONSES_RESP = {
    "id": "resp_67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",
    "object": "response",
    "created_at": 1741476542,
    "status": "completed",
    "model": "gpt-4.1-2025-04-14",
    "output": [
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {
            "type": "message",
            "id": "msg_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
            "status": "completed",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "In a peaceful grove ", "annotations": []},
                {"type": "refusal", "refusal": "nope"},
                {"type": "output_text", "text": "beneath a silver moon.", "annotations": []},
            ],
        },
    ],
    "usage": {
        "input_tokens": 36,
        "input_tokens_details": {"cached_tokens": 12},
        "output_tokens": 87,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 123,
    },
}

EMBEDDINGS_RESP = {
    "object": "list",
    "data": [{"object": "embedding", "embedding": [0.0023064255, -0.009327292], "index": 0}],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 8, "total_tokens": 8},
}


class TestNonStreaming:
    def test_anthropic(self):
        u = extract_usage("anthropic", ANTHROPIC_RESP)
        assert u == Usage(input_tokens=2095, output_tokens=503, cache_read_tokens=1800, cache_write_tokens=120)
        assert u.estimated is False
        assert extract_text("anthropic", ANTHROPIC_RESP) == "I'll check the weather. One moment."
        assert response_model("anthropic", ANTHROPIC_RESP) == "claude-sonnet-4-5-20250929"

    def test_anthropic_tool_only(self):
        resp = {"content": [{"type": "tool_use", "id": "t", "name": "f", "input": {}}], "usage": {"input_tokens": 3}}
        assert extract_text("anthropic", resp) == ""
        assert extract_usage("anthropic", resp) == Usage(input_tokens=3)

    def test_openai_chat(self):
        u = extract_usage("openai", OPENAI_CHAT_RESP)
        assert u == Usage(input_tokens=1117, output_tokens=46, cache_read_tokens=1024)
        assert extract_text("openai", OPENAI_CHAT_RESP) == "The image shows a wooden boardwalk."
        assert response_model("openai", OPENAI_CHAT_RESP) == "gpt-4o-2024-08-06"

    def test_openai_chat_tool_calls_null_content(self):
        resp = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "c"}]}}]}
        assert extract_text("openai", resp) == ""

    def test_openai_chat_content_parts(self):
        resp = {"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]}
        assert extract_text("openai", resp) == "ab"

    def test_openai_legacy_completions(self):
        resp = {"choices": [{"text": "hello", "index": 0}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}
        assert extract_text("openai", resp) == "hello"
        assert extract_usage("openai", resp) == Usage(input_tokens=2, output_tokens=1)

    def test_openai_responses(self):
        u = extract_usage("openai", OPENAI_RESPONSES_RESP)
        assert u == Usage(input_tokens=36, output_tokens=87, cache_read_tokens=12)
        assert extract_text("openai", OPENAI_RESPONSES_RESP) == "In a peaceful grove beneath a silver moon."
        assert response_model("openai", OPENAI_RESPONSES_RESP) == "gpt-4.1-2025-04-14"

    def test_openai_responses_top_level_output_text(self):
        resp = {"output_text": "shortcut", "output": [], "model": "gpt-5"}
        assert extract_text("openai", resp) == "shortcut"

    def test_embeddings(self):
        assert extract_usage("openai", EMBEDDINGS_RESP) == Usage(input_tokens=8)
        assert extract_text("openai", EMBEDDINGS_RESP) == ""
        assert response_model("openai", EMBEDDINGS_RESP) == "text-embedding-3-small"

    def test_error_responses(self):
        a = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        o = {"error": {"message": "bad key", "type": "invalid_request_error"}}
        for style, resp in (("anthropic", a), ("openai", o)):
            assert extract_usage(style, resp) == Usage()
            assert extract_text(style, resp) == ""
            assert response_model(style, resp) is None

    @pytest.mark.parametrize("bad", [None, [], "x", 3, {}, {"usage": None}, {"usage": "x"},
                                     {"choices": "x"}, {"choices": [None]}, {"choices": []},
                                     {"content": "plain"}, {"content": [1, None, {"type": "text"}]},
                                     {"output": [None, {"content": "x"}, {"content": [None]}]},
                                     {"model": 7}])
    def test_malformed_never_raises(self, bad):
        for style in ("anthropic", "openai"):
            assert isinstance(extract_usage(style, bad), Usage)
            assert isinstance(extract_text(style, bad), str)
            m = response_model(style, bad)
            assert m is None or isinstance(m, str)

    def test_weird_usage_values(self):
        u = extract_usage("openai", {"usage": {"prompt_tokens": None, "completion_tokens": "12",
                                               "prompt_tokens_details": None}})
        assert u == Usage(input_tokens=0, output_tokens=12)
        u = extract_usage("anthropic", {"usage": {"input_tokens": True, "output_tokens": -3}})
        assert u == Usage()
        u = extract_usage("openai", {"usage": {"prompt_tokens": 10.0, "completion_tokens": "abc"}})
        assert u == Usage(input_tokens=10)


# ------------------------------------------------------------------------------------------
# streaming
# ------------------------------------------------------------------------------------------
ANTHROPIC_SSE = """event: message_start
data: {"type":"message_start","message":{"id":"msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY","type":"message","role":"assistant","content":[],"model":"claude-opus-4-1-20250805","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"cache_creation_input_tokens":7,"cache_read_input_tokens":300,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Hmm, let me think"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQB"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: ping
data: {"type": "ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"!"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\": \\"SF\\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":2}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":15}}

event: message_stop
data: {"type":"message_stop"}

"""

OPENAI_CHAT_SSE = """data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini-2024-07-18","system_fingerprint":"fp_44709d6fcb","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini-2024-07-18","system_fingerprint":"fp_44709d6fcb","choices":[{"index":0,"delta":{"content":"Hello"},"logprobs":null,"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini-2024-07-18","system_fingerprint":"fp_44709d6fcb","choices":[{"index":0,"delta":{"content":" world"},"logprobs":null,"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini-2024-07-18","system_fingerprint":"fp_44709d6fcb","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}],"usage":null}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini-2024-07-18","system_fingerprint":"fp_44709d6fcb","choices":[],"usage":{"prompt_tokens":19,"completion_tokens":10,"total_tokens":29,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":0}}}

data: [DONE]

"""

OPENAI_RESPONSES_SSE = """event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_67c9fdcecf488190bdd9a0409de3a1ec07b8b0ad4e5eb654","object":"response","created_at":1741290958,"status":"in_progress","model":"gpt-4.1-2025-04-14","output":[],"usage":null}}

event: response.in_progress
data: {"type":"response.in_progress","sequence_number":1,"response":{"id":"resp_67c9","object":"response","status":"in_progress","model":"gpt-4.1-2025-04-14","output":[],"usage":null}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"msg_67c9","type":"message","status":"in_progress","role":"assistant","content":[]}}

event: response.content_part.added
data: {"type":"response.content_part.added","sequence_number":3,"item_id":"msg_67c9","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_67c9","output_index":0,"content_index":0,"delta":"Hi"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":5,"item_id":"msg_67c9","output_index":0,"content_index":0,"delta":" there!"}

event: response.output_text.done
data: {"type":"response.output_text.done","sequence_number":6,"item_id":"msg_67c9","output_index":0,"content_index":0,"text":"Hi there!"}

event: response.completed
data: {"type":"response.completed","sequence_number":7,"response":{"id":"resp_67c9","object":"response","status":"completed","model":"gpt-4.1-2025-04-14","output":[{"id":"msg_67c9","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hi there!","annotations":[]}]}],"usage":{"input_tokens":37,"input_tokens_details":{"cached_tokens":0},"output_tokens":11,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":48}}}

"""


class TestParseSSE:
    def test_anthropic(self):
        usage, text, model = parse_sse("anthropic", ANTHROPIC_SSE)
        assert text == "Hello!"
        assert model == "claude-opus-4-1-20250805"
        assert usage == Usage(input_tokens=25, output_tokens=15, cache_read_tokens=300, cache_write_tokens=7)

    def test_anthropic_crlf(self):
        usage, text, model = parse_sse("anthropic", ANTHROPIC_SSE.replace("\n", "\r\n"))
        assert (text, model, usage.output_tokens, usage.input_tokens) == ("Hello!", "claude-opus-4-1-20250805", 15, 25)

    def test_anthropic_cr_only(self):
        usage, text, _ = parse_sse("anthropic", ANTHROPIC_SSE.replace("\n", "\r"))
        assert text == "Hello!" and usage.output_tokens == 15

    def test_anthropic_message_delta_cumulative_usage(self):
        raw = (
            'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-haiku-4-5",'
            '"usage":{"input_tokens":0,"output_tokens":1}}}\n\n'
            'event: message_delta\ndata: {"type":"message_delta","usage":{"input_tokens":40,'
            '"cache_read_input_tokens":5,"output_tokens":9}}\n\n'
        )
        usage, _, _ = parse_sse("anthropic", raw)
        assert usage == Usage(input_tokens=40, output_tokens=9, cache_read_tokens=5)

    def test_anthropic_without_event_lines(self):
        raw = "\n\n".join(line for line in ANTHROPIC_SSE.split("\n") if line.startswith("data:"))
        usage, text, _ = parse_sse("anthropic", raw)
        assert text == "Hello!" and usage.output_tokens == 15

    def test_openai_chat(self):
        usage, text, model = parse_sse("openai", OPENAI_CHAT_SSE)
        assert text == "Hello world"
        assert model == "gpt-4o-mini-2024-07-18"
        assert usage == Usage(input_tokens=19, output_tokens=10, cache_read_tokens=4)

    def test_openai_chat_no_usage_chunk(self):
        raw = "\n".join(l for l in OPENAI_CHAT_SSE.split("\n") if '"usage":{' not in l)
        usage, text, _ = parse_sse("openai", raw)
        assert text == "Hello world"
        assert usage == Usage()

    def test_openai_chat_crlf_and_no_space(self):
        raw = OPENAI_CHAT_SSE.replace("data: ", "data:").replace("\n", "\r\n")
        usage, text, _ = parse_sse("openai", raw)
        assert text == "Hello world" and usage.input_tokens == 19

    def test_openai_multiple_choices_only_index0(self):
        raw = (
            'data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"A"}},{"index":1,"delta":{"content":"B"}}]}\n\n'
            'data: {"model":"gpt-4o","choices":[{"index":1,"delta":{"content":"B2"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        assert parse_sse("openai", raw)[1] == "A"

    def test_openai_tool_call_chunks(self):
        raw = (
            'data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":null,'
            '"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":""}}]}}]}\n\n'
            'data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}\n\n'
            "data: [DONE]\n\n"
        )
        usage, text, model = parse_sse("openai", raw)
        assert (text, model) == ("", "gpt-4o")

    def test_openai_legacy_completions_stream(self):
        raw = 'data: {"model":"gpt-3.5-turbo-instruct","choices":[{"text":"Hi","index":0}]}\n\ndata: [DONE]\n\n'
        assert parse_sse("openai", raw)[1] == "Hi"

    def test_openai_responses(self):
        usage, text, model = parse_sse("openai", OPENAI_RESPONSES_SSE)
        assert text == "Hi there!"
        assert model == "gpt-4.1-2025-04-14"
        assert usage == Usage(input_tokens=37, output_tokens=11)

    def test_openai_responses_without_deltas_uses_completed_output(self):
        raw = "\n\n".join(l for l in OPENAI_RESPONSES_SSE.split("\n\n") if "output_text.delta" not in l)
        _, text, _ = parse_sse("openai", raw)
        assert text == "Hi there!"

    def test_multiline_data(self):
        raw = (
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta",\n'
            'data:  "index":0,"delta":{"type":"text_delta","text":"multi"}}\n\n'
        )
        assert parse_sse("anthropic", raw)[1] == "multi"

    def test_comments_and_garbage_skipped(self):
        raw = (
            ": keep-alive comment\n\n"
            "data: {not json\n\n"
            "data: [1,2,3]\n\n"
            "data: null\n\n"
            "garbage line without colon\n\n"
            "retry: 1000\nid: 5\n\n"
            'data: {"model":"m","choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
            'data: {"choices":"bad"}\n\n'
            'data: {"choices":[null, 5, {"delta":"bad"}]}\n\n'
            'data: {"usage":"bad"}\n\n'
            "data: [DONE]\n\n"
            "data: [DONE]\n\n"
        )
        usage, text, model = parse_sse("openai", raw)
        assert (text, model, usage) == ("ok", "m", Usage())

    def test_truncated_transcript(self):
        cut = ANTHROPIC_SSE[: ANTHROPIC_SSE.index("message_delta") + 40]
        usage, text, model = parse_sse("anthropic", cut)
        assert text == "Hello!"
        assert usage.input_tokens == 25 and usage.output_tokens == 1  # final delta lost

    def test_no_trailing_blank_line(self):
        raw = 'data: {"model":"m","choices":[{"index":0,"delta":{"content":"x"}}]}'
        assert parse_sse("openai", raw)[1] == "x"

    def test_anthropic_error_event(self):
        raw = 'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n'
        assert parse_sse("anthropic", raw) == (Usage(), "", None)

    @pytest.mark.parametrize("raw", ["", "\n\n\n", "data:", "event: x", "data: \n\n", "\x00\x01"])
    def test_empty_and_weird(self, raw):
        for style in ("anthropic", "openai"):
            usage, text, model = parse_sse(style, raw)
            assert usage == Usage() and text == "" and model is None

    def test_bytes_input_tolerated(self):
        usage, text, _ = parse_sse("openai", OPENAI_CHAT_SSE.encode())  # type: ignore[arg-type]
        assert text == "Hello world"


# ------------------------------------------------------------------------------------------
# estimation
# ------------------------------------------------------------------------------------------
class TestEstimate:
    def test_estimate_tokens(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("a") == 1
        assert estimate_tokens("abcd" * 10) == 10
        assert estimate_tokens(None) == 0  # type: ignore[arg-type]

    def test_estimate_usage(self):
        req = {"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "x" * 400}]}],
               "max_tokens": 100}
        u = estimate_usage(req, "y" * 80)
        assert u.estimated is True
        assert u.output_tokens == 20
        assert u.input_tokens >= 100
        assert u.cache_read_tokens == 0 and u.cache_write_tokens == 0

    def test_estimate_usage_none(self):
        u = estimate_usage(None, "")
        assert u == Usage(estimated=True)

    def test_estimate_usage_deep_nesting(self):
        req: dict = {}
        cur = req
        for _ in range(500):
            cur["a"] = {}
            cur = cur["a"]
        cur["s"] = "hello"
        assert estimate_usage(req, "").estimated is True


# ------------------------------------------------------------------------------------------
# errors and headers
# ------------------------------------------------------------------------------------------
class TestErrorsAndHeaders:
    def test_error_body_anthropic(self):
        body = json.loads(error_body("anthropic", 429, "budget reached", "rate_limit_error"))
        assert body == {"type": "error", "error": {"type": "rate_limit_error", "message": "budget reached"}}

    def test_error_body_openai(self):
        body = json.loads(error_body("openai", 502, "upstream down"))
        assert body == {"error": {"message": "upstream down", "type": "llmtap_error", "code": "llmtap_error"}}

    def test_error_body_unicode(self):
        assert json.loads(error_body("openai", 400, "blocked … sk-ant")) ["error"]["message"] == "blocked … sk-ant"

    def test_hop_by_hop(self):
        expected = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
                    "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding"}
        assert HOP_BY_HOP == frozenset(expected)

    def test_forward_headers(self):
        h = {
            "Host": "127.0.0.1:4141",
            "Content-Length": "42",
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-abc",
            "Accept-Encoding": "gzip, br",
            "Connection": "keep-alive, X-Custom-Hop",
            "X-Custom-Hop": "1",
            "Transfer-Encoding": "chunked",
            "X-LLMTap-Session": "s1",
            "x-llmtap-tag-feature": "chat",
            "anthropic-version": "2023-06-01",
        }
        out = forward_headers(h)
        assert out == {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-abc",
            "anthropic-version": "2023-06-01",
            "accept-encoding": "identity",
        }

    def test_forward_headers_httpmessage_and_empty(self):
        m = Message()
        m["Host"] = "x"
        m["X-Api-Key"] = "k"
        assert forward_headers(m) == {"X-Api-Key": "k", "accept-encoding": "identity"}
        assert forward_headers({}) == {"accept-encoding": "identity"}

    def test_tags_from_headers(self):
        h = {"X-LLMTap-Tag-Feature": "summarize ", "x-llmtap-tag-User-ID": "42", "X-LLMTap-Session": "run-7",
             "X-LLMTap-Tag-": "ignored", "Authorization": "x"}
        assert tags_from_headers(h) == {"feature": "summarize", "user-id": "42", "session": "run-7"}
        assert tags_from_headers({}) == {}
