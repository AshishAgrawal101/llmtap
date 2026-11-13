"""Tests for Gemini route, usage extraction, streaming, and upstream proxying."""
from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from llmtap.config import Config, DEFAULT_UPSTREAMS
from llmtap.models import Usage
from llmtap.providers import (
    error_body,
    extract_text,
    extract_usage,
    parse_sse,
    resolve_route,
)
from llmtap.proxy import LLMTap


# =====================================================================================
# Step 1: Configuration & Default Upstreams
# =====================================================================================

def test_gemini_in_default_upstreams() -> None:
    """Verify gemini is registered in DEFAULT_UPSTREAMS with the correct base and style."""
    assert "gemini" in DEFAULT_UPSTREAMS, "gemini must be present in DEFAULT_UPSTREAMS"
    entry = DEFAULT_UPSTREAMS["gemini"]
    assert entry["base"] == "https://generativelanguage.googleapis.com"
    assert entry["style"] == "gemini"


def test_config_initializes_with_gemini_upstream() -> None:
    """Config() instances should include the default gemini upstream configuration."""
    cfg = Config()
    assert "gemini" in cfg.upstreams
    assert cfg.upstreams["gemini"]["base"] == "https://generativelanguage.googleapis.com"
    assert cfg.upstreams["gemini"]["style"] == "gemini"


# =====================================================================================
# Step 2: Route Resolution
# =====================================================================================

def test_resolve_route_generate_content() -> None:
    """Requests to :generateContent endpoints should resolve to gemini."""
    cfg = Config()
    r = resolve_route("/v1beta/models/gemini-2.5-flash:generateContent", {}, cfg)
    assert r.provider == "gemini"
    assert r.style == "gemini"
    assert r.upstream_base == "https://generativelanguage.googleapis.com"
    assert r.upstream_path == "/v1beta/models/gemini-2.5-flash:generateContent"


def test_resolve_route_stream_generate_content() -> None:
    """Requests to :streamGenerateContent should keep query parameters like ?alt=sse."""
    cfg = Config()
    path = "/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    r = resolve_route(path, {}, cfg)
    assert r.provider == "gemini"
    assert r.style == "gemini"
    assert r.upstream_base == "https://generativelanguage.googleapis.com"
    assert r.upstream_path == path


def test_resolve_route_by_google_api_key_header() -> None:
    """Requests carrying x-goog-api-key should route to gemini."""
    cfg = Config()
    r = resolve_route("/v1beta/models", {"x-goog-api-key": "AIzaSyFakeKey"}, cfg)
    assert r.provider == "gemini"
    assert r.style == "gemini"


def test_resolve_route_by_prefix() -> None:
    """Explicit /gemini/ prefix routes to gemini without /v1 prepend."""
    cfg = Config()
    r = resolve_route("/gemini/v1beta/models/gemini-pro:generateContent", {}, cfg)
    assert r.provider == "gemini"
    assert r.style == "gemini"
    assert r.upstream_path == "/v1beta/models/gemini-pro:generateContent"


# =====================================================================================
# Step 3: Usage & Text Extraction (Non-Streaming)
# =====================================================================================

def test_extract_usage_gemini() -> None:
    """Extract promptTokenCount, candidatesTokenCount, and cachedContentTokenCount."""
    payload = {
        "candidates": [{"content": {"parts": [{"text": "Hello world"}]}}],
        "usageMetadata": {
            "promptTokenCount": 15,
            "candidatesTokenCount": 35,
            "totalTokenCount": 50,
            "cachedContentTokenCount": 5,
        },
    }
    usage = extract_usage("gemini", payload)
    assert usage.input_tokens == 15
    assert usage.output_tokens == 35
    assert usage.cache_read_tokens == 5
    assert usage.total() == 50


def test_extract_text_gemini() -> None:
    """Extract assistant output text from Gemini candidates structure."""
    payload = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Hello, "},
                        {"text": "how can I assist you today?"},
                    ],
                }
            }
        ]
    }
    text = extract_text("gemini", payload)
    assert text == "Hello, how can I assist you today?"


# =====================================================================================
# Step 4: Streaming (SSE) Parser
# =====================================================================================

def test_parse_sse_gemini() -> None:
    """Parse Gemini Server-Sent Events stream, extract text and final usage."""
    raw_sse = (
        'data: {"candidates": [{"content": {"parts": [{"text": "Part 1, "}]}}]}\n\n'
        'data: {"candidates": [{"content": {"parts": [{"text": "Part 2."}]}}], '
        '"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 6, "totalTokenCount": 16}, '
        '"modelVersion": "gemini-2.5-flash"}\n\n'
    )
    usage, text, model = parse_sse("gemini", raw_sse)
    assert text == "Part 1, Part 2."
    assert usage.input_tokens == 10
    assert usage.output_tokens == 6
    assert model == "gemini-2.5-flash"


# =====================================================================================
# Step 5: Provider-Shaped Error Format
# =====================================================================================

def test_gemini_error_body() -> None:
    """Gemini errors should adhere to the Google RPC format."""
    err = json.loads(error_body("gemini", 400, "invalid argument", "invalid_argument"))
    assert "error" in err
    assert err["error"]["code"] == 400
    assert err["error"]["message"] == "invalid argument"


# =====================================================================================
# Step 6: End-to-End Route Test with Local Upstream Stub
# =====================================================================================

class GeminiStubHandler(BaseHTTPRequestHandler):
    """Minimal local HTTP server emulating Gemini generateContent and streamGenerateContent."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length") or 0)
        _ = self.rfile.read(content_length)

        if ":streamGenerateContent" in self.path:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            chunk1 = (
                b'data: {"candidates": [{"content": {"parts": [{"text": "Streamed "}]}}]}\n\n'
            )
            chunk2 = (
                b'data: {"candidates": [{"content": {"parts": [{"text": "Gemini response"}]}}], '
                b'"usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 4, "totalTokenCount": 16}, '
                b'"modelVersion": "gemini-2.5-flash"}\n\n'
            )
            for c in (chunk1, chunk2):
                self.wfile.write(f"{len(c):X}\r\n".encode() + c + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
        else:
            resp = {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Direct Gemini response"}],
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 8,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 11,
                },
                "modelVersion": "gemini-2.5-flash",
            }
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def test_proxy_e2e_gemini_generate_content(tmp_path) -> None:
    """Test LLMTap proxying a non-streaming Gemini call to a local stub."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), GeminiStubHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    stub_port = server.server_address[1]

    try:
        cfg = Config(port=0, db_path=str(tmp_path / "llmtap.db"))
        cfg.upstreams["gemini"] = {
            "base": f"http://127.0.0.1:{stub_port}",
            "style": "gemini",
        }

        with LLMTap(cfg) as tap:
            proxy_url = f"{tap.base_url}/v1beta/models/gemini-2.5-flash:generateContent"
            req = urllib.request.Request(
                proxy_url,
                data=json.dumps({"contents": [{"parts": [{"text": "Hi"}]}]}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode())
                assert "candidates" in data

            # Verify proxy recorded the exchange correctly
            exchanges = tap.store.list(limit=10)
            assert len(exchanges) == 1
            ex = exchanges[0]
            assert ex.provider == "gemini"
            assert ex.usage.input_tokens == 8
            assert ex.usage.output_tokens == 3
            assert ex.output_text == "Direct Gemini response"
    finally:
        server.shutdown()
        server.server_close()


def test_proxy_e2e_gemini_stream_generate_content(tmp_path) -> None:
    """Test LLMTap proxying a streaming Gemini call to a local stub."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), GeminiStubHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    stub_port = server.server_address[1]

    try:
        cfg = Config(port=0, db_path=str(tmp_path / "llmtap.db"))
        cfg.upstreams["gemini"] = {
            "base": f"http://127.0.0.1:{stub_port}",
            "style": "gemini",
        }

        with LLMTap(cfg) as tap:
            proxy_url = f"{tap.base_url}/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
            req = urllib.request.Request(
                proxy_url,
                data=json.dumps({"contents": [{"parts": [{"text": "Hi"}]}]}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                assert "text/event-stream" in resp.headers.get("Content-Type", "")
                raw_body = resp.read().decode()
                assert "Streamed Gemini response" in raw_body

            # Verify recorded exchange in proxy
            exchanges = tap.store.list(limit=10)
            assert len(exchanges) == 1
            ex = exchanges[0]
            assert ex.provider == "gemini"
            assert ex.streamed is True
            assert ex.usage.input_tokens == 12
            assert ex.usage.output_tokens == 4
            assert ex.output_text == "Streamed Gemini response"
    finally:
        server.shutdown()
        server.server_close()
