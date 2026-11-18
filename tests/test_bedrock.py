"""AWS Bedrock Runtime routing and transparent proxy tests."""
from __future__ import annotations

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from llmtap.config import Config, DEFAULT_UPSTREAMS
from llmtap.providers import resolve_route
from llmtap.proxy import LLMTap


AWS_HOST = "bedrock-runtime.us-east-1.amazonaws.com"
AWS_AUTH = (
    "AWS4-HMAC-SHA256 "
    "Credential=AKIAIOSFODNN7EXAMPLE/20261117/us-east-1/bedrock/aws4_request, "
    "SignedHeaders=content-type;host;x-amz-date;x-amz-security-token, "
    f"Signature={'0' * 64}"
)
INVOKE_PATH = "/model/anthropic.claude-v2/invoke?trace=true&format=json"
STREAM_PATH = "/model/anthropic.claude-v2/invoke-with-response-stream"


@pytest.fixture
def cfg() -> Config:
    return Config()


class TestBedrockRoute:
    def test_default_upstream(self):
        assert DEFAULT_UPSTREAMS["bedrock"] == {
            "base": "https://bedrock-runtime.{region}.amazonaws.com",
            "style": "bedrock",
        }

    @pytest.mark.parametrize(
        ("host", "expected_base"),
        [
            (
                "bedrock-runtime.us-east-1.amazonaws.com",
                "https://bedrock-runtime.us-east-1.amazonaws.com",
            ),
            (
                "bedrock-runtime-fips.us-gov-west-1.amazonaws.com",
                "https://bedrock-runtime-fips.us-gov-west-1.amazonaws.com",
            ),
            (
                "bedrock-runtime.eu-west-1.api.aws",
                "https://bedrock-runtime.eu-west-1.api.aws",
            ),
            (
                "bedrock-runtime.cn-north-1.amazonaws.com.cn",
                "https://bedrock-runtime.cn-north-1.amazonaws.com.cn",
            ),
        ],
    )
    def test_resolves_supported_hosts(self, cfg, host, expected_base):
        route = resolve_route(INVOKE_PATH, {"Host": host}, cfg)

        assert route.provider == "bedrock"
        assert route.style == "bedrock"
        assert route.upstream_base == expected_base
        assert route.upstream_path == INVOKE_PATH

    def test_host_lookup_is_case_insensitive(self, cfg):
        route = resolve_route(INVOKE_PATH, {"hOsT": AWS_HOST.upper()}, cfg)
        assert route.provider == "bedrock"
        assert route.upstream_base == f"https://{AWS_HOST}"

    def test_resolves_local_proxy_request_from_sigv4_scope(self, cfg):
        route = resolve_route(
            INVOKE_PATH,
            {"Host": "127.0.0.1:4141", "Authorization": AWS_AUTH},
            cfg,
        )

        assert route.provider == "bedrock"
        assert route.upstream_base == f"https://{AWS_HOST}"
        assert route.upstream_path == INVOKE_PATH

    def test_configured_stub_base_is_used_without_formatting(self, cfg):
        cfg.upstreams["bedrock"] = {
            "base": "http://127.0.0.1:54321",
            "style": "bedrock",
        }
        route = resolve_route(INVOKE_PATH, {"Host": AWS_HOST}, cfg)
        assert route.upstream_base == "http://127.0.0.1:54321"

    def test_path_and_query_are_preserved_exactly(self, cfg):
        path = (
            "/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3A"
            "inference-profile%2Fexample/invoke?z=last&a=first%20value"
        )
        route = resolve_route(path, {"Host": AWS_HOST}, cfg)
        assert route.upstream_path == path

    def test_bedrock_credential_does_not_capture_unrelated_path(self, cfg):
        route = resolve_route(
            "/v1/chat/completions",
            {"Host": "127.0.0.1:4141", "Authorization": AWS_AUTH},
            cfg,
        )
        assert route.provider == "openai"

    @pytest.mark.parametrize(
        "host",
        [
            "bedrock.amazonaws.com",
            "bedrock-runtime.example.com",
            "evil-bedrock-runtime.us-east-1.amazonaws.com",
        ],
    )
    def test_similar_hosts_do_not_match(self, cfg, host):
        # (a Bedrock Runtime path is enough on its own, so use a path that isn't one)
        route = resolve_route("/v1/chat/completions", {"Host": host}, cfg)
        assert route.provider == "openai"


class BedrockStub:
    """Local upstream that captures signed requests and can gate a binary stream."""

    first_stream_chunk = b"\x00\x00\x00\x10first-event"
    second_stream_chunk = b"\x00\x00\x00\x11second-event"

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.first_sent = threading.Event()
        self.release_stream = threading.Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length", "0"))
                body = self.rfile.read(length)
                owner.requests.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                    }
                )

                if self.path.endswith("invoke-with-response-stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/vnd.amazon.eventstream")
                    self.end_headers()
                    self.wfile.write(owner.first_stream_chunk)
                    self.wfile.flush()
                    owner.first_sent.set()
                    owner.release_stream.wait(5)
                    self.wfile.write(owner.second_stream_chunk)
                    self.wfile.flush()
                    return

                response = json.dumps({"completion": "hello from bedrock"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.release_stream.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def bedrock_stub():
    stub = BedrockStub()
    try:
        yield stub
    finally:
        stub.close()


@pytest.fixture
def bedrock_tap(bedrock_stub):
    cfg = Config(port=0, db_path=":memory:", quiet=True, guard="off")
    cfg.upstreams["bedrock"] = {
        "base": bedrock_stub.base_url,
        "style": "bedrock",
    }
    tap = LLMTap(cfg).start_background()
    try:
        yield tap
    finally:
        tap.shutdown()


def _signed_headers() -> dict[str, str]:
    return {
        "Host": AWS_HOST,
        "Authorization": AWS_AUTH,
        "Content-Type": "application/json",
        "X-Amz-Date": "20261117T120000Z",
        "X-Amz-Security-Token": "temporary-session-token",
    }


def test_proxy_preserves_signed_request(bedrock_tap, bedrock_stub):
    body = b'{"prompt": "preserve these exact bytes", "max_tokens": 8}'
    conn = http.client.HTTPConnection("127.0.0.1", bedrock_tap.port, timeout=5)
    conn.request("POST", INVOKE_PATH, body=body, headers=_signed_headers())
    response = conn.getresponse()
    response_body = response.read()
    conn.close()

    assert response.status == 200
    assert json.loads(response_body) == {"completion": "hello from bedrock"}
    assert len(bedrock_stub.requests) == 1

    captured = bedrock_stub.requests[0]
    assert captured["path"] == INVOKE_PATH
    assert captured["body"] == body
    assert captured["headers"]["host"] == AWS_HOST
    assert captured["headers"]["authorization"] == AWS_AUTH
    assert captured["headers"]["x-amz-date"] == "20261117T120000Z"
    assert captured["headers"]["x-amz-security-token"] == "temporary-session-token"


def test_proxy_relays_bedrock_eventstream_incrementally(bedrock_tap, bedrock_stub):
    received: list[bytes] = []
    errors: list[BaseException] = []
    got_first = threading.Event()

    def client() -> None:
        conn = http.client.HTTPConnection("127.0.0.1", bedrock_tap.port, timeout=5)
        try:
            conn.request(
                "POST",
                STREAM_PATH,
                body=b'{"prompt":"stream"}',
                headers=_signed_headers(),
            )
            response = conn.getresponse()
            assert response.status == 200
            assert response.getheader("content-type") == "application/vnd.amazon.eventstream"
            received.append(response.read(len(bedrock_stub.first_stream_chunk)))
            got_first.set()
            received.append(response.read())
        except BaseException as exc:  # make failures visible in the test thread
            errors.append(exc)
            got_first.set()
        finally:
            conn.close()

    thread = threading.Thread(target=client, daemon=True)
    thread.start()
    try:
        assert bedrock_stub.first_sent.wait(2), "stub never sent its first stream chunk"
        assert got_first.wait(0.75), "proxy buffered the Bedrock stream instead of relaying it"
        assert not errors
        assert received == [bedrock_stub.first_stream_chunk]
    finally:
        bedrock_stub.release_stream.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert not errors
    assert b"".join(received) == (
        bedrock_stub.first_stream_chunk + bedrock_stub.second_stream_chunk
    )
