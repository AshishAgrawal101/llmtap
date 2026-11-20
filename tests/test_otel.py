"""Tests for OpenTelemetry spend export."""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

import pytest

from llmtap.config import load_config
from llmtap.models import Exchange
from llmtap.otel import COST_UNIT, METRIC_NAME, OtelExporter, export_spend, format_otlp_payload


def _attrs(payload: dict[str, Any]) -> dict[str, str]:
    dp = payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"][0]["sum"]["dataPoints"][0]
    return {a["key"]: a["value"]["stringValue"] for a in dp["attributes"]}


def test_format_otlp_payload_unit_and_attributes() -> None:
    """OTLP JSON structure, metric name, USD unit and GenAI semantic-convention attributes."""
    ex = Exchange(provider="openai", model="gpt-4o", cost_usd=0.0025, ts=1_699_999_999.0)
    payload = format_otlp_payload(ex, timestamp_nano=1_700_000_000_000_000_000)

    metric = payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"][0]
    assert metric["name"] == METRIC_NAME == "llmtap.cost"
    assert metric["unit"] == COST_UNIT == "USD"
    assert metric["sum"]["aggregationTemporality"] == 1 and metric["sum"]["isMonotonic"] is True

    datapoint = metric["sum"]["dataPoints"][0]
    assert datapoint["asDouble"] == 0.0025
    assert datapoint["timeUnixNano"] == "1700000000000000000"
    assert datapoint["startTimeUnixNano"] == "1699999999000000000"

    attrs = _attrs(payload)
    assert attrs == {"gen_ai.provider.name": "openai", "gen_ai.request.model": "gpt-4o"}


def test_payload_includes_session_and_never_bodies() -> None:
    ex = Exchange(provider="anthropic", model="claude-haiku-4-5", cost_usd=0.001, tags={"session": "run-7"},
                  request_body='{"secret": "sk-do-not-export"}', response_body="hello there")
    payload = format_otlp_payload(ex)
    assert _attrs(payload)["llmtap.session"] == "run-7"
    text = json.dumps(payload)
    assert "sk-do-not-export" not in text and "hello there" not in text


def test_export_spend_skips_zero_cost() -> None:
    """Requests with zero cost (cache hits, replays) trigger no network call."""
    ex = Exchange(provider="anthropic", model="claude-haiku-4-5", cost_usd=0.0)
    assert export_spend("http://127.0.0.1:4318", ex) is False


def test_export_spend_bad_endpoint_never_raises() -> None:
    ex = Exchange(provider="openai", model="gpt-4o", cost_usd=0.01)
    assert export_spend("http://127.0.0.1:notaport/v1/metrics", ex) is False


class _Sink:
    """A local OTLP collector stand-in that records received payloads (optionally slowly)."""

    def __init__(self, delay_s: float = 0.0) -> None:
        sink = self
        self.received: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.delay_s = delay_s

        class H(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                data = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if sink.delay_s:
                    time.sleep(sink.delay_s)
                sink.paths.append(self.path)
                sink.received.append(json.loads(data.decode()))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1/metrics"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def wait_for(self, n: int, timeout_s: float = 5.0) -> None:
        deadline = time.time() + timeout_s
        while len(self.received) < n and time.time() < deadline:
            time.sleep(0.02)


@pytest.fixture()
def sink() -> Iterator[_Sink]:
    s = _Sink()
    try:
        yield s
    finally:
        s.close()


def test_export_spend_with_local_collector_sink(sink: _Sink) -> None:
    ex = Exchange(provider="gemini", model="gemini-2.5-flash", cost_usd=0.00015)
    assert export_spend(sink.url, ex) is True
    assert sink.paths == ["/v1/metrics"]
    metric = sink.received[0]["resourceMetrics"][0]["scopeMetrics"][0]["metrics"][0]
    assert metric["name"] == "llmtap.cost" and metric["sum"]["dataPoints"][0]["asDouble"] == 0.00015


def test_endpoint_without_path_posts_to_v1_metrics(sink: _Sink) -> None:
    base = sink.url.rsplit("/v1/metrics", 1)[0]
    assert export_spend(base, Exchange(provider="openai", model="gpt-4o", cost_usd=0.01)) is True
    assert sink.paths == ["/v1/metrics"]


def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_otel_collector_down_does_not_raise() -> None:
    ex = Exchange(provider="openai", model="gpt-4o", cost_usd=0.01)
    assert export_spend(f"http://127.0.0.1:{_closed_port()}/v1/metrics", ex, timeout_s=0.5) is False


def test_exporter_counts_and_drops_without_blocking() -> None:
    exp = OtelExporter(f"http://127.0.0.1:{_closed_port()}/v1/metrics", timeout_s=0.2)
    try:
        t0 = time.time()
        for _ in range(5):
            exp.submit(Exchange(provider="openai", model="gpt-4o", cost_usd=0.01))
        exp.submit(Exchange(provider="openai", model="gpt-4o", cost_usd=0.0))   # free: never queued
        assert time.time() - t0 < 0.1
    finally:
        exp.close(wait_s=5)
    assert exp.sent == 0 and exp.failed == 5


# --------------------------------------------------------------------------- through the proxy

def _anthropic_call(tap: Any, model: str = "claude-sonnet-4-5", session: str | None = None) -> tuple[int, float]:
    body = json.dumps({"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})
    headers = {"Content-Type": "application/json", "x-api-key": "test", "anthropic-version": "2023-06-01"}
    if session:
        headers["X-LLMTap-Session"] = session
    c = http.client.HTTPConnection("127.0.0.1", tap.port, timeout=10)
    t0 = time.time()
    c.request("POST", "/v1/messages", body=body, headers=headers)
    r = c.getresponse()
    r.read()
    c.close()
    return r.status, time.time() - t0


def test_proxy_exports_priced_calls(tap: Any, sink: _Sink) -> None:
    t = tap(otel_endpoint=sink.url)
    status, _ = _anthropic_call(t, session="s1")
    assert status == 200
    sink.wait_for(1)
    assert len(sink.received) == 1
    payload = sink.received[0]
    attrs = _attrs(payload)
    assert attrs["gen_ai.provider.name"] == "anthropic"
    assert attrs["gen_ai.request.model"].startswith("claude-sonnet-4-5")
    assert attrs["llmtap.session"] == "s1"
    cost = payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"][0]["sum"]["dataPoints"][0]["asDouble"]
    recorded = t.store.list(limit=1)[0]
    assert cost == pytest.approx(recorded.cost_usd) and cost > 0


def test_slow_collector_does_not_delay_requests(tap: Any) -> None:
    slow = _Sink(delay_s=1.5)
    try:
        t = tap(otel_endpoint=slow.url)
        _, first = _anthropic_call(t)
        _, second = _anthropic_call(t)
        assert first < 1.0 and second < 1.0      # the 1.5 s collector never sits on the request path
        slow.wait_for(2, timeout_s=8)
        assert len(slow.received) == 2
    finally:
        slow.close()


def test_no_export_without_endpoint(tap: Any) -> None:
    t = tap()
    assert t.otel is None
    assert _anthropic_call(t)[0] == 200


def test_generic_otel_env_vars_are_not_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """A collector configured for other tools must not silently receive llmtap data."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.example:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://collector.example:4318/v1/metrics")
    monkeypatch.delenv("LLMTAP_OTEL_ENDPOINT", raising=False)
    assert load_config(str(tmp_path / "none.toml")).otel_endpoint == ""
    monkeypatch.setenv("LLMTAP_OTEL_ENDPOINT", "http://127.0.0.1:4318/v1/metrics")
    assert load_config(str(tmp_path / "none.toml")).otel_endpoint == "http://127.0.0.1:4318/v1/metrics"
