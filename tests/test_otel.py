"""Tests for OpenTelemetry spend export."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from llmtap.config import Config
from llmtap.models import Exchange, Usage
from llmtap.otel import COST_UNIT, METRIC_NAME, export_spend, format_otlp_payload
from llmtap.proxy import LLMTap


def test_format_otlp_payload_unit_and_attributes() -> None:
    """Verify OTLP JSON structure, metric name, unit USD, and attributes."""
    ex = Exchange(provider="openai", model="gpt-4o", cost_usd=0.0025)
    payload = format_otlp_payload(ex, timestamp_nano=1_700_000_000_000_000_000)

    metric = payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"][0]
    assert metric["name"] == METRIC_NAME
    assert metric["unit"] == COST_UNIT
    assert metric["unit"] == "USD"

    datapoint = metric["sum"]["dataPoints"][0]
    assert datapoint["asDouble"] == 0.0025
    assert datapoint["timeUnixNano"] == "1700000000000000000"

    attrs = {a["key"]: a["value"]["stringValue"] for a in datapoint["attributes"]}
    assert attrs["provider"] == "openai"
    assert attrs["model"] == "gpt-4o"


def test_export_spend_skips_zero_cost() -> None:
    """Requests with zero cost should not trigger network calls."""
    ex = Exchange(provider="anthropic", model="claude-haiku-4-5", cost_usd=0.0)
    assert export_spend("http://127.0.0.1:4318", ex) is False


class OtelSinkHandler(BaseHTTPRequestHandler):
    """Local test sink recording received OTLP metrics payloads."""

    received: list[dict[str, Any]] = []

    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(length)
        try:
            OtelSinkHandler.received.append(json.loads(data.decode()))
        except Exception:
            pass
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")


def test_export_spend_with_local_collector_sink() -> None:
    """Test exporting spend to a local OTLP collector sink."""
    OtelSinkHandler.received.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), OtelSinkHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    sink_port = server.server_address[1]

    try:
        ex = Exchange(provider="gemini", model="gemini-2.5-flash", cost_usd=0.00015)
        ok = export_spend(f"http://127.0.0.1:{sink_port}/v1/metrics", ex)
        assert ok is True
        assert len(OtelSinkHandler.received) == 1

        payload = OtelSinkHandler.received[0]
        metric = payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"][0]
        assert metric["name"] == "gen_ai.client.cost"
        assert metric["unit"] == "USD"
        dp = metric["sum"]["dataPoints"][0]
        assert dp["asDouble"] == 0.00015
    finally:
        server.shutdown()
        server.server_close()


def test_otel_collector_down_does_not_raise() -> None:
    """Unreachable collector should return False and never raise an exception."""
    ex = Exchange(provider="openai", model="gpt-4o", cost_usd=0.01)
    # Closed local port
    ok = export_spend("http://127.0.0.1:59999/v1/metrics", ex, timeout_s=0.5)
    assert ok is False