"""OpenTelemetry spend exporter (stdlib only, OTLP/HTTP JSON).

Off unless you set ``otel_endpoint`` (``llmtap.toml``) or ``LLMTAP_OTEL_ENDPOINT``. Each priced upstream call is
sent to that collector as one data point of a delta counter:

- metric ``llmtap.cost``, unit ``USD`` (llmtap's own metric: OpenTelemetry has no standard cost metric)
- attributes ``gen_ai.provider.name`` and ``gen_ai.request.model`` (OpenTelemetry GenAI semantic conventions),
  plus ``llmtap.session`` when the request carried ``X-LLMTap-Session``

Only the provider, model, session id and cost are sent: never prompts, responses or keys. Exports happen on a
background thread with a bounded queue, so a slow or unreachable collector never delays a proxied request.
"""
from __future__ import annotations

import http.client
import json
import queue
import ssl
import threading
import time
from typing import Any
from urllib.parse import urlsplit

from .models import Exchange

METRIC_NAME = "llmtap.cost"
COST_UNIT = "USD"
_QUEUE_MAX = 1000           # data points waiting for the collector; newer ones are dropped beyond this


def _attr(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def format_otlp_payload(ex: Exchange, timestamp_nano: int | None = None) -> dict[str, Any]:
    """Format an Exchange's spend into an OTLP JSON ExportMetricsServiceRequest."""
    ts = timestamp_nano if timestamp_nano is not None else int(time.time() * 1e9)
    start = min(int(ex.ts * 1e9), ts)
    attrs = [
        _attr("gen_ai.provider.name", ex.provider or "unknown"),
        _attr("gen_ai.request.model", ex.model or "unknown"),
    ]
    session = (ex.tags or {}).get("session")
    if session:
        attrs.append(_attr("llmtap.session", session))
    return {
        "resourceMetrics": [{
            "resource": {"attributes": [_attr("service.name", "llmtap")]},
            "scopeMetrics": [{
                "scope": {"name": "llmtap"},
                "metrics": [{
                    "name": METRIC_NAME,
                    "description": "Spend on LLM API calls through llmtap, in USD",
                    "unit": COST_UNIT,
                    "sum": {
                        "dataPoints": [{
                            "attributes": attrs,
                            "asDouble": float(ex.cost_usd),
                            "startTimeUnixNano": str(start),
                            "timeUnixNano": str(ts),
                        }],
                        "aggregationTemporality": 1,  # AGGREGATION_TEMPORALITY_DELTA
                        "isMonotonic": True,
                    },
                }],
            }],
        }]
    }


def export_spend(endpoint: str, ex: Exchange, timeout_s: float = 2.0) -> bool:
    """Send one spend data point to an OTLP/HTTP collector. Never raises; returns True on a 2xx."""
    if not endpoint or ex.cost_usd <= 0:
        return False
    try:
        target = endpoint if endpoint.startswith(("http://", "https://")) else "http://" + endpoint
        u = urlsplit(target)
        path = u.path if u.path not in ("", "/") else "/v1/metrics"
        if u.query:
            path = f"{path}?{u.query}"
        body = json.dumps(format_otlp_payload(ex)).encode("utf-8")
        port = u.port or (443 if u.scheme == "https" else 4318)
        host = u.hostname or "127.0.0.1"
        if u.scheme == "https":
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                host, port, timeout=timeout_s, context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
        try:
            conn.request("POST", path, body=body,
                         headers={"Content-Type": "application/json", "Content-Length": str(len(body))})
            resp = conn.getresponse()
            resp.read()
            return 200 <= resp.status < 300
        finally:
            conn.close()
    except Exception:
        return False


class OtelExporter:
    """Sends spend data points from a background thread so the proxy never waits on the collector."""

    def __init__(self, endpoint: str, timeout_s: float = 2.0) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.sent = 0
        self.failed = 0
        self.dropped = 0
        self._q: queue.Queue[Exchange | None] = queue.Queue(maxsize=_QUEUE_MAX)
        self._thread = threading.Thread(target=self._run, name="llmtap-otel", daemon=True)
        self._thread.start()

    def submit(self, ex: Exchange) -> None:
        """Queue a priced exchange for export. Never blocks or raises."""
        if ex.cost_usd <= 0:
            return
        try:
            self._q.put_nowait(ex)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while True:
            ex = self._q.get()
            try:
                if ex is None:
                    return
                if export_spend(self.endpoint, ex, self.timeout_s):
                    self.sent += 1
                else:
                    self.failed += 1
            finally:
                self._q.task_done()

    def close(self, wait_s: float = 2.0) -> None:
        """Stop the worker, giving queued data points up to ``wait_s`` seconds to go out."""
        try:
            self._q.put(None, timeout=wait_s)
        except queue.Full:
            return
        self._thread.join(wait_s)
