"""OpenTelemetry spend exporter (stdlib only, OTLP/HTTP JSON).
Exports recorded LLM spend as an OTLP counter metric:
- Metric Name: gen_ai.client.cost (OpenTelemetry GenAI semantic convention)
- Cost Unit: USD
- Attributes: provider, model
"""
from __future__ import annotations
import http.client
import json
import ssl
import time
from typing import Any
from urllib.parse import urlsplit
from .models import Exchange
METRIC_NAME = "gen_ai.client.cost"
COST_UNIT = "USD"

def format_otlp_payload(ex: Exchange, timestamp_nano: int | None = None) -> dict[str, Any]:
    """Format an Exchange's spend into a standard OTLP JSON ExportMetricsServiceRequest."""
    ts = timestamp_nano if timestamp_nano is not None else int(time.time() * 1e9)
    attrs = [
        {"key": "provider", "value": {"stringValue": ex.provider or "unknown"}},
        {"key": "model", "value": {"stringValue": ex.model or "unknown"}},
    ]
    return {
        "resourceMetrics": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "llmtap"}}
                    ]
                },
                "scopeMetrics": [
                    {
                        "scope": {"name": "llmtap"},
                        "metrics": [
                            {
                                "name": METRIC_NAME,
                                "description": "LLM token spend in USD",
                                "unit": COST_UNIT,
                                "sum": {
                                    "dataPoints": [
                                        {
                                            "attributes": attrs,
                                            "asDouble": float(ex.cost_usd),
                                            "timeUnixNano": str(ts),
                                        }
                                    ],
                                    "aggregationTemporality": 1,  # AGGREGATION_TEMPORALITY_DELTA
                                    "isMonotonic": True,
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    }


def export_spend(endpoint: str, ex: Exchange, timeout_s: float = 2.0) -> bool:
    """Send spend metric to an OTLP/HTTP collector. Never raises; returns True on success."""
    if not endpoint or ex.cost_usd <= 0:
        return False
    target = endpoint
    if not target.startswith("http://") and not target.startswith("https://"):
        target = "http://" + target
    u = urlsplit(target)
    path = u.path or "/v1/metrics"
    if path == "/":
        path = "/v1/metrics"
    if u.query:
        path = f"{path}?{u.query}"
    payload = format_otlp_payload(ex)
    body = json.dumps(payload).encode("utf-8")
    port = u.port or (443 if u.scheme == "https" else 4318)
    host = u.hostname or "127.0.0.1"
    try:
        if u.scheme == "https":
            conn = http.client.HTTPSConnection(host, port, timeout=timeout_s, context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
        conn.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        resp = conn.getresponse()
        resp.read()
        ok = 200 <= resp.status < 300
        conn.close()
        return ok
    except Exception:
        return False