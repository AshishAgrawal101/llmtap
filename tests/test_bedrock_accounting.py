"""Bedrock calls are LLM calls: priced, budgeted, cached, recorded, guarded - and their signatures survive."""
from __future__ import annotations

import base64
import json
import re
import struct
import zlib
from typing import Any

import pytest

from llmtap import doctor
from llmtap.config import Config
from llmtap.models import Route
from llmtap.pricing import cost_usd
from llmtap.providers import (
    bedrock_route_problem,
    decode_eventstream,
    encode_eventstream,
    eventstream_to_text,
    extract_text,
    extract_usage,
    forward_headers,
    model_from_path,
    parse_bedrock_stream,
    resolve_route,
    text_to_eventstream,
)
from llmtap.proxy import _is_llm_call
from test_hardening import Up, call, last, mk, send  # noqa: F401  (mk: fixture)

ES = "application/vnd.amazon.eventstream"
CLAUDE = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
ARN = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0"
ARN_Q = ARN.replace(":", "%3A").replace("/", "%2F")
MSGS = [{"role": "user", "content": [{"text": "hi"}]}]
SIGV4 = ("AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE/20261117/eu-west-1/bedrock/aws4_request, "
         "SignedHeaders=content-type;host;x-amz-date, Signature=" + "0" * 64)


def frame(event: str, payload: Any, message_type: str = "event", **extra: str) -> bytes:
    h = {":event-type": event, ":content-type": "application/json", ":message-type": message_type, **extra}
    if message_type == "exception":
        h = {":exception-type": event, ":content-type": "application/json", ":message-type": "exception"}
    return encode_eventstream(h, json.dumps(payload).encode())


def chunk(obj: dict) -> bytes:
    return frame("chunk", {"bytes": base64.b64encode(json.dumps(obj).encode()).decode(), "p": "abc"})


CONVERSE_STREAM = [
    frame("messageStart", {"role": "assistant"}),
    frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Hel"}}),
    frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "lo"}}),
    frame("contentBlockStop", {"contentBlockIndex": 0}),
    frame("messageStop", {"stopReason": "end_turn"}),
    frame("metadata", {"usage": {"inputTokens": 1000, "outputTokens": 200, "cacheReadInputTokens": 4000,
                                 "cacheWriteInputTokens": 500, "totalTokens": 5700}, "metrics": {"latencyMs": 9}}),
]
CLAUDE_INVOKE_STREAM = [
    chunk({"type": "message_start", "message": {"model": "claude-sonnet-4-5-20250929", "usage": {
        "input_tokens": 1000, "output_tokens": 1, "cache_read_input_tokens": 4000,
        "cache_creation_input_tokens": 500}}}),
    chunk({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi there"}}),
    chunk({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 200}}),
    chunk({"type": "message_stop", "amazon-bedrock-invocationMetrics": {"inputTokenCount": 1000,
                                                                      "outputTokenCount": 200}}),
]
# 1000 in, 200 out, 4000 cache read, 500 cache write on Sonnet 4.5 ($3 / $15 / $0.30 / $3.75 per MTok)
EXPECTED = (1000 * 3 + 200 * 15 + 4000 * 0.3 + 500 * 3.75) / 1e6


def converse_json(text: str = "Hello", **usage: int) -> bytes:
    u = {"inputTokens": 1000, "outputTokens": 200, "cacheReadInputTokens": 4000, "cacheWriteInputTokens": 500}
    u.update(usage)
    return json.dumps({"output": {"message": {"role": "assistant", "content": [{"text": text}]}},
                       "stopReason": "end_turn", "usage": u, "metrics": {"latencyMs": 9}}).encode()


def bedrock_up(fn=None) -> Up:
    def default(h, rec):
        path = rec["path"].split("?")[0]
        if path.endswith("/converse-stream"):
            send(h, 200, {"Content-Type": ES}, b"".join(CONVERSE_STREAM))
        elif path.endswith("/invoke-with-response-stream"):
            send(h, 200, {"Content-Type": ES}, b"".join(CLAUDE_INVOKE_STREAM))
        else:
            send(h, 200, {"Content-Type": "application/json"}, converse_json())
    return Up(fn or default)


@pytest.fixture
def btap(mk):
    """``btap(up=None, **cfg) -> (tap, up)`` with the bedrock upstream pointed at a local stub."""
    def make(up: Up | None = None, **kw: Any):
        tap, u = mk(up or bedrock_up(), **kw)
        tap.cfg.upstreams["bedrock"] = {"base": u.base_url, "style": "bedrock"}
        return tap, u
    return make


def sigv4_headers(**extra: str) -> dict[str, str]:
    return {"Authorization": SIGV4, "X-Amz-Date": "20261117T120000Z", "Content-Type": "application/json", **extra}


# ------------------------------------------------------------------------------------------------
# EventStream codec
# ------------------------------------------------------------------------------------------------
def test_eventstream_round_trip_and_partial_tail():
    data = b"".join(CONVERSE_STREAM)
    frames, used = decode_eventstream(data)
    assert used == len(data) and [h[":event-type"] for h, _ in frames] == [
        "messageStart", "contentBlockDelta", "contentBlockDelta", "contentBlockStop", "messageStop", "metadata"]
    assert json.loads(frames[1][1]) == {"contentBlockIndex": 0, "delta": {"text": "Hel"}}
    # a partial trailing message is left undecoded, not misread
    frames, used = decode_eventstream(data + CONVERSE_STREAM[0][:20])
    assert len(frames) == 6 and used == len(data)


@pytest.mark.parametrize("pos", [5, 9, 30, -2])
def test_eventstream_corruption_stops_decoding(pos):
    one = bytearray(CONVERSE_STREAM[1])
    one[pos] ^= 0xFF
    frames, used = decode_eventstream(CONVERSE_STREAM[0] + bytes(one) + CONVERSE_STREAM[2])
    assert len(frames) == 1 and used == len(CONVERSE_STREAM[0])


def test_eventstream_decodes_every_header_type():
    hb = b""
    for name, t, val in [("t", 0, b""), ("f", 1, b""), ("b", 2, b"\x7f"), ("s", 3, struct.pack(">h", -2)),
                         ("i", 4, struct.pack(">i", 70000)), ("l", 5, struct.pack(">q", -(2 ** 40))),
                         ("x", 6, struct.pack(">H", 2) + b"\x00\x01"), ("str", 7, struct.pack(">H", 2) + b"ok"),
                         ("ts", 8, struct.pack(">q", 1700000000000)), ("u", 9, bytes(range(16)))]:
        hb += bytes([len(name)]) + name.encode() + bytes([t]) + val
    prelude = struct.pack(">II", 16 + len(hb) + 2, len(hb))
    head = prelude + struct.pack(">I", zlib.crc32(prelude)) + hb + b"{}"
    frames, _ = decode_eventstream(head + struct.pack(">I", zlib.crc32(head)))
    (h, payload), = frames
    assert payload == b"{}"
    assert h == {"t": True, "f": False, "b": 127, "s": -2, "i": 70000, "l": -(2 ** 40), "x": b"\x00\x01",
                 "str": "ok", "ts": 1700000000000, "u": bytes(range(16)).hex()}


def test_botocore_parses_our_frames():
    eventstream = pytest.importorskip("botocore.eventstream")
    buf = eventstream.EventStreamBuffer()
    buf.add_data(b"".join(text_to_eventstream(eventstream_to_text(b"".join(CLAUDE_INVOKE_STREAM)))))
    msgs = list(buf)
    assert [m.headers[":event-type"] for m in msgs] == ["chunk"] * 4
    inner = json.loads(base64.b64decode(json.loads(msgs[1].payload)["bytes"]))
    assert inner["delta"]["text"] == "Hi there"


def test_eventstream_text_is_readable_and_replayable():
    text = eventstream_to_text(b"".join(CLAUDE_INVOKE_STREAM))
    first = json.loads(text.splitlines()[0])
    assert first["headers"][":event-type"] == "chunk" and first["chunk"]["type"] == "message_start"
    assert "bytes" not in first["payload"] and first["payload"] == {"p": "abc"}
    again = eventstream_to_text(b"".join(text_to_eventstream(text)))
    assert again == text


# ------------------------------------------------------------------------------------------------
# parsing
# ------------------------------------------------------------------------------------------------
def test_stream_usage_text_and_errors():
    u, text, _, err = parse_bedrock_stream(eventstream_to_text(b"".join(CONVERSE_STREAM)))
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens) == (1000, 200, 4000, 500)
    assert text == "Hello" and err is None
    u, text, model, _ = parse_bedrock_stream(eventstream_to_text(b"".join(CLAUDE_INVOKE_STREAM)))
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens) == (1000, 200, 4000, 500)
    assert text == "Hi there" and model == "claude-sonnet-4-5-20250929"
    # Llama chunks carry no usage until the final invocation metrics
    llama = [chunk({"generation": "Hey", "prompt_token_count": 7, "generation_token_count": 1}),
             chunk({"generation": "!", "generation_token_count": 2,
                    "amazon-bedrock-invocationMetrics": {"inputTokenCount": 7, "outputTokenCount": 2}})]
    u, text, _, _ = parse_bedrock_stream(eventstream_to_text(b"".join(llama)))
    assert (u.input_tokens, u.output_tokens, text) == (7, 2, "Hey!")
    broken = CONVERSE_STREAM[:2] + [frame("throttlingException", {"message": "Too many tokens"}, "exception")]
    _, _, _, err = parse_bedrock_stream(eventstream_to_text(b"".join(broken)))
    assert err == "throttlingException: Too many tokens"


@pytest.mark.parametrize(("body", "expected"), [
    (json.loads(converse_json()), (1000, 200, 4000, 500, "Hello")),
    ({"id": "msg_1", "type": "message", "model": "claude-sonnet-4-5-20250929",
      "content": [{"type": "text", "text": "Hi"}],
      "usage": {"input_tokens": 9, "output_tokens": 4, "cache_read_input_tokens": 2}}, (9, 4, 2, 0, "Hi")),
    ({"generation": "Yo", "prompt_token_count": 11, "generation_token_count": 3}, (11, 3, 0, 0, "Yo")),
    ({"inputTextTokenCount": 5, "results": [{"tokenCount": 6, "outputText": "Titan"}]}, (5, 6, 0, 0, "Titan")),
    ({"output": {"message": {"content": [{"text": "Nova"}]}},
      "usage": {"inputTokens": 3, "outputTokens": 2, "cacheReadInputTokenCount": 1}}, (3, 2, 1, 0, "Nova")),
    ({"outputs": [{"text": "Mistral", "stop_reason": "stop"}]}, (0, 0, 0, 0, "Mistral")),
])
def test_invoke_and_converse_bodies(body, expected):
    u = extract_usage("bedrock", body)
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens,
            extract_text("bedrock", body)) == expected


def test_model_from_bedrock_paths_and_llm_calls():
    assert model_from_path(f"/model/{CLAUDE.replace(':', '%3A')}/converse") == CLAUDE
    assert model_from_path(f"/model/{ARN_Q}/invoke") == ARN
    assert model_from_path("/model/amazon.nova-pro-v1:0/converse-stream?x=1") == "amazon.nova-pro-v1:0"
    for op in ("invoke", "invoke-with-response-stream", "converse", "converse-stream"):
        assert _is_llm_call("POST", f"/model/{ARN_Q}/{op}")
    assert not _is_llm_call("POST", "/model/x/count-tokens") and not _is_llm_call("GET", "/model/x/converse")
    assert cost_usd(ARN, extract_usage("bedrock", {"usage": {"inputTokens": 1_000_000}}), style="bedrock") == 1.0


# ------------------------------------------------------------------------------------------------
# routing
# ------------------------------------------------------------------------------------------------
def _short_term_key(region: str) -> str:
    url = (f"bedrock.amazonaws.com/?Action=CallWithBearerToken&X-Amz-Algorithm=AWS4-HMAC-SHA256"
           f"&X-Amz-Credential=ASIAEXAMPLE%2F20261117%2F{region}%2Fbedrock%2Faws4_request&Version=1")
    return "bedrock-api-key-" + base64.b64encode(url.encode()).decode()


@pytest.mark.parametrize(("auth", "env", "base"), [
    (SIGV4, None, "https://bedrock-runtime.eu-west-1.amazonaws.com"),
    ("Bearer " + _short_term_key("ap-southeast-2"), None, "https://bedrock-runtime.ap-southeast-2.amazonaws.com"),
    ("Bearer ABSKQmVkcm9ja0FQSUtleS1leGFtcGxl", "us-west-2", "https://bedrock-runtime.us-west-2.amazonaws.com"),
    ("Bearer ABSKQmVkcm9ja0FQSUtleS1leGFtcGxl", "cn-north-1", "https://bedrock-runtime.cn-north-1.amazonaws.com.cn"),
    ("Bearer ABSKQmVkcm9ja0FQSUtleS1leGFtcGxl", None, "https://bedrock-runtime.{region}.amazonaws.com"),
])
def test_region_for_requests_sent_to_llmtap(monkeypatch, auth, env, base):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    if env:
        monkeypatch.setenv("AWS_REGION", env)
    for path in (f"/model/{ARN_Q}/converse", f"/bedrock/model/{ARN_Q}/converse"):
        route = resolve_route(path, {"Host": "127.0.0.1:4141", "Authorization": auth}, Config())
        assert (route.provider, route.style, route.upstream_base) == ("bedrock", "bedrock", base)
        assert route.upstream_path == f"/model/{ARN_Q}/converse"


def test_route_problems(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    route = resolve_route("/model/x/converse", {"Authorization": "Bearer ABSKxx"}, Config())
    assert "AWS_REGION" in (bedrock_route_problem(route, "/model/x/converse", {}) or "")
    signed = Route(provider="bedrock", upstream_base="https://b", upstream_path="/model/x/converse", style="bedrock")
    assert bedrock_route_problem(signed, "/model/x/converse", {"Authorization": SIGV4}) is None
    assert "endpoint_url" in (bedrock_route_problem(signed, "/bedrock/model/x/converse", {"Authorization": SIGV4})
                              or "")
    assert bedrock_route_problem(signed, "/bedrock/model/x/converse", {"Authorization": "Bearer k"}) is None


@pytest.mark.parametrize("host", ["bedrock-runtime.s3.amazonaws.com", "bedrock-runtime.us-east-1.amazonaws.com:8443",
                                  "bedrock-runtime.us-east-1.amazonaws.com.evil.com"])
def test_lookalike_bedrock_hosts(host):
    assert resolve_route("/v1/chat/completions", {"Host": host}, Config()).provider == "openai"


def test_signed_llmtap_headers_are_forwarded():
    auth = SIGV4.replace("x-amz-date", "x-amz-date;x-llmtap-tag-team")
    h = {"Authorization": auth, "X-LLMTap-Tag-Team": "a", "X-LLMTap-Session": "s"}
    out = {k.lower() for k in forward_headers(h, preserve_signed=True)}
    assert "x-llmtap-tag-team" in out and "x-llmtap-session" not in out
    assert "x-llmtap-tag-team" not in {k.lower() for k in forward_headers(h)}


def test_doctor_does_not_probe_the_region_template(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    probed: list[str] = []
    monkeypatch.setattr(doctor, "probe_host", lambda base, timeout: (probed.append(base), (True, "ok"))[1])
    checks = {c.title: c for c in doctor.check_online(Config())}
    assert not any("{" in b for b in probed)
    assert checks["bedrock not checked (its endpoint depends on the AWS region)"].level == doctor.INFO
    monkeypatch.setenv("AWS_REGION", "eu-central-1")
    probed.clear()
    doctor.check_online(Config())
    assert "https://bedrock-runtime.eu-central-1.amazonaws.com" in probed


# ------------------------------------------------------------------------------------------------
# through the proxy
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("op", ["converse", "converse-stream", "invoke-with-response-stream"])
def test_calls_are_priced_with_text_and_raw_bytes_relayed(btap, op):
    tap, up = btap()
    status, h, body = call(tap, "POST", f"/model/{CLAUDE.replace(':', '%3A')}/{op}",
                           {"messages": MSGS}, sigv4_headers())
    assert status == 200
    if op == "converse-stream":
        assert body == b"".join(CONVERSE_STREAM) and h["content-type"] == ES
    elif op == "invoke-with-response-stream":
        assert body == b"".join(CLAUDE_INVOKE_STREAM)
    ex = last(tap)
    assert (ex.provider, ex.style, ex.source) == ("bedrock", "bedrock", "upstream")
    assert ex.model in (CLAUDE, "claude-sonnet-4-5-20250929")
    assert ex.output_text in ("Hello", "Hi there") and ex.streamed == (op != "converse")
    assert ex.cost_usd == pytest.approx(EXPECTED)
    assert '"headers"' in ex.response_body or '"output"' in ex.response_body


def test_invoke_usage_from_headers(btap):
    def fn(h, rec):
        send(h, 200, {"Content-Type": "application/json", "X-Amzn-Bedrock-Input-Token-Count": "1000",
                      "X-Amzn-Bedrock-Output-Token-Count": "100"},
             json.dumps({"completion": "ok", "stop_reason": "stop_sequence"}).encode())
    tap, _ = btap(bedrock_up(fn))
    call(tap, "POST", "/model/anthropic.claude-sonnet-4-5-20250929-v1:0/invoke", {"prompt": "hi"}, sigv4_headers())
    ex = last(tap)
    assert (ex.usage.input_tokens, ex.usage.output_tokens, ex.output_text) == (1000, 100, "ok")
    assert ex.cost_usd == pytest.approx((1000 * 3 + 100 * 15) / 1e6)


def test_in_band_exception_is_an_error_and_not_cached(btap):
    def fn(h, rec):
        send(h, 200, {"Content-Type": ES}, b"".join(CONVERSE_STREAM[:2]) + frame(
            "modelStreamErrorException", {"message": "boom"}, "exception"))
    tap, up = btap(bedrock_up(fn), cache=True)
    for _ in range(2):
        call(tap, "POST", "/model/x/converse-stream", {"messages": MSGS}, sigv4_headers())
    ex = last(tap, 2)
    assert ex.source == "upstream" and ex.error == "modelStreamErrorException: boom"
    assert len(up.requests) == 2


def test_budget_stops_bedrock_spend(btap):
    tap, up = btap(budget_session_usd=EXPECTED * 1.5)
    path = f"/model/{CLAUDE}/converse"
    assert call(tap, "POST", path, {"messages": MSGS, "n": 1}, sigv4_headers())[0] == 200
    assert call(tap, "POST", path, {"messages": MSGS, "n": 2}, sigv4_headers())[0] == 200
    status, h, body = call(tap, "POST", path, {"messages": MSGS, "n": 3}, sigv4_headers())
    assert status == 429 and json.loads(body)["__type"] == "llmtap_budget_exceeded"
    assert len(up.requests) == 2


@pytest.mark.parametrize("op", ["converse", "converse-stream", "invoke-with-response-stream"])
def test_cache_and_cassette_replay(btap, tmp_path, op):
    cas = tmp_path / "bedrock.jsonl"
    path = f"/model/{CLAUDE}/{op}"
    tap, up = btap(cassette=str(cas), mode="record")
    _, _, recorded = call(tap, "POST", path, {"messages": MSGS}, sigv4_headers())
    tap.shutdown()
    tap2, up2 = btap(cassette=str(cas), mode="replay")
    status, h, replayed = call(tap2, "POST", path, {"messages": MSGS}, sigv4_headers())
    assert status == 200 and h["x-llmtap-source"] == "cassette" and not up2.requests
    if op == "converse":
        assert json.loads(replayed) == json.loads(recorded)
    else:
        assert h["content-type"] == ES
        assert eventstream_to_text(replayed) == eventstream_to_text(recorded)
    ex = last(tap2)
    assert ex.output_text in ("Hello", "Hi there") and ex.saved_usd == pytest.approx(EXPECTED)


def test_guard_redact_refuses_signed_requests_but_redacts_api_key_ones(btap):
    tap, up = btap(guard="redact")
    secret = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    body = {"messages": [{"role": "user", "content": [{"text": f"my token {secret}"}]}]}
    status, _, resp = call(tap, "POST", "/model/x/converse", body, sigv4_headers())
    assert status == 400 and "SigV4" in json.loads(resp)["message"] and not up.requests
    assert secret not in last(tap).request_body
    status, _, _ = call(tap, "POST", "/model/x/converse", body,
                        {"Authorization": "Bearer ABSKexample", "Content-Type": "application/json"})
    assert status == 200
    sent = up.requests[-1]
    assert secret.encode() not in sent["raw"] and int(sent["headers"]["content-length"]) == len(sent["raw"])


def test_no_region_and_signed_prefix_route_are_explained(mk, monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    tap, up = mk()
    status, _, body = call(tap, "POST", "/model/x/converse", {"messages": MSGS}, {"Authorization": "Bearer ABSKxx"})
    assert status == 400 and "AWS_REGION" in json.loads(body)["message"]
    status, _, body = call(tap, "POST", "/bedrock/model/x/converse", {"messages": MSGS}, sigv4_headers())
    assert status == 400 and "endpoint_url" in json.loads(body)["message"]
    assert not up.requests


# ------------------------------------------------------------------------------------------------
# real SigV4 signatures (botocore)
# ------------------------------------------------------------------------------------------------
def _signature_ok(rec: dict, secret: str = "secret") -> bool:
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials

    hd = rec["headers"]
    auth = hd["authorization"]
    signed = re.search(r"SignedHeaders=([^,\s]+)", auth).group(1).split(";")
    key, _, region, service = re.search(r"Credential=([^/]+)/(\d{8})/([^/]+)/([^/]+)/", auth).groups()
    req = AWSRequest(method=rec["method"], url=f"http://{hd['host']}{rec['path']}", data=rec["raw"],
                     headers={k: hd[k] for k in signed})
    req.context["timestamp"] = hd["x-amz-date"]
    signer = SigV4Auth(Credentials(key, secret), service, region)
    sig = signer.signature(signer.string_to_sign(req, signer.canonical_request(req)), req)
    return sig == re.search(r"Signature=([0-9a-f]{64})", auth).group(1)


@pytest.fixture
def client_for():
    botocore_session = pytest.importorskip("botocore.session")
    from botocore.config import Config as BotoConfig

    def make(tap, **kw):
        return botocore_session.get_session().create_client(
            "bedrock-runtime", region_name="us-east-1", endpoint_url=f"http://127.0.0.1:{tap.port}",
            aws_access_key_id="AKIAEXAMPLE", aws_secret_access_key="secret",
            config=BotoConfig(retries={"max_attempts": 1}), **kw)
    return make


def test_botocore_signatures_survive_the_proxy(btap, client_for):
    tap, up = btap(cache=True)
    client = client_for(tap)
    r = client.converse(modelId=ARN, messages=MSGS, inferenceConfig={"maxTokens": 64})
    assert r["output"]["message"]["content"] == [{"text": "Hello"}]
    events = [next(iter(e)) for e in client.converse_stream(modelId=CLAUDE, messages=MSGS)["stream"]]
    assert events == ["messageStart", "contentBlockDelta", "contentBlockDelta", "contentBlockStop", "messageStop",
                      "metadata"]
    assert len(up.requests) == 2 and all(_signature_ok(rec) for rec in up.requests)
    assert up.requests[0]["path"] == f"/model/{ARN_Q}/converse"
    # the verifier is not a rubber stamp
    tampered = dict(up.requests[0], raw=up.requests[0]["raw"].replace(b"hi", b"ho"))
    assert not _signature_ok(tampered)
    # a cached stream replays as EventStream frames botocore can parse
    again = [next(iter(e)) for e in client.converse_stream(modelId=CLAUDE, messages=MSGS)["stream"]]
    assert again == events and len(up.requests) == 2
    ex = last(tap, 3)
    assert ex.source == "cache" and ex.output_text == "Hello"


def test_botocore_sees_llmtap_refusals(btap, client_for):
    from botocore.exceptions import ClientError

    tap, up = btap(guard="redact")
    secret = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    with pytest.raises(ClientError) as info:
        client_for(tap).converse(modelId=CLAUDE, messages=[{"role": "user", "content": [{"text": secret}]}])
    err = info.value.response["Error"]
    assert err["Code"] == "llmtap_guard_blocked" and "SigV4" in err["Message"]
    assert not up.requests
