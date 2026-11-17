"""HTTP-layer robustness: request framing, interim upstream responses, live-feed lifecycle, concurrency."""
from __future__ import annotations

import json
import socket
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

import pytest

from test_hardening import Up, anth_json, call, last, mk, msg, raw_http, send, sse  # noqa: F401  (mk: fixture)

P = b"/anthropic/v1/messages"
BODY = b'{"a":1}'


def _post(extra: bytes, body: bytes = BODY) -> bytes:
    return b"POST " + P + b" HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n" + extra + b"\r\n" + body


def _responses(out: bytes) -> int:
    return out.count(b"HTTP/1.1 ")


def _wait(cond, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


# ------------------------------------------------------------------------------------------------
# request framing
# ------------------------------------------------------------------------------------------------
def test_conflicting_content_lengths_are_rejected_not_smuggled(mk):
    tap, up = mk()
    smuggled = b"GET /anthropic/v1/smuggled HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"
    body = BODY + smuggled
    out = raw_http(tap.port, _post(b"Content-Length: 7\r\nContent-Length: %d\r\n" % len(body), body), timeout=2)
    assert out.startswith(b"HTTP/1.1 400"), out[:80]
    assert _responses(out) == 1
    assert up.requests == []


def test_identical_duplicate_content_lengths_are_accepted(mk):
    tap, up = mk()
    out = raw_http(tap.port, _post(b"Content-Length: 7\r\nContent-Length: 7\r\nConnection: close\r\n"), timeout=2)
    assert out.startswith(b"HTTP/1.1 200"), out[:80]
    assert up.requests[0]["raw"] == BODY


@pytest.mark.parametrize("value", [b"+7", b"0_7", b"-0", b""])
def test_non_digit_content_length_is_rejected(mk, value):
    # int() accepts signs and underscores; another hop would frame the body differently
    tap, up = mk()
    out = raw_http(tap.port, _post(b"Content-Length: " + value + b"\r\nConnection: close\r\n"), timeout=2)
    assert out.startswith(b"HTTP/1.1 400"), out[:80]
    assert up.requests == []


@pytest.mark.parametrize("size", [b"-7", b"0x7", b"+7", b"0_7"])
def test_malformed_chunk_size_is_rejected(mk, size):
    tap, up = mk()
    payload = _post(b"Transfer-Encoding: chunked\r\nConnection: close\r\n", size + b"\r\n" + BODY + b"\r\n0\r\n\r\n")
    out = raw_http(tap.port, payload, timeout=2)
    assert out.startswith(b"HTTP/1.1 400"), out[:80]
    assert up.requests == []


def test_chunk_longer_than_its_size_is_rejected(mk):
    tap, up = mk()
    out = raw_http(tap.port, _post(b"Transfer-Encoding: chunked\r\nConnection: close\r\n", b"1\r\nab\r\n0\r\n\r\n"),
                   timeout=2)
    assert out.startswith(b"HTTP/1.1 400"), out[:80]
    assert up.requests == []


@pytest.mark.parametrize("te", [b"gzip", b"gzip, chunked", b"identity"])
def test_unsupported_request_transfer_encoding_is_rejected(mk, te):
    tap, up = mk()
    out = raw_http(tap.port, _post(b"Transfer-Encoding: " + te + b"\r\nContent-Length: 7\r\nConnection: close\r\n"),
                   timeout=2)
    assert out.startswith(b"HTTP/1.1 400"), out[:80]
    assert up.requests == []


def test_chunked_plus_content_length_is_served_then_connection_closed(mk):
    tap, up = mk()
    follow = b"GET /anthropic/v1/models HTTP/1.1\r\nHost: x\r\n\r\n"
    payload = _post(b"Transfer-Encoding: chunked\r\nContent-Length: 7\r\n", b"7\r\n" + BODY + b"\r\n0\r\n\r\n") + follow
    out = raw_http(tap.port, payload, timeout=2)
    assert out.startswith(b"HTTP/1.1 200"), out[:80]
    assert _responses(out) == 1 and b"Connection: close" in out
    assert [r["raw"] for r in up.requests] == [BODY]


def test_lying_content_length_does_not_preallocate_memory(mk):
    tap, up = mk()
    tracemalloc.start()
    try:
        s = socket.create_connection(("127.0.0.1", tap.port), timeout=2)
        s.sendall(b"POST " + P + b" HTTP/1.1\r\nHost: x\r\nContent-Length: 400000000\r\n\r\n" + BODY)
        time.sleep(0.3)
        peak = tracemalloc.get_traced_memory()[1]
        s.close()
    finally:
        tracemalloc.stop()
    assert peak < 50 * 2**20, f"{peak / 2**20:.0f} MiB allocated for a 7-byte body"
    assert up.requests == []


def test_pipelined_keepalive_requests_do_not_mix_bodies(mk):
    def fn(h, rec):
        text = json.loads(rec["raw"])["messages"][0]["content"] if rec["raw"] else "models"
        send(h, 200, {"Content-Type": "application/json"}, anth_json(text=text))

    up = Up(fn)
    tap, _ = mk(up)
    b1, b2 = json.dumps(msg("one")).encode(), json.dumps(msg("two")).encode()
    payload = (_post(b"Content-Length: %d\r\n" % len(b1), b1)
               + _post(b"Expect: 100-continue\r\nContent-Length: %d\r\n" % len(b2), b2)
               + b"GET /anthropic/v1/models HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
    out = raw_http(tap.port, payload, timeout=3)
    assert out.count(b"HTTP/1.1 200") == 3
    assert [r["raw"] for r in up.requests] == [b1, b2, b""]
    assert out.index(b'"one"') < out.index(b'"two"')


def test_http10_stream_is_close_delimited(mk):
    up = Up(lambda h, rec: sse(h, [b'data: {"a":1}\n\n'] * 3))
    tap, _ = mk(up)
    body = json.dumps(msg(stream=True)).encode()
    out = raw_http(tap.port, b"POST " + P + b" HTTP/1.0\r\nContent-Type: application/json\r\n"
                   b"Content-Length: %d\r\n\r\n" % len(body) + body, timeout=3)
    head, _, rest = out.partition(b"\r\n\r\n")
    assert b"Transfer-Encoding" not in head and b"Connection: close" in head
    assert rest == b'data: {"a":1}\n\n' * 3


# ------------------------------------------------------------------------------------------------
# upstream responses
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("interim", [b"103 Early Hints\r\nLink: </x>; rel=preload", b"102 Processing"],
                         ids=["103", "102"])
def test_interim_upstream_responses_are_not_relayed_as_final(mk, interim):
    def fn(h, rec):
        h.wfile.write(b"HTTP/1.1 " + interim + b"\r\n\r\n")
        h.wfile.flush()
        send(h, 200, {"Content-Type": "application/json"}, anth_json(text="final"))

    tap, _ = mk(Up(fn))
    status, _, body = call(tap, "POST", P.decode(), msg(), timeout=5)
    assert status == 200 and b"final" in body
    ex = last(tap)
    assert ex.status == 200 and ex.output_text == "final"


def test_client_disconnect_mid_stream_closes_upstream_and_records_once(mk):
    frames = [b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
              b'"delta":{"type":"text_delta","text":"x"}}\n\n'] * 200
    up = Up(lambda h, rec: sse(h, frames, delay=0.01))
    tap, _ = mk(up)
    body = json.dumps(msg(stream=True)).encode()
    s = socket.create_connection(("127.0.0.1", tap.port), timeout=2)
    s.sendall(_post(b"Content-Length: %d\r\n" % len(body), body))
    s.recv(4096)
    s.close()
    assert _wait(lambda: up.requests and up.live == 0, 3.0), "upstream still streaming to a gone client"
    ex = last(tap)
    assert ex.error == "client disconnected" and ex.streamed
    time.sleep(0.2)
    assert tap.store.count() == 1


def test_slow_upstream_first_byte_times_out_with_504(mk):
    gate = threading.Event()
    up = Up(lambda h, rec: (gate.wait(2), send(h, 200, {"Content-Type": "application/json"}, anth_json())))
    tap, _ = mk(up, upstream_timeout_s=0.3)
    t0 = time.time()
    status, _, _ = call(tap, "POST", P.decode(), msg(), timeout=5)
    gate.set()
    assert status == 504 and time.time() - t0 < 1.5


# ------------------------------------------------------------------------------------------------
# concurrency
# ------------------------------------------------------------------------------------------------
def test_many_concurrent_connections_complete_and_release_threads(mk):
    up = Up(lambda h, rec: (time.sleep(0.05), send(h, 200, {"Content-Type": "application/json"}, anth_json())))
    tap, _ = mk(up)
    base = threading.active_count()
    with ThreadPoolExecutor(120) as pool:
        statuses = list(pool.map(lambda i: call(tap, "POST", P.decode(), msg(str(i)), timeout=20)[0], range(120)))
    assert statuses == [200] * 120
    assert _wait(lambda: tap.store.count() == 120, 3.0)
    assert call(tap, "GET", "/_llmtap/api/stats")[0] == 200
    assert _wait(lambda: threading.active_count() <= base + 2, 3.0), threading.active_count() - base
