"""Realistic fake LLM upstream (Anthropic + OpenAI wire formats) for tests and `llmtap demo`.

Speaks HTTP/1.1 on 127.0.0.1 with proper chunked transfer-encoding for SSE streams. Replies are
deterministic (``"Echo: <last user text>"``) unless the prompt contains a magic word:
``MOCK_ERROR_500`` (500 provider error), ``MOCK_ERROR_429`` (429 rate limit), ``MOCK_SLOW`` (sleep 1s).
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

MOCK_MODELS: tuple[str, ...] = (
    "claude-opus-4-1", "claude-sonnet-4-5", "claude-haiku-4-5",
    "gpt-4o", "gpt-4o-mini", "gpt-5", "text-embedding-3-small",
)
EMBED_DIM = 8


# --------------------------------------------------------------------------------------- helpers
def _collect_strings(obj: Any, out: list[str]) -> None:
    """Append every string found in a JSON-ish object (depth first)."""
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_strings(v, out)


def _content_text(content: Any) -> str:
    """Text of a message `content` (string, or list of blocks with text / input_text)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
                elif isinstance(block.get("content"), (str, list)):  # tool_result blocks
                    parts.append(_content_text(block["content"]))
        return "\n".join(p for p in parts if p)
    return ""


def prompt_text(req: dict | None) -> str:
    """All prompt text of a request (system + messages / input / prompt), joined by newlines."""
    if not isinstance(req, dict):
        return ""
    out: list[str] = []
    for key in ("system", "instructions", "prompt"):
        val = req.get(key)
        if isinstance(val, (str, list)):
            out.append(_content_text(val))
    for key in ("messages", "input"):
        val = req.get(key)
        if isinstance(val, str):
            out.append(val)
        elif isinstance(val, list):
            for m in val:
                if isinstance(m, dict) and "content" in m:
                    c = m["content"]
                    out.append(_content_text(c) if isinstance(c, (str, list)) else json.dumps(c))
                elif isinstance(m, dict):
                    strs: list[str] = []
                    _collect_strings(m, strs)  # e.g. function_call_output items
                    out.append(" ".join(strs))
                elif isinstance(m, str):
                    out.append(m)
    return "\n".join(p for p in out if p)


def last_user_text(req: dict | None) -> str:
    """Text of the last user turn (messages API, chat completions, or responses API `input`)."""
    if not isinstance(req, dict):
        return ""
    msgs = req.get("messages")
    if msgs is None:
        msgs = req.get("input")
    if isinstance(msgs, str):
        return msgs
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role", "user") == "user":
                txt = _content_text(m.get("content"))
                if txt:
                    return txt
            elif isinstance(m, str):
                return m
    p = req.get("prompt")
    return p if isinstance(p, str) else ""


def reply_for(req: dict | None) -> str:
    """Deterministic assistant reply for a request."""
    return ("Echo: " + last_user_text(req))[:200]


def count_tokens(text: str) -> int:
    """Mock tokenizer: len(text)//4 + 1."""
    return len(text) // 4 + 1


def split_chunks(text: str) -> list[str]:
    """Split text into word-ish streaming chunks that concatenate back to `text`."""
    parts = re.findall(r"\s*\S+", text)
    if not parts:
        return [text] if text else []
    tail = text[len("".join(parts)):]
    if tail:
        parts[-1] += tail
    return parts


def embedding_for(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic unit-ish pseudo-vector derived from sha256(text)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [struct.unpack(">H", digest[2 * i:2 * i + 2])[0] / 65535.0 * 2 - 1 for i in range(dim)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [round(x / norm, 6) for x in raw]


def _anthropic_error(status: int, message: str) -> dict:
    etype = {429: "rate_limit_error", 500: "api_error", 400: "invalid_request_error",
             404: "not_found_error"}.get(status, "api_error")
    return {"type": "error", "error": {"type": etype, "message": message}}


def _openai_error(status: int, message: str) -> dict:
    etype = {429: "rate_limit_exceeded", 500: "server_error", 400: "invalid_request_error",
             404: "not_found"}.get(status, "server_error")
    return {"error": {"message": message, "type": etype, "param": None, "code": etype}}


# --------------------------------------------------------------------------------------- server
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "llmtap-mock/0.1"
    mock: "MockUpstream"  # set on the subclass created per MockUpstream

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - silence stderr
        return

    # ---- low level writers
    def _send_json(self, status: int, obj: Any, extra: dict[str, str] | None = None) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("request-id", "req_mock_" + uuid.uuid4().hex[:16])
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def _start_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("request-id", "req_mock_" + uuid.uuid4().hex[:16])
        self.end_headers()
        self.wfile.flush()

    def _chunk(self, data: bytes) -> None:
        if not data:
            return
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def _end_stream(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _stream(self, frames: Iterator[bytes]) -> None:
        self._start_stream()
        try:
            for frame in frames:
                self._chunk(frame)
            self._end_stream()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _sleep_token(self) -> None:
        if self.mock.token_delay_s > 0:
            time.sleep(self.mock.token_delay_s)

    # ---- request plumbing
    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            return self.rfile.read(length)
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            buf = b""
            while True:
                line = self.rfile.readline().strip()
                size = int(line.split(b";")[0] or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    break
                buf += self.rfile.read(size)
                self.rfile.readline()
            return buf
        return b""

    def _record(self, method: str, raw: bytes) -> dict | None:
        try:
            parsed = json.loads(raw) if raw else None
        except (ValueError, UnicodeDecodeError):
            parsed = None
        entry = {
            "method": method,
            "path": self.path,
            "headers": {k.lower(): v for k, v in self.headers.items()},
            "body": parsed,
            "raw": raw,
        }
        with self.mock._lock:
            self.mock.requests.append(entry)
        return parsed if isinstance(parsed, dict) else None

    def do_GET(self) -> None:  # noqa: N802
        self._record("GET", b"")
        path = self.path.split("?", 1)[0]
        if path in ("/v1/models", "/models"):
            now = int(time.time())
            self._send_json(200, {"object": "list", "data": [
                {"id": m, "object": "model", "type": "model", "created": now, "owned_by": "llmtap-mock",
                 "display_name": m} for m in MOCK_MODELS]})
        else:
            self._send_json(404, _openai_error(404, f"mock: no route for GET {path}"))

    def do_POST(self) -> None:  # noqa: N802
        raw = self._read_body()
        req = self._record("POST", raw)
        path = self.path.split("?", 1)[0]
        if self.mock.latency_s > 0:
            time.sleep(self.mock.latency_s)
        style = "anthropic" if path.startswith("/v1/messages") or path.startswith("/v1/complete") else "openai"
        if req is None:
            err = _anthropic_error if style == "anthropic" else _openai_error
            self._send_json(400, err(400, "mock: request body must be a JSON object"))
            return
        text = prompt_text(req)
        if "MOCK_SLOW" in text:
            time.sleep(1.0)
        if "MOCK_ERROR_500" in text:
            err = _anthropic_error if style == "anthropic" else _openai_error
            self._send_json(500, err(500, "mock: internal server error (MOCK_ERROR_500)"))
            return
        if "MOCK_ERROR_429" in text:
            err = _anthropic_error if style == "anthropic" else _openai_error
            self._send_json(429, err(429, "mock: rate limited (MOCK_ERROR_429)"), {"retry-after": "1"})
            return
        if path == "/v1/messages/count_tokens":
            self._send_json(200, {"input_tokens": count_tokens(text)})
        elif path == "/v1/messages":
            self._anthropic_messages(req, text)
        elif path in ("/v1/chat/completions", "/chat/completions"):
            self._openai_chat(req, text)
        elif path in ("/v1/responses", "/responses"):
            self._openai_responses(req, text)
        elif path in ("/v1/embeddings", "/embeddings"):
            self._openai_embeddings(req)
        else:
            err = _anthropic_error if style == "anthropic" else _openai_error
            self._send_json(404, err(404, f"mock: no route for POST {path}"))

    # ---- anthropic messages
    def _anthropic_messages(self, req: dict, text: str) -> None:
        model = req.get("model") or "claude-sonnet-4-5"
        reply = reply_for(req)
        in_tok, out_tok = count_tokens(text), count_tokens(reply)
        msg_id = "msg_mock_" + uuid.uuid4().hex[:24]
        if req.get("stream") is not True:
            self._send_json(200, {
                "id": msg_id, "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": reply}],
                "model": model, "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": in_tok, "output_tokens": out_tok,
                          "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            })
            return

        def ev(name: str, data: dict) -> bytes:
            return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode("utf-8")

        def frames() -> Iterator[bytes]:
            yield ev("message_start", {"type": "message_start", "message": {
                "id": msg_id, "type": "message", "role": "assistant", "content": [], "model": model,
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": in_tok, "output_tokens": 1,
                          "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}})
            yield ev("content_block_start", {"type": "content_block_start", "index": 0,
                                             "content_block": {"type": "text", "text": ""}})
            yield ev("ping", {"type": "ping"})
            for piece in split_chunks(reply):
                self._sleep_token()
                yield ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                 "delta": {"type": "text_delta", "text": piece}})
            yield ev("content_block_stop", {"type": "content_block_stop", "index": 0})
            yield ev("message_delta", {"type": "message_delta",
                                       "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                       "usage": {"output_tokens": out_tok}})
            yield ev("message_stop", {"type": "message_stop"})

        self._stream(frames())

    # ---- openai chat completions
    def _openai_chat(self, req: dict, text: str) -> None:
        model = req.get("model") or "gpt-4o-mini"
        reply = reply_for(req)
        in_tok, out_tok = count_tokens(text), count_tokens(reply)
        cid = "chatcmpl-mock" + uuid.uuid4().hex[:20]
        created = int(time.time())
        usage = {"prompt_tokens": in_tok, "completion_tokens": out_tok, "total_tokens": in_tok + out_tok,
                 "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                 "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0}}
        if req.get("stream") is not True:
            self._send_json(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply, "refusal": None},
                             "logprobs": None, "finish_reason": "stop"}],
                "usage": usage, "system_fingerprint": "fp_mock",
            })
            return
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))

        def chunk(delta: dict, finish: str | None = None, with_usage: bool = False) -> bytes:
            obj: dict[str, Any] = {"id": cid, "object": "chat.completion.chunk", "created": created,
                                   "model": model, "system_fingerprint": "fp_mock",
                                   "choices": [{"index": 0, "delta": delta, "logprobs": None,
                                                "finish_reason": finish}]}
            if include_usage:
                obj["usage"] = None
            return f"data: {json.dumps(obj)}\n\n".encode("utf-8")

        def frames() -> Iterator[bytes]:
            yield chunk({"role": "assistant", "content": "", "refusal": None})
            for piece in split_chunks(reply):
                self._sleep_token()
                yield chunk({"content": piece})
            yield chunk({}, "stop")
            if include_usage:
                final = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                         "system_fingerprint": "fp_mock", "choices": [], "usage": usage}
                yield f"data: {json.dumps(final)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        self._stream(frames())

    # ---- openai responses API
    def _openai_responses(self, req: dict, text: str) -> None:
        model = req.get("model") or "gpt-4o-mini"
        reply = reply_for(req)
        in_tok, out_tok = count_tokens(text), count_tokens(reply)
        rid = "resp_mock_" + uuid.uuid4().hex[:24]
        mid = "msg_mock_" + uuid.uuid4().hex[:24]
        created = int(time.time())
        usage = {"input_tokens": in_tok, "input_tokens_details": {"cached_tokens": 0},
                 "output_tokens": out_tok, "output_tokens_details": {"reasoning_tokens": 0},
                 "total_tokens": in_tok + out_tok}
        item = {"type": "message", "id": mid, "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "text": reply, "annotations": []}]}

        def response(status: str, output: list, use: dict | None) -> dict:
            return {"id": rid, "object": "response", "created_at": created, "status": status, "model": model,
                    "output": output, "usage": use, "error": None, "incomplete_details": None}

        if req.get("stream") is not True:
            body = response("completed", [item], usage)
            body["output_text"] = reply
            self._send_json(200, body)
            return
        seq = iter(range(10_000))

        def ev(name: str, data: dict) -> bytes:
            data = {"type": name, "sequence_number": next(seq), **data}
            return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode("utf-8")

        def frames() -> Iterator[bytes]:
            yield ev("response.created", {"response": response("in_progress", [], None)})
            yield ev("response.in_progress", {"response": response("in_progress", [], None)})
            yield ev("response.output_item.added", {"output_index": 0, "item": {
                "type": "message", "id": mid, "status": "in_progress", "role": "assistant", "content": []}})
            yield ev("response.content_part.added", {"item_id": mid, "output_index": 0, "content_index": 0,
                                                     "part": {"type": "output_text", "text": "", "annotations": []}})
            for piece in split_chunks(reply):
                self._sleep_token()
                yield ev("response.output_text.delta", {"item_id": mid, "output_index": 0,
                                                        "content_index": 0, "delta": piece})
            yield ev("response.output_text.done", {"item_id": mid, "output_index": 0, "content_index": 0,
                                                   "text": reply})
            yield ev("response.content_part.done", {"item_id": mid, "output_index": 0, "content_index": 0,
                                                    "part": item["content"][0]})
            yield ev("response.output_item.done", {"output_index": 0, "item": item})
            yield ev("response.completed", {"response": response("completed", [item], usage)})

        self._stream(frames())

    # ---- openai embeddings
    def _openai_embeddings(self, req: dict) -> None:
        model = req.get("model") or "text-embedding-3-small"
        inp = req.get("input", "")
        items = inp if isinstance(inp, list) else [inp]
        texts = [x if isinstance(x, str) else json.dumps(x) for x in items]
        tokens = sum(count_tokens(t) for t in texts)
        self._send_json(200, {
            "object": "list", "model": model,
            "data": [{"object": "embedding", "index": i, "embedding": embedding_for(t)} for i, t in enumerate(texts)],
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        })


class _MockServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128   # bursts of concurrent clients must not overflow the listen backlog


class MockUpstream:
    """A local fake Anthropic/OpenAI server. ``requests`` logs everything received (thread-safe append)."""

    def __init__(self, latency_s: float = 0.0, token_delay_s: float = 0.0) -> None:
        self.latency_s = latency_s
        self.token_delay_s = token_delay_s
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port: int = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "MockUpstream":
        """Bind 127.0.0.1:0 and serve in a daemon thread."""
        handler = type("MockHandler", (_Handler,), {"mock": self})
        self._server = _MockServer(("127.0.0.1", 0), handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, args=(0.05,),
                                        name="llmtap-mock", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and release the port (idempotent)."""
        srv, self._server = self._server, None
        if srv is not None:
            srv.shutdown()
            srv.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def reset(self) -> None:
        """Forget logged requests."""
        with self._lock:
            self.requests.clear()

    def __enter__(self) -> "MockUpstream":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()
