# llmtap — build contract (internal, for contributors)

llmtap is a **local HTTP proxy** between any app/agent and LLM APIs (Anthropic, OpenAI, any
OpenAI-compatible upstream). Apps point their base URL at it (or run `llmtap wrap -- <cmd>`, which
sets the env vars) and llmtap records, prices, caches, replays, guards and budgets every call.

HARD RULES
- **Python >= 3.11 stdlib only.** No third-party runtime deps. Tests use pytest only.
- Never make real network calls in tests. Use local fake upstream servers on 127.0.0.1 port 0.
- Only edit the files you own. Import other modules exactly by the signatures below.
- Already written (read them, don't rewrite): `llmtap/models.py`, `llmtap/config.py`, `llmtap/__init__.py`, `llmtap/__main__.py`, `pyproject.toml`.
- Type hints everywhere, small docstrings, no prints in library code except `term.py` / `cli.py`.
- Thread-safety: the proxy uses `ThreadingHTTPServer`, so Store/Cache/Budget must be thread-safe.

Default port **4141**. Internal routes live under `/_llmtap/` (dashboard + JSON API).

---------------------------------------------------------------------------------------------------
## llmtap/providers.py — routing + wire-format knowledge
```python
def resolve_route(path: str, headers: Mapping[str, str], cfg: Config) -> Route
```
Routing rules (first match wins):
1. Explicit prefix: `/<name>/rest` where `<name>` is a key in `cfg.upstreams` → Route(provider=name,
   upstream_base=cfg.upstreams[name]["base"], upstream_path="/rest" (keep query string), style=that upstream's style).
   e.g. `/openrouter/v1/chat/completions` → base `https://openrouter.ai/api`, path `/v1/chat/completions`.
2. Path starts with `/v1/messages` or `/v1/complete` → anthropic.
3. Headers contain `anthropic-version` or `x-api-key` (case-insensitive) → anthropic.
4. Otherwise → openai.
Paths without `/v1` prefix (e.g. `/chat/completions`, when user sets OPENAI_BASE_URL without /v1) get `/v1` prepended for openai style.

```python
def parse_json(body: bytes) -> dict | None              # None if not a JSON object
def request_model(req: dict | None) -> str | None
def is_stream_request(req: dict | None) -> bool          # req.get("stream") is True
def extract_usage(style: str, resp: dict) -> Usage       # non-streaming response JSON
def extract_text(style: str, resp: dict) -> str          # assistant text (concat text blocks / choices[0].message.content / responses-API output_text)
def response_model(style: str, resp: dict) -> str | None
def parse_sse(style: str, raw: str) -> tuple[Usage, str, str | None]
    # Parse a full raw SSE transcript ("event: x\ndata: {...}\n\n" ...). Return (usage, assembled_text, model).
    # anthropic: message_start.message.usage (input/cache tokens, model), message_delta.usage.output_tokens,
    #            content_block_delta.delta.text. openai chat: choices[0].delta.content, final chunk "usage"
    #            (prompt_tokens/completion_tokens/prompt_tokens_details.cached_tokens), ignore "data: [DONE]".
    #            openai responses API: event response.output_text.delta (delta), response.completed (response.usage).
def estimate_tokens(text: str) -> int                    # max(1, len(text)//4) for non-empty, else 0
def estimate_usage(req: dict | None, output_text: str) -> Usage   # estimated=True; input from all strings in req
def error_body(style: str, status: int, message: str, err_type: str = "llmtap_error") -> bytes
    # Provider-shaped JSON error so SDKs surface it nicely:
    # anthropic: {"type":"error","error":{"type":err_type,"message":...}}
    # openai:    {"error":{"message":...,"type":err_type,"code":err_type}}
HOP_BY_HOP: frozenset[str]   # lowercased: connection, keep-alive, proxy-authenticate, proxy-authorization, te, trailers, transfer-encoding, upgrade, host, content-length, accept-encoding
def forward_headers(headers: Mapping[str, str]) -> dict[str, str]   # drop HOP_BY_HOP and x-llmtap-*; force "accept-encoding: identity"
def tags_from_headers(headers: Mapping[str, str]) -> dict[str, str] # "X-LLMTap-Tag-Foo: bar" -> {"foo":"bar"}; "X-LLMTap-Session: s" -> {"session":"s"}
```
Usage field mapping: anthropic `input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens`;
openai chat `prompt_tokens` (minus nothing), `completion_tokens`, `prompt_tokens_details.cached_tokens`→cache_read;
openai responses/embeddings `input_tokens`/`output_tokens` or `prompt_tokens`. Missing → 0.

## llmtap/pricing.py
```python
PRICES: dict[str, dict[str, float]]    # model-prefix -> {"input","output","cache_read","cache_write"} USD per 1M tokens
def price_for(model: str | None, overrides: dict | None = None) -> dict[str, float] | None
    # exact match, then longest prefix match (strip provider prefixes like "anthropic/" or "openai/"), overrides win. None if unknown.
def cost_usd(model: str | None, usage: Usage, overrides: dict | None = None) -> float   # 0.0 if unknown model
def format_usd(x: float) -> str        # "$0.0042", "$1.23", "$0.00" (4 decimals under $1 if nonzero, else 2)
```
Cover Claude (opus/sonnet/haiku 3.x, 4.x, 4.5, 5 families), GPT-4o/4o-mini/4.1/4.1-mini/nano, o1/o3/o4-mini, gpt-5/5-mini/5-nano,
text-embedding-3-*, Gemini 2.x via openrouter names, llama/mixtral/deepseek common names. If cache_read missing default to 10% of input
(anthropic) / 50% (openai); cache_write default 125% of input. Local models (ollama: llama3, qwen, etc.) cost 0.

## llmtap/store.py — SQLite persistence
```python
class Store:
    def __init__(self, path: str)                 # ":memory:" ok; check_same_thread=False + a Lock; WAL mode for files
    def record(self, ex: Exchange) -> None        # insert (or replace by id)
    def get(self, id: str) -> Exchange | None     # also accepts unique id prefix (>=4 chars)
    def list(self, limit: int = 50, offset: int = 0, q: str | None = None, since: float | None = None,
             provider: str | None = None, model: str | None = None, source: str | None = None) -> list[Exchange]
        # newest first. q = substring search over request_body/output_text/model/path.
    def stats(self, since: float | None = None, group_by: str | None = None) -> dict
        # {"requests","input_tokens","output_tokens","cost_usd","saved_usd","avg_latency_ms","errors",
        #  "cache_hits","blocked","guard_findings", "groups": [{"key":..,"requests":..,"cost_usd":..,"input_tokens":..,"output_tokens":..}] }
        # group_by in {None,"model","provider","source","day","hour","tag:<name>"}; groups sorted by cost desc (day/hour: by key asc)
    def spent_since(self, since: float) -> float  # sum cost_usd where source=="upstream"
    def timeseries(self, since: float, bucket_s: int = 3600) -> list[dict]   # [{"t": bucket_start, "requests", "cost_usd", "tokens"}] asc
    def count(self) -> int
    def clear(self) -> None
    def close(self) -> None
```
Columns: all Exchange fields; usage flattened (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, usage_estimated);
tags & guard_findings as JSON text. Index on ts.

## llmtap/cache.py — response cache (dev money saver)
```python
def cache_key(provider: str, path: str, req_body: bytes) -> str
    # sha256 hex of provider + "\n" + path (without query) + "\n" + canonical JSON (sort_keys, compact) of body with
    # volatile keys removed at top level: "user", "metadata", "request_id". If body isn't JSON, hash raw bytes.
class ResponseCache:
    def __init__(self, path: str, ttl_s: int = 0)     # separate sqlite file or same db file (own table "cache"); thread-safe
    def get(self, key: str) -> CachedResponse | None  # honors ttl
    def put(self, key: str, resp: CachedResponse) -> None   # only call for 2xx
    def stats(self) -> dict                           # {"entries": n, "hits": n}  (hits counted in-process)
    def clear(self) -> None
```

## llmtap/cassette.py — VCR for LLMs (deterministic, keyless CI)
```python
class CassetteMiss(Exception): ...
class Cassette:
    def __init__(self, path: str, mode: str = "auto")  # mode: "record" | "replay" | "auto"
        # loads existing JSONL lines if file exists. "record" mode starts fresh (truncate on first write).
    def lookup(self, key: str) -> CachedResponse | None
        # replay/auto: return recorded response. Identical keys recorded multiple times are replayed in order
        # (1st call → 1st recording, 2nd → 2nd, then the last one repeats). record mode: always None.
    def should_record(self) -> bool                    # record, or auto
    def record(self, key: str, request: dict | str, resp: CachedResponse) -> None   # append a JSON line, flush; thread-safe
    def __len__(self) -> int
```
Line format: `{"key":..., "request": <parsed json or str>, "response": {"status","headers","body","streamed","cost_usd"}, "recorded_at": iso8601}`.
Proxy semantics: replay mode + miss → `CassetteMiss` handled by proxy as HTTP 599-ish? No: return **HTTP 409** with provider error
"llmtap cassette miss (replay mode): no recording for this request. Re-record with --mode record" unless cfg.allow_miss.

## llmtap/guard.py — secret/PII firewall for prompts
```python
@dataclass
class Finding: kind: str; path: str; preview: str; start: int; end: int   # preview = masked e.g. "sk-ant-…a1b2"
    def to_dict(self) -> dict
def scan_text(text: str, pii: bool = False) -> list[Finding]               # path = ""
def scan_obj(obj: Any, pii: bool = False) -> list[Finding]                 # walk dict/list strings; path like "messages[0].content"
def redact_obj(obj: Any, pii: bool = False) -> tuple[Any, list[Finding]]   # returns NEW object with matches replaced by "[REDACTED:<kind>]"
def apply_guard(body: bytes, mode: str, pii: bool = False) -> tuple[bytes, list[Finding], bool]
    # -> (body_to_send, findings, blocked). off: (body, [], False). warn: findings, unchanged body.
    # redact: re-serialized JSON with redactions (only if findings). block: blocked=True if findings.
    # Non-JSON bodies: scan as text; redact operates on text.
```
Kinds: aws_access_key (AKIA/ASIA…), aws_secret_key (contextual), anthropic_key (sk-ant-…), openai_key (sk-…/sk-proj-…),
github_token (ghp_/gho_/ghs_/github_pat_), slack_token (xox[baprs]-), stripe_key (sk_live_/rk_live_), google_api_key (AIza…),
private_key (-----BEGIN … PRIVATE KEY-----), jwt, generic_secret (password/secret/token/api_key = "<≥8 chars>").
PII (only if pii=True): email, phone (intl/US), credit_card (Luhn-validated). Keep false positives low; don't flag
"sk-" inside normal words; JWT must be 3 base64url segments starting "eyJ".

## llmtap/budget.py — spend caps + runaway-agent loop breaker
```python
@dataclass
class Verdict: allowed: bool; reason: str = ""; kind: str = ""    # kind: "budget" | "loop" | ""
class Budget:
    def __init__(self, cfg: Config, spent_since: Callable[[float], float])   # spent_since = store.spent_since
    def check(self, key: str) -> Verdict
        # 1) daily: spent_since(local midnight) >= cfg.budget_daily_usd > 0 → deny "daily budget $X reached ($Y spent)"
        # 2) session: in-process spent >= cfg.budget_session_usd > 0 → deny
        # 3) loop: if cfg.loop_threshold > 0 and this key was seen >= loop_threshold times in last loop_window_s → deny
        #    "loop detected: identical request sent N times in Ws". (Records the attempt timestamp every call.)
    def add_spend(self, usd: float) -> None   # session accumulator (call after each upstream response)
    def session_spent(self) -> float
```
Proxy returns HTTP 429 with provider-shaped error, source="budget" or "loop".

## llmtap/proxy.py — the core server
```python
class LLMTap:
    def __init__(self, cfg: Config, store: Store | None = None)   # builds Store(cfg.resolved_db_path()), cache (if cfg.cache), cassette (if cfg.cassette), Budget
    cfg; store; cache: ResponseCache | None; cassette: Cassette | None; budget: Budget; bus: EventBus
    def handle(self, method: str, path: str, headers: Mapping[str,str], body: bytes, respond: Responder) -> Exchange
    def serve_forever(self) -> None
    def start_background(self) -> "LLMTap"   # serve in daemon thread; sets self.port to actual port (cfg.port 0 = ephemeral)
    def shutdown(self) -> None
    port: int; base_url: str  (property)
class EventBus:   # pub/sub for the live dashboard
    def publish(self, event: dict) -> None
    def subscribe(self) -> queue.Queue       # each subscriber gets its own Queue
    def unsubscribe(self, q) -> None
```
Pipeline per request (non-`/_llmtap` paths): resolve_route → parse JSON → tags → apply_guard (block → 400 provider error,
source="blocked") → cache_key → budget.check (deny → 429) → cassette.lookup / cache.get (serve stored body verbatim; streamed bodies
are re-sent as `text/event-stream` chunked; source="cassette"/"cache"; saved_usd = stored cost) → forward upstream with
http.client (HTTPS via ssl.create_default_context(); honor timeout) → for `text/event-stream` responses, relay each chunk to the client
immediately (chunked transfer-encoding, flush) while accumulating; record ttft_ms → compute usage (parse_sse / extract_usage, fall back to
estimate_usage) → cost → budget.add_spend → cache.put/cassette.record for 2xx → store.record → bus.publish({"type":"exchange","exchange": summary dict})
→ term.log_exchange(ex) unless cfg.quiet. Upstream connection errors → 502 provider-shaped error, source="error".
Response headers to client: pass upstream headers except hop-by-hop; add `x-llmtap-id: <exchange id>` and `x-llmtap-source`.
Paths under `/_llmtap/` are delegated to `dashboard.handle_internal(tap, method, path, headers, body, respond) -> bool`.
`GET /` with no route → redirect 302 to `/_llmtap/`.

Responder interface (so dashboard and tests don't care about http.server):
```python
class Responder(Protocol):
    def start(self, status: int, headers: dict[str, str]) -> None
    def write(self, chunk: bytes) -> None   # when started with "transfer-encoding: chunked", writes a chunk frame
    def end(self) -> None
```

## llmtap/dashboard.py + llmtap/ui/index.html — live web UI
```python
def handle_internal(tap: "LLMTap", method: str, path: str, headers, body: bytes, respond: Responder) -> bool
```
Routes: `GET /_llmtap/` → ui/index.html (read via importlib.resources or Path(__file__).parent/"ui"); `GET /_llmtap/api/stats?since=<secs>&group_by=`
→ store.stats + {"session_spent","cache": cache.stats() or null, "cassette": {"path","mode","entries"} or null, "config": {guard, cache, budget_daily_usd, loop_threshold}};
`GET /_llmtap/api/exchanges?limit&offset&q&provider&model&source` → {"items":[summary]}; `GET /_llmtap/api/exchanges/<id>` → full exchange dict;
`GET /_llmtap/api/timeseries?since&bucket` ; `GET /_llmtap/api/events` → SSE stream (`data: <json>\n\n` per bus event, `: ping` every 15s);
`POST /_llmtap/api/clear` → store.clear(); `GET /_llmtap/health` → {"ok":true,"version"}.
summary dict = exchange dict minus request_body/response_body, plus "preview": last user message text (≤200 chars) and "output_preview" (≤200 chars).
UI: single self-contained file, vanilla JS, no CDN, works offline. Dark, gorgeous, dense.

## llmtap/term.py — pretty terminal output (ANSI, auto-disable when not a TTY or NO_COLOR)
```python
def banner(cfg: Config, base_url: str) -> str         # startup box: URLs, env exports to copy, enabled features
def log_exchange(ex: Exchange) -> None                 # one colored line to stderr: time, source badge, provider/model, status, tokens in→out, cost, latency, guard warnings
def summary(stats: dict, session_spent: float) -> str  # end-of-run report used by `llmtap wrap`
def table(rows: list[list[str]], headers: list[str]) -> str
```

## llmtap/cli.py — `llmtap` command
`main(argv: list[str] | None = None) -> int`. Subcommands:
- `serve` [--host --port --db --cache --cache-ttl --cassette --mode --allow-miss --guard --pii --budget-daily --budget-session --loop-threshold --loop-window --quiet --config]
- `wrap [same flags] -- <cmd...>`: start LLMTap on port 0 in background, run cmd with env OPENAI_BASE_URL=<base>/v1, OPENAI_API_BASE=<base>/v1,
  ANTHROPIC_BASE_URL=<base>, LLMTAP_URL=<base>; propagate exit code; print term.summary for that session to stderr.
- `env [--port]` → prints `export ...` lines (for `eval "$(llmtap env)"`).
- `stats [--since 24h|7d|30m|all] [--by model|provider|source|day|hour|tag:x]`, `ls [-n 20] [-q text]`, `show <id> [--raw]`,
  `tail` (poll store every 0.5s, print new via term.log_exchange), `export [--format jsonl|csv] [-o file]`, `prices [filter]`,
  `scan [file|-]` (guard.scan_text on a file/stdin, exit 1 if findings), `clear [--yes]`, `version`.
Time strings parsed by `parse_since("24h") -> float|None` (epoch cutoff).

## llmtap/mock.py — realistic fake LLM upstream (used by tests AND `llmtap demo`)
```python
class MockUpstream:
    def __init__(self, latency_s: float = 0.0, token_delay_s: float = 0.0)
    def start(self) -> "MockUpstream"      # ThreadingHTTPServer on 127.0.0.1:0 in daemon thread
    def stop(self) -> None
    port: int; base_url: str               # "http://127.0.0.1:<port>"
    requests: list[dict]                   # received: {"method","path","headers": dict(lowercased keys),"body": parsed json or None}
```
Endpoints: `POST /v1/messages` (anthropic, stream or not, proper SSE event sequence message_start/content_block_start/
content_block_delta/content_block_stop/message_delta/message_stop with usage), `POST /v1/chat/completions` (openai, stream or not;
streamed final usage chunk only if stream_options.include_usage), `POST /v1/responses`, `POST /v1/embeddings`, `GET /v1/models`.
Reply text is deterministic: `"Echo: <last user text>"` (truncated to 200 chars) unless the prompt contains one of the magic words:
`MOCK_ERROR_500` → 500 provider error; `MOCK_ERROR_429` → 429; `MOCK_SLOW` → sleeps 1s. Model in response = request model.
Usage: input_tokens = len(all prompt text)//4 + 1, output_tokens = len(reply)//4 + 1.

## llmtap/demo.py
```python
def generate_traffic(base_url: str, count: int = 40, delay_s: float = 0.3, stop: threading.Event | None = None) -> None
    # urllib-only client hitting the llmtap proxy: mix of anthropic/openai, streaming/non-streaming, several models
    # (claude-sonnet-4-5, claude-haiku-4-5, claude-opus-4-1, gpt-4o, gpt-4o-mini, gpt-5), repeated prompts (cache hits),
    # one prompt containing a fake leaked AWS key "AKIAIOSFODNN7EXAMPLE", a burst of identical requests (loop detector),
    # realistic prompts (code review, summarize, SQL, translate...). Sets X-LLMTap-Session / X-LLMTap-Tag-Feature headers.
```
CLI `llmtap demo [--port 4141] [--no-open]`: starts MockUpstream, points cfg.upstreams anthropic+openai at it, starts LLMTap
(cache on, guard redact, loop_threshold 4, db in a temp dir), generates traffic in a background thread, prints banner, opens browser
(webbrowser) unless --no-open, serves until Ctrl-C.
