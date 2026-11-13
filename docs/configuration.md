# Configuration

llmtap works with no configuration at all. When you want to change something, every setting can come from four
places, applied in this order (later wins):

```
built-in defaults  ←  llmtap.toml  ←  LLMTAP_* environment variables  ←  CLI flags
```

## Config file

llmtap uses the **first** of these that exists:

1. the path given with `--config path/to/file.toml`
2. `./llmtap.toml` in the current directory
3. `~/.config/llmtap/config.toml`

Settings can live at the top level of the file or inside an `[llmtap]` table. If an `[llmtap]` table exists, only its
contents are read. Dashes in keys are accepted (`budget-daily-usd` = `budget_daily_usd`). Unknown keys are ignored.

```toml
# llmtap.toml
port = 4141
cache = true
cache_ttl_s = 86400
guard = "redact"
guard_pii = false
budget_daily_usd = 10.0
budget_session_usd = 2.0
loop_threshold = 5
loop_window_s = 60
upstream_timeout_s = 300

# Extra upstreams: reachable at http://127.0.0.1:4141/<name>/...
[upstreams.vllm]
base = "http://gpu-box:8000"
style = "openai"

[upstreams.corp-gateway]
base = "https://llm-gateway.internal.example.com"
style = "anthropic"

# Price overrides / additions, USD per 1M tokens.
[pricing_overrides."my-finetune"]
input = 0.50
output = 1.50
cache_read = 0.05
cache_write = 0.625
```

A fully commented example is in [`llmtap.example.toml`](../llmtap.example.toml).

## Reference

| Field | Default | Env var | CLI flag | Description |
|---|---|---|---|---|
| `host` | `"127.0.0.1"` | `LLMTAP_HOST` | `--host` | Interface to bind. The dashboard has no auth, so keep it on loopback unless you trust the network. |
| `port` | `4141` | `LLMTAP_PORT` | `--port` | Port for `serve`. `wrap` always picks a free port. `0` = any free port. |
| `db_path` | `""` → `~/.llmtap/llmtap.db` | `LLMTAP_DB_PATH` | `--db` | SQLite history file. `":memory:"` keeps nothing on disk. |
| `quiet` | `false` | `LLMTAP_QUIET` | `--quiet` | Don't print a line per request. |
| `cache` | `false` | `LLMTAP_CACHE` | `--cache` | Serve identical requests from the local response cache. |
| `cache_ttl_s` | `0` | `LLMTAP_CACHE_TTL_S` | `--cache-ttl` | Cache entry lifetime in seconds. `0` = forever. |
| `cassette` | `""` | `LLMTAP_CASSETTE` | `--cassette` | Path to a `.jsonl` cassette. Empty = disabled. See [cassettes.md](cassettes.md). |
| `mode` | `"auto"` | `LLMTAP_MODE` | `--mode` | Cassette mode: `record`, `replay` or `auto`. |
| `allow_miss` | `false` | `LLMTAP_ALLOW_MISS` | `--allow-miss` | In `replay` mode, forward unrecorded requests upstream instead of returning 409. |
| `guard` | `"warn"` | `LLMTAP_GUARD` | `--guard` | Secret guard: `off`, `warn`, `redact` or `block`. See [guard.md](guard.md). |
| `guard_pii` | `false` | `LLMTAP_GUARD_PII` | `--pii` | Also detect emails, phone numbers and credit cards. |
| `budget_daily_usd` | `0.0` | `LLMTAP_BUDGET_DAILY_USD` | `--budget-daily` | Refuse requests (429) once today's upstream spend reaches this. `0` = unlimited. |
| `budget_session_usd` | `0.0` | `LLMTAP_BUDGET_SESSION_USD` | `--budget-session` | Same, for spend through this proxy process. `0` = unlimited. |
| `loop_threshold` | `0` | `LLMTAP_LOOP_THRESHOLD` | `--loop-threshold` | Refuse (429) an identical request once seen this many times within the window. `0` = off. |
| `loop_window_s` | `60` | `LLMTAP_LOOP_WINDOW_S` | `--loop-window` | Loop-detection window in seconds. |
| `upstreams` | see below | – | – | Named upstreams (`base`, `style`). File only. |
| `pricing_overrides` | `{}` | – | – | Per-model prices in USD per 1M tokens (`input`, `output`, `cache_read`, `cache_write`). File only, as `[pricing_overrides."<model>"]` tables. |
| `upstream_timeout_s` | `600.0` | `LLMTAP_UPSTREAM_TIMEOUT_S` | – | Timeout for upstream requests, in seconds. |

Other environment variables:

| Variable | Meaning |
|---|---|
| `LLMTAP_HOME` | Data directory (default `~/.llmtap`). The default database lives here. |
| `NO_COLOR` | Disable ANSI colours in terminal output. Colours are also off when stderr isn't a TTY. |

Boolean env vars are true for `1`, `true`, `yes` or `on` (case-insensitive) and false for anything else.

```bash
LLMTAP_CACHE=1 LLMTAP_GUARD=redact llmtap wrap -- python app.py
LLMTAP_DB_PATH=:memory: llmtap serve
```

## Upstreams

Built in. Any of these can be overridden in the config file, and new ones added:

| Name | `base` | `style` |
|---|---|---|
| `anthropic` | `https://api.anthropic.com` | `anthropic` |
| `openai` | `https://api.openai.com` | `openai` |
| `openrouter` | `https://openrouter.ai/api` | `openai` |
| `groq` | `https://api.groq.com/openai` | `openai` |
| `together` | `https://api.together.xyz` | `openai` |
| `mistral` | `https://api.mistral.ai` | `openai` |
| `deepseek` | `https://api.deepseek.com` | `openai` |
| `ollama` | `http://localhost:11434` | `openai` |

`style` tells llmtap which wire format to expect, for usage parsing and for the shape of its own error responses. Use
`openai` for any OpenAI-compatible server (vLLM, LM Studio, llama.cpp server, TGI, LiteLLM proxy…).

### Routing rules

For each request, the first matching rule wins:

1. **Explicit prefix.** `/<name>/rest` where `<name>` is a configured upstream → `<base>/rest` (query string kept).
   `/openrouter/v1/chat/completions` → `https://openrouter.ai/api/v1/chat/completions`.
2. Path starts with `/v1/messages` or `/v1/complete` → `anthropic`.
3. Request has an `anthropic-version` or `x-api-key` header → `anthropic`.
4. Everything else → `openai`.

For OpenAI-style routes, a path without a `/v1` prefix (e.g. `/chat/completions`, from a client configured with a base URL
missing `/v1`) gets `/v1` prepended.

Overriding the built-in `anthropic` / `openai` entries redirects *unprefixed* traffic too, which is handy for pointing
everything at a corporate gateway:

```toml
[upstreams.openai]
base = "https://llm-gateway.internal.example.com/openai"
style = "openai"
```

## Pricing

`llmtap prices` prints the built-in table, and `llmtap prices sonnet` filters it. Lookup is exact match first, then
the longest matching prefix, with provider prefixes like `anthropic/` or `openai/` stripped (so OpenRouter model names
work). Overrides always win.

When a price entry has no `cache_read` price, llmtap assumes 10% of input (Anthropic) or 50% (OpenAI). When it has no
`cache_write` price, it assumes 125% of input. Unknown models are recorded at $0, so add an override if you use a model
llmtap doesn't know.

## Where data lives

| What | Where |
|---|---|
| Request history | `~/.llmtap/llmtap.db` (SQLite, WAL). Change it with `--db`, `LLMTAP_DB_PATH` or `LLMTAP_HOME`. |
| Response cache | Local SQLite, managed by llmtap alongside the history |
| Cassettes | Wherever `--cassette` points (meant to be committed) |

`llmtap clear` wipes the history. Deleting `~/.llmtap/` removes everything.
