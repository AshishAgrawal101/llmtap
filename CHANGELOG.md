# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-11-28

### Added

- **X-ray in the dashboard and the report**: every request has an **X-ray** tab (context split, tool and MCP bloat,
  and the prompt-cache verdict against the previous request), the main view has a "Context X-ray · today" card, and
  `llmtap report` has an X-ray section (where input tokens go, the largest tool definitions, cache misses by cause
  with the $ lost). New read-only endpoints: `/_llmtap/api/exchanges/<id>/xray` and `/_llmtap/api/xray`.
- **`llmtap whatif`**: reprices your recorded calls on other models from the token counts they actually reported
  (no new calls), and shows what a working prompt cache would have saved, with misses split into fixable causes
  (a timestamp in the system prompt, reordered or edited tools) and cache expiries. `--json` too.
- **`--max-call-usd`**: refuses a single call likely to cost more than the limit (HTTP 403,
  `llmtap_call_too_expensive`), before it's sent. The estimate counts inline images as media, not by their base64
  length, and assumes up to 4,096 output tokens rather than the whole `max_tokens`.
- **[docs/agents.md](docs/agents.md)**: setup for Claude Code, Codex CLI, Aider and other agents, with recommended
  limits.

### Changed

- `llmtap report` applies `[pricing_overrides]` to its X-ray figures.
- PyPI page: links to the source, docs and changelog, and better search keywords. README: a line asking for a star.

## [0.4.1] - 2026-11-25

### Changed

- **Real keys are refused for providers with a stored key**: once you've stored a key for a provider, a request that
  carries a real key for it gets HTTP 403 (`llmtap_real_key_refused`), since it's someone else's account or the key
  you meant to stop handing out. `llmtap wrap` already swaps real keys for placeholders, so this only fires when
  something sends a real key on its own. Turn it off with `--allow-real-keys`. Thanks to the r/LLMDevs commenter who
  argued that an opt-in only protects people who were already going to be fine.
- The keys-only and real-key refusals now name the exact fix: the placeholder and environment variable to use
  (`OPENAI_API_KEY=llmtap-key-openai`), or the `llmtap keys add` command when nothing is stored.

### Fixed

- The test suite no longer reads the developer's real `~/.llmtap` or OS credential store.

## [0.4.0] - 2026-11-24

### Added

- **`llmtap xray`**: what your coding agent sends. Splits each request into system prompt, tool definitions, history
  and new input (scaled to the reported usage, with cost per part), ranks tool definitions and groups MCP tools by
  server, and explains prompt-cache misses: the first thing that changed since the previous request (a timestamp in
  the system prompt, an edited or reordered tool, a changed model, an edited message) or an expired cache, with the
  estimated cost of the miss. Works for Anthropic, OpenAI (chat and Responses), Gemini and Bedrock. `--json` too.
- **Leak tripwire** (on by default): llmtap fingerprints your real secrets (stored keys, values in `.env` / `.env.*`
  files in the folder it starts in, canaries) and blocks any request whose body contains one, as is, JSON-escaped,
  URL-encoded, hex or base64 at any alignment. Unlike the regex guard it never guesses. The secret is never written to
  the history, and an echo of it in a response is scrubbed. `--leak-check off|warn|block`, `--leak-files`. See
  [docs/what-may-leave.md](docs/what-may-leave.md).
- **Canaries**: `llmtap canary add --file .env` plants a fake secret; if it is ever sent, the request is blocked and
  you're told something read that file.
- **Keys-only mode** (`--keys-only`): refuses requests that carry a real API key instead of a `llmtap-key-<name>`
  placeholder, so a prompt injection can't send data out under someone else's account.
- **Request size limit** (`--max-request-kb`): refuses bigger request bodies with HTTP 413.
- **`--no-dashboard-writes`** (`dashboard_writes = false`): the dashboard API can't add or change stored keys or
  clear the history, so an agent that can reach llmtap's port can't loosen its own limits. Use the CLI instead.
- Compressed request bodies (`Content-Encoding`) are refused with HTTP 415, since no check could read them.
- Stored keys can't call endpoints that create other credentials (OpenAI realtime client secrets, Gemini ephemeral
  tokens, API keys): the result would work without llmtap and escape every limit.
- **Budget warnings** at 50, 80 and 95% (`--budget-warn`) of the daily and session budgets and of each stored key's
  caps: in the terminal, as a dashboard banner, and with `--notify` as a desktop notification (no dependencies).
- **`llmtap statusline`**: one line of today's spend, cache hit rate and last request size for Claude Code's status
  bar (`llmtap statusline --setup` prints the settings snippet).
- **Sandbox guide**: [docs/sandbox.md](docs/sandbox.md), how to run an agent where llmtap is its only way out (Linux
  and macOS recipes, not yet tested end to end).

### Changed

- **Stored keys refuse provider-side tools by default**: web search and fetch, code execution, remote MCP servers and
  `https://` images / documents / files the provider fetches itself make the provider send data on to other sites, so
  a request that uses a stored key and asks for one gets HTTP 403 (`llmtap_server_tool_blocked`). So does a model that
  searches by itself, a tool type llmtap doesn't know, and a request body it can't read (multipart uploads). Allow them per key
  with `llmtap keys limit NAME --allow-server-tools` (or the dashboard API's `allow_server_tools`). Requests with
  your own real key are not affected.

## [0.3.0] - 2026-11-21

### Added

- **Key vault**: store an API key once (`llmtap keys add openai`, or the dashboard's new **Keys** panel) and let apps
  and AI coding agents use it without seeing it. The secret lives in the OS credential store (Windows Credential
  Manager, macOS Keychain, Linux Secret Service; a private file otherwise); code uses a placeholder such as
  `llmtap-key-openai`, which `llmtap wrap` puts in `OPENAI_API_KEY` and friends, and llmtap swaps in the real key on
  the way out, only towards the upstream URL the key was stored for (the official API unless `--base`), and only for
  programs on this computer (not web pages or other machines). Per-key limits: daily and total dollar caps, allowed
  models, expiry and pause; they fail closed (unpriced models and endpoints llmtap can't check are refused for a
  limited key). The key is never returned by any API, never written to the history, and scrubbed from errors and
  echoed responses. See [docs/keys.md](docs/keys.md).
- **OpenTelemetry spend export** (thanks @KEYUR141, #8): set `otel_endpoint` (or `LLMTAP_OTEL_ENDPOINT`) to an
  OTLP/HTTP collector and every priced call is exported as the `llmtap.cost` counter in USD, with the provider and
  model as GenAI semantic-convention attributes. Off by default, sent in the background, never includes prompts,
  responses or keys.
- **Node wrapper** (thanks @KEYUR141, #9): `bin/llmtap.js` and a `package.json` for the npm package `llmtap-proxy`,
  so Node projects can run `npx llmtap-proxy wrap -- npm run dev` (Python 3.11+ must be installed). Not yet
  published to npm.

### Changed

- Dashboard: the header logo, browser-tab icon and wordmark now match the project logo (the tap with a waveform).
- Packaging: the license is declared as an SPDX expression (`license = "MIT"`, PEP 639) and the dashboard files as
  a package, so builds no longer warn and keep working with setuptools releases after February 2027. Building from
  source needs setuptools 77 or newer; installed files are unchanged.

## [0.2.2] - 2026-11-18

### Security

- `--guard block` no longer stores the blocked secret: the recorded request body is redacted.
- Gemini calls other than `generateContent` (e.g. `:embedContent?key=…`, `GET /v1beta/models?key=…`) were routed to
  the OpenAI upstream, sending the Google API key to api.openai.com. Any Generative Language path, `x-goog-api-key`
  header or `?key=` parameter now routes to Gemini.
- **Guard gaps:** Slack refresh/app tokens (`xoxe-`, `xapp-`) and incoming-webhook URLs are detected; `.env` lines
  with CRLF endings, keyed secrets inside structured tool arguments (Anthropic `tool_use.input`, Gemini
  `functionCall.args`) and UTF-8-BOM request bodies are scanned properly; a private key without its END line now
  has its key body redacted, not just the header.
- `llmtap scan` read UTF-16 files (what Windows PowerShell 5.1 writes) as NUL-interleaved text and missed every secret.
- **Request smuggling:** with two different `Content-Length` headers the proxy used the first and ran the rest of the
  body as a second request. Ambiguous framing (conflicting lengths, `+7`/`0x7`-style numbers, malformed chunks, a
  `Transfer-Encoding` other than `chunked`) now gets a 400, and a body is never read with one huge allocation (a
  3-byte body claiming `Content-Length: 3000000000` used to cost 2.9 GB).

### Fixed

- **Budgets:** concurrent requests could all pass the cap at once (spend was only added after each call finished).
  In-flight calls now reserve a worst-case cost estimate, so parallel agents can't overshoot by more than one call.
- **Budgets:** free cache and cassette hits were refused once the cap was reached. They are now always served; the
  loop breaker still counts them, so a stuck agent replaying a cached request is still stopped.
- **Cache + cassettes:** with `--cache`, record mode answered from the cache and never wrote the cassette, and auto
  mode didn't append cache-served responses. Record mode now skips the cache; auto mode records cache hits.
- **Cassettes:** a UTF-8 BOM (added by Notepad / PowerShell) silently dropped the first recording; parallel writers on
  Windows could overwrite each other's lines (appends are now file-locked); lines are written with `\n` on every OS.
- **Pricing:** Claude through OpenAI-compatible routers (OpenRouter, …) was charged twice for cached tokens: the
  formula now follows the API's wire format, not the model name. Non-generation calls (`GET /v1/responses/<id>`, …)
  are no longer priced or added to budgets.
- **Gemini:** `streamGenerateContent` without `?alt=sse` (a JSON array) now records real usage, text and cost, and the
  cache no longer answers an `?alt=sse` request with a cached JSON array.
- Proxied responses no longer carry duplicate `Date` / `Server` headers.
- **pytest plugin:** tests sharing a `cassette=` name no longer overwrite each other under `--llmtap-record`; nested
  test classes get distinct cassette files; the example project no longer fails with "Plugin already registered".
- **Dashboard:** "Copy as curl" dropped `-H` on tag/session headers; "avg TTFT" mixed in non-streamed, failed and
  out-of-range requests (now computed by the server); the spend axis showed `$0.00000` for small totals; replies that
  were only thinking showed as blank; the chart total was clipped on phones; personal data was labelled "secrets";
  invalid `group_by` returned 500 instead of 400.
- `llmtap report` printed a `file://` link that doesn't open on Windows.
- Guard: a lone surrogate escape in a JSON body crashed redact mode with a 500; the phone-number detector
  backtracked for ~20 s on 2 MB of digits.
- `report` / `stats` / `export` counted (or exported) rows twice when a running proxy recorded new rows meanwhile.
- `llmtap demo` wrote fake traffic into a project's configured cassette, obeyed its budgets, and went through
  `HTTP_PROXY`.
- `llmtap wrap`: a command the OS can't execute exits 126 with a message instead of a traceback; Ctrl-C on Windows no
  longer prints a thread traceback; the user's `NO_PROXY` / `no_proxy` hosts are kept.
- `llmtap doctor` reported a running llmtap as "port in use" with an IPv6 `--host` or behind `HTTP_PROXY`, and a
  database under a folder with `#` as unreadable.
- Token and duration formatting rolled over wrongly at unit boundaries (`1000k`, `1000ms`, `60.0s`); `~` shortening
  matched paths that merely start with the home directory's name.
- **Dashboard:** a tab left open on a stopped llmtap retried every second or so (tens of thousands of failed requests
  over an afternoon). It now backs off to 30 s, skips hidden tabs, reconnects by itself and shows "Offline".
- `llmtap stats` on an empty database showed `wrap`'s base-URL hint; it now says how to record traffic.
- An upstream's `102` / `103 Early Hints` interim response was relayed and recorded as the final response (and the
  real one dropped).
- Dashboard live feed: `HEAD /_llmtap/api/events` leaked a thread per request, and open feeds outlived `shutdown()`.
- Docs: install with `pip install llmtap-proxy`; routing, config flags, pipeline order and the budget API match the
  code.

### Added

- **AWS Bedrock Runtime** (`InvokeModel`, `InvokeModelWithResponseStream`, `Converse`, `ConverseStream`). Thanks to
  @KEYUR141 for the routing and the signature-preserving proxying (#7). Point the AWS SDK's `endpoint_url` at
  llmtap: calls are priced (Claude model IDs, inference profiles and ARNs), budget- and loop-checked, cached and
  recorded to cassettes, and EventStream responses are decoded for usage and text and replayed as EventStream.
  SigV4-signed requests are forwarded unchanged, including signed `X-LLMTap-*` headers, and Bedrock API keys
  (Bearer) are routed to Bedrock instead of OpenAI. `--guard redact` blocks a SigV4 request it would have to
  rewrite, since that would break the signature.
- Dashboard: an "errors (HTTP 4xx/5xx)" source filter, and search now matches tags and session ids.
- **Pricing follows more of the bill:** long-prompt rate cards (Claude Haiku 5.5 costs $0.50/$2.50 instead of
  $0.10/$0.50 once a prompt is over 100K tokens; Gemini 2.5/3 Pro and Claude Sonnet 4/4.5 above 200K), Anthropic fast
  mode (2x), 1-hour cache writes (2x input instead of 1.25x), server-side web searches ($10 per 1,000) and OpenAI's
  flex tier (half price). `llmtap prices` lists the long-prompt rates (and `--json` has a `long_context` key).

### Performance

- `llmtap stats` and `llmtap report` on large databases: on 200k rows `stats` went from 17 s to 3 s. Generation calls
  now store their cache key (database schema 2, upgraded in place; read-only and older databases still open), so
  the "potential savings" waste finder runs from an index; rows recorded before the upgrade are streamed instead of
  loaded at once; reports page with a keyset and skip response bodies.

## [0.2.1] - 2026-11-16

### Fixed

- Windows: `prices`, `doctor`, `scan` and other commands no longer crash with a `charmap` codec error when output is
  redirected to a file, a pipe or a CI log. Output falls back to UTF-8 when the console encoding can't show tables.

### Changed

- Published on PyPI as `llmtap-proxy` (`llmtap` was too close to an existing project). The command and the import name
  are still `llmtap`.

## [0.2.0] - 2026-11-16

### Added

- **Native Gemini API** (`generateContent` / `streamGenerateContent`): routing, usage and text extraction, and
  Google-style errors. Thanks to @KEYUR141 for the original implementation (#6). Gemini "thinking" tokens are billed as
  output, the model is read from the URL, and Gemini calls are cached and budget/loop-checked like other LLM calls.
- **pytest plugin** (`llmtap` fixture, `@pytest.mark.llmtap`, `--llmtap-mode`, `--llmtap-record`): per-test cassettes that
  record on first run and replay offline in CI. See `docs/pytest-plugin.md`.
- **`llmtap report`**: self-contained HTML / Markdown / JSON cost and safety report, including a waste finder that
  shows what `--cache` would have saved.
- **`llmtap doctor [--online]`**: setup checks with fix hints.
- `llmtap stats` shows potential savings from caching.

### Fixed

- `--by tag:<name>` on SQLite older than 3.45 (use `json_each` instead of a JSON path).
- Non-generation endpoints (`count_tokens`, batches, files) are recorded but no longer cached, budget- or loop-checked,
  or priced. In replay mode these requests now reach the real upstream.
- Multipart and binary request bodies are forwarded byte-for-byte, even with `--guard redact`.
- In-band stream errors (`event: error`, `response.failed`) are no longer recorded as successful, priced or cached.
- Stored bodies are capped at 2M characters; undecodable (br/zstd) responses are relayed untouched and not cached.
- HEAD/304 keep the upstream `Content-Length`; read timeouts after headers return 504.
- Claude Code OAuth traffic (`anthropic-*` headers only) is routed to the Anthropic upstream.
- Budget and cache updates now happen before the final response byte, fixing a race on rapid follow-up requests.

### Security

- Credentials echoed back by an upstream, and secret query parameters (`?key=...`), are scrubbed from the database,
  cache and cassettes.
- Dashboard routes reject requests with a non-local `Host` header (DNS-rebinding defense), and
  `POST /_llmtap/api/clear` rejects cross-site requests.
- A prominent warning is printed when binding to a non-loopback address.

## [0.1.0] - 2026-11-05

First public release.

### Added

- **Proxy**: local HTTP proxy for Anthropic, OpenAI and OpenAI-compatible APIs on `127.0.0.1:4141`, with live
  relaying of streamed (SSE) responses and time-to-first-token measurement.
- **Routing**: automatic Anthropic/OpenAI detection, plus `/<name>/` prefixes for the built-in upstreams OpenRouter,
  Groq, Together, Mistral, DeepSeek and Ollama, and for custom upstreams from `llmtap.toml`.
- **`llmtap wrap -- <cmd>`**: runs any command with `ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`, `OPENAI_API_BASE` and
  `LLMTAP_URL` pointed at a fresh proxy, then prints a cost/usage summary.
- **Cost tracking**: built-in price table for Claude, GPT, o-series, OpenAI embeddings, Gemini via OpenRouter, Llama,
  Mixtral and DeepSeek models, including prompt-cache reads and writes, with per-model overrides.
- **Dashboard**: single-file, offline web UI at `/_llmtap/` with a live feed (SSE), search, exchange detail, stats and
  time series, backed by a JSON API.
- **History**: SQLite storage with `llmtap ls`, `show`, `tail`, `stats` (by model, provider, source, day, hour or tag),
  `export` (JSONL/CSV) and `clear`.
- **Cache**: opt-in response cache for identical requests with TTL, tracking money saved.
- **Cassettes**: VCR-style record/replay to JSONL with `record`, `replay` and `auto` modes, ordered replay of repeated
  requests, and HTTP 409 on a replay miss (or `--allow-miss`).
- **Guard**: detection of AWS, Anthropic, OpenAI, GitHub, Slack, Stripe and Google keys, private keys, JWTs and generic
  secrets, plus optional PII (email, phone, Luhn-checked cards), in `warn`, `redact` or `block` mode. Also available as
  `llmtap scan`.
- **Budgets**: daily and per-session USD caps, plus a loop breaker for identical requests repeated within a window.
  Refusals use provider-shaped HTTP 429 errors.
- **Tagging**: `X-LLMTap-Session` and `X-LLMTap-Tag-*` request headers, stripped before forwarding.
- **`llmtap demo`**: built-in mock upstream and traffic generator for trying everything without an API key.
- Composite GitHub Action (`action.yml`) for replaying cassettes in CI.

[Unreleased]: https://github.com/AshishAgrawal101/llmtap/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/AshishAgrawal101/llmtap/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/AshishAgrawal101/llmtap/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/AshishAgrawal101/llmtap/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/AshishAgrawal101/llmtap/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/AshishAgrawal101/llmtap/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/AshishAgrawal101/llmtap/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/AshishAgrawal101/llmtap/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/AshishAgrawal101/llmtap/releases/tag/v0.1.0
