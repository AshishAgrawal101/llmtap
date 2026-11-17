# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- `--guard block` no longer stores the blocked secret: the recorded request body is redacted.
- Gemini calls other than `generateContent` (e.g. `:embedContent?key=…`, `GET /v1beta/models?key=…`) were routed to
  the OpenAI upstream, sending the Google API key to api.openai.com. Any Generative Language path, `x-goog-api-key`
  header or `?key=` parameter now routes to Gemini.

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

### Added

- Dashboard: an "errors (HTTP 4xx/5xx)" source filter, and search now matches tags and session ids.

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

[Unreleased]: https://github.com/AshishAgrawal101/llmtap/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/AshishAgrawal101/llmtap/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/AshishAgrawal101/llmtap/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/AshishAgrawal101/llmtap/releases/tag/v0.1.0
