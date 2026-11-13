# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/AshishAgrawal101/llmtap/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AshishAgrawal101/llmtap/releases/tag/v0.1.0
