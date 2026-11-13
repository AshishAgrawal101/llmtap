# Cassettes: record/replay testing for LLM apps

Tests that call a real LLM are slow, cost money, need secrets in CI, and give slightly different answers on every run.
llmtap cassettes fix that the way [VCR](https://github.com/vcr/vcr) fixed it for HTTP APIs: **record real traffic once,
commit it, and replay it deterministically everywhere else.**

Because llmtap works at the HTTP layer, the same cassette works whether your code is Python, Node, Go or a shell script,
and whether it uses the raw SDK, LangChain, LlamaIndex or an agent framework.

## The workflow

```bash
# 1. Record locally, with real API keys.
llmtap wrap --cassette tests/cassettes/llm.jsonl --mode record -- pytest -q

# 2. Review and commit the cassette.
git add tests/cassettes/llm.jsonl && git commit -m "Record LLM cassette"

# 3. Replay in CI (or locally, offline), with fake keys.
ANTHROPIC_API_KEY=fake-key OPENAI_API_KEY=fake-key \
  llmtap wrap --cassette tests/cassettes/llm.jsonl --mode replay -- pytest -q
```

The SDKs still want *some* API key to be set, but replayed requests never reach the provider, so any string works.

## Modes

| Mode | Lookup | On miss | Records |
|---|---|---|---|
| `record` | never | forwards upstream | every 2xx response; the file is truncated on the first write |
| `replay` | always | **HTTP 409** provider-shaped error (or forwards upstream with `--allow-miss`) | nothing |
| `auto` *(default)* | always | forwards upstream | new 2xx responses are appended |

The error body on a replay miss looks like whatever your provider would send, so your SDK raises its usual
`APIStatusError` / `ConflictError` with this message:

```
llmtap cassette miss (replay mode): no recording for this request. Re-record with --mode record
```

> **Turn off SDK retries in replayed tests.** The official Anthropic and OpenAI SDKs automatically retry HTTP 409,
> so a miss is retried a couple of times before failing. Create test clients with `max_retries=0` (Python) /
> `maxRetries: 0` (Node) to fail fast.

`auto` mode is the convenient one locally: run your tests and only new or changed prompts hit the API. Use `record` when you
want a clean cassette without stale entries, and **always use `replay` in CI** so a changed prompt fails the build
instead of silently calling the API, or failing for lack of a key.

## Matching semantics

Each request is reduced to a key:

```
sha256( provider + "\n" + path-without-query + "\n" + canonical_json(body) )
```

- **provider** is the resolved upstream (`anthropic`, `openai`, `openrouter`, a custom upstream name, …).
- **path** excludes the query string, so `/v1/messages?beta=true` matches `/v1/messages`.
- **canonical JSON** is the request body with keys sorted and whitespace removed, minus the volatile top-level fields
  `user`, `metadata` and `request_id`. Non-JSON bodies are hashed as raw bytes.
- **Headers are ignored.** API keys, `anthropic-version`, `User-Agent` and SDK telemetry headers don't affect matching,
  which is why fake keys work in CI.
- The guard runs before matching, so use the same `--guard` mode when recording and replaying.

Consequences:

- Changing anything that ends up in the body (the prompt, the model, `max_tokens`, `temperature`, tool definitions, the
  system prompt, even a timestamp you interpolate into a prompt) produces a miss. That's the point: if your prompt changed,
  the recording is stale.
- Keep prompts deterministic in tests. Freeze clocks, fix random seeds, and sort anything you serialize into a prompt.

### Repeated identical requests

If the same key was recorded several times (an agent asking the same question twice and getting different answers),
replay returns the recordings **in the order they were recorded**: first call → first recording, second → second. Once
they run out, the **last one repeats**. Multi-turn agent runs therefore replay faithfully even when a request repeats.

### Streaming

Streamed (SSE) responses are recorded verbatim and replayed as `text/event-stream`, so streaming code paths are
exercised in replay exactly as they were in recording.

## Cassette format

A cassette is a [JSON Lines](https://jsonlines.org/) file, one recorded exchange per line:

```json
{"key": "3f9c…", "request": {"model": "claude-haiku-4-5", "max_tokens": 64, "messages": [...]}, "response": {"status": 200, "headers": {"content-type": "application/json"}, "body": "{\"id\":\"msg_…\",…}", "streamed": false, "cost_usd": 0.00021}, "recorded_at": "2026-11-05T09:14:03+00:00"}
```

- `request` is the parsed request body, kept for humans and diffs; matching uses `key`.
- `response.body` is the raw JSON text, or the raw SSE transcript when `streamed` is `true`.
- `response.cost_usd` is the original cost. Every replay adds it to the **saved** total.

It's line-oriented text, so diffs in pull requests are readable and merge conflicts are rare.

### Before you commit a cassette

Cassettes contain full prompts and model responses, plus response headers (which can include request IDs or
organization identifiers). Request headers, including your API keys, are **not** recorded. Still:

- Skim the file before committing it to a public repo.
- Run `llmtap scan tests/cassettes/llm.jsonl` to check for secrets that ended up in prompts.
- Record with `--guard redact` if your tests deal with credential-like data.

## pytest

There's nothing to install in your test suite. Run pytest under `llmtap wrap`:

```python
# conftest.py
import os
import anthropic
import pytest


@pytest.fixture
def claude() -> anthropic.Anthropic:
    # ANTHROPIC_BASE_URL is set by `llmtap wrap`; max_retries=0 makes cassette misses fail fast.
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "fake-key"), max_retries=0)
```

```python
# test_sentiment.py
def test_sentiment(claude):
    msg = claude.messages.create(
        model="claude-haiku-4-5",
        max_tokens=5,
        messages=[{"role": "user", "content": "Reply with one word, positive or negative: 'I love this!'"}],
    )
    assert msg.content[0].text.strip().lower().startswith("positive")
```

A complete example, including a guard that skips LLM tests when they aren't running under llmtap, lives in
[`examples/pytest_cassette/`](../examples/pytest_cassette).

**Tip:** use one cassette per test module or suite (`--cassette tests/cassettes/<suite>.jsonl`) to keep diffs focused.

## GitHub Actions

Use the composite action that ships with this repo:

```yaml
- uses: AshishAgrawal101/llmtap@main
  with:
    cassette: tests/cassettes/llm.jsonl
    command: pytest -q
```

Or call the CLI directly:

```yaml
- run: pip install git+https://github.com/AshishAgrawal101/llmtap
- run: llmtap wrap --cassette tests/cassettes/llm.jsonl --mode replay -- pytest -q
  env:
    ANTHROPIC_API_KEY: fake-key
    OPENAI_API_KEY: fake-key
```

See [`examples/github-actions.yml`](../examples/github-actions.yml) for a complete workflow.

## Other languages

```bash
llmtap wrap --cassette test/llm.jsonl --mode replay -- npm test
llmtap wrap --cassette testdata/llm.jsonl --mode replay -- go test ./...
```

Any test runner works, as long as its LLM client reads `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL`, or you pass the
base URL from `LLMTAP_URL` yourself.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| 409 "cassette miss" in CI | The request changed since recording. Re-record locally with `--mode record`, or run with `--mode auto` locally to append the new request. |
| Miss although nothing changed | Something non-deterministic is in the body: timestamps, UUIDs, unordered sets, random few-shot examples. Make it deterministic. |
| Tests slow before failing on a miss | SDK retries on 409. Set `max_retries=0`. |
| Requests bypass the cassette entirely | The client isn't using the base URL: hard-coded URL, or an env var your SDK doesn't read. See [integrations.md](integrations.md). |
| Spend limits trip during recording | Budgets and the loop breaker still apply when recording. Raise or disable them for the recording run. |
