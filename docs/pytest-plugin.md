# pytest plugin: record once, replay forever

`pip install llmtap-proxy` also installs a pytest plugin. It turns testing LLM code into this:

- the **first** `pytest` run calls the real API and records every exchange next to the test;
- every run after that **replays** the recording: free, instant, deterministic, no network and no API key;
- in CI, a prompt that has no recording **fails the test** instead of silently spending money.

It is built on llmtap's [cassettes](cassettes.md), but you don't run a proxy or wrap your command. Each test gets its own
in-process proxy and its own cassette file, and the Anthropic/OpenAI SDKs are pointed at it through environment variables.

## 60-second quickstart

```bash
pip install llmtap-proxy pytest anthropic openai
```

```python
# test_summary.py
import anthropic

def test_summary(llmtap):                       # <- the only llmtap-specific line
    client = anthropic.Anthropic(max_retries=0)  # reads ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY
    msg = client.messages.create(
        model="claude-haiku-4-5", max_tokens=100,
        messages=[{"role": "user", "content": "Say hello in French."}],
    )
    assert "bonjour" in msg.content[0].text.lower()
```

OpenAI is the same:

```python
import openai

def test_chat(llmtap):
    client = openai.OpenAI(max_retries=0)        # reads OPENAI_BASE_URL / OPENAI_API_KEY
    r = client.chat.completions.create(model="gpt-4o-mini",
                                       messages=[{"role": "user", "content": "Say hello in French."}])
    assert r.choices[0].message.content
```

```bash
ANTHROPIC_API_KEY=sk-ant-... pytest      # 1st run: records   -> cassettes/test_summary__test_summary.jsonl
pytest                                   # 2nd run: replays   -> no key, no network, ~milliseconds
```

```
==================================== llmtap ====================================
llmtap: 1 call, $0.00 spent, $0.0004 saved by replay, 1 cassette
```

Don't want a fixture argument? A marker alone is enough, and it carries per-test settings:

```python
@pytest.mark.llmtap(guard="redact")
def test_agent():
    run_my_agent()          # whatever SDK/framework it uses, it already talks to the proxy
```

> Create SDK clients with `max_retries=0` in tests: the official SDKs retry HTTP 409, which is what a replay miss returns.

## What the plugin does for a test

For every test that requests the `llmtap` fixture (or carries `@pytest.mark.llmtap`), and only for those:

1. starts an in-process llmtap proxy on a free port (in-memory database, quiet) with the test's cassette;
2. sets `ANTHROPIC_BASE_URL=<base>`, `OPENAI_BASE_URL=<base>/v1`, `OPENAI_API_BASE=<base>/v1` and `LLMTAP_URL=<base>` (restored
   afterwards, so tests don't leak into each other);
3. in replay mode (or `auto` mode with an existing cassette), sets a harmless fake `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` if
   they are unset, so SDK clients can be constructed in CI without secrets. Real keys are never overwritten;
4. at teardown, fails the test if a replayed request had no recording (see below), then stops the proxy.

If no test uses the fixture or marker the plugin does nothing at all: no servers, no environment changes, no output.
(`pytest -p no:llmtap` disables it completely.)

### The `llmtap` fixture

| Attribute | Meaning |
|---|---|
| `.base_url` | the proxy's URL, e.g. `http://127.0.0.1:53211` |
| `.cassette_path` | `pathlib.Path` of this test's cassette |
| `.mode` | the effective mode: `auto`, `record` or `replay` |
| `.exchanges()` | every call so far, oldest first (`llmtap.models.Exchange`: `model`, `usage`, `source`, `output_text`, ...) |
| `.calls` | number of calls |
| `.cost_usd` | money actually spent upstream in this test (0 when replayed) |
| `.saved_usd` | money avoided by replaying |
| `.misses()` | replay-mode requests that had no recording |
| `.tap` | the underlying `llmtap.proxy.LLMTap` |

```python
def test_cost_guard(llmtap):
    run_pipeline()
    assert llmtap.calls <= 3, "pipeline got chattier"
    assert all(e.usage.output_tokens < 500 for e in llmtap.exchanges())
```

## Cassettes and modes

By default a cassette lives next to the test file: `<test dir>/cassettes/<module>__<test_name>.jsonl`. Parametrized
tests get one cassette per parameter set (`test_mod__test_p__a_b.jsonl`); characters outside `A-Za-z0-9_.-` are replaced and
a short hash is appended when that made the name ambiguous. Methods are named `Class.method`.

| Mode | Behavior |
|---|---|
| `auto` *(default locally)* | replay what's recorded, call the real API for anything new and record it |
| `record` | ignore the cassette, call the real API, rewrite the cassette from scratch |
| `replay` *(default when `CI` is set)* | never touch the network; a request with no recording **fails the test** |

Choose with, in order of precedence: the marker's `mode=`, `--llmtap-record`, `--llmtap-mode {auto,record,replay}`, then the
default (`replay` if the `CI` environment variable is set and not `0`/`false`/`no`/`off`, else `auto`).

### Options

| Option | Meaning |
|---|---|
| `--llmtap-mode {auto,record,replay}` | mode for the whole run |
| `--llmtap-record` | shortcut for "re-record every cassette that is used" |
| `--llmtap-cassette-dir DIR` | keep all cassettes in one directory (names then include the test file's path) |
| ini `llmtap_cassette_dir` | same, set permanently in `pytest.ini` / `pyproject.toml`; relative to the rootdir |

```toml
# pyproject.toml
[tool.pytest.ini_options]
llmtap_cassette_dir = "tests/cassettes"
```

### The marker

```python
@pytest.mark.llmtap(cassette="name", mode="replay", guard="redact", allow_miss=False)
```

| Argument | Meaning |
|---|---|
| `cassette` | use `<cassette dir>/name.jsonl` instead of the per-test file (tests sharing a name share the file, in order) |
| `mode` | `auto` / `record` / `replay`, overriding the command line |
| `guard` | llmtap [guard](guard.md) mode for this test: `off`, `warn`, `redact`, `block` |
| `allow_miss` | in replay mode, forward unrecorded requests upstream instead of failing |

The marker also requests the fixture, so `@pytest.mark.llmtap` on a test, class or module (`pytestmark = ...`) is enough.

### Replay misses

When a replayed request has no recording, the SDK receives an HTTP 409 and the test is failed at teardown with:

```
llmtap: 1 LLM request(s) had no recording in tests/cassettes/test_summary__test_summary.jsonl:
  - model=claude-haiku-4-5 /v1/messages: last user message: 'Say hello in Spanish.'
re-run with --llmtap-record to re-record this cassette (needs real API keys).
```

If the test had already failed on its own, its error is left untouched and the miss is reported in the `llmtap` section of the
terminal summary instead. A stale cassette can't pass silently.

### Secrets

Request headers (`x-api-key`, `Authorization`, ...) are never written to cassettes, and key-looking strings (`sk-...`,
`sk-ant-...`) in request bodies are redacted. Response headers that look sensitive are dropped. Matching ignores headers
entirely, which is why a fake key in CI works. Before committing a new cassette you can still run `llmtap scan` over it.

## Workflow

**Day to day**

```bash
pytest                       # auto: replay what exists, record what's new
```

**You changed a prompt** (the old recording no longer matches):

```bash
pytest tests/test_summary.py --llmtap-record     # re-record just that file (needs real keys)
git diff --stat tests/cassettes                  # review what changed
git add tests/cassettes && git commit
```

**Commit cassettes.** They're small, diff nicely (one JSON line per call) and are exactly what makes CI keyless.

### GitHub Actions

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install llmtap-proxy pytest anthropic openai
      - run: pytest -q          # CI=true is set by GitHub => replay mode; no API keys, no secrets
```

Missing `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` values are faked for you; a prompt that wasn't recorded fails the build with the message above.

## FAQ

**My model answers differently every time. Won't recordings go stale?**
Recordings are keyed on the *request*, not the answer. The same request always gets the same recorded answer, which is what
makes the test deterministic. They only go stale when the request changes (prompt, model, `max_tokens`, tools, ...), and then
replay fails loudly so you re-record. Assert on the shape of the answer, not exact wording, so a re-record doesn't break
assertions. Keep prompts deterministic: freeze clocks, seed randomness, sort dicts and sets before putting them in a prompt.

**The same request is made twice in one test.**
Identical requests are replayed in the order they were recorded; after the last one, it repeats. Multi-turn agents replay
faithfully.

**Async tests?**
Work as is. The proxy runs on its own thread, and the environment is set before your test body (and any async client) starts.
Create the client inside the test, or in a fixture that depends on `llmtap`, so it sees the environment variables. A client
built at import time will have already read the real base URL.

**pytest-xdist?**
Works. Every test has its own cassette file and its own proxy on its own port, so workers never share state. (Tests that
deliberately share one `cassette=` name should not run in parallel.) The summary line is merged across workers.

**Streaming?**
Streamed responses are recorded verbatim and replayed as streams.

**A provider other than Anthropic/OpenAI** (OpenRouter, Groq, Ollama, ...)?
The plugin reads `llmtap.toml` from the working directory, so custom `[upstreams.*]` and pricing apply. Point your client at
`llmtap.base_url + "/<upstream-name>/v1"` (explicit prefix routing), or set the environment variable your client uses.

**Can I use the proxy directly?**
Yes: `llmtap.base_url` is a plain HTTP URL, so any client works. `llmtap.tap` is the `LLMTap` object (store, cassette, budget).

**Where did the `llmtap` section in the output come from?**
`llmtap: N calls, $X spent, $Y saved by replay, M cassettes`, printed only when a test used the plugin.

**How do I switch it off?**
`pytest -p no:llmtap`. Without the fixture or marker it does nothing anyway.

For a runnable project see [`examples/pytest_plugin/`](../examples/pytest_plugin/).
