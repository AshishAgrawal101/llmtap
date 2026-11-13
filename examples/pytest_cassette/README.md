# pytest + llmtap cassettes

A tiny LLM feature (`sentiment.py`) with tests (`test_sentiment.py`) that run **offline, deterministically and for
free** in CI by replaying a recorded cassette.

```bash
cd examples/pytest_cassette
pip install anthropic pytest
```

**1. Record once, with a real key:**

```bash
ANTHROPIC_API_KEY=sk-ant-... \
  llmtap wrap --cassette cassettes/llm.jsonl --mode record -- pytest -q
```

**2. Commit** `cassettes/llm.jsonl` (run `llmtap scan cassettes/llm.jsonl` first).

**3. Replay anywhere, with no key and no network to the provider:**

```bash
ANTHROPIC_API_KEY=fake-key \
  llmtap wrap --cassette cassettes/llm.jsonl --mode replay -- pytest -q
```

Change a prompt in `sentiment.py` and replay fails with a clear `cassette miss` error until you re-record. A stale
recording can't silently pass.

Without `llmtap wrap`, the LLM tests are skipped (see `conftest.py`), so a plain `pytest` never spends money by accident.

In CI, see [`../github-actions.yml`](../github-actions.yml) or use the composite action:

```yaml
- uses: AshishAgrawal101/llmtap@main
  with:
    cassette: examples/pytest_cassette/cassettes/llm.jsonl
    command: cd examples/pytest_cassette && pytest -q
```
