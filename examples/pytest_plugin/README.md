# pytest plugin example

Three tests for a tiny LLM feature (`app.py`), recorded once and replayed forever by the llmtap pytest plugin.

```bash
pip install llmtap pytest      # the plugin registers itself, there is nothing to configure
cd examples/pytest_plugin
pytest                         # replays the committed cassettes: free, offline, no API key needed
```

The committed cassettes in `cassettes/` were generated against llmtap's mock upstream (the "answers" are just
`Echo: ...`), so you can run the example without a key. To record real answers:

```bash
ANTHROPIC_API_KEY=sk-ant-... pytest --llmtap-record     # re-records every cassette the tests use
git diff --stat cassettes/                              # review, then commit
```

Things to try:

- Change a prompt in `app.py` and run `CI=1 pytest`: the test fails with the prompt that has no recording and
  `re-run with --llmtap-record`.
- `pytest -p no:llmtap` turns the plugin off.

See [`docs/pytest-plugin.md`](../../docs/pytest-plugin.md) for the full guide. `conftest.py` only exists so the example
works from a source checkout; with `pip install llmtap` you don't need it.
