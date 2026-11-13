# Contributing to llmtap

Thanks for helping! llmtap aims to be the tool you reach for first when you want to see, test or guard LLM traffic,
and it stays small and dependency-free to get there.

## Ground rules

1. **Standard library only.** Runtime code may import nothing outside the Python 3.11+ standard library. `pytest` is
   the only dev dependency. This is the project's core promise, and CI checks it.
2. **No real network calls in tests.** Use `llmtap.mock.MockUpstream` (a realistic fake Anthropic/OpenAI server on
   `127.0.0.1:0`) and point `cfg.upstreams` at it. Use `port=0` and `db_path=":memory:"`.
3. **Thread safety.** The proxy is a `ThreadingHTTPServer`, so shared state (store, cache, cassette, budget) needs a lock.
4. **Type hints and short docstrings** on public functions. No `print` in library code (only `term.py` / `cli.py` write to the terminal).
5. **Don't break the wire.** llmtap must forward requests faithfully. Anything that changes bytes sent upstream
   (like `guard = "redact"`) has to be opt-in and visible.

## Dev setup

```bash
git clone https://github.com/AshishAgrawal101/llmtap && cd llmtap
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
llmtap demo            # fake traffic + dashboard, no keys needed
```

The module map and request pipeline are described in [docs/architecture.md](docs/architecture.md).

## Great first contributions

- **Prices.** Add or update a model in `llmtap/pricing.py` (USD per 1M tokens) and link the provider's pricing page in the PR.
- **Guard detectors.** Add a pattern in `llmtap/guard.py` with tests for true positives *and* realistic near-misses.
  Low false positives matter more than recall. Use obviously fake secrets in tests (e.g. AWS's documented
  `AKIAIOSFODNN7EXAMPLE`).
- **Upstream presets.** Add an OpenAI-compatible provider to `DEFAULT_UPSTREAMS` in `llmtap/config.py`.
- **Integration snippets.** Show how to use llmtap with another SDK or framework in [docs/integrations.md](docs/integrations.md).
  Please test the snippet.
- **Wire formats.** New streaming event types or usage fields belong in `llmtap/providers.py` and `llmtap/mock.py`, each with a test.

## Pull requests

- Keep PRs focused. One feature or fix per PR.
- Add tests. `pytest -q` must pass on Python 3.11–3.13 (CI runs Linux and macOS).
- Update the docs (`README.md`, `docs/`) and add a line under **Unreleased** in [CHANGELOG.md](CHANGELOG.md) for
  user-visible changes.
- For bigger changes (new subcommands, storage format changes, new config fields), open an issue first so we can agree
  on the design.

## Reporting bugs

Use the bug report template. The most useful things to include:

- `llmtap version`, Python version, OS
- the provider and SDK (with version) you were using
- the exchange from `llmtap show <id> --raw`, **with secrets removed**

Never post real API keys or private prompts in issues. For security problems, see [SECURITY.md](SECURITY.md).

## Code of conduct

Be kind, assume good faith, and keep discussions technical. Harassment of any kind isn't tolerated.

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
