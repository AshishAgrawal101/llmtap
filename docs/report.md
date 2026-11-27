# `llmtap report` and `llmtap doctor`

## `llmtap report`: a shareable cost & safety report

```bash
llmtap report                          # last 7 days -> llmtap-report.html
llmtap report --since 30d -o q4.html --open
llmtap report --format md              # compact Markdown on stdout (PR comments, Slack)
llmtap report -f json -o report.json   # the raw numbers
```

Options: `--since WHEN` (`30m`, `24h`, `7d`, `today`, `2026-01-31`, `all`; default `7d`), `-o FILE`
(`-` = stdout; the format is inferred from `.html` / `.md` / `.json`), `--format html|md|json`,
`--open` (open the HTML in your browser), `--db PATH`, `--config FILE`.

The HTML is **one self-contained file**: inline CSS and SVG, no JavaScript, no CDN, no network
requests. Prompts are untrusted, so every piece of recorded text is HTML-escaped. Email it, attach it to a
ticket, or open it offline. A generated example lives in [`assets/sample-report.html`](../assets/sample-report.html).

What is in it:

| Section | What it tells you |
|---|---|
| Headline cards | spend, requests, tokens, saved by cache/replay, guard findings |
| **Waste finder** | identical prompts sent upstream more than once, with the call count, total cost and what `--cache` would have saved. Only upstream duplicates count; the first call of a prompt is never waste. Duplicates are matched with the same key the cache uses (provider + path + canonical body, ignoring `user` / `metadata` / `request_id`) |
| Cost by model / day / provider / `tag:feature` | where the money goes (day chart is inline SVG) |
| Latency & reliability | p50 / p95 latency, TTFT p50, error rate, mix of upstream / cache / blocked requests |
| Most expensive requests | top 10 with model, cost, tokens and a 100-char prompt preview |
| Guard findings | secret / PII kinds with counts and **masked** previews (full secrets are never printed) |
| Runaway loops blocked | repeated identical requests the loop breaker stopped, with an estimate of the spend prevented |

The numbers come from `llmtap.report.build_report(store, since) -> dict`; `render_html`,
`render_markdown` and `render_json` just format that dict.

Markdown example (abridged; the full output also has the metrics, model, day, tag, expensive-request and guard tables):

```text
## llmtap report — Nov 10, 2026 → Nov 16, 2026

**Spend $16.91** · 270 requests · 1.8M tokens · saved $1.64 by cache/replay · 7 guard findings

- $3.33 (20% of spend) went to 79 duplicate requests; `llmtap serve --cache` would have served them free.
- 17 runaway-loop requests blocked (~$3.06 prevented).

### Waste finder — $3.33 avoidable

| Prompt | Calls | Total | Saved with `--cache` |
|---|---:|---:|---:|
| Summarize the following incident report in 5 bullet points… | ×14 | $2.02 | $1.87 |
| Review this pull request diff for concurrency bugs and sugg… | ×23 | $0.9453 | $0.9042 |
```

`llmtap stats` also prints one extra line using the same logic:

```text
Potential savings from caching: $3.33 (79 duplicate requests)
```

## `llmtap doctor`: is my setup right?

```bash
llmtap doctor            # offline checks
llmtap doctor --online   # also HEAD/GET each upstream host (5s timeout)
```

```text
llmtap doctor · v0.4.2

 ✓ Python 3.13.1
 ✓ Port 4141 is free
 ✓ Database writable: /home/you/.llmtap/llmtap.db  1,204 exchanges recorded
 ✓ SQLite 3.46.0
 · No config file found  using defaults
     → copy llmtap.example.toml to llmtap.toml to customize
 · No LLMTAP_* env vars set  using defaults / config file
 ! ANTHROPIC_BASE_URL not set  this SDK talks to the provider directly
     → run `eval "$(llmtap env)"` or `llmtap wrap -- <your command>`
 ✓ OPENAI_BASE_URL points at llmtap  http://127.0.0.1:4141
 ✓ ANTHROPIC_API_KEY set
 · OPENAI_API_KEY not set
     → export it if you call that provider (or use `llmtap demo` with no keys)

All good, with 1 warning
```

`✓` ok, `!` warning (works, but probably not what you want), `✗` hard failure, `·` informational. The
exit code is `1` only on a hard failure: Python older than 3.11, an unwritable or unreadable database path,
SQLite without JSON functions, or an invalid / missing `--config` file. A busy port, SDKs that bypass llmtap and
unreachable hosts are warnings. API keys are only ever reported as "set" / "not set", never printed. When port 4141
is busy, doctor tells you whether it is llmtap itself or another process (best effort via `ss` / `lsof`).
