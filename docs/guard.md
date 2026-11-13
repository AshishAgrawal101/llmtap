# Guard: a secret & PII firewall for prompts

Prompts are assembled from user input, files, logs, tool outputs and retrieved documents, and sooner or later one of
them contains a credential. llmtap scans every outgoing request body **before** it leaves your machine.

## Modes

| Mode | What happens when something is found | HTTP | Exchange `source` |
|---|---|---|---|
| `off` | No scanning. | as upstream | `upstream` |
| `warn` *(default)* | Findings are logged (terminal + dashboard) and the request is sent **unchanged**. | as upstream | `upstream` |
| `redact` | Each match is replaced with `[REDACTED:<kind>]` and the redacted request is sent. | as upstream | `upstream` |
| `block` | The request is refused and never sent. | **400**, provider-shaped error | `blocked` |

```bash
llmtap wrap --guard redact -- python app.py
llmtap serve --guard block --pii
LLMTAP_GUARD=block llmtap wrap -- ./run-agent.sh
```

```toml
# llmtap.toml
guard = "redact"
guard_pii = true
```

The request body stored in llmtap's history is the one sent upstream, so in `redact` mode the secret is not kept in your
local database either. In `warn` mode the original prompt, secret included, is stored locally.

In `block` mode the SDK raises its normal error for a 400 (`anthropic.BadRequestError`, `openai.BadRequestError`, …),
so the failure is visible and attributable instead of silent.

## What is detected

### Secrets (always on unless `guard = "off"`)

| Kind | Matches | Masked preview (illustrative) |
|---|---|---|
| `aws_access_key` | `AKIA…` / `ASIA…` access key IDs | `AKIA…MPLE` |
| `aws_secret_key` | 40-char AWS secret keys, when they appear next to AWS-ish context (`aws_secret_access_key = …`) | `wJal…EKEY` |
| `anthropic_key` | `sk-ant-…` | `sk-ant-…a1b2` |
| `openai_key` | `sk-…`, `sk-proj-…` (not `sk-` inside ordinary words) | `sk-proj-…9f3c` |
| `github_token` | `ghp_`, `gho_`, `ghs_`, `github_pat_` | `ghp_…x7Qa` |
| `slack_token` | `xoxb-`, `xoxa-`, `xoxp-`, `xoxr-`, `xoxs-` | `xoxb-…4kd2` |
| `stripe_key` | `sk_live_…`, `rk_live_…` | `sk_live_…Wd0e` |
| `google_api_key` | `AIza…` | `AIza…c8Rk` |
| `private_key` | `-----BEGIN … PRIVATE KEY-----` blocks (RSA, EC, OpenSSH, PKCS#8…) | `-----BEGIN…` |
| `jwt` | Three base64url segments starting with `eyJ` | `eyJh…sw5c` |
| `generic_secret` | `password`, `secret`, `token`, `api_key` assigned a value of 8+ characters (`password = "hunter22!"`, `"api_key": "…"`) | `pass…22!"` |

### PII (only with `--pii` / `guard_pii = true`)

| Kind | Matches |
|---|---|
| `email` | Email addresses |
| `phone` | International (`+44 20 7946 0958`) and US (`(415) 555-0100`) phone numbers |
| `credit_card` | Card numbers that pass the **Luhn** checksum (cuts false positives on random digit runs) |

PII detection is opt-in because many legitimate prompts contain email addresses or phone numbers. Turn it on when you
handle end-user data.

## Findings

Each finding records:

| Field | Meaning |
|---|---|
| `kind` | One of the kinds above |
| `path` | Where in the JSON body it was found, e.g. `messages[2].content` or `system` |
| `preview` | A **masked** preview such as `sk-ant-…a1b2`. The full secret is never shown in the dashboard or the logs. |
| `start`, `end` | Character offsets within that string |

Findings appear in the terminal line for the request, in the dashboard, and in `llmtap stats` (`guard_findings`,
`blocked`).

## Scanning outside the proxy

The same detectors are available as a command, which makes them handy in pre-commit hooks and CI:

```bash
llmtap scan prompts/system_prompt.txt     # exit code 1 if anything is found
cat transcript.txt | llmtap scan -
llmtap scan tests/cassettes/llm.jsonl     # check a cassette before committing it
```

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: llmtap-scan
      name: llmtap secret scan (prompts)
      entry: bash -c 'for f in "$@"; do llmtap scan "$f" || exit 1; done' --
      language: system
      files: ^prompts/
```

## Limits

The guard is a regex-based safety net, not a DLP product.

- It can't recognise secrets with no recognisable shape (a random password with no `password=` next to it).
- It scans request bodies, not responses or headers. Your API key in the `Authorization` header is supposed to go upstream.
- Unusually encoded content (base64 blobs, images, PDFs) isn't decoded.

False positives or misses? Please open an issue with a **fake** example. Never paste a real secret.
