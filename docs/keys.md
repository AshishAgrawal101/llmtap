# Key vault

Store an API key once, and let your code (or an AI coding agent working on it) use the key **without ever seeing
it**, within limits you set.

```bash
llmtap keys add openai          # asks for the key (input hidden), stores it in your OS credential store
llmtap wrap -- python app.py    # app.py sees OPENAI_API_KEY=llmtap-key-openai; llmtap sends the real key
```

Or open the dashboard (`llmtap serve`, then <http://127.0.0.1:4141/_llmtap/>), click **Keys**, and paste the key there.

## How it works

1. **You store the key.** The secret goes into the operating system's credential store. llmtap itself keeps only a
   masked preview (`sk-p…a1b2`), the provider and your limits, in `~/.llmtap/keys.json`.
2. **Your code uses a placeholder.** Every stored key has one: `llmtap-key-<name>`. `llmtap wrap` sets the usual
   environment variables to it (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`,
   `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `TOGETHER_API_KEY`), so most SDKs
   pick it up with no changes. With `llmtap serve`, `eval "$(llmtap env)"` prints the same variables.
3. **llmtap swaps it on the way out.** When a request carries a placeholder (in `Authorization: Bearer …`,
   `x-api-key`, `x-goog-api-key`, `api-key` or a `?key=` parameter), llmtap replaces it with the real key, checks the
   key's limits, and forwards the request **only if it is going to that key's own provider**.

The real key is never returned by the dashboard or any API, never written to the history database, and scrubbed
from anything an upstream echoes back. Requests that use a stored key are tagged `vault_key=<name>`, so
`llmtap stats --by tag:vault_key` shows spend per key.

If several keys exist for one provider, `wrap` uses the one named exactly like the provider (`openai`), else the
alphabetically first. To pick a specific key, set the variable yourself: `OPENAI_API_KEY=llmtap-key-work-openai`.

## Limits

Set them in the dashboard (Keys) or with `llmtap keys limit`:

```bash
llmtap keys limit openai --daily 2 --total 20          # dollars; 0 = no cap
llmtap keys limit openai --models "gpt-4o-mini,gpt-5-mini*"   # allowed models, * wildcards; "" = any
llmtap keys limit openai --expires 2026-12-31          # end of that day, local time; "never" to clear
llmtap keys limit openai --pause                       # refuse everything; --resume to undo
```

| Limit | Refused with |
|---|---|
| Paused, expired | 403 `llmtap_key_paused` / `llmtap_key_expired` |
| Model not in the allowed list (generation calls) | 403 `llmtap_key_model_not_allowed` |
| Daily / total spend reached (generation calls) | 429 `llmtap_key_limit` |
| Placeholder for a key that isn't stored | 401 `llmtap_key_not_found` |
| Key sent towards another provider | 403 `llmtap_key_wrong_provider` |

Errors are shaped like the provider's own, so your SDK raises its usual exception with llmtap's message. Spend caps
work like the global budgets: spend is read from llmtap's history database (calls since local midnight, or since the
key was stored), and in-flight calls reserve a worst-case estimate so parallel requests can't overshoot. Cache and
cassette hits are free and never reach the provider, so they don't need the key and aren't limited by it. The global
`--budget-daily` / `--budget-session` caps still apply on top.

## Commands

| Command | What it does |
|---|---|
| `llmtap keys` / `llmtap keys list` | Stored keys (masked), their placeholders and limits |
| `llmtap keys add NAME [--provider P]` | Store or replace a key. Prompts with hidden input, or reads one line from stdin |
| `llmtap keys limit NAME [--daily USD] [--total USD] [--models LIST] [--expires WHEN] [--pause\|--resume]` | Set limits |
| `llmtap keys remove NAME` | Delete the key from the credential store and llmtap |

`--provider` defaults to the name when the name is a provider (`openai`, `anthropic`, `gemini`, …). For a custom
upstream, use its name: `llmtap keys add work --provider my-gateway`.

## Where secrets are stored

| `LLMTAP_VAULT_BACKEND` | Store |
|---|---|
| `auto` (default) | Windows Credential Manager, the macOS login Keychain (`security`), or the Linux Secret Service (`secret-tool`: GNOME Keyring, KWallet). Falls back to `file` when none is available. |
| `os` | Like `auto`, but an error instead of the file fallback |
| `file` | `~/.llmtap/keys.secret.json`, readable only by you (mode 0600, like `~/.aws/credentials`). Not encrypted. |
| `memory` | This process only; gone when it exits (tests) |

The dashboard's Keys panel shows which store is in use.

## What this protects against, and what it doesn't

It keeps keys out of the places they usually leak from: prompts, logs, stack traces, `.env` files that get committed,
screenshots, and AI agents that print or upload their environment. Combined with the limits, a leaked placeholder is
worthless and a runaway agent can only spend what you allowed.

It is **not a sandbox**. Anything running as your user on the same computer, including an AI agent with shell access,
can read your OS credential store directly, call llmtap's local API to change limits, or make calls through llmtap
with the placeholder. Keep the dashboard on `127.0.0.1` (the default). The Keys API only accepts changes from the
dashboard page itself, so other websites can't add, read or change keys, but local programs can.

SigV4-signed AWS Bedrock requests can't use stored keys (AWS signs them before they leave your machine). A Bedrock API
key can be stored with `--provider bedrock` and sent as `Authorization: Bearer llmtap-key-<name>`; `wrap` does not set
`AWS_BEARER_TOKEN_BEDROCK` for you, and this path has not been tried against real AWS yet.
