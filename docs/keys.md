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
   key's limits, and forwards the request **only to the upstream the key was stored for**: the provider's official
   API (`https://api.openai.com`, …) unless you stored it with `--base <url>`. A project's `llmtap.toml` that points
   `openai` somewhere else can't redirect a stored key; the call is refused with `llmtap_key_wrong_upstream`.

Stored keys are for programs on this computer: requests from other machines, from web pages (a cross-site `Origin`
or `Sec-Fetch-Site`) and to unexpected `Host` names (DNS rebinding) are refused, and responses to stored-key requests
carry no `Access-Control-Allow-Origin`. Otherwise any website could spend `llmtap-key-openai`, since the name is easy
to guess.

The real key is never returned by the dashboard or any API, never written to the history database, and scrubbed
from error messages and from anything an upstream echoes back (headers and body, streamed or not). Stored-key
requests ask the upstream for uncompressed responses; a compressed one is decoded, scrubbed and relayed
uncompressed, or refused (502) if llmtap can't decode it.

Programs in Docker containers or WSL reach llmtap from a non-loopback address, so they can't use stored keys; run
them on the host, or pass the real key yourself. Requests that use a stored key are tagged
`vault_key=<name>`, so `llmtap stats --by tag:vault_key` shows spend per key.

If several keys exist for one provider, `wrap` uses the one named exactly like the provider (`openai`), else the
alphabetically first. To pick a specific key, set the variable yourself: `OPENAI_API_KEY=llmtap-key-work-openai`.

## Limits

Set them in the dashboard (Keys) or with `llmtap keys limit`:

```bash
llmtap keys limit openai --daily 2 --total 20          # dollars; 0 = no cap
llmtap keys limit openai --models "gpt-4o-mini,gpt-5-mini*"   # allowed models, * wildcards; "" = any
llmtap keys limit openai --expires 2026-12-31          # end of that day, local time; "never" to clear
llmtap keys limit openai --pause                       # refuse everything; --resume to undo
llmtap keys limit openai --allow-server-tools          # allow provider-side web search / fetch, code execution, MCP
```

| Limit | Refused with |
|---|---|
| Paused, expired | 403 `llmtap_key_paused` / `llmtap_key_expired` |
| A model the request names (URL or body) isn't in the allowed list | 403 `llmtap_key_model_not_allowed` |
| Daily / total spend reached | 429 `llmtap_key_limit` |
| A key with a spend limit, used with a model llmtap has no price for | 403 `llmtap_key_unpriced_model` |
| A key with model or spend limits, used for an endpoint llmtap can't check (batches, images, audio, fine-tuning, long-running jobs) | 403 `llmtap_key_endpoint_not_allowed` |
| Provider-side tools (web search / fetch, code execution, remote MCP, provider-fetched URLs), unless `--allow-server-tools` ([why](what-may-leave.md#provider-side-tools)) | 403 `llmtap_server_tool_blocked` |
| Endpoints that create other credentials (OpenAI realtime client secrets, Gemini ephemeral tokens, API keys), which would work without llmtap | 403 `llmtap_key_mint_refused` |
| A real key for a provider that has a stored key, unless `--allow-real-keys` ([why](what-may-leave.md#real-keys-and-keys-only-mode)) | 403 `llmtap_real_key_refused` |
| Placeholder for a key that isn't stored | 401 `llmtap_key_not_found` |
| Key sent towards another provider, or another URL than it was stored for | 403 `llmtap_key_wrong_provider` / `llmtap_key_wrong_upstream` |
| Request from a web page on another site, another machine, or an unexpected `Host` | 403 `llmtap_key_cross_origin` / `llmtap_key_remote_client` / `llmtap_key_bad_host` |

Requests an upstream could read differently from llmtap's checks are refused (400) when they use a stored key:
`.`/`..`/`//`/`\`/`#` in the path, method-override headers (`X-HTTP-Method-Override` and similar), a `GET` or `HEAD`
with a body, and (for keys with a model list) a body that repeats `"model"` in one object. With a model list, every
`"model"` anywhere in the body counts, e.g. `tools[].model` on the Responses API.

Limits only mean something if llmtap can count them, so they fail closed: a key with a spend limit refuses models
without a price (add one under `[pricing_overrides]`), and a key with any model or spend limit only reaches generation
calls, reads (`GET`) and token counting. Use a key without limits for anything else.

Errors are shaped like the provider's own, so your SDK raises its usual exception with llmtap's message. Spend caps
work like the global budgets: spend is read from llmtap's history database (calls since local midnight, or since the
key was stored), and in-flight calls reserve a worst-case estimate so parallel requests can't overshoot. Cache and
cassette hits are free and never reach the provider, so they don't need the key and aren't limited by it. The global
`--budget-daily` / `--budget-session` caps still apply on top.

## Commands

| Command | What it does |
|---|---|
| `llmtap keys` / `llmtap keys list` | Stored keys (masked), their placeholders and limits |
| `llmtap keys add NAME [--provider P] [--base URL]` | Store or replace a key (keeping its limits). Prompts with hidden input, or reads one line from stdin |
| `llmtap keys limit NAME [--daily USD] [--total USD] [--models LIST] [--expires WHEN] [--pause\|--resume]` | Set limits |
| `llmtap keys remove NAME` | Delete the key from the credential store and llmtap |

`--provider` defaults to the name when the name is a provider (`openai`, `anthropic`, `gemini`, …). For a custom
upstream, use its name: `llmtap keys add work --provider my-gateway`; the key is then bound to that upstream's URL in
your current configuration (or `--base`). To send a built-in provider's key through a gateway, store it with
`--base https://your-gateway`.

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
dashboard page itself, so other websites can't add, read or change keys, but local programs can, unless you start
llmtap with `--no-dashboard-writes`. To close that gap,
run the agent in a box where llmtap is its only way out: [sandbox.md](sandbox.md). The
[leak tripwire](what-may-leave.md#leak-tripwire) also blocks any request that contains a stored key.

SigV4-signed AWS Bedrock requests can't use stored keys (AWS signs them before they leave your machine). A Bedrock API
key can be stored with `--provider bedrock` and sent as `Authorization: Bearer llmtap-key-<name>`; `wrap` does not set
`AWS_BEARER_TOKEN_BEDROCK` for you, and this path has not been tried against real AWS yet.
