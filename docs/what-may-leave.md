# What may leave

Spend caps limit the bill. They don't limit **what** goes out: a prompt-injected agent with a $2 cap can still put
your `.env` file into a prompt. These checks look at the content of each request before it leaves your machine.

| Check | Default | What it stops |
|---|---|---|
| [Leak tripwire](#leak-tripwire) | **on (block)** | Your real secrets (stored keys, `.env` values, canaries) in a prompt, including base64, hex and URL-encoded copies |
| [Canaries](#canaries) | off (`llmtap canary add`) | Tells you an agent read your `.env` file |
| [Provider-side tools](#provider-side-tools) | **blocked for stored keys** | Web search / fetch, code execution, remote MCP and provider-fetched URLs, which send data on to other sites |
| [Keys-only mode](#keys-only-mode) | off (`--keys-only`) | Requests carrying a real key, such as an attacker's own, instead of your placeholder |
| [Size limit](#size-limit) | off (`--max-request-kb`) | Bulk uploads in one call |

None of this makes llmtap a sandbox: it only sees LLM traffic. An agent with network access can still `curl` data
out. For that, run the agent somewhere llmtap is its only way out: see [sandbox.md](sandbox.md).

## Leak tripwire

The [secret guard](guard.md) guesses: it looks for things shaped like keys. The tripwire doesn't guess. It
fingerprints the secrets you actually have and blocks any request that contains one:

- the keys stored in the [key vault](keys.md),
- values in `.env` and `.env.*` files in the folder where llmtap starts (not `.env.example` / `.sample` /
  `.template`), or the files you list with `--leak-files a.env,b.env`,
- your [canaries](#canaries).

From `.env` files it takes values that look generated: those of secret-named variables (`*_KEY`, `*_SECRET`,
`*_TOKEN`, `*_PASSWORD`, `*_DSN`, `*_WEBHOOK_URL`, ...) that are at least 10 characters with mixed characters, long
random-looking values of any variable, URLs carrying credentials, and multi-line PEM keys. It skips plain words
(`DB_PASSWORD=password`, `SESSION_DRIVER=database`), numbers, paths, `${VAR}` references, URLs without credentials,
weak passwords on `localhost` connection strings, and `llmtap-key-*` placeholders. For a connection string with a
real password it also watches the password on its own.

Each secret is matched as is, JSON-escaped, URL-encoded (upper or lower case), as hex and as base64 at every
alignment, also of its UTF-16 form (PowerShell). Line breaks, spaces, colons and `\x` are ignored when looking for the
encoded forms, so `base64` output wrapped at 76 columns and `xxd`-style hex are caught. It checks the body, the URL
and every header except the credential ones. The files are re-read when they change.

It catches secrets copied or encoded in common ways, not every transformation: an agent determined to hide a secret
(reversing it, splitting it, encrypting it) can. That is what the [sandbox](sandbox.md) is for.

```bash
llmtap wrap -- claude                         # default: block
llmtap wrap --leak-check warn -- python app.py   # send it, but log and flag it
llmtap wrap --leak-check off -- python app.py
```

A blocked call gets HTTP 403 with the provider's error shape, naming what was found but never the value:

```
llmtap blocked this request: it contains STRIPE_SECRET from .env (base64).
```

The secret is never written to llmtap's history: the stored request has it replaced with `[LEAK-REDACTED]`, and an
echo of it in a response (warn mode) is scrubbed too. With `--notify`, you also get a desktop notification.

Cost: one substring search per secret form over the request body, about 50 ms for an 800 KB prompt and 30 secrets.

Compressed request bodies (`Content-Encoding: gzip` and friends) can't be checked, so they are refused with HTTP 415.
The official SDKs don't compress requests.

## Canaries

A canary is a fake secret you plant in a `.env` file. Nothing legitimate ever sends it, so if one shows up in a
prompt, something read that file.

```bash
llmtap canary add --file .env                  # appends ADMIN_API_TOKEN=tk_... and starts watching it
llmtap canary add --file .env --name OLD_STRIPE_KEY
llmtap canary list
llmtap canary remove OLD_STRIPE_KEY            # then delete the line from .env yourself
```

Canaries are kept in `~/.llmtap/canaries.json`. Like any leak, a canary in a request is blocked (or flagged with
`--leak-check warn`), and the message says it was a canary.

## Provider-side tools

Some tools run on the provider's servers, not on your machine: Anthropic's web search, web fetch and code execution,
the MCP connector (`mcp_servers`), OpenAI's `web_search` / `code_interpreter` / remote `mcp` tools and
`web_search_options`, Gemini's Google Search grounding, URL context and code execution. So do images, documents and
files given as `https://` URLs: the provider fetches them, and a URL can carry data in its query string.

Models that search by themselves (`*search*` models, OpenRouter `:online` and `plugins`, Groq `compound`, xAI
`search_parameters`, Perplexity `sonar`) count too. In a request's `tools` list, any tool type llmtap doesn't know
is treated as provider-side, so a new server tool is refused until llmtap learns about it.

These make the provider send data to other places, where llmtap can't see it. So a request that uses a **stored key**
and asks for one of them is refused with HTTP 403 (`llmtap_server_tool_blocked`), unless you allow it for that key.
So is a request body llmtap can't read (multipart file uploads, JSONL), since it can't be checked:

```bash
llmtap keys limit anthropic --allow-server-tools
llmtap keys limit anthropic --block-server-tools     # back to the default
```

Client-side tools (your own functions, Anthropic's bash / text editor / computer use, Claude Code's own `WebFetch`)
run on your machine and are not affected. Requests with your own real key, rather than a stored one, are not
checked: that's your key.

## Keys-only mode

```bash
llmtap wrap --keys-only -- claude
```

Refuses (HTTP 403, `llmtap_keys_only`) any request that carries a real credential instead of exactly a
`llmtap-key-<name>` placeholder, in a credential header or a credential-like URL parameter (`key`, `token`,
`X-Amz-Signature`, ...). A prompt injection could otherwise make the agent send your data under *someone else's* API key, to an
account the attacker can read. Requests with no credentials at all (a local Ollama model) still work. AWS SigV4
requests carry a real credential, so they are refused too; use a Bedrock API key in the vault instead.

## Size limit

```bash
llmtap wrap --max-request-kb 512 -- python agent.py
```

Refuses request bodies over the limit with HTTP 413 (`llmtap_request_too_large`). It limits how much one call can
carry out and what it can cost. Coding agents send large requests, so set it
well above your normal traffic: check `llmtap xray` or the dashboard for your usual sizes first.

## Locking the dashboard

Anything that can reach llmtap's port can use the dashboard's API, including an agent in a [sandbox](sandbox.md)
whose only way out is llmtap. With `--no-dashboard-writes` (`dashboard_writes = false`), the API can't add or change
stored keys (so it can't allow server tools, or add its own key to get past `--keys-only`) or clear the history.
Make those changes with `llmtap keys` and `llmtap clear` in a terminal instead.

## Config file

```toml
leak_check = "block"           # off | warn | block
leak_files = ""                # "" = .env* in the folder llmtap starts in
keys_only = false
max_request_kb = 0             # 0 = no limit
dashboard_writes = true        # false: keys and history can only be changed from the CLI
notify = false                 # desktop notifications for budget warnings and blocked leaks
```
