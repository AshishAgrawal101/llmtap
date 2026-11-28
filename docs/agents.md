# Coding agents

llmtap works with any agent that lets you point it at a base URL. The general recipe:

```bash
llmtap keys add anthropic              # optional: the agent gets a placeholder, never the real key
llmtap wrap --max-call-usd 1 --budget-daily 10 -- <agent command>
```

`wrap` starts the proxy, sets `ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL` and `OPENAI_API_BASE`, puts placeholders in
the API key variables of providers you've stored a key for, and runs the command. Afterwards:

```bash
llmtap xray          # what the agent sent: system prompt, tools, history, and why the cache missed
llmtap whatif        # what the same session would have cost on other models
llmtap report --open # a shareable HTML report
```

> **Status:** the setups below follow each agent's own documentation. They have not all been tested end to end yet.
> If you try one, please tell us how it went in an issue.

## Claude Code

Claude Code reads `ANTHROPIC_BASE_URL`, so `wrap` is enough when you use an API key:

```bash
llmtap wrap -- claude
```

With a Claude subscription login (instead of an API key), traffic may not use the base URL in the same way; this
hasn't been verified.

**Status line**: show today's spend, cache hit rate and the last request's size at the bottom of Claude Code:

```bash
llmtap statusline --setup      # prints the snippet for ~/.claude/settings.json
```

```json
{ "statusLine": { "type": "command", "command": "llmtap statusline" } }
```

Claude Code's built-in web search runs on Anthropic's servers. If you use a stored key, allow it explicitly:
`llmtap keys limit anthropic --allow-server-tools` ([why](what-may-leave.md#provider-side-tools)).

## Codex CLI

Codex takes its endpoint from `~/.codex/config.toml` rather than `OPENAI_BASE_URL`. Add a provider that points at
llmtap (run `llmtap serve` in another terminal, or use the port `wrap` prints):

```toml
model_provider = "llmtap"

[model_providers.llmtap]
name = "llmtap"
base_url = "http://127.0.0.1:4141/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
```

Then run `codex` with `OPENAI_API_KEY` set to your key or to `llmtap-key-openai`. Signing in with a ChatGPT account
instead of an API key sends traffic elsewhere, so llmtap won't see it.

## Aider

Aider talks to OpenAI-compatible APIs through `OPENAI_API_BASE`, which `wrap` sets:

```bash
llmtap wrap -- aider --model gpt-5
```

For other providers, pass Aider's own base-URL option for that provider and point it at llmtap
(`http://127.0.0.1:<port>/<provider>/...`, see [integrations.md](integrations.md)).

## Anything else

If the tool has a "base URL", "API base" or "endpoint" setting, point it at `http://127.0.0.1:4141` (Anthropic
style) or `http://127.0.0.1:4141/v1` (OpenAI style), or `http://127.0.0.1:4141/<provider>/...` for the other built-in
providers. Tools that call the model from their own servers (Cursor, for example) can't be seen by a local proxy.

## Recommended limits for agents

| Flag | Why |
|---|---|
| `--max-call-usd 1` | Refuses one giant call (a whole repository in the context) before it's sent |
| `--budget-daily 10` | A hard daily cap, with warnings at 50 / 80 / 95% |
| `--loop-threshold 5` | Stops an agent sending the same request over and over |
| `--keys-only` | Refuses any real API key, only placeholders get through ([why](what-may-leave.md#real-keys-and-keys-only-mode)) |

To stop an agent sending data anywhere except through llmtap, run it in a box: [sandbox.md](sandbox.md).
