# Integrations

llmtap is a base URL. Anything that lets you change where it sends LLM requests works with it.

The easiest path is always:

```bash
llmtap wrap -- <your command>
```

`wrap` sets `ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`, `OPENAI_API_BASE` and `LLMTAP_URL` for the child process. Many SDKs
read these automatically, so there's **nothing to change in code**. When a library doesn't read them, pass the base URL
explicitly as shown below.

The snippets use the default `llmtap serve` address. Under `wrap`, read `LLMTAP_URL` instead:

| Talking to | Base URL |
|---|---|
| Anthropic | `http://127.0.0.1:4141` |
| OpenAI | `http://127.0.0.1:4141/v1` |
| Other upstream `<name>` (OpenAI-compatible) | `http://127.0.0.1:4141/<name>/v1` |

Your API keys are passed through to the provider unchanged. Keep setting `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` /
`OPENROUTER_API_KEY` as usual.

---

## Python: Anthropic SDK

Reads `ANTHROPIC_BASE_URL` automatically.

```python
import anthropic

client = anthropic.Anthropic()                                    # under `llmtap wrap`
client = anthropic.Anthropic(base_url="http://127.0.0.1:4141")    # explicit

with client.messages.stream(
    model="claude-sonnet-4-5", max_tokens=512,
    messages=[{"role": "user", "content": "Stream me a limerick."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

## Python: OpenAI SDK

Reads `OPENAI_BASE_URL` automatically.

```python
from openai import OpenAI

client = OpenAI()                                                 # under `llmtap wrap`
client = OpenAI(base_url="http://127.0.0.1:4141/v1")              # explicit

# Ask for usage on streams so llmtap gets exact token counts instead of estimates.
stream = client.chat.completions.create(
    model="gpt-4o-mini", stream=True, stream_options={"include_usage": True},
    messages=[{"role": "user", "content": "Count to five."}],
)
```

The Responses API (`client.responses.create`) and embeddings work the same way.

## Node / TypeScript: official SDKs

`openai` reads `OPENAI_BASE_URL` and `@anthropic-ai/sdk` reads `ANTHROPIC_BASE_URL`.

```ts
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

const openai = new OpenAI();          // under `llmtap wrap -- node app.mjs`
const anthropic = new Anthropic();

// explicit:
const openai2 = new OpenAI({ baseURL: "http://127.0.0.1:4141/v1" });
const anthropic2 = new Anthropic({ baseURL: "http://127.0.0.1:4141" });
```

## Vercel AI SDK

```ts
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { generateText } from "ai";

const openai = createOpenAI({ baseURL: "http://127.0.0.1:4141/v1" });
const anthropic = createAnthropic({ baseURL: "http://127.0.0.1:4141/v1" }); // the AI SDK's Anthropic baseURL includes /v1

const { text } = await generateText({
  model: anthropic("claude-haiku-4-5"),
  prompt: "Name three uses for a proxy.",
});
```

## Go

```go
import (
    "os"

    "github.com/anthropics/anthropic-sdk-go"
    "github.com/anthropics/anthropic-sdk-go/option"
    "github.com/openai/openai-go"
    oaioption "github.com/openai/openai-go/option"
)

base := os.Getenv("LLMTAP_URL") // set by `llmtap wrap -- go run .`
if base == "" {
    base = "http://127.0.0.1:4141"
}
claude := anthropic.NewClient(option.WithBaseURL(base))
gpt := openai.NewClient(oaioption.WithBaseURL(base + "/v1"))
```

## curl / any HTTP client

```bash
curl -N http://127.0.0.1:4141/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01" -H "content-type: application/json" \
  -H "X-LLMTap-Session: manual-test" \
  -d '{"model":"claude-haiku-4-5","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}'

curl http://127.0.0.1:4141/groq/v1/chat/completions \
  -H "authorization: Bearer $GROQ_API_KEY" -H "content-type: application/json" \
  -d '{"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":"hi"}]}'
```

## LangChain (Python)

```python
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

claude = ChatAnthropic(model="claude-sonnet-4-5", base_url="http://127.0.0.1:4141")
gpt = ChatOpenAI(model="gpt-4o-mini", base_url="http://127.0.0.1:4141/v1")
```

Under `llmtap wrap`, `ChatOpenAI` also picks up `OPENAI_BASE_URL` / `OPENAI_API_BASE` from the environment. LangChain.js
is the same idea: `new ChatOpenAI({ configuration: { baseURL: "http://127.0.0.1:4141/v1" } })`.

This is where llmtap shines: open the dashboard and see the *actual* prompt your chain, agent or retriever assembled.

## LlamaIndex

```python
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o-mini", api_base="http://127.0.0.1:4141/v1")
claude = Anthropic(model="claude-sonnet-4-5", base_url="http://127.0.0.1:4141")
```

## Claude Agent SDK / Claude Code

Both honour `ANTHROPIC_BASE_URL`, so wrapping is enough:

```bash
llmtap wrap -- python my_agent.py      # Claude Agent SDK app
llmtap wrap -- claude                  # watch what a coding agent sends, live
```

Long agent sessions are a good place for the loop breaker and a session budget:

```bash
llmtap wrap --budget-session 3 --loop-threshold 6 -- python my_agent.py
```

## LiteLLM (SDK)

```python
import litellm

resp = litellm.completion(
    model="openai/gpt-4o-mini",
    api_base="http://127.0.0.1:4141/v1",
    messages=[{"role": "user", "content": "hi"}],
)
```

To put llmtap *in front of* a LiteLLM proxy server, add it as an upstream and point clients at `/litellm/v1`:

```toml
[upstreams.litellm]
base = "http://localhost:4000"
style = "openai"
```

## DSPy

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini", api_base="http://127.0.0.1:4141/v1")
dspy.configure(lm=lm)
```

## Instructor, Pydantic AI and other SDK wrappers

Libraries that wrap the official `openai` / `anthropic` clients inherit their base URL, so `llmtap wrap` just works.
Otherwise, build the client with the base URL yourself and hand it over:

```python
import instructor
from openai import OpenAI

client = instructor.from_openai(OpenAI(base_url="http://127.0.0.1:4141/v1"))
```

## OpenRouter, Groq, Together, Mistral, DeepSeek

Use the OpenAI SDK (or anything OpenAI-compatible) with the provider's prefix:

```python
import os
from openai import OpenAI

openrouter = OpenAI(base_url="http://127.0.0.1:4141/openrouter/v1", api_key=os.environ["OPENROUTER_API_KEY"])
groq       = OpenAI(base_url="http://127.0.0.1:4141/groq/v1",       api_key=os.environ["GROQ_API_KEY"])
together   = OpenAI(base_url="http://127.0.0.1:4141/together/v1",   api_key=os.environ["TOGETHER_API_KEY"])
mistral    = OpenAI(base_url="http://127.0.0.1:4141/mistral/v1",    api_key=os.environ["MISTRAL_API_KEY"])
deepseek   = OpenAI(base_url="http://127.0.0.1:4141/deepseek/v1",   api_key=os.environ["DEEPSEEK_API_KEY"])
```

## Ollama and local servers

Ollama's OpenAI-compatible endpoint is preconfigured as `ollama` (`http://localhost:11434`):

```python
from openai import OpenAI

local = OpenAI(base_url="http://127.0.0.1:4141/ollama/v1", api_key="ollama")
local.chat.completions.create(model="llama3.1", messages=[{"role": "user", "content": "hi"}])
```

Local models are priced at $0, but you still get latency, token counts, history, caching and cassettes. vLLM, LM Studio,
llama.cpp's server and TGI work the same way once added as an upstream (see [configuration.md](configuration.md#upstreams)).

## Aider and other coding agents

Any CLI that reads `OPENAI_API_BASE` / `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` or has a "custom base URL" setting can be wrapped:

```bash
llmtap wrap -- aider --model sonnet
```

Some editors and hosted agents send requests from the vendor's servers rather than from your machine. Those can't reach
a proxy on `127.0.0.1`, and llmtap won't see their traffic.

## Tagging requests from any client

Add headers. llmtap strips them before forwarding:

```python
client = OpenAI(default_headers={"X-LLMTap-Session": "eval-2026-11-05", "X-LLMTap-Tag-Feature": "search"})
client.chat.completions.create(..., extra_headers={"X-LLMTap-Tag-User": "u_123"})   # per request
```

```bash
llmtap stats --by tag:feature
llmtap stats --by tag:session --since 7d
```

## Using llmtap from your own code

The dashboard's JSON API is available to scripts:

| Endpoint | Returns |
|---|---|
| `GET /_llmtap/health` | `{"ok": true, "version": …}` |
| `GET /_llmtap/api/stats?since=<secs>&group_by=model` | Totals plus grouped breakdown, session spend, cache and cassette status |
| `GET /_llmtap/api/exchanges?limit=&offset=&q=&provider=&model=&source=` | `{"items": [summary, …]}` |
| `GET /_llmtap/api/exchanges/<id>` | One full exchange, including request and response bodies |
| `GET /_llmtap/api/timeseries?since=&bucket=` | Requests, cost and tokens per time bucket |
| `GET /_llmtap/api/events` | Server-Sent Events stream of new exchanges |
| `POST /_llmtap/api/clear` | Clears history |

Every proxied response also carries `x-llmtap-id` (look it up with `llmtap show <id>`) and `x-llmtap-source`
(`upstream`, `cache`, `cassette`, …) headers.
