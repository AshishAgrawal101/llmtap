// llmtap from Node.js: zero dependencies (Node 18+ has fetch built in).
//
//   export ANTHROPIC_API_KEY=sk-ant-...
//   llmtap wrap -- node examples/node_quickstart.mjs
//
// With the official SDKs you need no code at all: `openai` reads OPENAI_BASE_URL and
// `@anthropic-ai/sdk` reads ANTHROPIC_BASE_URL, both of which `llmtap wrap` sets.
// This file uses plain fetch to show that llmtap is just a base URL.

const base = process.env.LLMTAP_URL ?? process.env.ANTHROPIC_BASE_URL ?? "http://127.0.0.1:4141";
const model = process.env.MODEL ?? "claude-haiku-4-5";

const headers = {
  "content-type": "application/json",
  "x-api-key": process.env.ANTHROPIC_API_KEY ?? "",
  "anthropic-version": "2023-06-01",
  // Tags for the dashboard; stripped by llmtap before forwarding.
  "X-LLMTap-Session": "node-quickstart",
  "X-LLMTap-Tag-Feature": "demo",
};

async function ask(question) {
  const res = await fetch(`${base}/v1/messages`, {
    method: "POST",
    headers,
    body: JSON.stringify({ model, max_tokens: 300, messages: [{ role: "user", content: question }] }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(`${res.status}: ${data?.error?.message ?? JSON.stringify(data)}`);
  console.log(`[llmtap id=${res.headers.get("x-llmtap-id")} source=${res.headers.get("x-llmtap-source")}]`);
  return data.content.filter((b) => b.type === "text").map((b) => b.text).join("");
}

async function stream(question) {
  const res = await fetch(`${base}/v1/messages`, {
    method: "POST",
    headers,
    body: JSON.stringify({ model, max_tokens: 300, stream: true, messages: [{ role: "user", content: question }] }),
  });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of res.body) {
    buffer += decoder.decode(chunk, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const event = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const line = event.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      const payload = JSON.parse(line.slice(6));
      if (payload.type === "content_block_delta" && payload.delta?.type === "text_delta") {
        process.stdout.write(payload.delta.text);
      }
    }
  }
  process.stdout.write("\n");
}

console.log("llmtap at", base);
console.log(await ask("In one sentence: why put a proxy in front of an LLM API?"));
await stream("Write a two-line poem about tokens.");
