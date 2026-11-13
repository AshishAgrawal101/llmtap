"""Model price table and cost calculation.

Prices are **published list prices in USD per 1M tokens** as of ``PRICES_AS_OF``
(standard tier: no batch discount, no long-context surcharge, no regional uplift).
They drift; override or extend them with the config ``pricing_overrides`` table
(``[llmtap.pricing_overrides."my-model"] input = 1.0 output = 2.0``), which wins
over this table.  Entries marked ``# est.`` are best estimates.

Lookup (``price_for``):
  1. normalize the name: lowercase; drop routing prefixes (``anthropic/``, ``openai/``,
     ``google/``, ``meta-llama/`` ... - everything up to the last ``/``); drop Bedrock
     ``[us.]anthropic.`` style prefixes; drop ``@version`` suffixes (Vertex); drop
     ``ft:`` fine-tune wrappers; Claude ``3.5`` -> ``3-5`` (OpenRouter spelling).
  2. ``:tag`` suffixes: numeric (Bedrock ``-v2:0``) and OpenRouter variants
     (``:thinking``, ``:beta``) are dropped; ``:free`` is free; any other tag on a bare
     name (``llama3:8b``, ``qwen2.5:7b``) is an Ollama local model -> cost 0.
  3. exact match, then the LONGEST key that is a prefix of the name on a word boundary
     (``gpt-4o-mini-2024-07-18`` -> ``gpt-4o-mini``, never ``gpt-4o``).
     Overrides are matched the same way and win ties.

Cost (``cost_usd``) - the providers report cached tokens differently:
  * Anthropic (``claude-*``): ``input_tokens`` EXCLUDES cache tokens, so
    ``cost = input*in + cache_read*cache_read + cache_write*cache_write + output*out``.
  * Everyone else (OpenAI and OpenAI-compatible APIs, Gemini, DeepSeek, ...): cached
    tokens are a SUBSET of ``prompt_tokens``, so
    ``cost = (input - cache_read)*in + cache_read*cache_read + output*out``
    (cache writes are not billed).  If ``cache_read > input`` the numbers are clearly
    exclusive, so the input is charged in full.
Missing cache prices default to 10% of input (Anthropic) / 50% (others); cache-write
price defaults to 125% of input.
"""
from __future__ import annotations

import re
from typing import Any

from .models import Usage

PRICES_AS_OF = "2026-11"

_ZERO: dict[str, float] = {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0}


def _p(inp: float, out: float, cr: float | None = None, cw: float | None = None) -> dict[str, float]:
    d = {"input": float(inp), "output": float(out)}
    if cr is not None:
        d["cache_read"] = float(cr)
    if cw is not None:
        d["cache_write"] = float(cw)
    return d


_RAW: dict[str, dict[str, float]] = {
    # ------------------------------------------------------------------ Anthropic
    # cache_read = 0.1x input, cache_write (5-min TTL) = 1.25x input unless noted.
    "claude-instant": _p(0.8, 2.4),
    "claude-2": _p(8, 24),
    "claude-3-haiku": _p(0.25, 1.25, 0.03, 0.30),
    "claude-3-sonnet": _p(3, 15),
    "claude-3-opus": _p(15, 75),
    "claude-3-5-haiku": _p(0.8, 4, 0.08, 1.0),
    "claude-3-5-sonnet": _p(3, 15),
    "claude-3-7-sonnet": _p(3, 15),
    "claude-sonnet-4": _p(3, 15),
    "claude-sonnet-4-5": _p(3, 15),
    "claude-sonnet-4-6": _p(3, 15),
    "claude-opus-4": _p(15, 75),
    "claude-opus-4-1": _p(15, 75),
    "claude-opus-4-5": _p(5, 25),
    "claude-opus-4-6": _p(5, 25),
    "claude-opus-4-7": _p(5, 25),
    "claude-opus-4-8": _p(5, 25),
    "claude-haiku-4-5": _p(1, 5),
    # Claude 5 family (newer than most published tables; best-known list prices).
    "claude-opus-5": _p(5, 25),
    "claude-opus-5-5": _p(4, 20, 0.20),
    "claude-sonnet-5": _p(2, 10),
    "claude-sonnet-5-5": _p(2, 10, 0.10),
    "claude-haiku-5-5": _p(0.10, 0.50),           # <=100K-token prompts
    "claude-fable-5": _p(10, 50, 1.0),
    "claude-fable-5-1": _p(10, 50, 0.25),
    "claude-mythos-5": _p(10, 50, 1.0),
    "claude-mythos-5-1": _p(10, 50, 0.25),
    "claude-fable-5-2": _p(8, 40, 0.20),
    "claude-mythos-5-2": _p(8, 40, 0.20),
    # Aliases without a minor version.
    "claude-opus": _p(4, 20, 0.20),               # est. (latest opus)
    "claude-sonnet": _p(2, 10, 0.10),             # est. (latest sonnet)
    "claude-haiku": _p(0.10, 0.50),               # est. (latest haiku)
    # ------------------------------------------------------------------ OpenAI
    "gpt-4o": _p(2.5, 10, 1.25),
    "gpt-4o-2024-05-13": _p(5, 15),
    "chatgpt-4o-latest": _p(5, 15),
    "gpt-4o-mini": _p(0.15, 0.6, 0.075),
    "gpt-4o-audio-preview": _p(2.5, 10),
    "gpt-4o-mini-audio-preview": _p(0.15, 0.6),
    "gpt-4o-realtime-preview": _p(5, 20, 2.5),
    "gpt-4o-mini-realtime-preview": _p(0.6, 2.4, 0.3),
    "gpt-4o-search-preview": _p(2.5, 10),
    "gpt-4o-mini-search-preview": _p(0.15, 0.6),
    "gpt-4.1": _p(2, 8, 0.5),
    "gpt-4.1-mini": _p(0.4, 1.6, 0.1),
    "gpt-4.1-nano": _p(0.1, 0.4, 0.025),
    "gpt-4.5-preview": _p(75, 150, 37.5),
    "gpt-4-turbo": _p(10, 30),
    "gpt-4-1106-preview": _p(10, 30),
    "gpt-4-0125-preview": _p(10, 30),
    "gpt-4-vision-preview": _p(10, 30),
    "gpt-4": _p(30, 60),
    "gpt-4-32k": _p(60, 120),
    "gpt-3.5-turbo": _p(0.5, 1.5),
    "gpt-3.5-turbo-instruct": _p(1.5, 2),
    "o1": _p(15, 60, 7.5),
    "o1-preview": _p(15, 60, 7.5),
    "o1-mini": _p(1.1, 4.4, 0.55),
    "o1-pro": _p(150, 600),
    "o3": _p(2, 8, 0.5),
    "o3-pro": _p(20, 80),
    "o3-mini": _p(1.1, 4.4, 0.55),
    "o3-deep-research": _p(10, 40, 2.5),
    "o4-mini": _p(1.1, 4.4, 0.275),
    "o4-mini-deep-research": _p(2, 8, 0.5),
    "codex-mini-latest": _p(1.5, 6, 0.375),
    "gpt-5": _p(1.25, 10, 0.125),
    "gpt-5-mini": _p(0.25, 2, 0.025),
    "gpt-5-nano": _p(0.05, 0.4, 0.005),
    "gpt-5-pro": _p(15, 120),
    "gpt-5-codex": _p(1.25, 10, 0.125),
    "gpt-5-chat": _p(1.25, 10, 0.125),
    "gpt-5.1": _p(1.25, 10, 0.125),
    "gpt-5.1-codex": _p(1.25, 10, 0.125),
    "gpt-5.1-codex-mini": _p(0.25, 2, 0.025),
    "gpt-5.2": _p(1.75, 14, 0.175),               # est.
    "gpt-oss-120b": _p(0.15, 0.6),                # hosted (groq/together), est.
    "gpt-oss-20b": _p(0.075, 0.3),                # hosted, est.
    "text-embedding-3-small": _p(0.02, 0),
    "text-embedding-3-large": _p(0.13, 0),
    "text-embedding-ada-002": _p(0.10, 0),
    "omni-moderation": _p(0, 0),
    "text-moderation": _p(0, 0),
    "davinci-002": _p(2, 2),
    "babbage-002": _p(0.4, 0.4),
    # ------------------------------------------------------------------ Google
    # (<=200K-token prompt tier; cache_read = context-cache price)
    "gemini-3-pro": _p(2, 12, 0.2),               # est.
    "gemini-2.5-pro": _p(1.25, 10, 0.125),
    "gemini-2.5-flash": _p(0.30, 2.50, 0.03),
    "gemini-2.5-flash-lite": _p(0.10, 0.40, 0.01),
    "gemini-2.0-flash": _p(0.10, 0.40, 0.025),
    "gemini-2.0-flash-lite": _p(0.075, 0.30),
    "gemini-1.5-pro": _p(1.25, 5, 0.3125),
    "gemini-1.5-flash": _p(0.075, 0.30, 0.01875),
    "gemini-1.5-flash-8b": _p(0.0375, 0.15, 0.01),
    "gemini-embedding-001": _p(0.15, 0),
    "text-embedding-004": _p(0, 0),
    # ------------------------------------------------------------------ Mistral
    "mistral-large": _p(2, 6),
    "mistral-medium": _p(0.4, 2),
    "mistral-small": _p(0.1, 0.3),
    "mistral-tiny": _p(0.25, 0.25),
    "mistral-saba": _p(0.2, 0.6),
    "mistral-embed": _p(0.1, 0),
    "mistral-7b-instruct": _p(0.03, 0.055),       # openrouter, est.
    "open-mistral-7b": _p(0.25, 0.25),
    "open-mistral-nemo": _p(0.15, 0.15),
    "mistral-nemo": _p(0.15, 0.15),
    "open-mixtral-8x7b": _p(0.7, 0.7),
    "open-mixtral-8x22b": _p(2, 6),
    "mixtral-8x7b": _p(0.24, 0.24),               # groq / together
    "mixtral-8x22b": _p(1.2, 1.2),                # together
    "codestral": _p(0.3, 0.9),
    "devstral-small": _p(0.1, 0.3),
    "devstral-medium": _p(0.4, 2),
    "magistral-medium": _p(2, 5),
    "magistral-small": _p(0.5, 1.5),
    "ministral-8b": _p(0.1, 0.1),
    "ministral-3b": _p(0.04, 0.04),
    "pixtral-large": _p(2, 6),
    "pixtral-12b": _p(0.15, 0.15),
    # ------------------------------------------------------------------ DeepSeek
    # Official API (V3.2 pricing for both aliases; cache_read = cache-hit price).
    "deepseek-chat": _p(0.28, 0.42, 0.028),
    "deepseek-reasoner": _p(0.28, 0.42, 0.028),
    "deepseek-v3": _p(0.27, 1.10, 0.07),          # openrouter-style names
    "deepseek-v3.1": _p(0.27, 1.10, 0.07),        # est.
    "deepseek-v3.2": _p(0.28, 0.42, 0.028),
    "deepseek-r1": _p(0.55, 2.19, 0.14),
    "deepseek-r1-distill-llama-70b": _p(0.75, 0.99),   # groq
    # ------------------------------------------------------------------ Meta Llama (hosted)
    # groq / openrouter spellings
    "llama-3.1-8b": _p(0.05, 0.08),
    "llama-3.1-70b": _p(0.59, 0.79),
    "llama-3.1-405b": _p(3.5, 3.5),
    "llama-3.2-1b": _p(0.04, 0.04),
    "llama-3.2-3b": _p(0.06, 0.06),
    "llama-3.2-11b": _p(0.18, 0.18),
    "llama-3.2-90b": _p(0.9, 0.9),
    "llama-3.3-70b": _p(0.59, 0.79),
    "llama-4-scout": _p(0.11, 0.34),
    "llama-4-maverick": _p(0.20, 0.60),
    "llama3-8b": _p(0.05, 0.08),                  # groq legacy llama3-8b-8192
    "llama3-70b": _p(0.59, 0.79),                 # groq legacy llama3-70b-8192
    "llama-guard": _p(0.2, 0.2),
    # together spellings (meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo)
    "meta-llama-3.1-8b": _p(0.18, 0.18),
    "meta-llama-3.1-70b": _p(0.88, 0.88),
    "meta-llama-3.1-405b": _p(3.5, 3.5),
    "meta-llama-3-8b": _p(0.2, 0.2),
    "meta-llama-3-70b": _p(0.88, 0.88),
    "llama-3.3-70b-instruct-turbo": _p(0.88, 0.88),
    "llama-4-scout-17b-16e-instruct": _p(0.18, 0.59),   # together, est.
    "llama-4-maverick-17b-128e-instruct-fp8": _p(0.27, 0.85),  # together, est.
    # ------------------------------------------------------------------ Qwen (hosted)
    "qwen-max": _p(1.6, 6.4),
    "qwen-plus": _p(0.4, 1.2),
    "qwen-turbo": _p(0.05, 0.2),
    "qwen-2.5-72b": _p(0.35, 0.40),               # openrouter, est.
    "qwen2.5-72b": _p(1.2, 1.2),                  # together
    "qwen2.5-7b": _p(0.3, 0.3),                   # together
    "qwen2.5-coder-32b": _p(0.8, 0.8),            # together
    "qwen-2.5-coder-32b": _p(0.8, 0.8),
    "qwq-32b": _p(1.2, 1.2),                      # together, est.
    "qwen3-235b": _p(0.2, 0.6),                   # est.
    "qwen3-32b": _p(0.29, 0.59),                  # groq, est.
    "qwen3-coder": _p(0.4, 1.6),                  # est.
    # ------------------------------------------------------------------ others
    "grok-4": _p(3, 15, 0.75),
    "grok-4-fast": _p(0.2, 0.5, 0.05),
    "grok-3": _p(3, 15, 0.75),
    "grok-3-mini": _p(0.3, 0.5, 0.075),
    "grok-code-fast": _p(0.2, 1.5, 0.02),
    "command-a": _p(2.5, 10),
    "command-r-plus": _p(2.5, 10),
    "command-r": _p(0.15, 0.6),
    "command-r7b": _p(0.0375, 0.15),
    "kimi-k2": _p(0.6, 2.5, 0.15),                # est.
    "glm-4.6": _p(0.6, 2.2, 0.11),                # est.
    # ------------------------------------------------------------------ local (Ollama etc.) = free
    "llama2": _ZERO, "llama3": _ZERO, "llama4": _ZERO, "codellama": _ZERO, "llava": _ZERO,
    "qwen2": _ZERO, "qwen2.5": _ZERO, "qwen3": _ZERO, "qwq": _ZERO,
    "gemma2": _ZERO, "gemma3": _ZERO, "phi3": _ZERO, "phi4": _ZERO,
    "tinyllama": _ZERO, "starcoder2": _ZERO, "deepseek-coder-v2": _ZERO,
    "nomic-embed-text": _ZERO, "mxbai-embed-large": _ZERO, "all-minilm": _ZERO,
    "dolphin-mixtral": _ZERO, "neural-chat": _ZERO, "openchat": _ZERO, "zephyr": _ZERO,
    "vicuna": _ZERO, "orca-mini": _ZERO, "smollm": _ZERO, "smollm2": _ZERO,
    "mistral": _ZERO, "mixtral": _ZERO, "qwen": _ZERO, "phi": _ZERO, "gemma": _ZERO,
}

# Bare local names that must only match exactly (as prefixes they'd swallow paid
# models such as "mistral-large" or "qwen-max").
_EXACT_ONLY: frozenset[str] = frozenset({"mistral", "mixtral", "qwen", "phi", "gemma"})


def _is_anthropic(key: str) -> bool:
    return key.startswith("claude")


def _complete(key: str, entry: dict[str, Any]) -> dict[str, float]:
    """Fill in default cache prices for one table entry."""
    inp = float(entry.get("input", 0.0) or 0.0)
    out = float(entry.get("output", 0.0) or 0.0)
    cr = entry.get("cache_read")
    cw = entry.get("cache_write")
    if cr is None:
        cr = inp * (0.10 if _is_anthropic(key) else 0.50)
    if cw is None:
        cw = inp * 1.25
    return {"input": inp, "output": out, "cache_read": float(cr), "cache_write": float(cw)}


PRICES: dict[str, dict[str, float]] = {k: _complete(k, v) for k, v in _RAW.items()}

_BEDROCK_PREFIX = re.compile(
    r"^(?:(?:us|eu|apac|au|ca|jp|global)\.)?(?:anthropic|meta|mistral|amazon|cohere|ai21|deepseek|openai|qwen)\."
)
_CLAUDE_DOT = re.compile(r"(\d)\.(\d)")


def _normalize(model: str) -> tuple[str, bool]:
    """Return (normalized name, is_free_local)."""
    name = model.strip().lower()
    if name.startswith("ft:"):                     # ft:gpt-4o-mini-2024-07-18:org::id
        name = name[3:].split(":", 1)[0]
    had_slash = "/" in name
    if had_slash:
        name = name.rstrip("/").rsplit("/", 1)[-1]
    name = _BEDROCK_PREFIX.sub("", name)
    if "@" in name:
        name = name.split("@", 1)[0]
    if ":" in name:
        base, tag = name.split(":", 1)
        if tag == "free":
            return base, True
        if tag.isdigit() or had_slash:
            name = base
        else:
            return base, True                      # ollama "llama3:8b" -> local
    if name.startswith("claude"):
        name = _CLAUDE_DOT.sub(r"\1-\2", name)
    return name, False


def normalize_model(model: str) -> str:
    """Canonical lookup form of a model name (lowercase, prefixes/suffixes stripped)."""
    return _normalize(model)[0]


def _best_key(name: str, table: dict[str, Any]) -> str | None:
    """Exact key, else the longest key that prefixes `name` at a word boundary."""
    if name in table:
        return name
    best: str | None = None
    for key in table:
        if key in _EXACT_ONLY or not name.startswith(key):
            continue
        nxt = name[len(key):len(key) + 1]
        if nxt and nxt.isalnum():
            continue
        if best is None or len(key) > len(best):
            best = key
    return best


def price_for(model: str | None, overrides: dict | None = None) -> dict[str, float] | None:
    """USD-per-1M-token prices for `model` ({"input","output","cache_read","cache_write"}), or None."""
    if not model or not isinstance(model, str) or not model.strip():
        return None
    name, free = _normalize(model)
    builtin_key = _best_key(name, PRICES)
    if overrides:
        norm_over: dict[str, dict[str, Any]] = {}
        for k, v in overrides.items():
            if isinstance(k, str) and isinstance(v, dict):
                norm_over[normalize_model(k)] = v
        okey = _best_key(name, norm_over) if norm_over else None
        if okey is not None and (builtin_key is None or len(okey) >= len(builtin_key) or free):
            base: dict[str, Any] = {}
            if builtin_key is not None and okey.startswith(builtin_key) and not free:
                base = dict(_RAW[builtin_key])
            base.update({k: v for k, v in norm_over[okey].items() if v is not None})
            return _complete(okey, base)
    if free:
        return dict(_ZERO)
    if builtin_key is None:
        return None
    return dict(PRICES[builtin_key])


def cost_usd(model: str | None, usage: Usage, overrides: dict | None = None) -> float:
    """Dollar cost of one call; 0.0 for unknown models. See module docstring for the formula."""
    price = price_for(model, overrides)
    if price is None:
        return 0.0
    inp = max(int(usage.input_tokens or 0), 0)
    out = max(int(usage.output_tokens or 0), 0)
    cr = max(int(usage.cache_read_tokens or 0), 0)
    cw = max(int(usage.cache_write_tokens or 0), 0)
    name = normalize_model(model or "")
    if _is_anthropic(name):
        total = (inp * price["input"] + cr * price["cache_read"]
                 + cw * price["cache_write"] + out * price["output"])
    else:
        uncached = inp - cr if cr <= inp else inp
        total = uncached * price["input"] + cr * price["cache_read"] + out * price["output"]
    return round(total / 1_000_000, 10)


def format_usd(x: float) -> str:
    """'$0.0042' (4 decimals under $1), '$1.23', '$0.00' for zero."""
    x = float(x or 0.0)
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a == 0:
        return "$0.00"
    if a < 0.00005:
        return f"{sign}<$0.0001"
    if a < 1:
        return f"{sign}${a:.4f}"
    return f"{sign}${a:.2f}"
