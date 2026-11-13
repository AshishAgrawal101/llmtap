"""Tests for llmtap.pricing."""
from __future__ import annotations

import math

import pytest

from llmtap.models import Usage
from llmtap.pricing import PRICES, PRICES_AS_OF, cost_usd, format_usd, normalize_model, price_for


def approx(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


# ----------------------------------------------------------------- table shape

def test_table_entries_complete_and_non_negative() -> None:
    assert len(PRICES) > 100
    for key, p in PRICES.items():
        assert key == key.lower(), key
        assert set(p) == {"input", "output", "cache_read", "cache_write"}, key
        assert all(isinstance(v, float) and v >= 0 for v in p.values()), key


def test_prices_as_of_present() -> None:
    assert PRICES_AS_OF


@pytest.mark.parametrize("model,inp,out", [
    ("claude-3-haiku", 0.25, 1.25),
    ("claude-3-opus", 15, 75),
    ("claude-3-5-haiku", 0.8, 4),
    ("claude-3-5-sonnet", 3, 15),
    ("claude-3-7-sonnet", 3, 15),
    ("claude-sonnet-4", 3, 15),
    ("claude-sonnet-4-5", 3, 15),
    ("claude-opus-4", 15, 75),
    ("claude-opus-4-1", 15, 75),
    ("claude-opus-4-5", 5, 25),
    ("claude-haiku-4-5", 1, 5),
    ("gpt-4o", 2.5, 10),
    ("gpt-4o-mini", 0.15, 0.6),
    ("gpt-4.1", 2, 8),
    ("gpt-4.1-mini", 0.4, 1.6),
    ("gpt-4.1-nano", 0.1, 0.4),
    ("o1", 15, 60),
    ("o3", 2, 8),
    ("o3-mini", 1.1, 4.4),
    ("o4-mini", 1.1, 4.4),
    ("gpt-5", 1.25, 10),
    ("gpt-5-mini", 0.25, 2),
    ("gpt-5-nano", 0.05, 0.4),
    ("text-embedding-3-small", 0.02, 0),
    ("text-embedding-3-large", 0.13, 0),
])
def test_known_list_prices(model: str, inp: float, out: float) -> None:
    p = price_for(model)
    assert p is not None
    assert approx(p["input"], inp) and approx(p["output"], out)


@pytest.mark.parametrize("model", [
    "gpt-4-turbo", "gpt-3.5-turbo", "o1-mini", "gemini-2.5-pro", "gemini-2.5-flash",
    "gemini-2.0-flash", "mistral-large-latest", "deepseek-chat", "deepseek-reasoner",
    "llama-3.1-70b-versatile", "llama-3.3-70b-versatile", "qwen-max", "claude-opus-5",
    "claude-sonnet-5", "mixtral-8x7b-32768",
])
def test_covered_families(model: str) -> None:
    p = price_for(model)
    assert p is not None and p["input"] > 0 and p["output"] > 0


# ----------------------------------------------------------------- default cache prices

def test_anthropic_cache_defaults() -> None:
    p = price_for("claude-sonnet-4-5")
    assert p is not None
    assert approx(p["cache_read"], 0.30)       # 10% of input
    assert approx(p["cache_write"], 3.75)      # 125% of input


def test_openai_cache_default_is_half() -> None:
    p = price_for("gpt-4-turbo")
    assert p is not None
    assert approx(p["cache_read"], 5.0)
    assert approx(p["cache_write"], 12.5)


def test_explicit_cache_price_kept() -> None:
    assert approx(price_for("gpt-4o-mini")["cache_read"], 0.075)  # type: ignore[index]
    assert approx(price_for("claude-3-haiku")["cache_write"], 0.30)  # type: ignore[index]


# ----------------------------------------------------------------- lookup / normalization

@pytest.mark.parametrize("model,expected", [
    ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
    ("gpt-4o-2024-08-06", "gpt-4o"),
    ("gpt-4o-2024-05-13", "gpt-4o-2024-05-13"),
    ("claude-3-5-sonnet-20241022", "claude-3-5-sonnet"),
    ("claude-3-5-sonnet-latest", "claude-3-5-sonnet"),
    ("claude-opus-4-20250514", "claude-opus-4"),
    ("claude-opus-4-1-20250805", "claude-opus-4-1"),
    ("claude-opus-4-5-20251101", "claude-opus-4-5"),
    ("claude-sonnet-4-20250514", "claude-sonnet-4"),
    ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5"),
    ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
    ("o1-mini-2024-09-12", "o1-mini"),
    ("o3-mini-2025-01-31", "o3-mini"),
    ("o1-2024-12-17", "o1"),
    ("gpt-5-mini-2025-08-07", "gpt-5-mini"),
    ("gpt-4.1-nano-2025-04-14", "gpt-4.1-nano"),
    ("gemini-2.5-flash-lite-preview-06-17", "gemini-2.5-flash-lite"),
    ("gemini-2.0-flash-001", "gemini-2.0-flash"),
])
def test_longest_prefix_match(model: str, expected: str) -> None:
    assert price_for(model) == PRICES[expected]


def test_word_boundary_prevents_false_prefix() -> None:
    # "o1" must not match "o1x"-like names; "gpt-4" must not match "gpt-4o..." (gpt-4o wins)
    assert price_for("o1xyz") is None
    assert price_for("gpt-4o-something") == PRICES["gpt-4o"]


@pytest.mark.parametrize("model,expected", [
    ("anthropic/claude-3.5-sonnet", "claude-3-5-sonnet"),
    ("anthropic/claude-sonnet-4.5", "claude-sonnet-4-5"),
    ("anthropic/claude-3.7-sonnet:thinking", "claude-3-7-sonnet"),
    ("openai/gpt-4o-mini", "gpt-4o-mini"),
    ("openai/gpt-4.1", "gpt-4.1"),
    ("google/gemini-2.5-pro", "gemini-2.5-pro"),
    ("google/gemini-2.5-flash", "gemini-2.5-flash"),
    ("meta-llama/llama-3.1-70b-instruct", "llama-3.1-70b"),
    ("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "meta-llama-3.1-70b"),
    ("openrouter/anthropic/claude-opus-4", "claude-opus-4"),
    ("deepseek/deepseek-r1", "deepseek-r1"),
    ("mistralai/mistral-large-2411", "mistral-large"),
    ("Qwen/Qwen2.5-72B-Instruct-Turbo", "qwen2.5-72b"),
])
def test_provider_prefixes_stripped(model: str, expected: str) -> None:
    assert price_for(model) == PRICES[expected]


@pytest.mark.parametrize("model,expected", [
    ("claude-3-5-sonnet-v2@20241022", "claude-3-5-sonnet"),       # vertex
    ("claude-opus-4-5@20251101", "claude-opus-4-5"),
    ("anthropic.claude-3-5-sonnet-20241022-v2:0", "claude-3-5-sonnet"),  # bedrock
    ("us.anthropic.claude-sonnet-4-20250514-v1:0", "claude-sonnet-4"),
    ("ft:gpt-4o-mini-2024-07-18:my-org::abc123", "gpt-4o-mini"),  # fine-tune -> base rate
    ("GPT-4o", "gpt-4o"),
    ("  gpt-4o  ", "gpt-4o"),
])
def test_version_suffixes_and_wrappers(model: str, expected: str) -> None:
    assert price_for(model) == PRICES[expected]


@pytest.mark.parametrize("model", [
    "llama3:8b", "llama3", "llama3.1", "llama3.2:latest", "qwen2.5:7b", "mistral",
    "mistral:latest", "gpt-oss:20b", "deepseek-r1:14b", "phi3:mini", "gemma3:27b",
    "nomic-embed-text", "meta-llama/llama-3.1-8b-instruct:free",
])
def test_local_and_free_models_cost_zero(model: str) -> None:
    p = price_for(model)
    assert p is not None
    assert all(v == 0 for v in p.values())
    assert cost_usd(model, Usage(1000, 1000, 100, 100)) == 0.0


def test_exact_only_local_names_dont_swallow_paid_models() -> None:
    assert price_for("mistral-large-latest") == PRICES["mistral-large"]
    assert price_for("qwen-max-2025-01-25") == PRICES["qwen-max"]
    assert price_for("mistral-unknown-model") is None


def test_groq_llama_not_local() -> None:
    assert price_for("llama3-70b-8192") == PRICES["llama3-70b"]
    assert price_for("llama-3.1-8b-instant") == PRICES["llama-3.1-8b"]


@pytest.mark.parametrize("model", [None, "", "   ", "totally-unknown-model", "foo/bar"])
def test_unknown_returns_none(model: str | None) -> None:
    assert price_for(model) is None
    assert cost_usd(model, Usage(1000, 1000)) == 0.0


def test_normalize_model() -> None:
    assert normalize_model("anthropic/claude-3.5-sonnet") == "claude-3-5-sonnet"
    assert normalize_model("claude-3-5-sonnet@20240620") == "claude-3-5-sonnet"
    assert normalize_model("openai/gpt-4.1") == "gpt-4.1"   # dots kept outside claude


def test_price_for_returns_copy() -> None:
    p = price_for("gpt-4o")
    assert p is not None
    p["input"] = 999
    assert PRICES["gpt-4o"]["input"] == 2.5


# ----------------------------------------------------------------- overrides

def test_override_new_model() -> None:
    ov = {"my-model": {"input": 1.0, "output": 2.0}}
    p = price_for("my-model-v3", ov)
    assert p == {"input": 1.0, "output": 2.0, "cache_read": 0.5, "cache_write": 1.25}
    assert approx(cost_usd("my-model-v3", Usage(1_000_000, 1_000_000), ov), 3.0)


def test_override_wins_and_merges_partial() -> None:
    ov = {"gpt-4o": {"input": 1.0}}
    p = price_for("gpt-4o-2024-08-06", ov)
    assert p is not None
    assert p["input"] == 1.0 and p["output"] == 10.0 and p["cache_read"] == 1.25


def test_more_specific_builtin_beats_shorter_override() -> None:
    ov = {"gpt-4o": {"input": 1.0, "output": 1.0}}
    assert price_for("gpt-4o-mini", ov) == PRICES["gpt-4o-mini"]


def test_override_exact_and_normalized_keys() -> None:
    ov = {"anthropic/claude-3.5-sonnet": {"input": 2.0, "output": 9.0}}
    p = price_for("claude-3-5-sonnet-20241022", ov)
    assert p is not None and p["input"] == 2.0 and p["output"] == 9.0
    assert approx(p["cache_read"], 0.2)  # anthropic default = 10%


def test_override_can_price_local_model() -> None:
    ov = {"llama3": {"input": 0.1, "output": 0.1}}
    assert price_for("llama3:8b", ov)["input"] == 0.1  # type: ignore[index]


def test_override_ignores_garbage() -> None:
    ov = {"gpt-4o": "nope", 5: {"input": 1}}  # type: ignore[dict-item]
    assert price_for("gpt-4o", ov) == PRICES["gpt-4o"]  # type: ignore[arg-type]


# ----------------------------------------------------------------- cost formula

def test_cost_simple() -> None:
    # gpt-4o: 2.5/10 per 1M
    assert approx(cost_usd("gpt-4o", Usage(1000, 500)), (1000 * 2.5 + 500 * 10) / 1e6)


def test_cost_anthropic_additive_cache() -> None:
    u = Usage(input_tokens=100, output_tokens=200, cache_read_tokens=10_000, cache_write_tokens=2_000)
    expected = (100 * 3 + 200 * 15 + 10_000 * 0.3 + 2_000 * 3.75) / 1e6
    assert approx(cost_usd("claude-sonnet-4-5-20250929", u), expected)


def test_cost_anthropic_via_openrouter_name_still_additive() -> None:
    u = Usage(input_tokens=100, output_tokens=0, cache_read_tokens=1000)
    assert approx(cost_usd("anthropic/claude-3.5-sonnet", u), (100 * 3 + 1000 * 0.3) / 1e6)


def test_cost_openai_cached_is_subset() -> None:
    u = Usage(input_tokens=10_000, output_tokens=1_000, cache_read_tokens=8_000)
    expected = (2_000 * 2.5 + 8_000 * 1.25 + 1_000 * 10) / 1e6
    assert approx(cost_usd("gpt-4o", u), expected)


def test_cost_openai_ignores_cache_write() -> None:
    u = Usage(input_tokens=1000, output_tokens=0, cache_write_tokens=1000)
    assert approx(cost_usd("gpt-4o", u), 1000 * 2.5 / 1e6)


def test_cost_openai_cache_read_exceeds_input_charges_full_input() -> None:
    u = Usage(input_tokens=100, output_tokens=0, cache_read_tokens=1000)
    assert approx(cost_usd("gpt-4.1", u), (100 * 2 + 1000 * 0.5) / 1e6)


def test_cost_deepseek_subset_formula() -> None:
    u = Usage(input_tokens=1_000_000, output_tokens=0, cache_read_tokens=500_000)
    assert approx(cost_usd("deepseek-chat", u), 0.5 * 0.28 + 0.5 * 0.028)


def test_cost_negative_tokens_clamped() -> None:
    assert cost_usd("gpt-4o", Usage(-5, -5, -5, -5)) == 0.0


def test_cost_big_numbers() -> None:
    assert approx(cost_usd("claude-3-opus", Usage(1_000_000, 1_000_000)), 90.0)


# ----------------------------------------------------------------- format_usd

@pytest.mark.parametrize("x,s", [
    (0, "$0.00"),
    (0.0, "$0.00"),
    (0.0042, "$0.0042"),
    (0.00421, "$0.0042"),
    (0.5, "$0.5000"),
    (1.234, "$1.23"),
    (1, "$1.00"),
    (12345.678, "$12345.68"),
    (-0.0042, "-$0.0042"),
    (-3.5, "-$3.50"),
])
def test_format_usd(x: float, s: str) -> None:
    assert format_usd(x) == s


def test_claude_5_family_prices_match_published_list():
    """Regression: 'claude-fable-5-2' must not fall through to the older 'claude-fable-5' prefix."""
    expect = {
        "claude-fable-5-2": (8, 40), "claude-mythos-5-2": (8, 40), "claude-fable-5-1": (10, 50),
        "claude-opus-5-5": (4, 20), "claude-sonnet-5-5": (2, 10), "claude-haiku-5-5": (0.10, 0.50),
    }
    for model, (i, o) in expect.items():
        p = price_for(model)
        assert (p["input"], p["output"]) == (i, o), model
