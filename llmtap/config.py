"""Configuration: defaults <- llmtap.toml <- LLMTAP_* env vars <- CLI flags."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


def llmtap_home() -> Path:
    return Path(os.environ.get("LLMTAP_HOME", Path.home() / ".llmtap"))


DEFAULT_UPSTREAMS: dict[str, dict[str, str]] = {
    # name -> {"base": url, "style": "anthropic"|"openai"|"bedrock"}
    "anthropic": {"base": "https://api.anthropic.com", "style": "anthropic"},
    "openai": {"base": "https://api.openai.com", "style": "openai"},
    "openrouter": {"base": "https://openrouter.ai/api", "style": "openai"},
    "groq": {"base": "https://api.groq.com/openai", "style": "openai"},
    "together": {"base": "https://api.together.xyz", "style": "openai"},
    "mistral": {"base": "https://api.mistral.ai", "style": "openai"},
    "deepseek": {"base": "https://api.deepseek.com", "style": "openai"},
    "ollama": {"base": "http://localhost:11434", "style": "openai"},
    "gemini": {
        "base": "https://generativelanguage.googleapis.com",
        "style": "gemini",
    },
    "bedrock": {
        "base": "https://bedrock-runtime.{region}.amazonaws.com",
        "style": "bedrock",
    },
}


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 4141
    db_path: str = ""                       # "" -> llmtap_home()/llmtap.db ; ":memory:" allowed
    quiet: bool = False                     # suppress per-request console lines
    # caching
    cache: bool = False
    cache_ttl_s: int = 0                    # 0 = forever
    # cassettes (record/replay)
    cassette: str = ""                      # path to .jsonl cassette; "" = disabled
    mode: str = "auto"                      # "record" | "replay" | "auto"
    allow_miss: bool = False                # in replay mode, forward misses upstream instead of erroring
    # secret guard
    guard: str = "warn"                     # "off" | "warn" | "redact" | "block"
    guard_pii: bool = False                 # also detect emails / phone numbers / credit cards
    # budgets & loop detection
    budget_daily_usd: float = 0.0           # 0 = unlimited
    budget_session_usd: float = 0.0         # 0 = unlimited (per proxy process)
    loop_threshold: int = 0                 # 0 = off; block when identical request seen N times...
    loop_window_s: int = 60                 # ...within this many seconds
    budget_warn: str = "50,80,95"           # warn when a budget / stored-key cap reaches these percentages ("" = off)
    notify: bool = False                    # also show budget warnings as desktop notifications
    # what may leave
    keys_only: bool = False                 # refuse requests that carry a real API key instead of a llmtap placeholder
    allow_real_keys: bool = False           # False: a real key is refused for any provider that has a stored key
    leak_check: str = "block"               # "off" | "warn" | "block": your real secrets (stored keys, .env values,
                                            # canaries) in a request body, also base64 / hex / URL-encoded
    leak_files: str = ""                    # comma-separated KEY=VALUE files to fingerprint; "" = .env* in the cwd
    max_call_usd: float = 0.0               # 0 = off; refuse a single call likely to cost more than this
    max_request_kb: int = 0                 # 0 = off; refuse request bodies bigger than this (a bulk-upload limit)
    dashboard_writes: bool = True           # False: the dashboard API can't add / change keys or clear history (use
                                            # the CLI), so an agent that can reach llmtap's port can't change them
    # upstreams & pricing
    upstreams: dict[str, dict[str, str]] = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_UPSTREAMS.items()})
    pricing_overrides: dict[str, dict[str, float]] = field(default_factory=dict)  # model -> {"input":..,"output":..,"cache_read":..,"cache_write":..} USD per 1M tokens
    upstream_timeout_s: float = 600.0

    # OpenTelemetry OTLP/HTTP metrics endpoint, e.g. "http://localhost:4318/v1/metrics". Off when empty. Set it
    # explicitly (llmtap.toml / LLMTAP_OTEL_ENDPOINT): the generic OTEL_EXPORTER_OTLP_* variables are not read, so a
    # collector configured for other tools never receives llmtap data by surprise.
    otel_endpoint: str = ""

    def resolved_db_path(self) -> str:
        if self.db_path:
            return self.db_path
        home = llmtap_home()
        home.mkdir(parents=True, exist_ok=True)
        return str(home / "llmtap.db")

    def base_url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        return f"http://{host}:{self.port}"


_ENV_PREFIX = "LLMTAP_"
_TRUE, _FALSE = ("1", "true", "yes", "on"), ("0", "false", "no", "off", "")
_CHOICES: dict[str, tuple[str, ...]] = {
    "mode": ("record", "replay", "auto"),
    "guard": ("off", "warn", "redact", "block"),
    "leak_check": ("off", "warn", "block"),
}
_STYLES = ("anthropic", "openai", "gemini")


def _coerce(value: str, typ: Any) -> Any:
    t = typ if isinstance(typ, type) else type(typ)
    if t is bool:
        return value.strip().lower() in _TRUE
    if t is int:
        return int(value)
    if t is float:
        return float(value)
    return value


def _check(name: str, value: Any, where: str) -> Any:
    """Type-check / coerce one scalar field value; ValueError names `where` and the key."""
    t = type(next(f.default for f in fields(Config) if f.name == name))

    def bad(expected: str) -> ValueError:
        return ValueError(f"{where}{name}: expected {expected}, got {value!r}")

    if t is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in _TRUE + _FALSE:
            return _coerce(value, bool)
        raise bad("true or false")
    if t in (int, float):
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise bad("a number")
        try:
            num = _coerce(str(value).strip(), float)
            out = t(num) if t is float or float(num).is_integer() else None
        except (ValueError, OverflowError):
            out = None
        if out is None:
            raise bad("an integer" if t is int else "a number")
        if name == "port" and not 0 <= out <= 65535:
            raise bad("a port number 0-65535")
        return out
    if not isinstance(value, str):
        raise bad("a string")
    if name in _CHOICES and value not in _CHOICES[name]:
        raise bad("one of " + ", ".join(_CHOICES[name]))
    return value


def load_config(path: str | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Build a Config. `overrides` (e.g. from CLI) win; keys with value None are ignored.

    Raises ValueError (naming the file / variable and key) for invalid TOML or values."""
    cfg = Config()
    candidates = [Path(path)] if path else [Path("llmtap.toml"), Path.home() / ".config" / "llmtap" / "config.toml"]
    for p in candidates:
        if p.is_file():
            apply_file(cfg, p)
            break
    for f in fields(Config):
        var = _ENV_PREFIX + f.name.upper()
        env = os.environ.get(var)
        if env is not None and f.name not in ("upstreams", "pricing_overrides"):
            setattr(cfg, f.name, _check(f.name, env, f"environment variable {var}: "))
    if overrides:
        _apply(cfg, {k: v for k, v in overrides.items() if v is not None})
    return cfg


def apply_file(cfg: Config, path: Path) -> None:
    """Read a TOML config file (UTF-8) into `cfg`; ValueError names the file and offending key."""
    try:
        data = tomllib.loads(path.read_bytes().decode("utf-8"))
    except UnicodeDecodeError as e:
        raise ValueError(f"{path}: not valid UTF-8 ({e.reason})") from None
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path}: invalid TOML: {e}") from None
    section = data.get("llmtap", data)
    if not isinstance(section, dict):
        raise ValueError(f"{path}: llmtap: expected a table, got {section!r}")
    _apply(cfg, section, f"{path}: ")


def _apply(cfg: Config, data: dict[str, Any], where: str = "") -> None:
    names = {f.name for f in fields(Config)}
    for k, v in data.items():
        key = k
        k = k.replace("-", "_")
        if k == "pricing":
            k = "pricing_overrides"
        if k not in names:
            continue
        if k in ("upstreams", "pricing_overrides") and not isinstance(v, dict):
            raise ValueError(f"{where}{key}: expected a table, got {v!r}")
        if k == "upstreams":
            for name, spec in v.items():
                if not isinstance(spec, dict) or not isinstance(spec.get("base"), str) or not spec["base"].strip():
                    raise ValueError(f'{where}upstreams.{name}: expected a table with base = "http://..."')
                style = spec.get("style", "openai")
                if style not in _STYLES:
                    raise ValueError(f"{where}upstreams.{name}: style must be one of {', '.join(_STYLES)}, got {style!r}")
                cfg.upstreams[name] = {"base": spec["base"].strip().rstrip("/"), "style": style}
        elif k == "pricing_overrides":
            for model, price in v.items():
                if not isinstance(price, dict) or not all(
                        isinstance(x, (int, float)) and not isinstance(x, bool) for x in price.values()):
                    raise ValueError(f"{where}{key}.{model}: expected a table like {{ input = 1.0, output = 2.0 }}")
            cfg.pricing_overrides.update(v)
        else:
            setattr(cfg, k, _check(k, v, where))
