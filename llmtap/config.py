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
    # name -> {"base": url, "style": "anthropic"|"openai"}
    "anthropic": {"base": "https://api.anthropic.com", "style": "anthropic"},
    "openai": {"base": "https://api.openai.com", "style": "openai"},
    "openrouter": {"base": "https://openrouter.ai/api", "style": "openai"},
    "groq": {"base": "https://api.groq.com/openai", "style": "openai"},
    "together": {"base": "https://api.together.xyz", "style": "openai"},
    "mistral": {"base": "https://api.mistral.ai", "style": "openai"},
    "deepseek": {"base": "https://api.deepseek.com", "style": "openai"},
    "ollama": {"base": "http://localhost:11434", "style": "openai"},
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
    # upstreams & pricing
    upstreams: dict[str, dict[str, str]] = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_UPSTREAMS.items()})
    pricing_overrides: dict[str, dict[str, float]] = field(default_factory=dict)  # model -> {"input":..,"output":..,"cache_read":..,"cache_write":..} USD per 1M tokens
    upstream_timeout_s: float = 600.0

    def resolved_db_path(self) -> str:
        if self.db_path:
            return self.db_path
        home = llmtap_home()
        home.mkdir(parents=True, exist_ok=True)
        return str(home / "llmtap.db")

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


_ENV_PREFIX = "LLMTAP_"


def _coerce(value: str, typ: Any) -> Any:
    t = typ if isinstance(typ, type) else type(typ)
    if t is bool:
        return value.strip().lower() in ("1", "true", "yes", "on")
    if t is int:
        return int(value)
    if t is float:
        return float(value)
    return value


def load_config(path: str | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Build a Config. `overrides` (e.g. from CLI) win; keys with value None are ignored."""
    cfg = Config()
    candidates = [Path(path)] if path else [Path("llmtap.toml"), Path.home() / ".config" / "llmtap" / "config.toml"]
    for p in candidates:
        if p.is_file():
            data = tomllib.loads(p.read_text())
            _apply(cfg, data.get("llmtap", data))
            break
    for f in fields(Config):
        env = os.environ.get(_ENV_PREFIX + f.name.upper())
        if env is not None and f.name not in ("upstreams", "pricing_overrides"):
            setattr(cfg, f.name, _coerce(env, getattr(cfg, f.name)))
    if overrides:
        _apply(cfg, {k: v for k, v in overrides.items() if v is not None})
    return cfg


def _apply(cfg: Config, data: dict[str, Any]) -> None:
    names = {f.name for f in fields(Config)}
    for k, v in data.items():
        k = k.replace("-", "_")
        if k == "pricing":
            k = "pricing_overrides"
        if k not in names:
            continue
        if k == "upstreams":
            for name, spec in v.items():
                cfg.upstreams[name] = {"base": spec["base"].rstrip("/"), "style": spec.get("style", "openai")}
        elif k == "pricing_overrides":
            cfg.pricing_overrides.update(v)
        else:
            setattr(cfg, k, v)
