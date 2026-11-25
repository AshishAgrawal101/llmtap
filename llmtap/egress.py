"""What may leave: checks on the *content* of outgoing requests (the spend caps only limit the bill).

* **Leak tripwire** (:class:`Tripwire`): fingerprints of your real secrets (stored keys, values in ``.env``
  files, canaries) and the encodings an agent might use to smuggle them (base64 at every alignment, hex,
  URL-encoding). Any outgoing request body that contains one is blocked (or warned about). Unlike the regex
  guard, it never guesses: it only matches values you actually have.
* **Canaries**: fake secrets written into a ``.env`` file. Nothing legitimate ever sends them, so seeing one in a
  prompt means an agent read the file.
* **Provider-side tools** (:func:`server_tools`): web search / fetch, code execution, remote MCP servers and URLs
  the provider fetches itself. They make the provider send data to other places, so stored keys refuse them
  unless the key allows it.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import secrets as _secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urlsplit

from .config import llmtap_home

__all__ = ["Leak", "Tripwire", "server_tools", "parse_env", "secretish", "variants", "new_canary",
           "load_canaries", "add_canary", "remove_canary"]

MIN_SECRET_LEN = 10           # shorter values ("postgres", "password") are words, not secrets worth blocking on
MIN_VARIANT_LEN = 12          # shorter encoded fragments match too much by chance
# variable names that hold secrets, matched on whole name parts (SESSION_DRIVER and AUTH_DOMAIN are not secrets)
_SECRET_NAME = re.compile(r"(?i)(?:^|[_.-])(?:API_?KEY|KEY|SECRET|TOKEN|PASSWORD|PASSWD|PASS|PWD|CREDENTIALS?|DSN|"
                          r"WEBHOOK(?:_URL)?|PRIVATE_KEY|SIGNING_KEY|ACCESS_KEY|CLIENT_SECRET|SALT)(?:$|[_.-])")
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_.-]*)\s*[=:]\s*(.*?)\s*$")
_VAR_REF = re.compile(r"^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?$|^%[A-Za-z_][A-Za-z0-9_]*%$")
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0", "db", "postgres", "mysql", "redis", "host.docker.internal")


# ------------------------------------------------------------------------------------------------ .env files

def parse_env(text: str) -> list[tuple[str, str]]:
    """``KEY=value`` pairs of a dotenv file (quotes removed, ``# comments`` skipped). Never raises.

    A quoted value that spans several lines (a PEM key) is read up to its closing quote."""
    out: list[tuple[str, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            continue
        name, value = m.group(1), m.group(2)
        if value[:1] in ("\"", "'", "`"):
            q = value[0]
            end = value.find(q, 1)
            if end < 0:                                     # multi-line: read on to the closing quote
                parts = [value[1:]]
                while i < len(lines):
                    nxt = lines[i]
                    i += 1
                    if q in nxt:
                        parts.append(nxt[:nxt.index(q)])
                        break
                    parts.append(nxt)
                value = "\n".join(parts)
            else:
                value = value[1:end]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        out.append((name, value))
    return out


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    n = len(s)
    return -sum(k / n * math.log2(k / n) for k in counts.values())


def _char_classes(s: str) -> int:
    return (any(c.islower() for c in s) + any(c.isupper() for c in s) + any(c.isdigit() for c in s)
            + any(not c.isalnum() for c in s))


def _shaped_like_secret(v: str, strict: bool = False) -> bool:
    """A value that looks generated rather than typed: long enough, no spaces, mixed characters."""
    if len(v) < (20 if strict else MIN_SECRET_LEN) or any(c.isspace() for c in v):
        return False
    if strict:
        return _char_classes(v) >= 2 and _entropy(v) >= 3.5
    return _char_classes(v) >= 2 or _entropy(v) >= 3.2


def secretish(name: str, value: str) -> bool:
    """Is this ``.env`` value worth fingerprinting? Values of secret-named variables (``*_KEY``, ``*_SECRET``,
    ``*_TOKEN``, ``*_PASSWORD``...) that look generated, and long random-looking values of any variable. Plain
    words, numbers, paths, variable references and URLs without credentials are not."""
    v = value.strip()
    if v.startswith("-----BEGIN") and "\n" in v:            # a multi-line PEM block: its body is the secret
        return True
    if len(v) < MIN_SECRET_LEN or v.startswith("llmtap-key-") or _VAR_REF.match(v):
        return False
    if v.replace(".", "").replace("-", "").isdigit():
        return False
    if v.startswith(("/", "./", "../", "~")) or re.match(r"^[A-Za-z]:[\/]", v):
        return False
    low = v.lower()
    if any(w in low for w in ("your_", "your-", "example", "changeme", "placeholder", "xxxxxx", "<", "${")):
        return False
    named = bool(_SECRET_NAME.search(name))
    if "://" in v:
        try:
            u = urlsplit(v)
        except ValueError:
            return False
        if u.password and (u.hostname or "").lower() in _LOCAL_HOSTS and not _shaped_like_secret(u.password):
            return False                                    # postgres://postgres:postgres@localhost/app
        # a URL is a secret when it carries credentials, or is a webhook / DSN by name (the path is the secret)
        return bool(u.password or (u.username and len(u.username) >= 16) or u.query) or named
    if named:
        return _shaped_like_secret(v)
    return _shaped_like_secret(v, strict=True)


def url_password(value: str) -> str | None:
    """The password of a connection string, when it is worth fingerprinting on its own."""
    try:
        u = urlsplit(value)
        pw, user, host = u.password, u.username, (u.hostname or "").lower()
    except ValueError:
        return None
    if not pw or pw == user or host in _LOCAL_HOSTS or not _shaped_like_secret(pw):
        return None
    return pw


def _label(text: str) -> str:
    """File and variable names go into messages, terminal output and notifications: printable ASCII only."""
    return "".join(c if 32 <= ord(c) < 127 and c not in "'\"`$;&|<>" else "?" for c in text)[:80]


# ------------------------------------------------------------------------------------------------- encodings

def _b64_cores(raw: bytes, alphabet: str) -> list[str]:
    """The base64 characters that depend only on ``raw``, for each of the 3 possible offsets inside a larger
    encoded blob (an agent that base64-encodes a whole .env file puts the secret at any offset)."""
    cores = []
    for pad in range(3):
        enc = base64.b64encode(b"\0" * pad + raw)
        if alphabet == "url":
            enc = enc.translate(bytes.maketrans(b"+/", b"-_"))
        start = -(-pad * 4 // 3)                 # ceil(4*pad/3): first char made only of the secret's bits
        end = (pad + len(raw)) * 4 // 3          # floor: last char made only of the secret's bits
        core = enc[start:end].decode("ascii")
        if len(core) >= MIN_VARIANT_LEN:
            cores.append(core)
    return cores


def variants(secret: str) -> list[tuple[str, str]]:
    """(encoding, text) forms of a secret to look for: as is, JSON-escaped, URL-encoded, hex, base64."""
    raw = secret.encode("utf-8")
    out: list[tuple[str, str]] = [("plain", secret)]
    esc = json.dumps(secret)[1:-1]
    if esc != secret:
        out.append(("json-escaped", esc))
    for safe in ("", "/"):
        q = quote(secret, safe=safe)
        if q != secret:
            out.append(("url-encoded", q))
            out.append(("url-encoded", re.sub(r"%[0-9A-F]{2}", lambda m: m.group(0).lower(), q)))
    out.append(("hex", raw.hex()))
    out.append(("hex", raw.hex().upper()))
    for data in (raw, secret.encode("utf-16-le")):         # UTF-16: PowerShell's -EncodedCommand / [Convert]
        for alphabet in ("std", "url"):
            for core in _b64_cores(data, alphabet):
                out.append(("base64", core))
    seen: set[str] = set()
    uniq = []
    for enc, text in out:
        if text not in seen and (enc == "plain" or len(text) >= MIN_VARIANT_LEN):
            seen.add(text)
            uniq.append((enc, text))
    return uniq


# --------------------------------------------------------------------------------------------------- canaries

def _canary_path(home: Path | None = None) -> Path:
    return (home or llmtap_home()) / "canaries.json"


def new_canary() -> str:
    """A fake secret that looks like a real token (and matches no real provider's format)."""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "tk_" + "".join(_secrets.choice(alphabet) for _ in range(40))


def load_canaries(home: Path | None = None) -> list[dict[str, Any]]:
    try:
        data = json.loads(_canary_path(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = data.get("canaries") if isinstance(data, dict) else None
    return [c for c in items or [] if isinstance(c, dict) and isinstance(c.get("value"), str)]


def _save_canaries(items: list[dict[str, Any]], home: Path | None = None) -> None:
    from .vault import _atomic_write
    path = _canary_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps({"version": 1, "canaries": items}, indent=2) + "\n", private=True)


def add_canary(env_file: Path, name: str, home: Path | None = None) -> dict[str, Any]:
    """Append ``NAME=<fake secret>`` to ``env_file`` and remember the value. Returns the canary record."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name):
        raise ValueError("canary name: use letters, digits and '_' (like a normal environment variable)")
    env_file = Path(env_file)
    existing = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    if any(k == name for k, _ in parse_env(existing)):
        raise ValueError(f"{env_file} already has a variable named {name}")
    value = new_canary()
    sep = "" if not existing or existing.endswith("\n") else "\n"
    with open(env_file, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"{sep}{name}={value}\n")
    rec = {"name": name, "file": str(env_file.resolve()), "value": value, "created": time.time()}
    items = load_canaries(home)
    items.append(rec)
    _save_canaries(items, home)
    return rec


def remove_canary(name: str, home: Path | None = None) -> int:
    """Forget canaries named ``name`` (the line in the .env file is left for you to delete). Returns how many."""
    items = load_canaries(home)
    keep = [c for c in items if c.get("name") != name]
    if len(keep) != len(items):
        _save_canaries(keep, home)
    return len(items) - len(keep)


# --------------------------------------------------------------------------------------------------- tripwire

@dataclass
class Leak:
    """One fingerprinted secret found in a request. Never contains the secret itself."""
    label: str                 # e.g. "OPENAI_API_KEY from .env", "stored key 'openai'", "canary ADMIN_TOKEN"
    encoding: str              # plain / base64 / hex / url-encoded / json-escaped
    canary: bool = False
    forms: list[str] = field(default_factory=list, repr=False)   # every encoded form, for scrubbing storage

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "canary" if self.canary else "leak", "path": self.label, "preview": self.encoding,
                "start": 0, "end": 0}


@dataclass
class _Print:
    label: str
    canary: bool
    forms: list[tuple[str, bytes]] = field(default_factory=list)


class Tripwire:
    """Exact-match detector for the user's real secrets in outgoing request bodies.

    Sources are re-read when they change (checked at most every ``refresh_s`` seconds), so a key added with
    ``llmtap keys add`` or a new canary is covered without restarting the proxy."""

    def __init__(self, env_files: Iterable[Path] = (), vault: Any = None, home: Path | None = None,
                 extra: Iterable[tuple[str, str]] = (), refresh_s: float = 2.0) -> None:
        self.env_files = [Path(p) for p in env_files]
        self._vault_src = vault          # a Vault, or a callable returning one (opened on first use)
        self.home = home
        self.extra = list(extra)
        self.refresh_s = refresh_s
        self._lock = threading.Lock()
        self._prints: list[_Print] = []
        self._stamp: tuple[Any, ...] | None = None
        self._checked = 0.0

    @property
    def vault(self) -> Any:
        src = self._vault_src
        if callable(src) and not hasattr(src, "list"):
            try:
                return src()
            except Exception:
                return None
        return src

    # -- sources ----------------------------------------------------------------------------------
    def _sources_stamp(self) -> tuple[Any, ...]:
        parts: list[Any] = []
        for p in self.env_files + [_canary_path(self.home)]:
            try:
                st = p.stat()
                parts.append((str(p), st.st_mtime_ns, st.st_size))
            except OSError:
                parts.append((str(p), None))
        vault = self.vault
        if vault is not None:
            try:
                parts.append(tuple((k.name, k.rev, k.uid) for k in vault.list()))
            except Exception:
                parts.append("vault?")
        return tuple(parts)

    def _collect(self) -> list[tuple[str, str, bool]]:
        found: list[tuple[str, str, bool]] = []
        canary_values = set()
        for c in load_canaries(self.home):
            found.append((f"canary {_label(str(c.get('name', '?')))} from "
                          f"{_label(Path(str(c.get('file', '.env'))).name)}", c["value"], True))
            canary_values.add(c["value"])
        for p in self.env_files:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for name, value in parse_env(text):
                if value in canary_values or not secretish(name, value):
                    continue
                found.append((f"{_label(name)} from {_label(p.name)}", value, False))
                if "://" in value:                 # a connection string: its password on its own too
                    pw = url_password(value)
                    if pw:
                        found.append((f"the password in {name} from {_label(p.name)}", pw, False))
        vault = self.vault
        if vault is not None:
            try:
                for k in vault.list():
                    s = vault.secret(k.name)
                    if s:
                        found.append((f"the stored key {k.name!r}", s, False))
            except Exception:
                pass
        for label, value in self.extra:
            if len(value) >= MIN_SECRET_LEN:
                found.append((label, value, False))
        return found

    def _refresh(self) -> list[_Print]:
        now = time.monotonic()
        with self._lock:
            if self._stamp is not None and now - self._checked < self.refresh_s:
                return self._prints
            self._checked = now
            stamp = self._sources_stamp()
            if stamp == self._stamp:
                return self._prints
            prints: list[_Print] = []
            seen: set[str] = set()
            for label, value, canary in self._collect():
                if value in seen:
                    continue
                seen.add(value)
                prints.append(_Print(label, canary, [(enc, t.encode("utf-8")) for enc, t in variants(value)]))
            self._prints, self._stamp = prints, stamp
            return prints

    def count(self) -> int:
        return len(self._refresh())

    # -- matching ---------------------------------------------------------------------------------
    def scan(self, body: bytes) -> list[Leak]:
        """Every fingerprinted secret present in ``body`` (raw bytes as they will be sent)."""
        if not body:
            return []
        prints = self._refresh()
        if not prints:
            return []
        hay = [body]
        # base64 wrapped at 76 columns, "xx:xx:xx" / "\xNN" / spaced hex: also search with separators removed
        squeezed = _SQUEEZE.sub(b"", body)
        if squeezed != body:
            hay.append(squeezed)
        # JSON escapes ("A", "\/") hide a plain value from a byte search: also look at the decoded strings
        if b"\\" in body:
            try:
                decoded = _json_strings(json.loads(body.decode("utf-8")))
                if decoded:
                    hay.append(decoded.encode("utf-8", "surrogatepass"))
            except (UnicodeDecodeError, ValueError, RecursionError):
                pass
        out: list[Leak] = []
        for p in prints:
            for enc, form in p.forms:
                if any(form in h for h in hay):
                    out.append(Leak(p.label, enc, p.canary, [f.decode("utf-8", "replace") for _, f in p.forms]))
                    break
        return out

    def redact(self, body: bytes) -> bytes:
        """``body`` with every fingerprinted form replaced (for storage; the original is never stored)."""
        for p in self._refresh():
            for _enc, form in sorted(p.forms, key=lambda f: len(f[1]), reverse=True):
                if form in body:
                    body = body.replace(form, b"[LEAK-REDACTED]")
        return body


_SQUEEZE = re.compile(rb"\\\\x|\\x|\\[nrt]|[\s:]")


def _json_strings(obj: Any) -> str:
    parts: list[str] = []
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, str):
            parts.append(o)
        elif isinstance(o, dict):
            for k, v in o.items():
                parts.append(k)
                stack.append(v)
        elif isinstance(o, list):
            stack.extend(o)
    return "\n".join(parts)


def default_env_files(cwd: Path | None = None) -> list[Path]:
    """``.env`` and ``.env.*`` in the working directory (not ``.env.example`` style templates)."""
    base = Path(cwd or os.getcwd())
    out = []
    try:
        for p in sorted(base.iterdir()):
            n = p.name
            if (n == ".env" or n.startswith(".env.")) and p.is_file() and not n.endswith(
                    (".example", ".sample", ".template", ".dist")):
                out.append(p)
    except OSError:
        pass
    return out


# ---------------------------------------------------------------------------------------- provider-side tools

# tool "type" values (Anthropic / OpenAI / Groq) that run on the provider and can reach other sites
_SERVER_TYPE = re.compile(r"^(web_search|web_fetch|code_execution|bash_code_execution|code_interpreter|mcp|"
                          r"mcp_toolset|url_context|google_search|browser_search|computer_use_preview_remote)(_|$)",
                          re.I)
# top-level tools[] entries with these types run on the caller's machine, or never leave the provider
_CLIENT_TYPE = re.compile(r"^(function|custom|bash|text_editor|str_replace_editor|str_replace_based_edit_tool|"
                          r"computer|computer_use_preview|memory|local_shell|shell|apply_patch|tool_search_tool|"
                          r"file_search|image_generation)(_|$)", re.I)
# object keys that switch a provider-side tool on (compared lowercase, without "_" / "-")
_SERVER_KEYS = {
    "mcpservers": "remote MCP servers (mcp_servers)",
    "websearchoptions": "web search (web_search_options)",
    "searchparameters": "web search (search_parameters)",
    "googlesearch": "Google Search grounding",
    "googlesearchretrieval": "Google Search grounding",
    "googlemaps": "Google Maps grounding",
    "urlcontext": "URL context (the provider fetches URLs)",
    "codeexecution": "code execution",
    "enterprisewebsearch": "web search",
    "container": "a code execution container",
    "plugins": "provider plugins (e.g. OpenRouter web search)",
}
# Gemini tools[] objects / Bedrock toolConfig.tools[] objects: anything but these keys is a built-in tool
_GEMINI_TOOL_KEYS = frozenset({"functiondeclarations", "toolspec", "cachepoint"})
# models that search the web by themselves, with no tool in the body
_SEARCH_MODEL = re.compile(r"(?i)(search|:online$|^groq/compound|/compound|sonar|deep-research)")
_URL_KEYS = {"image_url", "file_url", "fileuri", "file_uri", "document_url", "audio_url", "video_url"}
_PROVIDER_FILE_HOSTS = ("generativelanguage.googleapis.com", "api.openai.com", "api.anthropic.com")


def _remote_url(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    s = v.strip()
    low = s.lower()
    if low.startswith("gs://"):
        return True           # a bucket the provider reads: the object path reaches that bucket's logs
    if not low.startswith(("http://", "https://")):
        return False          # data: URLs and file ids are not fetched from the open web
    if "\\" in s or "@" in s:
        return True           # parsers disagree on the host of these: never treat them as the provider's own
    try:
        host = (urlsplit(s).hostname or "").lower()
    except ValueError:
        return True
    return not any(host == h or host.endswith("." + h) for h in _PROVIDER_FILE_HOSTS)


class _Pairs(list):
    """A JSON object kept as its (key, value) pairs, duplicates included."""


# subtrees that hold a client tool's schema, arguments or results: their keys are data, not request options
_DATA_KEYS = frozenset({"inputschema", "parameters", "input", "arguments", "args", "schema", "properties",
                        "responseformat", "text", "jsonschema", "metadata", "response", "output",
                        "functionresponse", "toolresult"})


def server_tools(body: bytes) -> list[str] | None:
    """Provider-side tools, web-searching models and provider-fetched URLs a JSON request body asks for.

    [] when there are none; None when the body isn't JSON llmtap can read (multipart uploads, JSONL, broken or
    absurdly nested JSON), so the caller can refuse what it can't check. Walks every object, duplicate keys
    included; tool schemas and tool-call arguments are skipped (a client tool with a ``url`` parameter is not a
    fetch). Top-level tool lists are an allowlist: an unknown tool type counts as provider-side."""
    found: list[str] = []

    def add(what: str) -> None:
        if what not in found:
            found.append(what)

    if not body.strip():
        return []
    try:
        root = json.loads(body.decode("utf-8-sig"), object_pairs_hook=_Pairs)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    stack: list[tuple[Any, int, str]] = [(root, 0, "")]
    while stack:
        node, depth, parent = stack.pop()
        if isinstance(node, _Pairs):
            is_url_source = any(k == "type" and v == "url" for k, v in node)
            in_tool_list = parent in ("tools",)
            keys = {k.lower().replace("_", "").replace("-", "") for k, _ in node}
            if in_tool_list and "type" not in keys and "name" not in keys and "toolspec" not in keys:
                for k in keys - _GEMINI_TOOL_KEYS:          # Gemini / Bedrock built-in tools: {"googleSearch": {}}
                    add(f"the built-in tool {k!r}")
            for k, v in node:
                kl = k.lower()
                norm = kl.replace("_", "").replace("-", "")
                if kl == "type" and isinstance(v, str):
                    if _SERVER_TYPE.match(v):
                        add(f"the {v} tool")
                    elif in_tool_list and depth <= 2 and not _CLIENT_TYPE.match(v):
                        add(f"the {v} tool (unknown tool types are treated as provider-side)")
                elif kl in ("model", "agent") and isinstance(v, str) and _SEARCH_MODEL.search(v):
                    add(f"the model {v} (it searches the web by itself)")
                elif kl == "systemtool":
                    add("a Bedrock system tool")
                elif norm in _SERVER_KEYS and not (v is None or v is False or v == "" or (
                        type(v) is list and not v)) and (norm != "container" or depth == 0):
                    add(_SERVER_KEYS[norm])
                elif kl in _URL_KEYS or norm in _URL_KEYS:
                    url = dict(v).get("url") if isinstance(v, _Pairs) else v
                    if _remote_url(url):
                        add("a URL the provider fetches itself")
                elif kl == "url" and is_url_source and _remote_url(v):
                    add("a URL the provider fetches itself")
                elif kl == "server_url" and isinstance(v, str):
                    add("a remote MCP server")
                if norm in _DATA_KEYS and depth > 0:
                    continue
                if isinstance(v, (list, _Pairs)):
                    stack.append((v, depth + 1, kl))
        elif isinstance(node, list):
            stack.extend((v, depth, parent) for v in node)
    return found
