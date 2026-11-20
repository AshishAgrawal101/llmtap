"""Key vault: store provider API keys once and let apps (or AI coding agents) use them without ever seeing them.

You save a key once (dashboard "Keys" panel or ``llmtap keys add NAME``). The secret goes into the operating
system's credential store; llmtap keeps only non-secret metadata (provider, a masked preview, limits) in
``~/.llmtap/keys.json``.

Code then uses a *placeholder* instead of the key, e.g. ``OPENAI_API_KEY=llmtap-key-openai`` (``llmtap wrap`` sets
these for you). When a request carrying a placeholder passes through llmtap, the real key is swapped in on the way
out, only towards that key's own provider, and only if the key's limits (paused, expiry, allowed models, daily /
total spend) allow it. The real key is never returned by any API, never written to the history database, and
scrubbed from anything an upstream echoes back.

What this protects against: keys pasted into prompts, printed in logs, committed in ``.env`` files, or read by an
agent that dumps its environment. What it does not: a deliberately malicious program running as your own user can
read your OS credential store, or change the limits through the local API, just as you can.

Secret backends (``LLMTAP_VAULT_BACKEND``): ``auto`` (default) uses Windows Credential Manager, the macOS Keychain
(``security``) or the Linux Secret Service (``secret-tool``), and falls back to ``file``: ``~/.llmtap/keys.secret.json``
readable only by you (like ``~/.aws/credentials``; not encrypted). ``memory`` keeps secrets in this process only.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.parse import parse_qsl, quote, urlsplit

from .config import llmtap_home

SERVICE = "llmtap"
PLACEHOLDER_PREFIX = "llmtap-key-"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
PLACEHOLDER_RE = re.compile(re.escape(PLACEHOLDER_PREFIX) + r"([a-z0-9][a-z0-9_-]{0,39})(?![a-z0-9_-])")
MAX_SECRET_LEN = 2000
# Credential headers a placeholder may travel in (lowercase), plus the ``?key=`` query parameter (Gemini).
PLACEHOLDER_HEADERS = ("authorization", "x-api-key", "x-goog-api-key", "api-key", "openai-api-key")
PLACEHOLDER_QUERY = ("key", "api_key", "apikey")
TAG = "vault_key"          # Exchange tag naming the stored key a request used (clients cannot set it)

# Environment variables ``llmtap wrap`` fills with a placeholder for each provider that has a stored key.
PROVIDER_ENV: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
}


class VaultError(ValueError):
    """Invalid vault input or a failing secret backend; the message is safe to show (never contains a secret)."""


# ----------------------------------------------------------------------------------------------------- backends

class SecretBackend(Protocol):
    name: str

    def get(self, key: str) -> str | None: ...
    def set(self, key: str, secret: str) -> None: ...
    def delete(self, key: str) -> None: ...


class MemoryBackend:
    """Secrets in this process only (tests, throwaway sessions)."""
    name = "memory"

    def __init__(self) -> None:
        self._d: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._d.get(key)

    def set(self, key: str, secret: str) -> None:
        with self._lock:
            self._d[key] = secret

    def delete(self, key: str) -> None:
        with self._lock:
            self._d.pop(key, None)


class FileBackend:
    """A JSON file readable only by the current user (mode 0600 on POSIX). Not encrypted."""
    name = "file"

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            raise VaultError(f"cannot read {self.path}: {e}") from None
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def _write(self, data: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, self.path)

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._read().get(key)

    def set(self, key: str, secret: str) -> None:
        with self._lock:
            data = self._read()
            data[key] = secret
            self._write(data)

    def delete(self, key: str) -> None:
        with self._lock:
            data = self._read()
            if data.pop(key, None) is not None:
                self._write(data)


class WindowsBackend:
    """Windows Credential Manager (generic credentials named ``llmtap:<name>``), through advapi32."""
    name = "windows-credential-manager"
    _CRED_TYPE_GENERIC = 1
    _CRED_PERSIST_LOCAL_MACHINE = 2
    _ERROR_NOT_FOUND = 1168

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class FILETIME(ctypes.Structure):
            _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

        class CREDENTIALW(ctypes.Structure):
            _fields_ = [("Flags", wintypes.DWORD), ("Type", wintypes.DWORD), ("TargetName", wintypes.LPWSTR),
                        ("Comment", wintypes.LPWSTR), ("LastWritten", FILETIME),
                        ("CredentialBlobSize", wintypes.DWORD), ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
                        ("Persist", wintypes.DWORD), ("AttributeCount", wintypes.DWORD),
                        ("Attributes", ctypes.c_void_p), ("TargetAlias", wintypes.LPWSTR),
                        ("UserName", wintypes.LPWSTR)]

        self._ct, self._CRED = ctypes, CREDENTIALW
        adv = ctypes.WinDLL("advapi32", use_last_error=True)
        self._write = adv.CredWriteW
        self._write.argtypes = [ctypes.POINTER(CREDENTIALW), wintypes.DWORD]
        self._write.restype = wintypes.BOOL
        self._read = adv.CredReadW
        self._read.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                               ctypes.POINTER(ctypes.POINTER(CREDENTIALW))]
        self._read.restype = wintypes.BOOL
        self._delete = adv.CredDeleteW
        self._delete.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        self._delete.restype = wintypes.BOOL
        self._free = adv.CredFree
        self._free.argtypes = [ctypes.c_void_p]
        self._free.restype = None

    @staticmethod
    def _target(key: str) -> str:
        return f"{SERVICE}:{key}"

    def get(self, key: str) -> str | None:
        ct = self._ct
        p = ct.POINTER(self._CRED)()
        if not self._read(self._target(key), self._CRED_TYPE_GENERIC, 0, ct.byref(p)):
            err = ct.get_last_error()
            if err == self._ERROR_NOT_FOUND:
                return None
            raise VaultError(f"Windows Credential Manager read failed (error {err})")
        try:
            cred = p.contents
            raw = ct.string_at(cred.CredentialBlob, cred.CredentialBlobSize)
            return raw.decode("utf-8")
        finally:
            self._free(p)

    def set(self, key: str, secret: str) -> None:
        ct = self._ct
        blob = secret.encode("utf-8")
        buf = (ct.c_ubyte * len(blob)).from_buffer_copy(blob)
        cred = self._CRED()
        cred.Type = self._CRED_TYPE_GENERIC
        cred.TargetName = self._target(key)
        cred.Comment = "llmtap API key vault"
        cred.CredentialBlobSize = len(blob)
        cred.CredentialBlob = ct.cast(buf, ct.POINTER(ct.c_ubyte))
        cred.Persist = self._CRED_PERSIST_LOCAL_MACHINE
        cred.UserName = SERVICE
        if not self._write(ct.byref(cred), 0):
            raise VaultError(f"Windows Credential Manager write failed (error {ct.get_last_error()})")

    def delete(self, key: str) -> None:
        if not self._delete(self._target(key), self._CRED_TYPE_GENERIC, 0):
            err = self._ct.get_last_error()
            if err != self._ERROR_NOT_FOUND:
                raise VaultError(f"Windows Credential Manager delete failed (error {err})")


def _run(cmd: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=20)


class MacKeychainBackend:
    """The macOS login Keychain through ``security``. The secret goes over stdin, never on a command line."""
    name = "macos-keychain"

    def __init__(self, exe: str) -> None:
        self.exe = exe

    def get(self, key: str) -> str | None:
        r = _run([self.exe, "find-generic-password", "-s", SERVICE, "-a", key, "-w"])
        if r.returncode == 44:                      # errSecItemNotFound
            return None
        if r.returncode != 0:
            raise VaultError(f"macOS Keychain read failed (security exit {r.returncode})")
        return r.stdout.rstrip("\n")

    def set(self, key: str, secret: str) -> None:
        # `security -i` reads commands from stdin; secrets are validated to contain no quotes or whitespace.
        line = f'add-generic-password -U -s "{SERVICE}" -a "{key}" -l "llmtap: {key}" -w "{secret}"\n'
        r = _run([self.exe, "-i"], stdin=line)
        if r.returncode != 0 or "error" in (r.stderr or "").lower():
            raise VaultError(f"macOS Keychain write failed (security exit {r.returncode})")

    def delete(self, key: str) -> None:
        r = _run([self.exe, "delete-generic-password", "-s", SERVICE, "-a", key])
        if r.returncode not in (0, 44):
            raise VaultError(f"macOS Keychain delete failed (security exit {r.returncode})")


class SecretToolBackend:
    """The Linux Secret Service (GNOME Keyring, KWallet...) through ``secret-tool``; secrets go over stdin."""
    name = "secret-service"

    def __init__(self, exe: str) -> None:
        self.exe = exe

    def get(self, key: str) -> str | None:
        r = _run([self.exe, "lookup", "service", SERVICE, "account", key])
        if r.returncode != 0:
            return None if not r.stderr.strip() else self._fail("read", r)
        return r.stdout.rstrip("\n") or None

    def set(self, key: str, secret: str) -> None:
        r = _run([self.exe, "store", f"--label=llmtap: {key}", "service", SERVICE, "account", key], stdin=secret)
        if r.returncode != 0:
            self._fail("write", r)

    def delete(self, key: str) -> None:
        r = _run([self.exe, "clear", "service", SERVICE, "account", key])
        if r.returncode != 0 and r.stderr.strip():
            self._fail("delete", r)

    @staticmethod
    def _fail(what: str, r: subprocess.CompletedProcess) -> Any:
        raise VaultError(f"Secret Service {what} failed (secret-tool exit {r.returncode})")


_MEMORY = MemoryBackend()     # shared by every Vault in this process when LLMTAP_VAULT_BACKEND=memory


def _secret_tool_works(exe: str) -> bool:
    """secret-tool exists on many headless boxes without a running Secret Service; probe with a harmless lookup."""
    try:
        r = _run([exe, "lookup", "service", SERVICE, "account", "__llmtap_probe__"])
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode in (0, 1) and "error" not in (r.stderr or "").lower()


def make_backend(kind: str | None = None, home: Path | None = None) -> SecretBackend:
    kind = (kind or os.environ.get("LLMTAP_VAULT_BACKEND") or "auto").strip().lower()
    home = home or llmtap_home()
    if kind == "memory":
        return _MEMORY
    if kind == "file":
        return FileBackend(home / "keys.secret.json")
    if kind not in ("auto", "os"):
        raise VaultError(f"LLMTAP_VAULT_BACKEND: expected auto, os, file or memory, got {kind!r}")
    if sys.platform == "win32":
        return WindowsBackend()
    if sys.platform == "darwin" and shutil.which("security"):
        return MacKeychainBackend(shutil.which("security") or "security")
    st = shutil.which("secret-tool")
    if st and _secret_tool_works(st):
        return SecretToolBackend(st)
    if kind == "os":
        raise VaultError("no OS credential store found (install secret-tool, or set LLMTAP_VAULT_BACKEND=file)")
    return FileBackend(home / "keys.secret.json")


# ----------------------------------------------------------------------------------------------------- metadata

@dataclass
class KeyInfo:
    """Everything llmtap knows about a stored key, except the secret itself."""
    name: str
    provider: str
    preview: str = ""
    created: float = field(default_factory=time.time)
    rev: int = 1                       # bumped whenever the secret changes (other processes drop their cache)
    daily_usd: float = 0.0             # 0 = no daily cap
    total_usd: float = 0.0             # 0 = no lifetime cap (counted from `created`)
    models: list[str] = field(default_factory=list)   # allowed model patterns (fnmatch, case-insensitive); [] = any
    expires: float = 0.0               # epoch seconds; 0 = never
    paused: bool = False

    @property
    def placeholder(self) -> str:
        return PLACEHOLDER_PREFIX + self.name

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["placeholder"] = self.placeholder
        return d


def mask(secret: str) -> str:
    """A preview that identifies a key without revealing it: ``sk-p…a1b2``."""
    s = secret.strip()
    if len(s) >= 16:
        return f"{s[:4]}…{s[-4:]}"
    return f"…{s[-2:]}" if len(s) >= 6 else "…"


def _validate_name(name: str) -> str:
    n = (name or "").strip().lower()
    if not NAME_RE.match(n):
        raise VaultError("key name: use 1-40 lowercase letters, digits, '-' or '_', starting with a letter or digit")
    return n


def _validate_secret(secret: str) -> str:
    s = (secret or "").strip()
    if not s:
        raise VaultError("the key is empty")
    if len(s) > MAX_SECRET_LEN:
        raise VaultError(f"the key is longer than {MAX_SECRET_LEN} characters")
    if any(c.isspace() or c in "\"'\\" or ord(c) < 32 for c in s):
        raise VaultError("the key contains spaces, quotes or control characters; paste only the key itself")
    if s.startswith(PLACEHOLDER_PREFIX):
        raise VaultError("that is a llmtap placeholder, not a real API key")
    return s


def _validate_limits(limits: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in limits.items():
        if v is None:
            continue
        if k in ("daily_usd", "total_usd"):
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise VaultError(f"{k}: expected a number of dollars") from None
            if not (f >= 0) or f == float("inf"):
                raise VaultError(f"{k}: expected 0 (no cap) or a positive number")
            out[k] = round(f, 6)
        elif k == "expires":
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise VaultError("expires: expected a timestamp (0 = never)") from None
            if f < 0:
                raise VaultError("expires: expected a timestamp (0 = never)")
            out[k] = f
        elif k == "models":
            items = v.split(",") if isinstance(v, str) else v
            if not isinstance(items, (list, tuple)):
                raise VaultError("models: expected a list of model names or patterns")
            pats = [str(m).strip() for m in items if str(m).strip()]
            if any(len(p) > 200 for p in pats) or len(pats) > 100:
                raise VaultError("models: too many or too long patterns")
            out[k] = pats
        elif k == "paused":
            if not isinstance(v, bool):
                raise VaultError("paused: expected true or false")
            out[k] = v
        else:
            raise VaultError(f"unknown limit {k!r}")
    return out


# --------------------------------------------------------------------------------------------------------- vault

class Vault:
    """Key metadata (``keys.json``) + secrets (a backend). Safe to share between threads; other processes'
    changes (CLI vs. running proxy) are picked up on the next access."""

    def __init__(self, home: Path | None = None, backend: SecretBackend | None = None) -> None:
        self.home = Path(home) if home else llmtap_home()
        self.meta_path = self.home / "keys.json"
        self._backend = backend               # opened on first secret access (listing keys never needs it)
        self._lock = threading.RLock()
        self._keys: dict[str, KeyInfo] = {}
        self._mtime: float | None = None
        self._secrets: dict[str, tuple[int, str]] = {}     # name -> (rev, secret) cache
        self._inflight: dict[str, float] = {}               # name -> reserved USD of in-flight calls

    @property
    def backend(self) -> SecretBackend:
        with self._lock:
            if self._backend is None:
                self._backend = make_backend(home=self.home)
            return self._backend

    # -- metadata file --------------------------------------------------------------------------------
    def _load(self) -> None:
        try:
            mtime = self.meta_path.stat().st_mtime
        except FileNotFoundError:
            self._keys, self._mtime = {}, None
            return
        if mtime == self._mtime:
            return
        try:
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise VaultError(f"cannot read {self.meta_path}: {e}") from None
        keys: dict[str, KeyInfo] = {}
        known = set(KeyInfo.__dataclass_fields__)
        for item in (data.get("keys") if isinstance(data, dict) else None) or []:
            if isinstance(item, dict) and NAME_RE.match(str(item.get("name", ""))):
                keys[item["name"]] = KeyInfo(**{k: v for k, v in item.items() if k in known})
        self._keys, self._mtime = keys, mtime

    def _save(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        tmp = self.meta_path.with_name(self.meta_path.name + ".tmp")
        body = {"version": 1, "keys": [asdict(k) for k in sorted(self._keys.values(), key=lambda k: k.name)]}
        tmp.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.meta_path)
        try:
            self._mtime = self.meta_path.stat().st_mtime
        except OSError:
            self._mtime = None

    # -- reading ------------------------------------------------------------------------------------------
    def list(self) -> list[KeyInfo]:
        with self._lock:
            self._load()
            return sorted(self._keys.values(), key=lambda k: k.name)

    def get(self, name: str) -> KeyInfo | None:
        with self._lock:
            self._load()
            return self._keys.get((name or "").lower())

    def secret(self, name: str) -> str | None:
        """The real key, for the proxy's outgoing request only. Cached per revision."""
        with self._lock:
            self._load()
            info = self._keys.get(name)
            if info is None:
                return None
            hit = self._secrets.get(name)
            if hit and hit[0] == info.rev:
                return hit[1]
        value = self.backend.get(name)
        if value:
            with self._lock:
                self._secrets[name] = (info.rev, value)
        return value

    # -- writing ------------------------------------------------------------------------------------------
    def add(self, name: str, provider: str, secret: str, **limits: Any) -> KeyInfo:
        n = _validate_name(name)
        p = (provider or "").strip().lower()
        if not NAME_RE.match(p):
            raise VaultError("provider: expected an upstream name such as openai, anthropic or gemini")
        s = _validate_secret(secret)
        lim = _validate_limits(limits)
        with self._lock:
            self._load()
            old = self._keys.get(n)
            self.backend.set(n, s)
            info = KeyInfo(name=n, provider=p, preview=mask(s), rev=(old.rev + 1) if old else 1,
                           created=old.created if old and old.provider == p else time.time())
            for k, v in lim.items():
                setattr(info, k, v)
            if old and not lim:        # replacing the secret keeps the limits you set
                for k in ("daily_usd", "total_usd", "models", "expires", "paused"):
                    setattr(info, k, getattr(old, k))
            self._keys[n] = info
            self._secrets.pop(n, None)
            self._save()
            return info

    def update(self, name: str, **limits: Any) -> KeyInfo:
        lim = _validate_limits(limits)
        with self._lock:
            self._load()
            info = self._keys.get((name or "").lower())
            if info is None:
                raise VaultError(f"no stored key named {name!r}")
            for k, v in lim.items():
                setattr(info, k, v)
            self._save()
            return info

    def remove(self, name: str) -> bool:
        n = (name or "").lower()
        with self._lock:
            self._load()
            if n not in self._keys:
                return False
            self.backend.delete(n)
            del self._keys[n]
            self._secrets.pop(n, None)
            self._save()
            return True

    # -- wiring -------------------------------------------------------------------------------------------
    def child_env(self, current: Mapping[str, str]) -> dict[str, str]:
        """Placeholders for ``llmtap wrap``: one per provider env var that has a stored key.

        A key named exactly like its provider wins, else the alphabetically first one. A variable that already
        holds a placeholder is left alone (you picked a key explicitly)."""
        chosen: dict[str, KeyInfo] = {}
        for k in self.list():
            if k.provider not in PROVIDER_ENV:
                continue
            cur = chosen.get(k.provider)
            if cur is None or (k.name == k.provider and cur.name != cur.provider):
                chosen[k.provider] = k
        out: dict[str, str] = {}
        for prov, info in chosen.items():
            for var in PROVIDER_ENV[prov]:
                if not str(current.get(var, "")).startswith(PLACEHOLDER_PREFIX):
                    out[var] = info.placeholder
        return out

    # -- spend reservations (per-key caps) ---------------------------------------------------------------
    def admit(self, info: KeyInfo, model: str | None, spent_today: float, spent_total: float,
              estimate: float, priced: bool) -> Refusal | None:
        """Check ``info``'s limits and, if allowed and priced, reserve ``estimate`` atomically.

        Call :meth:`release` with the same estimate when the request has finished."""
        with self._lock:
            refusal = check_limits(info, model, spent_today, spent_total, self._inflight.get(info.name, 0.0), priced)
            if refusal is None and priced:
                self._inflight[info.name] = self._inflight.get(info.name, 0.0) + max(0.0, estimate)
            return refusal

    def release(self, name: str, usd: float) -> None:
        with self._lock:
            left = self._inflight.get(name, 0.0) - max(0.0, usd)
            if left > 1e-12:
                self._inflight[name] = left
            else:
                self._inflight.pop(name, None)

    def inflight(self, name: str) -> float:
        with self._lock:
            return self._inflight.get(name, 0.0)


# ------------------------------------------------------------------------------------------- placeholder swap

@dataclass
class Found:
    name: str
    where: str          # header name (lowercase) or "?<param>"


def find_placeholder(headers: Mapping[str, str], path: str) -> Found | None:
    """The first ``llmtap-key-<name>`` placeholder in a credential header or the ``?key=`` parameter."""
    for h in PLACEHOLDER_HEADERS:
        v = headers.get(h)
        if v and PLACEHOLDER_PREFIX in v:
            m = PLACEHOLDER_RE.search(v)
            if m:
                return Found(m.group(1), h)
    if "?" in path and PLACEHOLDER_PREFIX in path:
        for k, v in parse_qsl(urlsplit(path).query, keep_blank_values=True):
            if k.lower() in PLACEHOLDER_QUERY:
                m = PLACEHOLDER_RE.fullmatch(v.strip())
                if m:
                    return Found(m.group(1), "?" + k)
    return None


def substitute(headers: dict[str, str], path: str, found: Found, secret: str) -> tuple[dict[str, str], str]:
    """Copies of ``headers`` / ``path`` with the placeholder replaced by the real key (in that one place)."""
    ph = PLACEHOLDER_PREFIX + found.name
    hdrs = dict(headers)
    if not found.where.startswith("?"):
        hdrs[found.where] = PLACEHOLDER_RE.sub(lambda m: secret if m.group(1) == found.name else m.group(0),
                                               hdrs[found.where])
        return hdrs, path
    param = found.where[1:]
    base, _, query = path.partition("?")
    parts = []
    for piece in query.split("&"):
        k, eq, v = piece.partition("=")
        if eq and k == param and v.strip() == ph:
            piece = f"{k}={quote(secret, safe='')}"
        parts.append(piece)
    return hdrs, base + "?" + "&".join(parts)


@dataclass
class Refusal:
    status: int
    message: str
    err_type: str


def _local_midnight(now: float | None = None) -> float:
    t = time.localtime(now if now is not None else time.time())
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))


def model_allowed(info: KeyInfo, model: str | None) -> bool:
    if not info.models:
        return True
    m = (model or "").lower()
    return bool(m) and any(fnmatch.fnmatchcase(m, p.lower()) for p in info.models)


def check_limits(info: KeyInfo, model: str | None, spent_today: float, spent_total: float,
                 pending: float, priced: bool, now: float | None = None) -> Refusal | None:
    """Why a call through stored key ``info`` must be refused, or None.

    Spend caps apply to priced (generation) calls and work like the global budgets: refused once what was spent
    plus what is in flight (``pending``) reaches the cap."""
    now = time.time() if now is None else now
    if info.paused:
        return Refusal(403, f"llmtap: the stored key {info.name!r} is paused. Resume it in the llmtap dashboard "
                            f"(Keys) or with `llmtap keys limit {info.name} --resume`.", "llmtap_key_paused")
    if info.expires and now >= info.expires:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(info.expires))
        return Refusal(403, f"llmtap: the stored key {info.name!r} expired on {when}.", "llmtap_key_expired")
    if priced and not model_allowed(info, model):
        allowed = ", ".join(info.models)
        return Refusal(403, f"llmtap: the stored key {info.name!r} may only be used with: {allowed} "
                            f"(this request asked for {model or 'no model'}).", "llmtap_key_model_not_allowed")
    if priced:
        if info.daily_usd and spent_today + pending >= info.daily_usd:
            return Refusal(429, f"llmtap: daily limit of ${info.daily_usd:.2f} for the stored key {info.name!r} "
                                f"reached (${spent_today:.4f} spent today). Raise it in the llmtap dashboard (Keys) "
                                "or wait until midnight.", "llmtap_key_limit")
        if info.total_usd and spent_total + pending >= info.total_usd:
            return Refusal(429, f"llmtap: total limit of ${info.total_usd:.2f} for the stored key {info.name!r} "
                                f"reached (${spent_total:.4f} spent). Raise it in the llmtap dashboard (Keys).",
                           "llmtap_key_limit")
    return None
