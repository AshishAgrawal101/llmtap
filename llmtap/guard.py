"""Secret / PII firewall for prompts.

Scans request bodies for leaked credentials (and optionally PII) before they leave the machine.
Designed to run on every request: all regexes are precompiled, cheap literal prefilters skip
patterns that cannot match, and base64 media payloads (images, audio, data: URLs) are skipped.
"""
from __future__ import annotations

import bisect
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable

__all__ = ["Finding", "KINDS", "scan_text", "scan_obj", "redact_obj", "apply_guard"]

KINDS: dict[str, str] = {
    "aws_access_key": "AWS access key ID (AKIA…/ASIA…)",
    "aws_secret_key": "AWS secret access key",
    "anthropic_key": "Anthropic API key (sk-ant-…)",
    "openai_key": "OpenAI API key (sk-…/sk-proj-…)",
    "github_token": "GitHub token (ghp_/gho_/ghs_/github_pat_…)",
    "slack_token": "Slack token (xoxb-/xoxp-…)",
    "stripe_key": "Stripe live secret key (sk_live_/rk_live_…)",
    "google_api_key": "Google API key (AIza…)",
    "private_key": "PEM private key block",
    "jwt": "JSON Web Token",
    "generic_secret": "Hard-coded password / secret / token assignment",
    "email": "Email address (PII)",
    "phone": "Phone number (PII)",
    "credit_card": "Credit card number (PII, Luhn-validated)",
}

PII_KINDS: frozenset[str] = frozenset({"email", "phone", "credit_card"})


@dataclass
class Finding:
    """One detected secret. `start`/`end` are offsets into the string located at `path`."""
    kind: str
    path: str
    preview: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------------
# validators (return True to keep the match)

_PLACEHOLDER_WORDS = ("your", "example", "placeholder", "xxxx", "changeme", "redacted",
                      "dummy", "<", ">", "${", "{{", "}}", "****", "....")
_PATHLIKE_PREFIXES = ("/", "~", "./", "../", "$", "%", "http://", "https://", "file://")
_DOTTED_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


def _char_classes(s: str) -> int:
    lo = up = dg = ot = False
    for c in s:
        if c.islower():
            lo = True
        elif c.isupper():
            up = True
        elif c.isdigit():
            dg = True
        else:
            ot = True
    return lo + up + dg + ot


def _plausible_secret_value(v: str) -> bool:
    low = v.lower()
    if low.startswith(_PATHLIKE_PREFIXES):
        return False
    if any(w in low for w in _PLACEHOLDER_WORDS):
        return False
    if len(set(v)) <= 2:            # "********", "aaaaaaaa", "xxxxxxxx"
        return False
    if _DOTTED_IDENT.fullmatch(v):  # os.environ, self.token, settings.SECRET
        return False
    if v.lower() in ("true", "false", "null", "none", "undefined"):
        return False
    return _char_classes(v) >= 2


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _card_brand_ok(d: str) -> bool:
    n = len(d)
    if d[0] == "4":
        return n in (13, 16, 19)                                   # Visa
    p2, p4 = int(d[:2]), int(d[:4])
    if 51 <= p2 <= 55 or 2221 <= p4 <= 2720:
        return n == 16                                             # Mastercard
    if p2 in (34, 37):
        return n == 15                                             # Amex
    if d.startswith("6011") or p2 == 65 or 644 <= int(d[:3]) <= 649:
        return 16 <= n <= 19                                       # Discover
    if p2 in (36, 38, 39) or 300 <= int(d[:3]) <= 305:
        return 14 <= n <= 19                                       # Diners
    if p2 == 35:
        return 16 <= n <= 19                                       # JCB
    if d[:2] in ("62", "81"):
        return 16 <= n <= 19                                       # UnionPay
    return False


def _credit_card_ok(v: str) -> bool:
    seps = {c for c in v if not c.isdigit()}
    if len(seps) > 1:
        return False
    d = "".join(c for c in v if c.isdigit())
    if not 13 <= len(d) <= 19 or len(set(d)) == 1:
        return False
    return _card_brand_ok(d) and _luhn_ok(d)


def _phone_ok(v: str) -> bool:
    n = sum(c.isdigit() for c in v)
    return 10 <= n <= 15


_FILE_EXTS = frozenset({"png", "jpg", "jpeg", "gif", "svg", "webp", "ico", "bmp", "js", "ts", "css",
                        "py", "json", "html", "md", "txt", "pdf", "zip", "gz", "tar", "map", "mjs"})


def _email_ok(v: str) -> bool:
    return v.rsplit(".", 1)[-1].lower() not in _FILE_EXTS


# --------------------------------------------------------------------------------------------
# patterns

@dataclass(frozen=True)
class _Pattern:
    kind: str
    regex: re.Pattern[str]
    group: int                       # capture group holding the secret (0 = whole match)
    priority: int                    # lower = more specific; wins overlaps
    literals: tuple[str, ...] = ()   # prefilter: at least one must occur (in text, or lowered text if lower_lits)
    lower_lits: bool = False
    validate: Callable[[str], bool] | None = None


_GENERIC_KW = (r"(?:password|passwd|passphrase|secret[_-]?key|secret|token|api[_-]?key|apikey|"
               r"access[_-]?key|private[_-]?key|credentials?)")

_PATTERNS: tuple[_Pattern, ...] = (
    _Pattern("private_key", re.compile(
        r"-----BEGIN (?:[A-Z0-9]+ ){0,3}PRIVATE KEY(?: BLOCK)?-----"
        r"(?:(?:[^-]|-(?!----)){0,20000}-----END (?:[A-Z0-9]+ ){0,3}PRIVATE KEY(?: BLOCK)?-----)?"),
        0, 0, ("PRIVATE KEY",)),
    _Pattern("anthropic_key", re.compile(r"(?<![A-Za-z0-9_-])sk-ant-[A-Za-z0-9_-]{20,}"),
             0, 1, ("sk-ant-",)),
    _Pattern("aws_access_key", re.compile(r"(?<![A-Za-z0-9])(?:AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}(?![A-Za-z0-9])"),
             0, 1, ("AKIA", "ASIA", "ABIA", "ACCA")),
    _Pattern("github_token", re.compile(
        r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{30,255})(?![A-Za-z0-9_])"),
        0, 1, ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_")),
    _Pattern("slack_token", re.compile(r"(?<![A-Za-z0-9_-])xox[baprs]-(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{10,255}"),
             0, 1, ("xox",)),
    _Pattern("stripe_key", re.compile(r"(?<![A-Za-z0-9_])(?:sk|rk)_live_[A-Za-z0-9]{16,255}"),
             0, 1, ("_live_",)),
    _Pattern("google_api_key", re.compile(r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])"),
             0, 1, ("AIza",)),
    _Pattern("jwt", re.compile(
        r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}(?![A-Za-z0-9_-])"),
        0, 2, ("eyJ",)),
    _Pattern("aws_secret_key", re.compile(
        r"(?i)(?:aws[A-Za-z0-9_.-]{0,20}?secret|secret[_.-]?access[_.-]?key)[A-Za-z0-9_.-]{0,20}"
        r"[\"']?\s*(?::=|=>|[:=])\s*[\"']?([A-Za-z0-9/+]{40})(?![A-Za-z0-9/+=])"),
        1, 2, ("secret",), True),
    _Pattern("openai_key", re.compile(
        r"(?<![A-Za-z0-9_-])sk-(?:(?:proj|svcacct|admin|None)-[A-Za-z0-9_-]{20,}"
        r"|(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]{32,})"),
        0, 3, ("sk-",)),
    # password = "hunter2!!"  /  "api_key": "..."  /  token := '...'
    _Pattern("generic_secret", re.compile(
        r"(?i)(?<![A-Za-z0-9])" + _GENERIC_KW +
        r"\\?[\"']?\s*(?::=|=>|==|[:=])\s*\\?([\"'`])([^\s\"'`\\]{8,256})\\?\1"),
        2, 5, ("pass", "secret", "token", "key", "credential"), True, _plausible_secret_value),
    # .env style: DB_PASSWORD=s3cr3tP4ss
    _Pattern("generic_secret", re.compile(
        r"(?im)^[ \t]{0,16}(?:export[ \t]+)?[A-Za-z0-9_]{0,48}" + _GENERIC_KW +
        r"[A-Za-z0-9_]{0,32}[ \t]{0,4}=[ \t]{0,4}([^\s\"'`#;,()\[\]{}]{8,256})[ \t]*$"),
        1, 5, ("pass", "secret", "token", "key", "credential"), True, _plausible_secret_value),
    # Authorization: Bearer <token>
    _Pattern("generic_secret", re.compile(r"(?i)(?<![A-Za-z0-9])bearer\s+([A-Za-z0-9._~+/-]{20,}=*)"),
             1, 5, ("bearer",), True, _plausible_secret_value),
    # ---- PII ----
    _Pattern("credit_card", re.compile(r"(?<![\d.-])\d(?:[ -]?\d){12,18}(?![\d-])"),
             0, 6, (), False, _credit_card_ok),
    _Pattern("email", re.compile(
        r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9][A-Za-z0-9._%+-]{0,63}@"
        r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24}(?![A-Za-z0-9-])"),
        0, 7, ("@",), False, _email_ok),
    _Pattern("phone", re.compile(
        r"(?<![\w.+-])(?:\+?1[ .-]?)?(?:\(\d{3}\)[ ]?|\d{3}[ .-])\d{3}[ .-]\d{4}(?![\w-])"),
        0, 8, (), False, _phone_ok),
    _Pattern("phone", re.compile(r"(?<![\w.+-])\+[1-9]\d{0,2}(?:[ .-]?\(?\d{1,4}\)?){2,5}(?![\w-])"),
             0, 8, ("+",), False, _phone_ok),
)

_DATA_URL = re.compile(r"data:[A-Za-z0-9.+/-]*(?:;[A-Za-z0-9=._-]+)*;base64,[A-Za-z0-9+/=]*")
_BASE64_BLOB = re.compile(r"[A-Za-z0-9+/=\r\n]+")
_BLOB_MIN = 1024


def _mask(kind: str, s: str) -> str:
    """Masked preview: keep a short prefix and the last 4 chars."""
    if kind == "private_key":
        header = s.split("-----", 2)[1] if s.count("-----") >= 2 else "BEGIN PRIVATE KEY"
        return f"-----{header}-----…"
    n = len(s)
    if n >= 24:
        return f"{s[:6]}…{s[-4:]}"
    if n >= 16:
        return f"{s[:4]}…{s[-4:]}"
    if n >= 12:
        return f"{s[:2]}…{s[-4:]}"
    if n >= 8:
        return f"{s[:2]}…{s[-2:]}"
    return "…"


def _is_base64_blob(text: str) -> bool:
    return len(text) >= _BLOB_MIN and _BASE64_BLOB.fullmatch(text) is not None


def scan_text(text: str, pii: bool = False, path: str = "") -> list[Finding]:
    """Find secrets (and PII if `pii`) in `text`. Overlaps resolve to the most specific kind."""
    if not text or len(text) < 8 or _is_base64_blob(text):
        return []
    lowered: str | None = None
    excluded: list[tuple[int, int]] = []
    if ";base64," in text:
        excluded = [(m.start(), m.end()) for m in _DATA_URL.finditer(text)]
        if excluded and excluded[0] == (0, len(text)):
            return []

    cands: list[tuple[int, int, int, str]] = []  # (priority, start, end, kind)
    for p in _PATTERNS:
        if not pii and p.kind in PII_KINDS:
            continue
        if p.literals:
            if p.lower_lits:
                if lowered is None:
                    lowered = text.lower()
                hay = lowered
            else:
                hay = text
            if not any(lit in hay for lit in p.literals):
                continue
        for m in p.regex.finditer(text):
            s, e = m.span(p.group)
            if s < 0 or e <= s:
                continue
            if p.validate is not None and not p.validate(m.group(p.group)):
                continue
            cands.append((p.priority, s, e, p.kind))
    if not cands:
        return []

    # Greedy selection: most specific kind first, then longest span; reject overlaps.
    cands.sort(key=lambda c: (c[0], -(c[2] - c[1]), c[1]))
    starts: list[int] = []
    chosen: list[tuple[int, int, str]] = []
    for ex_s, ex_e in excluded:
        bisect.insort(starts, ex_s)
        chosen.insert(bisect.bisect_left(starts, ex_s), (ex_s, ex_e, ""))
    for _, s, e, kind in cands:
        i = bisect.bisect_left(starts, s)
        if i > 0 and chosen[i - 1][1] > s:
            continue
        if i < len(starts) and starts[i] < e:
            continue
        starts.insert(i, s)
        chosen.insert(i, (s, e, kind))
    return [Finding(kind, path, _mask(kind, text[s:e]), s, e) for s, e, kind in chosen if kind]


# --------------------------------------------------------------------------------------------
# object walking

def _child_path(path: str, key: Any) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}.{key}" if path else str(key)


def _skip_key(d: dict[str, Any], key: str, parent_key: Any) -> bool:
    """True for opaque binary payloads that must not be scanned (images, audio, signatures)."""
    t = d.get("type")
    if key == "data" and (t == "base64" or parent_key == "input_audio" or t == "redacted_thinking"):
        return True
    if key == "signature" and t in ("thinking", "redacted_thinking"):
        return True
    return False


def _walk(obj: Any, pii: bool, path: str, parent_key: Any, out: list[Finding],
          redact: bool) -> Any:
    if isinstance(obj, str):
        found = scan_text(obj, pii, path)
        if not found:
            return obj
        out.extend(found)
        if not redact:
            return obj
        return _redact_text(obj, found)
    if isinstance(obj, dict):
        res: dict[Any, Any] = {}
        for k, v in obj.items():
            if isinstance(v, str) and _skip_key(obj, k, parent_key):
                res[k] = v
            else:
                res[k] = _walk(v, pii, _child_path(path, k), k, out, redact)
        return res if redact else obj
    if isinstance(obj, (list, tuple)):
        items = [_walk(v, pii, _child_path(path, i), parent_key, out, redact) for i, v in enumerate(obj)]
        return (type(obj)(items) if isinstance(obj, tuple) else items) if redact else obj
    return obj


def _redact_text(text: str, findings: list[Finding]) -> str:
    parts: list[str] = []
    pos = 0
    for f in sorted(findings, key=lambda f: f.start):
        parts.append(text[pos:f.start])
        parts.append(f"[REDACTED:{f.kind}]")
        pos = f.end
    parts.append(text[pos:])
    return "".join(parts)


def scan_obj(obj: Any, pii: bool = False) -> list[Finding]:
    """Walk dicts/lists and scan every string value. Paths look like "messages[0].content"."""
    out: list[Finding] = []
    _walk(obj, pii, "", None, out, False)
    return out


def redact_obj(obj: Any, pii: bool = False) -> tuple[Any, list[Finding]]:
    """Return a NEW object with each match replaced by "[REDACTED:<kind>]", plus the findings."""
    out: list[Finding] = []
    new = _walk(obj, pii, "", None, out, True)
    return new, out


def apply_guard(body: bytes, mode: str, pii: bool = False) -> tuple[bytes, list[Finding], bool]:
    """Apply guard `mode` ("off" | "warn" | "redact" | "block") to a request body.

    Returns (body_to_send, findings, blocked). Unknown modes behave like "warn".
    """
    if mode == "off" or not body:
        return body, [], False
    text = body.decode("utf-8", errors="surrogateescape")
    obj: Any = None
    is_json = False
    try:
        obj = json.loads(text)
        is_json = True
    except ValueError:
        pass

    if mode == "redact":
        if is_json:
            new, findings = redact_obj(obj, pii)
            if not findings:
                return body, [], False
            return json.dumps(new, ensure_ascii=False).encode("utf-8", errors="surrogateescape"), findings, False
        findings = scan_text(text, pii)
        if not findings:
            return body, [], False
        return _redact_text(text, findings).encode("utf-8", errors="surrogateescape"), findings, False

    findings = scan_obj(obj, pii) if is_json else scan_text(text, pii)
    blocked = mode == "block" and bool(findings)
    return body, findings, blocked
