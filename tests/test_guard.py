"""Tests for llmtap.guard — secret/PII firewall."""
from __future__ import annotations

import base64
import json
import random
import time
import uuid

import pytest

from llmtap.guard import KINDS, Finding, apply_guard, redact_obj, scan_obj, scan_text

# Fake (but well-formed) secrets.
ANTHROPIC = "sk-ant-api03-" + "aB3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1xY3zA5bC7dE9fG1hJ3kL5mN7pQ9rS1tU3vW5xY7zA9bC1d-EfGhIjAA"
OPENAI_PROJ = "sk-proj-" + "Ab12Cd34Ef56Gh78Ij90Kl12Mn34Op56Qr78St90Uv12Wx34Yz56_-aBcD"
OPENAI_LEGACY = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8s9T0u1V2w3X4"
AWS_ID = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
GITHUB = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
GITHUB_PAT = "github_pat_" + "11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ0123456789abcdefghij"
SLACK = "xoxb-" + "123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"
STRIPE = "sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc"
GOOGLE = "AIza" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q"
JWT = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
       "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ."
       "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c")
PEM = ("-----BEGIN RSA PRIVATE KEY-----\n"
       "MIIEowIBAAKCAQEA1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y\n"
       "z5A6B7C8D9E0F1G2H3I4J5K6L7M8N9O0P1Q2R3S4T5U6V7W8X9Y0Z1a2b3c4d5e6\n"
       "-----END RSA PRIVATE KEY-----")


def kinds(text: str, pii: bool = False) -> list[str]:
    return [f.kind for f in scan_text(text, pii=pii)]


# ------------------------------------------------------------------------------------ basics

def test_kinds_registry() -> None:
    expected = {"aws_access_key", "aws_secret_key", "anthropic_key", "openai_key", "github_token",
                "slack_token", "stripe_key", "google_api_key", "private_key", "jwt", "generic_secret",
                "email", "phone", "credit_card"}
    assert expected <= set(KINDS)
    assert all(isinstance(v, str) and v for v in KINDS.values())


def test_finding_to_dict() -> None:
    f = Finding(kind="openai_key", path="a.b", preview="sk-…abcd", start=1, end=9)
    assert f.to_dict() == {"kind": "openai_key", "path": "a.b", "preview": "sk-…abcd", "start": 1, "end": 9}
    json.dumps(f.to_dict())


def test_empty_and_short() -> None:
    assert scan_text("") == []
    assert scan_text("hi") == []
    assert scan_obj(None) == []
    assert scan_obj({"a": 1, "b": [True, 2.5, None]}) == []


# ------------------------------------------------------------------------------------ positives

@pytest.mark.parametrize("secret,kind", [
    (ANTHROPIC, "anthropic_key"),
    ("sk-ant-admin01-" + "Zx9Yw8Vu7Ts6Rq5Po4Nm3Lk2", "anthropic_key"),
    (OPENAI_PROJ, "openai_key"),
    (OPENAI_LEGACY, "openai_key"),
    ("sk-svcacct-" + "Q1w2E3r4T5y6U7i8O9p0A1s2D3f4", "openai_key"),
    (AWS_ID, "aws_access_key"),
    ("ASIA" + "Y34FZKBOKMUTVV7A", "aws_access_key"),
    (GITHUB, "github_token"),
    ("gho_" + "16C7e42F292c6912E7710c838347Ae178B4a", "github_token"),
    ("ghs_" + "16C7e42F292c6912E7710c838347Ae178B4a", "github_token"),
    (GITHUB_PAT, "github_token"),
    (SLACK, "slack_token"),
    ("xoxp-" + "1234567890-0987654321-abcdef", "slack_token"),
    (STRIPE, "stripe_key"),
    ("rk_live_" + "51H8xYzAbCdEfGhIjKlMnOp", "stripe_key"),
    (GOOGLE, "google_api_key"),
    (JWT, "jwt"),
])
def test_token_kinds_detected(secret: str, kind: str) -> None:
    text = f"here is my key: {secret} please don't share"
    found = scan_text(text)
    assert [f.kind for f in found] == [kind]
    f = found[0]
    assert text[f.start:f.end] == secret
    assert f.path == ""
    # preview is masked: short prefix + … + last 4
    assert "…" in f.preview
    assert f.preview.endswith(secret[-4:])
    assert secret not in f.preview
    assert len(f.preview) <= 11


def test_anthropic_preview_format() -> None:
    f = scan_text(ANTHROPIC)[0]
    assert f.preview == "sk-ant…" + ANTHROPIC[-4:]


def test_private_key() -> None:
    text = f"config:\n{PEM}\nthanks"
    found = scan_text(text)
    assert [f.kind for f in found] == ["private_key"]
    f = found[0]
    assert text[f.start:f.end] == PEM
    assert "MIIE" not in f.preview
    assert f.preview.startswith("-----BEGIN RSA PRIVATE KEY")
    for header in ("-----BEGIN PRIVATE KEY-----", "-----BEGIN OPENSSH PRIVATE KEY-----",
                   "-----BEGIN EC PRIVATE KEY-----", "-----BEGIN ENCRYPTED PRIVATE KEY-----"):
        assert kinds(header + "\nabc\n") == ["private_key"]


def test_private_key_redacted_whole_block() -> None:
    new, found = redact_obj({"t": f"a {PEM} b"})
    assert new == {"t": "a [REDACTED:private_key] b"}


def test_public_key_not_flagged() -> None:
    assert kinds("-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA\n-----END PUBLIC KEY-----") == []
    assert kinds("-----BEGIN CERTIFICATE-----\nMIIDdzCCAl+gAwIBAgIEAgAAuTANBgkqhkiG\n-----END CERTIFICATE-----") == []


@pytest.mark.parametrize("text", [
    f"aws_secret_access_key = {AWS_SECRET}",
    f"AWS_SECRET_ACCESS_KEY={AWS_SECRET}",
    f'"aws_secret_access_key": "{AWS_SECRET}"',
    f"aws_secret_key: '{AWS_SECRET}'",
    f"SecretAccessKey: {AWS_SECRET}",
])
def test_aws_secret_contextual(text: str) -> None:
    found = scan_text(text)
    assert [f.kind for f in found] == ["aws_secret_key"]
    assert text[found[0].start:found[0].end] == AWS_SECRET


def test_aws_secret_requires_context() -> None:
    assert kinds(f"the value {AWS_SECRET} alone") == []
    # a 40-hex git sha next to the word 'commit' is not an aws secret
    assert kinds("commit = 9fceb02d0ae598e95dc970b74767f19372d61af8") == []


def test_aws_pair_in_credentials_file() -> None:
    text = f"[default]\naws_access_key_id = {AWS_ID}\naws_secret_access_key = {AWS_SECRET}\n"
    assert kinds(text) == ["aws_access_key", "aws_secret_key"]


@pytest.mark.parametrize("text,value", [
    ('password = "hunter2hunter"', "hunter2hunter"),
    ("db_password: 'S3cure!Passw0rd'", "S3cure!Passw0rd"),
    ('{"api_key": "a8f5f167f44f4964e6c998dee827110c"}', "a8f5f167f44f4964e6c998dee827110c"),
    ('client_secret = "Zm9vYmFyYmF6cXV4"', "Zm9vYmFyYmF6cXV4"),
    ("const TOKEN = `abc123XYZ789def`", "abc123XYZ789def"),
    ('secret := "q1w2e3r4t5y6"', "q1w2e3r4t5y6"),
    ("DATABASE_PASSWORD=s3cr3tP4ss!", "s3cr3tP4ss!"),
    ("export SECRET_KEY_BASE=8f14e45fceea167a5a36dedd4bea2543", "8f14e45fceea167a5a36dedd4bea2543"),
    ("Authorization: Bearer abcDEF123456ghiJKL789mno", "abcDEF123456ghiJKL789mno"),
])
def test_generic_secret(text: str, value: str) -> None:
    found = scan_text(text)
    assert [f.kind for f in found] == ["generic_secret"]
    f = found[0]
    assert text[f.start:f.end] == value


def test_generic_secret_redacts_only_value() -> None:
    new, found = redact_obj('password = "hunter2hunter" # todo')
    assert new == 'password = "[REDACTED:generic_secret]" # todo'
    assert len(found) == 1


@pytest.mark.parametrize("text", [
    'password = "short1"',                     # < 8 chars
    'password = "********"',                    # mask
    'password = "<your-password-here>"',        # placeholder
    'api_key = "YOUR_API_KEY_HERE"',            # placeholder
    'token = "${GITHUB_TOKEN}"',                # template
    'password = "{{ db_password }}"',
    "token = os.environ.get('TOKEN')",          # code
    "token=self.token_value",
    'password: "required"',                     # schema word
    '"max_tokens": 1024',
    "token_count = 12345678",
    "The secret to good bread is patience, not a password manager.",
    "Please enter your password: it must be at least 8 characters.",
    'password = getpass.getpass("Password: ")',
    "PWD=/home/user/projects/llmtap",
    "Bearer tokens are sent in the Authorization header.",
])
def test_generic_secret_negatives(text: str) -> None:
    assert kinds(text) == [], text


# ------------------------------------------------------------------------------------ overlaps

def test_overlap_prefers_specific_kind() -> None:
    # sk-ant- key also looks like an openai sk- key
    assert kinds(ANTHROPIC) == ["anthropic_key"]
    # generic assignment wrapping a specific key → specific kind wins
    for secret, kind in [(ANTHROPIC, "anthropic_key"), (OPENAI_PROJ, "openai_key"), (STRIPE, "stripe_key"),
                         (GITHUB, "github_token")]:
        assert kinds(f'api_key = "{secret}"') == [kind]
        assert kinds(f"OPENAI_API_KEY={secret}") == [kind]
    assert kinds(f"Authorization: Bearer {JWT}") == ["jwt"]


def test_multiple_findings_sorted() -> None:
    text = f"{GITHUB} and {AWS_ID} and {OPENAI_LEGACY}"
    found = scan_text(text)
    assert [f.kind for f in found] == ["github_token", "aws_access_key", "openai_key"]
    assert [f.start for f in found] == sorted(f.start for f in found)


# ------------------------------------------------------------------------------------ negatives

@pytest.mark.parametrize("text", [
    "I use sk-learn and scikit-learn for machine learning tasks every day.",
    "the task-list and disk-usage commands; also risk-assessment-framework-2024-q3",
    "sk-2024-roadmap-planning-document-v2",
    "ask-anything mask-ant-colony desk-ant-1234567890abcdefghijklmn",
    "Here is a UUID: 123e4567-e89b-12d3-a456-426614174000 and " + str(uuid.uuid4()),
    "commit 9fceb02d0ae598e95dc970b74767f19372d61af8 fixed it; sha256 "
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "AKIA is a prefix but AKIASHORT is not a key, nor is AKIAIOSFODNN7EXAMPLEEXTRA",
    "The ASIA PACIFIC region grew 12% while EMEA fell.",
    "xoxo hugs and kisses, xox-b",
    "eyJ is how base64 JSON starts, e.g. eyJhbGci but that's not a full JWT.",
    "version 1.2.3.4567 at 2024-01-15T10:30:00Z, port 8080, pid 12345",
    "def get_api_key(self) -> str:\n    return self._api_key\n",
    "const token = await getToken();\nif (password.length < 8) throw new Error('too short');",
    "ghp_short and gho_ and github_pat_ are prefixes only",
    "sk_test_example is a test key",
])
def test_no_false_positives(text: str) -> None:
    assert kinds(text) == [], text


PROSE_CORPUS = """
The quick brown fox jumps over the lazy dog. In 2023, revenue grew 14.5% to $3,200,000.
Our API returns a token count with every response; the secret sauce is caching.
To reset your password, click "Forgot password" on the login page.

```python
import os, uuid, hashlib
def handler(event, context):
    api_key = os.environ["OPENAI_API_KEY"]
    request_id = str(uuid.uuid4())
    digest = hashlib.sha256(b"hello").hexdigest()  # 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    return {"statusCode": 200, "body": json.dumps({"id": request_id, "tokens": 1234})}
```

```js
const config = { apiKey: process.env.API_KEY, timeout: 30000, retries: 3 };
fetch(`${base}/v1/messages`, { headers: { "x-api-key": config.apiKey } });
```

SELECT id, email_verified, created_at FROM users WHERE id = 42 ORDER BY created_at DESC LIMIT 10;
git log --oneline: a1b2c3d Fix bug; 9fceb02 Add feature; merge 4b825dc642cb6eb9a060e54bf8d69288fbee4904
Docker image sha256:4c8a9f3e2b1d0e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f
Kubernetes pod my-app-7d4b9c8f5-x2k9p restarted 3 times. Build #48213 passed in 4m12s.
"""


def test_prose_and_code_corpus_clean() -> None:
    assert scan_text(PROSE_CORPUS) == []
    assert scan_text(PROSE_CORPUS, pii=True) == []


# ------------------------------------------------------------------------------------ PII

def test_pii_off_by_default() -> None:
    text = "mail jane.doe@example.com or call (555) 123-4567, card 4111 1111 1111 1111"
    assert scan_text(text) == []
    assert sorted(kinds(text, pii=True)) == ["credit_card", "email", "phone"]


@pytest.mark.parametrize("email", ["jane.doe@example.com", "a+tag@sub.domain.co.uk", "X_Y-z@corp.io"])
def test_email_positive(email: str) -> None:
    found = scan_text(f"contact <{email}> today", pii=True)
    assert [f.kind for f in found] == ["email"]


@pytest.mark.parametrize("text", ["@property decorator", "npm i @types/node", "logo@2x.png", "user@localhost",
                                  "email me at"])
def test_email_negative(text: str) -> None:
    assert kinds(text, pii=True) == []


@pytest.mark.parametrize("phone", ["(555) 123-4567", "555-123-4567", "555.123.4567", "+1 555 123 4567",
                                   "+1-555-123-4567", "+44 20 7946 0958", "+49 30 1234 5678", "+33 1 23 45 67 89"])
def test_phone_positive(phone: str) -> None:
    found = scan_text(f"call me at {phone} tomorrow", pii=True)
    assert [f.kind for f in found] == ["phone"], phone
    assert found[0].end - found[0].start == len(phone)


@pytest.mark.parametrize("text", ["2024-01-15", "id 5551234567", "version v10.555.123.4567", "+1 2", "12:30:45",
                                  "192.168.100.200", "ISBN 978-3-16-148410-0"])
def test_phone_negative(text: str) -> None:
    assert kinds(text, pii=True) == []


@pytest.mark.parametrize("card", ["4111111111111111", "4111 1111 1111 1111", "4111-1111-1111-1111",
                                  "5500 0000 0000 0004", "378282246310005", "6011111111111117",
                                  "3530111333300000", "2223003122003222"])
def test_credit_card_positive(card: str) -> None:
    found = scan_text(f"card: {card} exp 12/29", pii=True)
    assert [f.kind for f in found] == ["credit_card"], card
    assert found[0].preview.endswith(card[-4:])


@pytest.mark.parametrize("text", [
    "4111111111111112",          # Luhn fails
    "4111 1111 1111 1112",
    "1234567890123456",          # no brand + Luhn fail
    "0000000000000000",
    "4111-1111 1111-1111",       # inconsistent separators
    "timestamp 1699999999999 ms",
    "order 79927398713",         # valid Luhn but too short / no brand
])
def test_credit_card_negative(text: str) -> None:
    assert "credit_card" not in kinds(text, pii=True)


def test_luhn_validation() -> None:
    assert kinds("4012888888881881", pii=True) == ["credit_card"]
    assert kinds("4012888888881882", pii=True) == []


# ------------------------------------------------------------------------------------ objects

def anthropic_body() -> dict:
    return {
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": [
                {"type": "text", "text": f"my key is {AWS_ID}, ok?"},
                {"type": "text", "text": "and ünïcödé ✓ stays"},
            ]},
        ],
        "metadata": {"user_id": "u-1"},
    }


def test_scan_obj_paths() -> None:
    found = scan_obj(anthropic_body())
    assert len(found) == 1
    f = found[0]
    assert f.kind == "aws_access_key"
    assert f.path == "messages[1].content[0].text"
    text = anthropic_body()["messages"][1]["content"][0]["text"]
    assert text[f.start:f.end] == AWS_ID


def test_scan_obj_path_shapes() -> None:
    assert scan_obj(GITHUB)[0].path == ""
    assert scan_obj([1, GITHUB])[0].path == "[1]"
    assert scan_obj({"a": {"b": [[0, {"c": GITHUB}]]}})[0].path == "a.b[0][1].c"
    assert scan_obj({"system": GITHUB})[0].path == "system"


def test_scan_obj_tool_arguments_string() -> None:
    body = {"messages": [{"role": "assistant", "tool_calls": [
        {"function": {"name": "f", "arguments": json.dumps({"password": "Tr0ub4dor&3x"})}}]}]}
    found = scan_obj(body)
    assert [(f.kind, f.path) for f in found] == [("generic_secret", "messages[0].tool_calls[0].function.arguments")]


def test_redact_obj_returns_new_object() -> None:
    body = anthropic_body()
    original = json.loads(json.dumps(body))
    new, found = redact_obj(body)
    assert body == original  # input untouched
    assert new is not body
    assert len(found) == 1
    assert new["messages"][1]["content"][0]["text"] == "my key is [REDACTED:aws_access_key], ok?"
    # everything else preserved
    new["messages"][1]["content"][0]["text"] = original["messages"][1]["content"][0]["text"]
    assert new == original


def test_redact_obj_multiple_in_one_string() -> None:
    new, found = redact_obj({"x": f"a {GITHUB} b {STRIPE} c"})
    assert new == {"x": "a [REDACTED:github_token] b [REDACTED:stripe_key] c"}
    assert len(found) == 2


def test_redact_no_findings_same_value() -> None:
    body = {"a": ["x", {"b": 1}]}
    new, found = redact_obj(body)
    assert found == [] and new == body


# ------------------------------------------------------------------------------------ base64 media

def _random_b64(n: int, seed: int = 0) -> str:
    rnd = random.Random(seed)
    return base64.b64encode(bytes(rnd.getrandbits(8) for _ in range(n))).decode()


def test_base64_image_block_skipped() -> None:
    # A base64 payload that (by construction) contains secret-looking substrings must not be flagged.
    payload = _random_b64(3000) + "AKIAIOSFODNN7EXAMPLE" + "/" + GOOGLE.replace("-", "A") + _random_b64(3000, 1)
    body = {"messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": payload}},
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "AKIAIOSFODNN7EXAMPLE"}},
        {"type": "text", "text": "what is this?"},
    ]}]}
    assert scan_obj(body) == []
    new, found = redact_obj(body)
    assert found == [] and new == body


def test_data_url_skipped() -> None:
    url = "data:image/png;base64," + _random_b64(2000) + "AKIAIOSFODNN7EXAMPLE" + _random_b64(100, 2)
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "input_audio", "input_audio": {"data": "AKIAIOSFODNN7EXAMPLE", "format": "wav"}},
    ]}]}
    assert scan_obj(body) == []
    # data URL embedded in raw text is skipped, but a real secret next to it is still caught
    text = f"img {url} and key {GITHUB}"
    assert kinds(text) == ["github_token"]


def test_large_plain_base64_blob_skipped() -> None:
    blob = _random_b64(5000) + "AKIAIOSFODNN7EXAMPLE" + _random_b64(5000, 3)
    assert scan_text(blob) == []


def test_thinking_signature_skipped() -> None:
    body = {"messages": [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "hmm", "signature": "AKIAIOSFODNN7EXAMPLE"},
        {"type": "redacted_thinking", "data": "AKIAIOSFODNN7EXAMPLE"}]}]}
    assert scan_obj(body) == []


# ------------------------------------------------------------------------------------ apply_guard

def _body(obj: object) -> bytes:
    return json.dumps(obj).encode()


def test_apply_guard_off() -> None:
    b = _body(anthropic_body())
    assert apply_guard(b, "off") == (b, [], False)


def test_apply_guard_warn() -> None:
    b = _body(anthropic_body())
    out, found, blocked = apply_guard(b, "warn")
    assert out is b and blocked is False
    assert [f.path for f in found] == ["messages[1].content[0].text"]


def test_apply_guard_block() -> None:
    b = _body(anthropic_body())
    out, found, blocked = apply_guard(b, "block")
    assert blocked is True and len(found) == 1 and out == b
    clean = _body({"messages": [{"role": "user", "content": "hi"}]})
    assert apply_guard(clean, "block") == (clean, [], False)


def test_apply_guard_redact_json() -> None:
    b = _body(anthropic_body())
    out, found, blocked = apply_guard(b, "redact")
    assert blocked is False and len(found) == 1
    text = out.decode("utf-8")
    assert AWS_ID not in text
    assert "ünïcödé ✓" in text  # ensure_ascii=False
    obj = json.loads(text)
    expected = anthropic_body()
    expected["messages"][1]["content"][0]["text"] = "my key is [REDACTED:aws_access_key], ok?"
    assert obj == expected
    assert list(obj) == list(expected)  # key order preserved


def test_apply_guard_redact_no_findings_unchanged() -> None:
    b = b'{"messages":  [ {"role":"user","content":"hi"} ] }'
    assert apply_guard(b, "redact") == (b, [], False)


def test_apply_guard_non_json() -> None:
    b = f"plain text body with {OPENAI_LEGACY} inside".encode()
    out, found, blocked = apply_guard(b, "warn")
    assert out == b and [f.kind for f in found] == ["openai_key"] and found[0].path == ""
    out, found, blocked = apply_guard(b, "redact")
    assert out == b"plain text body with [REDACTED:openai_key] inside"
    out, found, blocked = apply_guard(b, "block")
    assert blocked is True


def test_apply_guard_empty_and_invalid_utf8() -> None:
    assert apply_guard(b"", "block") == (b"", [], False)
    raw = b"\xff\xfe binary " + GITHUB.encode() + b" \x80"
    out, found, blocked = apply_guard(raw, "redact")
    assert [f.kind for f in found] == ["github_token"]
    assert out.startswith(b"\xff\xfe binary [REDACTED:github_token]") and out.endswith(b"\x80")


def test_apply_guard_pii_flag() -> None:
    b = _body({"messages": [{"role": "user", "content": "email bob@example.org"}]})
    assert apply_guard(b, "block")[2] is False
    out, found, blocked = apply_guard(b, "block", pii=True)
    assert blocked is True and found[0].kind == "email"
    out, found, _ = apply_guard(b, "redact", pii=True)
    assert json.loads(out)["messages"][0]["content"] == "email [REDACTED:email]"


def test_findings_serializable() -> None:
    _, found, _ = apply_guard(_body({"a": f"{ANTHROPIC} {JWT}"}), "warn")
    json.dumps([f.to_dict() for f in found])
    assert {f.kind for f in found} == {"anthropic_key", "jwt"}


# ------------------------------------------------------------------------------------ performance

def test_performance_1mb_prompt() -> None:
    para = PROSE_CORPUS + " Contact support for help. The token budget is 4096 per request.\n"
    text = (para * (1_048_576 // len(para) + 1))[:1_048_576]
    text = text[:500_000] + f" leaked {AWS_ID} here " + text[500_000:]
    scan_text(text[:1000])  # warm up
    t0 = time.perf_counter()
    found = scan_text(text, pii=True)
    elapsed = time.perf_counter() - t0
    assert [f.kind for f in found] == ["aws_access_key"]
    assert elapsed < 0.5, f"scan took {elapsed:.3f}s"


def test_performance_1mb_json_body() -> None:
    msgs = [{"role": "user" if i % 2 else "assistant", "content": PROSE_CORPUS} for i in range(1_048_576 // len(PROSE_CORPUS))]
    body = json.dumps({"model": "gpt-4o", "messages": msgs}).encode()
    t0 = time.perf_counter()
    _, found, _ = apply_guard(body, "redact")
    elapsed = time.perf_counter() - t0
    assert found == []
    assert elapsed < 0.5, f"apply_guard took {elapsed:.3f}s"


@pytest.mark.parametrize("text", [
    "-----BEGIN PRIVATE KEY-----" * 30000,            # many unterminated PEM headers
    "TOKEN_" * 170000 + "=x",                         # one huge env-style line
    "password: \"" * 90000,
    "a@" + "b." * 500000,
    "(555) " * 170000,
])
def test_performance_adversarial(text: str) -> None:
    t0 = time.perf_counter()
    scan_text(text, pii=True)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"scan took {elapsed:.3f}s"
