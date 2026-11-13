# Security Policy

## Supported versions

llmtap is pre-1.0. Security fixes go into the latest release only.

| Version | Supported |
|---|---|
| 0.1.x | ✅ |

## Reporting a vulnerability

**Please don't open a public issue for security problems.** Report them privately through
[GitHub Security Advisories](https://github.com/AshishAgrawal101/llmtap/security/advisories/new) ("Report a vulnerability").

Please include the affected version, a description, reproduction steps and the impact you see. We aim to acknowledge
reports within 3 working days and to ship a fix or mitigation for confirmed issues as quickly as we can. We're happy to
credit you in the release notes.

## Security model

llmtap is a **local developer tool**. It's worth knowing what it does and doesn't protect:

- **Bind address.** By default llmtap listens on `127.0.0.1` only. The dashboard and its JSON API (`/_llmtap/`)
  **have no authentication** and expose full prompts and responses. Don't use `--host 0.0.0.0` on an untrusted network.
- **API keys.** Request headers (including `Authorization` / `x-api-key`) are forwarded to the upstream you configured
  and are **not** written to the history database or cassettes.
- **Stored data.** Prompts and responses are stored in plaintext in SQLite (`~/.llmtap/llmtap.db` by default). Treat it
  like a log file. `llmtap clear` deletes the history, and `--db :memory:` avoids writing it at all.
- **Cassettes** contain prompts, responses and response headers, and are designed to be committed. Review them and run
  `llmtap scan <cassette>` before committing to a public repository.
- **Guard.** The secret guard is a regex-based safety net. It reduces the chance of leaking credentials into prompts,
  but it isn't a guarantee or a substitute for proper secret management. In `warn` mode the original (unredacted)
  prompt is stored locally. Use `redact` or `block` if that matters to you.
- **Upstream TLS.** HTTPS upstreams are verified with the system trust store (`ssl.create_default_context()`).
  Traffic between your app and llmtap is plain HTTP over loopback.
- **Custom upstreams.** Anything in `llmtap.toml` `[upstreams.*]` receives your requests *and API keys*. Only add
  upstreams you trust, and be careful with `llmtap.toml` files from repositories you didn't write.

In scope: anything that leaks keys or prompts beyond what's described above, lets a remote party reach the proxy or
dashboard under the default configuration, lets a crafted upstream response execute code or script in the dashboard
(XSS), or bypasses guard `block` mode for a supported secret type.
