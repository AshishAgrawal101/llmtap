# Running an agent where llmtap is the only way out

llmtap checks what goes to the model: [stored keys](keys.md), [spend caps](../README.md#-budgets--loop-breaker) and
[what may leave](what-may-leave.md). It can't see anything else. An agent that can run `curl` can send your data
anywhere, and an agent running as you can read your keychain.

The fix is to run the agent in a box where:

1. its only network access is llmtap on `127.0.0.1:4141`, and
2. it can't read your secrets (keychain, `~/.llmtap`, other projects).

Then every byte that leaves goes through llmtap's checks, and there is no real key inside the box to steal.

> **Status:** these recipes are written from each platform's documentation and have **not yet been tested end to end
> by the maintainer**. If you try one, please report back in
> [the tracking issue](https://github.com/AshishAgrawal101/llmtap/issues), whether it works or not.

## Start llmtap outside the box

```bash
llmtap keys add anthropic            # the real key stays in your OS keychain, outside the box
llmtap serve --keys-only --no-dashboard-writes
```

`--keys-only` refuses any real key coming from inside the box, so the agent can't use its own account to send data
out. `--no-dashboard-writes` stops it from using the dashboard's API (on the same port) to add a key, loosen a key's
limits or wipe the history.

Inside the box, the agent only needs the placeholder and the base URL (print them with `llmtap env`):

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:4141
ANTHROPIC_API_KEY=llmtap-key-anthropic
```

Stored keys are only used for clients on the same computer (loopback), so the box must share the host's loopback
interface. That rules out plain Docker for now: see [Docker](#docker) below.

## Linux: a separate user plus a firewall rule

Run the agent as its own user. It can't read your files, your keychain (Secret Service) or `~/.llmtap`, and an
`nftables` rule limits that user's traffic to llmtap's port:

```bash
sudo useradd --create-home agent
sudo nft add table inet llmtap_box
sudo nft add chain inet llmtap_box out '{ type filter hook output priority 0 ; }'
sudo nft add rule inet llmtap_box out meta skuid agent oif lo tcp dport 4141 accept
sudo nft add rule inet llmtap_box out meta skuid agent drop
```

Give the `agent` user access to the project only (for example with a group, or a copy of the repository), then:

```bash
sudo -u agent env ANTHROPIC_BASE_URL=http://127.0.0.1:4141 ANTHROPIC_API_KEY=llmtap-key-anthropic \
  bash -lc 'cd /path/to/project && claude'
```

Notes:

- The rule blocks DNS too, which is what you want: the agent doesn't need it to reach `127.0.0.1`.
- Package installs (`npm install`, `pip install`) won't work inside the box. Install dependencies first, as your user.
- Remove the rules with `sudo nft delete table inet llmtap_box`.

## macOS: `sandbox-exec`

`sandbox-exec` is deprecated by Apple but still ships with macOS. This profile allows everything except network
traffic other than llmtap's port, and reading llmtap's files and the keychains:

```scheme
; llmtap-box.sb
(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:4141"))
(allow network-outbound (remote unix-socket))
(deny file-read* (subpath (string-append (param "HOME") "/.llmtap")))
(deny file-read* (subpath (string-append (param "HOME") "/Library/Keychains")))
```

```bash
sandbox-exec -D HOME="$HOME" -f llmtap-box.sb env ANTHROPIC_BASE_URL=http://127.0.0.1:4141 \
  ANTHROPIC_API_KEY=llmtap-key-anthropic claude
```

The keychain can also be reached through system services rather than files, so treat the file rules as a second
line of defence and prefer the separate-user approach (a second macOS account) when it matters.

## Windows

Windows has no simple built-in per-process network filter. The practical options are a separate local account with
Windows Defender Firewall rules scoped to that account's programs, or running both llmtap and the agent inside WSL
with the Linux recipe above. Help writing and testing a Windows recipe is very welcome.

## Docker

A container has its own loopback interface, so it reaches llmtap through the Docker bridge, and llmtap refuses
stored keys for non-local clients. Supporting this safely needs a per-client proxy token (so a container proves it
may use a key without being on loopback). That is planned; until then, use the Linux or macOS recipes for agents
that should use stored keys.

## What the box doesn't fix

- The agent can still put secrets it can read **inside the box** into prompts. The
  [leak tripwire](what-may-leave.md#leak-tripwire) blocks the ones it knows about; keep real secrets out of the box.
- Provider-side tools (web fetch, MCP connectors) send data from the provider's servers. They are
  [refused for stored keys](what-may-leave.md#provider-side-tools) unless you allow them.
- Anything you allow through (for example an MCP server on another port) is a way out llmtap doesn't check.
