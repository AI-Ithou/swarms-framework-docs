# Offline DecisionModel documentation checks

These tests exercise the documentation's actual runnable scripts against the
Swarms implementation. All provider responses and agent replies in these tests
are synthetic fixtures. They are never presented as live example output.

Use an isolated environment and the inspected framework revision:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "swarms @ git+https://github.com/kyegomez/swarms.git@c5e71b85ddc98c60951039c716970a72f61ea6db" pytest pytest-socket
export SWARMS_TELEMETRY_ON=false
export LITELLM_LOCAL_MODEL_COST_MAP=True
python -m pytest --disable-socket --allow-unix-socket tests/decision_models -q
```

`--disable-socket` prevents accidental outbound connections. UNIX sockets remain
allowed for local event-loop plumbing. Set `WORKSPACE_DIR` to a temporary directory
if you do not want Swarms to create its local workspace under this checkout.
Environments using a SOCKS proxy also need `python -m pip install "httpx[socks]"`.

The checks cover:

- Exact parity between runnable files and guide snippets
- Executable Python syntax and the reference's display-only API signatures
- Unique navigation entries and internal page links, including heading anchors
- Triage and routing boundary policies, declined and escalated paths
- One batched request and selected-agent-only execution
- Custom request/response normalization, validation, URLs, and cleanup

They do not validate real TypeSafe, OpenRouter, or OpenAI behavior. Before claiming
live verification, run the three guide scripts with authorized provider credentials,
capture actual output, and record the model IDs. A real custom-provider integration
also needs a service implementing the illustrated envelope schema.
