"""Offline checks only: every numeric response below is a synthetic fixture.

Run with pytest-socket's --disable-socket --allow-unix-socket options.
No fixture is evidence of a live model response.
"""

import ast
import importlib.util
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from swarms import DecisionModel
from swarms.structs import decision_model

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "decision-models"
CODE = EXAMPLES / "code"
REFERENCE = ROOT / "api" / "decision-model.mdx"
PAGES = [REFERENCE, *EXAMPLES.glob("*.mdx")]


def reference_signature(code):
    """Parse the reference's display-only signatures without executing them."""
    source = code.strip()
    if source.startswith("DecisionModel(\n"):
        source = "def __init__(" + source[len("DecisionModel("):]
    elif not re.fullmatch(r"(?:async )?def \w+\([\s\S]*\) -> [^\n]+", source):
        return None
    return ast.parse(source + ":\n    ...").body[0]


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ticket = load_example("ticket_triage")
router = load_example("agent_router")
custom = load_example("custom_decision_model")


def synthetic_answers(confidence=0.9, urgency=0.9, frustration=1.0):
    return {
        "department": {
            "type": "choice", "choice": "technical",
            "confidence": confidence,
            "probabilities": {"technical": 0.9, "billing": 0.1, "sales": 0.0},
        },
        "frustration": {
            "type": "score", "score": frustration, "confidence": 0.9,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Angry"},
            "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0},
        },
        "is_urgent": {"type": "noul", "noul": urgency},
    }


def routing_answers(scope=0.9, confidence=0.9, choice="Technical-Agent"):
    return {
        "agent": {
            "type": "choice", "choice": choice, "confidence": confidence,
            "probabilities": {"Technical-Agent": 0.9, "Billing-Agent": 0.1},
        },
        "in_scope": {"type": "noul", "noul": scope},
    }


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch):
    for name in (
        "TYPESAFE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
        "CUSTOM_DECISION_API_KEY", "CLOUDFLARE_AUTH_TOKEN", "CLOUDFLARE_ACCOUNT_ID",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("confidence,urgency,frustration,action", [
    (0.59, 0.99, 2.0, "Send to a human"),
    (0.60, 0.81, 1.0, "Page the technical"),
    (0.90, 0.80, 1.0, "Queue for the technical"),
    (0.90, 0.99, 0.99, "Queue for the technical"),
    (0.90, 0.20, 0.0, "Queue for the technical"),
])
def test_triage_policy_boundaries(confidence, urgency, frustration, action):
    assert ticket.triage_action(
        synthetic_answers(confidence, urgency, frustration)
    ).startswith(action)


@pytest.mark.parametrize("scope,confidence,choice,action", [
    (0.49, 0.99, "Technical-Agent", "decline"),
    (0.49, 0.10, "Technical-Agent", "decline"),
    (0.50, 0.49, "Technical-Agent", "escalate"),
    (0.50, 0.50, "Technical-Agent", "run"),
    (0.90, 0.90, "Unknown-Agent", "escalate"),
])
def test_routing_policy_boundaries(scope, confidence, choice, action):
    assert router.choose_route(
        routing_answers(scope, confidence, choice), {"Technical-Agent"}
    )[0] == action


@pytest.mark.parametrize("scope,confidence,expected_calls", [
    (0.49, 0.90, []), (0.90, 0.49, []), (0.90, 0.90, ["Technical-Agent"]),
])
def test_real_client_routes_once_and_only_runs_selected_agent(
    scope, confidence, expected_calls, capsys,
):
    requests, calls = [], []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "model": "offline-fixture", "usage": {},
            "answers": routing_answers(scope, confidence),
        })

    def agent(name, description):
        return SimpleNamespace(
            agent_name=name, agent_description=description,
            run=lambda task: calls.append(name) or "offline agent fixture",
        )

    agents = {
        "Billing-Agent": agent("Billing-Agent", "Invoices and payments"),
        "Technical-Agent": agent("Technical-Agent", "Bugs and outages"),
    }
    model = DecisionModel(api_key="offline-fixture-not-a-credential")
    model._client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        router.route("Our webhook is down.", model, agents)
    finally:
        model.close()
    assert len(requests) == 1
    assert requests[0]["questions"]["agent"]["criteria"] == {
        name: agent.agent_description for name, agent in agents.items()
    }
    assert set(requests[0]["questions"]) == {"agent", "in_scope"}
    assert calls == expected_calls
    assert '"model": "offline-fixture"' in capsys.readouterr().out


def test_ticket_main_sends_three_questions_once_and_closes(monkeypatch, capsys):
    requests, clients = [], []
    real_client = httpx.Client

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "model": "offline-fixture", "usage": {}, "answers": synthetic_answers(),
        })

    def create_client(**kwargs):
        client = real_client(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-fixture-not-a-credential")
    monkeypatch.setattr(decision_model.httpx, "Client", create_client)
    ticket.main()
    assert len(requests) == 1
    assert {q["type"] for q in requests[0]["questions"].values()} == {
        "choice", "score", "noul",
    }
    assert clients[0].is_closed
    assert "Action: Page the technical" in capsys.readouterr().out


def test_custom_adapter_retains_extra_fields_and_normalizes_response():
    model = custom.EnvelopeDecisionModel(
        api_key="offline-fixture-not-a-credential", extra_body={"trace_label": "test"},
    )
    questions = {"urgent": {"type": "noul", "instructions": "Urgent?"}}
    payload = model.build_payload("Support question", questions)
    assert payload == {
        "model_id": "jev-latest", "context": "Support question",
        "prompts": questions, "trace_label": "test",
    }
    response = {
        "model_id": "offline-fixture",
        "decisions": {"urgent": {"type": "noul", "noul": 0.2}},
        "token_usage": {"input_tokens": 1, "output_tokens": 1},
    }
    assert model.parse_response(response, questions) == {
        "model": response["model_id"], "answers": response["decisions"],
        "usage": response["token_usage"],
    }
    model.close()


def test_custom_adapter_preserves_question_and_answer_validation():
    model = custom.EnvelopeDecisionModel(api_key="offline-fixture-not-a-credential")
    with pytest.raises(ValueError):
        model.build_payload("Support question", {"team": {"type": "choice"}})
    questions = {"urgent": {"type": "noul", "instructions": "Urgent?"}}
    for decisions in ({}, {"urgent": {"type": "choice"}}):
        with pytest.raises(RuntimeError):
            model.parse_response({
                "model_id": "offline-fixture", "decisions": decisions, "token_usage": {},
            }, questions)
    model.close()


@pytest.mark.parametrize("mode,url,model_id", [
    ("typesafe", "https://api.typesafe.ai/v1/systemone", "jev-1.13.0"),
    ("openrouter", "https://openrouter.ai/api/v1/systemone", "~typesafe/jev-latest"),
    ("custom", "https://provider.example/evaluate", "test-model"),
])
def test_custom_script_modes_use_expected_payload_and_close(
    mode, url, model_id, monkeypatch, capsys,
):
    requests, clients = [], []
    real_client = httpx.Client
    env = {
        "typesafe": "TYPESAFE_API_KEY", "openrouter": "OPENROUTER_API_KEY",
        "custom": "CUSTOM_DECISION_API_KEY",
    }
    monkeypatch.setenv(env[mode], "offline-fixture-not-a-credential")
    args = ["custom_decision_model.py", "--provider", mode]
    if mode == "custom":
        args += ["--base-url", "https://provider.example", "--model", model_id]
    monkeypatch.setattr("sys.argv", args)

    def respond(request):
        payload = json.loads(request.content)
        requests.append((str(request.url), payload))
        answers = {
            "team": {"type": "choice", "choice": "sales"},
            "urgent": {"type": "noul", "noul": 0.1},
        }
        if mode == "custom":
            return httpx.Response(200, json={
                "model_id": model_id, "decisions": answers, "token_usage": {},
            })
        return httpx.Response(200, json={
            "model": model_id, "answers": answers, "usage": {},
        })

    def create_client(**kwargs):
        client = real_client(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(decision_model.httpx, "Client", create_client)
    custom.main()
    assert len(requests) == 1
    assert requests[0][0] == url
    model_key = "model_id" if mode == "custom" else "model"
    assert requests[0][1][model_key] == model_id
    assert clients[0].is_closed
    assert f'"model": "{model_id}"' in capsys.readouterr().out


@pytest.mark.parametrize("page", PAGES)
def test_python_snippets_parse(page):
    for index, code in enumerate(re.findall(r"```python\n(.*?)```", page.read_text(), re.DOTALL)):
        if page == REFERENCE and reference_signature(code) is not None:
            continue
        ast.parse(code, filename=f"{page.name}:{index}")


@pytest.mark.parametrize("slug,name", [
    ("ticket-triage", "ticket_triage"), ("agent-router", "agent_router"),
    ("custom-decision-model", "custom_decision_model"),
])
def test_guide_contains_exact_runnable_file(slug, name):
    assert (CODE / f"{name}.py").read_text() in (EXAMPLES / f"{slug}.mdx").read_text()


def test_reference_signatures_match_installed_source():
    blocks = re.findall(r"```python\n(.*?)```", REFERENCE.read_text(), re.DOTALL)
    functions = [function for code in blocks
                 if (function := reference_signature(code)) is not None]
    expected = {
        "__init__", "run", "arun", "choice", "score", "noul", "list_models",
        "close", "build_headers", "build_payload", "parse_response", "get_decision_models",
    }
    assert {function.name for function in functions} == expected
    assert len(functions) == len(expected)
    for function in functions:
        target = (decision_model.get_decision_models
                  if function.name == "get_decision_models"
                  else getattr(DecisionModel, function.name))
        parameters = [param for name, param in inspect.signature(target).parameters.items()
                      if name != "self"]
        assert [arg.arg for arg in function.args.args] == [param.name for param in parameters]
        assert [ast.literal_eval(value) for value in function.args.defaults] == [
            param.default for param in parameters if param.default is not inspect.Parameter.empty
        ]
        assert isinstance(function, ast.AsyncFunctionDef) == inspect.iscoroutinefunction(target)


@pytest.mark.parametrize("code", [
    "DecisionModel(\n    model_name: str =\n)",
    "def run(state: str, questions: Dict[str, Any] -> Dict[str, Any]",
])
def test_reference_signature_parser_rejects_malformed_signatures(code):
    with pytest.raises(SyntaxError):
        if reference_signature(code) is None:
            ast.parse(code)


def test_navigation_has_all_four_pages():
    tabs = json.loads((ROOT / "docs.json").read_text())["navigation"]["tabs"]
    api_tab = next(tab for tab in tabs if tab["tab"] == "API Reference")
    core = next(group for group in api_tab["groups"] if group["group"] == "Core Classes")
    assert core["pages"].count("api/decision-model") == 1
    examples = next(tab for tab in tabs if tab["tab"] == "Examples")
    group = next(group for group in examples["groups"] if group["group"] == "Decision Models")
    assert group["pages"] == [
        "examples/decision-models/ticket-triage", "examples/decision-models/agent-router",
        "examples/decision-models/custom-decision-model",
    ]
    for path in ["api/decision-model", *group["pages"]]:
        assert (ROOT / f"{path}.mdx").is_file()


@pytest.mark.parametrize("page", PAGES)
def test_new_pages_have_frontmatter_and_resolvable_internal_links(page):
    text = page.read_text()
    assert text.startswith("---\n")
    assert "title:" in text.split("---", 2)[1]
    assert "description:" in text.split("---", 2)[1]
    for link in re.findall(r"\]\((/[^)]+)\)", text):
        path, _, fragment = link.partition("#")
        target = ROOT / f"{path.lstrip('/')}.mdx"
        assert target.is_file(), (page, link)
        if fragment:
            headings = re.findall(r"^#{1,6} (.+)$", target.read_text(), re.MULTILINE)
            anchors = {
                re.sub(r"\s+", "-", re.sub(r"[^\w\s-]", "", heading.lower()))
                for heading in headings
            }
            assert fragment in anchors, (page, link)


def test_router_main_constructs_real_agents_without_live_generation(monkeypatch, capsys):
    """Exercise real Agent construction; only run() replies are synthetic."""
    from swarms import Agent

    requests, clients, runs = [], [], []
    real_client = httpx.Client

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        state = payload["state"]
        selected = "Technical-Agent"
        if "invoice" in state:
            selected = "Billing-Agent"
        elif "enterprise" in state:
            selected = "Sales-Agent"
        scope = 0.1 if "history essay" in state else 0.9
        return httpx.Response(200, json={
            "model": "offline-fixture", "usage": {},
            "answers": routing_answers(scope=scope, choice=selected),
        })

    def create_client(**kwargs):
        client = real_client(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(client)
        return client

    def no_live_generation(self, task):
        runs.append((self.agent_name, task))
        return "offline agent reply fixture"

    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-fixture-not-a-credential")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture-not-a-credential")
    monkeypatch.setattr(decision_model.httpx, "Client", create_client)
    monkeypatch.setattr(Agent, "run", no_live_generation)
    router.main()
    assert len(requests) == 4
    assert [name for name, _ in runs] == ["Billing-Agent", "Technical-Agent", "Sales-Agent"]
    assert clients[-1].is_closed
    assert "Declined: out of scope for support." in capsys.readouterr().out
