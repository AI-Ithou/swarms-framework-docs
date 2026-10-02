"""Route a support task to one agent after scope and confidence checks."""

import json


def choose_route(answers: dict, agent_names: set[str]) -> tuple[str, str]:
    """Return an action and its message or agent name; no network calls."""
    if answers["in_scope"]["noul"] < 0.5:
        return "decline", "Declined: out of scope for support."

    decision = answers["agent"]
    if decision["confidence"] < 0.5:
        return "escalate", "Escalated to a human: routing confidence is below 0.5."
    if decision["choice"] not in agent_names:
        return "escalate", "Escalated to a human: the selected agent is unknown."
    return "run", decision["choice"]


def route(task: str, router, agents_by_name: dict) -> str:
    """Evaluate both routing questions, then run only the selected agent."""
    result = router.run(
        state=task,
        questions={
            "agent": {
                "type": "choice",
                "instructions": "Which agent should handle this request?",
                "criteria": {
                    agent.agent_name: agent.agent_description
                    for agent in agents_by_name.values()
                },
            },
            "in_scope": {
                "type": "noul",
                "instructions": (
                    "This is a request a software company's support team "
                    "should handle."
                ),
            },
        },
    )
    # Print the full decision response before any agent executes.
    print(json.dumps(result, indent=2))
    action, target = choose_route(result["answers"], set(agents_by_name))
    if action != "run":
        return target

    confidence = result["answers"]["agent"]["confidence"]
    print(f"Routing to {target} (confidence {confidence:.2f})")
    return agents_by_name[target].run(task)


def main() -> None:
    from swarms import Agent, DecisionModel

    agents = [
        Agent(
            agent_name="Billing-Agent",
            agent_description=(
                "Refunds, invoices, failed payments and subscription changes."
            ),
            system_prompt="You resolve billing questions clearly and briefly.",
            model_name="gpt-5.4",
            max_loops=1,
            output_type="str",
        ),
        Agent(
            agent_name="Technical-Agent",
            agent_description="Bugs, API errors, integrations and outages.",
            system_prompt="You debug technical problems step by step.",
            model_name="gpt-5.4",
            max_loops=1,
            output_type="str",
        ),
        Agent(
            agent_name="Sales-Agent",
            agent_description="Pricing, plan upgrades and new accounts.",
            system_prompt="You answer pricing and plan questions.",
            model_name="gpt-5.4",
            max_loops=1,
            output_type="str",
        ),
    ]
    agents_by_name = {agent.agent_name: agent for agent in agents}
    tasks = [
        "I was charged twice for my March invoice, can you refund one?",
        "Our webhook endpoint returns 500 since your API update this morning.",
        "What does the enterprise plan cost for 200 seats?",
        "Can you write my history essay on the French Revolution?",
    ]
    router = DecisionModel(model_name="jev-latest")
    try:
        for task in tasks:
            print(f"\nTask: {task}")
            print(route(task, router, agents_by_name))
    finally:
        router.close()


if __name__ == "__main__":
    main()
