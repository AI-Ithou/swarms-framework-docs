"""Ask three typed questions in one request, then apply a local triage policy."""

import json


def triage_action(answers: dict) -> str:
    """Choose an action without making network requests or paging anyone."""
    department = answers["department"]
    frustration = answers["frustration"]["score"]
    urgency = answers["is_urgent"]["noul"]

    if department["confidence"] < 0.6:
        return "Send to a human: department confidence is below 0.6."
    if urgency > 0.8 and frustration >= 1:
        return f"Page the {department['choice']} on-call team."
    return f"Queue for the {department['choice']} team."


def main() -> None:
    from swarms import DecisionModel

    ticket = {
        "message": (
            "Hi, I've been trying to connect my Stripe account for 3 days "
            "and the integration keeps failing. I'm losing sales. "
            "Please help ASAP."
        ),
        "customer_plan": "enterprise",
    }
    model = DecisionModel(model_name="jev-latest")
    try:
        # One run call evaluates Choice, Score, and Noul together.
        result = model.run(
            state=ticket,
            questions={
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this ticket?",
                    "criteria": {
                        "billing": "Payment or subscription issues",
                        "technical": "Bugs or integration problems",
                        "sales": "Pricing or account questions",
                    },
                },
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated does the customer appear?",
                    "criteria": [
                        "Calm, just stating facts",
                        "Frustrated but civil",
                        "Very angry, strong language",
                    ],
                },
                "is_urgent": {
                    "type": "noul",
                    "instructions": (
                        "The message conveys urgency or time-sensitivity."
                    ),
                },
            },
        )
    finally:
        model.close()

    # Keep the full provider response available for inspection.
    print(json.dumps(result, indent=2))
    answers = result["answers"]
    department = answers["department"]
    print(f"Answered by {result['model']}")
    print(
        f"Department: {department['choice']} "
        f"(confidence {department['confidence']:.2f})"
    )
    print(f"Probabilities: {department['probabilities']}")
    print(f"Frustration: {answers['frustration']['score']:.2f} on a 0-2 scale")
    print(f"Urgency: {answers['is_urgent']['noul']:.2f}")
    print(f"Action: {triage_action(answers)}")


if __name__ == "__main__":
    main()
