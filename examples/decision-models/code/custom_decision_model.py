"""Use a pinned Jev, OpenRouter, or a provider with an envelope schema.

Every mode calls a real provider when executed. The custom schema is an
illustration, not a claim that a public service implements this protocol.
"""

import argparse
import json

from swarms import DecisionModel


class EnvelopeDecisionModel(DecisionModel):
    """Adapt context/prompts requests and decisions/token_usage responses."""

    def build_payload(self, state, questions):
        # Keep the base class's local validation before changing field names.
        payload = super().build_payload(state, questions)
        return {
            "model_id": payload.pop("model"),
            "context": payload.pop("state"),
            "prompts": payload.pop("questions"),
            **payload,
        }

    def parse_response(self, data, questions):
        # This provider keeps each typed answer's fields unchanged.
        normalized = {
            "model": data["model_id"],
            "answers": data["decisions"],
            "usage": data["token_usage"],
        }
        return super().parse_response(normalized, questions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", choices=("typesafe", "openrouter", "custom"),
        default="typesafe",
    )
    parser.add_argument("--base-url", help="Required for the custom provider")
    parser.add_argument("--model", help="Required for the custom provider")
    args = parser.parse_args()

    if args.provider == "openrouter":
        model = DecisionModel(
            base_url="https://openrouter.ai/api",
            endpoint="/v1/systemone",
            model_name="~typesafe/jev-latest",
            api_key_env="OPENROUTER_API_KEY",
        )
    elif args.provider == "custom":
        if not args.base_url or not args.model:
            parser.error("custom requires --base-url and --model")
        model = EnvelopeDecisionModel(
            base_url=args.base_url,
            endpoint="/evaluate",
            model_name=args.model,
            api_key_env="CUSTOM_DECISION_API_KEY",
        )
    else:
        model = DecisionModel(model_name="jev-1.13.0")

    try:
        result = model.run(
            state="Could you explain the difference between your plans?",
            questions={
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle this request?",
                    "criteria": {
                        "sales": "Plan comparisons and pricing",
                        "technical": "Software bugs and integrations",
                        "billing": "Invoices and existing payments",
                    },
                },
                "urgent": {
                    "type": "noul",
                    "instructions": "Does the request express urgency?",
                },
            },
        )
        print(json.dumps(result, indent=2))
    finally:
        model.close()


if __name__ == "__main__":
    main()
