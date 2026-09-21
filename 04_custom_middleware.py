from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, Runtime
from langchain_typesafe import Choice, ChoiceAnswer, TypeSafeClassifier
from typing_extensions import NotRequired
from dotenv import load_dotenv

load_dotenv()

class TriageState(AgentState):
    triage: NotRequired[ChoiceAnswer]


class TriageMiddleware(AgentMiddleware[TriageState]):
    state_schema = TriageState

    def __init__(self) -> None:
        self.classifier = TypeSafeClassifier()

    def before_agent(
        self, state: TriageState, runtime: Runtime
    ) -> dict[str, ChoiceAnswer]:
        response = self.classifier.invoke(
            {
                "state": state["messages"],
                "questions": {
                    "triage": Choice(
                        instructions="Which team should handle this conversation?",
                        criteria={
                            "billing": "Payments, invoices, and subscriptions.",
                            "infra": "Deploys, availability, and incidents.",
                            "other": "Requests that belong to another team.",
                        },
                    )
                },
            }
        )
        return {"triage": response.choices["triage"]}


agent = create_agent(
    "google_genai:gemini-2.5-flash",
    middleware=[TriageMiddleware()],
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Customers are seeing 500 errors."}]}
)
print(result["triage"].choice, result["triage"].confidence)