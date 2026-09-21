from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier
from dotenv import load_dotenv

load_dotenv()

classifier = TypeSafeClassifier()

response = classifier.invoke(
    {
        "state": (
            "The deploy failed twice and customers are seeing 500s. "
            "Can someone look now?"
        ),
        "questions": {
            "urgent": Noul(instructions="Does this need attention right now?"),
            "team": Choice(
                instructions="Which team should pick this up?",
                criteria={
                    "infra": "Deploys, availability, and on-call incidents.",
                    "billing": "Payments, invoices, and subscriptions.",
                },
            ),
            "severity": Score(
                instructions="How severe is the impact?",
                criteria=["Cosmetic.", "Degraded for some users.", "Full outage."],
            ),
        },
    }
)

print(response.nouls["urgent"].noul)
print(response.choices["team"].choice, response.choices["team"].confidence)
print(response.scores["severity"].score)