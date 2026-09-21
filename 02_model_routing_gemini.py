from langchain.agents import create_agent
from langchain_typesafe.experimental.middleware import (
    ModelChoice,
    ModelRouterMiddleware,
)
from dotenv import load_dotenv

load_dotenv()

router = ModelRouterMiddleware(
    choices={
        "fast": ModelChoice(
            model="google_genai:gemini-2.5-flash",
            criteria="Direct lookups, extraction, and localized changes with explicit targets.",
        ),
        "powerful": ModelChoice(
            model="google_genai:gemini-3.1-pro-preview",
            criteria="Architecture, novel root-cause reasoning, and high-stakes decisions.",
        ),
    },
    instructions="Choose the least costly model that can complete the task safely.",
)

agent = create_agent(
    "google_genai:gemini-2.5-flash",
    middleware=[router],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Prove there is only one GOD!",
            }
        ]
    }
)

# result = agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "Five boxes contain coins. Exactly one box contains only gold coins, exactly one contains only silver coins, and the remaining three contain a mixture of gold and silver coins. Each box is labeled either “Gold,” “Silver,” “Mixed,” “Gold or Mixed,” or “Silver or Mixed.”You are told that every label is wrong. You may draw one coin from exactly one box, without looking inside.Which box should you draw from, and how can you determine the contents of all five boxes from that single coin?"
#                 "Explain your reasoning step by step and prove that your strategy always works.",
#             }
#         ]
#     }
# )

print(result["model_route"].choice)