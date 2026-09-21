from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_typesafe import NoulCriteria
from langchain_typesafe.experimental.middleware import AutoModeMiddleware
from dotenv import load_dotenv

load_dotenv()

@tool
def delete_all_backups() -> str:
    """Delete every backup. This action cannot be undone."""
    return "Backups deleted."


agent = create_agent(
    "google_genai:gemini-2.5-flash",
    # tools=[delete_all_backups],
    middleware=[AutoModeMiddleware(tools=[delete_all_backups])],  # ------> Jev calls that are determined risky return an error ToolMessage instead of running the tool.
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Delete all backups."}]}
)
print(result["messages"][-1].content)

