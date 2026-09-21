# Building an Agent Harness with Jev & LangChain

An evaluation, routing, and guardrail harness showcasing how to integrate **Jev** / **TypeSafe Classifiers** (`langchain-typesafe`) into **LangChain** agent architectures.

This repository provides hands-on patterns for building safe, cost-efficient, and deterministic agent systems—covering semantic classification, dynamic model routing, automated tool risk gating, and custom agent middleware.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Prerequisites & Setup](#prerequisites--setup)
- [Environment Configuration](#environment-configuration)
- [Walkthrough & Examples](#walkthrough--examples)
  - [0. Basic Binary Classification (`Noul`)](#0-basic-binary-classification-noul)
  - [1. Multi-Dimensional Classification (`Choice` & `Score`)](#1-multi-dimensional-classification-choice--score)
  - [2. Dynamic Model Routing (`ModelRouterMiddleware`)](#2-dynamic-model-routing-modelroutermiddleware)
  - [3. Tool Risk Gating (`AutoModeMiddleware`)](#3-tool-risk-gating-automodemiddleware)
  - [4. Custom Agent Middleware (`AgentMiddleware`)](#4-custom-agent-middleware-agentmiddleware)
- [Core TypeSafe Primitives](#core-typesafe-primitives)
- [Running the Scripts](#running-the-scripts)
- [Tech Stack](#tech-stack)

---

## 🚀 Overview

Modern agentic workflows require fast, deterministic guardrails and decisions—such as deciding whether an incoming prompt needs urgent escalation, picking between an inexpensive flash model vs. a deep-reasoning model, or preventing dangerous tool invocations like deleting databases.

Using **`langchain-typesafe`**, this project implements:
1. **Semantic triage & scoring** directly on conversation state.
2. **Dynamic model routing** to minimize LLM inference costs and latency while preserving quality.
3. **Automated risk gating** to intercept high-risk tool executions before execution.
4. **Custom middleware hooks** to inject typed classification metadata into the agent lifecycle.

---

## ✨ Key Features

- **Typed Classifiers**: Get structured answers (`Noul`, `Choice`, `Score`) with confidence metrics directly from conversation state.
- **Cost-Optimized Model Routing**: Automatically route simple lookup tasks to faster models (e.g., `gemini-2.5-flash`) and complex analytical reasoning to deeper models (e.g., `gemini-3.1-pro-preview` or `gpt-6-astra`).
- **Autonomous Tool Gating**: Prevent catastrophic actions (data deletion, privilege escalation) by checking tool safety policies dynamically.
- **Stateful Middleware**: Intercept agent requests in `before_agent` hooks to enrich agent state with triage and routing info.

---

## 📂 Project Structure

```text
.
├── 00_test.py                  # Single-question binary classification (Noul)
├── 01_test.py                  # Multi-question classification (Noul, Choice, Score)
├── 02_model_routing_gemini.py  # Model routing between Gemini Flash and Pro Preview
├── 02_model_routing_openai.py  # Model routing pattern using OpenAI models
├── 03_tool_risk_gating.py      # Automated risk interception for sensitive tools
├── 04_custom_middleware.py     # Custom TriageMiddleware extending LangChain AgentMiddleware
├── .env.example                # Template for environment variables
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🛠 Prerequisites & Setup

### 1. Clone the repository

```bash
git clone https://github.com/p2kalita/Building-a-Harness-with-Jev-LangChain.git
cd Building-a-Harness-with-Jev-LangChain
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Configuration

Create a `.env` file in the root directory by copying `.env.example`:

```bash
cp .env.example .env
```

Set your API credentials in `.env`:

```env
TYPESAFE_API_KEY="your_typesafe_api_key_here"
GEMINI_API_KEY="your_gemini_api_key_here"
GROQ_API_KEY="your_groq_api_key_here"       # Optional
OPENAI_API_KEY="your_openai_api_key_here"   # Optional, for OpenAI routing
```

---

## 📖 Walkthrough & Examples

### 0. Basic Binary Classification (`Noul`)
*File: `00_test.py`*

Demonstrates binary query classification using `Noul` (evaluating a condition as yes/no / true/false) to detect whether an operational ticket requires immediate attention:

```python
from langchain_typesafe import Noul, TypeSafeClassifier

classifier = TypeSafeClassifier()
response = classifier.invoke({
    "state": "The deploy failed twice and customers are seeing 500s. Can someone look now?",
    "questions": {
        "urgent": Noul(instructions="Does this need attention right now?"),
    },
})

print(response.nouls["urgent"].noul)  # True
```

---

### 1. Multi-Dimensional Classification (`Choice` & `Score`)
*File: `01_test.py`*

Evaluates multiple dimensions simultaneously over the same state:
- **`Noul`**: Is this incident urgent?
- **`Choice`**: Categorical team routing (`infra` vs `billing`).
- **`Score`**: Impact rubric scoring (`Cosmetic`, `Degraded for some users`, `Full outage`).

```python
from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier

classifier = TypeSafeClassifier()
response = classifier.invoke({
    "state": "The deploy failed twice and customers are seeing 500s. Can someone look now?",
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
})

print("Urgent:", response.nouls["urgent"].noul)
print("Team:", response.choices["team"].choice, "Confidence:", response.choices["team"].confidence)
print("Severity:", response.scores["severity"].score)
```

---

### 2. Dynamic Model Routing (`ModelRouterMiddleware`)
*Files: `02_model_routing_gemini.py` & `02_model_routing_openai.py`*

Dynamically selects the appropriate model tier based on the user request before initiating agent execution:

```python
from langchain.agents import create_agent
from langchain_typesafe.experimental.middleware import ModelChoice, ModelRouterMiddleware

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

result = agent.invoke({"messages": [{"role": "user", "content": "Prove there is only one GOD!"}]})
print("Selected Route:", result["model_route"].choice)
```

---

### 3. Tool Risk Gating (`AutoModeMiddleware`)
*File: `03_tool_risk_gating.py`*

Protects systems by gating dangerous tool calls. If a tool call is flagged as unsafe or irreversible, the middleware blocks execution and injects an error `ToolMessage`:

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_typesafe.experimental.middleware import AutoModeMiddleware

@tool
def delete_all_backups() -> str:
    """Delete every backup. This action cannot be undone."""
    return "Backups deleted."

agent = create_agent(
    "google_genai:gemini-2.5-flash",
    middleware=[AutoModeMiddleware(tools=[delete_all_backups])],
)

result = agent.invoke({"messages": [{"role": "user", "content": "Delete all backups."}]})
print(result["messages"][-1].content)  # Blocked safely by middleware
```

---

### 4. Custom Agent Middleware (`AgentMiddleware`)
*File: `04_custom_middleware.py`*

Build custom lifecycle middleware to enrich `AgentState` before the agent runs:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, Runtime
from langchain_typesafe import Choice, ChoiceAnswer, TypeSafeClassifier
from typing_extensions import NotRequired

class TriageState(AgentState):
    triage: NotRequired[ChoiceAnswer]

class TriageMiddleware(AgentMiddleware[TriageState]):
    state_schema = TriageState

    def __init__(self) -> None:
        self.classifier = TypeSafeClassifier()

    def before_agent(self, state: TriageState, runtime: Runtime) -> dict[str, ChoiceAnswer]:
        response = self.classifier.invoke({
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
        })
        return {"triage": response.choices["triage"]}

agent = create_agent(
    "google_genai:gemini-2.5-flash",
    middleware=[TriageMiddleware()],
)

result = agent.invoke({"messages": [{"role": "user", "content": "Customers are receiving overdue notifications."}]})
print("Triage:", result["triage"].choice, "Confidence:", result["triage"].confidence)
```

---

## 🧩 Core TypeSafe Primitives

| Primitive | Purpose | Output |
| :--- | :--- | :--- |
| **`Noul`** | Binary condition evaluator (True / False) | `response.nouls[key].noul` |
| **`Choice`** | Multi-class categorizer with defined criteria | `response.choices[key].choice`, `.confidence` |
| **`Score`** | Ordinal rubric / severity assessment | `response.scores[key].score` |
| **`ModelRouterMiddleware`** | Routes request to optimal LLM based on complexity | `result["model_route"].choice` |
| **`AutoModeMiddleware`** | Safeguards risky or destructive tool calls | Gated execution / Error `ToolMessage` |

---

## 🏃 Running the Scripts

Run any script with Python from the project root:

```bash
# Run binary classifier test
python 00_test.py

# Run multi-dimensional triage & scoring
python 01_test.py

# Run dynamic model router with Gemini
python 02_model_routing_gemini.py

# Run dynamic model router with OpenAI
python 02_model_routing_openai.py

# Run tool risk gating
python 03_tool_risk_gating.py

# Run custom triage middleware
python 04_custom_middleware.py
```

---

## 📦 Tech Stack

- **[LangChain](https://github.com/langchain-ai/langchain)** - Agent creation, tooling, and middleware abstraction
- **[langchain-typesafe](https://docs.langchain.com/oss/python/integrations/providers/typesafe)** - TypeSafe / Jev classifier and experimental middleware
- **[Google GenAI SDK](https://github.com/langchain-ai/langchain-google)** - Gemini 2.5 Flash / 3.1 Pro integration
- **Python-dotenv** - Environment variable management