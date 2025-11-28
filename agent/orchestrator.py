"""
Lightweight orchestrator agent built on top of Meta's ARE `BaseAgent`.

Design goals:
- Reuse the existing ReAct-style `BaseAgent` + `JsonActionExecutor`.
- Expose subagents (e.g. AppAgent.expert_agent) as tools the orchestrator can call.
- Keep configuration simple and local to this repository for now.

This module does **not** wire the orchestrator into ARE's CLI yet; it is meant to be
imported from your own code or scenarios. Promotion to a first-class ARE agent can
reuse the same `OrchestratorAgent` implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional

from evals.are.simulation.agents.default_agent.base_agent import (
    BaseAgent as MetaBaseAgent,
)
from evals.are.simulation.agents.default_agent.tools.json_action_executor import (
    JsonActionExecutor,
)
from evals.are.simulation.agents.default_agent.default_tools import FinalAnswerTool
from evals.are.simulation.tools import SystemPrompt, Tool


ROOT_DIR = Path(__file__).resolve().parent
ORCHESTRATOR_PROMPT_PATH = ROOT_DIR / "orchestrator_prompt.txt"
SUBAGENT_PROMPT_PATH = ROOT_DIR / "subagent_prompt.txt"


def _load_prompt(path: Path, default: str) -> str:
    """
    Load a prompt from disk, falling back to `default` if the file is missing or empty.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    stripped = text.strip()
    return stripped or default


DEFAULT_ORCHESTRATOR_PROMPT = _load_prompt(
    ORCHESTRATOR_PROMPT_PATH,
    default=(
        "You are an orchestrator agent. You can solve tasks yourself or delegate\n"
        "subtasks to specialized subagents exposed as tools.\n\n"
        "- Think step by step.\n"
        "- When a subagent is clearly a better fit (e.g. Calendar, Email, DB), call\n"
        "  its tool with a clear, self-contained instruction.\n"
        "- After delegating, read the tool's result and decide whether additional\n"
        "  steps or subagent calls are needed.\n"
        "- Always finish by calling the `final_answer` tool with the best overall\n"
        "  answer you can provide.\n"
    ),
)

DEFAULT_SUBAGENT_PROMPT = _load_prompt(
    SUBAGENT_PROMPT_PATH,
    default=(
        "You are an expert subagent for a specific application (e.g. calendar,\n"
        "email, database, or messaging).\n\n"
        "You receive tasks from an orchestrator agent that already did some\n"
        "planning. Your job is to:\n"
        "- Use your application tools to complete the requested work.\n"
        "- Be precise about any changes you make to the app state.\n"
        "- When you are done, call the `final_answer` tool summarizing what you did\n"
        "  and returning any requested result.\n"
    ),
)


class SubagentTool(Tool):
    """
    Minimal Tool wrapper that delegates to a subagent callable.

    The callable should have the signature: (task: str) -> str | object.
    """

    def __init__(
        self,
        name: str,
        description: str,
        delegate: Callable[[str], object],
    ) -> None:
        self.name = name
        self.description = description
        self.inputs = {
            "task": {
                "type": "string",
                "description": "Task description for this subagent to execute.",
            }
        }
        self.output_type = "any"
        super().__init__()
        self._delegate = delegate

    def forward(self, task: str) -> object:
        return self._delegate(task)


def make_app_agent_subagent_tool(app_name: str, app_agent) -> Tool:
    """
    Wrap an ARE `AppAgent` (from `app_agent.py`) as a Tool the orchestrator can call.

    The returned Tool uses the `expert_agent` entry point on the given app_agent.
    """
    description = (
        f"Delegate a task to the {app_name} expert agent via its `expert_agent` tool. "
        "Use this when the task clearly requires that application."
    )
    return SubagentTool(
        name=f"{app_name}__expert_agent",
        description=description,
        delegate=app_agent.expert_agent,
    )


class OrchestratorAgent(MetaBaseAgent):
    """
    Orchestrator built on top of Meta's ARE `BaseAgent`.

    Parameters
    ----------
    llm_engine:
        Callable implementing the ARE LLM engine interface:
        llm_engine(prompt: list[dict], **kwargs) -> str | (str, metadata).
    subagent_tools:
        Mapping from tool name to Tool, typically created via `make_app_agent_subagent_tool`.
    system_prompt:
        Optional override for the orchestrator system prompt. If omitted, a default
        multi-tool orchestration prompt is used.
    """

    def __init__(
        self,
        llm_engine: Callable,
        subagent_tools: Optional[Mapping[str, Tool]] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        tools: Dict[str, Tool] = {"final_answer": FinalAnswerTool()}
        if subagent_tools:
            tools.update(dict(subagent_tools))

        system_prompts = {
            "system_prompt": SystemPrompt(
                prompt=system_prompt or DEFAULT_ORCHESTRATOR_PROMPT
            )
        }

        action_executor = JsonActionExecutor(tools=tools)

        super().__init__(
            llm_engine=llm_engine,
            system_prompts=system_prompts,
            tools=tools,
            action_executor=action_executor,
        )


def create_orchestrator(
    llm_engine: Callable,
    subagent_tools: Iterable[Tool] | Mapping[str, Tool],
    system_prompt: Optional[str] = None,
) -> OrchestratorAgent:
    """
    Convenience builder for a fully-wired OrchestratorAgent.

    `subagent_tools` can be:
    - A mapping `name -> Tool`, or
    - An iterable of Tool instances, in which case their `name` attributes are used.
    """
    if isinstance(subagent_tools, Mapping):
        tool_map = dict(subagent_tools)
    else:
        tool_map = {tool.name: tool for tool in subagent_tools}

    return OrchestratorAgent(
        llm_engine=llm_engine,
        subagent_tools=tool_map,
        system_prompt=system_prompt,
    )


