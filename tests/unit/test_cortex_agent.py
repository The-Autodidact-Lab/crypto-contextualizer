from __future__ import annotations

from typing import Any
import os

import pytest

from are.simulation.agents.llm.types import MMObservation
from are.simulation.agents.llm.llm_engine import LLMEngine, MockLLMEngine

from cortex import cortex_agent as ca
from cortex.context_cortex import cortex


def reset_cortex() -> None:
    cortex._agents.clear()
    cortex._episodes.clear()


def _make_mock_llm_engine_with_response(response: str) -> MockLLMEngine:
    """
    Helper to build a MockLLMEngine that always returns `response` for the first call.
    The underlying engine is a stub that should never actually be used.
    """

    class _StubEngine(LLMEngine):
        def chat_completion(self, messages, stop_sequences=None, **kwargs):
            raise RuntimeError("StubEngine.chat_completion should not be called")

        def simple_call(self, prompt: str) -> str:
            raise RuntimeError("StubEngine.simple_call should not be called")

    stub = _StubEngine(model_name="stub")
    return MockLLMEngine(mock_responses=[response], engine=stub)


def test_ingest_tool_without_pending_context_returns_message():
    # Directly exercise the defensive branch in IngestEpisodeTool.
    tool = ca.IngestEpisodeTool(get_pending_context=lambda: None)
    out = tool("summary", "1")
    assert "No pending episode context" in out


def test_cortex_agent_run_integration_with_dummy_llm(monkeypatch):
    """
    Full-path integration test:
    - Use MockLLMEngine so the ReAct loop emits a real JSON tool call.
    - Verify that the ingest tool writes into the shared cortex singleton.
    """
    reset_cortex()

    # Patch LLMEngineBuilder.create_engine used inside CortexAgent to return
    # a MockLLMEngine with a deterministic ReAct-style response.
    def fake_create_engine(self, engine_config, mock_responses=None):
        response = (
            "Thought: I will summarize the trace and store it in the Cortex.\n"
            "Action:\n"
            "{\n"
            '  \"action\": \"ingest_episode\",\n'
            '  \"action_input\": {\n'
            '    \"trace_summary\": \"Deterministic summary of the trace.\",\n'
            '    \"mask_str\": \"10\"\n'
            "  }\n"
            "}<end_action>\n"
        )
        return _make_mock_llm_engine_with_response(response)

    monkeypatch.setattr(ca.LLMEngineBuilder, "create_engine", fake_create_engine)

    agent = ca.CortexAgent(api_key="dummy-key")

    episode_id = "ep_integration"
    source_agent_id = "orchestrator"
    raw_trace = {"logs": ["step 1", "step 2"]}

    result = agent.run(
        episode_id=episode_id,
        source_agent_id=source_agent_id,
        raw_trace=raw_trace,
        metadata={"step": 2},
    )

    # BaseAgent may return None; CortexAgent normalizes that to an empty string.
    assert isinstance(result, str)

    # Verify that an episode was actually written.
    episodes = list(cortex._episodes.values())
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.episode_id == episode_id
    assert ep.source_agent_id == source_agent_id
    assert ep.raw_trace == raw_trace
    assert ep.summary == "Deterministic summary of the trace."
    assert ep.access_mask == 0b10
    assert ep.metadata["step"] == 2


def test_cortex_agent_run_return_branches(monkeypatch):
    """
    Unit-style tests that exercise the return-value branches in CortexAgent.run:
    - When BaseAgent.run returns an MMObservation.
    - When BaseAgent.run returns a plain string.
    """
    reset_cortex()

    # Patch engine creation to avoid any real network calls; the engine won't
    # actually be used because we monkeypatch BaseAgent.run below.
    def fake_create_engine(self, engine_config, mock_responses=None):
        return _make_mock_llm_engine_with_response("Thought: unused\n")

    monkeypatch.setattr(ca.LLMEngineBuilder, "create_engine", fake_create_engine)

    agent = ca.CortexAgent(api_key="dummy-key")

    # 1) When superclass run returns MMObservation, we should unwrap .content.
    def fake_run_obs(self, task: str, reset: bool = True, **kwargs):
        # Construct a real MMObservation pydantic model
        return MMObservation(content="obs-content", attachments=[])

    original_run = ca.BaseAgent.run
    monkeypatch.setattr(ca.BaseAgent, "run", fake_run_obs)

    res_obs = agent.run(
        episode_id="ep_obs",
        source_agent_id="src",
        raw_trace="trace",
        metadata=None,
    )
    assert res_obs == "obs-content"

    # 2) When superclass run returns a plain string, we should return that string.
    def fake_run_str(self, task: str, reset: bool = True, **kwargs):
        return "plain-string"

    monkeypatch.setattr(ca.BaseAgent, "run", fake_run_str)

    res_str = agent.run(
        episode_id="ep_str",
        source_agent_id="src",
        raw_trace="trace",
        metadata=None,
    )
    assert res_str == "plain-string"

    # Restore original for cleanliness (not strictly necessary in pytest).
    monkeypatch.setattr(ca.BaseAgent, "run", original_run)


@pytest.mark.skipif(
    "GEMINI_API_KEY" not in os.environ,
    reason="Requires a real GEMINI_API_KEY to call the Gemini LLM.",
)
def test_cortex_agent_with_real_gemini():
    """
    End-to-end test that uses the real Gemini-backed CortexAgent.

    This verifies that:
    - The prompt is accepted by the model.
    - The agent emits a valid JSON tool call to `ingest_episode`.
    - The ingest tool writes a non-empty summary and non-zero mask into the cortex.
    """
    reset_cortex()

    agent = ca.CortexAgent(api_key=os.environ["GEMINI_API_KEY"])

    episode_id = "ep_real"
    source_agent_id = "orchestrator"
    raw_trace = {"logs": ["user asked to schedule a meeting", "agent delegated to calendar"]}

    result = agent.run(
        episode_id=episode_id,
        source_agent_id=source_agent_id,
        raw_trace=raw_trace,
        metadata={"kind": "real_gemini_integration"},
    )

    # We don't depend on the exact text, only that something came back.
    assert isinstance(result, str)

    episodes = list(cortex._episodes.values())
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.episode_id == episode_id
    assert ep.source_agent_id == source_agent_id
    assert ep.raw_trace == raw_trace
    # Real model should produce a non-empty summary and a non-zero mask.
    assert isinstance(ep.summary, str) and ep.summary.strip()
    assert ep.access_mask != 0


