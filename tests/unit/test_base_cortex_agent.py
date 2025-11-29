# tests to ensure BaseCortexAgent is working as expected

import pytest
from are.simulation.agents.default_agent.base_cortex_agent import BaseCortexAgent
from are.simulation.agents.default_agent.tools.json_action_executor import JsonActionExecutor
from are.simulation.agents.default_agent.default_tools import FinalAnswerTool
from are.simulation.agents.llm.llm_engine import LLMEngine, MockLLMEngine
from cortex.context_cortex import ContextCortex, cortex
from cortex.cortex_agent import CortexAgent


def reset_cortex():
    cortex._agents.clear()
    cortex._episodes.clear()


class StubLLMEngine(LLMEngine):
    def chat_completion(self, messages, stop_sequences=None, **kwargs):
        raise RuntimeError("StubLLMEngine.chat_completion should not be called")
    
    def simple_call(self, prompt: str) -> str:
        raise RuntimeError("StubLLMEngine.simple_call should not be called")


def _make_mock_llm_engine_with_response(response: str) -> MockLLMEngine:
    stub = StubLLMEngine(model_name="stub")
    return MockLLMEngine(mock_responses=[response], engine=stub)


# initialisation
def test_base_cortex_agent_initialization():
    reset_cortex()
    
    cortex_instance = ContextCortex()
    cortex_agent = CortexAgent(api_key="dummy-key")
    
    llm_engine = _make_mock_llm_engine_with_response("Thought: test\nAction: {\"action\": \"final_answer\", \"action_input\": {\"answer\": \"done\"}}<end_action>")
    
    agent = BaseCortexAgent(
        cortex=cortex_instance,
        cortex_agent=cortex_agent,
        agent_id="test_agent",
        agent_mask=0b1,
        llm_engine=llm_engine,
        tools={"final_answer": FinalAnswerTool()},
        action_executor=JsonActionExecutor(tools={"final_answer": FinalAnswerTool()}),
        system_prompts={"system_prompt": "You are a test agent."},
    )
    
    assert agent.cortex == cortex_instance
    assert agent.cortex_agent == cortex_agent
    assert agent.agent_id == "test_agent"
    assert agent.agent_mask == 0b1
    assert agent._last_cortex_ingestion_index == -1
    assert len(agent.conditional_post_steps) > 0
    assert any(step.name == "test_agent_cortex_post" for step in agent.conditional_post_steps)


# post-step hook
def test_cortex_post_step_ingests_trace(monkeypatch):
    reset_cortex()
    
    cortex_instance = ContextCortex()
    
    # Mock cortex_agent.run to capture calls
    ingest_calls = []
    def mock_cortex_agent_run(episode_id, source_agent_id, raw_trace, metadata):
        ingest_calls.append({
            "episode_id": episode_id,
            "source_agent_id": source_agent_id,
            "raw_trace": raw_trace,
            "metadata": metadata,
        })
        return "ingested"
    
    cortex_agent = CortexAgent(api_key="dummy-key")
    monkeypatch.setattr(cortex_agent, "run", mock_cortex_agent_run)
    
    llm_engine = _make_mock_llm_engine_with_response("Thought: test\nAction: {\"action\": \"final_answer\", \"action_input\": {\"answer\": \"done\"}}<end_action>")
    
    agent = BaseCortexAgent(
        cortex=cortex_instance,
        cortex_agent=cortex_agent,
        agent_id="test_agent",
        agent_mask=0b1,
        llm_engine=llm_engine,
        tools={"final_answer": FinalAnswerTool()},
        action_executor=JsonActionExecutor(tools={"final_answer": FinalAnswerTool()}),
        system_prompts={"system_prompt": "You are a test agent."},
    )
    
    # Run agent to generate logs
    agent.run("Test task", reset=True)
    
    # Post-step hook should have been called, ingesting trace
    assert len(ingest_calls) > 0
    assert ingest_calls[0]["source_agent_id"] == "test_agent"
    assert "logs" in ingest_calls[0]["raw_trace"]
    assert agent._last_cortex_ingestion_index >= 0


def test_build_history_injects_cortex_episodes():
    reset_cortex()
    
    cortex_instance = ContextCortex()
    cortex_instance.register_agent("other_agent", mask=0b10)
    cortex_instance.ingest_episode(
        episode_id="ep_1",
        source_agent_id="other_agent",
        raw_trace={"logs": ["test"]},
        trace_summary="Other agent did something",
        mask_str="11",
    )
    
    cortex_agent = CortexAgent(api_key="dummy-key")
    llm_engine = _make_mock_llm_engine_with_response("Thought: test\nAction: {\"action\": \"final_answer\", \"action_input\": {\"answer\": \"done\"}}<end_action>")
    
    agent = BaseCortexAgent(
        cortex=cortex_instance,
        cortex_agent=cortex_agent,
        agent_id="test_agent",
        agent_mask=0b1,
        llm_engine=llm_engine,
        tools={"final_answer": FinalAnswerTool()},
        action_executor=JsonActionExecutor(tools={"final_answer": FinalAnswerTool()}),
        system_prompts={"system_prompt": "You are a test agent."},
    )
    
    # Initialize to create system prompt log
    agent.initialize()
    
    # Build history should inject cortex episodes
    history = agent.build_history_from_logs()
    
    # Find system prompt message
    system_msg = next((msg for msg in history if msg.get("role") == "system"), None)
    assert system_msg is not None
    assert "<relevant_multiagent_context>" in system_msg["content"]
    assert "other_agent" in system_msg["content"]
    assert "Other agent did something" in system_msg["content"]


# error modes
def test_cortex_post_step_handles_missing_cortex():
    cortex_agent = CortexAgent(api_key="dummy-key")
    llm_engine = _make_mock_llm_engine_with_response("Thought: test\nAction: {\"action\": \"final_answer\", \"action_input\": {\"answer\": \"done\"}}<end_action>")
    
    agent = BaseCortexAgent(
        cortex=None,  # Missing cortex
        cortex_agent=cortex_agent,
        agent_id="test_agent",
        agent_mask=0b1,
        llm_engine=llm_engine,
        tools={"final_answer": FinalAnswerTool()},
        action_executor=JsonActionExecutor(tools={"final_answer": FinalAnswerTool()}),
        system_prompts={"system_prompt": "You are a test agent."},
    )
    
    # Should not raise error when cortex is None
    agent._cortex_post_step(agent)
    assert agent._last_cortex_ingestion_index == -1


def test_build_history_handles_no_cortex():
    cortex_agent = CortexAgent(api_key="dummy-key")
    llm_engine = _make_mock_llm_engine_with_response("Thought: test\nAction: {\"action\": \"final_answer\", \"action_input\": {\"answer\": \"done\"}}<end_action>")
    
    agent = BaseCortexAgent(
        cortex=None,
        cortex_agent=cortex_agent,
        agent_id="test_agent",
        agent_mask=0b1,
        llm_engine=llm_engine,
        tools={"final_answer": FinalAnswerTool()},
        action_executor=JsonActionExecutor(tools={"final_answer": FinalAnswerTool()}),
        system_prompts={"system_prompt": "You are a test agent."},
    )
    
    agent.initialize()
    history = agent.build_history_from_logs()
    
    # Should return normal history without cortex injection
    assert len(history) > 0
    system_msg = next((msg for msg in history if msg.get("role") == "system"), None)
    assert system_msg is not None
    assert "<relevant_multiagent_context>" not in system_msg["content"]
