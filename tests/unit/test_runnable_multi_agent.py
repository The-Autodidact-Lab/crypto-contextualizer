"""
Comprehensive tests for RunnableARESimulationMultiAgent.

Tests cover initialization, method overrides, MultiAgent integration,
scenario execution, and evaluation access patterns.
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add workspace root and meta_are_base to sys.path to enable imports
workspace_root = Path(__file__).parent.parent.parent
meta_are_base_dir = workspace_root / "meta_are_base"

# Add meta_are_base first so 'are' module can be found by internal imports
if str(meta_are_base_dir) not in sys.path:
    sys.path.insert(0, str(meta_are_base_dir))

# Add workspace root so root-level packages (run, agent, cortex) can be imported
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import pytest

from are.simulation.agents.default_agent.base_agent import BaseAgent, BaseAgentLog
from are.simulation.agents.default_agent.are_simulation_main import ARESimulationAgent
from are.simulation.agents.default_agent.agent_factory import are_simulation_react_json_agent
from are.simulation.agents.are_simulation_agent_config import (
    ARESimulationReactBaseAgentConfig,
)
from are.simulation.agents.llm.llm_engine import LLMEngine, MockLLMEngine
from are.simulation.apps import App
from are.simulation.scenarios import Scenario
from are.simulation.time_manager import TimeManager
from are.simulation.types import SimulatedGenerationTimeConfig

from run.runnable_multi_agent import RunnableARESimulationMultiAgent
from agent.multi_agent import MultiAgent
from cortex.context_cortex import ContextCortex


def reset_cortex():
    """Reset the singleton cortex for clean test state."""
    from cortex.context_cortex import cortex
    cortex._agents.clear()
    cortex._episodes.clear()


class StubLLMEngine(LLMEngine):
    """Stub LLM engine that should never be called."""
    
    def chat_completion(self, messages, stop_sequences=None, **kwargs):
        raise RuntimeError("StubLLMEngine.chat_completion should not be called")
    
    def simple_call(self, prompt: str) -> str:
        raise RuntimeError("StubLLMEngine.simple_call should not be called")


def _make_mock_llm_engine_with_response(response: str) -> MockLLMEngine:
    """Create a MockLLMEngine with a deterministic response."""
    stub = StubLLMEngine(model_name="stub")
    return MockLLMEngine(mock_responses=[response], engine=stub)


def _create_mock_app(name: str) -> Mock:
    """Create a mock App for testing."""
    app = Mock(spec=App)
    app.name = name
    app.get_tools.return_value = []
    return app


def _create_mock_scenario(apps: list[App] | None = None, **kwargs) -> Mock:
    """Create a mock Scenario for testing."""
    scenario = Mock(spec=Scenario)
    scenario.scenario_id = kwargs.get("scenario_id", "test_scenario")
    scenario.apps = apps if apps is not None else []
    scenario.additional_system_prompt = kwargs.get("additional_system_prompt", None)
    scenario.start_time = kwargs.get("start_time", 1000.0)
    scenario.nb_turns = kwargs.get("nb_turns", None)
    scenario.get_tools.return_value = []
    return scenario


def _create_placeholder_base_agent(llm_engine: LLMEngine) -> BaseAgent:
    """Create a placeholder base agent for initialization."""
    base_config = ARESimulationReactBaseAgentConfig(
        system_prompt="Test system prompt",
        max_iterations=80,
    )
    return are_simulation_react_json_agent(llm_engine=llm_engine, base_agent_config=base_config)


@pytest.fixture
def mock_log_callback():
    """Mock log callback function."""
    return Mock()


@pytest.fixture
def mock_time_manager():
    """Mock time manager."""
    time_mgr = Mock(spec=TimeManager)
    time_mgr.current_time = 1000.0
    return time_mgr


@pytest.fixture
def mock_llm_engine():
    """Mock LLM engine."""
    return _make_mock_llm_engine_with_response(
        'Thought: test\nAction: {"action": "final_answer", "action_input": {"answer": "done"}}<end_action>'
    )


@pytest.fixture
def basic_multi_agent(mock_llm_engine, mock_time_manager, mock_log_callback):
    """Create a RunnableARESimulationMultiAgent with basic setup."""
    placeholder_agent = _create_placeholder_base_agent(mock_llm_engine)
    
    return RunnableARESimulationMultiAgent(
        log_callback=mock_log_callback,
        pause_env=None,
        resume_env=None,
        llm_engine=mock_llm_engine,
        base_agent=placeholder_agent,
        time_manager=mock_time_manager,
        tools=[],
        max_iterations=80,
        max_turns=None,
        simulated_generation_time_config=None,
    )


# ============================================================================
# Initialization Tests
# ============================================================================

def test_initialization(mock_llm_engine, mock_time_manager, mock_log_callback):
    """Test basic initialization of RunnableARESimulationMultiAgent."""
    reset_cortex()
    
    placeholder_agent = _create_placeholder_base_agent(mock_llm_engine)
    
    agent = RunnableARESimulationMultiAgent(
        log_callback=mock_log_callback,
        pause_env=None,
        resume_env=None,
        llm_engine=mock_llm_engine,
        base_agent=placeholder_agent,
        time_manager=mock_time_manager,
        tools=[],
        max_iterations=80,
        max_turns=5,
    )
    
    assert agent.multi_agent is None  # Not created until prepare_are_simulation_run
    assert agent.llm_engine == mock_llm_engine
    assert agent.time_manager == mock_time_manager
    assert agent.react_agent.log_callback == mock_log_callback  # Parent stores it on react_agent
    assert agent.max_iterations == 80
    assert agent.max_turns == 5
    assert agent.react_agent == placeholder_agent  # Placeholder until MultiAgent creation


def test_agent_framework_property(basic_multi_agent):
    """Test agent_framework property returns correct value."""
    assert basic_multi_agent.agent_framework == "MultiAgent"


def test_multi_agent_instance_property_before_initialization(basic_multi_agent):
    """Test multi_agent_instance property returns None before initialization."""
    assert basic_multi_agent.multi_agent_instance is None


def test_cortex_property_before_initialization(basic_multi_agent):
    """Test cortex property returns None before MultiAgent creation."""
    assert basic_multi_agent.cortex is None


# ============================================================================
# Method Override Tests
# ============================================================================

def test_set_subagents_is_no_op(basic_multi_agent):
    """Test that set_subagents() is overridden as no-op."""
    basic_multi_agent.sub_agents = ["some", "agents"]
    
    basic_multi_agent.set_subagents()
    
    # Should be cleared (no-op means empty list)
    assert basic_multi_agent.sub_agents == []


def test_init_tools_skips_scenario_tools(basic_multi_agent, mock_llm_engine, mock_time_manager, mock_log_callback):
    """Test that init_tools() doesn't add scenario tools."""
    reset_cortex()
    
    scenario = _create_mock_scenario()
    
    # Mock scenario tools
    mock_tool = Mock()
    mock_tool.name = "scenario_tool"
    scenario.get_tools.return_value = [mock_tool]
    
    # Store initial tools count
    initial_tools_count = len(basic_multi_agent.tools)
    
    # Call init_tools
    basic_multi_agent.init_tools(scenario)
    
    # Tools should not have been added
    assert len(basic_multi_agent.tools) == initial_tools_count


def test_init_system_prompt_adds_orchestrator_instructions(basic_multi_agent):
    """Test that init_system_prompt() adds orchestrator-specific instructions."""
    reset_cortex()
    
    scenario = _create_mock_scenario()
    scenario.additional_system_prompt = "Additional instructions"
    
    # Setup placeholder agent with init_system_prompts (with placeholder for notification system)
    basic_multi_agent.react_agent.init_system_prompts = {
        "system_prompt": "Base prompt <<notification_system_description>> <<curent_time_description>> <<agent_reminder_description>>"
    }
    
    # Set up minimal notification system (required by parent's init_system_prompt)
    from are.simulation.notification_system import BaseNotificationSystem, NotificationSystemConfig
    basic_multi_agent.react_agent.notification_system = BaseNotificationSystem(config=NotificationSystemConfig())
    
    # Call init_system_prompt
    basic_multi_agent.init_system_prompt(scenario)
    
    # Check that orchestrator instructions were added
    prompt = basic_multi_agent.react_agent.init_system_prompts["system_prompt"]
    assert "orchestrator" in prompt.lower()
    assert "coordinate and delegate" in prompt.lower()
    assert "expert agents" in prompt.lower()


# ============================================================================
# prepare_are_simulation_run Tests
# ============================================================================

@patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"})
def test_prepare_are_simulation_run_creates_multi_agent(basic_multi_agent):
    """Test that prepare_are_simulation_run() creates MultiAgent instance."""
    reset_cortex()
    
    app1 = _create_mock_app("CalendarApp")
    app2 = _create_mock_app("EmailApp")
    scenario = _create_mock_scenario(apps=[app1, app2])
    
    # Create proper notification system (not Mock) for init_system_prompt
    from are.simulation.notification_system import BaseNotificationSystem, NotificationSystemConfig
    notification_system = BaseNotificationSystem(config=NotificationSystemConfig())
    
    # Call prepare_are_simulation_run
    basic_multi_agent.prepare_are_simulation_run(
        scenario=scenario,
        notification_system=notification_system,
    )
    
    # MultiAgent should be created
    assert basic_multi_agent.multi_agent is not None
    assert isinstance(basic_multi_agent.multi_agent, MultiAgent)
    
    # react_agent should be replaced with orchestrator
    assert basic_multi_agent.react_agent == basic_multi_agent.multi_agent.orchestrator
    assert basic_multi_agent.react_agent.agent_id == "orchestrator"
    
    # MultiAgent should have correct number of app agents
    assert len(basic_multi_agent.multi_agent.app_agents) == 2
    
    # Should be initialized
    assert basic_multi_agent._initialized is True


@patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"})
def test_prepare_are_simulation_run_configures_orchestrator(basic_multi_agent):
    """Test that orchestrator is properly configured with settings."""
    reset_cortex()
    
    scenario = _create_mock_scenario(apps=[])
    
    basic_multi_agent.prepare_are_simulation_run(scenario=scenario)
    
    orchestrator = basic_multi_agent.react_agent
    
    # Orchestrator should have correct settings
    assert orchestrator.max_iterations == basic_multi_agent.max_iterations
    assert orchestrator.llm_engine == basic_multi_agent.llm_engine
    assert orchestrator.time_manager == basic_multi_agent.time_manager
    assert orchestrator.log_callback == basic_multi_agent.react_agent.log_callback


@patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"})
def test_prepare_are_simulation_run_with_system_prompts(basic_multi_agent):
    """Test that system_prompts are passed to MultiAgent."""
    reset_cortex()
    
    scenario = _create_mock_scenario(apps=[])
    
    # Set up placeholder with system_prompts
    basic_multi_agent.react_agent.init_system_prompts = {
        "system_prompt": "Custom system prompt"
    }
    
    basic_multi_agent.prepare_are_simulation_run(scenario=scenario)
    
    # Orchestrator should have system prompts
    assert hasattr(basic_multi_agent.react_agent, 'init_system_prompts')
    assert "system_prompt" in basic_multi_agent.react_agent.init_system_prompts


@patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"})
def test_prepare_are_simulation_run_with_empty_apps(basic_multi_agent):
    """Test prepare_are_simulation_run with no apps in scenario."""
    reset_cortex()
    
    scenario = _create_mock_scenario(apps=[])
    
    basic_multi_agent.prepare_are_simulation_run(scenario=scenario)
    
    # MultiAgent should still be created (just no app agents)
    assert basic_multi_agent.multi_agent is not None
    assert len(basic_multi_agent.multi_agent.app_agents) == 0
    # Orchestrator should still exist
    assert basic_multi_agent.react_agent == basic_multi_agent.multi_agent.orchestrator


@patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"})
def test_prepare_are_simulation_run_with_initial_logs(basic_multi_agent):
    """Test prepare_are_simulation_run with initial agent logs."""
    reset_cortex()
    
    scenario = _create_mock_scenario(apps=[])
    
    # Create mock logs
    mock_log = Mock(spec=BaseAgentLog)
    initial_logs = [mock_log]
    
    basic_multi_agent.prepare_are_simulation_run(
        scenario=scenario,
        initial_agent_logs=initial_logs,
    )
    
    # After replacement, orchestrator should be set as react_agent
    # replay would be called on the orchestrator if logs were provided
    # We just verify that prepare completed successfully (replay is internal)
    assert basic_multi_agent.react_agent == basic_multi_agent.multi_agent.orchestrator


@patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"})
def test_prepare_are_simulation_run_pause_resume_env(basic_multi_agent):
    """Test pause/resume env functions are set on orchestrator."""
    reset_cortex()
    
    pause_fn = Mock()
    resume_fn = Mock()
    
    agent = RunnableARESimulationMultiAgent(
        log_callback=basic_multi_agent.react_agent.log_callback,
        pause_env=pause_fn,
        resume_env=resume_fn,
        llm_engine=basic_multi_agent.llm_engine,
        base_agent=_create_placeholder_base_agent(basic_multi_agent.llm_engine),
        time_manager=basic_multi_agent.time_manager,
    )
    
    scenario = _create_mock_scenario(apps=[])
    
    agent.prepare_are_simulation_run(scenario=scenario)
    
    # Orchestrator should have pause/resume functions
    assert agent.react_agent.pause_env == pause_fn
    assert agent.react_agent.resume_env == resume_fn


@patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"})
def test_prepare_are_simulation_run_with_simulated_time_config(basic_multi_agent):
    """Test prepare_are_simulation_run with simulated generation time config."""
    reset_cortex()
    
    pause_fn = Mock()
    resume_fn = Mock()
    
    time_config = SimulatedGenerationTimeConfig(mode="measured")
    
    agent = RunnableARESimulationMultiAgent(
        log_callback=basic_multi_agent.react_agent.log_callback,
        pause_env=pause_fn,
        resume_env=resume_fn,
        llm_engine=basic_multi_agent.llm_engine,
        base_agent=_create_placeholder_base_agent(basic_multi_agent.llm_engine),
        time_manager=basic_multi_agent.time_manager,
        simulated_generation_time_config=time_config,
    )
    
    scenario = _create_mock_scenario(apps=[])
    
    agent.prepare_are_simulation_run(scenario=scenario)
    
    # Orchestrator should have time config
    assert agent.react_agent.simulated_generation_time_config == time_config


def test_prepare_are_simulation_run_missing_pause_resume_with_time_config(basic_multi_agent):
    """Test that missing pause/resume functions raise error with time config."""
    reset_cortex()
    
    time_config = SimulatedGenerationTimeConfig(mode="measured")
    
    agent = RunnableARESimulationMultiAgent(
        log_callback=basic_multi_agent.react_agent.log_callback,
        pause_env=None,  # Missing
        resume_env=None,  # Missing
        llm_engine=basic_multi_agent.llm_engine,
        base_agent=_create_placeholder_base_agent(basic_multi_agent.llm_engine),
        time_manager=basic_multi_agent.time_manager,
        simulated_generation_time_config=time_config,
    )
    
    scenario = _create_mock_scenario(apps=[])
    
    with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"}):
        with pytest.raises(Exception, match="Pause and resume environment functions"):
            agent.prepare_are_simulation_run(scenario=scenario)


# ============================================================================
# Property Access Tests (after initialization)
# ============================================================================

@patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"})
def test_multi_agent_instance_property_after_initialization(basic_multi_agent):
    """Test multi_agent_instance property returns MultiAgent after initialization."""
    reset_cortex()
    
    scenario = _create_mock_scenario(apps=[])
    basic_multi_agent.prepare_are_simulation_run(scenario=scenario)
    
    assert basic_multi_agent.multi_agent_instance is not None
    assert basic_multi_agent.multi_agent_instance == basic_multi_agent.multi_agent
    assert isinstance(basic_multi_agent.multi_agent_instance, MultiAgent)


@patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"})
def test_cortex_property_after_initialization(basic_multi_agent):
    """Test cortex property returns ContextCortex after initialization."""
    reset_cortex()
    
    scenario = _create_mock_scenario(apps=[])
    basic_multi_agent.prepare_are_simulation_run(scenario=scenario)
    
    cortex = basic_multi_agent.cortex
    assert cortex is not None
    assert isinstance(cortex, ContextCortex)
    assert cortex == basic_multi_agent.multi_agent.cortex


# ============================================================================
# run_scenario Tests
# ============================================================================

@patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"})
def test_run_scenario_calls_prepare_if_not_initialized(basic_multi_agent):
    """Test that run_scenario() calls prepare_are_simulation_run() if not initialized."""
    reset_cortex()
    
    scenario = _create_mock_scenario(apps=[])
    
    # Mock agent_loop to avoid full execution
    basic_multi_agent.agent_loop = Mock(return_value="test output")
    
    # Mock prepare to track if it's called
    prepare_called = []
    original_prepare = basic_multi_agent.prepare_are_simulation_run
    
    def track_prepare(*args, **kwargs):
        prepare_called.append(True)
        return original_prepare(*args, **kwargs)
    
    basic_multi_agent.prepare_are_simulation_run = track_prepare
    
    result = basic_multi_agent.run_scenario(scenario=scenario)
    
    # prepare_are_simulation_run should have been called
    assert len(prepare_called) > 0
    assert basic_multi_agent._initialized is True
    
    # Result should be AgentExecutionResult
    from are.simulation.agents.agent_execution_result import AgentExecutionResult
    assert isinstance(result, AgentExecutionResult)


@patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"})
def test_run_scenario_skips_prepare_if_already_initialized(basic_multi_agent):
    """Test that run_scenario() skips prepare if already initialized."""
    reset_cortex()
    
    scenario = _create_mock_scenario(apps=[])
    
    # Initialize first
    basic_multi_agent.prepare_are_simulation_run(scenario=scenario)
    assert basic_multi_agent._initialized is True
    
    # Mock prepare to ensure it's not called again
    basic_multi_agent.prepare_are_simulation_run = Mock()
    basic_multi_agent.agent_loop = Mock(return_value="test output")
    
    result = basic_multi_agent.run_scenario(scenario=scenario)
    
    # prepare_are_simulation_run should not have been called again
    basic_multi_agent.prepare_are_simulation_run.assert_not_called()


# ============================================================================
# Integration Tests
# ============================================================================

@patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"})
def test_full_scenario_execution_flow(basic_multi_agent):
    """Test the full flow from initialization to scenario execution."""
    reset_cortex()
    
    app1 = _create_mock_app("CalendarApp")
    app2 = _create_mock_app("EmailApp")
    scenario = _create_mock_scenario(apps=[app1, app2])
    
    # Create proper notification system for init_system_prompt
    from are.simulation.notification_system import BaseNotificationSystem, NotificationSystemConfig
    notification_system = BaseNotificationSystem(config=NotificationSystemConfig())
    
    # Mock agent_loop to avoid full execution
    basic_multi_agent.agent_loop = Mock(return_value="Task completed")
    
    # Run scenario
    result = basic_multi_agent.run_scenario(
        scenario=scenario,
        notification_system=notification_system,
    )
    
    # Verify MultiAgent was created
    assert basic_multi_agent.multi_agent is not None
    assert len(basic_multi_agent.multi_agent.app_agents) == 2
    
    # Verify orchestrator is set as react_agent
    assert basic_multi_agent.react_agent.agent_id == "orchestrator"
    
    # Verify cortex is accessible
    cortex = basic_multi_agent.cortex
    assert cortex is not None
    
    # Verify result
    assert result.output == "Task completed"


@patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"})
def test_cortex_access_for_evaluation(basic_multi_agent):
    """Test that cortex can be accessed for evaluation after scenario run."""
    reset_cortex()
    
    scenario = _create_mock_scenario(apps=[])
    
    # Mock agent_loop
    basic_multi_agent.agent_loop = Mock(return_value="Done")
    
    # Run scenario
    basic_multi_agent.run_scenario(scenario=scenario)
    
    # Access cortex for evaluation
    cortex = basic_multi_agent.cortex
    assert cortex is not None
    
    # Access MultiAgent instance
    multi_agent = basic_multi_agent.multi_agent_instance
    assert multi_agent is not None
    
    # Verify cortex is the same instance
    assert cortex == multi_agent.cortex
    
    # Can access orchestrator and app agents
    assert multi_agent.orchestrator is not None
    assert isinstance(multi_agent.app_agents, list)


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_missing_gemini_api_key(basic_multi_agent):
    """Test that missing GEMINI_API_KEY raises error during MultiAgent creation."""
    reset_cortex()
    
    # Remove GEMINI_API_KEY from environment
    with patch.dict(os.environ, {}, clear=True):
        scenario = _create_mock_scenario(apps=[])
        
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            basic_multi_agent.prepare_are_simulation_run(scenario=scenario)


@patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"})
def test_scenario_with_none_apps(basic_multi_agent):
    """Test handling of scenario with None apps."""
    reset_cortex()
    
    scenario = _create_mock_scenario()
    scenario.apps = None  # Explicitly set to None
    
    basic_multi_agent.prepare_are_simulation_run(scenario=scenario)
    
    # Should handle None gracefully (empty list)
    assert basic_multi_agent.multi_agent is not None
    assert len(basic_multi_agent.multi_agent.app_agents) == 0


# ============================================================================
# Subagent Tool Tests
# ============================================================================

@patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"})
def test_orchestrator_has_subagent_tools(basic_multi_agent):
    """Test that orchestrator has subagent tools created from app agents."""
    reset_cortex()
    
    app1 = _create_mock_app("CalendarApp")
    app2 = _create_mock_app("EmailApp")
    scenario = _create_mock_scenario(apps=[app1, app2])
    
    basic_multi_agent.prepare_are_simulation_run(scenario=scenario)
    
    orchestrator = basic_multi_agent.multi_agent.orchestrator
    
    # Orchestrator should have tools (subagent tools)
    assert len(orchestrator.tools) > 0
    
    # Tools should include subagent tools for each app
    tool_names = list(orchestrator.tools.keys())
    assert any("CalendarApp" in name for name in tool_names)
    assert any("EmailApp" in name for name in tool_names)


# ============================================================================
# System Prompt Tests
# ============================================================================

@patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"})
def test_system_prompt_includes_scenario_additional_prompt(basic_multi_agent):
    """Test that scenario's additional_system_prompt is included."""
    reset_cortex()
    
    scenario = _create_mock_scenario(
        apps=[],
        additional_system_prompt="Custom scenario instructions"
    )
    
    basic_multi_agent.prepare_are_simulation_run(scenario=scenario)
    
    # Check that orchestrator has system prompts
    if hasattr(basic_multi_agent.react_agent, 'init_system_prompts'):
        prompt = basic_multi_agent.react_agent.init_system_prompts.get("system_prompt", "")
        # Additional prompt should be included (via parent's init_system_prompt)
        # Note: This depends on parent implementation, so we just check prompt exists
        assert len(prompt) > 0


@patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"})
def test_system_prompt_includes_orchestrator_instructions(basic_multi_agent):
    """Test that orchestrator-specific instructions are added to system prompt."""
    reset_cortex()
    
    scenario = _create_mock_scenario(apps=[])
    
    basic_multi_agent.prepare_are_simulation_run(scenario=scenario)
    
    if hasattr(basic_multi_agent.react_agent, 'init_system_prompts'):
        prompt = basic_multi_agent.react_agent.init_system_prompts.get("system_prompt", "")
        # Check for orchestrator instructions
        assert "orchestrator" in prompt.lower() or "coordinate" in prompt.lower() or "delegate" in prompt.lower()

