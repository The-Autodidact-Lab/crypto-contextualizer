"""
Integration tests for ContextCortex that make real API calls to Gemini.

To run these tests:
1. Install google-generativeai: `pip install google-generativeai`
2. Set GEMINI_API_KEY in your environment (or .env file):
   export GEMINI_API_KEY="your-api-key-here"
3. Run with pytest:
   pytest agent/tests/test_context_cortex_integration.py -m integration -v

These tests will be automatically skipped if:
- google-generativeai is not installed
- GEMINI_API_KEY is not set
"""
import os
import uuid

import pytest

from agent.context_cortex import (
    ContextCortex,
    ContextEpisode,
    cortex_add_episode,
    cortex_list_accessible_episodes,
    cortex_update_agent_mask,
    cortex_update_episode_mask,
    make_cortex_after_step_hook,
)
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai

    _HAS_GENAI = True
except Exception:  # pragma: no cover - import error path
    _HAS_GENAI = False


GEMINI_MODEL_NAME = "gemini-2.5-flash"


def _require_gemini() -> None:
    """
    Skip the test if the Google GenAI SDK or GEMINI_API_KEY is not available.
    This keeps the tests honest (no mocks) while allowing them to be skipped
    in environments where live calls are impossible.
    """
    if not _HAS_GENAI:
        pytest.skip("google-generativeai SDK is not installed")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY is not set in environment")
    genai.configure(api_key=api_key)


def _make_gemini_summarizer():
    """
    Create a small-LLM summarizer callable using Gemini 2.5 Flash.

    The callable matches the ContextCortex ingestion API:
        (episode: ContextEpisode) -> summary: str
    """

    _require_gemini()
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    def summarizer(ep: ContextEpisode) -> str:
        prompt = (
            "You are a memory curator for an AI agent.\n\n"
            f"Episode ID: {ep.episode_id}\n"
            f"Source agent: {ep.source_agent_id}\n"
            f"Access mask (binary): {ContextCortex.format_mask(ep.access_mask)}\n\n"
            "Here is the raw trace:\n"
            f"{ep.raw_trace}\n\n"
            "Write a single concise English sentence summarizing this episode."
        )
        response = model.generate_content(prompt)
        return (response.text or "").strip()

    return summarizer


@pytest.mark.integration
def test_end_to_end_add_episode_and_summarize_with_gemini():
    """
    Full workflow:
    - Create cortex.
    - Register agents with overlapping masks.
    - Add an episode via cortex_add_episode with a mask string.
    - Summarize using a real Gemini call.
    - Fetch episodes for an agent and ensure the summary is present.
    """
    _require_gemini()

    cortex = ContextCortex()
    # Agent A: sees only mask 0b01 episodes; B: sees 0b01 and 0b10.
    cortex.register_agent("A", 0b01)
    cortex.register_agent("B", 0b11)

    episode_id = cortex_add_episode(
        cortex,
        source_agent_id="A",
        raw_mask_str="1",  # 0b01, visible to both A and B
        raw_trace_snippet={"messages": ["user: hello", "agent: hi there"]},
        make_episode_id=lambda: f"ep-{uuid.uuid4()}",
        metadata={"scenario": "integration-basic"},
    )

    # Summarize with real Gemini call
    summarizer = _make_gemini_summarizer()
    updated = cortex.ingest_with_llm(episode_id, summarizer)
    assert updated is not None
    assert isinstance(updated.summary, str)
    assert len(updated.summary) > 0

    # Agent A should see the episode
    eps_A = cortex.get_episodes_for_agent("A", include_raw=False)
    assert any(ep.episode_id == episode_id for ep in eps_A)

    # Agent B should also see the episode
    eps_B = cortex.get_episodes_for_agent("B", include_raw=False)
    assert any(ep.episode_id == episode_id for ep in eps_B)


@pytest.mark.integration
def test_after_step_hook_integration_with_gemini_summarizer():
    """
    Workflow:
    - Create cortex and register one agent.
    - Build an after-step hook wired to Gemini summarizer.
    - Call the hook with a synthetic trace.
    - Verify an episode is created and summarized; then fetch via list helper.
    """
    _require_gemini()

    cortex = ContextCortex()
    cortex.register_agent("orchestrator", 0b11)

    def make_episode_id() -> str:
        return f"ep-{uuid.uuid4()}"

    summarizer = _make_gemini_summarizer()

    hook = make_cortex_after_step_hook(
        cortex,
        agent_id="orchestrator",
        default_mask=0b11,
        make_episode_id=make_episode_id,
        llm_summarizer=summarizer,
    )

    # Simulate a small trace representing a single agent step
    trace = {
        "step": 0,
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
        ],
    }

    episode = hook(trace, metadata={"turn": 0})
    assert isinstance(episode, ContextEpisode)
    assert episode.summary is not None
    assert len(episode.summary) > 0

    # The episode should be visible via list helper
    listed = cortex_list_accessible_episodes(cortex, agent_id="orchestrator")
    ids = {item["episode_id"] for item in listed}
    assert episode.episode_id in ids


@pytest.mark.integration
def test_visibility_masks_for_multiple_agents_and_episodes_with_gemini():
    """
    Verify that different agent masks and episode masks interact correctly,
    with real Gemini summaries attached to each episode.
    """
    _require_gemini()

    cortex = ContextCortex()
    # A: 0b01, B: 0b10, C: 0b11
    cortex.register_agent("A", 0b01)
    cortex.register_agent("B", 0b10)
    cortex.register_agent("C", 0b11)

    # Episodes with different access masks
    ep_shared = cortex_add_episode(
        cortex,
        source_agent_id="A",
        raw_mask_str="11",  # visible to C only (0b11 & 0b11), not to A(01) or B(10) individually
        raw_trace_snippet={"messages": ["shared context"]},
        make_episode_id=lambda: "ep-shared",
        metadata={"case": "shared"},
    )
    ep_A_only = cortex_add_episode(
        cortex,
        source_agent_id="A",
        raw_mask_str="1",  # 0b01
        raw_trace_snippet={"messages": ["A-only context"]},
        make_episode_id=lambda: "ep-A",
        metadata={"case": "A-only"},
    )
    ep_B_only = cortex_add_episode(
        cortex,
        source_agent_id="B",
        raw_mask_str="10",  # 0b10
        raw_trace_snippet={"messages": ["B-only context"]},
        make_episode_id=lambda: "ep-B",
        metadata={"case": "B-only"},
    )

    summarizer = _make_gemini_summarizer()
    for eid in (ep_shared, ep_A_only, ep_B_only):
        updated = cortex.ingest_with_llm(eid, summarizer)
        assert updated is not None
        assert isinstance(updated.summary, str)
        assert len(updated.summary) > 0

    # A (0b01) should see ep-A and ep-shared only if there is a shared bit
    eps_A = cortex.get_episodes_for_agent("A", include_raw=False)
    ids_A = {ep.episode_id for ep in eps_A}
    assert "ep-A" in ids_A
    # 0b01 & 0b10 == 0 => A cannot see B-only
    assert "ep-B" not in ids_A
    # 0b01 & 0b11 == 0b01 => A can see ep-shared
    assert "ep-shared" in ids_A

    # B (0b10) should see ep-B and ep-shared
    eps_B = cortex.get_episodes_for_agent("B", include_raw=False)
    ids_B = {ep.episode_id for ep in eps_B}
    assert "ep-B" in ids_B
    assert "ep-A" not in ids_B
    assert "ep-shared" in ids_B

    # C (0b11) should see all three
    eps_C = cortex.get_episodes_for_agent("C", include_raw=False)
    ids_C = {ep.episode_id for ep in eps_C}
    assert {"ep-A", "ep-B", "ep-shared"} <= ids_C


@pytest.mark.integration
def test_visibility_changes_after_episode_mask_update():
    """
    Changing an episode's access mask should immediately change which agents
    can see it.
    """
    _require_gemini()

    cortex = ContextCortex()
    cortex.register_agent("A", 0b01)
    cortex.register_agent("B", 0b10)

    episode_id = cortex_add_episode(
        cortex,
        source_agent_id="A",
        raw_mask_str="1",  # initially visible only to A
        raw_trace_snippet={"messages": ["initial visibility for A only"]},
        make_episode_id=lambda: "ep-vis-change",
        metadata={"case": "episode-mask-update"},
    )

    # Summarize once to ensure LLM path is exercised
    summarizer = _make_gemini_summarizer()
    cortex.ingest_with_llm(episode_id, summarizer)

    # Initial visibility: A sees it, B does not
    ids_A_before = {ep.episode_id for ep in cortex.get_episodes_for_agent("A")}
    ids_B_before = {ep.episode_id for ep in cortex.get_episodes_for_agent("B")}
    assert "ep-vis-change" in ids_A_before
    assert "ep-vis-change" not in ids_B_before

    # Update mask to "10" (0b10), now only B should see it
    ok = cortex_update_episode_mask(
        cortex,
        episode_id="ep-vis-change",
        raw_mask_str="10",
    )
    assert ok is True

    ids_A_after = {ep.episode_id for ep in cortex.get_episodes_for_agent("A")}
    ids_B_after = {ep.episode_id for ep in cortex.get_episodes_for_agent("B")}
    assert "ep-vis-change" not in ids_A_after
    assert "ep-vis-change" in ids_B_after


@pytest.mark.integration
def test_visibility_changes_after_agent_mask_update():
    """
    Changing an agent's mask should immediately change which episodes it can
    access.
    """
    _require_gemini()

    cortex = ContextCortex()
    # A starts with no access to 0b10 episodes
    cortex.register_agent("A", 0b01)

    episode_id = cortex_add_episode(
        cortex,
        source_agent_id="B",
        raw_mask_str="10",  # 0b10
        raw_trace_snippet={"messages": ["B-only context initially"]},
        make_episode_id=lambda: "ep-agent-mask",
        metadata={"case": "agent-mask-update"},
    )

    summarizer = _make_gemini_summarizer()
    cortex.ingest_with_llm(episode_id, summarizer)

    ids_before = {ep.episode_id for ep in cortex.get_episodes_for_agent("A")}
    assert "ep-agent-mask" not in ids_before

    # Now give A access to 0b10 episodes by updating its mask to 0b11
    ok = cortex_update_agent_mask(
        cortex,
        agent_id="A",
        raw_mask_str="11",
    )
    assert ok is True

    ids_after = {ep.episode_id for ep in cortex.get_episodes_for_agent("A")}
    assert "ep-agent-mask" in ids_after



