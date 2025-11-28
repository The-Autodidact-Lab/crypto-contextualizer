import pytest

from agent.context_cortex import (
    AgentIdentity,
    ContextCortex,
    ContextEpisode,
    cortex_add_episode,
    cortex_list_accessible_episodes,
    cortex_update_agent_mask,
    cortex_update_episode_mask,
    make_cortex_after_step_hook,
)


# ---------------------------------------------------------------------------
# Bitmask helper tests
# ---------------------------------------------------------------------------


def test_parse_mask_binary_plain():
    cortex = ContextCortex()
    assert cortex.parse_mask("1") == 0b1
    assert cortex.parse_mask("11") == 0b11
    assert cortex.parse_mask("101") == 0b101


def test_parse_mask_binary_with_prefix():
    cortex = ContextCortex()
    assert cortex.parse_mask("0b1") == 0b1
    assert cortex.parse_mask("0b101") == 0b101


def test_parse_mask_decimal():
    cortex = ContextCortex()
    # Decimal numbers that don't look like binary parse correctly
    assert cortex.parse_mask("5") == 5
    # Decimal numbers that look like binary need explicit base=10
    assert cortex.parse_mask("10", base=10) == 10
    assert cortex.parse_mask("100", base=10) == 100
    assert cortex.parse_mask("1010", base=10) == 1010


def test_parse_mask_with_explicit_base():
    cortex = ContextCortex()
    # Explicit base=2 for binary
    assert cortex.parse_mask("10", base=2) == 0b10
    assert cortex.parse_mask("101", base=2) == 0b101
    # Explicit base=10 for decimal
    assert cortex.parse_mask("10", base=10) == 10
    assert cortex.parse_mask("100", base=10) == 100
    # Explicit base=16 for hex (if needed)
    assert cortex.parse_mask("FF", base=16) == 255


def test_parse_mask_ambiguous_defaults_to_binary():
    cortex = ContextCortex()
    # Numbers like "10", "100" that could be binary or decimal
    # default to binary for backward compatibility
    assert cortex.parse_mask("10") == 0b10  # binary 2, not decimal 10
    assert cortex.parse_mask("100") == 0b100  # binary 4, not decimal 100


def test_parse_mask_empty_raises():
    cortex = ContextCortex()
    with pytest.raises(ValueError):
        cortex.parse_mask("")


def test_format_mask_basic():
    assert ContextCortex.format_mask(0) == "0"
    assert ContextCortex.format_mask(1) == "1"
    assert ContextCortex.format_mask(3) == "11"
    assert ContextCortex.format_mask(5) == "101"


# ---------------------------------------------------------------------------
# Agent registry tests
# ---------------------------------------------------------------------------


def test_register_and_get_agent():
    cortex = ContextCortex()
    identity = cortex.register_agent("orchestrator", 0b11)
    assert isinstance(identity, AgentIdentity)
    assert identity.agent_id == "orchestrator"
    assert identity.mask == 0b11

    fetched = cortex.get_agent("orchestrator")
    assert fetched is identity


def test_update_agent_mask():
    cortex = ContextCortex()
    cortex.register_agent("agentA", 0b01)
    updated = cortex.update_agent_mask("agentA", 0b10)
    assert updated is not None
    assert updated.mask == 0b10
    assert cortex.get_agent("agentA").mask == 0b10


def test_set_agent_mask_from_str():
    cortex = ContextCortex()
    cortex.register_agent("agentB", 0b01)
    updated = cortex.set_agent_mask_from_str("agentB", "11")
    assert updated is not None
    assert updated.mask == 0b11
    assert cortex.get_agent("agentB").mask == 0b11


def test_update_agent_mask_unknown_returns_none():
    cortex = ContextCortex()
    assert cortex.update_agent_mask("missing", 0b1) is None
    assert cortex.set_agent_mask_from_str("missing", "1") is None


# ---------------------------------------------------------------------------
# Episode storage tests
# ---------------------------------------------------------------------------


def test_add_episode_stores_full_trace_and_metadata():
    cortex = ContextCortex()
    trace = {"messages": ["hello", "world"]}
    metadata = {"turn": 1}
    episode = cortex.add_episode(
        episode_id="ep1",
        source_agent_id="agentX",
        access_mask=0b01,
        raw_trace=trace,
        metadata=metadata,
    )

    assert isinstance(episode, ContextEpisode)
    assert episode.episode_id == "ep1"
    assert episode.source_agent_id == "agentX"
    assert episode.access_mask == 0b01
    assert episode.raw_trace == trace
    assert episode.metadata == metadata


def test_set_episode_mask_from_str_updates_mask():
    cortex = ContextCortex()
    cortex.add_episode(
        episode_id="ep1",
        source_agent_id="agentX",
        access_mask=0b01,
        raw_trace={},
        metadata=None,
    )
    updated = cortex.set_episode_mask_from_str("ep1", "10")
    assert updated is not None
    assert updated.access_mask == 0b10


def test_set_episode_mask_from_str_unknown_returns_none():
    cortex = ContextCortex()
    assert cortex.set_episode_mask_from_str("missing", "1") is None


# ---------------------------------------------------------------------------
# LLM ingestion / summarization tests (unit-level, pure callables)
# ---------------------------------------------------------------------------


def test_ingest_with_llm_applies_summarizer():
    cortex = ContextCortex()
    ep = cortex.add_episode(
        episode_id="ep1",
        source_agent_id="agentX",
        access_mask=0b01,
        raw_trace="raw trace",
        metadata=None,
    )

    def summarizer(episode: ContextEpisode) -> str:
        return f"summary of {episode.raw_trace}"

    updated = cortex.ingest_with_llm("ep1", summarizer)
    assert updated is ep
    assert ep.summary == "summary of raw trace"


def test_ingest_with_llm_unknown_episode_returns_none():
    cortex = ContextCortex()

    def summarizer(_: ContextEpisode) -> str:  # pragma: no cover - not called
        return "unused"

    assert cortex.ingest_with_llm("missing", summarizer) is None


def test_after_turn_ingest_creates_episode_and_summarizes():
    cortex = ContextCortex()

    def make_episode_id() -> str:
        return "ep1"

    def summarizer(episode: ContextEpisode) -> str:
        return f"summary: {episode.raw_trace}"

    episode = cortex.after_turn_ingest(
        agent_id="agentX",
        turn_trace={"tool": "call"},
        initial_mask=0b11,
        make_episode_id=make_episode_id,
        llm_summarizer=summarizer,
        metadata={"turn": 1},
    )

    assert episode.episode_id == "ep1"
    assert episode.source_agent_id == "agentX"
    assert episode.access_mask == 0b11
    assert episode.raw_trace == {"tool": "call"}
    assert episode.metadata == {"turn": 1}
    assert episode.summary == "summary: {'tool': 'call'}"


# ---------------------------------------------------------------------------
# Access control & retrieval tests
# ---------------------------------------------------------------------------


def test_get_episodes_for_agent_filters_by_bitmask():
    cortex = ContextCortex()
    cortex.register_agent("A", 0b01)
    cortex.register_agent("B", 0b11)

    cortex.add_episode("ep1", "A", 0b01, raw_trace="trace1", metadata=None)
    cortex.add_episode("ep2", "B", 0b01, raw_trace="trace2", metadata=None)
    cortex.add_episode("ep3", "B", 0b10, raw_trace="trace3", metadata=None)

    eps_for_A = cortex.get_episodes_for_agent("A", include_raw=True)
    assert {e.episode_id for e in eps_for_A} == {"ep1", "ep2"}

    eps_for_B = cortex.get_episodes_for_agent("B", include_raw=True)
    assert {e.episode_id for e in eps_for_B} == {"ep1", "ep2", "ep3"}


def test_get_episodes_for_agent_excludes_unknown_agents():
    cortex = ContextCortex()
    cortex.add_episode("ep1", "A", 0b01, raw_trace="trace", metadata=None)
    assert cortex.get_episodes_for_agent("missing") == []


def test_get_episodes_for_agent_omit_raw_trace_when_flag_false():
    cortex = ContextCortex()
    cortex.register_agent("A", 0b01)
    cortex.add_episode("ep1", "A", 0b01, raw_trace={"secret": "data"}, metadata=None)

    eps = cortex.get_episodes_for_agent("A", include_raw=False)
    assert len(eps) == 1
    assert eps[0].episode_id == "ep1"
    assert eps[0].raw_trace is None


# ---------------------------------------------------------------------------
# Tool-style helper tests
# ---------------------------------------------------------------------------


def test_cortex_add_episode_uses_mask_string_and_returns_id():
    cortex = ContextCortex()

    def make_episode_id() -> str:
        return "ep1"

    episode_id = cortex_add_episode(
        cortex,
        source_agent_id="agentX",
        raw_mask_str="11",
        raw_trace_snippet="snippet",
        make_episode_id=make_episode_id,
        metadata={"foo": "bar"},
    )
    assert episode_id == "ep1"
    ep = cortex._episodes[episode_id]
    assert ep.access_mask == 0b11
    assert ep.raw_trace == "snippet"
    assert ep.metadata == {"foo": "bar"}


def test_cortex_update_episode_mask_returns_boolean():
    cortex = ContextCortex()
    cortex.add_episode("ep1", "A", 0b01, raw_trace="trace", metadata=None)

    ok = cortex_update_episode_mask(cortex, episode_id="ep1", raw_mask_str="10")
    assert ok is True
    assert cortex._episodes["ep1"].access_mask == 0b10

    fail = cortex_update_episode_mask(cortex, episode_id="missing", raw_mask_str="1")
    assert fail is False


def test_cortex_update_agent_mask_returns_boolean():
    cortex = ContextCortex()
    cortex.register_agent("A", 0b01)

    ok = cortex_update_agent_mask(cortex, agent_id="A", raw_mask_str="11")
    assert ok is True
    assert cortex.get_agent("A").mask == 0b11

    fail = cortex_update_agent_mask(cortex, agent_id="missing", raw_mask_str="1")
    assert fail is False


def test_cortex_list_accessible_episodes_returns_lightweight_dicts():
    cortex = ContextCortex()
    cortex.register_agent("A", 0b01)
    cortex.add_episode("ep1", "A", 0b01, raw_trace="trace1", metadata=None)
    cortex.add_episode("ep2", "A", 0b10, raw_trace="trace2", metadata=None)
    cortex._episodes["ep1"].summary = "summary1"

    result = cortex_list_accessible_episodes(cortex, agent_id="A")
    assert isinstance(result, list)
    assert {"episode_id", "mask", "summary"} <= set(result[0].keys())
    ids = {item["episode_id"] for item in result}
    assert "ep1" in ids and "ep2" not in ids  # A only sees mask 0b01


def test_make_cortex_after_step_hook_creates_episodes():
    cortex = ContextCortex()

    def make_episode_id_factory():
        counter = {"i": 0}

        def make_episode_id():
            counter["i"] += 1
            return f"ep{counter['i']}"

        return make_episode_id

    hook = make_cortex_after_step_hook(
        cortex,
        agent_id="agentX",
        default_mask=0b01,
        make_episode_id=make_episode_id_factory(),
        llm_summarizer=None,
    )

    ep = hook({"msg": "hello"}, metadata={"turn": 1})
    assert ep.episode_id == "ep1"
    assert ep.source_agent_id == "agentX"
    assert ep.access_mask == 0b01
    assert ep.metadata == {"turn": 1}


