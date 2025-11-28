"""
Context Cortex: shared episodic memory with bitmask-based access control.

This module is intentionally framework-agnostic. It can be wired to ARE agents
and logs, but does not depend on them directly.

Key ideas:
- Every agent has an integer bitmask (`agent_mask`) that encodes which
  context-groups it belongs to.
- Every episode (stored trace segment) has an integer bitmask (`access_mask`)
  that encodes which context-groups should see it.
- Access rule: an agent with mask A can access an episode with mask E iff
  `(A & E) != 0`.

Bitmasks are the *primary language* for access control: the LLM is expected to
reason about and propose masks, typically as binary strings like "1", "11",
"101". Helper functions are provided to parse and format these masks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentIdentity:
    """Represents an agent and its access-control mask."""

    agent_id: str
    mask: int


@dataclass
class ContextEpisode:
    """
    A single episode of context.

    raw_trace should contain the full underlying trace / logs / messages for
    this episode. summary is an optional LLM-produced abstraction.
    """

    episode_id: str
    source_agent_id: str
    access_mask: int
    raw_trace: Any
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core cortex
# ---------------------------------------------------------------------------


class ContextCortex:
    """
    Central store for contextual episodes with bitmask-based access control.

    The cortex itself is model-agnostic. LLMs interact with it via:
    - Mask strings (e.g. "1", "11", "101") parsed to integers.
    - Summarizer / policy callables passed in from the outside.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, AgentIdentity] = {}
        self._episodes: Dict[str, ContextEpisode] = {}

    # ------------------------------------------------------------------
    # Bitmask helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_mask(mask_str: str, base: Optional[int] = None) -> int:
        """
        Parse a mask string into an integer.

        Args:
            mask_str: String representation of the mask.
            base: Optional base for parsing. If None, uses auto-detection:
                - Strings starting with "0b" are always parsed as binary.
                - Otherwise, defaults to binary (2) for backward compatibility.
                - To parse decimal numbers like "10" or "100", explicitly pass base=10.

        Accepted forms (when base=None):
        - Binary-like strings of 0/1, e.g. "1", "11", "101" (parsed as binary)
        - Strings with "0b" prefix, e.g. "0b101" (parsed as binary)
        - Decimal strings, e.g. "5" (parsed as decimal if base=10 is provided)

        Examples:
            parse_mask("101") -> 5 (binary)
            parse_mask("10") -> 2 (binary, ambiguous with decimal 10)
            parse_mask("10", base=10) -> 10 (decimal)
            parse_mask("0b101") -> 5 (binary, explicit prefix)
        """
        s = mask_str.strip()
        if not s:
            raise ValueError("Empty mask string.")

        # If base is explicitly provided, use it
        if base is not None:
            return int(s, base)

        # Auto-detection: "0b" prefix always means binary
        if s.startswith("0b"):
            return int(s, 2)

        # Default to binary for backward compatibility with LLM use cases
        # Note: This means "10" will parse as binary 2, not decimal 10.
        # To parse as decimal, explicitly pass base=10.
        if all(ch in "01" for ch in s):
            return int(s, 2)

        # Fallback: try decimal
        return int(s, 10)

    @staticmethod
    def format_mask(mask: int) -> str:
        """
        Format a mask integer as a minimal binary string without leading zeros.
        """
        if mask <= 0:
            return "0"
        return bin(mask)[2:]

    # ------------------------------------------------------------------
    # Agent registry
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str, mask: int) -> AgentIdentity:
        """
        Register or update an agent's identity and access mask.
        """
        identity = AgentIdentity(agent_id=agent_id, mask=mask)
        self._agents[agent_id] = identity
        return identity

    def update_agent_mask(self, agent_id: str, new_mask: int) -> Optional[AgentIdentity]:
        """
        Update an agent's mask. Returns the updated identity, or None if missing.
        """
        identity = self._agents.get(agent_id)
        if identity is None:
            return None
        identity.mask = new_mask
        return identity

    def set_agent_mask_from_str(self, agent_id: str, mask_str: str) -> Optional[AgentIdentity]:
        """
        Convenience helper to update an agent's mask from a string representation.
        Intended to be called with mask proposals from an LLM.
        """
        new_mask = self.parse_mask(mask_str)
        return self.update_agent_mask(agent_id, new_mask)

    def get_agent(self, agent_id: str) -> Optional[AgentIdentity]:
        return self._agents.get(agent_id)

    # ------------------------------------------------------------------
    # Episode storage
    # ------------------------------------------------------------------

    def add_episode(
        self,
        episode_id: str,
        source_agent_id: str,
        access_mask: int,
        raw_trace: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextEpisode:
        """
        Store a new episode with full raw_trace.
        """
        episode = ContextEpisode(
            episode_id=episode_id,
            source_agent_id=source_agent_id,
            access_mask=access_mask,
            raw_trace=raw_trace,
            metadata=metadata or {},
        )
        self._episodes[episode_id] = episode
        return episode

    def set_episode_mask_from_str(self, episode_id: str, mask_str: str) -> Optional[ContextEpisode]:
        """
        Update an episode's access_mask based on a mask string.
        Intended to be called with mask proposals from an LLM.
        """
        episode = self._episodes.get(episode_id)
        if episode is None:
            return None
        episode.access_mask = self.parse_mask(mask_str)
        return episode

    # ------------------------------------------------------------------
    # LLM curation hooks
    # ------------------------------------------------------------------

    def ingest_with_llm(
        self,
        episode_id: str,
        llm_summarizer: Callable[[ContextEpisode], str],
    ) -> Optional[ContextEpisode]:
        """
        Run a small LLM over an existing episode to curate/summarize it.

        llm_summarizer receives the full ContextEpisode and should return a
        short summary string. The cortex does not import any model clients
        directly; it only calls this function.
        """
        episode = self._episodes.get(episode_id)
        if episode is None:
            return None
        episode.summary = llm_summarizer(episode)
        return episode

    def after_turn_ingest(
        self,
        *,
        agent_id: str,
        turn_trace: Any,
        initial_mask: int,
        make_episode_id: Callable[[], str],
        llm_summarizer: Optional[Callable[[ContextEpisode], str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextEpisode:
        """
        Convenience function to be called after an agent turn.

        - Creates a ContextEpisode containing the full turn_trace.
        - Sets its access_mask to initial_mask.
        - Optionally runs a small LLM summarizer to populate `summary`.

        Example of small-LLM usage pattern (pseudocode):

        >>> def summarizer(ep: ContextEpisode) -> str:
        ...     prompt = f\"\"\"You are a memory curator. Here is a trace:
        ... {ep.raw_trace}
        ...
        ... The agent's mask is {ContextCortex.format_mask(
        ...     cortex.get_agent(ep.source_agent_id).mask
        ... ) if cortex.get_agent(ep.source_agent_id) else 'unknown'}.
        ...
        ... Summarize this trace in at most 3 sentences.\"\"\"
        ...     return call_small_llm(prompt)
        """
        episode_id = make_episode_id()
        episode = self.add_episode(
            episode_id=episode_id,
            source_agent_id=agent_id,
            access_mask=initial_mask,
            raw_trace=turn_trace,
            metadata=metadata,
        )
        if llm_summarizer is not None:
            episode.summary = llm_summarizer(episode)
        return episode

    # ------------------------------------------------------------------
    # Access checking & retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _has_access(agent_mask: int, episode_mask: int) -> bool:
        """
        Core access rule: any shared bit between masks grants access.
        """
        return (agent_mask & episode_mask) != 0

    def get_episodes_for_agent(
        self,
        agent_id: str,
        include_raw: bool = False,
    ) -> List[ContextEpisode]:
        """
        Return all episodes this agent is allowed to see based on masks.

        When include_raw is False, raw_trace is omitted in the returned objects
        to keep them lightweight; the cortex still retains the full trace.
        """
        identity = self._agents.get(agent_id)
        if identity is None:
            return []

        accessible: List[ContextEpisode] = []
        for episode in self._episodes.values():
            if self._has_access(identity.mask, episode.access_mask):
                if include_raw:
                    accessible.append(episode)
                else:
                    accessible.append(
                        ContextEpisode(
                            episode_id=episode.episode_id,
                            source_agent_id=episode.source_agent_id,
                            access_mask=episode.access_mask,
                            raw_trace=None,
                            summary=episode.summary,
                            metadata=episode.metadata,
                        )
                    )
        return accessible


# ---------------------------------------------------------------------------
# Optional helper: cortex \"tools\" and simple integration hook
# ---------------------------------------------------------------------------

def cortex_add_episode(
    cortex: ContextCortex,
    *,
    source_agent_id: str,
    raw_mask_str: str,
    raw_trace_snippet: Any,
    make_episode_id: Callable[[], str],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Tool-style helper to add an episode given a mask string.

    Intended to be exposed to a maintenance agent, which:
    - Proposes raw_mask_str (e.g. \"1\", \"11\", \"101\").
    - Provides a compact raw_trace_snippet (e.g. recent messages or logs).
    """
    access_mask = cortex.parse_mask(raw_mask_str)
    episode_id = make_episode_id()
    cortex.add_episode(
        episode_id=episode_id,
        source_agent_id=source_agent_id,
        access_mask=access_mask,
        raw_trace=raw_trace_snippet,
        metadata=metadata,
    )
    return episode_id


def cortex_update_episode_mask(
    cortex: ContextCortex,
    *,
    episode_id: str,
    raw_mask_str: str,
) -> bool:
    """
    Tool-style helper to update an episode's access mask from a string.
    """
    episode = cortex.set_episode_mask_from_str(episode_id, raw_mask_str)
    return episode is not None


def cortex_update_agent_mask(
    cortex: ContextCortex,
    *,
    agent_id: str,
    raw_mask_str: str,
) -> bool:
    """
    Tool-style helper to update an agent's mask from a string.
    """
    identity = cortex.set_agent_mask_from_str(agent_id, raw_mask_str)
    return identity is not None


def cortex_list_accessible_episodes(
    cortex: ContextCortex,
    *,
    agent_id: str,
) -> List[Dict[str, Any]]:
    """
    Tool-style helper to list episodes visible to an agent.

    Returns a lightweight list of dicts suitable for LLM consumption:
    - episode_id
    - mask_str
    - summary
    """
    out: List[Dict[str, Any]] = []
    episodes = cortex.get_episodes_for_agent(agent_id, include_raw=False)
    for ep in episodes:
        out.append(
            {
                "episode_id": ep.episode_id,
                "mask": ContextCortex.format_mask(ep.access_mask),
                "summary": ep.summary,
            }
        )
    return out


def make_cortex_after_step_hook(
    cortex: ContextCortex,
    *,
    agent_id: str,
    default_mask: int,
    make_episode_id: Callable[[], str],
    llm_summarizer: Optional[Callable[[ContextEpisode], str]] = None,
) -> Callable[[Any, Optional[Dict[str, Any]]], ContextEpisode]:
    """
    Create a small hook that can be called after each agent step/turn.

    The returned function has signature:
        hook(turn_trace, metadata=None) -> ContextEpisode

    and internally calls `cortex.after_turn_ingest(...)`.

    This is designed to be easy to plug into existing event loops or logging
    callbacks without modifying core agent logic.
    """

    def hook(turn_trace: Any, metadata: Optional[Dict[str, Any]] = None) -> ContextEpisode:
        return cortex.after_turn_ingest(
            agent_id=agent_id,
            turn_trace=turn_trace,
            initial_mask=default_mask,
            make_episode_id=make_episode_id,
            llm_summarizer=llm_summarizer,
            metadata=metadata,
        )

    return hook


