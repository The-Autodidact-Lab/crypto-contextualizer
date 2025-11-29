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
from typing import Any, Dict, List, Optional


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

    def update_agent_mask(self, agent_id: str, mask_str: str) -> Optional[AgentIdentity]:
        """
        Update an agent's mask. Returns the updated identity, or None if missing.
        """
        identity = self._agents.get(agent_id)
        if identity is None:
            return None
        
        identity.mask = self.parse_mask(mask_str)
        return identity

    def get_agent(self, agent_id: str) -> Optional[AgentIdentity]:
        return self._agents.get(agent_id)

    def get_all_agents(self) -> List[AgentIdentity]:
        """
        Return a list of all registered agents with their IDs and masks.
        
        Returns:
            List of AgentIdentity objects containing agent_id and mask for each registered agent.
        """
        return list(self._agents.values())

    
    # ------------------------------------------------------------------
    # LLM operations
    # ------------------------------------------------------------------
    # raw ingest episode function
    def ingest_episode(
        self,
        episode_id: str,
        source_agent_id: str,
        raw_trace: Any,
        trace_summary: str,
        mask_str: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextEpisode:
        """
        Ingest an episode into the cortex.

        This is the primary write API used by `CortexAgent`'s ingest tool. The LLM
        provides only `trace_summary` and `mask_str`; all other fields come from
        the surrounding agent logic.
        """
        episode = self.add_episode(
            episode_id=episode_id,
            source_agent_id=source_agent_id,
            access_mask=self.parse_mask(mask_str),
            raw_trace=raw_trace,
            summary=trace_summary,
            metadata=metadata,
        )
        return episode

    # ------------------------------------------------------------------
    # Episode storage
    # ------------------------------------------------------------------

    def add_episode(
        self,
        episode_id: str,
        source_agent_id: str,
        access_mask: int,
        raw_trace: Any,
        summary: Optional[str] = None,
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
            summary=summary,
            metadata=metadata or {},
        )
        self._episodes[episode_id] = episode
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

# easy singleton for tool call
cortex = ContextCortex()