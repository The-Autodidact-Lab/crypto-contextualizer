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

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from termcolor import colored


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
        self.logger = logging.getLogger(__name__)

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
        mask_str = self.format_mask(mask)
        self.logger.debug(
            colored(
                f"[CORTEX] Registered agent '{agent_id}' with mask={mask} (binary: 0b{mask_str})",
                "cyan",
            )
        )
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
    
    def get_episode_count(self) -> int:
        """Return the total number of episodes in the cortex."""
        return len(self._episodes)
    
    def get_all_episodes_info(self) -> List[Dict[str, Any]]:
        """
        Return debug information about all episodes (for logging purposes).
        Returns a list of dicts with episode_id, source_agent_id, and access_mask.
        """
        return [
            {
                "episode_id": ep.episode_id,
                "source_agent_id": ep.source_agent_id,
                "access_mask": ep.access_mask,
                "access_mask_str": self.format_mask(ep.access_mask),
            }
            for ep in self._episodes.values()
        ]

    
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
        access_mask = self.parse_mask(mask_str)
        mask_str_formatted = self.format_mask(access_mask)
        self.logger.info(
            colored(
                f"[CORTEX] Ingesting episode '{episode_id}' from '{source_agent_id}' "
                f"with mask_str='{mask_str}' → mask={access_mask} (binary: 0b{mask_str_formatted})",
                "green",
                attrs=["bold"],
            )
        )
        self.logger.debug(
            colored(
                f"[CORTEX] Episode summary: {trace_summary[:100] if len(trace_summary) > 100 else trace_summary}",
                "green",
            )
        )
        episode = self.add_episode(
            episode_id=episode_id,
            source_agent_id=source_agent_id,
            access_mask=access_mask,
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
        mask_str = self.format_mask(access_mask)
        self.logger.debug(
            colored(
                f"[CORTEX] Stored episode '{episode_id}' from '{source_agent_id}' "
                f"with access_mask={access_mask} (binary: 0b{mask_str}), "
                f"total_episodes={len(self._episodes)}",
                "cyan",
            )
        )
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
        self.logger.debug(
            colored(
                f"[CORTEX] get_episodes_for_agent called for agent_id='{agent_id}', include_raw={include_raw}",
                "cyan",
                attrs=["bold"],
            )
        )
        
        identity = self._agents.get(agent_id)
        if identity is None:
            self.logger.warning(
                colored(
                    f"[CORTEX] ⚠ Agent '{agent_id}' not found in registry! Available agents: {list(self._agents.keys())}",
                    "red",
                    attrs=["bold"],
                )
            )
            return []

        agent_mask_str = self.format_mask(identity.mask)
        self.logger.debug(
            colored(
                f"[CORTEX] Agent '{agent_id}' has mask={identity.mask} (binary: 0b{agent_mask_str})",
                "cyan",
            )
        )

        total_episodes = len(self._episodes)
        self.logger.debug(
            colored(
                f"[CORTEX] Total episodes in cortex: {total_episodes}",
                "cyan",
            )
        )

        accessible: List[ContextEpisode] = []
        for episode in self._episodes.values():
            episode_mask_str = self.format_mask(episode.access_mask)
            has_access = self._has_access(identity.mask, episode.access_mask)
            bitwise_result = identity.mask & episode.access_mask
            
            if has_access:
                self.logger.debug(
                    colored(
                        f"[CORTEX] ✓ Episode '{episode.episode_id}' (source: {episode.source_agent_id}) "
                        f"ACCESSIBLE: agent_mask={identity.mask} (0b{agent_mask_str}) & "
                        f"episode_mask={episode.access_mask} (0b{episode_mask_str}) = {bitwise_result} != 0",
                        "green",
                    )
                )
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
            else:
                self.logger.debug(
                    colored(
                        f"[CORTEX] ✗ Episode '{episode.episode_id}' (source: {episode.source_agent_id}) "
                        f"BLOCKED: agent_mask={identity.mask} (0b{agent_mask_str}) & "
                        f"episode_mask={episode.access_mask} (0b{episode_mask_str}) = {bitwise_result} == 0",
                        "yellow",
                    )
                )
        
        self.logger.info(
            colored(
                f"[CORTEX] Agent '{agent_id}' can access {len(accessible)}/{total_episodes} episode(s)",
                "green" if accessible else "yellow",
                attrs=["bold"],
            )
        )
        
        # Log summary at the end
        if total_episodes > 0:
            self._log_cortex_summary()
        
        return accessible
    
    def _log_cortex_summary(self) -> None:
        """
        Log a summary of all episodes and which agents can access them.
        """
        all_agents = self.get_all_agents()
        all_episodes = list(self._episodes.values())
        
        if not all_episodes:
            return
        
        self.logger.info(
            colored(
                f"[CORTEX] === CORTEX SUMMARY ===",
                "cyan",
                attrs=["bold"],
            )
        )
        self.logger.info(
            colored(
                f"[CORTEX] Total episodes: {len(all_episodes)}, Total agents: {len(all_agents)}",
                "cyan",
            )
        )
        
        # For each agent, show which episodes they can access
        for agent in all_agents:
            agent_mask_str = self.format_mask(agent.mask)
            accessible_episodes = [
                ep for ep in all_episodes
                if self._has_access(agent.mask, ep.access_mask)
            ]
            self.logger.info(
                colored(
                    f"[CORTEX] Agent '{agent.agent_id}' (mask=0b{agent_mask_str}): "
                    f"can access {len(accessible_episodes)}/{len(all_episodes)} episode(s)",
                    "green" if accessible_episodes else "yellow",
                )
            )
            if accessible_episodes:
                episode_list = ", ".join([
                    f"{ep.episode_id}(mask=0b{self.format_mask(ep.access_mask)}, source={ep.source_agent_id})"
                    for ep in accessible_episodes
                ])
                self.logger.debug(
                    colored(
                        f"[CORTEX]   Accessible episodes: {episode_list}",
                        "green",
                    )
                )
        
        # Show all episodes and their access masks
        self.logger.debug(
            colored(
                f"[CORTEX] All episodes:",
                "cyan",
            )
        )
        for ep in all_episodes:
            ep_mask_str = self.format_mask(ep.access_mask)
            accessible_by = [
                agent.agent_id for agent in all_agents
                if self._has_access(agent.mask, ep.access_mask)
            ]
            self.logger.debug(
                colored(
                    f"[CORTEX]   {ep.episode_id}: mask=0b{ep_mask_str}, "
                    f"source={ep.source_agent_id}, accessible_by=[{', '.join(accessible_by)}]",
                    "cyan",
                )
            )
        
        self.logger.info(
            colored(
                f"[CORTEX] === END CORTEX SUMMARY ===",
                "cyan",
                attrs=["bold"],
            )
        )

# easy singleton for tool call
cortex = ContextCortex()