from __future__ import annotations

from decimal import Context
from typing import Any, Dict, Optional, Union
import os
from dotenv import load_dotenv

load_dotenv()

from are.simulation.agents.agent_log import ToolCallLog
from are.simulation.agents.default_agent.base_agent import (
    BaseAgent,
    TerminationStep,
)
from are.simulation.agents.default_agent.tools.json_action_executor import (
    JsonActionExecutor,
)
from are.simulation.agents.default_agent.prompts.system_prompt import (
    REACT_LOOP_JSON_SYSTEM_PROMPT,
    JSON_AGENT_HINTS,
)
from are.simulation.agents.llm.llm_engine_builder import LLMEngineBuilder
from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
from are.simulation.agents.llm.types import MMObservation
from are.simulation.tools import Tool

from cortex.context_cortex import ContextCortex


class IngestEpisodeTool(Tool):
    """
    Tool used by `CortexAgent` to write a new episode into the shared cortex.

    LLM-visible arguments:
    - trace_summary: concise natural-language summary of the episode.
    - mask_str:      access-control mask as a binary-like string (e.g. "1", "11"),
                     chosen by the CortexAgent itself.

    All structural fields (episode_id, source_agent_id, raw_trace, metadata) are
    injected from the surrounding `CortexAgent` via a pending-context getter.
    """

    def __init__(self, get_pending_context, cortex: ContextCortex) -> None:
        self.name = "ingest_episode"
        self.description = (
            "Ingest the current trace into the shared Cortex. "
            "Call this exactly once per run with a concise trace summary and "
            "the provided access mask string."
        )
        self.inputs = {
            "trace_summary": {
                "type": "string",
                "description": (
                    "Short natural-language summary (2–5 sentences) of the trace "
                    "you were given."
                ),
            },
            "mask_str": {
                "type": "string",
                "description": (
                    "Access-control bitmask string (e.g. '1', '10', '11') that "
                    "determines which agents can see this episode. You, the "
                    "Cortex agent, must decide this mask based on which agents "
                    "should have access."
                ),
            },
        }
        self.output_type = "string"
        super().__init__()
        self._get_pending_context = get_pending_context
        self._cortex = cortex

    def forward(self, trace_summary: str, mask_str: str) -> str:
        ctx = self._get_pending_context()
        if not ctx:
            return "No pending episode context found; nothing was ingested."

        episode_id: str = ctx["episode_id"]
        source_agent_id: str = ctx["source_agent_id"]
        raw_trace: Any = ctx["raw_trace"]
        metadata: Optional[Dict[str, Any]] = ctx.get("metadata")

        episode = self._cortex.ingest_episode(
            episode_id=episode_id,
            source_agent_id=source_agent_id,
            raw_trace=raw_trace,
            trace_summary=trace_summary,
            mask_str=mask_str,
            metadata=metadata,
        )
        return f"Episode {episode.episode_id} ingested into Cortex."


class CortexAgent(BaseAgent):
    """
    Lightweight agent that summarizes traces and ingests them into the Cortex.

    Contract:
    - Caller passes in a structured trace plus episode metadata.
    - The agent summarizes the trace and must call `ingest_episode` exactly once
      with `trace_summary` and `mask_str`.
    - The ingest tool then writes the full episode into the shared `cortex`.
    """

    api_key: str = os.getenv("GEMINI_API_KEY")
    model: str = "gemini-2.0-flash"
    provider: str = "gemini"
    system_prompt: str = ""

    def __init__(self, api_key: str, cortex: ContextCortex):
        self.api_key = api_key
        self.cortex = cortex
        self._pending_ingest: Optional[Dict[str, Any]] = None

        # agent config
        ingest_tool = IngestEpisodeTool(
            get_pending_context=self._get_pending_ingest,
            cortex=self.cortex
        )
        tools: Dict[str, Tool] = {"ingest_episode": ingest_tool}

        engine_config = LLMEngineConfig(
            model_name=self.model,
            provider=self.provider
        )
        llm_engine = LLMEngineBuilder().create_engine(engine_config)

        action_executor = JsonActionExecutor(tools=tools)

        termination_step = TerminationStep(
            condition=self._termination_condition_ingest_episode,
            function=lambda _: None,
            name="cortex_ingest_termination",
        )

        system_prompt = REACT_LOOP_JSON_SYSTEM_PROMPT.format(
            json_agent_hints=JSON_AGENT_HINTS
        )

        super().__init__(
            llm_engine=llm_engine,
            system_prompts={"system_prompt": system_prompt},
            tools=tools,
            action_executor=action_executor,
            max_iterations=2,
            termination_step=termination_step,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pending_ingest(self) -> Optional[Dict[str, Any]]:
        return self._pending_ingest
    
    def _termination_condition_ingest_episode(self, agent: BaseAgent) -> bool:
        """
        Termination condition: terminate immediately after ingest_episode is called successfully.
        """
        tool_call = agent.get_last_log_of_type(ToolCallLog)
        tool_name = tool_call.tool_name if tool_call else ""
        if isinstance(agent.action_executor, JsonActionExecutor):
            if tool_name == "ingest_episode":
                return True
        return agent.iterations >= agent.max_iterations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        episode_id: str,
        source_agent_id: str,
        raw_trace: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Summarize a trace and ingest it into the Cortex via the ingest tool.

        The caller is responsible for:
        - Choosing `episode_id` (unique per episode) and `source_agent_id`.
        - Providing `raw_trace` in any serializable / stringifiable form.
        - Choosing an appropriate access-control mask string; the agent will
          encode this as `mask_str` in the ingest tool call.
        """
        self._pending_ingest = {
            "episode_id": episode_id,
            "source_agent_id": source_agent_id,
            "raw_trace": raw_trace,
            "metadata": metadata or {},
        }

        # format trace and prompt
        trace_repr = str(raw_trace)
        
        # Get list of registered agents from cortex if available
        agents_info = ""
        registered_agents = self.cortex.get_all_agents()
        if registered_agents:
            agents_list = []
            for agent_identity in registered_agents:
                mask_str = ContextCortex.format_mask(agent_identity.mask)
                agents_list.append(f"- {agent_identity.agent_id} (mask: {mask_str}, binary: 0b{mask_str})")
            agents_info = f"""
=== REGISTERED AGENTS ===
The following agents are registered in the system:
{chr(10).join(agents_list)}

Use these agent masks when deciding which agents should see this episode. Match the mask format exactly.
=== END REGISTERED AGENTS ===

"""
        
        task = f"""You are given an agent trace to ingest into the shared Cortex memory.

[TRACE]
{trace_repr}

{agents_info}

Your task:
1. Read and understand the trace.
2. Produce a concise, high-signal summary (2–5 sentences).
3. Decide on an appropriate access-control mask string `mask_str` (binary like "1", "10", "11") indicating which agent groups should see this episode.
4. Call the `ingest_episode` tool EXACTLY ONCE with:
   - trace_summary: your summary
   - mask_str: the mask string you chose

After successfully calling `ingest_episode`, the task is complete. Do not provide any additional output or confirmation.

=== ACCESS MASK GUIDELINES ===
The access mask determines which agents can see this episode. Use bitwise binary strings where each bit position represents an agent group:
- "1" (binary 1, mask 0b1): Only visible to agents in group 1 (bit 0 set)
- "10" (binary 2, mask 0b10): Only visible to agents in group 2 (bit 1 set)
- "11" (binary 3, mask 0b11): Visible to agents in both group 1 AND group 2 (bits 0 and 1 set)
- "101" (binary 5, mask 0b101): Visible to agents in group 1 (bit 0) AND group 3 (bit 2)
- "111" (binary 7, mask 0b111): Visible to agents in groups 1, 2, and 3 (all bits set)
- etc. (you may invent whatever masks you need to use)

How bitwise masks work:
- Each bit position represents a different agent group
- Setting multiple bits makes the episode visible to multiple groups
- An agent can access an episode if their group mask shares at least one bit with the episode mask

Examples:
- Trace from a single agent group → use mask matching that group (e.g., "1" for group 1, "10" for group 2)
- Trace showing collaboration between two groups → use combined mask (e.g., "11" for groups 1 and 2)
- Trace relevant to all groups → use mask with all bits set (e.g., "111" for three groups)

Choose the mask based on which agent groups would benefit from seeing this context. You should be MINIMALIST in your choice of mask; that is, only agents that NEED this information to continue should see the episode.
=== END ACCESS MASK GUIDELINES ===
"""

        try:
            result: Union[str, MMObservation, None] = super().run(task, reset=True)
        finally:
            self._pending_ingest = None

        if isinstance(result, MMObservation):
            return result.content
        if result is None:
            return ""
        return str(result)