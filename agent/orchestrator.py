from typing import Any, Dict

from are.simulation.agents.default_agent.base_cortex_agent import BaseCortexAgent
from are.simulation.tools import Tool

from cortex.context_cortex import ContextCortex
from cortex.cortex_agent import CortexAgent

class OrchestratorAgent(BaseCortexAgent):
    def __init__(
        self,
        cortex: ContextCortex,
        cortex_agent: CortexAgent,
        agent_id: str,
        agent_mask: int,
        tools: Dict[str, Tool],
        **kwargs: Any,
    ):
        super().__init__(
            cortex=cortex,
            cortex_agent=cortex_agent,
            agent_id=agent_id,
            agent_mask=agent_mask,
            tools=tools,
            **kwargs
        )