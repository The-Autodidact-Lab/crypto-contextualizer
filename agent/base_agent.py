# use this as a template for any custom agents
# this imports from Meta's ARE base agent to make it compatible with other evals
from evals.are.simulation.agents.default_agent.base_agent import BaseAgent, ConditionalStep
from evals.are.simulation.agents.default_agent.tools.json_action_executor import JsonActionExecutor

class BaseAgent(BaseAgent):
    def __init__(self, name: str, model: str, api_key: str):
        super().__init__(name, model, api_key)

    def step(self, query: str) -> str:
        pass

    def run(self, query: str) -> str:
        pass

    