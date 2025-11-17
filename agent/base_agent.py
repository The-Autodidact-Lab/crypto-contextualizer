from openai import OpenAI

class BaseAgent:
    def __init__(self, name: str, model: str, api_key: str):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
    
    def step(self, query: str) -> str:
        pass

    def run(self, query: str) -> str:
        pass

    