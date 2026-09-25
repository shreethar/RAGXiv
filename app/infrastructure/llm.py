from abc import ABC, abstractmethod
from openai import OpenAI

DEFAULT_LLM_MODEL="gpt-5.4-mini-2026-03-17"

class LLMService(ABC):
    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        ...

class OpenAILLMService(LLMService):
    def __init__(
        self,
        *,
        client: OpenAI,
        model: str = DEFAULT_LLM_MODEL,
    ):
        self.client = client
        self.model = model

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        return response.output_text