from abc import ABC, abstractmethod
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

DEFAULT_LLM_MODEL="gpt-5.4-mini-2026-03-17"
StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)

class LLMService(ABC):
    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        ...

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_type: type[StructuredOutput],
    ) -> StructuredOutput:
        raise NotImplementedError(
            "Structured output is not supported by this LLM service"
        )

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

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_type: type[StructuredOutput],
    ) -> StructuredOutput:
        response = self.client.responses.parse(
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
            text_format=output_type,
        )

        if response.output_parsed is None:
            raise RuntimeError(
                "LLM did not return the requested structured output"
            )

        return response.output_parsed
