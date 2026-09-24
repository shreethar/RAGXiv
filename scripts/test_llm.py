from types import SimpleNamespace
from unittest.mock import Mock

from app.infrastructure.llm import OpenAILLMService


def test_generate_returns_model_output():
    client = Mock()

    client.responses.create.return_value = SimpleNamespace(
        output_text="Self-attention allows each token to attend to other tokens."
    )

    service = OpenAILLMService(
        client=client,
        model="test-model",
    )

    result = service.generate(
        system_prompt="Answer using the provided context.",
        user_prompt="What is self-attention?",
    )

    assert (
        result
        == "Self-attention allows each token to attend to other tokens."
    )

    client.responses.create.assert_called_once_with(
        model="test-model",
        input=[
            {
                "role": "system",
                "content": "Answer using the provided context.",
            },
            {
                "role": "user",
                "content": "What is self-attention?",
            },
        ],
    )

def test_generate_returns_empty_output_when_model_returns_empty_text():
    client = Mock()

    client.responses.create.return_value = SimpleNamespace(
        output_text=""
    )

    service = OpenAILLMService(
        client=client,
        model="test-model",
    )

    result = service.generate(
        system_prompt="System",
        user_prompt="User",
    )

    assert result == ""