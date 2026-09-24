import pytest
from openai import OpenAI
from dotenv import load_dotenv
import os
from app.infrastructure.llm import OpenAILLMService

load_dotenv()

@pytest.mark.integration
def test_openai_llm_generate():
    service = OpenAILLMService(
        client=OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
    )

    result = service.generate(
        system_prompt=(
            "Answer briefly. Return only one sentence."
        ),
        user_prompt=(
            "What is self-attention in a Transformer?"
        ),
    )

    assert isinstance(result, str)
    assert result.strip()