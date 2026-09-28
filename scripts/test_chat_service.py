from types import SimpleNamespace
from unittest.mock import MagicMock, call
from uuid import uuid4

import pytest

from app.agents.research_agent import ResearchAgentResult, ResearchMessage
from app.domain.citations import CitationReference
from app.services.chat_service import ChatService


@pytest.fixture
def service_context():
    user_id = uuid4()
    project_id = uuid4()
    chat_id = uuid4()
    chat_repository = MagicMock()
    message_repository = MagicMock()
    citation_repository = MagicMock()
    project_repository = MagicMock()
    research_agent = MagicMock()

    chat_repository.get_by_id.return_value = SimpleNamespace(
        chat_id=chat_id,
        project_id=project_id,
    )
    project_repository.get_by_id.return_value = SimpleNamespace(
        project_id=project_id,
        user_id=user_id,
    )

    service = ChatService(
        chat_repository=chat_repository,
        message_repository=message_repository,
        citation_repository=citation_repository,
        project_repository=project_repository,
        research_agent=research_agent,
    )

    return SimpleNamespace(
        service=service,
        user_id=user_id,
        project_id=project_id,
        chat_id=chat_id,
        chat_repository=chat_repository,
        message_repository=message_repository,
        citation_repository=citation_repository,
        project_repository=project_repository,
        research_agent=research_agent,
    )


def _message(*, role: str, content: str):
    return SimpleNamespace(
        message_id=uuid4(),
        role=role,
        content=content,
    )


def test_create_get_and_list_messages_preserve_chat_ownership(
    service_context,
):
    chat = service_context.chat_repository.get_by_id.return_value
    created_chat = SimpleNamespace(
        chat_id=uuid4(),
        project_id=service_context.project_id,
        title="Research chat",
    )
    messages = [_message(role="user", content="Question")]
    service_context.chat_repository.create.return_value = created_chat
    service_context.message_repository.list_by_chat.return_value = messages

    created = service_context.service.create_chat(
        user_id=service_context.user_id,
        project_id=service_context.project_id,
        title="  Research chat  ",
    )
    fetched = service_context.service.get_chat(
        user_id=service_context.user_id,
        chat_id=service_context.chat_id,
    )
    listed = service_context.service.list_messages(
        user_id=service_context.user_id,
        chat_id=service_context.chat_id,
    )

    assert created is created_chat
    assert fetched is chat
    assert listed == messages
    service_context.chat_repository.create.assert_called_once_with(
        project_id=service_context.project_id,
        title="Research chat",
    )
    service_context.message_repository.list_by_chat.assert_called_once_with(
        service_context.chat_id
    )
    service_context.research_agent.invoke.assert_not_called()


def test_create_chat_rejects_project_owned_by_another_user(service_context):
    service_context.project_repository.get_by_id.return_value = (
        SimpleNamespace(
            project_id=service_context.project_id,
            user_id=uuid4(),
        )
    )

    with pytest.raises(ValueError, match="Chat does not belong to user"):
        service_context.service.create_chat(
            user_id=service_context.user_id,
            project_id=service_context.project_id,
            title="Research chat",
        )

    service_context.chat_repository.create.assert_not_called()


def test_send_message_persists_conversation_and_passes_ordered_history(
    service_context,
):
    previous_user = _message(
        role="user",
        content="How does self-attention work in robotics?",
    )
    previous_assistant = _message(
        role="assistant",
        content="It lets each token weigh other relevant tokens.",
    )
    current_user = _message(
        role="user",
        content="What about its computational cost?",
    )
    final_assistant = _message(
        role="assistant",
        content="Its standard cost is quadratic in sequence length.",
    )
    service_context.message_repository.create.side_effect = [
        current_user,
        final_assistant,
    ]
    service_context.message_repository.list_by_chat.return_value = [
        previous_user,
        previous_assistant,
        current_user,
    ]
    service_context.research_agent.invoke.return_value = (
        ResearchAgentResult(
            answer=(
                "Its standard cost is quadratic in sequence length."
            )
        )
    )

    result = service_context.service.send_message(
        user_id=service_context.user_id,
        chat_id=service_context.chat_id,
        content="  What about its computational cost?  ",
    )

    assert service_context.message_repository.create.call_args_list == [
        call(
            chat_id=service_context.chat_id,
            role="user",
            content="What about its computational cost?",
        ),
        call(
            chat_id=service_context.chat_id,
            role="assistant",
            content=(
                "Its standard cost is quadratic in sequence length."
            ),
        ),
    ]
    service_context.message_repository.list_by_chat.assert_called_once_with(
        service_context.chat_id
    )

    agent_messages = (
        service_context.research_agent.invoke.call_args.kwargs["messages"]
    )
    assert all(
        isinstance(message, ResearchMessage)
        for message in agent_messages
    )
    assert [message.role for message in agent_messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert [message.content for message in agent_messages] == [
        "How does self-attention work in robotics?",
        "It lets each token weigh other relevant tokens.",
        "What about its computational cost?",
    ]
    assert sum(
        message.content == "What about its computational cost?"
        for message in agent_messages
    ) == 1
    assert result.chat_id == service_context.chat_id
    assert result.user_message_id == current_user.message_id
    assert result.assistant_message_id == final_assistant.message_id
    assert result.answer == final_assistant.content
    assert result.citations == ()
    service_context.citation_repository.create_many.assert_not_called()


def test_missing_chat_does_not_persist_or_invoke_agent(service_context):
    service_context.chat_repository.get_by_id.return_value = None

    with pytest.raises(ValueError, match="Chat not found"):
        service_context.service.send_message(
            user_id=service_context.user_id,
            chat_id=service_context.chat_id,
            content="Hello",
        )

    service_context.message_repository.create.assert_not_called()
    service_context.research_agent.invoke.assert_not_called()


def test_chat_owned_by_another_user_does_not_persist_or_invoke_agent(
    service_context,
):
    service_context.project_repository.get_by_id.return_value = (
        SimpleNamespace(
            project_id=service_context.project_id,
            user_id=uuid4(),
        )
    )

    with pytest.raises(
        ValueError,
        match="Chat does not belong to user",
    ):
        service_context.service.send_message(
            user_id=service_context.user_id,
            chat_id=service_context.chat_id,
            content="Hello",
        )

    service_context.message_repository.create.assert_not_called()
    service_context.research_agent.invoke.assert_not_called()


def test_agent_failure_persists_no_assistant_and_propagates(
    service_context,
):
    current_user = _message(role="user", content="A question")
    service_context.message_repository.create.return_value = current_user
    service_context.message_repository.list_by_chat.return_value = [
        current_user
    ]
    service_context.research_agent.invoke.side_effect = RuntimeError(
        "agent failed"
    )

    with pytest.raises(RuntimeError, match="agent failed"):
        service_context.service.send_message(
            user_id=service_context.user_id,
            chat_id=service_context.chat_id,
            content="A question",
        )

    service_context.message_repository.create.assert_called_once_with(
        chat_id=service_context.chat_id,
        role="user",
        content="A question",
    )
    service_context.citation_repository.create_many.assert_not_called()


def test_selected_citations_are_persisted_for_assistant_message(
    service_context,
):
    current_user = _message(role="user", content="Question")
    assistant = _message(role="assistant", content="Answer")
    reference = CitationReference(
        paper_id=uuid4(),
        chunk_id=uuid4(),
    )
    persisted_citation = SimpleNamespace(citation_id=uuid4())
    service_context.message_repository.create.side_effect = [
        current_user,
        assistant,
    ]
    service_context.message_repository.list_by_chat.return_value = [
        current_user
    ]
    service_context.research_agent.invoke.return_value = ResearchAgentResult(
        answer="Answer",
        citations=(reference,),
    )
    service_context.citation_repository.create_many.return_value = [
        persisted_citation
    ]

    result = service_context.service.send_message(
        user_id=service_context.user_id,
        chat_id=service_context.chat_id,
        content="Question",
    )

    service_context.citation_repository.create_many.assert_called_once_with(
        message_id=assistant.message_id,
        references=(reference,),
    )
    assert result.citations == (persisted_citation,)


def test_get_citation_verifies_ownership_through_message_chat(
    service_context,
):
    citation_id = uuid4()
    citation = SimpleNamespace(
        citation_id=citation_id,
        message_id=uuid4(),
    )
    message = SimpleNamespace(
        message_id=citation.message_id,
        chat_id=service_context.chat_id,
    )
    service_context.citation_repository.get_by_id.return_value = citation
    service_context.message_repository.get_by_id.return_value = message

    result = service_context.service.get_citation(
        user_id=service_context.user_id,
        citation_id=citation_id,
    )

    assert result is citation
    service_context.chat_repository.get_by_id.assert_called_with(
        service_context.chat_id
    )


def test_get_citation_rejects_cross_user_access(service_context):
    citation = SimpleNamespace(
        citation_id=uuid4(),
        message_id=uuid4(),
    )
    service_context.citation_repository.get_by_id.return_value = citation
    service_context.message_repository.get_by_id.return_value = (
        SimpleNamespace(
            message_id=citation.message_id,
            chat_id=service_context.chat_id,
        )
    )
    service_context.project_repository.get_by_id.return_value = (
        SimpleNamespace(
            project_id=service_context.project_id,
            user_id=uuid4(),
        )
    )

    with pytest.raises(ValueError, match="Chat does not belong to user"):
        service_context.service.get_citation(
            user_id=service_context.user_id,
            citation_id=citation.citation_id,
        )


@pytest.mark.parametrize("content", ["", " ", "\n\t"])
def test_empty_message_is_rejected_before_persistence(
    service_context,
    content,
):
    with pytest.raises(
        ValueError,
        match="Message content must not be empty",
    ):
        service_context.service.send_message(
            user_id=service_context.user_id,
            chat_id=service_context.chat_id,
            content=content,
        )

    service_context.chat_repository.get_by_id.assert_not_called()
    service_context.message_repository.create.assert_not_called()
    service_context.research_agent.invoke.assert_not_called()


def test_unsupported_persisted_role_does_not_invoke_agent(
    service_context,
):
    current_user = _message(role="user", content="Hello")
    service_context.message_repository.create.return_value = current_user
    service_context.message_repository.list_by_chat.return_value = [
        _message(role="tool", content="internal result"),
        current_user,
    ]

    with pytest.raises(
        ValueError,
        match="Unsupported persisted message role",
    ):
        service_context.service.send_message(
            user_id=service_context.user_id,
            chat_id=service_context.chat_id,
            content="Hello",
        )

    service_context.research_agent.invoke.assert_not_called()
    service_context.message_repository.create.assert_called_once()
