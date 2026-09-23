from uuid import uuid4

from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models.chat import Chat
from app.db.models.message import Message
from app.db.models.project import Project
from app.db.models.user import User
from app.repositories.chat import ChatRepository


def main():
    with SessionLocal() as session:
        try:
            # 1. Create test user
            user = User(
                name="Chat Repository Test User",
                email=f"chat-test-{uuid4()}@example.com",
                password_hash="test-password-hash",
            )

            session.add(user)
            session.flush()

            # 2. Create test project
            project = Project(
                user_id=user.user_id,
                name="Chat Repository Test Project",
            )

            session.add(project)
            session.flush()

            chat_repo = ChatRepository(session)

            # 3. Create Chat
            chat = chat_repo.create(
                project_id=project.project_id,
                title="Test Chat",
            )

            print(f"Created chat: {chat.chat_id}")

            assert chat.chat_id is not None
            assert chat.project_id == project.project_id
            assert chat.title == "Test Chat"

            # 4. Get by ID
            found_chat = chat_repo.get_by_id(
                chat.chat_id
            )

            assert found_chat is not None
            assert found_chat.chat_id == chat.chat_id

            print("get_by_id(): passed")

            # 5. List by project
            chats = chat_repo.list_by_project(
                project.project_id
            )

            assert len(chats) == 1
            assert chats[0].chat_id == chat.chat_id

            print("list_by_project(): passed")

            # 6. Update
            updated_chat = chat_repo.update(
                chat.chat_id,
                title="Updated Test Chat",
            )

            assert updated_chat.title == "Updated Test Chat"

            print("update(): passed")

            # 7. Verify update
            verified_chat = chat_repo.get_by_id(
                chat.chat_id
            )

            assert verified_chat is not None
            assert verified_chat.title == "Updated Test Chat"

            print("update verification: passed")

            # 8. Create messages directly for cascade test
            message_1 = Message(
                chat_id=chat.chat_id,
                role="user",
                content="Hello",
            )

            message_2 = Message(
                chat_id=chat.chat_id,
                role="assistant",
                content="Hello! How can I help?",
            )

            session.add_all([
                message_1,
                message_2,
            ])

            session.flush()

            # Verify messages exist
            messages_before = list(
                session.scalars(
                    select(Message).where(
                        Message.chat_id == chat.chat_id
                    )
                )
            )

            assert len(messages_before) == 2

            print("message creation: passed")

            # 9. Delete Chat
            chat_id = chat.chat_id

            chat_repo.delete(chat_id)

            # 10. Verify Chat deleted
            deleted_chat = chat_repo.get_by_id(chat_id)

            assert deleted_chat is None

            print("delete(): passed")

            # 11. Verify Messages were deleted
            messages_after = list(
                session.scalars(
                    select(Message).where(
                        Message.chat_id == chat_id
                    )
                )
            )

            assert len(messages_after) == 0

            print("message cascade: passed")

            # 12. Nonexistent Chat
            nonexistent_id = uuid4()

            try:
                chat_repo.delete(nonexistent_id)

                assert False, (
                    "Expected ValueError for "
                    "nonexistent chat"
                )

            except ValueError:
                print(
                    "delete(nonexistent): passed"
                )

            # Clean up Project and User
            session.delete(project)
            session.delete(user)

            session.commit()

            print()
            print("All ChatRepository tests passed.")

        except Exception:
            session.rollback()
            raise


if __name__ == "__main__":
    main()