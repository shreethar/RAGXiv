from uuid import uuid4

from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models.chat import Chat
from app.db.models.message import Message
from app.db.models.project import Project
from app.db.models.user import User
from app.repositories.message import MessageRepository


def main():
    with SessionLocal() as session:
        try:
            # 1. Create test user
            user = User(
                name="Message Repository Test User",
                email=f"message-test-{uuid4()}@example.com",
                password_hash="test-password-hash",
            )

            session.add(user)
            session.flush()

            # 2. Create project
            project = Project(
                user_id=user.user_id,
                name="Message Repository Test Project",
            )

            session.add(project)
            session.flush()

            # 3. Create chat
            chat = Chat(
                project_id=project.project_id,
                title="Message Repository Test Chat",
            )

            session.add(chat)
            session.flush()

            message_repo = MessageRepository(session)

            # 4. Create first message
            message_1 = message_repo.create(
                chat_id=chat.chat_id,
                role="user",
                content="Hello",
            )

            print(f"Created message: {message_1.message_id}")

            assert message_1.message_id is not None
            assert message_1.chat_id == chat.chat_id
            assert message_1.role == "user"
            assert message_1.content == "Hello"

            print("create(): passed")

            # 5. Create second message
            message_2 = message_repo.create(
                chat_id=chat.chat_id,
                role="assistant",
                content="Hello! How can I help?",
            )

            assert message_2.message_id is not None

            # 6. Get by ID
            found_message = message_repo.get_by_id(
                message_1.message_id
            )

            assert found_message is not None
            assert found_message.message_id == message_1.message_id
            assert found_message.content == "Hello"

            print("get_by_id(): passed")

            # 7. List by chat
            messages = message_repo.list_by_chat(
                chat.chat_id
            )

            assert len(messages) == 2

            assert messages[0].message_id == message_1.message_id
            assert messages[1].message_id == message_2.message_id

            assert messages[0].role == "user"
            assert messages[1].role == "assistant"

            print("list_by_chat(): passed")

            # 8. Delete first message
            message_id = message_1.message_id

            message_repo.delete(message_id)

            deleted_message = message_repo.get_by_id(
                message_id
            )

            assert deleted_message is None

            print("delete(): passed")

            # 9. Verify only second message remains
            remaining_messages = (
                message_repo.list_by_chat(
                    chat.chat_id
                )
            )

            assert len(remaining_messages) == 1
            assert (
                remaining_messages[0].message_id
                == message_2.message_id
            )

            print("delete verification: passed")

            # 10. Nonexistent message
            nonexistent_id = uuid4()

            try:
                message_repo.delete(
                    nonexistent_id
                )

                assert False, (
                    "Expected ValueError for "
                    "nonexistent message"
                )

            except ValueError:
                print(
                    "delete(nonexistent): passed"
                )

            # 11. Verify database state directly
            db_messages = list(
                session.scalars(
                    select(Message).where(
                        Message.chat_id == chat.chat_id
                    )
                )
            )

            assert len(db_messages) == 1
            assert (
                db_messages[0].message_id
                == message_2.message_id
            )

            print("database verification: passed")

            # Clean up
            session.delete(project)
            session.delete(user)

            session.commit()

            print()
            print("All MessageRepository tests passed.")

        except Exception:
            session.rollback()
            raise


if __name__ == "__main__":
    main()