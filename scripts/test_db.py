import uuid

from app.db.database import SessionLocal
from app.db.models import (
    Chat,
    Message,
    Paper,
    Project,
    ProjectPaper,
    User,
)


def main():
    db = SessionLocal()

    try:
        # ---------------------------------------------------------
        # 1. Create the objects
        # ---------------------------------------------------------

        user = User(
            name="Test User",
            email=f"test-{uuid.uuid4()}@example.com",
            password_hash="fake-hash",
        )

        project = Project(
            name="Test Project",
            user=user,
        )

        paper = Paper(
            arxiv_id=f"test-{uuid.uuid4()}",
            title="Test Paper",
            abstract="This is a test paper.",
            authors="Test Author",
            version="v1",
            pdf_url="https://arxiv.org/pdf/test",
        )

        project_paper = ProjectPaper(
            project=project,
            paper=paper,
        )

        chat = Chat(
            project=project,
            title="Test Chat",
        )

        message = Message(
            chat=chat,
            role="user",
            content="What is this paper about?",
        )

        db.add(user)
        db.commit()

        # ---------------------------------------------------------
        # 2. Verify IDs were generated
        # ---------------------------------------------------------

        print("\nCreated objects:")
        print(f"User:          {user.user_id}")
        print(f"Project:       {project.project_id}")
        print(f"Paper:         {paper.paper_id}")
        print(f"ProjectPaper:  {project_paper.project_paper_id}")
        print(f"Chat:          {chat.chat_id}")
        print(f"Message:       {message.message_id}")

        # ---------------------------------------------------------
        # 3. Verify relationships
        # ---------------------------------------------------------

        print("\nRelationship checks:")

        print(
            "project.user == user:",
            project.user.user_id == user.user_id,
        )

        print(
            "project_paper.project == project:",
            project_paper.project.project_id == project.project_id,
        )

        print(
            "project_paper.paper == paper:",
            project_paper.paper.paper_id == paper.paper_id,
        )

        print(
            "chat.project == project:",
            chat.project.project_id == project.project_id,
        )

        print(
            "message.chat == chat:",
            message.chat.chat_id == chat.chat_id,
        )

        print(
            "project has 1 paper association:",
            len(project.project_papers) == 1,
        )

        print(
            "project has 1 chat:",
            len(project.chats) == 1,
        )

        print(
            "chat has 1 message:",
            len(chat.messages) == 1,
        )

        # ---------------------------------------------------------
        # 4. Remember IDs for deletion test
        # ---------------------------------------------------------

        user_id = user.user_id
        project_id = project.project_id
        paper_id = paper.paper_id
        chat_id = chat.chat_id
        project_paper_id = project_paper.project_paper_id
        message_id = message.message_id

        # ---------------------------------------------------------
        # 5. Delete the project
        # ---------------------------------------------------------

        print("\nDeleting project...")

        db.delete(project)
        db.commit()

        # ---------------------------------------------------------
        # 6. Verify project-owned data disappeared
        # ---------------------------------------------------------

        remaining_project = db.get(Project, project_id)
        remaining_project_paper = db.get(
            ProjectPaper,
            project_paper_id,
        )
        remaining_chat = db.get(Chat, chat_id)
        remaining_message = db.get(Message, message_id)

        print("\nAfter deleting project:")

        print(
            "Project deleted:",
            remaining_project is None,
        )

        print(
            "ProjectPaper deleted:",
            remaining_project_paper is None,
        )

        print(
            "Chat deleted:",
            remaining_chat is None,
        )

        print(
            "Message deleted:",
            remaining_message is None,
        )

        # ---------------------------------------------------------
        # 7. Verify global Paper survived
        # ---------------------------------------------------------

        remaining_paper = db.get(Paper, paper_id)

        print(
            "Paper survived:",
            remaining_paper is not None,
        )

        # ---------------------------------------------------------
        # 8. Clean up remaining objects
        # ---------------------------------------------------------

        if remaining_paper is not None:
            db.delete(remaining_paper)

        remaining_user = db.get(User, user_id)

        if remaining_user is not None:
            db.delete(remaining_user)

        db.commit()

        print("\nCleanup complete.")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()