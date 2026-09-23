from uuid import uuid4

from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models.paper import Paper
from app.db.models.project import Project
from app.db.models.project_paper import ProjectPaper
from app.db.models.user import User
from app.repositories.project import ProjectRepository


def main():
    with SessionLocal() as session:
        try:
            # 1. Create test user
            user = User(
                name="John Doe",
                password_hash = "asda",
                email=f"repo-test-{uuid4()}@example.com",
            )
            session.add(user)
            session.flush()

            # 2. Create project
            project_repo = ProjectRepository(session)

            project = project_repo.create(
                user_id=user.user_id,
                name="Repository Test Project",
            )

            # 3. Create paper
            paper = Paper(
                arxiv_id=f"repo-test-{uuid4()}",
                title="Repository Test Paper",
                project_count=0,
            )
            session.add(paper)
            session.flush()

            print("Created:")
            print(f"  User:    {user.user_id}")
            print(f"  Project: {project.project_id}")
            print(f"  Paper:   {paper.paper_id}")
            print()

            # 4. Add paper
            added = project_repo.add_paper(
                project.project_id,
                paper.paper_id,
            )

            session.refresh(paper)

            print(f"First add_paper(): {added}")
            print(f"project_count: {paper.project_count}")

            assert added is True
            assert paper.project_count == 1

            # 5. Add paper again
            added_again = project_repo.add_paper(
                project.project_id,
                paper.paper_id,
            )

            session.refresh(paper)

            print(f"Second add_paper(): {added_again}")
            print(f"project_count: {paper.project_count}")

            assert added_again is False
            assert paper.project_count == 1

            # 6. Check has_paper()
            has_paper = project_repo.has_paper(
                project.project_id,
                paper.paper_id,
            )

            print(f"has_paper(): {has_paper}")

            assert has_paper is True

            # 7. Check list_papers()
            papers = project_repo.list_papers(
                project.project_id
            )

            print(f"list_papers(): {len(papers)} paper(s)")

            assert len(papers) == 1
            assert papers[0].paper_id == paper.paper_id

            # 8. Remove paper
            removed = project_repo.remove_paper(
                project.project_id,
                paper.paper_id,
            )

            session.refresh(paper)

            print(f"First remove_paper(): {removed}")
            print(f"project_count: {paper.project_count}")

            assert removed is True
            assert paper.project_count == 0

            # 9. Remove paper again
            removed_again = project_repo.remove_paper(
                project.project_id,
                paper.paper_id,
            )

            session.refresh(paper)

            print(f"Second remove_paper(): {removed_again}")
            print(f"project_count: {paper.project_count}")

            assert removed_again is False
            assert paper.project_count == 0

            # 10. Add again for cascade test
            project_repo.add_paper(
                project.project_id,
                paper.paper_id,
            )

            session.refresh(paper)

            assert paper.project_count == 1

            # 11. Delete project
            project_repo.delete(project.project_id)

            # Verify ProjectPaper was deleted
            association = session.scalar(
                select(ProjectPaper).where(
                    ProjectPaper.project_id == project.project_id,
                    ProjectPaper.paper_id == paper.paper_id,
                )
            )

            assert association is None

            # Verify Paper still exists
            remaining_paper = session.scalar(
                select(Paper).where(
                    Paper.paper_id == paper.paper_id
                )
            )

            assert remaining_paper is not None
            assert remaining_paper.project_count == 0

            print()
            print("All ProjectRepository tests passed.")

            # Clean up test user/paper
            session.delete(remaining_paper)
            session.delete(user)

            session.commit()

        except Exception:
            session.rollback()
            raise


if __name__ == "__main__":
    main()