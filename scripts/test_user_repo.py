from uuid import uuid4

from app.db.database import SessionLocal
from app.repositories.user import UserRepository


def main():
    with SessionLocal() as session:
        try:
            user_repo = UserRepository(session)

            email = f"repo-test-{uuid4()}@example.com"

            # 1. Create
            user = user_repo.create(
                name="Repository Test User",
                email=email,
                password_hash="test-password-hash",
            )

            print(f"Created user: {user.user_id}")

            assert user.user_id is not None
            assert user.name == "Repository Test User"
            assert user.email == email
            assert user.password_hash == "test-password-hash"

            # 2. Get by ID
            found_by_id = user_repo.get_by_id(
                user.user_id
            )

            assert found_by_id is not None
            assert found_by_id.user_id == user.user_id

            print("get_by_id(): passed")

            # 3. Get by email
            found_by_email = user_repo.get_by_email(email)

            assert found_by_email is not None
            assert found_by_email.user_id == user.user_id

            print("get_by_email(): passed")

            # 4. Update
            updated_user = user_repo.update(
                user.user_id,
                name="Updated User",
                email=email,
                password_hash="new-password-hash",
            )

            assert updated_user.name == "Updated User"
            assert updated_user.email == email
            assert (
                updated_user.password_hash
                == "new-password-hash"
            )

            print("update(): passed")

            # 5. Verify updated values through database query
            verified_user = user_repo.get_by_id(
                user.user_id
            )

            assert verified_user is not None
            assert verified_user.name == "Updated User"
            assert (
                verified_user.password_hash
                == "new-password-hash"
            )

            print("update verification: passed")

            # 6. Delete
            user_id = user.user_id

            user_repo.delete(user_id)

            print("delete(): passed")

            # 7. Verify deletion
            deleted_user = user_repo.get_by_id(user_id)

            assert deleted_user is None

            print("delete verification: passed")

            # 8. Nonexistent user
            nonexistent_id = uuid4()

            try:
                user_repo.delete(nonexistent_id)
                assert False, (
                    "Expected ValueError for "
                    "nonexistent user"
                )
            except ValueError:
                print(
                    "delete(nonexistent): passed"
                )

            session.commit()

            print()
            print("All UserRepository tests passed.")

        except Exception:
            session.rollback()
            raise


if __name__ == "__main__":
    main()