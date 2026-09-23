from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models.user import User


class UserRepository:

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, user_id: UUID) -> User | None:
        stmt = select(User).where(
            User.user_id == user_id
        )

        return self.session.scalar(stmt)

    def get_by_email(self, email: str) -> User | None:
        stmt = select(User).where(
            User.email == email
        )

        return self.session.scalar(stmt)

    def create(
        self,
        *,
        name: str,
        email: str,
        password_hash: str,
    ) -> User:
        user = User(
            name=name,
            email=email,
            password_hash=password_hash,
        )

        self.session.add(user)
        self.session.flush()

        return user

    def update(
        self,
        user_id: UUID,
        *,
        name: str,
        email: str,
        password_hash: str,
    ) -> User:
        user = self.get_by_id(user_id)

        if user is None:
            raise ValueError(
                f"User {user_id} not found"
            )

        user.name = name
        user.email = email
        user.password_hash = password_hash

        self.session.flush()

        return user

    def delete(self, user_id: UUID) -> None:
        user = self.get_by_id(user_id)

        if user is None:
            raise ValueError(
                f"User {user_id} not found"
            )

        self.session.delete(user)
        self.session.flush()