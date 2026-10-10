from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from dotenv import load_dotenv
import os

load_dotenv()


def get_database_url() -> str | URL:
    configured_url = os.getenv("DATABASE_URL")
    if configured_url:
        return configured_url

    required_variables = {
        "POSTGRES_USER": os.getenv("POSTGRES_USER"),
        "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD"),
        "POSTGRES_DB": os.getenv("POSTGRES_DB"),
        "POSTGRES_HOST": os.getenv("POSTGRES_HOST"),
    }
    missing = [
        name
        for name, value in required_variables.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Database configuration is incomplete; missing: "
            + ", ".join(missing)
        )

    port_value = os.getenv("POSTGRES_PORT", "5432")
    try:
        port = int(port_value)
    except ValueError as error:
        raise RuntimeError("POSTGRES_PORT must be an integer") from error

    return URL.create(
        drivername="postgresql+psycopg",
        username=required_variables["POSTGRES_USER"],
        password=required_variables["POSTGRES_PASSWORD"],
        host=required_variables["POSTGRES_HOST"],
        port=port,
        database=required_variables["POSTGRES_DB"],
    )


DATABASE_URL = get_database_url()
SQL_ECHO = os.getenv("SQL_ECHO", "false").lower() == "true"

engine = create_engine(
    DATABASE_URL,
    echo=SQL_ECHO,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    pass
