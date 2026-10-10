from sqlalchemy.engine import URL

from app.db.database import get_database_url


def test_database_url_can_be_built_from_container_environment(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_USER", "ragxiv")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss:word")
    monkeypatch.setenv("POSTGRES_DB", "ragxiv")
    monkeypatch.setenv("POSTGRES_HOST", "postgres")
    monkeypatch.setenv("POSTGRES_PORT", "5432")

    url = get_database_url()

    assert isinstance(url, URL)
    assert url.username == "ragxiv"
    assert url.password == "p@ss:word"
    assert url.host == "postgres"
    assert url.port == 5432
    assert url.database == "ragxiv"


def test_explicit_database_url_takes_precedence(monkeypatch):
    explicit_url = "postgresql+psycopg://user:password@localhost/database"
    monkeypatch.setenv("DATABASE_URL", explicit_url)

    assert get_database_url() == explicit_url
