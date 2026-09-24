import httpx
import pytest

from app.infrastructure.arxiv import ArxivService


@pytest.mark.integration
def test_get_real_arxiv_metadata():
    with httpx.Client(timeout=30.0) as client:
        service = ArxivService(client=client)

        result = service.get_metadata(
            arxiv_id="1706.03762",
        )

    assert result.arxiv_id == "1706.03762"
    assert result.version.startswith("v")
    assert result.title
    assert result.abstract
    assert result.authors
    assert result.pdf_url.endswith(
        "/1706.03762.pdf"
    )