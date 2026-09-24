import httpx
import pytest

from app.infrastructure.arxiv import (
    ARXIV_PDF_URL,
    ArxivService,
)


def test_download_pdf():
    pdf_content = b"%PDF-test-content"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == (
            f"{ARXIV_PDF_URL}/2401.12345.pdf"
        )

        return httpx.Response(
            status_code=200,
            content=pdf_content,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as client:
        service = ArxivService(client=client)

        result = service.download_pdf(
            arxiv_id="2401.12345",
        )

    assert result == pdf_content


def test_download_pdf_raises_for_http_error():
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=404,
            content=b"Not found",
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as client:
        service = ArxivService(client=client)

        with pytest.raises(httpx.HTTPStatusError):
            service.download_pdf(
                arxiv_id="does-not-exist",
            )


ATOM_RESPONSE = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
    <entry>
        <id>http://arxiv.org/abs/2401.12345v2</id>
        <title>Retrieval Augmented Generation</title>
        <summary>
            A study of retrieval augmented generation.
        </summary>
        <published>2024-01-15T12:00:00Z</published>
        <author>
            <name>Alice Smith</name>
        </author>
        <author>
            <name>Bob Jones</name>
        </author>
    </entry>
</feed>
"""

def test_get_metadata():
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.url.params["id_list"] == "2401.12345"

        return httpx.Response(
            status_code=200,
            text=ATOM_RESPONSE,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as client:
        service = ArxivService(client=client)

        result = service.get_metadata(
            arxiv_id="2401.12345",
        )

    assert result.arxiv_id == "2401.12345"
    assert result.version == "v2"
    assert result.title == "Retrieval Augmented Generation"
    assert (
        result.abstract
        == "A study of retrieval augmented generation."
    )
    assert result.authors == [
        "Alice Smith",
        "Bob Jones",
    ]
    assert result.pdf_url.endswith(
        "/2401.12345.pdf"
    )

def test_search():
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.url.params["search_query"] == (
            "cat:cs.AI"
        )

        return httpx.Response(
            status_code=200,
            text=ATOM_RESPONSE,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as client:
        service = ArxivService(client=client)

        results = service.search(
            query="cat:cs.AI",
        )

    assert len(results) == 1

    result = results[0]

    assert result.arxiv_id == "2401.12345"
    assert result.title == (
        "Retrieval Augmented Generation"
    )
    assert result.authors == [
        "Alice Smith",
        "Bob Jones",
    ]

def test_get_metadata_raises_when_paper_not_found():
    response = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
    </feed>
    """

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            text=response,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as client:
        service = ArxivService(client=client)

        with pytest.raises(
            ValueError,
            match="paper not found",
        ):
            service.get_metadata(
                arxiv_id="does-not-exist",
            )

            