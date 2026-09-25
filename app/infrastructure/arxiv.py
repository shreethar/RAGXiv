from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET

import httpx


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_PDF_URL = "https://arxiv.org/pdf"


@dataclass(frozen=True)
class ArxivSearchResult:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published_at: datetime
    pdf_url: str


@dataclass(frozen=True)
class ArxivPaperMetadata:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    version: str
    published_at: datetime
    pdf_url: str


class ArxivService:
    def __init__(
        self,
        *,
        client: httpx.Client,
    ):
        self.client = client

    def search(
        self,
        *,
        query: str,
    ) -> list[ArxivSearchResult]:
        response = self.client.get(
            ARXIV_API_URL,
            params={
                "search_query": query,
            },
        )

        response.raise_for_status()

        root = ET.fromstring(response.text)

        namespace = "{http://www.w3.org/2005/Atom}"

        entries = root.findall(
            f"{namespace}entry"
        )

        results = []

        for entry in entries:
            metadata = self._parse_entry(entry)

            results.append(
                ArxivSearchResult(
                    arxiv_id=metadata.arxiv_id,
                    title=metadata.title,
                    abstract=metadata.abstract,
                    authors=metadata.authors,
                    published_at=metadata.published_at,
                    pdf_url=metadata.pdf_url,
                )
            )

        return results

    def get_metadata(
        self,
        *,
        arxiv_id: str,
    ) -> ArxivPaperMetadata:
        response = self.client.get(
            ARXIV_API_URL,
            params={
                "id_list": arxiv_id,
                "max_results": 1,
            },
        )

        response.raise_for_status()

        root = ET.fromstring(response.text)

        namespace = "{http://www.w3.org/2005/Atom}"

        entry = root.find(f"{namespace}entry")

        if entry is None:
            raise ValueError(
                f"arXiv paper not found: {arxiv_id}"
            )

        return self._parse_entry(entry)

    def download_pdf(
        self,
        *,
        arxiv_id: str,
    ) -> bytes:
        url = f"{ARXIV_PDF_URL}/{arxiv_id}.pdf"

        response = self.client.get(
            url,
            follow_redirects=True
            )
        response.raise_for_status()

        return response.content
    
    def _parse_entry(
        self,
        entry: ET.Element,
    ) -> ArxivPaperMetadata:
        namespace = "{http://www.w3.org/2005/Atom}"

        id_element = entry.find(f"{namespace}id")
        title_element = entry.find(f"{namespace}title")
        summary_element = entry.find(f"{namespace}summary")
        published_element = entry.find(
            f"{namespace}published"
        )

        if (
            id_element is None
            or title_element is None
            or summary_element is None
            or published_element is None
        ):
            raise ValueError(
                "arXiv response is missing required fields"
            )

        arxiv_id = self._extract_arxiv_id(
            id_element.text or ""
        )

        authors = [
            author.findtext(
                f"{namespace}name",
                default="",
            )
            for author in entry.findall(
                f"{namespace}author"
            )
        ]

        version = self._extract_version(
            id_element.text or ""
        )

        pdf_url = f"{ARXIV_PDF_URL}/{arxiv_id}.pdf"

        published_at = datetime.fromisoformat(
            (published_element.text or "").replace(
                "Z",
                "+00:00",
            )
        )

        return ArxivPaperMetadata(
            arxiv_id=arxiv_id,
            title=" ".join(
                (title_element.text or "").split()
            ),
            abstract=" ".join(
                (summary_element.text or "").split()
            ),
            authors=authors,
            version=version,
            published_at=published_at,
            pdf_url=pdf_url,
        )

    @staticmethod
    def _extract_arxiv_id(value: str) -> str:
        value = value.rstrip("/")

        identifier = value.rsplit("/", 1)[-1]

        if "v" in identifier:
            identifier = identifier.split("v", 1)[0]

        return identifier

    @staticmethod
    def _extract_version(value: str) -> str:
        value = value.rstrip("/")

        identifier = value.rsplit("/", 1)[-1]

        if "v" not in identifier:
            return "v1"

        return "v" + identifier.split("v", 1)[1]