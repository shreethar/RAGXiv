from app.db.models.paper import Paper
from app.infrastructure.arxiv import ArxivService
from app.infrastructure.chunker import Chunker
from app.infrastructure.document_processor import DocumentProcessor
from app.infrastructure.embeddings import EmbeddingService
from app.infrastructure.vector_store import VectorStore
from app.repositories.paper import PaperRepository


class PaperService:
    def __init__(
        self,
        *,
        paper_repository: PaperRepository,
        arxiv_service: ArxivService,
        document_processor: DocumentProcessor,
        chunker: Chunker,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ):
        self.paper_repository = paper_repository
        self.arxiv_service = arxiv_service
        self.document_processor = document_processor
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def ingest(
        self,
        *,
        arxiv_id: str,
    ) -> Paper:
        existing_paper = self.paper_repository.get_by_arxiv_id(
            arxiv_id
        )

        if existing_paper is not None:
            return existing_paper

        metadata = self.arxiv_service.get_metadata(
            arxiv_id=arxiv_id
        )

        pdf = self.arxiv_service.download_pdf(
            arxiv_id=arxiv_id
        )

        text = self.document_processor.extract_text(
            document=pdf
        )

        chunks = self.chunker.chunk(
            text=text
        )

        if not chunks:
            raise ValueError(
                f"No text could be extracted from paper: {arxiv_id}"
            )

        embeddings = self.embedding_service.embed(
            texts=[chunk.text for chunk in chunks]
        )

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "Embedding count does not match chunk count"
            )

        paper = self.paper_repository.create(
            arxiv_id=metadata.arxiv_id,
            title=metadata.title,
            abstract=metadata.abstract,
            authors=", ".join(metadata.authors),
            version=metadata.version,
            published_at=metadata.published_at,
            pdf_url=metadata.pdf_url,
            status="ready",
        )

        self.vector_store.add_chunks(
            paper_id=paper.paper_id,
            chunks=chunks,
            embeddings=embeddings,
        )

        return paper