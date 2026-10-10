from hashlib import sha256
import logging
from pathlib import Path

from app.db.models.paper import Paper
from app.infrastructure.arxiv import ArxivService
from app.infrastructure.chunker import Chunker
from app.infrastructure.document_processor import DocumentProcessor
from app.infrastructure.embeddings import EmbeddingService
from app.infrastructure.vector_store import VectorStore
from app.repositories.paper import PaperRepository


logger = logging.getLogger(__name__)


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

    def ingest(self, *, arxiv_id: str) -> Paper:
        normalized_arxiv_id = arxiv_id.strip()
        if not normalized_arxiv_id:
            raise ValueError("arxiv_id must not be empty")

        existing_paper = self.paper_repository.get_by_arxiv_id(
            normalized_arxiv_id
        )
        if existing_paper is not None and existing_paper.status != "failed":
            self._log_duplicate(existing_paper)
            return existing_paper

        if existing_paper is None:
            paper = self.paper_repository.create(
                source_type="arxiv",
                arxiv_id=normalized_arxiv_id,
                title=normalized_arxiv_id,
                status="pending",
            )
        else:
            paper = existing_paper
            self.vector_store.delete_paper(paper_id=paper.paper_id)
            self._log_retry(paper)
        self.paper_repository.set_ingestion_status(
            paper,
            status="processing",
        )
        self._log_started(paper)

        stage = "metadata_lookup"
        try:
            metadata = self.arxiv_service.get_metadata(
                arxiv_id=normalized_arxiv_id
            )
            self.paper_repository.update_metadata(
                paper,
                title=metadata.title,
                abstract=metadata.abstract,
                authors=", ".join(metadata.authors),
                version=metadata.version,
                published_at=metadata.published_at,
                pdf_url=metadata.pdf_url,
            )

            stage = "pdf_download"
            document = self.arxiv_service.download_pdf(
                arxiv_id=normalized_arxiv_id
            )
            stage = "document_processing"
            return self._process_document(
                paper=paper,
                document=document,
            )
        except Exception as error:
            return self._mark_failed(
                paper=paper,
                stage=stage,
                error=error,
            )

    def ingest_pdf(
        self,
        *,
        document: bytes,
        filename: str,
        title: str | None = None,
        abstract: str | None = None,
        authors: str | None = None,
    ) -> Paper:
        if not document:
            raise ValueError("Uploaded PDF must not be empty")
        if not document.lstrip().startswith(b"%PDF-"):
            raise ValueError("Uploaded file is not a valid PDF")

        normalized_filename = Path(filename).name.strip()
        if not normalized_filename:
            raise ValueError("Uploaded PDF must have a filename")

        content_hash = sha256(document).hexdigest()
        existing_paper = self.paper_repository.get_by_content_hash(
            content_hash
        )
        if existing_paper is not None and existing_paper.status != "failed":
            self._log_duplicate(existing_paper)
            return existing_paper

        normalized_title = title.strip() if title is not None else ""
        if not normalized_title:
            normalized_title = Path(normalized_filename).stem.strip()
        if not normalized_title:
            raise ValueError("Paper title must not be empty")

        if existing_paper is None:
            paper = self.paper_repository.create(
                source_type="upload",
                arxiv_id=None,
                content_hash=content_hash,
                original_filename=normalized_filename,
                title=normalized_title,
                abstract=self._normalize_optional(abstract),
                authors=self._normalize_optional(authors),
                status="pending",
            )
        else:
            paper = existing_paper
            self.vector_store.delete_paper(paper_id=paper.paper_id)
            self._log_retry(paper)
        self.paper_repository.set_ingestion_status(
            paper,
            status="processing",
        )
        self._log_started(paper)

        try:
            return self._process_document(
                paper=paper,
                document=document,
            )
        except Exception as error:
            return self._mark_failed(
                paper=paper,
                stage="document_processing",
                error=error,
            )

    def _process_document(
        self,
        *,
        paper: Paper,
        document: bytes,
    ) -> Paper:
        text = self.document_processor.extract_text(document=document)
        chunks = self.chunker.chunk(text=text)
        if not chunks:
            raise ValueError("No text could be extracted from the paper")

        embeddings = self.embedding_service.embed(
            texts=[chunk.text for chunk in chunks]
        )
        if len(embeddings) != len(chunks):
            raise RuntimeError("Embedding count does not match chunk count")

        self.vector_store.add_chunks(
            paper_id=paper.paper_id,
            chunks=chunks,
            embeddings=embeddings,
        )
        self.paper_repository.set_ingestion_status(
            paper,
            status="ready",
        )
        logger.info(
            "Paper ingestion completed",
            extra={
                "event": "paper_ingestion_completed",
                "paper_id": paper.paper_id,
                "source_type": paper.source_type,
                "chunk_count": len(chunks),
            },
        )
        return paper

    def _mark_failed(
        self,
        *,
        paper: Paper,
        stage: str,
        error: Exception,
    ) -> Paper:
        reason = f"Ingestion failed during {stage} ({type(error).__name__})"
        self.paper_repository.set_ingestion_status(
            paper,
            status="failed",
            ingestion_error=reason,
        )
        logger.exception(
            "Paper ingestion failed",
            extra={
                "event": "paper_ingestion_failed",
                "paper_id": paper.paper_id,
                "source_type": paper.source_type,
                "ingestion_stage": stage,
            },
        )
        return paper

    @staticmethod
    def _normalize_optional(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _log_started(paper: Paper) -> None:
        logger.info(
            "Paper ingestion started",
            extra={
                "event": "paper_ingestion_started",
                "paper_id": paper.paper_id,
                "source_type": paper.source_type,
            },
        )

    @staticmethod
    def _log_duplicate(paper: Paper) -> None:
        logger.info(
            "Existing paper reused",
            extra={
                "event": "paper_ingestion_duplicate",
                "paper_id": paper.paper_id,
                "source_type": paper.source_type,
                "status": paper.status,
            },
        )

    @staticmethod
    def _log_retry(paper: Paper) -> None:
        logger.info(
            "Failed paper ingestion retried",
            extra={
                "event": "paper_ingestion_retried",
                "paper_id": paper.paper_id,
                "source_type": paper.source_type,
            },
        )
