from dataclasses import dataclass
import logging
from uuid import UUID

from pydantic import BaseModel, Field

from app.db.models.paper import Paper
from app.domain.citations import CitationReference, SourceCitation
from app.infrastructure.llm import LLMService
from app.infrastructure.vector_store import RetrievedChunk, VectorStore
from app.infrastructure.embeddings import EmbeddingService
from app.repositories.paper import PaperRepository


logger = logging.getLogger(__name__)


class RagCitationSelection(BaseModel):
    source_id: str = Field(min_length=1)


class RagAnswerOutput(BaseModel):
    answer: str = Field(min_length=1)
    citations: list[RagCitationSelection] = Field(default_factory=list)


@dataclass(frozen=True)
class RagResult:
    query: str
    answer: str
    chunks: list[RetrievedChunk]
    citations: list[SourceCitation]


class RAGService:
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm_service: LLMService,
        paper_repository: PaperRepository,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.paper_repository = paper_repository

    def query_paper(
        self,
        *,
        question: str,
        paper_id: UUID,
        limit: int = 10,
    ) -> RagResult:
        paper = self.paper_repository.get_by_id(paper_id)

        if paper is None:
            raise ValueError("Paper not found")
        if paper.status != "ready":
            raise ValueError(
                f"Paper is not ready for retrieval: {paper.status}"
            )

        query_embedding = self.embedding_service.embed(
            texts=[question]
        )[0]

        chunks = self.vector_store.search(
            paper_id=paper_id,
            query_embedding=query_embedding,
            limit=limit,
        )

        source_map = {
            f"source_{index}": chunk
            for index, chunk in enumerate(chunks, start=1)
        }
        context = self._build_context(
            paper=paper,
            source_map=source_map,
        )
        logger.info(
            "Paper retrieval completed",
            extra={
                "event": "paper_retrieval_completed",
                "paper_id": paper_id,
                "retrieved_chunk_count": len(chunks),
                "retrieval_limit": limit,
            },
        )

        output = self.llm_service.generate_structured(
            system_prompt=self._system_prompt(),
            user_prompt=self._build_user_prompt(
                question=question,
                context=context,
            ),
            output_type=RagAnswerOutput,
        )

        citations = self._resolve_citations(
            paper_id=paper_id,
            selections=output.citations,
            source_map=source_map,
        )
        logger.info(
            "Grounded paper answer generated",
            extra={
                "event": "paper_answer_generated",
                "paper_id": paper_id,
                "retrieved_chunk_count": len(chunks),
                "citation_count": len(citations),
            },
        )

        return RagResult(
            query=question,
            answer=output.answer,
            chunks=chunks,
            citations=citations,
        )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "Answer questions using only the provided research-paper sources. "
            "Each source has an invocation-local source_id. Return structured "
            "output containing the answer and only the source_ids that directly "
            "support that answer. Cite only source_ids present in the context. "
            "Do not cite a source merely because it was provided. If the sources "
            "do not support an answer, say so and return an empty citations list."
        )

    @staticmethod
    def _build_context(
        *,
        paper: Paper,
        source_map: dict[str, RetrievedChunk],
    ) -> str:
        if not source_map:
            return "No sources were retrieved."

        return "\n\n".join(
            (
                f"[Source ID: {source_id}]\n"
                f"Paper: {paper.title}\n"
                f"Source: {RAGService._paper_source_label(paper)}\n"
                f"Chunk index: {chunk.chunk_index}\n\n"
                f"{chunk.text}"
            )
            for source_id, chunk in source_map.items()
        )

    @staticmethod
    def _paper_source_label(paper: Paper) -> str:
        if paper.arxiv_id is not None:
            return f"arXiv {paper.arxiv_id}"
        return paper.original_filename or "uploaded PDF"

    @staticmethod
    def _build_user_prompt(
        *,
        question: str,
        context: str,
    ) -> str:
        return (
            f"Question:\n{question}\n\n"
            f"Sources:\n{context}"
        )

    @staticmethod
    def _resolve_citations(
        *,
        paper_id: UUID,
        selections: list[RagCitationSelection],
        source_map: dict[str, RetrievedChunk],
    ) -> list[SourceCitation]:
        citations: list[SourceCitation] = []
        seen_source_ids: set[str] = set()

        for selection in selections:
            if selection.source_id in seen_source_ids:
                continue

            chunk = source_map.get(selection.source_id)

            if chunk is None:
                raise ValueError(
                    "LLM returned an unknown citation source: "
                    f"{selection.source_id}"
                )

            seen_source_ids.add(selection.source_id)
            citations.append(
                SourceCitation(
                    source_id=selection.source_id,
                    reference=CitationReference(
                        paper_id=paper_id,
                        chunk_id=chunk.chunk_id,
                    ),
                )
            )

        return citations
