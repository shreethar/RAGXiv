from dataclasses import dataclass
from uuid import UUID

from app.infrastructure.llm import LLMService
from app.infrastructure.vector_store import RetrievedChunk, VectorStore
from app.infrastructure.embeddings import EmbeddingService


@dataclass(frozen=True)
class RagResult:
    query: str
    answer: str
    chunks: list[RetrievedChunk]


class RAGService:
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm_service: LLMService,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service

    def query_paper(
        self,
        *,
        question: str,
        paper_id: UUID,
        limit: int = 10,
    ) -> RagResult:
        query_embedding = self.embedding_service.embed(
            texts=[question]
        )[0]

        chunks = self.vector_store.search(
            paper_id=paper_id,
            query_embedding=query_embedding,
            limit=limit,
        )

        context = self._build_context(chunks)

        answer = self.llm_service.generate(
            system_prompt=self._system_prompt(),
            user_prompt=self._build_user_prompt(
                question=question,
                context=context,
            ),
        )

        return RagResult(
            query=question,
            answer=answer,
            chunks=chunks,
        )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You answer questions about a research paper using only "
            "the provided context. If the context does not contain "
            "enough information to answer the question, say so."
        )

    @staticmethod
    def _build_context(
        chunks: list[RetrievedChunk],
    ) -> str:
        return "\n\n".join(
            f"[Chunk {chunk.chunk_index}]\n{chunk.text}"
            for chunk in chunks
        )

    @staticmethod
    def _build_user_prompt(
        *,
        question: str,
        context: str,
    ) -> str:
        return (
            f"Question:\n{question}\n\n"
            f"Context:\n{context}"
        )