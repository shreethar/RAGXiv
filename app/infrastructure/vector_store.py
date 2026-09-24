from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    chunk_index: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    chunk_index: int


class VectorStore(ABC):
    @abstractmethod
    def add_chunks(
        self,
        *,
        paper_id: UUID,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        ...

    @abstractmethod
    def search(
        self,
        *,
        paper_id: UUID,
        query_embedding: list[float],
        limit: int,
    ) -> list[RetrievedChunk]:
        ...

    @abstractmethod
    def delete_paper(
        self,
        *,
        paper_id: UUID,
    ) -> None:
        ...
