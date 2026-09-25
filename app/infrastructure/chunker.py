from abc import ABC, abstractmethod

from app.infrastructure.vector_store import DocumentChunk


class Chunker(ABC):
    @abstractmethod
    def chunk(self, *, text: str) -> list[DocumentChunk]:
        ...