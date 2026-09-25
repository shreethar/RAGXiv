from app.infrastructure.chunker import Chunker
from app.infrastructure.vector_store import DocumentChunk


class TextChunker(Chunker):
    def __init__(
        self,
        *,
        chunk_size: int = 4000,
        overlap: int = 400,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if overlap < 0:
            raise ValueError("overlap must not be negative")

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, *, text: str) -> list[DocumentChunk]:
        if not text:
            return []

        chunks: list[DocumentChunk] = []

        step = self.chunk_size - self.overlap
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(
                start + self.chunk_size,
                len(text),
            )

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            start += step

        return chunks