from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class CitationReference:
    paper_id: UUID
    chunk_id: UUID


@dataclass(frozen=True)
class SourceCitation:
    source_id: str
    reference: CitationReference
