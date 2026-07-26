from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentMetadata(StrictModel):
    document_id: str
    checksum: str
    path: str
    title: str
    author: str | None
    format: str


class DocumentSection(StrictModel):
    document_id: str
    section_id: str
    order: int
    title: str
    text: str
    page: int | None


class ParsedDocument(StrictModel):
    metadata: DocumentMetadata
    sections: list[DocumentSection]


class TextChunk(StrictModel):
    chunk_id: str
    document_id: str
    document_title: str
    author: str | None
    section_id: str
    section_title: str
    section_order: int
    chunk_order: int
    text: str
    page: int | None
    source_path: str


class EntityType(StrEnum):
    CHARACTER = "character"
    LOCATION = "location"
    OBJECT = "object"
    ORGANIZATION = "organization"
    CONCEPT = "concept"
    EVENT = "event"
    OTHER = "other"


class EntityCandidate(StrictModel):
    name: str
    type: EntityType
    aliases: list[str]
    description: str
    traits: list[str]
    evidence_chunk_ids: list[str]


class RelationCandidate(StrictModel):
    source: str
    target: str
    type: str
    description: str
    evidence_chunk_ids: list[str]


class EventCandidate(StrictModel):
    name: str
    description: str
    participants: list[str]
    locations: list[str]
    evidence_chunk_ids: list[str]


class GraphExtraction(StrictModel):
    entities: list[EntityCandidate]
    relations: list[RelationCandidate]
    events: list[EventCandidate]


class CommunitySummary(StrictModel):
    community_id: str
    title: str
    summary: str
    entity_ids: list[str]
    evidence_chunk_ids: list[str]


class VectorRecord(StrictModel):
    record_id: str
    kind: str
    text: str
    metadata: dict[str, Any]


class RetrievedItem(StrictModel):
    item_id: str
    kind: str
    content: str
    score: float
    metadata: dict[str, Any]


class RetrievedContext(StrictModel):
    prompt: str
    items: list[RetrievedItem]
    estimated_tokens: int


class StoryRequest(StrictModel):
    corpus: str
    prompt: str
    max_words: int = Field(default=1800, ge=200, le=5000)


class GroundingReference(StrictModel):
    kind: str
    item_id: str
    document_title: str | None
    section_title: str | None
    chunk_id: str | None


class StoryResult(StrictModel):
    story: str
    grounding_report: list[GroundingReference]


class CorpusStats(StrictModel):
    corpus: str
    documents: int
    chunks: int
    entities: int
    relations: int
    communities: int
    indexed_records: int
    root: Path
