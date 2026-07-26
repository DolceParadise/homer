from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Sequence

import pytest

from homer.models import (
    EntityCandidate,
    EntityType,
    EventCandidate,
    GraphExtraction,
    RelationCandidate,
    RetrievedContext,
    StoryRequest,
    TextChunk,
)


ROOT = Path(__file__).resolve().parents[1]
ADVENTURES = ROOT / "data" / "pg1661-images-3.epub"
SIGN_OF_FOUR = ROOT / "data" / "pg2097-images-3.epub"


@pytest.fixture(scope="session")
def corpus_files() -> list[Path]:
    assert ADVENTURES.is_file()
    assert SIGN_OF_FOUR.is_file()
    return [ADVENTURES, SIGN_OF_FOUR]


class TokenHashEmbeddings:
    model_name = "test/token-hash"
    dimension = 64

    @staticmethod
    def _embed(text: str) -> list[float]:
        vector = [0.0] * TokenHashEmbeddings.dimension
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:2], "big") % len(vector)
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeLiteraryProvider:
    def __init__(self) -> None:
        self.extract_calls = 0
        self.write_context: RetrievedContext | None = None

    def extract(self, chunks: Sequence[TextChunk]) -> GraphExtraction:
        self.extract_calls += 1
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        text = " ".join(chunk.text for chunk in chunks).casefold()
        entities = []
        if "holmes" in text:
            entities.append(
                EntityCandidate(
                    name="Sherlock Holmes",
                    type=EntityType.CHARACTER,
                    aliases=["Holmes", "Sherlock"],
                    description="A consulting detective known for observation and deduction.",
                    traits=["observant", "analytical"],
                    evidence_chunk_ids=chunk_ids,
                )
            )
        if "watson" in text:
            entities.append(
                EntityCandidate(
                    name="Dr. John Watson",
                    type=EntityType.CHARACTER,
                    aliases=["Watson", "Dr. Watson"],
                    description="Holmes's friend, companion, and chronicler.",
                    traits=["loyal"],
                    evidence_chunk_ids=chunk_ids,
                )
            )
        if "irene adler" in text:
            entities.append(
                EntityCandidate(
                    name="Irene Adler",
                    type=EntityType.CHARACTER,
                    aliases=["the woman"],
                    description="An intelligent performer who outwitted Holmes.",
                    traits=["resourceful"],
                    evidence_chunk_ids=chunk_ids,
                )
            )
        relations = []
        names = {entity.name for entity in entities}
        if {"Sherlock Holmes", "Dr. John Watson"} <= names:
            relations.append(
                RelationCandidate(
                    source="Sherlock Holmes",
                    target="Dr. John Watson",
                    type="FRIEND_OF",
                    description="Watson assists Holmes and records his cases.",
                    evidence_chunk_ids=chunk_ids,
                )
            )
        if {"Sherlock Holmes", "Irene Adler"} <= names:
            relations.append(
                RelationCandidate(
                    source="Irene Adler",
                    target="Sherlock Holmes",
                    type="OUTWITTED",
                    description="Adler anticipated Holmes's plan and escaped.",
                    evidence_chunk_ids=chunk_ids,
                )
            )
        return GraphExtraction(
            entities=entities,
            relations=relations,
            events=[
                EventCandidate(
                    name="Narrated investigation",
                    description="Holmes investigates a case described in these passages.",
                    participants=["Sherlock Holmes"] if "Sherlock Holmes" in names else [],
                    locations=[],
                    evidence_chunk_ids=chunk_ids,
                )
            ]
            if entities
            else [],
        )

    def summarize(self, payload: dict) -> tuple[str, str]:
        names = [node["name"] for node in payload["nodes"][:4]]
        return " and ".join(names) or "Literary community", (
            "This community connects " + ", ".join(names) + "."
        )

    def write(self, request: StoryRequest, context: RetrievedContext) -> str:
        self.write_context = context
        return (
            "It was after dusk when Holmes laid the recovered clue upon our table. "
            "His grey eyes held the quiet light that I had learned never to disregard."
        )
