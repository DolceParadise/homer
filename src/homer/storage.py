from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Iterable

from qdrant_client import QdrantClient, models as qmodels

from homer.models import (
    CommunitySummary,
    DocumentMetadata,
    RetrievedItem,
    TextChunk,
    VectorRecord,
)


class CorpusPaths:
    def __init__(self, root: Path, corpus: str) -> None:
        safe = "".join(char for char in corpus if char.isalnum() or char in "-_").strip()
        if not safe:
            raise ValueError("Corpus name must contain letters or numbers")
        self.corpus = safe
        self.root = root.resolve() / safe
        self.root.mkdir(parents=True, exist_ok=True)
        self.documents = self.root / "documents.json"
        self.chunks = self.root / "chunks.json"
        self.graph = self.root / "graph.json"
        self.communities = self.root / "communities.json"
        self.state = self.root / "state.json"
        self.cache = self.root / "cache"
        self.cache.mkdir(exist_ok=True)
        self.qdrant = self.root / "qdrant"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


class CorpusStore:
    def __init__(self, paths: CorpusPaths) -> None:
        self.paths = paths

    def documents(self) -> list[DocumentMetadata]:
        return [
            DocumentMetadata.model_validate(value)
            for value in _read_json(self.paths.documents, [])
        ]

    def upsert_documents(self, documents: Iterable[DocumentMetadata]) -> None:
        values = {item.document_id: item for item in self.documents()}
        values.update({item.document_id: item for item in documents})
        ordered = sorted(values.values(), key=lambda item: (item.title, item.path))
        _write_json(
            self.paths.documents,
            [item.model_dump(mode="json") for item in ordered],
        )

    def chunks(self) -> list[TextChunk]:
        return [
            TextChunk.model_validate(value)
            for value in _read_json(self.paths.chunks, [])
        ]

    def upsert_chunks(self, chunks: Iterable[TextChunk]) -> None:
        values = {item.chunk_id: item for item in self.chunks()}
        values.update({item.chunk_id: item for item in chunks})
        ordered = sorted(
            values.values(),
            key=lambda item: (
                item.document_title,
                item.section_order,
                item.chunk_order,
            ),
        )
        _write_json(
            self.paths.chunks,
            [item.model_dump(mode="json") for item in ordered],
        )

    def chunk_map(self) -> dict[str, TextChunk]:
        return {item.chunk_id: item for item in self.chunks()}

    def communities(self) -> list[CommunitySummary]:
        return [
            CommunitySummary.model_validate(value)
            for value in _read_json(self.paths.communities, [])
        ]

    def save_communities(self, values: Iterable[CommunitySummary]) -> None:
        _write_json(
            self.paths.communities,
            [item.model_dump(mode="json") for item in values],
        )

    def processed_chunk_ids(self) -> set[str]:
        state = _read_json(self.paths.state, {})
        return set(state.get("processed_chunk_ids", []))

    def mark_processed(self, chunk_ids: Iterable[str]) -> None:
        state = _read_json(self.paths.state, {})
        processed = set(state.get("processed_chunk_ids", []))
        processed.update(chunk_ids)
        state["processed_chunk_ids"] = sorted(processed)
        _write_json(self.paths.state, state)

    def cache_get(self, namespace: str, key: str) -> dict | None:
        path = self.paths.cache / namespace / f"{key}.json"
        return _read_json(path, None)

    def cache_put(self, namespace: str, key: str, value: dict) -> None:
        path = self.paths.cache / namespace / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, value)


class LocalVectorStore:
    COLLECTION = "homer"

    def __init__(self, path: Path, dimension: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))
        self.dimension = dimension
        if not self.client.collection_exists(self.COLLECTION):
            self.client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        else:
            config = self.client.get_collection(self.COLLECTION)
            vectors = config.config.params.vectors
            existing_size = getattr(vectors, "size", None)
            if existing_size is not None and int(existing_size) != dimension:
                raise ValueError(
                    f"Index dimension is {existing_size}, embedding dimension is {dimension}"
                )

    @staticmethod
    def _point_id(record_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"homer:{record_id}"))

    def upsert(
        self,
        records: list[VectorRecord],
        vectors: list[list[float]],
    ) -> None:
        if len(records) != len(vectors):
            raise ValueError("Records and vectors must have equal length")
        if not records:
            return
        points = [
            qmodels.PointStruct(
                id=self._point_id(record.record_id),
                vector=vector,
                payload={
                    "record_id": record.record_id,
                    "kind": record.kind,
                    "text": record.text,
                    **record.metadata,
                },
            )
            for record, vector in zip(records, vectors, strict=True)
        ]
        self.client.upsert(
            collection_name=self.COLLECTION,
            points=points,
            wait=True,
        )

    def query(
        self,
        vector: list[float],
        limit: int = 12,
        kind: str | None = None,
    ) -> list[RetrievedItem]:
        query_filter = None
        if kind is not None:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="kind",
                        match=qmodels.MatchValue(value=kind),
                    )
                ]
            )
        response = self.client.query_points(
            collection_name=self.COLLECTION,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        items = []
        for point in response.points:
            payload = dict(point.payload or {})
            items.append(
                RetrievedItem(
                    item_id=str(payload.pop("record_id", point.id)),
                    kind=str(payload.pop("kind", "unknown")),
                    content=str(payload.pop("text", "")),
                    score=float(point.score),
                    metadata=payload,
                )
            )
        return items

    def count(self) -> int:
        return int(
            self.client.count(
                collection_name=self.COLLECTION,
                exact=True,
            ).count
        )

    def close(self) -> None:
        self.client.close()
