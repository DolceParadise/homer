from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from homer.chunking import SceneChunker
from homer.embeddings import EmbeddingProvider, QwenEmbeddingProvider
from homer.graph import LiteraryGraph, community_from_payload
from homer.llm import CommunitySummarizer, GraphExtractor, StoryWriter
from homer.models import (
    CommunitySummary,
    CorpusStats,
    GraphExtraction,
    GroundingReference,
    RetrievedContext,
    RetrievedItem,
    StoryRequest,
    StoryResult,
    TextChunk,
    VectorRecord,
)
from homer.parsers import parse_document
from homer.storage import CorpusPaths, CorpusStore, LocalVectorStore


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class CorpusEngine:
    def __init__(
        self,
        corpus: str,
        root: Path = Path(".homer"),
        embedding_provider: EmbeddingProvider | None = None,
        graph_extractor: GraphExtractor | None = None,
        community_summarizer: CommunitySummarizer | None = None,
        story_writer: StoryWriter | None = None,
        chunker: SceneChunker | None = None,
    ) -> None:
        self.paths = CorpusPaths(root, corpus)
        self.corpus = self.paths.corpus
        self.store = CorpusStore(self.paths)
        self.embedding_provider = embedding_provider or QwenEmbeddingProvider()
        self.graph_extractor = graph_extractor
        self.community_summarizer = community_summarizer
        self.story_writer = story_writer
        self.chunker = chunker or SceneChunker()
        self.graph = LiteraryGraph(self.paths.graph)
        self._vectors: LocalVectorStore | None = None

    @property
    def vectors(self) -> LocalVectorStore:
        if self._vectors is None:
            self._vectors = LocalVectorStore(
                self.paths.qdrant,
                self.embedding_provider.dimension,
            )
        return self._vectors

    def close(self) -> None:
        if self._vectors is not None:
            self._vectors.close()
            self._vectors = None

    def _chunk_records(self, chunks: Sequence[TextChunk]) -> list[VectorRecord]:
        return [
            VectorRecord(
                record_id=f"chunk:{chunk.chunk_id}",
                kind="chunk",
                text=(
                    f"{chunk.document_title}\n{chunk.section_title}\n\n{chunk.text}"
                ),
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "document_title": chunk.document_title,
                    "section_id": chunk.section_id,
                    "section_title": chunk.section_title,
                    "page": chunk.page,
                    "source_path": chunk.source_path,
                },
            )
            for chunk in chunks
        ]

    def _index_records(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        batch_size = 24
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            vectors = self.embedding_provider.embed_documents(
                [record.text for record in batch]
            )
            self.vectors.upsert(batch, vectors)

    def _extract_graph(self, chunks: list[TextChunk], batch_size: int) -> None:
        if self.graph_extractor is None:
            return
        processed = self.store.processed_chunk_ids()
        pending = [chunk for chunk in chunks if chunk.chunk_id not in processed]
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            key = _digest("|".join(chunk.chunk_id for chunk in batch))
            cached = self.store.cache_get("extractions", key)
            if cached is None:
                extraction = self.graph_extractor.extract(batch)
                self.store.cache_put(
                    "extractions",
                    key,
                    extraction.model_dump(mode="json"),
                )
            else:
                extraction = GraphExtraction.model_validate(cached)
            self.graph.merge(extraction)
            self.graph.save()
            self.store.mark_processed(chunk.chunk_id for chunk in batch)

    def _summarize_communities(self) -> list[CommunitySummary]:
        if self.community_summarizer is None:
            return self.store.communities()
        summaries = []
        for node_ids in self.graph.communities():
            payload = self.graph.community_payload(node_ids)
            signature = _digest(json.dumps(payload, sort_keys=True))
            cached = self.store.cache_get("communities", signature)
            if cached is None:
                title, summary = self.community_summarizer.summarize(payload)
                value = community_from_payload(
                    community_id=signature[:20],
                    title=title,
                    summary=summary,
                    node_ids=node_ids,
                    payload=payload,
                )
                self.store.cache_put(
                    "communities",
                    signature,
                    value.model_dump(mode="json"),
                )
            else:
                value = CommunitySummary.model_validate(cached)
            summaries.append(value)
        self.store.save_communities(summaries)
        return summaries

    def _index_graph(self, communities: list[CommunitySummary]) -> None:
        records = []
        for node_id, text, metadata in self.graph.entity_records():
            records.append(
                VectorRecord(
                    record_id=f"entity:{node_id}",
                    kind="entity",
                    text=text,
                    metadata={"entity_id": node_id, **metadata},
                )
            )
        for value in communities:
            records.append(
                VectorRecord(
                    record_id=f"community:{value.community_id}",
                    kind="community",
                    text=f"{value.title}\n{value.summary}",
                    metadata={
                        "community_id": value.community_id,
                        "title": value.title,
                        "entity_ids": value.entity_ids,
                        "evidence_chunk_ids": value.evidence_chunk_ids,
                    },
                )
            )
        self._index_records(records)

    def ingest(
        self,
        paths: Sequence[Path],
        extraction_batch_size: int = 4,
    ) -> CorpusStats:
        if extraction_batch_size < 1:
            raise ValueError("extraction_batch_size must be positive")
        parsed = [parse_document(path) for path in paths]
        chunks = [
            chunk
            for document in parsed
            for chunk in self.chunker.chunk(document)
        ]
        existing_ids = set(self.store.chunk_map())
        new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
        self.store.upsert_documents(document.metadata for document in parsed)
        self.store.upsert_chunks(chunks)
        self._index_records(self._chunk_records(new_chunks))
        self._extract_graph(chunks, extraction_batch_size)
        communities = self._summarize_communities()
        self._index_graph(communities)
        return self.inspect()

    @staticmethod
    def _dedupe_rank(items: list[RetrievedItem]) -> list[RetrievedItem]:
        values: dict[str, RetrievedItem] = {}
        for rank, item in enumerate(items):
            rrf = 1.0 / (60 + rank)
            current = values.get(item.item_id)
            adjusted = item.model_copy(
                update={"score": float(item.score) + rrf}
            )
            if current is None or adjusted.score > current.score:
                values[item.item_id] = adjusted
        return sorted(values.values(), key=lambda item: item.score, reverse=True)

    def retrieve(
        self,
        prompt: str,
        limit: int = 14,
        context_tokens: int = 7000,
    ) -> RetrievedContext:
        vector = self.embedding_provider.embed_query(prompt)
        # Search a wider candidate pool so less numerous record types such as
        # community summaries are available for the diversity quotas.
        semantic = [
            *self.vectors.query(vector, limit=max(limit * 6, 80)),
            *self.vectors.query(vector, limit=4, kind="community"),
            *self.vectors.query(vector, limit=4, kind="entity"),
        ]
        seeds = self.graph.match_prompt_entities(prompt)
        seed_communities = [
            RetrievedItem(
                item_id=f"community:{community.community_id}",
                kind="community",
                content=f"{community.title}\n{community.summary}",
                score=1.25,
                metadata={
                    "community_id": community.community_id,
                    "title": community.title,
                    "entity_ids": community.entity_ids,
                    "evidence_chunk_ids": community.evidence_chunk_ids,
                },
            )
            for community in self.store.communities()
            if seeds.intersection(community.entity_ids)
        ]
        graph_items = self.graph.neighborhood_items(seeds, hops=2)
        chunk_map = self.store.chunk_map()
        enriched_graph = []
        for item in graph_items:
            metadata = dict(item.metadata)
            evidence_ids = metadata.get("evidence_chunk_ids", [])
            evidence = next(
                (
                    chunk_map[chunk_id]
                    for chunk_id in evidence_ids
                    if chunk_id in chunk_map
                ),
                None,
            )
            if evidence is not None:
                metadata.update(
                    {
                        "chunk_id": evidence.chunk_id,
                        "document_title": evidence.document_title,
                        "section_title": evidence.section_title,
                        "source_path": evidence.source_path,
                    }
                )
            enriched_graph.append(item.model_copy(update={"metadata": metadata}))

        ranked = self._dedupe_rank(
            [*enriched_graph, *seed_communities, *semantic]
        )
        relation_quota = max(1, min(5, limit // 3))
        chunk_quota = max(1, min(6, limit // 2))
        remaining = max(0, limit - relation_quota - chunk_quota)
        quotas = {
            "relation": relation_quota,
            "chunk": chunk_quota,
            "community": min(1, max(0, remaining - 1)),
            "entity": 1 if remaining else 0,
        }
        diverse = []
        selected_ids = set()
        for kind, quota in quotas.items():
            for item in (value for value in ranked if value.kind == kind):
                if len([value for value in diverse if value.kind == kind]) >= quota:
                    break
                if item.item_id not in selected_ids:
                    diverse.append(item)
                    selected_ids.add(item.item_id)

        selected = []
        characters = 0
        character_budget = context_tokens * 4
        for item in diverse:
            if len(selected) >= limit:
                break
            size = len(item.content)
            if selected and characters + size > character_budget:
                continue
            selected.append(item)
            characters += size
        return RetrievedContext(
            prompt=prompt,
            items=selected,
            estimated_tokens=max(1, characters // 4),
        )

    def write(
        self,
        prompt: str,
        max_words: int = 1800,
        context_tokens: int = 7000,
    ) -> StoryResult:
        if self.story_writer is None:
            raise RuntimeError("No story writer is configured")
        request = StoryRequest(
            corpus=self.corpus,
            prompt=prompt,
            max_words=max_words,
        )
        context = self.retrieve(prompt, context_tokens=context_tokens)
        story = self.story_writer.write(request, context)
        references = []
        for item in context.items:
            metadata = item.metadata
            references.append(
                GroundingReference(
                    kind=item.kind,
                    item_id=item.item_id,
                    document_title=metadata.get("document_title"),
                    section_title=metadata.get("section_title")
                    or metadata.get("title"),
                    chunk_id=metadata.get("chunk_id"),
                )
            )
        return StoryResult(story=story, grounding_report=references)

    def inspect(self) -> CorpusStats:
        return CorpusStats(
            corpus=self.corpus,
            documents=len(self.store.documents()),
            chunks=len(self.store.chunks()),
            entities=self.graph.entity_count,
            relations=self.graph.relation_count,
            communities=len(self.store.communities()),
            indexed_records=(
                self.vectors.count() if self.paths.qdrant.exists() else 0
            ),
            root=self.paths.root,
        )
