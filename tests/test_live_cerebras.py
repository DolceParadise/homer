from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from homer.embeddings import QwenEmbeddingProvider
from homer.llm import CerebrasProvider
from homer.pipeline import CorpusEngine


@pytest.mark.live
def test_live_two_book_ingestion_retrieval_and_generation(tmp_path, corpus_files):
    load_dotenv(override=False)
    if not os.getenv("CEREBRAS_API_KEY"):
        pytest.skip("CEREBRAS_API_KEY is not configured")
    provider = CerebrasProvider()
    engine = CorpusEngine(
        corpus="sherlock-live",
        root=tmp_path,
        embedding_provider=QwenEmbeddingProvider(batch_size=8),
        graph_extractor=provider,
        community_summarizer=provider,
        story_writer=provider,
    )
    try:
        stats = engine.ingest(corpus_files, extraction_batch_size=4)
        assert stats.documents == 2
        assert stats.entities > 10
        assert stats.relations > 5

        carbuncle = engine.retrieve(
            "Write a follow-up to The Adventure of the Blue Carbuncle"
        )
        assert any("carbuncle" in item.content.casefold() for item in carbuncle.items)

        result = engine.write(
            "What would have happened if Sherlock Holmes had caught Irene Adler?",
            max_words=350,
        )
        assert len(result.story.split()) >= 100
        assert result.grounding_report
    finally:
        engine.close()
