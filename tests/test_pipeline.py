from __future__ import annotations

from homer.pipeline import CorpusEngine

from conftest import FakeLiteraryProvider, TokenHashEmbeddings


def test_two_book_ingestion_is_idempotent_and_retrievable(tmp_path, corpus_files):
    provider = FakeLiteraryProvider()
    engine = CorpusEngine(
        corpus="sherlock",
        root=tmp_path,
        embedding_provider=TokenHashEmbeddings(),
        graph_extractor=provider,
        community_summarizer=provider,
        story_writer=provider,
    )
    try:
        first = engine.ingest(corpus_files, extraction_batch_size=6)
        first_calls = provider.extract_calls
        second = engine.ingest(corpus_files, extraction_batch_size=6)

        assert first.documents == 2
        assert first.chunks > 100
        assert first.entities >= 3
        assert first.relations > 0
        assert first.communities > 0
        assert second.chunks == first.chunks
        assert second.entities == first.entities
        assert second.relations == first.relations
        assert provider.extract_calls == first_calls

        context = engine.retrieve(
            "Write a follow-up to The Adventure of the Blue Carbuncle",
            limit=10,
        )
        assert context.items
        assert any(
            "carbuncle" in item.content.casefold()
            for item in context.items
            if item.kind == "chunk"
        )

        result = engine.write(
            "What would have happened if Sherlock Holmes caught Irene Adler?",
            max_words=300,
        )
        assert result.story
        assert result.grounding_report
        assert provider.write_context is not None
        assert any(
            "Irene Adler" in item.content or "OUTWITTED" in item.content
            for item in provider.write_context.items
        )
    finally:
        engine.close()
