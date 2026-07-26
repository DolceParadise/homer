from __future__ import annotations

from homer.chunking import SceneChunker
from homer.parsers import parse_document


def test_real_epubs_preserve_metadata_and_spine(corpus_files):
    documents = [parse_document(path) for path in corpus_files]

    assert [document.metadata.title for document in documents] == [
        "The Adventures of Sherlock Holmes",
        "The Sign of the Four",
    ]
    assert all(document.metadata.author == "Arthur Conan Doyle" for document in documents)
    assert all(len(document.sections) >= 10 for document in documents)
    assert all(
        [section.order for section in document.sections]
        == list(range(len(document.sections)))
        for document in documents
    )
    assert all(section.text.strip() for document in documents for section in document.sections)


def test_chunk_ids_are_stable_and_keep_provenance(corpus_files):
    document = parse_document(corpus_files[0])
    chunker = SceneChunker(target_words=350, max_words=500, overlap_words=40)

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len(first) > len(document.sections)
    assert all(chunk.document_title == document.metadata.title for chunk in first)
    assert all(chunk.section_title for chunk in first)
    assert all(chunk.source_path == document.metadata.path for chunk in first)
