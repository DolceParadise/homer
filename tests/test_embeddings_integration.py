from __future__ import annotations

import numpy as np
import pytest

from homer.embeddings import QwenEmbeddingProvider
from homer.parsers import parse_document


@pytest.mark.integration
def test_qwen_embedding_shape_normalization_and_literary_ranking(corpus_files):
    adventures = parse_document(corpus_files[0])
    sign = parse_document(corpus_files[1])
    carbuncle = next(
        section
        for section in adventures.sections
        if "carbuncle" in section.text.casefold()
    )
    unrelated = sign.sections[-1]
    provider = QwenEmbeddingProvider(batch_size=2)

    documents = provider.embed_documents([carbuncle.text[:5000], unrelated.text[:5000]])
    query = provider.embed_query("The Adventure of the Blue Carbuncle")
    scores = np.asarray(documents) @ np.asarray(query)

    assert np.asarray(documents).shape == (2, 1024)
    assert np.allclose(np.linalg.norm(documents, axis=1), 1.0, atol=1e-4)
    assert scores[0] > scores[1]
