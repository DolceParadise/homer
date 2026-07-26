from __future__ import annotations

from homer.models import CommunitySummary
from homer.pipeline import CorpusEngine
from homer.web import graph_payload


def test_graph_payload_filters_to_grounding_evidence(tmp_path):
    engine = CorpusEngine(corpus="web", root=tmp_path)
    try:
        graph = engine.graph.graph
        graph.add_node("holmes", name="Sherlock Holmes", type="character")
        graph.add_node("adler", name="Irene Adler", type="character")
        graph.add_node("watson", name="Dr. Watson", type="character")
        graph.add_edge(
            "holmes",
            "adler",
            key="adler-case",
            type="PURSUED",
            description="Holmes pursues Adler.",
        )
        graph.add_edge(
            "holmes",
            "watson",
            key="watson-case",
            type="FRIEND_OF",
            description="Watson assists Holmes.",
        )
        engine.store.save_communities(
            [
                CommunitySummary(
                    community_id="bohemia",
                    title="The Adler affair",
                    summary="Holmes and Adler clash over a royal photograph.",
                    entity_ids=["holmes", "adler"],
                    evidence_chunk_ids=["chunk-1"],
                )
            ]
        )

        full = graph_payload(engine)
        focused = graph_payload(
            engine,
            {"relation:holmes:adler:adler-case", "community:bohemia"},
        )

        assert len(full["elements"]) == 5
        assert {item["data"]["id"] for item in focused["elements"]} == {
            "holmes",
            "adler",
            "edge:holmes:adler:adler-case",
        }
        assert focused["communities"][0]["title"] == "The Adler affair"
    finally:
        engine.close()
