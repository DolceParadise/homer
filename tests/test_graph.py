from __future__ import annotations

from homer.graph import LiteraryGraph
from homer.models import (
    EntityCandidate,
    EntityType,
    GraphExtraction,
    RelationCandidate,
)


def test_aliases_merge_and_relationship_evidence_is_preserved(tmp_path):
    graph = LiteraryGraph(tmp_path / "graph.json")
    graph.merge(
        GraphExtraction(
            entities=[
                EntityCandidate(
                    name="Sherlock Holmes",
                    type=EntityType.CHARACTER,
                    aliases=["Holmes"],
                    description="A consulting detective.",
                    traits=["observant"],
                    evidence_chunk_ids=["c1"],
                ),
                EntityCandidate(
                    name="Holmes",
                    type=EntityType.CHARACTER,
                    aliases=["Sherlock"],
                    description="Watson's friend.",
                    traits=["analytical"],
                    evidence_chunk_ids=["c2"],
                ),
                EntityCandidate(
                    name="Irene Adler",
                    type=EntityType.CHARACTER,
                    aliases=["the woman"],
                    description="A singer who anticipated Holmes.",
                    traits=["resourceful"],
                    evidence_chunk_ids=["c2"],
                ),
            ],
            relations=[
                RelationCandidate(
                    source="Irene Adler",
                    target="Holmes",
                    type="OUTWITTED",
                    description="She escaped before he could stop her.",
                    evidence_chunk_ids=["c2"],
                )
            ],
            events=[],
        )
    )
    graph.save()
    restored = LiteraryGraph(tmp_path / "graph.json")

    assert restored.entity_count == 2
    assert restored.relation_count == 1
    seeds = restored.match_prompt_entities("What if Sherlock caught Irene Adler?")
    facts = restored.neighborhood_items(seeds)
    assert len(seeds) == 2
    assert any("OUTWITTED" in item.content for item in facts)
    assert facts[0].metadata["evidence_chunk_ids"] == ["c2"]
