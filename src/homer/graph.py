from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import networkx as nx
from networkx.readwrite import json_graph

from homer.models import (
    CommunitySummary,
    EntityCandidate,
    EntityType,
    GraphExtraction,
    RelationCandidate,
    RetrievedItem,
)


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()


def _dedupe(values: Iterable[str]) -> list[str]:
    return sorted({value.strip() for value in values if value and value.strip()})


class LiteraryGraph:
    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.graph = json_graph.node_link_graph(
                data,
                directed=True,
                multigraph=True,
                edges="edges",
            )
        else:
            self.graph = nx.MultiDiGraph()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = json_graph.node_link_data(self.graph, edges="edges")
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _match_node(self, name: str) -> str | None:
        normalized = normalize_name(name)
        if not normalized:
            return None
        for node_id, attrs in self.graph.nodes(data=True):
            names = [attrs.get("name", ""), *attrs.get("aliases", [])]
            normalized_names = {normalize_name(value) for value in names}
            if normalized in normalized_names:
                return str(node_id)
            tokens = normalized.split()
            for existing in normalized_names:
                existing_tokens = existing.split()
                if (
                    len(tokens) == 1
                    and len(tokens[0]) > 3
                    and existing_tokens
                    and tokens[0] == existing_tokens[-1]
                ):
                    return str(node_id)
                if (
                    len(existing_tokens) == 1
                    and len(existing_tokens[0]) > 3
                    and tokens
                    and existing_tokens[0] == tokens[-1]
                ):
                    return str(node_id)
        return None

    def upsert_entity(self, entity: EntityCandidate) -> str:
        node_id = self._match_node(entity.name)
        if node_id is None:
            node_id = hashlib.sha256(
                f"{normalize_name(entity.name)}:{entity.type}".encode()
            ).hexdigest()[:20]
            self.graph.add_node(
                node_id,
                name=entity.name.strip(),
                type=entity.type.value,
                aliases=[],
                descriptions=[],
                traits=[],
                evidence_chunk_ids=[],
            )
        attrs = self.graph.nodes[node_id]
        aliases = [*attrs.get("aliases", []), *entity.aliases]
        if normalize_name(entity.name) != normalize_name(attrs.get("name", "")):
            aliases.append(entity.name)
        attrs["aliases"] = _dedupe(aliases)
        attrs["descriptions"] = _dedupe(
            [*attrs.get("descriptions", []), entity.description]
        )
        attrs["traits"] = _dedupe([*attrs.get("traits", []), *entity.traits])
        attrs["evidence_chunk_ids"] = _dedupe(
            [*attrs.get("evidence_chunk_ids", []), *entity.evidence_chunk_ids]
        )
        return node_id

    def _placeholder(self, name: str) -> str:
        return self.upsert_entity(
            EntityCandidate(
                name=name,
                type=EntityType.OTHER,
                aliases=[],
                description="Referenced in an extracted relationship.",
                traits=[],
                evidence_chunk_ids=[],
            )
        )

    def add_relation(self, relation: RelationCandidate) -> None:
        source = self._match_node(relation.source) or self._placeholder(relation.source)
        target = self._match_node(relation.target) or self._placeholder(relation.target)
        relation_type = re.sub(
            r"[^A-Z0-9]+",
            "_",
            relation.type.upper().strip(),
        ).strip("_") or "RELATED_TO"
        key_source = "|".join(
            [
                source,
                target,
                relation_type,
                relation.description,
            ]
        )
        key = hashlib.sha256(key_source.encode()).hexdigest()[:20]
        if self.graph.has_edge(source, target, key):
            attrs = self.graph[source][target][key]
            attrs["evidence_chunk_ids"] = _dedupe(
                [
                    *attrs.get("evidence_chunk_ids", []),
                    *relation.evidence_chunk_ids,
                ]
            )
            return
        self.graph.add_edge(
            source,
            target,
            key=key,
            type=relation_type,
            description=relation.description,
            evidence_chunk_ids=_dedupe(relation.evidence_chunk_ids),
        )

    def merge(self, extraction: GraphExtraction) -> None:
        for entity in extraction.entities:
            self.upsert_entity(entity)
        for relation in extraction.relations:
            self.add_relation(relation)
        for event in extraction.events:
            event_id = self.upsert_entity(
                EntityCandidate(
                    name=event.name,
                    type=EntityType.EVENT,
                    aliases=[],
                    description=event.description,
                    traits=[],
                    evidence_chunk_ids=event.evidence_chunk_ids,
                )
            )
            for participant in event.participants:
                participant_id = self._match_node(participant) or self._placeholder(
                    participant
                )
                self.add_relation(
                    RelationCandidate(
                        source=self.graph.nodes[participant_id]["name"],
                        target=self.graph.nodes[event_id]["name"],
                        type="PARTICIPATED_IN",
                        description=event.description,
                        evidence_chunk_ids=event.evidence_chunk_ids,
                    )
                )
            for location in event.locations:
                location_id = self._match_node(location) or self._placeholder(location)
                self.add_relation(
                    RelationCandidate(
                        source=self.graph.nodes[event_id]["name"],
                        target=self.graph.nodes[location_id]["name"],
                        type="OCCURRED_AT",
                        description=event.description,
                        evidence_chunk_ids=event.evidence_chunk_ids,
                    )
                )

    def communities(self) -> list[set[str]]:
        undirected = nx.Graph()
        undirected.add_nodes_from(self.graph.nodes)
        undirected.add_edges_from((source, target) for source, target in self.graph.edges())
        if not undirected.nodes:
            return []
        if not undirected.edges:
            return []
        values = nx.community.greedy_modularity_communities(undirected)
        # Isolated entities do not form a meaningful narrative community and
        # summarizing each one would add cost without useful global context.
        return [
            set(map(str, value))
            for value in values
            if len(value) >= 2
        ]

    def community_payload(self, node_ids: set[str]) -> dict:
        nodes = []
        evidence = set()
        for node_id in sorted(node_ids):
            attrs = self.graph.nodes[node_id]
            nodes.append(
                {
                    "id": node_id,
                    "name": attrs.get("name"),
                    "type": attrs.get("type"),
                    "descriptions": attrs.get("descriptions", []),
                    "traits": attrs.get("traits", []),
                }
            )
            evidence.update(attrs.get("evidence_chunk_ids", []))
        relations = []
        for source, target, attrs in self.graph.edges(data=True):
            if str(source) in node_ids and str(target) in node_ids:
                relations.append(
                    {
                        "source": self.graph.nodes[source].get("name"),
                        "target": self.graph.nodes[target].get("name"),
                        "type": attrs.get("type"),
                        "description": attrs.get("description"),
                    }
                )
                evidence.update(attrs.get("evidence_chunk_ids", []))
        return {
            "nodes": nodes,
            "relations": relations,
            "evidence_chunk_ids": sorted(evidence),
        }

    def entity_records(self) -> list[tuple[str, str, dict]]:
        records = []
        for node_id, attrs in self.graph.nodes(data=True):
            text = "\n".join(
                [
                    f"{attrs.get('name')} ({attrs.get('type')})",
                    *attrs.get("descriptions", []),
                    *[f"Trait: {trait}" for trait in attrs.get("traits", [])],
                ]
            )
            records.append(
                (
                    str(node_id),
                    text,
                    {
                        "name": attrs.get("name"),
                        "entity_type": attrs.get("type"),
                        "evidence_chunk_ids": attrs.get("evidence_chunk_ids", []),
                    },
                )
            )
        return records

    def match_prompt_entities(self, prompt: str) -> set[str]:
        normalized_prompt = f" {normalize_name(prompt)} "
        matches = set()
        for node_id, attrs in self.graph.nodes(data=True):
            names = [attrs.get("name", ""), *attrs.get("aliases", [])]
            if any(
                value and f" {normalize_name(value)} " in normalized_prompt
                for value in names
            ):
                matches.add(str(node_id))
        return matches

    def neighborhood_items(
        self,
        seeds: set[str],
        hops: int = 2,
    ) -> list[RetrievedItem]:
        selected = set(seeds)
        frontier = set(seeds)
        undirected = self.graph.to_undirected()
        for _ in range(hops):
            frontier = {
                str(neighbor)
                for node in frontier
                for neighbor in undirected.neighbors(node)
            } - selected
            selected.update(frontier)
        items = []
        seen = set()
        for source, target, key, attrs in self.graph.edges(keys=True, data=True):
            if str(source) not in selected and str(target) not in selected:
                continue
            item_id = f"relation:{source}:{target}:{key}"
            if item_id in seen:
                continue
            seen.add(item_id)
            source_name = self.graph.nodes[source].get("name", str(source))
            target_name = self.graph.nodes[target].get("name", str(target))
            content = (
                f"{source_name} --{attrs.get('type', 'RELATED_TO')}--> "
                f"{target_name}: {attrs.get('description', '')}"
            )
            items.append(
                RetrievedItem(
                    item_id=item_id,
                    kind="relation",
                    content=content,
                    score=(
                        1.35
                        if str(source) in seeds and str(target) in seeds
                        else 1.1
                        if str(source) in seeds or str(target) in seeds
                        else 0.95
                    ),
                    metadata={
                        "source_entity": source_name,
                        "target_entity": target_name,
                        "evidence_chunk_ids": attrs.get("evidence_chunk_ids", []),
                    },
                )
            )
        return items

    @property
    def entity_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def relation_count(self) -> int:
        return self.graph.number_of_edges()


def community_from_payload(
    community_id: str,
    title: str,
    summary: str,
    node_ids: set[str],
    payload: dict,
) -> CommunitySummary:
    return CommunitySummary(
        community_id=community_id,
        title=title,
        summary=summary,
        entity_ids=sorted(node_ids),
        evidence_chunk_ids=payload["evidence_chunk_ids"],
    )
