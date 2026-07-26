from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from homer.embeddings import QwenEmbeddingProvider
from homer.llm import CerebrasProvider
from homer.pipeline import CorpusEngine


PACKAGE_ROOT = Path(__file__).parent
STATIC_ROOT = PACKAGE_ROOT / "static"
DATA_ROOT = Path(os.getenv("HOMER_DATA_DIR", ".homer"))

app = FastAPI(title="Homer", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=4000)
    corpus: str = Field(default="sherlock", min_length=1, max_length=64)
    max_words: int = Field(default=1200, ge=200, le=5000)


_engine_lock = threading.RLock()
_engines: dict[str, CorpusEngine] = {}


def _engine(corpus: str) -> CorpusEngine:
    with _engine_lock:
        if corpus not in _engines:
            _engines[corpus] = CorpusEngine(
                corpus=corpus,
                root=DATA_ROOT,
                embedding_provider=QwenEmbeddingProvider(),
            )
        return _engines[corpus]


def _edge_id(source: str, target: str, key: str) -> str:
    return f"edge:{source}:{target}:{key}"


def graph_payload(engine: CorpusEngine, used_item_ids: set[str] | None = None) -> dict[str, Any]:
    graph = engine.graph.graph
    community_by_id = {item.community_id: item for item in engine.store.communities()}
    used_nodes: set[str] | None = None
    used_edges: set[str] | None = None
    used_communities: set[str] | None = None

    if used_item_ids is not None:
        used_nodes, used_edges, used_communities = set(), set(), set()
        for item_id in used_item_ids:
            kind, _, value = item_id.partition(":")
            if kind == "relation":
                parts = value.split(":", 2)
                if len(parts) == 3:
                    source, target, key = parts
                    used_nodes.update((source, target))
                    used_edges.add(_edge_id(source, target, key))
            elif kind == "entity":
                used_nodes.add(value)
            elif kind == "community":
                used_communities.add(value)

    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        node_id = str(node_id)
        if used_nodes is not None and node_id not in used_nodes:
            continue
        nodes.append(
            {
                "data": {
                    "id": node_id,
                    "label": attrs.get("name", node_id),
                    "type": attrs.get("type", "other"),
                    "description": " ".join(attrs.get("descriptions", [])[:2]),
                    "traits": ", ".join(attrs.get("traits", [])),
                }
            }
        )

    edges = []
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        edge_id = _edge_id(str(source), str(target), str(key))
        if used_edges is not None and edge_id not in used_edges:
            continue
        edges.append(
            {
                "data": {
                    "id": edge_id,
                    "source": str(source),
                    "target": str(target),
                    "label": attrs.get("type", "RELATED_TO"),
                    "description": attrs.get("description", ""),
                }
            }
        )

    communities = []
    for community_id, community in community_by_id.items():
        if used_communities is not None and community_id not in used_communities:
            continue
        communities.append(
            {
                "id": community_id,
                "title": community.title,
                "summary": community.summary,
                "entities": len(community.entity_ids),
            }
        )
    return {"elements": [*nodes, *edges], "communities": communities}


@asynccontextmanager
async def corpus_lifespan(_: FastAPI):
    """Open Qdrant on the server thread so it can also be closed there."""
    engine = _engine("sherlock")
    if engine.paths.qdrant.exists():
        _ = engine.vectors
    try:
        yield
    finally:
        with _engine_lock:
            for engine in _engines.values():
                engine.close()


app.router.lifespan_context = corpus_lifespan


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/graph")
def get_graph(corpus: str = "sherlock") -> dict[str, Any]:
    engine = _engine(corpus)
    with _engine_lock:
        if not engine.paths.graph.exists():
            raise HTTPException(status_code=404, detail=f"Corpus '{corpus}' is not indexed")
        payload = graph_payload(engine)
        payload["stats"] = engine.inspect().model_dump(mode="json")
        return payload


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    engine = _engine(request.corpus)
    with _engine_lock:
        if not engine.paths.graph.exists():
            raise HTTPException(status_code=404, detail=f"Corpus '{request.corpus}' is not indexed")
        if engine.story_writer is None:
            engine.story_writer = CerebrasProvider()
        result = engine.write(request.prompt, max_words=request.max_words)
        used_ids = {reference.item_id for reference in result.grounding_report}
        return {
            "story": result.story,
            "grounding": [item.model_dump(mode="json") for item in result.grounding_report],
            "graph": graph_payload(engine, used_ids),
        }


def run() -> None:
    import uvicorn

    uvicorn.run("homer.web:app", host="127.0.0.1", port=8000, reload=True)
