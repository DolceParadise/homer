from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Callable, Protocol, Sequence

from dotenv import find_dotenv, load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict

from homer.models import (
    GraphExtraction,
    RetrievedContext,
    StoryRequest,
    TextChunk,
)


class GraphExtractor(Protocol):
    def extract(self, chunks: Sequence[TextChunk]) -> GraphExtraction: ...


class CommunitySummarizer(Protocol):
    def summarize(self, payload: dict) -> tuple[str, str]: ...


class StoryWriter(Protocol):
    def write(
        self,
        request: StoryRequest,
        context: RetrievedContext,
    ) -> str: ...


class _CommunityOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    summary: str


class CerebrasProvider(GraphExtractor, CommunitySummarizer, StoryWriter):
    def __init__(
        self,
        api_key: str | None = None,
        extraction_model: str = "gpt-oss-120b",
        writer_model: str = "zai-glm-4.7",
        writer_fallback_model: str = "gpt-oss-120b",
        max_attempts: int = 6,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path, override=False)
        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "CEREBRAS_API_KEY is required. Add it to .env or the process environment."
            )
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.cerebras.ai/v1",
            timeout=180.0,
        )
        self.extraction_model = extraction_model
        self.writer_model = writer_model
        self.writer_fallback_model = writer_fallback_model
        self.max_attempts = max_attempts
        self.sleep = sleep

    @staticmethod
    def _retry_delay(error: Exception, attempt: int) -> float:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return min(60.0, (2**attempt) + random.random())

    def _call(self, **kwargs):
        retryable = (RateLimitError, APIConnectionError, APIStatusError)
        for attempt in range(self.max_attempts):
            try:
                return self.client.chat.completions.create(**kwargs)
            except retryable as error:
                status = getattr(error, "status_code", None)
                if (
                    attempt + 1 >= self.max_attempts
                    or status is not None
                    and status < 500
                    and status != 429
                ):
                    raise
                self.sleep(self._retry_delay(error, attempt))
        raise RuntimeError("Unreachable retry state")

    def extract(self, chunks: Sequence[TextChunk]) -> GraphExtraction:
        if not chunks:
            return GraphExtraction(entities=[], relations=[], events=[])
        corpus = "\n\n".join(
            f"<chunk id={chunk.chunk_id} book={json.dumps(chunk.document_title)} "
            f"section={json.dumps(chunk.section_title)}>\n{chunk.text}\n</chunk>"
            for chunk in chunks
        )
        response = self._call(
            model=self.extraction_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract a conservative literary knowledge graph from the supplied "
                        "passages. Include only facts supported by the passages. Canonicalize "
                        "names, record useful aliases, express relationship types as concise "
                        "uppercase verbs, and attach valid supplied chunk IDs to every claim. "
                        "Return empty arrays rather than inventing information."
                    ),
                },
                {"role": "user", "content": corpus},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "literary_graph",
                    "strict": True,
                    "schema": GraphExtraction.model_json_schema(),
                },
            },
            reasoning_effort="low",
            temperature=0.1,
            max_completion_tokens=6000,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Cerebras returned an empty graph extraction")
        return GraphExtraction.model_validate_json(content)

    def summarize(self, payload: dict) -> tuple[str, str]:
        response = self._call(
            model=self.extraction_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize this connected literary subgraph. Give it a short factual "
                        "title and explain the central characters, events, motivations, and "
                        "relationships without adding unsupported details."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False)[:30000],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "community_summary",
                    "strict": True,
                    "schema": _CommunityOutput.model_json_schema(),
                },
            },
            reasoning_effort="low",
            temperature=0.2,
            max_completion_tokens=1200,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Cerebras returned an empty community summary")
        parsed = _CommunityOutput.model_validate_json(content)
        return parsed.title, parsed.summary

    def write(self, request: StoryRequest, context: RetrievedContext) -> str:
        context_blocks = "\n\n".join(
            f"[{index}. {item.kind} | {item.item_id}]\n{item.content}"
            for index, item in enumerate(context.items, start=1)
        )
        kwargs = dict(
            model=self.writer_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Write a polished literary continuation grounded in the supplied canon. "
                        "Use the corpus's narrative viewpoint, period vocabulary, characterization, "
                        "and established relationships. Treat graph facts as canon and excerpts as "
                        "stylistic and factual evidence. Do not mention retrieval, context blocks, "
                        "or citations in the story. Do not contradict the evidence. When the user's "
                        "premise changes canon, change only what the premise requires and develop "
                        "plausible consequences."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"REQUEST:\n{request.prompt}\n\n"
                        f"TARGET LENGTH: no more than {request.max_words} words.\n\n"
                        f"RETRIEVED CANON AND STYLE:\n{context_blocks}"
                    ),
                },
            ],
            reasoning_effort="none",
            temperature=0.9,
            top_p=0.95,
            max_completion_tokens=min(8000, max(1200, request.max_words * 2)),
        )
        try:
            response = self._call(**kwargs)
        except RateLimitError:
            kwargs.update(
                {
                    "model": self.writer_fallback_model,
                    "reasoning_effort": "low",
                }
            )
            response = self._call(**kwargs)
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Cerebras returned an empty story")
        return content.strip()
