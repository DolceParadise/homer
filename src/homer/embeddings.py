from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class QwenEmbeddingProvider:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 8,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._dimension = 1024

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._model is not None:
            return int(self._model.get_sentence_embedding_dimension())
        return self._dimension

    def _load(self):
        if self._model is None:
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError
            from sentence_transformers import SentenceTransformer

            kwargs = {}
            if self.device:
                kwargs["device"] = self.device
            model_source = self._model_name
            try:
                # Loading from the resolved snapshot prevents Transformers from
                # making model-metadata requests after the files are cached.
                model_source = snapshot_download(
                    repo_id=self._model_name,
                    local_files_only=True,
                )
            except LocalEntryNotFoundError:
                pass
            self._model = SentenceTransformer(model_source, **kwargs)
            self._dimension = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @staticmethod
    def _as_lists(values: np.ndarray) -> list[list[float]]:
        return values.astype(np.float32, copy=False).tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self.batch_size,
        )
        return self._as_lists(np.asarray(vectors))

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        prompt_name = "query" if "query" in model.prompts else None
        vector = model.encode(
            [text],
            prompt_name=prompt_name,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self._as_lists(np.asarray(vector))[0]
