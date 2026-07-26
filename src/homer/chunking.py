from __future__ import annotations

import re

from homer.models import ParsedDocument, TextChunk
from homer.parsers import stable_id


SCENE_BREAK = re.compile(r"^\s*(?:\*\s*){3,}$|^\s*[-—]\s*[-—]\s*[-—]\s*$")


class SceneChunker:
    def __init__(
        self,
        target_words: int = 650,
        max_words: int = 850,
        overlap_words: int = 80,
    ) -> None:
        if not 0 <= overlap_words < target_words <= max_words:
            raise ValueError("Expected 0 <= overlap < target <= max")
        self.target_words = target_words
        self.max_words = max_words
        self.overlap_words = overlap_words

    def chunk(self, document: ParsedDocument) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for section in document.sections:
            paragraphs = [
                value.strip()
                for value in re.split(r"\n\s*\n", section.text)
                if value.strip()
            ]
            current: list[str] = []
            current_words = 0
            section_chunks: list[str] = []
            for paragraph in paragraphs:
                words = paragraph.split()
                if SCENE_BREAK.match(paragraph):
                    if current:
                        section_chunks.append("\n\n".join(current))
                        current, current_words = [], 0
                    continue
                if current and current_words + len(words) > self.max_words:
                    completed = "\n\n".join(current)
                    section_chunks.append(completed)
                    overlap = completed.split()[-self.overlap_words :]
                    current = [" ".join(overlap)] if overlap else []
                    current_words = len(overlap)
                current.append(paragraph)
                current_words += len(words)
                if current_words >= self.target_words:
                    completed = "\n\n".join(current)
                    section_chunks.append(completed)
                    overlap = completed.split()[-self.overlap_words :]
                    current = [" ".join(overlap)] if overlap else []
                    current_words = len(overlap)
            if current and current_words > self.overlap_words:
                section_chunks.append("\n\n".join(current))

            for chunk_order, text in enumerate(section_chunks):
                chunks.append(
                    TextChunk(
                        chunk_id=stable_id(
                            document.metadata.checksum,
                            section.section_id,
                            chunk_order,
                            text,
                        ),
                        document_id=document.metadata.document_id,
                        document_title=document.metadata.title,
                        author=document.metadata.author,
                        section_id=section.section_id,
                        section_title=section.title,
                        section_order=section.order,
                        chunk_order=chunk_order,
                        text=text,
                        page=section.page,
                        source_path=document.metadata.path,
                    )
                )
        return chunks
