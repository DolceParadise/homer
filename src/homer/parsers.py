from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from pathlib import Path

from bs4 import BeautifulSoup

from homer.models import (
    DocumentMetadata,
    DocumentSection,
    ParsedDocument,
)


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(*parts: object, length: int = 24) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def clean_text(value: str) -> str:
    value = value.replace("\xa0", " ").replace("\u00ad", "")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        raise NotImplementedError


class EpubParser(DocumentParser):
    def parse(self, path: Path) -> ParsedDocument:
        import ebooklib
        from ebooklib import epub

        path = path.resolve()
        checksum = file_checksum(path)
        document_id = stable_id(checksum, path.name)
        book = epub.read_epub(str(path), options={"ignore_ncx": True})

        title_values = book.get_metadata("DC", "title")
        creator_values = book.get_metadata("DC", "creator")
        title = str(title_values[0][0]).strip() if title_values else path.stem
        author = str(creator_values[0][0]).strip() if creator_values else None

        sections: list[DocumentSection] = []
        order = 0
        for item_id, linear in book.spine:
            if str(linear).lower() == "no":
                continue
            item = book.get_item_with_id(item_id)
            if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for tag in soup(["script", "style", "nav"]):
                tag.decompose()
            heading = soup.find(["h1", "h2", "h3", "title"])
            section_title = (
                clean_text(heading.get_text(" ", strip=True))
                if heading is not None
                else f"Section {order + 1}"
            )
            blocks = []
            for element in soup.find_all(["h1", "h2", "h3", "p", "blockquote"]):
                block = clean_text(element.get_text(" ", strip=True))
                if block and (not blocks or block != blocks[-1]):
                    blocks.append(block)
            text = clean_text("\n\n".join(blocks))
            if len(text.split()) < 20:
                continue
            sections.append(
                DocumentSection(
                    document_id=document_id,
                    section_id=stable_id(document_id, order, item.get_name()),
                    order=order,
                    title=section_title,
                    text=text,
                    page=None,
                )
            )
            order += 1

        if not sections:
            raise ValueError(f"No readable spine documents found in {path}")
        return ParsedDocument(
            metadata=DocumentMetadata(
                document_id=document_id,
                checksum=checksum,
                path=str(path),
                title=title,
                author=author,
                format="epub",
            ),
            sections=sections,
        )


class PdfParser(DocumentParser):
    def parse(self, path: Path) -> ParsedDocument:
        import fitz

        path = path.resolve()
        checksum = file_checksum(path)
        document_id = stable_id(checksum, path.name)
        document = fitz.open(path)
        metadata = document.metadata or {}
        title = (metadata.get("title") or path.stem).strip()
        author = (metadata.get("author") or "").strip() or None
        toc = document.get_toc(simple=True)

        def page_title(page_number: int) -> str:
            candidates = [
                entry[1]
                for entry in toc
                if len(entry) >= 3 and int(entry[2]) <= page_number
            ]
            return candidates[-1] if candidates else f"Page {page_number}"

        sections: list[DocumentSection] = []
        for index, page in enumerate(document):
            text = clean_text(page.get_text("text"))
            if len(text.split()) < 20:
                continue
            page_number = index + 1
            sections.append(
                DocumentSection(
                    document_id=document_id,
                    section_id=stable_id(document_id, page_number),
                    order=index,
                    title=page_title(page_number),
                    text=text,
                    page=page_number,
                )
            )
        document.close()
        if not sections:
            raise ValueError(f"No readable pages found in {path}")
        return ParsedDocument(
            metadata=DocumentMetadata(
                document_id=document_id,
                checksum=checksum,
                path=str(path),
                title=title,
                author=author,
                format="pdf",
            ),
            sections=sections,
        )


def parser_for(path: Path) -> DocumentParser:
    suffix = path.suffix.lower()
    if suffix == ".epub":
        return EpubParser()
    if suffix == ".pdf":
        return PdfParser()
    raise ValueError(f"Unsupported document format: {path.suffix}")


def parse_document(path: Path) -> ParsedDocument:
    if not path.is_file():
        raise FileNotFoundError(path)
    return parser_for(path).parse(path)
