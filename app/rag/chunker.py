"""Markdown-aware recursive chunker.

The corpus uses ``## `` (and occasionally ``### ``) sub-headers to delineate
sections. We split the body by those headers first so each chunk inherits the
section title for retrieval. Within a section we run
``RecursiveCharacterTextSplitter`` configured with ``tiktoken`` for token
counting (chunk size and overlap come from settings).
"""

import re
from dataclasses import dataclass
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

# Matches "## Title" and "### Title" at line start. Anything below H3 is
# treated as inline content rather than a section boundary — corpus uses at
# most two header levels.
_HEADER_PATTERN = re.compile(r"^(#{2,3})\s+(.+?)\s*$", re.MULTILINE)

_TIKTOKEN_ENCODING = "cl100k_base"


@dataclass(frozen=True)
class Chunk:
    text: str
    section: str
    chunk_index: int


@lru_cache(maxsize=1)
def _get_splitter() -> RecursiveCharacterTextSplitter:
    s = get_settings()
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=_TIKTOKEN_ENCODING,
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _split_by_sections(body: str, default_section: str) -> list[tuple[str, str]]:
    """Return a list of ``(section_title, section_text)`` tuples in document order."""
    matches = list(_HEADER_PATTERN.finditer(body))
    if not matches:
        text = body.strip()
        return [(default_section, text)] if text else []

    sections: list[tuple[str, str]] = []

    pre = body[: matches[0].start()].strip()
    if pre:
        sections.append((default_section, pre))

    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            sections.append((title, text))

    return sections


def chunk_document(body: str, default_section: str) -> list[Chunk]:
    """Split a markdown document body into chunks tagged with section titles."""
    splitter = _get_splitter()
    sections = _split_by_sections(body, default_section)
    chunks: list[Chunk] = []
    for section_title, section_text in sections:
        for piece in splitter.split_text(section_text):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(text=piece, section=section_title, chunk_index=len(chunks))
            )
    return chunks
