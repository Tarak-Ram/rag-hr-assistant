"""Heading-aware Markdown chunking.

Policies are split on headings first so a chunk never mixes two topics, then
long sections are split by paragraph with overlap. Every chunk keeps its
document and section title, which become the citation shown to the user.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    id: str
    doc: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.doc} § {self.section}"


_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    pieces, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) + 1 > size:
            pieces.append(current)
            current = current[-overlap:] + " " + p if overlap else p
        else:
            current = f"{current} {p}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(path: Path, size: int = 600, overlap: int = 100) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    doc_title_match = re.search(r"^#\s+(.*)$", raw, re.MULTILINE)
    doc_title = doc_title_match.group(1).strip() if doc_title_match else path.stem

    matches = list(_HEADING.finditer(raw))
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        if body:
            sections.append((m.group(2).strip(), body))

    chunks: list[Chunk] = []
    for section, body in sections:
        for n, piece in enumerate(_split_long(body, size, overlap)):
            digest = hashlib.sha1(f"{path.name}|{section}|{n}".encode()).hexdigest()[:12]
            chunks.append(
                Chunk(
                    id=digest,
                    doc=doc_title,
                    section=section,
                    text=piece,
                    metadata={"source": path.name},
                )
            )
    return chunks


def load_corpus(docs_dir: Path, size: int = 600, overlap: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path, size, overlap))
    return chunks
