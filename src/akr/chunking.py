"""Split Markdown documents into citable chunks, one per `##` section."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str       # citable id: "<document>#<section-slug>", e.g. "payments#deposit"
    doc: str      # file stem
    title: str    # document title (the `#` heading)
    section: str  # section heading (the `##` heading)
    text: str     # section body


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def chunk_markdown(doc: str, text: str) -> list[Chunk]:
    """One chunk per `##` section. Blockquotes (disclaimers) are skipped.

    A document without `##` sections becomes a single "overview" chunk.
    """
    title, section, buf, out = doc, None, [], []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            name = section or "Overview"
            out.append(Chunk(f"{doc}#{slugify(name)}", doc, title, name, body))

    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            section, buf = line[3:].strip(), []
        elif not line.startswith(">"):
            buf.append(line)
    flush()
    return out


def load_corpus(path: str | Path) -> list[Chunk]:
    folder = Path(path)
    files = sorted(folder.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No Markdown files found in {folder}")
    chunks: list[Chunk] = []
    for f in files:
        chunks.extend(chunk_markdown(f.stem, f.read_text(encoding="utf-8")))
    ids = [c.id for c in chunks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate chunk ids (rename a section): {sorted(dupes)}")
    return chunks
