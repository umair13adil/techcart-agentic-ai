"""Markdown-aware chunking with contextual headers.

* split on headings first (keeps semantic sections together)
* never split inside a markdown table row; long tables are split by row groups with the header repeated
* each chunk is prefixed with "<doc title> > <section>" so a chunk like "| 2,001 - 5,000 | 299 | 499 |" still
  carries its meaning when retrieved on its own (a cheap version of "contextual retrieval")
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    meta: dict = field(default_factory=dict)


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out, cur, header = [], "", []
    for ln in text.split("\n"):
        is_row = ln.startswith("|")
        if is_row and len(header) < 2:
            header.append(ln)           # remember table header + separator row
        elif not is_row:
            header = []
        if cur and len(cur) + len(ln) + 1 > max_chars:
            out.append(cur.strip())
            if is_row and len(header) == 2 and ln not in header:
                cur = "\n".join(header) + "\n"   # repeat table header in the next chunk
            else:
                cur = cur[-overlap:] + "\n" if not is_row else ""
        cur += ln + "\n"
    if cur.strip():
        out.append(cur.strip())
    return out


def chunk_markdown(doc_id: str, markdown: str, meta: dict, max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    title_match = re.match(r"#\s+(.+)", markdown)
    title = title_match.group(1).strip() if title_match else doc_id
    body = markdown[title_match.end():] if title_match else markdown
    sections = re.split(r"\n(?=#{2,3}\s)", body)
    chunks: list[Chunk] = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        h = re.match(r"#{2,3}\s+(.+)", sec)
        heading = h.group(1).strip() if h else "Overview"
        for piece in _split_long(sec, max_chars, overlap):
            n = len(chunks)
            chunks.append(Chunk(f"{doc_id}#{n}", doc_id, f"[{title} > {heading}]\n{piece}",
                                {**meta, "title": title, "section": heading}))
    # small docs (products) are better as one chunk: merge if total is short
    if len(chunks) > 1 and sum(len(c.text) for c in chunks) < max_chars * 1.6:
        merged = "\n".join([chunks[0].text] + [c.text.split("\n", 1)[1] for c in chunks[1:]])
        chunks = [Chunk(f"{doc_id}#0", doc_id, merged, {**meta, "title": title, "section": "all"})]
    return chunks
