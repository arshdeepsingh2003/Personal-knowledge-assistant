"""
chunker.py

Tested and confirmed working. The exact logic was verified against the real
SCALE-split-across-pages PDF before writing this file.

Key insight discovered from testing:
  - strip_headers=False keeps "## **4.1 The SCALE Framework**" inside content
    but our substring check "heading in content" FAILS because heading in
    metadata is "4.1 The SCALE Framework" (no ## or **) while content has
    "## **4.1 The SCALE Framework**". String mismatch → no prefix injected.

  - strip_headers=True removes the heading line from content entirely.
    Heading lives ONLY in metadata as clean text: "4.1 The SCALE Framework"
    We ALWAYS prepend it — no substring check needed — so it is guaranteed
    to appear in every chunk.

Verified output for chunk_size=400 on the SCALE PDF:
  Chunk 1: ✓ SCALE | intro text
  Chunk 2: ✓ SCALE | S and C items
  Chunk 3: ✓ SCALE | A and L items      ← was failing before
  Chunk 4: ✓ SCALE | E item             ← was failing before
"""

import hashlib
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.core.config import settings

DEFAULT_CHUNK_SIZE    = settings.chunking_default_size
DEFAULT_CHUNK_OVERLAP = settings.chunking_default_overlap


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_documents(
    documents:     List[Document],
    chunk_size:    int   = None,
    chunk_overlap: int   = None,
    strategy:      str = None,
) -> List[Document]:
    chunk_size    = chunk_size    or DEFAULT_CHUNK_SIZE
    chunk_overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP
    strategy      = strategy      or settings.chunking_default_strategy

    if strategy in ("structure_aware", "markdown", "semantic"):
        return _smart_chunk(documents, chunk_size, chunk_overlap)
    return _chunk_recursive(documents, chunk_size, chunk_overlap)


# ── Smart chunker ──────────────────────────────────────────────────────────────

def _smart_chunk(
    documents:     List[Document],
    chunk_size:    int,
    chunk_overlap: int,
) -> List[Document]:
    all_chunks: List[Document] = []
    for doc in documents:
        ctype = doc.metadata.get("content_type", "prose")
        if ctype == "pdf_markdown":
            all_chunks.extend(_chunk_pdf_markdown(doc, chunk_size, chunk_overlap))
        else:
            all_chunks.extend(_chunk_recursive([doc], chunk_size, chunk_overlap))
    return _label_chunks(all_chunks)


# ── PDF markdown chunker — tested and confirmed ────────────────────────────────

def _chunk_pdf_markdown(
    doc:           Document,
    chunk_size:    int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Three-phase chunking for pymupdf4llm markdown output.

    Phase 1 — MarkdownHeaderTextSplitter with strip_headers=TRUE
      Cuts at ##/### boundaries. Heading goes to metadata ONLY (not content).
      One Document per section. Multi-page sections stay together because
      pymupdf4llm puts no ## inside a section.

    Phase 2 — ALWAYS prepend heading to content
      Since strip_headers=True guarantees the heading is NOT in content,
      we unconditionally prepend it. No substring check needed.
      Every Document now starts with the section heading keyword.

    Phase 3 — Split oversized sections
      If a section > chunk_size, RecursiveCharacterTextSplitter splits it.
      After splitting, any sub-chunk missing the heading gets it re-prepended.
      This is the fix for the exact failure: L and E sub-chunks now start
      with "4.1 The SCALE Framework for Enterprise AI" so retrieval works.
    """

    # ── Phase 1 ────────────────────────────────────────────────────────────────
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",   "h1"),
            ("##",  "h2"),
            ("###", "h3"),
        ],
        strip_headers=True,   # heading removed from content → lives in metadata only
    )

    sections = header_splitter.split_text(doc.page_content)

    # Transfer source metadata (file_name, source, etc.) to each section
    source_meta = {k: v for k, v in doc.metadata.items()
                   if k not in ("content_type",)}

    # ── Phase 2 ────────────────────────────────────────────────────────────────
    section_docs: List[Document] = []
    for s in sections:
        content = s.page_content.strip()
        if not content:
            continue

        # Build clean heading from metadata (strip markdown bold markers)
        h_parts = [
            s.metadata.get("h1", ""),
            s.metadata.get("h2", ""),
            s.metadata.get("h3", ""),
        ]
        heading = " — ".join(
            p.replace("**", "").replace("*", "").strip()
            for p in h_parts if p
        ).strip()

        # Generate unique section_id from source file + heading
        section_key = f"{source_meta.get('file_name', '')}:{heading}"
        section_id = hashlib.md5(section_key.encode()).hexdigest()[:12]

        # ALWAYS prepend — strip_headers=True guarantees it's not in content
        if heading:
            content = f"{heading}\n\n{content}"

        section_docs.append(Document(
            page_content=content,
            metadata={
                **source_meta,
                **s.metadata,
                "content_type": "section",
                "heading":      heading,
                "section_id":   section_id,
            }
        ))

    # ── Phase 3 ────────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    final: List[Document] = []
    for sdoc in section_docs:
        heading = sdoc.metadata.get("heading", "")
        content = sdoc.page_content

        if len(content) <= chunk_size:
            sdoc.metadata["section_chunk_index"] = 0
            sdoc.metadata["section_total_chunks"] = 1
            final.append(sdoc)
            continue

        # Split into sub-chunks
        subs = splitter.split_documents([sdoc])
        heading_lower = heading.lower()
        section_total = len(subs)

        for idx, sc in enumerate(subs):
            # Re-inject heading into any sub-chunk that lost it
            if heading and heading_lower not in sc.page_content.lower():
                sc.page_content = f"{heading}\n\n{sc.page_content}"
                sc.metadata["section_prefix"] = heading
            sc.metadata["section_chunk_index"] = idx
            sc.metadata["section_total_chunks"] = section_total
            final.append(sc)

    return final


# ── Legacy chunker ─────────────────────────────────────────────────────────────

def _chunk_recursive(
    documents:     List[Document],
    chunk_size:    int,
    chunk_overlap: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )
    return _label_chunks(splitter.split_documents(documents))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _label_chunks(chunks: List[Document]) -> List[Document]:
    total = len(chunks)

    source_groups: dict = {}
    for c in chunks:
        src = c.metadata.get("file_name", c.metadata.get("source", "__unknown__"))
        source_groups.setdefault(src, []).append(c)

    for src, group in source_groups.items():
        gtotal = len(group)
        for pos, c in enumerate(group):
            c.metadata["position_ratio"] = round(pos / max(gtotal - 1, 1), 4)

    for i, c in enumerate(chunks):
        c.metadata["chunk_index"]  = i
        c.metadata["total_chunks"] = total
    return chunks


def get_chunk_stats(chunks: List[Document]) -> dict:
    if not chunks:
        return {}
    lengths  = [len(c.page_content) for c in chunks]
    sections = sum(1 for c in chunks if c.metadata.get("content_type") == "section")
    prefixed = sum(1 for c in chunks if c.metadata.get("section_prefix"))
    return {
        "total_chunks":    len(chunks),
        "section_chunks":  sections,
        "prefixed_chunks": prefixed,
        "avg_length":      round(sum(lengths) / len(lengths)),
        "min_length":      min(lengths),
        "max_length":      max(lengths),
        "total_chars":     sum(lengths),
    }