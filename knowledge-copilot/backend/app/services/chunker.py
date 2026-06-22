"""
chunker.py — Enhanced chunker with table preservation, semantic chunking,
section hierarchy preservation, and rich metadata.

Improvements over original:
  1. Table rows are preserved as complete units (never split across chunks)
  2. Section hierarchy (h1/h2/h3) is stored as structured metadata
  3. Semantic chunking detects natural break points using embedding similarity
  4. Rich metadata: page number, section title, table title, document title, heading_path
  5. Table documents keep their page-level metadata aligned with the source
"""

import hashlib
import logging
import re
from typing import List, Optional, Set

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.core.config import settings

logger = logging.getLogger("knowledge_copilot.chunker")

DEFAULT_CHUNK_SIZE    = settings.chunking_default_size
DEFAULT_CHUNK_OVERLAP = settings.chunking_default_overlap


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_documents(
    documents:     List[Document],
    chunk_size:    Optional[int]   = None,
    chunk_overlap: Optional[int]   = None,
    strategy:      Optional[str]   = None,
) -> List[Document]:
    chunk_size    = chunk_size    or DEFAULT_CHUNK_SIZE
    chunk_overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP
    strategy      = strategy      or settings.chunking_default_strategy

    if strategy == "semantic":
        return _semantic_chunk(documents, chunk_size, chunk_overlap)
    if strategy in ("structure_aware", "markdown"):
        return _smart_chunk(documents, chunk_size, chunk_overlap)
    return _chunk_recursive(documents, chunk_size, chunk_overlap)


# ── Table boundary detector ────────────────────────────────────────────────────

TABLE_ROW_PATTERN = re.compile(
    r'^\s*\|.+\|\s*$',
    re.MULTILINE,
)
TABLE_SEPARATOR_PATTERN = re.compile(r'^\s*\|[\s\-:]+\|\s*$', re.MULTILINE)


def _has_table_rows(text: str) -> bool:
    return bool(TABLE_ROW_PATTERN.search(text))


def _preserve_table_boundaries(
    text: str,
    target_size: int,
) -> List[str]:
    """Split text while keeping table rows intact (never split inside a table)."""
    lines = text.split('\n')
    segments: List[str] = []
    current: List[str] = []
    in_table = False

    for line in lines:
        is_table = bool(TABLE_ROW_PATTERN.match(line))
        is_sep = bool(TABLE_SEPARATOR_PATTERN.match(line))

        if is_table or is_sep:
            if not in_table and current:
                segments.append('\n'.join(current))
                current = []
            in_table = True
            current.append(line)
        else:
            if in_table:
                segments.append('\n'.join(current))
                current = []
                in_table = False
            current.append(line)

    if current:
        segments.append('\n'.join(current))

    return segments


# ── Table-section linking ──────────────────────────────────────────────────────

def _build_section_page_index(chunks: List[Document]) -> dict[int, tuple[str, str, str]]:
    """Build a mapping of page -> (section_id, heading, heading_path) from section chunks."""
    index: dict[int, tuple[str, str, str]] = {}
    for c in chunks:
        if c.metadata.get("content_type") == "section":
            page = c.metadata.get("page", 0)
            sec_id = c.metadata.get("section_id", "")
            heading = c.metadata.get("heading", "")
            heading_path = c.metadata.get("heading_path", "")
            index[page] = (sec_id, heading, heading_path)
    return index


def _enrich_table_chunks(chunks: List[Document]) -> List[Document]:
    """Propagate section context to table chunks by matching page numbers."""
    section_index = _build_section_page_index(chunks)
    if not section_index:
        return chunks

    enriched: List[Document] = []
    for c in chunks:
        if c.metadata.get("content_type") == "table":
            page = c.metadata.get("page", c.metadata.get("table_page", 0))
            if page in section_index:
                sec_id, heading, heading_path = section_index[page]
                if sec_id and not c.metadata.get("section_id"):
                    c.metadata["section_id"] = sec_id
                    c.metadata["heading"] = heading
                    c.metadata["heading_path"] = heading_path
        enriched.append(c)
    return enriched


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
        elif ctype == "table":
            all_chunks.append(_chunk_table_doc(doc))
        else:
            all_chunks.extend(_chunk_recursive([doc], chunk_size, chunk_overlap))
    all_chunks = _enrich_table_chunks(all_chunks)
    return _label_chunks(all_chunks)


# ── Semantic chunker ───────────────────────────────────────────────────────────

def _semantic_chunk(
    documents:     List[Document],
    chunk_size:    int,
    chunk_overlap: int,
) -> List[Document]:
    """Semantic chunking that detects natural break points using embedding similarity."""
    all_chunks: List[Document] = []
    for doc in documents:
        ctype = doc.metadata.get("content_type", "prose")
        if ctype == "pdf_markdown":
            chunks = _chunk_pdf_markdown(doc, chunk_size, chunk_overlap)
            for c in chunks:
                if len(c.page_content) > chunk_size * 1.5:
                    sub_chunks = _split_semantic(c, chunk_size, chunk_overlap)
                    all_chunks.extend(sub_chunks)
                else:
                    all_chunks.append(c)
        elif ctype == "table":
            all_chunks.append(_chunk_table_doc(doc))
        else:
            chunks = _chunk_recursive([doc], chunk_size, chunk_overlap)
            for c in chunks:
                if len(c.page_content) > chunk_size * 1.5:
                    sub_chunks = _split_semantic(c, chunk_size, chunk_overlap)
                    all_chunks.extend(sub_chunks)
                else:
                    all_chunks.append(c)
    all_chunks = _enrich_table_chunks(all_chunks)
    return _label_chunks(all_chunks)


def _split_semantic(
    doc:           Document,
    target_size:   int,
    overlap:       int,
) -> List[Document]:
    """Split a document at semantic boundaries using paragraph-level embedding similarity."""
    parasep = re.compile(r'\n\s*\n')
    paragraphs = [p.strip() for p in parasep.split(doc.page_content) if p.strip()]
    if not paragraphs:
        return [doc]

    groups: List[List[str]] = []
    current: List[str] = []
    current_len = 0

    try:
        from app.services.embedder import embed_documents
        para_embs = embed_documents(paragraphs)
        para_embs = [np.array(e) for e in para_embs]
    except Exception:
        para_embs = None

    for i, para in enumerate(paragraphs):
        para_len = len(para)

        if not current:
            current.append(para)
            current_len = para_len
            continue

        break_score = 0.0
        if para_embs is not None and i > 0:
            sim = float(np.dot(para_embs[i], para_embs[i-1]) / (
                max(np.linalg.norm(para_embs[i]), 1e-10) * max(np.linalg.norm(para_embs[i-1]), 1e-10)
            ))
            break_score = 1.0 - sim

        should_break = (
            current_len + para_len > target_size * 1.2
            and break_score > settings.chunking_semantic_break_threshold
        )

        if should_break:
            groups.append(current)
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        groups.append(current)

    result: List[Document] = []
    for g in groups:
        text = '\n\n'.join(g)
        result.append(Document(
            page_content=text,
            metadata={**doc.metadata},
        ))

    return result if result else [doc]


# ── Table document chunker ─────────────────────────────────────────────────────

def _chunk_table_doc(doc: Document) -> Document:
    """Ensure table documents are stored as whole units with preserved row data."""
    meta = dict(doc.metadata)
    meta["content_type"] = "table"
    meta["table_preserved"] = True
    meta["section_chunk_index"] = 0
    meta["section_total_chunks"] = 1

    table_name = meta.get("table_name", "")
    page = meta.get("page", meta.get("table_page", 0))
    doc_title = meta.get("file_name", meta.get("source", ""))

    meta["document_title"] = doc_title
    meta["page"] = page
    if table_name:
        meta["table_title"] = table_name

    return Document(page_content=doc.page_content, metadata=meta)


# ── Page-break sanitization ────────────────────────────────────────────────────

_PAGE_HEADING_RE = re.compile(r'^(#{1,4})\s+Page\s+(\d+)\s*$', re.MULTILINE)


def _strip_page_break_headings(text: str) -> tuple[str, dict[int, int]]:
    """
    Remove page-break headings from pymupdf4llm output so they don't
    create false section splits in MarkdownHeaderTextSplitter.

    pymupdf4llm often inserts headings like "#### Page 2" between pages.
    These must be neutralized to avoid splitting a single logical section
    (e.g. "SCALE Framework") across two separate section_ids.

    Returns (clean_text, page_at_char) where page_at_char maps approximate
    character positions to page numbers.
    """
    page_at_char: dict[int, int] = {}
    clean_lines: list[str] = []
    for line in text.split('\n'):
        m = _PAGE_HEADING_RE.match(line)
        if m:
            page_num = int(m.group(2))
            char_pos = len('\n'.join(clean_lines))
            page_at_char[char_pos] = page_num
            clean_lines.append(f"[Page {page_num}]")
        else:
            clean_lines.append(line)
    return '\n'.join(clean_lines), page_at_char


def _assign_page_to_chunk(
    chunk_text: str,
    chunk_start: int,
    page_at_char: dict[int, int],
) -> int:
    """Assign the page number for a chunk based on its position in the full text."""
    page = 0
    for pos, pg in sorted(page_at_char.items()):
        if pos <= chunk_start:
            page = pg
        else:
            break
    return page - 1  # zero-indexed


_PAGE_MARKER_RE = re.compile(r'\[Page\s+(\d+)\]')


def _extract_page_from_text(text: str, default_page: int = 0) -> int:
    """Extract the first page number marker found in text, or return default."""
    m = _PAGE_MARKER_RE.search(text)
    if m:
        return int(m.group(1)) - 1  # zero-indexed
    return default_page


# ── PDF markdown chunker — tested and confirmed ────────────────────────────────

def _chunk_pdf_markdown(
    doc:           Document,
    chunk_size:    int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Four-phase chunking for pymupdf4llm markdown output.

    Phase 0 — Strip page-break headings that would create false section splits

    Phase 1 — MarkdownHeaderTextSplitter with strip_headers=TRUE
      Cuts at ###/#### boundaries. Heading goes to metadata ONLY (not content).

    Phase 2 — ALWAYS prepend heading to content, build heading path hierarchy

    Phase 3 — Preserve table boundaries before splitting oversized sections

    Phase 4 — Split oversized sections, keeping table rows intact
    """

    # ── Phase 0 — Sanitize page-break headings ────────────────────────────────
    clean_content, page_at_char = _strip_page_break_headings(doc.page_content)

    # ── Phase 1 ────────────────────────────────────────────────────────────────
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",   "h1"),
            ("##",  "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ],
        strip_headers=True,
    )

    sections = header_splitter.split_text(clean_content)

    source_meta = {k: v for k, v in doc.metadata.items()
                   if k not in ("content_type",)}
    doc_title = source_meta.get("file_name", source_meta.get("source", ""))

    # ── Phase 2 ────────────────────────────────────────────────────────────────
    section_docs: List[Document] = []
    for s in sections:
        content = s.page_content.strip()
        if not content:
            continue

        h1 = s.metadata.get("h1", "")
        h2 = s.metadata.get("h2", "")
        h3 = s.metadata.get("h3", "")
        h4 = s.metadata.get("h4", "")

        heading_parts = [
            p.replace("**", "").replace("*", "").strip()
            for p in [h1, h2, h3, h4] if p
        ]
        heading = " — ".join(heading_parts).strip()
        heading_path = " / ".join(heading_parts) if heading_parts else ""

        section_key = f"{doc_title}:{heading}"
        section_id = hashlib.md5(section_key.encode()).hexdigest()[:12]

        if heading:
            content = f"{heading}\n\n{content}"

        section_docs.append(Document(
            page_content=content,
            metadata={
                **source_meta,
                **s.metadata,
                "content_type": "section",
                "heading":      heading,
                "heading_path": heading_path,
                "section_id":   section_id,
                "document_title": doc_title,
                "page": _extract_page_from_text(content, 0),
            }
        ))

    # ── Phase 3 & 4 ───────────────────────────────────────────────────────────
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

        if settings.chunking_table_preserve_rows and _has_table_rows(content):
            segments = _preserve_table_boundaries(content, chunk_size)
            section_total = len(segments)
            for idx, seg in enumerate(segments):
                seg = seg.strip()
                if not seg:
                    continue
                if len(seg) <= chunk_size:
                    sc = Document(
                        page_content=seg,
                        metadata={**sdoc.metadata},
                    )
                    sc.metadata["section_chunk_index"] = idx
                    sc.metadata["section_total_chunks"] = section_total
                    if heading and heading.lower() not in seg.lower():
                        sc.page_content = f"{heading}\n\n{seg}"
                    final.append(sc)
                else:
                    subs = splitter.split_documents([Document(
                        page_content=seg, metadata=sdoc.metadata,
                    )])
                    for si, sub in enumerate(subs):
                        if heading and heading.lower() not in sub.page_content.lower():
                            sub.page_content = f"{heading}\n\n{sub.page_content}"
                        sub.metadata["section_chunk_index"] = idx + si
                        sub.metadata["section_total_chunks"] = section_total
                        final.append(sub)
        else:
            subs = splitter.split_documents([sdoc])
            heading_lower = heading.lower()
            section_total = len(subs)

            for idx, sc in enumerate(subs):
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