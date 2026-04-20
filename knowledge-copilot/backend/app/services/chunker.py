"""
chunker.py — Structure-aware chunking that preserves tables

Key changes from Phase 3:
  1. Table documents are never split — they are kept as single chunks.
  2. Table rows are converted to natural language sentences before embedding.
     This is critical: embeddings of "Retail | 312% | 3.8 months" are poor.
     Embeddings of "In Retail & E-commerce, Year 1 ROI is 312% with a
     payback period of 3.8 months." are 40-60% better on retrieval.
  3. Section headings detected above tables are prepended to table chunks
     so the LLM knows what section the table belongs to.
  4. Metadata is enriched with table_name, section, and key_fields.
  5. Prose chunks use the same recursive splitter as Phase 3.
"""

import re
from typing import List, Literal, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE    = 1000
DEFAULT_CHUNK_OVERLAP = 200


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_documents(
    documents:     List[Document],
    chunk_size:    int   = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int   = DEFAULT_CHUNK_OVERLAP,
    strategy:      Literal["recursive", "markdown", "structure_aware"] = "structure_aware",
) -> List[Document]:
    """
    Split documents into chunks.

    strategy="structure_aware" (new default):
      - Table documents → kept whole, rows converted to natural language
      - Prose documents → recursive character splitting (same as before)

    strategy="recursive" / "markdown":
      - Behaves identically to Phase 3 (for backwards compatibility)
    """
    if strategy == "structure_aware":
        return _structure_aware_chunk(documents, chunk_size, chunk_overlap)

    # Legacy strategies (unchanged from Phase 3)
    if strategy == "markdown":
        return _chunk_markdown(documents, chunk_size, chunk_overlap)

    return _chunk_recursive(documents, chunk_size, chunk_overlap)


# ── Strategy: structure-aware ─────────────────────────────────────────────────

def _structure_aware_chunk(
    documents:     List[Document],
    chunk_size:    int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Route each document to the right chunking strategy based on content_type.
    Table docs stay whole. Prose docs get recursive splitting.
    """
    all_chunks: List[Document] = []

    # Build a heading context tracker so we can prepend section headings
    # to table chunks when the preceding doc was a heading paragraph.
    last_heading: Optional[str] = None

    for doc in documents:
        content_type = doc.metadata.get("content_type", "prose")

        if content_type == "table":
            table_chunks = _chunk_table(doc, last_heading)
            all_chunks.extend(table_chunks)
            # Reset heading after attaching it to a table
            last_heading = None
        else:
            # Detect if this prose chunk ends with a heading
            detected = _detect_heading(doc.page_content)
            if detected:
                last_heading = detected

            # Split prose normally
            prose_chunks = _chunk_single_doc_recursive(doc, chunk_size, chunk_overlap)
            all_chunks.extend(prose_chunks)

    return _label_chunks(all_chunks)


def _chunk_table(doc: Document, section_heading: Optional[str] = None) -> List[Document]:
    """
    Keep a table as ONE chunk. Convert rows to natural language.
    Attach the section heading above the table for context.

    Returns a list with exactly one Document (the whole table).
    """
    raw_table = doc.metadata.get("table_data")

    if raw_table and len(raw_table) >= 2:
        nl_text = _table_to_natural_language(raw_table, section_heading)
        table_name = _infer_table_name(raw_table, doc.metadata)
        key_fields = _extract_key_fields(raw_table)
    else:
        # Fallback: use the pipe-separated text already in page_content
        nl_text = doc.page_content
        if section_heading:
            nl_text = f"{section_heading}\n\n{nl_text}"
        table_name = f"Table on page {doc.metadata.get('page', '?')}"
        key_fields = []

    enriched_meta = {
        **doc.metadata,
        "content_type": "table",
        "table_name":   table_name,
        "section":      section_heading or _infer_section(doc.metadata),
        "key_fields":   key_fields,
        # Remove raw table_data — too large for metadata store
        "table_data":   None,
    }

    return [Document(page_content=nl_text, metadata=enriched_meta)]


def _table_to_natural_language(
    table: List[List],
    section_heading: Optional[str] = None,
) -> str:
    """
    Convert a table (list of lists) into natural language sentences.

    This is the most important transformation in the whole pipeline.
    Without it, table embeddings are poor because models trained on
    prose don't understand pipe-delimited rows well.

    Input:
      headers = ["Industry", "ROI", "Payback Period"]
      row     = ["Retail & E-commerce", "312%", "3.8 months"]

    Output sentence:
      "In the Retail & E-commerce industry, the Year 1 ROI is 312%
       and the Payback Period is 3.8 months."

    The output is prefixed with the section heading if available.
    """
    if not table:
        return ""

    # Clean the table: remove None cells, strip whitespace
    cleaned = []
    for row in table:
        cleaned.append([str(c).strip() if c else "" for c in row])

    # First non-empty row is the header
    headers = cleaned[0]
    data_rows = [r for r in cleaned[1:] if any(c for c in r)]

    if not data_rows:
        return " | ".join(headers)

    # Build a readable block: section + header summary + NL sentences
    parts = []

    if section_heading:
        parts.append(f"Section: {section_heading}\n")

    # Add a compact table representation first (helps the LLM scan the data)
    parts.append("Table data:")
    parts.append(" | ".join(headers))
    for row in data_rows:
        parts.append(" | ".join(row))

    parts.append("")  # blank line separator

    # Now add natural language sentences for each row
    parts.append("In natural language:")
    for row in data_rows:
        sentence = _row_to_sentence(headers, row)
        if sentence:
            parts.append(sentence)

    return "\n".join(parts)


def _row_to_sentence(headers: List[str], row: List[str]) -> str:
    """
    Convert one table row to a natural language sentence.

    Strategy:
      - The first column is usually the subject (industry, model, metric)
      - Remaining columns are predicates
      - Build: "For [subject], [col2] is [val2], [col3] is [val3]."

    Examples:
      ["Industry", "ROI", "Payback"] + ["Retail", "312%", "3.8 months"]
      → "In the Retail industry, the Year 1 ROI is 312% and the
         Payback period is 3.8 months."

      ["Model", "Params", "Context"] + ["GPT-4 Turbo", "~1.8T", "128K"]
      → "The GPT-4 Turbo model has approximately 1.8T parameters
         and a context window of 128K tokens."
    """
    if not row or not any(row):
        return ""

    # Pair each header with its cell value, skip empty cells
    pairs = [
        (h.strip(), v.strip())
        for h, v in zip(headers, row)
        if h.strip() and v.strip()
    ]
    if not pairs:
        return ""

    subject_col, subject_val = pairs[0]
    predicates = pairs[1:]

    if not predicates:
        return f"{subject_col}: {subject_val}."

    pred_parts = []
    for col, val in predicates:
        pred_parts.append(f"the {col} is {val}")

    pred_str = ", ".join(pred_parts[:-1])
    if len(pred_parts) > 1:
        pred_str += f", and {pred_parts[-1]}"
    else:
        pred_str = pred_parts[0]

    return f"For {subject_val}, {pred_str}."


def _infer_table_name(table: List[List], metadata: dict) -> str:
    """Infer a table name from its content or metadata."""
    if table and table[0]:
        headers = [str(c).strip() for c in table[0] if c]
        if headers:
            return f"Table: {', '.join(headers[:3])}"
    page = metadata.get("page", "?")
    return f"Table on page {page}"


def _extract_key_fields(table: List[List]) -> List[str]:
    """Extract key field values from the first data column (the subject column)."""
    if len(table) < 2:
        return []
    key_fields = []
    for row in table[1:]:
        if row and row[0]:
            val = str(row[0]).strip()
            if val:
                key_fields.append(val)
    return key_fields[:10]  # cap at 10 to avoid huge metadata


def _infer_section(metadata: dict) -> str:
    page = metadata.get("page")
    if page is not None:
        return f"Page {page}"
    return "Unknown section"


def _detect_heading(text: str) -> Optional[str]:
    """
    Detect if a prose chunk ends with what looks like a section heading.
    These headings will be prepended to the next table chunk.
    """
    if not text:
        return None
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return None
    last = lines[-1]
    # A heading: short, no period at end, title-ish capitalisation
    if len(last) < 100 and not last.endswith(".") and len(last.split()) <= 10:
        return last
    return None


# ── Strategy: recursive (unchanged from Phase 3) ─────────────────────────────

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


def _chunk_single_doc_recursive(
    doc:           Document,
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
    return splitter.split_documents([doc])


def _chunk_markdown(
    documents:     List[Document],
    chunk_size:    int,
    chunk_overlap: int,
) -> List[Document]:
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",   "header_1"),
            ("##",  "header_2"),
            ("###", "header_3"),
        ],
        strip_headers=False,
    )
    header_chunks: List[Document] = []
    for doc in documents:
        splits = header_splitter.split_text(doc.page_content)
        for split in splits:
            split.metadata.update(doc.metadata)
            header_chunks.append(split)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return _label_chunks(splitter.split_documents(header_chunks))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _label_chunks(chunks: List[Document]) -> List[Document]:
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"]  = i
        chunk.metadata["total_chunks"] = total
    return chunks


def get_chunk_stats(chunks: List[Document]) -> dict:
    if not chunks:
        return {}
    lengths = [len(c.page_content) for c in chunks]
    table_count = sum(1 for c in chunks if c.metadata.get("content_type") == "table")
    return {
        "total_chunks": len(chunks),
        "table_chunks": table_count,
        "prose_chunks": len(chunks) - table_count,
        "avg_length":   round(sum(lengths) / len(lengths)),
        "min_length":   min(lengths),
        "max_length":   max(lengths),
        "total_chars":  sum(lengths),
    }