from typing import List, Literal
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

'''
Raw Documents
   ↓
chunk_documents()
   ↓
Choose strategy
   ↓
Split text (recursive / markdown)
   ↓
Add metadata (chunk_index, total_chunks)
   ↓
Return chunked documents
'''


# Sensible defaults — tunable per project
DEFAULT_CHUNK_SIZE    = 1000   # characters
DEFAULT_CHUNK_OVERLAP = 200    # characters

#It decides:Which splitting method to use?
def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    strategy: Literal["recursive", "markdown"] = "recursive",
) -> List[Document]:
    """
    Split a list of Documents into smaller chunks.
    Returns a new list of Documents with updated metadata.
    """
    if strategy == "markdown":
        return _chunk_markdown(documents, chunk_size, chunk_overlap)

    return _chunk_recursive(documents, chunk_size, chunk_overlap)


# ── Strategy 1: Recursive Character Splitter ─────────────────────────────────
# Tries to split on paragraphs → sentences → words → characters (in that order).
# Falls back to the next separator only when the current one produces chunks
# that are still too large.

#Tries splitting like a human:Paragraph → Line → Sentence → Word → Character

""""
\n\n" → paragraph
"\n"   → line
". "   → sentence
" "    → word
""     → character (last fallback)

"""

def _chunk_recursive(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # priority order
        length_function=len,
        add_start_index=True,  # tracks character offset in original doc
    )

    chunks = splitter.split_documents(documents)
    return _label_chunks(chunks)


# ── Strategy 2: Markdown-Aware Splitter ──────────────────────────────────────
# First splits on Markdown headers (H1 → H2 → H3), then applies the
# recursive splitter on each section. This keeps header context in metadata.

def _chunk_markdown(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",   "header_1"),
            ("##",  "header_2"),
            ("###", "header_3"),
        ],
        strip_headers=False,  # keep headers in text so context isn't lost
    )

    # Phase 1: split on headers
    """ 
    splits based on: 
    # Heading 1
    ## Heading 2
    ### Heading 3
    """
    header_chunks: List[Document] = []
    for doc in documents:
        splits = header_splitter.split_text(doc.page_content)
        for split in splits:
            split.metadata.update(doc.metadata)  # carry original metadata
            header_chunks.append(split)

    # Phase 2: apply recursive splitter on each header section
    #Header split → then fine-grained split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks = splitter.split_documents(header_chunks)
    return _label_chunks(chunks)


# ── Helpers ───────────────────────────────────────────────────────────────────

#Meta Data Injection
'''
So each chunk knows:

where it came from
its position
context about the whole document
'''


def _label_chunks(chunks: List[Document]) -> List[Document]:
    """Add chunk_index and total_chunks to every chunk's metadata."""
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = total
    return chunks


def get_chunk_stats(chunks: List[Document]) -> dict:
    """Return basic statistics about a set of chunks — useful for debugging."""
    if not chunks:
        return {}

    lengths = [len(c.page_content) for c in chunks]

    return {
        "total_chunks": len(chunks),
        "avg_length":   round(sum(lengths) / len(lengths)),
        "min_length":   min(lengths),
        "max_length":   max(lengths),
        "total_chars":  sum(lengths),
    }