"""
validate_table_rag.py
Validates that the table RAG pipeline correctly answers
questions that depend on tabular data.

Run from backend/ with venv active:
    python validate_table_rag.py

Prerequisites:
    1. All 6 files replaced (document_loader, chunker, embedder,
       retriever, llm, config)
    2. pip install pdfplumber sentence-transformers langchain-groq
    3. GROQ_API_KEY set in .env
    4. Re-index your PDF: POST /api/v1/documents (re-upload the file)
       IMPORTANT: you MUST re-index after changing the embedding model
       because the new model produces different vector dimensions.
"""

import sys, os, time

# ── 1. Check dependencies ─────────────────────────────────────────────────────
print("Checking dependencies...")
missing = []
for pkg in ["pdfplumber", "sentence_transformers", "langchain_groq", "groq"]:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} — MISSING")
        missing.append(pkg)

if missing:
    print(f"\nInstall missing packages:")
    print(f"  pip install {' '.join(pkg.replace('_','-') for pkg in missing)}")
    sys.exit(1)

print()

# ── 2. Test pdfplumber table extraction ───────────────────────────────────────
print("Testing pdfplumber table extraction...")
try:
    import pdfplumber

    # Look for any PDF in data/uploads
    pdf_path = None
    uploads = "data/uploads"
    if os.path.exists(uploads):
        for f in os.listdir(uploads):
            if f.endswith(".pdf"):
                pdf_path = os.path.join(uploads, f)
                break

    if not pdf_path:
        print("  ⚠ No PDF found in data/uploads — skipping extraction test")
        print("  Upload a PDF first via POST /api/v1/documents")
    else:
        with pdfplumber.open(pdf_path) as pdf:
            total_tables = 0
            for page in pdf.pages:
                tables = page.extract_tables()
                total_tables += len(tables)

        print(f"  ✓ Opened: {os.path.basename(pdf_path)}")
        print(f"  ✓ Tables found: {total_tables}")
        if total_tables == 0:
            print("  ⚠ No tables detected — check if the PDF has real tables vs images")
except Exception as e:
    print(f"  ✗ pdfplumber error: {e}")

print()

# ── 3. Test chunker natural language conversion ───────────────────────────────
print("Testing NL table conversion...")
try:
    from app.services.chunker import _row_to_sentence, _table_to_natural_language

    # Simulate the ROI table from the AI Research Report
    headers = ["Industry", "Avg Implementation Cost", "Year 1 ROI",
               "Payback Period", "Primary Use Case"]
    rows = [
        ["Financial Services", "$2.8M", "187%", "7.2 months", "Fraud detection"],
        ["Retail & E-commerce", "$1.2M", "312%", "3.8 months", "Personalisation"],
        ["Manufacturing", "$1.9M", "221%", "5.4 months", "Predictive maintenance"],
    ]

    for row in rows:
        sentence = _row_to_sentence(headers, row)
        print(f"  ✓ {sentence}")

    full_text = _table_to_natural_language([headers] + rows, "ROI Analysis")
    print(f"\n  Full NL table preview (first 300 chars):")
    print(f"  {full_text[:300]}...")
    print()
except Exception as e:
    print(f"  ✗ Chunker error: {e}")

print()

# ── 4. Test embedding dimension ───────────────────────────────────────────────
print("Testing new embedding model (bge-large)...")
try:
    from app.services.embedder import embed_query, get_embedding_dimension

    t0  = time.time()
    dim = get_embedding_dimension()
    elapsed = time.time() - t0

    if dim == 1024:
        print(f"  ✓ bge-large-en-v1.5 loaded ({dim} dims, {elapsed:.1f}s)")
    elif dim == 384:
        print(f"  ⚠ Got {dim} dims — still using old model (all-MiniLM).")
        print(f"    Check EMBEDDING_MODEL_LOCAL=BAAI/bge-large-en-v1.5 in .env")
    else:
        print(f"  ✓ Model loaded ({dim} dims)")

    # Test query embedding (BGE query instruction applied automatically)
    vec = embed_query("What is the ROI for retail industry?")
    print(f"  ✓ Query embedded ({len(vec)} dims)")
except Exception as e:
    print(f"  ✗ Embedder error: {e}")

print()

# ── 5. Test reranker ──────────────────────────────────────────────────────────
print("Testing BGE reranker...")
try:
    from app.services.retriever import _get_reranker, _rerank_bge

    reranker = _get_reranker()
    if reranker is None:
        print("  ⚠ Reranker is disabled (RERANKER_PROVIDER=none)")
    else:
        # Simulate reranking: the retail ROI chunk should outscore
        # the generic AI market overview chunk for a retail ROI query
        mock_results = [
            {"text": "The global AI market reached $142.3 billion in 2023 representing 38.1% YoY growth.",
             "score": 0.45, "metadata": {"content_type": "prose"}},
            {"text": "For Retail & E-commerce, the Year 1 ROI is 312% and the Payback Period is 3.8 months.",
             "score": 0.38, "metadata": {"content_type": "table"}},
            {"text": "For Financial Services, the Year 1 ROI is 187% and the Payback Period is 7.2 months.",
             "score": 0.36, "metadata": {"content_type": "table"}},
        ]

        query = "What is the ROI for retail industry?"
        reranked = _rerank_bge(reranker, query, mock_results, top_n=3)

        print(f"  Query: '{query}'")
        print(f"  Before rerank (vector scores):")
        for r in mock_results:
            print(f"    [{r['score']:.2f}] {r['text'][:60]}...")
        print(f"  After rerank (cross-encoder scores):")
        for r in reranked:
            print(f"    [{r.get('rerank_score', 0):.4f}] {r['text'][:60]}...")

        # Verify the table chunk with retail ROI moved to top
        if "Retail" in reranked[0]["text"] or "312%" in reranked[0]["text"]:
            print("  ✓ Reranker correctly prioritised the retail ROI chunk")
        else:
            print("  ⚠ Reranker did not prioritise the retail chunk — check model")
except Exception as e:
    print(f"  ✗ Reranker error: {e}")

print()

# ── 6. Test Groq LLM ──────────────────────────────────────────────────────────
print("Testing Groq LLM with table-aware prompt...")
try:
    from app.services.llm import generate_answer

    mock_context = """[1] (ROI Analysis — Table: Industry, Avg Implementation Cost, Year 1 ROI)
Section: ROI Analysis
Table data:
Industry | Year 1 ROI | Payback Period
Retail & E-commerce | 312% | 3.8 months
Financial Services | 187% | 7.2 months
Manufacturing | 221% | 5.4 months

In natural language:
For Retail & E-commerce, the Year 1 ROI is 312% and the Payback Period is 3.8 months.
For Financial Services, the Year 1 ROI is 187% and the Payback Period is 7.2 months.
For Manufacturing, the Year 1 ROI is 221% and the Payback Period is 5.4 months."""

    t0 = time.time()
    answer = generate_answer(
        query   = "What is the ROI for retail and e-commerce? Give the exact number.",
        context = mock_context,
    )
    elapsed = time.time() - t0

    print(f"  Answer ({elapsed:.1f}s): {answer[:200]}")

    if "312" in answer:
        print("  ✓ LLM correctly extracted 312% from table context")
    else:
        print("  ⚠ LLM did not return 312% — check system prompt or model")

except Exception as e:
    print(f"  ✗ LLM error: {e}")

print()
print("=" * 55)
print("Validation complete.")
print()
print("IMPORTANT: If you changed the embedding model,")
print("you MUST re-index all your documents:")
print("  1. DELETE /api/v1/vectorstore/clear")
print("  2. Re-upload each file via POST /api/v1/documents")
print("  Old vectors (384d) are incompatible with new model (1024d).")
print("=" * 55)