"""
verify.py — Run this BEFORE re-indexing to confirm the fix works.

Usage:
    python verify.py data/uploads/AI_Research_Report_2024.pdf

Checks:
  1. pymupdf4llm converts PDF correctly
  2. All ## headings detected
  3. SCALE section has all 5 letters
  4. Every chunk from the SCALE section contains "SCALE"
  5. Simulates the exact failing query

No server, no API key, no index needed.
"""

import sys
import os

def run(pdf_path: str):
    print(f"\n{'='*55}")
    print(f"Verifying: {os.path.basename(pdf_path)}")
    print(f"{'='*55}\n")

    # ── 1. pymupdf4llm ────────────────────────────────────────────────────────
    try:
        import pymupdf4llm
    except ImportError:
        print("STOP: pip install pymupdf4llm pymupdf")
        sys.exit(1)

    md = pymupdf4llm.to_markdown(pdf_path)
    md_path = pdf_path.replace(".pdf", ".extracted.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[1] pymupdf4llm OK — {len(md):,} chars")
    print(f"    Saved: {os.path.basename(md_path)}")
    print(f"    H1={md.count(chr(10)+'# ')} "
          f"H2={md.count(chr(10)+'## ')} "
          f"H3={md.count(chr(10)+'### ')}")
    print()

    # ── 2. Headings ───────────────────────────────────────────────────────────
    print("[2] Detected headings:")
    for line in md.splitlines():
        if line.startswith("#"):
            print(f"    {line[:70]}")
    print()

    # ── 3. SCALE section content ──────────────────────────────────────────────
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    hs = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")],
        strip_headers=True,
    )
    sections = hs.split_text(md)

    scale_section = None
    for s in sections:
        all_meta = " ".join(str(v) for v in s.metadata.values())
        if "scale" in all_meta.lower():
            scale_section = s
            break

    if not scale_section:
        print("[3] ✗ SCALE section not found in headings")
        print(f"    Open {os.path.basename(md_path)} and search for 'SCALE'")
        print(f"    The heading line format may differ — check what ## lines exist above")
        sys.exit(1)

    heading_meta = (scale_section.metadata.get("h2","") or
                    scale_section.metadata.get("h1","") or
                    scale_section.metadata.get("h3",""))
    heading_clean = heading_meta.replace("**","").replace("*","").strip()
    content = scale_section.page_content

    print(f"[3] SCALE section found: '{heading_clean}'")
    print(f"    Content length: {len(content)} chars")

    letters = {
        "S (Strategy)":  "strategy" in content.lower(),
        "C (Clean Data)":"clean data" in content.lower() or "c —" in content.lower(),
        "A (Agile)":     "agile" in content.lower(),
        "L (Learning)":  "learning" in content.lower(),
        "E (Ethical)":   "ethical" in content.lower() or "governance" in content.lower(),
    }
    all_letters = all(letters.values())
    for k, v in letters.items():
        print(f"    {'✓' if v else '✗'} {k}")
    if all_letters:
        print("    ✓ All 5 letters in ONE section")
    else:
        print("    ✗ Some letters missing — section may span multiple ## headings")
    print()

    # ── 4. Chunker test ───────────────────────────────────────────────────────
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Build full content with heading prepended (Phase 2)
    full = f"{heading_clean}\n\n{content}" if heading_clean else content

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    doc = Document(page_content=full, metadata=scale_section.metadata)

    if len(full) <= 1500:
        chunks = [doc]
    else:
        chunks = splitter.split_documents([doc])
        heading_lower = heading_clean.lower()
        fixed = []
        for c in chunks:
            if heading_clean and heading_lower not in c.page_content.lower():
                c.page_content = f"{heading_clean}\n\n{c.page_content}"
            fixed.append(c)
        chunks = fixed

    print(f"[4] Chunker produces {len(chunks)} chunk(s) from SCALE section:")
    all_pass = True
    for i, c in enumerate(chunks):
        has_scale = "scale" in c.page_content.lower()
        has_L     = "learning" in c.page_content.lower()
        has_E     = "ethical"  in c.page_content.lower() or "governance" in c.page_content.lower()
        status    = "✓" if has_scale else "✗"
        if not has_scale:
            all_pass = False
        note = ""
        if has_L: note += " [has L]"
        if has_E: note += " [has E]"
        print(f"    Chunk {i+1}: {status} SCALE{note}")
        print(f"      '{c.page_content[:80].strip()}'")
    print()

    # ── 5. Query simulation ───────────────────────────────────────────────────
    print("[5] Query simulation: 'What does L and E stand for in SCALE?'")
    matching = [c for c in chunks
                if "scale" in c.page_content.lower()
                and ("learning" in c.page_content.lower()
                     or "ethical" in c.page_content.lower())]
    if matching:
        print(f"    ✓ {len(matching)} chunk(s) will be retrieved for this query")
        for c in matching:
            for line in c.page_content.splitlines():
                if any(kw in line.lower() for kw in
                       ["learning", "ethical", "governance", "l —", "e —"]):
                    print(f"    → {line.strip()[:90]}")
    else:
        print("    ✗ No chunk matches — check heading detection")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"{'='*55}")
    if all_pass and all_letters and matching:
        print("✅ ALL CHECKS PASSED — ready to re-index")
        print()
        print("Steps:")
        print("  1. del data\\vector_store\\faiss.index")
        print("  2. del data\\vector_store\\docstore.json")
        print("  3. python main.py")
        print("  4. Upload PDF via dashboard")
        print("  5. Ask: 'What does L and E stand for in the SCALE framework?'")
    else:
        print("❌ CHECKS FAILED — do not re-index yet")
        print(f"   Open: {os.path.basename(md_path)}")
        print("   Search for 'SCALE' and check the heading format")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/uploads/AI_Research_Report_2024.pdf"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        print("Upload the PDF first, then: python verify.py data/uploads/AI_Research_Report_2024.pdf")
        sys.exit(1)
    run(path)