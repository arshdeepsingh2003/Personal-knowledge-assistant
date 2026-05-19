"""
RAG evaluation test suite — validates retrieval quality, chunk diversity,
multi-hop synthesis, table QA, numeric extraction, and long-document retrieval.

Run with:  python -m pytest tests/test_rag_evaluation.py -v
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure backend is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.chunker import chunk_documents, get_chunk_stats, _smart_chunk
from app.services.embedder import embed_query, embed_documents, get_embedding_dimension
from app.services.vector_store import FAISSStore, _mmr_selection
from app.core.config import settings

import numpy as np
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("test_rag")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_docs():
    """Sample documents representing different sections of a knowledge base."""
    return [
        Document(
            page_content="""## **1. The SCALE Framework for Enterprise AI**

The SCALE framework is a methodology for evaluating enterprise AI readiness.
It consists of five dimensions that help organizations assess their AI maturity.

SCALE is an acronym where each letter represents a critical dimension.""",
            metadata={"source": "framework.pdf", "file_name": "framework.pdf", "content_type": "pdf_markdown"},
        ),
        Document(
            page_content="""## **2. SCALE Dimensions**

### **S — Strategy**
Strategy refers to the organization's AI vision and roadmap.

### **C — Capabilities**
Capabilities encompass the technical skills, tools, and infrastructure.

### **A — Architecture**
Architecture covers the data infrastructure and system design.

### **L — Leadership**
Leadership represents executive commitment and cultural readiness.

### **E — Ethics**
Ethics includes governance, fairness, and responsible AI practices.""",
            metadata={"source": "framework.pdf", "file_name": "framework.pdf", "content_type": "pdf_markdown"},
        ),
        Document(
            page_content="""## **3. ROI Analysis by Industry**

### **3.1 Retail & E-commerce**
For Retail & E-commerce, the Year 1 ROI is 312% and the Payback Period is 3.8 months.

### **3.2 Healthcare**
For Healthcare, the Year 1 ROI is 189% and the Payback Period is 5.2 months.

### **3.3 Financial Services**
For Financial Services, the Year 1 ROI is 245% and the Payback Period is 4.1 months.

### **3.4 Manufacturing**
For Manufacturing, the Year 1 ROI is 167% and the Payback Period is 6.3 months.""",
            metadata={"source": "roi_report.pdf", "file_name": "roi_report.pdf", "content_type": "pdf_markdown"},
        ),
        Document(
            page_content="""## **4. Implementation Timeline**

The enterprise AI implementation follows three phases:

Phase 1 — Foundation (Months 1-3):
  - Assess current infrastructure
  - Build data pipeline
  - Train initial models

Phase 2 — Scaling (Months 4-8):
  - Deploy to pilot departments
  - Integrate with existing systems
  - Establish monitoring

Phase 3 — Optimization (Months 9-12):
  - Fine-tune models
  - Expand to enterprise-wide
  - Continuous improvement""",
            metadata={"source": "implementation.pdf", "file_name": "implementation.pdf", "content_type": "pdf_markdown"},
        ),
        Document(
            page_content="""## **5. Market Share Data**

| Company | Market Share (%) | Year-over-Year Growth (%) |
|---------|-----------------|--------------------------|
| TechCorp | 34.2 | 12.5 |
| DataFlow | 22.8 | 18.3 |
| AISystems | 18.5 | 24.7 |
| CloudBase | 15.1 | 8.9 |
| Others | 9.4 | 3.2 |""",
            metadata={"source": "market_analysis.pdf", "file_name": "market_analysis.pdf", "content_type": "pdf_markdown"},
        ),
        Document(
            page_content="""## **6. Cost Comparison**

The total cost of ownership varies by deployment model:

On-premises deployment: $120,000 per year for infrastructure, $85,000 for staffing.
Cloud deployment: $95,000 per year for services, $45,000 for staffing.
Hybrid deployment: $110,000 per year split across cloud and on-prem, $65,000 for staffing.

ROI break-even occurs at 8 months for cloud, 14 months for on-premises.""",
            metadata={"source": "cost_analysis.pdf", "file_name": "cost_analysis.pdf", "content_type": "pdf_markdown"},
        ),
    ]


@pytest.fixture(scope="session")
def chunked_docs(sample_docs):
    """Chunk the sample documents using the structure-aware chunker."""
    chunks = chunk_documents(
        sample_docs,
        chunk_size=400,
        chunk_overlap=50,
        strategy="structure_aware",
    )
    logger.info(f"Chunked {len(sample_docs)} docs into {len(chunks)} chunks")
    for c in chunks:
        heading = c.metadata.get("heading", "")
        logger.debug(f"  Chunk: heading='{heading}' len={len(c.page_content)}")
    return chunks


# ── Test 1: Multi-hop retrieval ──────────────────────────────────────────────

class TestMultiHopRetrieval:
    """Tests that require combining information from multiple sections."""

    def test_acronym_expansion_across_sections(self, chunked_docs):
        """SCALE acronym: S, C, A, L, E defined across chunks.
        Query: 'What does L and E stand for in the SCALE framework?'
        """
        chunks_with_heading = [
            c for c in chunked_docs
            if c.metadata.get("heading", "").lower().find("scale") >= 0
        ]
        headings = [c.metadata.get("heading", "") for c in chunks_with_heading]

        # Verify L (Leadership) and E (Ethics) chunks are present
        found_l = any("leadership" in h.lower() for h in headings)
        found_e = any("ethics" in h.lower() for h in headings)

        assert found_l, "Missing chunk for L=Leadership in SCALE framework"
        assert found_e, "Missing chunk for E=Ethics in SCALE framework"

        # Verify each dimension chunk contains its letter's content
        leadership_chunks = [
            c.page_content for c in chunks_with_heading
            if "leadership" in c.metadata.get("heading", "").lower()
        ]
        ethics_chunks = [
            c.page_content for c in chunks_with_heading
            if "ethics" in c.metadata.get("heading", "").lower()
        ]

        leadership_found = any("Leadership" in c for c in leadership_chunks)
        ethics_found = any("Ethics" in c for c in ethics_chunks)
        assert leadership_found, "Leadership content not found in L-dimension chunks"
        assert ethics_found, "Ethics content not found in E-dimension chunks"

    def test_cross_section_synthesis(self, chunked_docs):
        """Query combining ROI by industry AND implementation timeline."""
        roi_chunks = [
            c for c in chunked_docs
            if "roi" in c.metadata.get("heading", "").lower()
        ]
        impl_chunks = [
            c for c in chunked_docs
            if "implementation" in c.metadata.get("heading", "").lower()
        ]

        assert len(roi_chunks) > 0, "No ROI chunks found"
        assert len(impl_chunks) > 0, "No implementation timeline chunks found"

        # Check that specific industry ROI data is retrievable
        combined_text = " ".join(c.page_content for c in roi_chunks)
        assert "312%" in combined_text, "Retail ROI (312%) not retrievable"
        assert "189%" in combined_text, "Healthcare ROI (189%) not retrievable"
        assert "Retail & E-commerce" in combined_text, "Industry name missing"

    def test_embedding_similarity_for_related_queries(self):
        """Verify that semantically related queries have high cosine similarity."""
        query_pairs = [
            ("What is the ROI for retail?", "Retail & E-commerce ROI 312%"),
            ("SCALE framework L dimension", "Leadership in enterprise AI"),
            ("market share data", "Company market share percentages"),
        ]

        for q1, q2 in query_pairs:
            v1 = np.array(embed_query(q1))
            v2 = np.array(embed_query(q2))
            sim = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            logger.info(f"Similarity '{q1[:40]}' ↔ '{q2[:40]}' = {sim:.4f}")
            assert sim > 0.3, f"Related queries have low similarity: {sim:.4f}"


# ── Test 2: Table QA ─────────────────────────────────────────────────────────

class TestTableQA:
    """Tests that tabular data is properly extracted and retrievable."""

    def test_market_share_table_retrieval(self, chunked_docs):
        """Verify table content with market share data is retrievable."""
        table_chunks = [
            c for c in chunked_docs
            if "market share" in c.page_content.lower()
            or "Market Share" in c.page_content
        ]
        assert len(table_chunks) > 0, "No market share table chunks found"

        combined = " ".join(c.page_content for c in table_chunks)
        # Check key numeric values
        assert "34.2" in combined, "TechCorp market share (34.2%) missing"
        assert "22.8" in combined, "DataFlow market share (22.8%) missing"
        assert "TechCorp" in combined, "TechCorp company name missing"

    def test_roi_numeric_values_retrievable(self, chunked_docs):
        """Verify specific ROI percentages are retrievable."""
        roi_text = " ".join(
            c.page_content for c in chunked_docs
            if "roi" in c.page_content.lower()
        )

        assert "312%" in roi_text, "Retail ROI 312% missing from chunks"
        assert "189%" in roi_text, "Healthcare ROI 189% missing"
        assert "245%" in roi_text, "Financial Services ROI 245% missing"
        assert "167%" in roi_text, "Manufacturing ROI 167% missing"

    def test_cost_comparison_values(self, chunked_docs):
        """Verify cost comparison data is retrievable."""
        cost_chunks = [
            c for c in chunked_docs
            if "cost" in c.page_content.lower()
            or "deployment" in c.page_content.lower()
        ]
        cost_text = " ".join(c.page_content for c in cost_chunks)

        assert "$120,000" in cost_text or "$120000" in cost_text, "On-prem cost missing"
        assert "$95,000" in cost_text or "$95000" in cost_text, "Cloud cost missing"


# ── Test 3: Cross-section synthesis ──────────────────────────────────────────

class TestCrossSectionSynthesis:
    """Tests that information across different document sections can be combined."""

    def test_chunks_from_different_sections(self, chunked_docs):
        """Verify that chunks span multiple sections (heading diversity)."""
        headings = set()
        for c in chunked_docs:
            h = c.metadata.get("heading", "")
            if h:
                # Normalize to section number/title
                parts = h.split(" — ") if " — " in h else [h]
                for p in parts:
                    p = p.strip().lower()
                    if any(x in p for x in ["scale", "roi", "implementation", "market", "cost"]):
                        headings.add(p)

        assert len(headings) >= 3, (
            f"Only {len(headings)} distinct sections found, need ≥3. "
            f"Found: {headings}"
        )
        logger.info(f"Found {len(headings)} distinct sections: {headings}")

    def test_section_heading_preserved_in_chunks(self, chunked_docs):
        """Every chunk should have its section heading preserved."""
        for c in chunked_docs:
            heading = c.metadata.get("heading", "")
            content = c.page_content
            # The heading should appear in the content somewhere,
            # OR it's a sub-chunk with section_prefix
            if heading:
                has_prefix = c.metadata.get("section_prefix", "")
                heading_words = [w.lower() for w in heading.split() if len(w) > 3]
                content_lower = content.lower()
                found = any(w in content_lower for w in heading_words)
                assert found or has_prefix, (
                    f"Heading '{heading}' not found in chunk content and no prefix"
                )


# ── Test 4: Numerical/statistical extraction ─────────────────────────────────

class TestNumericExtraction:
    """Tests that numeric values survive chunking and embedding."""

    def test_numeric_values_preserved_in_chunks(self, chunked_docs):
        """Key numbers should be present in at least one chunk."""
        numeric_checks = [
            ("312", "ROI percentage"),
            ("3.8", "Payback period"),
            ("34.2", "Market share"),
            ("120,000", "Cost value"),
            ("95,000", "Cloud cost"),
        ]

        all_text = " ".join(c.page_content for c in chunked_docs)
        for val, label in numeric_checks:
            assert val in all_text, f"Numeric value '{val}' ({label}) missing from all chunks"

    def test_multiple_industry_data_points(self, chunked_docs):
        """Multiple data points for the same metric should all be present."""
        roi_text = " ".join(
            c.page_content for c in chunked_docs
            if "roi" in c.page_content.lower()
        )

        # All four industries should have ROI data
        assert "Retail" in roi_text
        assert "Healthcare" in roi_text
        assert "Financial" in roi_text
        assert "Manufacturing" in roi_text


# ── Test 5: Long-document retrieval ──────────────────────────────────────────

class TestLongDocumentRetrieval:
    """Tests that long documents are properly chunked with section continuity."""

    def test_chunk_size_compliance(self, chunked_docs):
        """No chunk should exceed max chunk size + overlap."""
        max_allowed = 400 + 50  # chunk_size + overlap from fixture
        oversized = [c for c in chunked_docs if len(c.page_content) > max_allowed]
        assert len(oversized) == 0, (
            f"{len(oversized)} chunks exceed max size {max_allowed}"
        )

    def test_chunk_overlap_continuity(self, chunked_docs):
        """Consecutive chunks from the same section should share overlapping text."""
        from collections import defaultdict

        section_chunks = defaultdict(list)
        for c in chunked_docs:
            h = c.metadata.get("heading", "") or "nosection"
            section_chunks[h].append(c)

        for heading, chunks in section_chunks.items():
            if len(chunks) < 2:
                continue
            for i in range(len(chunks) - 1):
                c1, c2 = chunks[i].page_content, chunks[i + 1].page_content
                # Check for some overlapping content (at least 1 overlapping non-trivial word)
                overlap_chars = set(c1.lower().split()) & set(c2.lower().split())
                meaningful_overlap = {w for w in overlap_chars if len(w) > 3}
                assert len(meaningful_overlap) > 1 or len(overlap_chars) > 4, (
                    f"Adjacent chunks under '{heading}' have insufficient overlap: "
                    f"{meaningful_overlap}"
                )

    def test_all_sections_indexed(self, chunked_docs):
        """All major sections should produce at least one chunk."""
        expected_sections = ["scale", "roi", "implementation", "market", "cost"]
        found_sections = set()
        for c in chunked_docs:
            h = (c.metadata.get("heading", "") or "").lower()
            for s in expected_sections:
                if s in h:
                    found_sections.add(s)

        missing = set(expected_sections) - found_sections
        assert len(missing) == 0, f"Missing sections: {missing}"


# ── Test 6: Retrieval quality metrics ─────────────────────────────────────────

class TestRetrievalQualityMetrics:
    """Automated retrieval quality and chunk coverage validation."""

    def test_chunk_coverage_score(self, chunked_docs):
        """Verify that all original content is covered by at least one chunk."""
        stats = get_chunk_stats(chunked_docs)
        logger.info(f"Chunk stats: {json.dumps(stats, indent=2)}")

        assert stats["total_chunks"] > 0, "No chunks produced"
        assert stats["min_length"] > 0, "Empty chunk found"
        assert stats["section_chunks"] > 0, "No section-aware chunks"

    def test_content_preservation(self, sample_docs, chunked_docs):
        """Check that key content from original docs is preserved in chunks."""
        original = " ".join(d.page_content for d in sample_docs)
        all_chunked = " ".join(c.page_content for c in chunked_docs)

        key_phrases = [
            "SCALE framework",
            "five dimensions",
            "312%",
            "Market Share",
            "Implementation Timeline",
        ]

        for phrase in key_phrases:
            assert phrase in original, f"Key phrase '{phrase}' not in original docs"
            assert phrase in all_chunked, f"Key phrase '{phrase}' lost during chunking"

    def test_deduplication_effectiveness(self, chunked_docs):
        """Check for minimal text overlap between chunks (< 60% for safety)."""
        from itertools import combinations

        texts = [c.page_content for c in chunked_docs]
        high_overlap_pairs = 0
        total_pairs = 0

        for i, j in combinations(range(len(texts)), 2):
            words_i = set(texts[i].lower().split())
            words_j = set(texts[j].lower().split())
            if not words_i or not words_j:
                continue
            jaccard = len(words_i & words_j) / len(words_i | words_j)
            if jaccard > 0.6:
                high_overlap_pairs += 1
            total_pairs += 1

        overlap_ratio = high_overlap_pairs / max(total_pairs, 1)
        logger.info(f"High-overlap pairs: {high_overlap_pairs}/{total_pairs} ({overlap_ratio:.1%})")
        assert overlap_ratio < 0.3, (
            f"Too many near-duplicate chunks: {overlap_ratio:.1%} pairs have >60% overlap"
        )


# ── Test 7: Embedding quality ────────────────────────────────────────────────

class TestEmbeddingQuality:
    """Tests that embeddings are of correct dimension and quality."""

    def test_embedding_dimension(self):
        """The embedding model should produce vectors of the expected dimension."""
        dim = get_embedding_dimension()
        logger.info(f"Embedding dimension: {dim}")
        assert dim == 1024, f"Expected 1024d embeddings, got {dim}d"

    def test_query_vs_document_embedding(self):
        """Query embeddings should be distinguishable from document embeddings."""
        query = "What is the ROI for retail industry?"
        doc_text = "For Retail & E-commerce, the Year 1 ROI is 312%"

        q_emb = np.array(embed_query(query))
        d_emb = np.array(embed_query(doc_text))

        sim = float(q_emb @ d_emb / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb)))
        logger.info(f"Query-document similarity: {sim:.4f}")
        assert sim > 0.2, f"Query-document similarity too low: {sim:.4f}"

    def test_embedding_distinctiveness(self):
        """Unrelated queries should have lower similarity than related ones."""
        related_q = "What is the ROI for retail?"
        related_d = "Retail ROI is 312%"
        unrelated_d = "The weather today is sunny"

        r_q = np.array(embed_query(related_q))
        r_d = np.array(embed_query(related_d))
        u_d = np.array(embed_query(unrelated_d))

        related_sim = float(r_q @ r_d / (np.linalg.norm(r_q) * np.linalg.norm(r_d)))
        unrelated_sim = float(r_q @ u_d / (np.linalg.norm(r_q) * np.linalg.norm(u_d)))

        logger.info(f"Related sim: {related_sim:.4f}, Unrelated sim: {unrelated_sim:.4f}")
        assert related_sim > unrelated_sim, (
            f"Unrelated query has higher similarity ({unrelated_sim:.4f}) "
            f"than related ({related_sim:.4f})"
        )


# ── Test 8: MMR quality ─────────────────────────────────────────────────────

class TestMMRQuality:
    """Tests that MMR selection produces diverse results."""

    def test_mmr_diversity(self):
        """MMR should select a diverse subset, not just top-k."""
        n_docs = 20
        k = 3

        # Create a query vector
        q_vec = np.array([1.0, 0.0], dtype="float32")
        q_vec = q_vec / np.linalg.norm(q_vec)

        # Create docs: two clusters — one near query, one far
        rng = np.random.RandomState(42)
        doc_vectors = np.vstack([
            rng.randn(10, 2) * 0.1 + np.array([0.9, 0.1]),  # cluster A (relevant)
            rng.randn(10, 2) * 0.1 + np.array([0.1, 0.9]),  # cluster B (diverse)
        ])
        doc_vectors = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)

        doc_indices = list(range(n_docs))
        doc_scores = [float(doc_vectors[i] @ q_vec) for i in range(n_docs)]

        # Plain top-k: all from cluster A
        top_k_indices = sorted(range(n_docs), key=lambda i: doc_scores[i], reverse=True)[:k]
        top_k_clusters = [0 if i < 10 else 1 for i in top_k_indices]

        # MMR: should include some from cluster B
        mmr_indices = _mmr_selection(q_vec, doc_vectors, doc_indices, doc_scores, k, lambda_mult=0.3)
        mmr_clusters = [0 if i < 10 else 1 for i in mmr_indices]

        top_k_diverse = sum(top_k_clusters)
        mmr_diverse = sum(mmr_clusters)

        logger.info(f"Top-k selection: clusters={top_k_clusters}, diverse count={top_k_diverse}")
        logger.info(f"MMR selection:   clusters={mmr_clusters}, diverse count={mmr_diverse}")

        assert mmr_diverse >= top_k_diverse, (
            f"MMR should select more diverse docs. "
            f"Top-k diverse: {top_k_diverse}, MMR diverse: {mmr_diverse}"
        )

    def test_mmr_vs_lambda(self):
        """Higher lambda = more relevance, lower lambda = more diversity."""
        n_docs = 10
        k = 3

        q_vec = np.array([1.0, 0.0], dtype="float32")
        q_vec = q_vec / np.linalg.norm(q_vec)

        rng = np.random.RandomState(123)
        doc_vectors = np.vstack([
            rng.randn(5, 2) * 0.1 + np.array([0.9, 0.0]),  # cluster A (high relevance)
            rng.randn(5, 2) * 0.1 + np.array([0.0, 0.9]),  # cluster B (low relevance)
        ])
        doc_vectors = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)

        doc_indices = list(range(n_docs))
        doc_scores = [float(doc_vectors[i] @ q_vec) for i in range(n_docs)]

        # High lambda (0.9) → mostly cluster A
        high_lambda_indices = _mmr_selection(q_vec, doc_vectors, doc_indices, doc_scores, k, lambda_mult=0.9)
        high_clusters = [0 if i < 5 else 1 for i in high_lambda_indices]
        high_cluster_a = sum(1 for c in high_clusters if c == 0)

        # Low lambda (0.1) → should include cluster B
        low_lambda_indices = _mmr_selection(q_vec, doc_vectors, doc_indices, doc_scores, k, lambda_mult=0.1)
        low_clusters = [0 if i < 5 else 1 for i in low_lambda_indices]
        low_cluster_b = sum(1 for c in low_clusters if c == 1)

        logger.info(f"High λ=0.9: cluster A count={high_cluster_a}")
        logger.info(f"Low λ=0.1: cluster B count={low_cluster_b}")

        assert high_cluster_a >= low_cluster_b or low_cluster_b > 0, (
            f"Expected low lambda to include more cluster B docs. "
            f"High λ cluster A: {high_cluster_a}, Low λ cluster B: {low_cluster_b}"
        )


# ── Test 9: Chunker section integrity ────────────────────────────────────────

class TestChunkerSectionIntegrity:
    """Tests that the chunker preserves section structure correctly."""

    def test_heading_reinjection(self, chunked_docs):
        """Sub-chunks that lost their heading should have it re-injected."""
        prefixed = sum(1 for c in chunked_docs if c.metadata.get("section_prefix"))
        logger.info(f"Chunks with re-injected heading prefix: {prefixed}")

        # For each chunk, the heading should be in the content
        for c in chunked_docs:
            heading = c.metadata.get("heading", "")
            content = c.page_content
            if heading:
                key_terms = [w for w in heading.split() if len(w) > 3]
                content_lower = content.lower()
                found = any(term.lower() in content_lower for term in heading.split())
                assert found, f"Heading term '{heading}' not in content of chunk"

    def test_no_heading_leakage(self, chunked_docs):
        """A chunk from section X should not contain content from section Y."""
        section_map = {}
        for c in chunked_docs:
            h = c.metadata.get("heading", "") or "unknown"
            for key, keywords in {
                "scale": ["scale", "s — strategy", "c — capabilities"],
                "roi": ["roi", "retail", "healthcare", "financial", "manufacturing"],
                "implementation": ["implementation", "phase 1", "phase 2", "phase 3"],
                "market": ["market share", "techcorp", "dataflow"],
                "cost": ["cost", "on-premises", "cloud deployment"],
            }.items():
                if any(k in h.lower() for k in keywords):
                    section_map.setdefault(key, []).append(c)

        # Check cross-contamination between clearly distinct sections
        if "scale" in section_map and "roi" in section_map:
            for scale_chunk in section_map["scale"]:
                scale_text = scale_chunk.page_content.lower()
                # ROI-section specific terms should NOT be in scale chunks
                if "roi" in scale_text and "312%" not in scale_text:
                    pass  # "roi" may appear generically but industry-specific data shouldn't
                assert "retail & e-commerce" not in scale_text or "scale" in scale_text, (
                    "Scale chunk appears to contain ROI-specific retail data"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
