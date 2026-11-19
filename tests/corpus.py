"""A tiny two-paper corpus used by retrieval and research-graph tests."""

PAPER_A, PAPER_B = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"
COLL_X, COLL_Y = "c-x", "c-y"
CORPUS = [
    (
        PAPER_A,
        COLL_X,
        "1 Introduction",
        2020,
        "Retrieval augmented generation conditions a language model on retrieved passages.",
    ),
    (
        PAPER_A,
        COLL_X,
        "3 Method",
        2020,
        "We use DPR, a dense passage retriever with a BERT bi-encoder, and BM25 as a baseline.",
    ),
    (
        PAPER_A,
        COLL_X,
        "5 Limitations",
        2020,
        "Limitations include retrieval errors that propagate into hallucinated answers.",
    ),
    (
        PAPER_B,
        COLL_Y,
        "1 Introduction",
        2024,
        "GraphRAG builds a knowledge graph of entities and summarises graph communities.",
    ),
    (
        PAPER_B,
        COLL_Y,
        "4 Results",
        2024,
        "GraphRAG improves comprehensiveness on global sensemaking questions over naive RAG.",
    ),
    (
        PAPER_B,
        COLL_Y,
        "6 Limitations",
        2024,
        "Building the graph index is expensive, and evaluation relies on an LLM judge.",
    ),
]
