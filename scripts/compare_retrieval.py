"""Side-by-side comparison of retrieval strategies E1-E5 on live data.

A qualitative inspection tool: shows which chunks each strategy returns and how long it takes.
Quantitative evaluation (Recall@K, MRR, nDCG against labelled data) is the evaluation module's
job (M5).

Usage:
    python scripts/compare_retrieval.py "What retriever does RAG use?" ["another question" ...]
        [--top-k 3] [--strategies dense,sparse,hybrid,hybrid_rerank,hybrid_qt_rerank]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.core.resources import Resources  # noqa: E402
from app.core.types import RetrievalStrategy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions", nargs="+")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--strategies", default=",".join(s.value for s in RetrievalStrategy))
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING", json_output=False)
    resources = Resources.from_settings(settings)
    strategies = [RetrievalStrategy(s.strip()) for s in args.strategies.split(",")]
    engine = resources.retrieval_engine

    # Warm-up so model loading time is not attributed to the first strategy.
    engine.dense.embedder.embed_query("warm up")
    if resources.reranker is not None:
        resources.reranker.score("warm up", ["warm up"])

    try:
        for question in args.questions:
            print(f"\n{'=' * 100}\nQ: {question}")
            for strategy in strategies:
                result = engine.retrieve(question, strategy=strategy, top_k=args.top_k)
                t = result.timings_ms
                timing = " ".join(f"{k}={v:.0f}ms" for k, v in t.items())
                print(f"\n  [{strategy.value}] {timing}")
                if len(result.queries) > 1:
                    for q in result.queries[1:]:
                        print(f"      + query: {q}")
                for rank, chunk in enumerate(result.chunks, start=1):
                    pages = chunk.payload.get("page_start"), chunk.payload.get("page_end")
                    page = f"p{pages[0]}" if pages[0] == pages[1] else f"p{pages[0]}-{pages[1]}"
                    snippet = " ".join(chunk.text.split())[:90]
                    print(f"    {rank}. {chunk.score:8.4f} {page:>7} [{chunk.section}] {snippet}")
    finally:
        resources.close()


if __name__ == "__main__":
    main()
