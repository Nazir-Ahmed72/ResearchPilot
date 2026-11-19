"""Retrieval and reranking nodes.

``retrieve`` fetches fused candidates for all planned queries in one Qdrant round trip.
``rerank`` scores them against the ORIGINAL question with the cross-encoder (if the strategy
includes reranking), merges them with evidence from earlier rounds, enforces a per-paper cap for
cross-paper questions, and assigns the short citation IDs (E1, E2, ...).
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig, RunnableLambda

from app.core.types import RetrievalStrategy
from app.orchestration.dependencies import ResearchDependencies
from app.orchestration.state import EvidenceItem, ResearchState
from app.retrieval.reranker import rerank as cross_encoder_rerank
from app.retrieval.types import RetrievalFilters, RetrievedChunk

RERANK_STRATEGIES = {RetrievalStrategy.HYBRID_RERANK, RetrievalStrategy.HYBRID_QT_RERANK}


def to_evidence(chunk: RetrievedChunk, eid: str = "") -> EvidenceItem:
    p = chunk.payload
    return EvidenceItem(
        eid=eid,
        chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        title=p.get("title"),
        year=p.get("year"),
        section=p.get("section"),
        page_start=p.get("page_start"),
        page_end=p.get("page_end"),
        text=chunk.text,
        score=round(float(chunk.score), 6),
        scores={k: round(float(v), 6) for k, v in chunk.scores.items()},
    )


def _from_evidence(item: EvidenceItem) -> RetrievedChunk:
    payload = {
        k: item[k]
        for k in ("paper_id", "title", "year", "section", "page_start", "page_end", "text")
    }  # type: ignore[literal-required]
    return RetrievedChunk(item["chunk_id"], payload, item["score"], dict(item["scores"]))


def _is_complex(state: ResearchState) -> bool:
    return (state.get("analysis") or {}).get("complexity") == "complex"


def retrieve(
    state: ResearchState, config: RunnableConfig, deps: ResearchDependencies
) -> dict[str, Any]:
    strategy = RetrievalStrategy(state["strategy"])
    queries = state.get("search_queries") or [state["question"]]
    cfg = deps.config
    if strategy in RERANK_STRATEGIES:
        k = cfg.candidate_k_complex if _is_complex(state) else cfg.candidate_k_simple
    else:
        k = cfg.evidence_k_complex if _is_complex(state) else cfg.evidence_k_simple
    filters = RetrievalFilters(**{**(state.get("filters") or {})})

    def search(_: dict[str, Any]) -> list[RetrievedChunk]:
        return deps.engine.hybrid.search(
            queries,
            max(k, deps.engine.config.candidate_k),
            filters,
            use_dense=strategy != RetrievalStrategy.SPARSE,
            use_sparse=strategy != RetrievalStrategy.DENSE,
        )

    # A named child run, so traces show queries in and chunk ids/scores out.
    fused = RunnableLambda(search, name="hybrid_search").invoke(
        {"queries": queries, "strategy": strategy.value, "k": k}, config
    )
    cross_paper = (state.get("analysis") or {}).get("requires_cross_paper", False)
    if cross_paper:  # noqa: SIM108 - kept as a block for the explanation below
        # Diversify BEFORE reranking: otherwise a term unique to one paper (e.g. "GraphRAG")
        # can fill every candidate slot, and a later per-paper cap has nothing else to choose.
        chunks = diversify(fused, k, deps.config.max_chunks_per_paper)
    else:
        chunks = fused[:k]
    return {
        "candidates": [to_evidence(c) for c in chunks],
        "_trace": {
            "queries": len(queries),
            "candidates": len(chunks),
            "chunk_ids": [c.chunk_id for c in chunks],
        },
    }


def diversify(ranked: list[RetrievedChunk], k: int, per_paper_cap: int) -> list[RetrievedChunk]:
    """Top-k by rank with at most ``per_paper_cap`` per paper; if that leaves slots empty (few
    papers in scope), fill them with the best remaining chunks."""
    chosen: list[RetrievedChunk] = []
    per_paper: dict[str, int] = {}
    for chunk in ranked:
        if per_paper.get(chunk.paper_id, 0) < per_paper_cap:
            chosen.append(chunk)
            per_paper[chunk.paper_id] = per_paper.get(chunk.paper_id, 0) + 1
            if len(chosen) >= k:
                return chosen
    chosen_ids = {c.chunk_id for c in chosen}
    chosen += [c for c in ranked if c.chunk_id not in chosen_ids][: k - len(chosen)]
    return chosen


def select_evidence(
    ranked: list[EvidenceItem], k: int, per_paper_cap: int | None
) -> list[EvidenceItem]:
    chosen: list[EvidenceItem] = []
    per_paper: dict[str, int] = {}
    seen: set[str] = set()
    for item in ranked:
        if item["chunk_id"] in seen:
            continue
        if per_paper_cap is not None and per_paper.get(item["paper_id"], 0) >= per_paper_cap:
            continue
        seen.add(item["chunk_id"])
        per_paper[item["paper_id"]] = per_paper.get(item["paper_id"], 0) + 1
        chosen.append(item)
        if len(chosen) >= k:
            break
    return [EvidenceItem(**{**item, "eid": f"E{i}"}) for i, item in enumerate(chosen, start=1)]


def rerank(
    state: ResearchState, config: RunnableConfig, deps: ResearchDependencies
) -> dict[str, Any]:
    strategy = RetrievalStrategy(state["strategy"])
    candidates = state.get("candidates") or []
    previous = state.get("evidence") or []
    complex_q = _is_complex(state)
    k = deps.config.evidence_k_complex if complex_q else deps.config.evidence_k_simple

    if strategy in RERANK_STRATEGIES and candidates:
        if deps.engine.reranker is None:
            raise RuntimeError(f"Strategy {strategy.value} requires a reranker")
        # Earlier evidence is re-scored too, so all items share one comparable scale.
        pool = [_from_evidence(e) for e in previous + candidates]
        unique = list({c.chunk_id: c for c in pool}.values())
        reranker = deps.engine.reranker

        def score(_: dict[str, Any]) -> list[RetrievedChunk]:
            return cross_encoder_rerank(reranker, state["question"], unique, len(unique))

        reranked = RunnableLambda(score, name="cross_encoder_rerank").invoke(
            {
                "question": state["question"],
                "candidates": len(unique),
                "model": getattr(reranker, "model_name", "unknown"),
            },
            config,
        )
        ranked = [to_evidence(c) for c in reranked]
    else:
        ranked = sorted(previous + candidates, key=lambda e: -e["score"])

    cross_paper = (state.get("analysis") or {}).get("requires_cross_paper", False)
    cap = deps.config.max_chunks_per_paper if cross_paper else None
    evidence = select_evidence(ranked, k, cap)
    return {
        "evidence": evidence,
        "_trace": {
            "reranked": strategy in RERANK_STRATEGIES,
            "evidence": len(evidence),
            "papers": len({e["paper_id"] for e in evidence}),
        },
    }
