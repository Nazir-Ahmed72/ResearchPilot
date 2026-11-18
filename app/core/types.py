"""Domain enums shared by the database, retrieval, orchestration and API layers."""

from __future__ import annotations

import enum


class AnalysisType(str, enum.Enum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    LIMITATIONS = "limitations"
    RESEARCH_GAPS = "research_gaps"
    LITERATURE_REVIEW = "literature_review"


class RetrievalStrategy(str, enum.Enum):
    """The experiment configurations E1-E5."""

    DENSE = "dense"  # E1
    SPARSE = "sparse"  # E2
    HYBRID = "hybrid"  # E3
    HYBRID_RERANK = "hybrid_rerank"  # E4
    HYBRID_QT_RERANK = "hybrid_qt_rerank"  # E5
