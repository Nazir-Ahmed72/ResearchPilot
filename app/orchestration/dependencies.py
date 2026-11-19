"""What the research nodes need, injected once when the graph is built.

LLM steps are typed as anything with ``invoke(input, config) -> BaseModel``, so tests pass
deterministic fakes and production passes ``prompt | llm.with_structured_output(Schema)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.orchestration import prompts
from app.orchestration.schemas import (
    EvidenceAssessment,
    QueryAnalysis,
    Synthesis,
    VerificationResult,
)
from app.retrieval.engine import RetrievalEngine
from app.retrieval.query_transformer import QueryTransformer


class StructuredStep(Protocol):
    def invoke(self, input: dict[str, Any], config: Any = None) -> BaseModel: ...  # noqa: A002


def structured_step(
    llm: BaseChatModel, prompt: ChatPromptTemplate, schema: type[BaseModel]
) -> StructuredStep:
    return prompt | llm.with_structured_output(schema)  # type: ignore[return-value]


@dataclass(frozen=True)
class ResearchConfig:
    max_retrieval_retries: int = 1  # extra plan -> retrieve rounds after insufficient evidence
    max_repair_attempts: int = 1  # extra synthesis rounds after failed citation verification
    evidence_k_simple: int = 6
    evidence_k_complex: int = 10
    max_chunks_per_paper: int = 4  # diversity cap for cross-paper questions
    candidate_k_simple: int = 10  # candidates reranked for simple questions
    candidate_k_complex: int = 16
    evidence_chars: int = 2000
    timeout_seconds: float = 300.0


@dataclass
class ResearchDependencies:
    engine: RetrievalEngine
    transformer: QueryTransformer
    analyzer: StructuredStep | None  # None -> heuristic query understanding
    evaluator: StructuredStep | None  # None -> evidence assumed sufficient if non-empty
    synthesizer: StructuredStep
    verifier: StructuredStep | None  # None -> only deterministic citation checks
    config: ResearchConfig


def llm_steps(
    main: BaseChatModel, fast: BaseChatModel, synthesis: BaseChatModel | None = None
) -> dict[str, StructuredStep]:
    """Lightweight decisions use the fast model; verification the main model; synthesis the main
    model, optionally with a higher reasoning effort."""
    return {
        "analyzer": structured_step(fast, prompts.ANALYZE_PROMPT, QueryAnalysis),
        "evaluator": structured_step(fast, prompts.EVALUATE_PROMPT, EvidenceAssessment),
        "synthesizer": structured_step(synthesis or main, prompts.SYNTHESIZE_PROMPT, Synthesis),
        "verifier": structured_step(main, prompts.VERIFY_PROMPT, VerificationResult),
    }
