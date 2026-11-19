"""Long-lived infrastructure clients and components shared by a process.

Created once at startup (FastAPI lifespan, or the first task in a Celery worker) and closed on
shutdown, so connections and ML models are not re-created per request. Heavy components (the
embedding model) load lazily on first use. Tests inject in-memory/fake parts via ``from_settings``
keyword overrides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import QdrantClient
from redis import Redis
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.database.session import create_db_engine, create_session_factory
from app.embeddings.dense import DenseEmbedder, SentenceTransformerEmbedder
from app.embeddings.sparse import Bm25SparseEncoder
from app.ingestion.pipeline import IngestionConfig, IngestionPipeline
from app.ingestion.storage import LocalFileStorage
from app.orchestration.dependencies import (
    ResearchConfig,
    ResearchDependencies,
    StructuredStep,
    llm_steps,
)
from app.orchestration.graph import build_research_graph
from app.retrieval.engine import RetrievalConfig, RetrievalEngine
from app.retrieval.qdrant_search import QdrantSearcher
from app.retrieval.query_transformer import QueryPlanner, QueryTransformer, create_query_planner
from app.retrieval.reranker import CrossEncoderReranker, Reranker
from app.retrieval.retrievers import DenseRetriever, HybridRetriever, SparseRetriever
from app.vectorstore.qdrant_store import QdrantChunkStore
from app.workers.dispatch import CeleryDispatcher, InlineDispatcher, TaskDispatcher

logger = logging.getLogger(__name__)


def create_qdrant_client(settings: Settings) -> QdrantClient:
    location = settings.qdrant_location
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    timeout = int(max(1, settings.health_check_timeout_seconds))
    if location.startswith(("http://", "https://")):
        return QdrantClient(url=location, api_key=api_key, timeout=max(timeout, 30))
    if location == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(path=location)  # embedded local mode, persisted to disk


@dataclass
class Resources:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    qdrant: QdrantClient
    redis: Redis
    dense_embedder: DenseEmbedder
    sparse_encoder: Bm25SparseEncoder = field(default_factory=Bm25SparseEncoder)
    reranker: Reranker | None = None
    _dispatcher: TaskDispatcher | None = None
    _query_planner: QueryPlanner | None = None
    _llm_steps: dict[str, StructuredStep] | None = None
    _checkpointer: BaseCheckpointSaver | None = None
    _closers: list = field(default_factory=list)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        dense_embedder: DenseEmbedder | None = None,
        dispatcher: TaskDispatcher | None = None,
        qdrant: QdrantClient | None = None,
        reranker: Reranker | None = None,
        query_planner: QueryPlanner | None = None,
        llm_steps: dict[str, StructuredStep] | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> Resources:
        engine = create_db_engine(settings)
        timeout = settings.health_check_timeout_seconds
        return cls(
            settings=settings,
            engine=engine,
            session_factory=create_session_factory(engine),
            qdrant=qdrant or create_qdrant_client(settings),
            redis=Redis.from_url(
                settings.redis_url, socket_connect_timeout=timeout, socket_timeout=timeout
            ),
            dense_embedder=dense_embedder
            or SentenceTransformerEmbedder(
                settings.embedding_model,
                cache_dir=settings.hf_home,
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
            ),
            reranker=reranker
            or CrossEncoderReranker(
                settings.reranker_model,
                cache_dir=settings.hf_home,
                device=settings.embedding_device,
                batch_size=settings.reranker_batch_size,
                max_length=settings.reranker_max_length,
            ),
            _dispatcher=dispatcher,
            _query_planner=query_planner,
            _llm_steps=llm_steps,
            _checkpointer=checkpointer,
        )

    @cached_property
    def storage(self) -> LocalFileStorage:
        return LocalFileStorage(self.settings.upload_dir)

    @cached_property
    def chunk_store(self) -> QdrantChunkStore:
        return QdrantChunkStore(self.qdrant, self.settings.qdrant_collection)

    @cached_property
    def ingestion_pipeline(self) -> IngestionPipeline:
        s = self.settings
        return IngestionPipeline(
            session_factory=self.session_factory,
            storage=self.storage,
            store=self.chunk_store,
            dense=self.dense_embedder,
            sparse=self.sparse_encoder,
            config=IngestionConfig(
                chunk_size=s.chunk_size_tokens,
                chunk_overlap=s.chunk_overlap_tokens,
                include_references=s.index_references,
                upsert_batch_size=s.qdrant_upsert_batch_size,
            ),
        )

    @cached_property
    def query_transformer(self) -> QueryTransformer:
        planner = self._query_planner
        if planner is None and self.settings.groq_api_key is not None:
            from app.llm.factory import create_chat_model

            planner = create_query_planner(create_chat_model(self.settings, tier="fast"))
        # Without an LLM, query transformation degrades to "search the original question".
        return QueryTransformer(planner, max_queries=self.settings.max_query_variants)

    @cached_property
    def retrieval_engine(self) -> RetrievalEngine:
        s = self.settings
        searcher = QdrantSearcher(self.qdrant, s.qdrant_collection)
        return RetrievalEngine(
            dense=DenseRetriever(searcher, self.dense_embedder),
            sparse=SparseRetriever(searcher, self.sparse_encoder),
            hybrid=HybridRetriever(
                searcher, self.dense_embedder, self.sparse_encoder, rrf_k=s.rrf_k
            ),
            reranker=self.reranker,
            transformer=self.query_transformer,
            config=RetrievalConfig(
                top_k=s.retrieval_top_k,
                candidate_k=s.retrieval_candidate_k,
                rerank_candidate_k=s.rerank_candidate_k,
            ),
        )

    @property
    def checkpointer(self) -> BaseCheckpointSaver:
        if self._checkpointer is None:
            from app.orchestration.checkpoint import create_checkpointer

            self._checkpointer, close = create_checkpointer(self.settings)
            self._closers.append(close)
        return self._checkpointer

    @cached_property
    def research_dependencies(self) -> ResearchDependencies:
        """Built on first use: needs the LLM (GROQ_API_KEY)."""
        steps = self._llm_steps
        if steps is None:
            from app.llm.factory import create_chat_model

            steps = llm_steps(
                create_chat_model(self.settings, "main"),
                create_chat_model(self.settings, "fast"),
                create_chat_model(
                    self.settings,
                    "main",
                    reasoning_effort=self.settings.llm_reasoning_effort_synthesis,
                ),
            )
        s = self.settings
        return ResearchDependencies(
            engine=self.retrieval_engine,
            transformer=self.query_transformer,
            analyzer=steps.get("analyzer"),
            evaluator=steps.get("evaluator"),
            synthesizer=steps["synthesizer"],
            verifier=steps.get("verifier"),
            config=ResearchConfig(
                max_retrieval_retries=s.research_max_retrieval_retries,
                max_repair_attempts=s.research_max_repair_attempts,
                evidence_k_simple=s.research_evidence_k_simple,
                evidence_k_complex=s.research_evidence_k_complex,
                max_chunks_per_paper=s.research_max_chunks_per_paper,
                candidate_k_simple=s.rerank_candidate_k,
                candidate_k_complex=s.research_candidate_k_complex,
                timeout_seconds=s.research_timeout_seconds,
            ),
        )

    @cached_property
    def research_graph(self) -> CompiledStateGraph:
        """The production graph: durable checkpoints (Neon) and configured interrupts."""
        return build_research_graph(
            self.research_dependencies,
            checkpointer=self.checkpointer,
            interrupt_before=self.settings.research_interrupt_before,
        )

    @cached_property
    def evaluation_graph(self) -> CompiledStateGraph:
        """Same workflow without checkpoints or interrupts: evaluation runs need neither, and
        should not leave checkpoint threads behind in the database."""
        return build_research_graph(self.research_dependencies)

    @cached_property
    def research_runner(self):  # type: ignore[no-untyped-def]
        from app.services.research import ResearchRunner

        return ResearchRunner(
            session_factory=self.session_factory,
            graph=self.research_graph,
            settings=self.settings,
            tracer=self.tracer,
        )

    @cached_property
    def tracer(self):  # type: ignore[no-untyped-def]
        """LangSmith tracer, or None when tracing is disabled (the default)."""
        from app.observability.tracing import create_tracer

        tracer = create_tracer(self.settings)
        if tracer is not None:
            from app.observability.tracing import flush

            self._closers.append(lambda: flush(tracer))
        return tracer

    @property
    def dispatcher(self) -> TaskDispatcher:
        if self._dispatcher is None:
            if self.settings.task_backend == "inline":
                # The runner is resolved lazily so a missing LLM key fails the run, not startup.
                from app.services.evaluations import execute_batch

                self._dispatcher = InlineDispatcher(
                    self.ingestion_pipeline,
                    lambda rid: self.research_runner.run(rid),
                    evaluation_runner=lambda bid: execute_batch(self, bid),
                )
            else:
                from app.workers.celery_config import create_celery_app

                self._dispatcher = CeleryDispatcher(create_celery_app(self.settings))
        return self._dispatcher

    def close(self) -> None:
        closers = [self.redis.close, self.qdrant.close, self.engine.dispose, *self._closers]
        if self._dispatcher is not None:
            closers.insert(0, self._dispatcher.close)
        for closer in closers:
            try:
                closer()
            except Exception:  # noqa: BLE001 - shutdown must continue even if one close fails
                logger.warning("Failed to close resource", exc_info=True)
