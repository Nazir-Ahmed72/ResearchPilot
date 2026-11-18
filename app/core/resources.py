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
    def dispatcher(self) -> TaskDispatcher:
        if self._dispatcher is None:
            if self.settings.task_backend == "inline":
                self._dispatcher = InlineDispatcher(self.ingestion_pipeline)
            else:
                from app.workers.celery_config import create_celery_app

                self._dispatcher = CeleryDispatcher(create_celery_app(self.settings))
        return self._dispatcher

    def close(self) -> None:
        closers = [self.redis.close, self.qdrant.close, self.engine.dispose]
        if self._dispatcher is not None:
            closers.insert(0, self._dispatcher.close)
        for closer in closers:
            try:
                closer()
            except Exception:  # noqa: BLE001 - shutdown must continue even if one close fails
                logger.warning("Failed to close resource", exc_info=True)
