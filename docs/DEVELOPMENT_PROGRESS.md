# Development Progress

## Environment

| Item | Value |
|---|---|
| Python | 3.10.10 (only interpreter on the machine; spec asked for 3.11+, all chosen libraries support 3.10) |
| Virtual env | `E:\ResearchPilot\.venv` — all packages installed here only |
| Caches | pip → `.cache/pip` (via `.venv/pip.ini`), pytest → `.cache/pytest`; set `TMP`/`TEMP` to `.cache/tmp` when installing |
| Database | Neon serverless PostgreSQL (SQLite fallback for local dev/tests) |
| Redis / Qdrant | Upstash Redis, Qdrant Cloud (free tiers) |
| LLM | Groq |
| Docker | Installed but not used yet (deferred by user) |

---

## M1 — Foundation ✅ (2026-11-19)

### Completed
* Project packaging (`pyproject.toml`), pinned lock file (`requirements/lock.txt`).
* Typed settings from environment (`app/core/config.py`), incl. Neon pooled/direct URLs and Groq.
* Structured JSON logging with request IDs and secret redaction (`app/core/logging.py`).
* Request-context ASGI middleware: request ID header, access log, safe 500 handling.
* Centralised error schema and exception handlers (`app/core/exceptions.py`).
* Shared infrastructure clients (`app/core/resources.py`): DB engine, Qdrant, Redis.
* SQLAlchemy models for all six required tables (`app/database/models.py`).
* Neon-aware engine: prepared statements off on pooled endpoint, pre-ping, recycle, timeouts.
* Alembic set up; initial migration generated and hand-reviewed (fixed JSONB `Text` import and
  portable `now()` defaults). Migrations use the direct Neon endpoint.
* Health endpoints: `GET /api/v1/health` (readiness, parallel checks with timeouts, 503 when
  degraded) and `GET /api/v1/health/live`.
* `.env.example`, `.gitignore`, `docs/architecture.md` (with Mermaid diagrams).

### Files created
`pyproject.toml`, `requirements/lock.txt`, `alembic.ini`, `migrations/env.py`,
`migrations/versions/20261119_76b16bea068b_initial_schema.py`, `app/main.py`,
`app/core/{config,logging,exceptions,middleware,resources}.py`,
`app/database/{models,session}.py`, `app/services/health.py`, `app/api/dependencies.py`,
`app/api/routes/health.py`, `tests/conftest.py`, `tests/unit/{test_config,test_logging,
test_api_errors,test_session}.py`, `tests/integration/{test_health,test_models,
test_migrations}.py`, `.env.example`, `.gitignore`, `README.md`, `docs/architecture.md`.

### Dependencies added
fastapi, uvicorn[standard], pydantic, pydantic-settings, sqlalchemy, alembic, psycopg[binary],
redis, qdrant-client, python-multipart; dev: pytest, pytest-cov, httpx2, ruff.
(`httpx2` replaces `httpx` for tests: Starlette 1.7's TestClient deprecates `httpx`.)

### Tests executed
`pytest`: **51 passed, 1 skipped**. The skip is the PostgreSQL migration test, which runs only
when `TEST_DATABASE_URL` is set (point it at a Neon branch's direct URL).
`ruff check` and `ruff format --check`: clean.
Manual smoke test with real uvicorn on SQLite: `/docs` 200, readiness correctly degraded while
Redis was unavailable.

### Neon verification (2026-11-19)
* Pooled and direct endpoints both connect (PostgreSQL 18.7, region ap-southeast-1).
* Migration round-trip test passed on Neon (`TEST_DATABASE_URL` = direct URL): schema matches
  models exactly; downgrade removes all tables. Run before `upgrade` while the DB was empty.
* `alembic upgrade head` applied to Neon: 6 tables + `alembic_version` (revision 76b16bea068b).
* Bug found and fixed: Alembic `env.py` and the migration test did not normalise raw Neon
  `postgresql://` URLs, so SQLAlchemy fell back to the missing `psycopg2` driver.

### Hosted Redis + Qdrant (2026-11-19)
Decision: run the real architecture (separate Celery worker) without Docker by using free
managed services. Qdrant's embedded folder mode was rejected because only one process can open
it, and the API and worker both need it.
* Redis: Upstash (TLS, `rediss://`). Read/write verified.
* Qdrant: Qdrant Cloud free cluster (ap-northeast-2), API key auth. Verified.
* `GET /api/v1/health` returns **200 ok** with all three dependencies (database 2.9 s cold /
  0.4 s warm; Redis 0.07 s warm; Qdrant 0.2 s warm). No secrets in logs.
* Health-check timeout default raised 2 s to 5 s so a Neon cold start is not reported as down.

### Known problems / not yet verified
* Upstash free tier limits monthly commands; Celery idle polling must be tuned in M2.
* `GROQ_API_KEY` not set yet (not needed until M4).


---

## M2 — Ingestion ✅ (2026-11-19)

### Completed
* Upload validation (extension, MIME, size, signature, parse, encryption, page limit) and
  filename sanitisation.
* PyMuPDF extraction with page numbers, font-based heading detection, title/author/year/DOI/arXiv
  metadata; text cleaning (ligatures, hyphenation, running headers/footers).
* Structure-aware chunking with token budgets, sentence-level overlap, page ranges, deterministic
  IDs, references excluded by default.
* Dense embeddings (BGE-small, lazy-loaded once per process, cached on E:) and an own BM25
  sparse encoder; Qdrant collection with named dense + sparse (IDF) vectors and payload indexes.
* Idempotent pipeline with permanent/transient error classification, compensation, redacted
  error storage, and protection against deletes racing ingestion.
* Celery worker (Upstash TLS, low-traffic polling), plus an in-process dispatcher.
* Repositories; `PaperService`; endpoints: `POST /papers/upload` (multi-file, per-file outcomes),
  `GET /papers` (pagination, status/collection filters), `GET /papers/{id}`,
  `DELETE /papers/{id}`, `POST /papers/{id}/reingest`.
* Docs: `docs/ingestion_pipeline.md`; architecture updated.

### Decisions
* `app/embeddings` and `app/vectorstore` are separate packages (not inside `ingestion/`) because
  retrieval (M3) also needs them; keeps ingestion and retrieval independent.
* Own BM25 encoder instead of fastembed: no second runtime, deterministic, testable.
* Chunk size 512 → 350 tokens: BGE-small's limit is 512 and each chunk gets a context header.
* Upload status codes: 202 if any file accepted; otherwise the shared reason (409 duplicate,
  413 too large, 415 not PDF, 503 queue down), else 400; per-file details always included.

### Bugs found by tests and real-data runs (all fixed)
* Secrets in `key=value` form were not redacted in logs or stored errors.
* SQLAlchemy 2.0 does not cascade objects added via backrefs: re-ingest jobs were never saved.
* Pydantic read the `collection` relationship object instead of its name (500 on GET).
* Real arXiv PDF: section numbers on separate lines were missed; wrapped headings split in two;
  arXiv ID (rotated stamp) not found, so the year came from the v4 PDF date.

### Dependencies added
pymupdf, sentence-transformers (torch 2.14 CPU build from the PyTorch CPU index), celery[redis],
snowballstemmer. BGE-small model (129 MB) cached in `E:\ResearchPilot\.cache\hf`.

### Tests executed
* `pytest`: **148 passed, 4 skipped** (PostgreSQL migration test without `TEST_DATABASE_URL`;
  3 real-model tests without `RUN_MODEL_TESTS=1`).
* `RUN_MODEL_TESTS=1 pytest tests/integration/test_real_embedder.py`: **3 passed** (real BGE).
* `ruff check` / `ruff format --check`: clean.
* **Live end-to-end** (Neon + Upstash + Qdrant Cloud, real Celery worker, real BGE):
  arXiv 2005.11401 → `ready` in ~12 s, 43 chunks, 25 headings, correct title/year/arXiv id;
  Qdrant schema and payload indexes verified; delete → 204, 0 vectors left, file removed;
  re-upload → 409 with existing id. No secrets found in API/worker logs.

### Known problems / limitations
See `docs/ingestion_pipeline.md` → Known limitations (no OCR, equations, lone-letter appendix
headings, authors only from metadata, local-disk file storage).
* The worker re-checks Hugging Face for model updates on each start (needs network); setting
  `HF_HUB_OFFLINE=1` after the first download avoids that.
* Upstash command usage has not been measured over a long idle period yet.

### State left in the cloud services
* Neon: paper `08d84b5e-924d-4ee6-ba0d-4f0cd12e056d` (RAG paper, collection `rag-papers`) and
  its job. Kept intentionally as test data for M3 retrieval.
* Qdrant Cloud: collection `researchpilot_chunks` with that paper's 43 points.


---

## M3 — Retrieval engine ✅ (2026-11-19)

### Completed
* Dense, sparse and hybrid retrievers over Qdrant with collection/paper/section/year filters;
  all searches of a request batched into one Qdrant round trip.
* Reciprocal Rank Fusion with deduplication, weights and full per-stage score/rank history.
* Cross-encoder reranking (bge-reranker-base), model loaded once per process.
* LLM query transformation (rewrite, alternatives, decomposition) with structured output,
  validation, caps and graceful fallback; provider-neutral LLM factory (Groq).
* `RetrievalEngine` implementing strategies E1–E5 with timings and structured logs.
* `POST /api/v1/retrieval/search` and `scripts/compare_retrieval.py`.
* Shared enums moved to `app/core/types.py` (no schema change).
* Docs: `docs/retrieval_engine.md`; architecture updated.

### Decisions
* Groq's `llama-3.3-70b-versatile` is no longer available to this key: default models are now
  `openai/gpt-oss-120b` (main) and `openai/gpt-oss-20b` (fast; used for query transformation).
* The three retrievers live in one module (`retrieval/retrievers.py`, ~100 lines) instead of three
  files from the suggested layout, because they share the batched searcher.
* Rerank candidates 20 → 10 after measuring CPU latency (12.8 s → 7.5 s).
* Reranker outputs are documented as uncalibrated; no fixed relevance threshold is assumed.

### Dependencies added
langchain-core 1.7, langchain-groq 1.1 (groq 0.37). Reranker model bge-reranker-base (~1.1 GB)
cached in `E:\ResearchPilot\.cache\hf` (total model cache 1.2 GB).

### Tests executed
* `pytest`: **187 passed, 8 skipped** (opt-in: 1 PostgreSQL, 4 real-model, 3 live Groq).
* `RUN_MODEL_TESTS=1`: **4 passed** (real BGE embedder + reranker).
* `RUN_LIVE_TESTS=1 pytest tests/integration/test_live_groq.py`: **3 passed**.
* `ruff check` / `ruff format --check`: clean.
* Live E1–E5 comparison on the RAG paper in Qdrant Cloud (see `docs/retrieval_engine.md`).

### Bugs / findings
* Fixed: a 3-character junk query ("ok?") passed validation; now ≥3 letters/digits required.
* Finding: reranker scores are not calibrated (a relevant passage scored 0.007).
* Finding: the chunk answering "What retriever does RAG use?" ranks low in every strategy (math-
  heavy text; single-topic corpus weakens BM25). To be quantified in M5, not patched ad hoc.

### Known problems
* Reranking takes ~7.5 s per query on this CPU.
* Quality of E1–E5 is not yet measured (needs the labelled evaluation set, M5).


---

## M4 — LangGraph orchestration ✅ (2026-11-19)

### Completed
* Typed, JSON-serialisable `ResearchState` with reducers for errors, trace and node timings.
* Eight nodes: understand_query, plan_queries, retrieve, rerank, evaluate_evidence, synthesize,
  verify_citations, respond. Conditional routing on complexity/intent, evidence sufficiency,
  verification results; bounded retrieval retries (1) and repair passes (1).
* Structured LLM outputs (QueryAnalysis, EvidenceAssessment, Synthesis with reported/inferred
  claims, VerificationResult); graceful fallbacks for auxiliary LLM failures.
* Two-layer citation verification and deterministic response rendering with numbered references.
* Prompt-injection defences for evidence and questions.
* PostgresSaver checkpoints on Neon (direct URL, health-checked pool), resume endpoint,
  crash-redelivery resume, human-in-the-loop `interrupt_before`; RetryPolicy on I/O and synthesis
  nodes; whole-run deadline.
* ResearchService/ResearchRunner, Celery `research` queue, inline dispatcher, progress via
  `current_node`, metrics (latency per node, tokens per model, optional cost, retries, errors).
* Endpoints: `POST /research/query`, `GET /research/{id}`, `POST /research/{id}/resume`.
* Ingested 3 more arXiv papers (DPR, GraphRAG, Self-RAG): 4 papers / 243 chunks.
* Docs: `docs/orchestration.md`; README and architecture updated.

### Decisions
* Synchronous graph: LangGraph node timeouts need async nodes, and psycopg async is awkward on
  Windows; per-call timeouts + a whole-run deadline instead (deadline in the run config so a
  resumed run gets a fresh budget).
* Response generation is deterministic (no LLM) so it cannot add unverified statements.
* Fast model for understanding/planning/evaluation; main model for synthesis (reasoning medium)
  and verification (reasoning low).
* Cross-paper questions diversify candidates per paper before reranking.
* Ingestion metadata is extraction-owned: re-ingestion refreshes title/authors/year/DOI.

### Schema / database changes (non-destructive)
* Migration `6d4ebaab7ead`: nullable `research_requests.current_node`. Applied to Neon.
* `JobStatus.INTERRUPTED` (string enum, no migration needed).
* LangGraph `checkpoint*` tables created by `PostgresSaver.setup()` in Neon; Alembic configured
  to ignore them (autogenerate diff vs Neon: empty).

### Dependencies added
langgraph 1.2, langgraph-checkpoint-postgres 3.2, psycopg-pool 3.3.

### Tests executed
* `pytest`: **241 passed, 9 skipped** (opt-in: PostgreSQL migration, real models, live Groq,
  live Neon checkpoint).
* `RUN_LIVE_TESTS=1 pytest tests/integration/test_live_checkpoint.py`: **1 passed** (crash in one
  process, resume from Neon in a new one, thread cleaned up).
* `ruff check` / `ruff format --check`: clean.
* Live end-to-end with Celery worker + Groq + Neon + Upstash + Qdrant Cloud: 3 questions
  (factual, comparison, limitations) succeeded; results in `docs/orchestration.md`.

### Bugs found and fixed
* Self-RAG title extracted as only its second line (small caps sizes 13.7/13.8 pt).
* Re-ingestion did not refresh extracted metadata.
* Tokenizer warning when counting long paragraphs before splitting.
* Node mutated the LLM's returned object in place (caught by a cross-test failure).
* **Safety:** the PostgreSQL migration test runs `downgrade base`; it now refuses to run unless
  the target database is empty (it was previously only safe because Neon was empty).
* Live: summaries had no citations (repair on every run), single-paper evidence for comparisons,
  verification prompt over Groq's 8k TPM limit, unbounded reasoning latency. See
  `docs/orchestration.md` → "What the first live runs exposed".

### Known problems
* Groq free tier (8k tokens/min on gpt-oss-120b) throttles back-to-back runs.
* One-sentence multi-hop questions may be classified simple and skip evidence evaluation.
* No streaming responses (progress by polling).
* Answer quality is not yet measured systematically (M5).

### Next milestone: M5 — see below

---

## M5 — Observability and evaluation ✅ (2026-11-19)

### Completed
* LangSmith tracing (opt-in, explicit tracer with input/output redaction; named
  `hybrid_search` / `cross_encoder_rerank` child runs; run metadata and tags; flush on finish).
* Evaluation dataset `rag_papers_v1`: 20 questions in 6 categories incl. 3 unanswerable,
  content-based evidence labels written from the indexed text and validated before every run.
* Metrics: Recall@K, Precision@K, MRR, nDCG@K; end-to-end completion, abstention, false
  abstention, evidence recall, citation label accuracy, citation validity/coverage, support rate,
  latency (total/per node), tokens, cost; Ragas 0.4 (faithfulness, answer relevancy, answer
  correctness, context recall) with Groq as judge and BGE embeddings.
* Runner + CLI (`scripts/run_evaluation.py`), reproducible results under `experiments/results/`
  (config with git commit, label check, per-question JSONL, summary, Markdown report), Ragas on
  saved answers (`--ragas-from`), LangSmith dataset + one experiment per strategy (no
  recomputation).
* API: `POST /evaluations/run`, `GET /evaluations/results`, `GET /evaluations/results/{id}`;
  Celery task on the research queue; `EvaluationRun` rows per experiment.
* Docs: `docs/evaluation.md`, `docs/observability.md`; README updated.

### Fixes to the research workflow found by evaluation
* Unanswerable questions on the simple path were reported as sufficient: added
  `evidence_answers_question` (decided first), a prompt rule, and a deterministic guard that moves
  absence statements ("the evidence does not mention X") to uncertainty and marks the answer
  insufficient when no finding remains. Then tuned so partially answerable questions still answer
  the part they can (smoke tests: abstention 3/3, no false abstention).
* `basis: "uncertainty"` from the model made Groq reject the whole synthesis (HTTP 400
  tool_use_failed): `basis` is now a normalised string, and tool_use_failed is retried.
* Secret redaction masked `input_tokens` / `max_tokens` (key pattern too broad): fixed.
* Evaluation latency excluded model loading (warm-up), and summaries tolerate incomplete rows.

### Results (details in docs/evaluation.md)
* Retrieval Recall@5: E1 0.49, E2 0.47, E3 0.58, E4 0.59, E5 0.59 (identical on a re-run).
* End to end (E4, 18/20 completed): abstention 1.00, false abstention 0.00, citation validity
  1.00, support rate 0.97, evidence recall 0.50, citation label accuracy 0.30, p50 13.5 s.
* Key finding: LLM-judged support (0.97) hides that half of the needed evidence is not retrieved;
  overview chunks win over specific ones for questions phrased with the system's name.

### Dependencies added
ragas 0.4.4 (+ openai 1.109 for Groq's OpenAI-compatible API; `[eval]` extra), langsmith (explicit).

### Tests executed
* `pytest`: **283 passed, 9 skipped**. `ruff`: clean.
* Live: retrieval E1–E5 (twice), generation on 20 questions with LangSmith tracing, Ragas subset,
  LangSmith dataset + 5 experiments + 20 traces verified via the LangSmith API.

### Known problems
* Groq free tier: **200k tokens/day** on gpt-oss-120b; the generation run hit it after 18
  questions. End-to-end comparison of all five strategies needs more quota.
* Retrieval weaknesses identified (see improvement candidates in docs/evaluation.md): simple
  factual recall 0.50, reranking hurts multi-part questions, QT has no measurable effect here.
* The dataset is small (20 questions); results are directional.

### Follow-up (same day)
* Ragas on 6 saved answers: answer relevancy 0.90 (6/6), context recall 0.50 (5/6, agrees with
  the label-based evidence recall). Faithfulness and answer correctness failed on Groq's daily
  token limit for the judge model; command to re-run on a fresh quota day in docs/evaluation.md.
* Ragas scoring can be paced (`--pace`) and run on saved answers (`--ragas-from`).

---

## M6 — Containerisation and CI ✅ (2026-11-19)

Decision (user): **Option A** — no Docker on the development machine (Docker Desktop data lives
on C:); GitHub Actions builds and smoke-tests the image instead.

### Completed
* `Dockerfile`: multi-stage, Python 3.11-slim, CPU PyTorch, versions constrained by the lock
  file, dependency layer cached separately from code, non-root user (uid 10001), liveness
  HEALTHCHECK, exec-form CMD with graceful shutdown, models in a volume (not baked in).
* `docker-compose.yml`: api, worker, redis 8.8.4, qdrant v1.20.1, one-off `migrate` job, optional
  `local-db` profile (postgres 18.7); health checks for every service, restart policies,
  persistent volumes, private network (only the API port published), memory limits, startup
  ordering (`service_healthy` / `service_completed_successfully`), Compose-level overrides so a
  cloud `.env` still uses local Redis/Qdrant.
* Worker health check (`python -m app.workers.healthcheck`: ready file + broker ping), since
  Celery remote control is disabled.
* `.dockerignore` (no `.env`, venv, caches, data, results, tests, docs in the build context).
* `.github/workflows/ci.yml`: ruff; tests on Python 3.10 / 3.11 / 3.12 with CPU torch and lock
  constraints; compose validation; image build with GHA cache; container smoke test (health,
  docs, non-root, Alembic heads).
* `docs/deployment.md` with the deployment diagram, configuration precedence, shutdown/recovery
  behaviour, CI description and Render/AWS mapping (nothing deployed, no resources created).

### Fixes
* Empty secret values (`QDRANT_API_KEY=`) are now "not set" instead of an empty key.

### Verification
* `docker compose config -q` (both profiles) passes locally (no daemon needed); resolved
  environment checked: local Redis/Qdrant inside Compose, database from `.env`.
* Workflow YAML parsed.
* **First CI run failed** in the test jobs on all Python versions (lint and Docker passed):
  `pyproject.toml` sets `--basetemp=.cache/pytest`, pytest only creates the last folder, and the
  git-ignored `.cache/` does not exist in a fresh checkout, so 83 tests using `tmp_path` errored
  in setup. Reproduced locally in a fresh clone (206 passed, 83 errors). Fixed by a CI step
  `mkdir -p .cache`; fresh-clone check: 288 passed, 9 skipped.
* **CI green** after the fix: lint, tests on Python 3.10 / 3.11 / 3.12, Docker image build and
  container smoke test all pass.
* Remaining CI warning: actions running on Node.js 20 (deprecated by GitHub); bump action
  versions when new majors are available.

---

## M7 — Final documentation and validation ✅ (2026-11-19)

### Completed
* `docs/api.md` (conventions, error codes, every endpoint with payloads; generated route list
  cross-checked against the live OpenAPI schema).
* README: Docker section, roadmap complete, status and limitations; architecture doc updated.
* Final validation: all modules compile; every setting documented in `.env.example`;
  `alembic check` against Neon: no pending changes; **288 passed, 9 skipped**; ruff clean; live
  API smoke test against Neon/Upstash/Qdrant Cloud: `/health` 200 (all checks ok), `/docs` 200,
  4 papers ready, evaluations endpoint 200.

### Issues found by the final validation (fixed)
* The Groq organisation id (from rate-limit error messages) was stored in two evaluation result
  files: error messages are now redacted before storage, a redaction pattern for provider
  organisation ids was added, and the saved files were cleaned. It is still present in the
  already-pushed M5 commit history (an account identifier, not a credential; history was not
  rewritten).
* Tests wrote result folders into the real `experiments/results/` (a default argument bound
  `RESULTS_ROOT` at import time, so patching it had no effect): fixed, and the 11 test folders
  were deleted.
* A generated regex contained a literal backspace character instead of ``, silently
  disabling the new redaction rule: fixed; the source tree was scanned for other control
  characters (none).

### Remaining work / known limitations
* Groq free tier limits (8k tokens/min, 200k tokens/day for gpt-oss-120b) restrict end-to-end
  evaluation: Ragas faithfulness/correctness and a full end-to-end E1–E5 comparison are pending.
* Retrieval quality improvements identified by the evaluation (docs/evaluation.md).
* No authentication; local-disk file storage shared by API and worker (see docs/deployment.md).
* Streaming responses not implemented (progress via polling).
