# API Reference

Base URL: `http://localhost:8000/api/v1`. Interactive documentation, generated from the code, is
at `/docs` (Swagger UI) and `/redoc`; the OpenAPI schema is at `/openapi.json`. This page
summarises the contract and the conventions shared by all endpoints.

## Conventions

* **JSON** request and response bodies (uploads use `multipart/form-data`).
* **Long-running work is asynchronous.** Uploads, research and evaluations return
  `202 Accepted` immediately and run in the background; poll the returned resource for status.
* **Request IDs.** Every response carries `X-Request-ID`. Send your own (`[A-Za-z0-9._-]`, up to
  128 characters) to correlate client and server logs; otherwise one is generated.
* **Errors** always have the same shape:

  ```json
  {"error": {"code": "not_found", "message": "Paper not found",
             "details": {"paper_id": "..."}, "request_id": "4f1c..."}}
  ```

  | Status | `code` | When |
  |---|---|---|
  | 400 | `invalid_input`, `upload_rejected` | Invalid request content (e.g. not a PDF) |
  | 404 | `not_found` | Unknown paper, collection, request, run or dataset |
  | 409 | `conflict`, `upload_rejected` | Duplicate upload; operation not allowed in the current state |
  | 413 | `upload_rejected` | Every uploaded file exceeds `MAX_UPLOAD_SIZE_MB` |
  | 415 | `upload_rejected` | Every uploaded file is not a PDF |
  | 422 | `validation_error` | Request body/parameters fail schema validation (`details` lists fields) |
  | 503 | `dependency_unavailable`, `llm_not_configured` | Redis, Qdrant, the database or the LLM is unavailable |
  | 500 | `internal_error` | Unexpected error (details only in server logs) |

* **Status values** for jobs, research requests and evaluation runs: `queued`, `running`,
  `succeeded`, `failed`, `interrupted` (research paused for human review).
* **Authentication:** none yet (see [deployment.md](deployment.md)); it is designed to be added as
  a FastAPI dependency without changing route code.

## Health

| Method | Path | Description |
|---|---|---|
| GET | `/health/live` | Liveness: `{"status": "ok"}` if the process is up |
| GET | `/health` | Readiness: checks database, Redis and Qdrant in parallel (5 s timeout each); `200` when all are ok, `503` with per-check details otherwise |

## Papers

### `POST /papers/upload`
Multipart form: `files` (1–`MAX_FILES_PER_UPLOAD` PDFs, each ≤ `MAX_UPLOAD_SIZE_MB`) and an
optional `collection` (created if new). Returns `202` if at least one file was accepted, with
one result per file:

```json
{"accepted": 1, "results": [
  {"filename": "rag.pdf", "status": "accepted", "paper_id": "…", "job_id": "…"},
  {"filename": "rag-copy.pdf", "status": "duplicate", "paper_id": "<existing>", "error_code": "duplicate"},
  {"filename": "notes.txt", "status": "rejected", "error_code": "unsupported_media_type"}]}
```

If no file is accepted, the status reflects the shared reason (409 duplicates, 413 too large,
415 not PDF, 503 queue unavailable, otherwise 400), with the per-file results in `details`.

### Other paper endpoints
| Method | Path | Description |
|---|---|---|
| GET | `/papers?limit=20&offset=0&status=ready&collection=rag-papers` | List with pagination and filters; `total` included |
| GET | `/papers/{paper_id}` | Metadata (title, authors, year, DOI, pages, chunk count, collection) and ingestion jobs with timings |
| DELETE | `/papers/{paper_id}` | `204`. Removes vectors, stored file and row. `409` while processing; `503` if Qdrant is down (the paper stays `deleting` and the delete can be repeated) |
| POST | `/papers/{paper_id}/reingest` | `202`. Re-run ingestion for a `ready` or `failed` paper |

## Retrieval

### `POST /retrieval/search`
Direct access to the retrieval engine, for inspecting strategies.

```json
{"query": "Compare RAG-Sequence and RAG-Token", "strategy": "hybrid_qt_rerank", "top_k": 5,
 "collection": "rag-papers", "paper_ids": [], "sections": [], "year_from": 2019, "year_to": 2024}
```

`strategy`: `dense` (E1), `sparse` (E2), `hybrid` (E3), `hybrid_rerank` (E4, default),
`hybrid_qt_rerank` (E5). The response lists the queries actually searched, the query
transformation (if any), per-stage timings, and per result: chunk/paper ids, title, section,
pages, text, and the score and rank of every stage (`dense`, `sparse`, `rrf`, `rerank`).

## Research

### `POST /research/query`
```json
{"question": "Compare traditional RAG and GraphRAG approaches",
 "collection": "rag-papers", "paper_ids": [], "analysis_type": null, "strategy": null}
```

`analysis_type` (optional override): `factual`, `comparison`, `limitations`, `research_gaps`,
`literature_review`. Returns `202 {"request_id", "status": "queued", "status_url"}`.
`404` for an unknown collection/paper, `409` if papers are not ready or no ready paper is in scope.

### `GET /research/{request_id}`
```json
{"request_id": "…", "status": "running", "current_node": "verify_citations",
 "question": "…", "analysis_type": "comparison", "strategy": "hybrid_rerank",
 "result": null}
```

When `status` is `succeeded`, `result` contains:

| Field | Content |
|---|---|
| `answer_markdown` | Rendered answer with numbered citations and a reference list |
| `summary`, `sections` | Structured answer; each claim has `text`, `basis` (reported/inferred), `verdict`, `citations` |
| `references` | Number, paper id, chunk id, title, year, section, page range |
| `evidence_sufficient`, `uncertainty` | Transparency about missing or conflicting evidence |
| `removed_statements` | Claims removed by citation verification, with reasons |
| `queries` | Search queries used |
| `verification` | Citation validity, coverage, support rate, claim counts |
| `metrics` | Latency (total and per node), retries, repairs, token usage per model, cost (if priced), models, errors |

### `POST /research/{request_id}/resume`
`202`. Continues a `failed` or `interrupted` run from its last checkpoint (completed steps are not
repeated). `409` for other states.

## Evaluations

### `POST /evaluations/run`
```json
{"name": "nightly", "dataset": "rag_papers_v1", "experiments": ["E1","E2","E3","E4","E5"],
 "top_k": 5, "include_generation": false, "generation_strategy": "hybrid_rerank",
 "question_ids": null, "limit": null, "ragas": false, "pace_seconds": 0,
 "collection": null, "langsmith": false}
```

Creates one evaluation run per experiment (plus `GEN` for the end-to-end workflow) and returns
`202 {"batch_id", "runs": [...]}`. `400` for unknown experiments or nothing selected, `404` for
an unknown dataset.

| Method | Path | Description |
|---|---|---|
| GET | `/evaluations/results?limit=20&experiment=E4` | Runs with status, metrics and results directory |
| GET | `/evaluations/results/{run_id}` | One run |

Retrieval runs report `recall@K`, `precision@K`, `mrr`, `ndcg@K`, latency and per-category
recall; `GEN` runs report the end-to-end metrics described in [evaluation.md](evaluation.md).
