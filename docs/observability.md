# Observability

ResearchPilot is observable at three levels: structured logs (always on), metrics stored per
research request (always on), and LangSmith traces (opt-in).

## 1. Structured logs

* JSON lines (`LOG_JSON=true`) or readable console output, via the standard `logging` module.
* Every line written while handling an HTTP request carries the `request_id` (also returned in
  the `X-Request-ID` header and in every error response).
* One access-log line per request: method, path, status, duration.
* Ingestion, retrieval and research steps log their timings (`ms_extract`, `ms_rerank`, ...),
  chunk ids, strategy and counts.
* **Redaction.** Before output, every message and extra field passes through `redact()`. It masks
  values whose key is or ends with `api_key`, `token`, `secret`, `password`, `authorization`
  or `cookie`; `key=value` pairs with such names; Groq, LangSmith and OpenAI-style keys; and
  passwords in connection URLs. Token *counters* (`input_tokens`, `max_tokens`) are kept.

## 2. Metrics stored per research request

`research_results.metrics` (also returned by `GET /api/v1/research/{id}`):

| Field | Meaning |
|---|---|
| `latency_ms` | End-to-end workflow time |
| `node_timings_ms` | Time per graph node (summed when a node runs more than once) |
| `retry_count`, `repair_count` | Retrieval retries and citation repair passes |
| `token_usage` | Input/output/total tokens per model (LangChain usage callback) |
| `cost_usd` | Only when `LLM_PRICING` is configured for every model used |
| `models` | Main and fast model names |
| `errors` | Redacted, non-fatal errors (e.g. a failed auxiliary LLM call) |
| `trace` | Compact per-node trace: node, ms and node-specific details (queries, chunk ids, sufficiency, verification counts) |

`research_requests.current_node` shows live progress while a run executes.

## 3. LangSmith tracing (opt-in)

Enable it in `.env`:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...            # smith.langchain.com > Settings > API Keys
LANGSMITH_PROJECT=researchpilot
```

### How it is wired
* `app/observability/tracing.py` creates a `LangChainTracer` only when tracing is enabled and a
  key is present. It is attached **explicitly** to the callbacks of research runs and
  evaluation runs. LangChain's global env-var tracing is not used, so tests and local
  development never send anything and never need an account.
* The LangSmith client is created with `hide_inputs` / `hide_outputs` hooks that apply the same
  `redact()` as the logs, so secrets never leave the process.
* Traces are flushed at the end of each run and on shutdown.

### What a research trace contains
* **Root run** `research_workflow` with metadata: request id, strategy, collection id, model
  provider, main and fast model names, app version; and tags: `research`, strategy.
* **One child run per graph node**, in execution order (retries and repairs appear as repeated
  nodes), with the node's input and output state.
* Inside `retrieve`: a `hybrid_search` run (queries in; chunk ids, payloads and dense/sparse/RRF
  scores out). Inside `rerank`: a `cross_encoder_rerank` run (candidate count and reranker model
  in; reranked chunks with scores out).
* Every LLM call (query understanding, query transformation, evidence evaluation, synthesis,
  verification): prompt, structured output, model name, latency and token usage.
* Exceptions, with the failing node.

Evaluation runs (`scripts/run_evaluation.py --generation`) are traced as `eval:<question id>`
with question id, category, strategy and dataset as metadata.

### LangSmith datasets and experiments
`scripts/run_evaluation.py --langsmith` (or `"langsmith": true` in `POST /evaluations/run`)
creates the dataset `researchpilot-<dataset name>` (questions, reference answers, evidence
labels) once, and records one LangSmith experiment per retrieval strategy (E1–E5) with Recall@K,
Precision@K, MRR, nDCG@K and latency per question. Experiments reuse the locally computed results,
so publishing makes no extra retrieval or LLM calls, and can be compared side by side in the
LangSmith UI.

### Privacy note
With tracing enabled, retrieved paper text and answers are sent to LangSmith; that is what makes
RAG traces useful. Credentials are always redacted. For confidential collections, leave
`LANGSMITH_TRACING=false` (the default).
