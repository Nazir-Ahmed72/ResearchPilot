# Evaluation report: ragas-subset

* Dataset: `rag_papers_v1` (6 questions, 6 answerable)
* Date (UTC): 2026-11-19T14:56:50+00:00
* Code version: `f67bf88-dirty`
* Models: main `openai/gpt-oss-120b`, fast `openai/gpt-oss-20b`, embeddings `BAAI/bge-small-en-v1.5`, reranker `BAAI/bge-reranker-base`
* Retrieval top-K: 5

## End-to-end research workflow (strategy `hybrid_rerank`)

| Metric | Value |
|---|---|
| Completion rate | 0.000 (0/6) |
| Abstention accuracy (unanswerable) | – |
| False abstention rate (answerable) | – |
| Evidence recall (labels covered by used evidence) | – |
| Citation label accuracy (cited chunks matching labels) | – |
| Citation validity (IDs exist) | – |
| Citation coverage (reported claims cited) | – |
| Support rate (LLM verifier) | – |
| Retrieval retry rate | – |
| Repair rate | – |
| Latency mean / p50 / p95 (ms) | – / – / – |
| Tokens per query (mean) | – |
| Cost per query (USD) | – |

### Mean node latency (ms)

| Node | ms |
|---|---|

## Interpreting these results

* **Small dataset.** 6 questions over 4 papers; differences of a few points between
  strategies are within noise. Treat results as directional.
* **Single run.** LLM steps (query transformation, synthesis, judging) are not fully
  deterministic; repeated runs vary.
* **Content-based labels.** A chunk counts as relevant if it comes from the labelled paper and
  contains the labelled phrases. Paraphrased evidence elsewhere is counted as *not* relevant, so
  precision is a lower bound.
* **LLM-as-a-judge limits** (support rate, Ragas): the judge is another LLM. It can be wrong or
  inconsistent, is sensitive to prompt wording, may favour fluent or longer answers, and here
  uses the same model family that generated the answers (self-preference bias). The label-based
  metrics (recall, citation label accuracy, abstention) do not depend on an LLM.
* **Latency** was measured on a laptop CPU (reranker) against free-tier cloud services and
  includes network round trips and provider rate-limit waits.
