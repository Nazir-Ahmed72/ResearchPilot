# Evaluation report: final-generation

* Dataset: `rag_papers_v1` (20 questions, 17 answerable)
* Date (UTC): 2026-11-19T14:41:03+00:00
* Code version: `f67bf88-dirty`
* Models: main `openai/gpt-oss-120b`, fast `openai/gpt-oss-20b`, embeddings `BAAI/bge-small-en-v1.5`, reranker `BAAI/bge-reranker-base`
* Retrieval top-K: 5

## End-to-end research workflow (strategy `hybrid_rerank`)

| Metric | Value |
|---|---|
| Completion rate | 0.900 (18/20) |
| Abstention accuracy (unanswerable) | 1.000 |
| False abstention rate (answerable) | 0.000 |
| Evidence recall (labels covered by used evidence) | 0.500 |
| Citation label accuracy (cited chunks matching labels) | 0.297 |
| Citation validity (IDs exist) | 1.000 |
| Citation coverage (reported claims cited) | 1.000 |
| Support rate (LLM verifier) | 0.973 |
| Retrieval retry rate | 0.000 |
| Repair rate | 0.167 |
| Latency mean / p50 / p95 (ms) | 17,043 / 13,487 / 72,202 |
| Tokens per query (mean) | 7,200 |
| Cost per query (USD) | – |

### Mean node latency (ms)

| Node | ms |
|---|---|
| evaluate_evidence | 883 |
| plan_queries | 542 |
| rerank | 8,184 |
| respond | 0.278 |
| retrieve | 1,022 |
| synthesize | 3,911 |
| understand_query | 762 |
| verify_citations | 2,581 |

## Interpreting these results

* **Small dataset.** 20 questions over 4 papers; differences of a few points between
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
