# Evaluation report: api-test

* Dataset: `mini` (3 questions, 2 answerable)
* Date (UTC): 2026-11-19T15:56:49+00:00
* Code version: `f67bf88-dirty`
* Models: main `openai/gpt-oss-120b`, fast `openai/gpt-oss-20b`, embeddings `BAAI/bge-small-en-v1.5`, reranker `BAAI/bge-reranker-base`
* Retrieval top-K: 3

## Retrieval (E1-E5)

| Exp | Strategy | recall@3 | precision@3 | mrr | ndcg@3 | Latency mean (ms) | p95 (ms) | Errors |
|---|---|---|---|---|---|---|---|---|
| E2 | `sparse` | 0.750 | 0.333 | 0.750 | 0.694 | 2 | 2 | 0 |
| E3 | `hybrid` | 0.500 | 0.167 | 0.250 | 0.316 | 2 | 2 | 0 |

### Recall@3 by question category

| Exp | comparison | simple_factual |
|---|---|---|
| E2 | 0.500 | 1.000 |
| E3 | 0.000 | 1.000 |

## End-to-end research workflow (strategy `hybrid_rerank`)

| Metric | Value |
|---|---|
| Completion rate | 1.000 (3/3) |
| Abstention accuracy (unanswerable) | 0.000 |
| False abstention rate (answerable) | 0.000 |
| Evidence recall (labels covered by used evidence) | 1.000 |
| Citation label accuracy (cited chunks matching labels) | 0.500 |
| Citation validity (IDs exist) | 1.000 |
| Citation coverage (reported claims cited) | 1.000 |
| Support rate (LLM verifier) | 1.000 |
| Retrieval retry rate | 0.000 |
| Repair rate | 0.000 |
| Latency mean / p50 / p95 (ms) | 11 / 10 / 12 |
| Tokens per query (mean) | 0.000 |
| Cost per query (USD) | – |

### Mean node latency (ms)

| Node | ms |
|---|---|
| rerank | 0.733 |
| respond | 0.100 |
| retrieve | 2 |
| synthesize | 0.133 |
| understand_query | 0.433 |
| verify_citations | 0.133 |

## Interpreting these results

* **Small dataset.** 3 questions over 2 papers; differences of a few points between
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
