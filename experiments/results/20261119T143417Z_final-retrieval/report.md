# Evaluation report: final-retrieval

* Dataset: `rag_papers_v1` (20 questions, 17 answerable)
* Date (UTC): 2026-11-19T14:34:17+00:00
* Code version: `f67bf88-dirty`
* Models: main `openai/gpt-oss-120b`, fast `openai/gpt-oss-20b`, embeddings `BAAI/bge-small-en-v1.5`, reranker `BAAI/bge-reranker-base`
* Retrieval top-K: 5

## Retrieval (E1-E5)

| Exp | Strategy | recall@5 | precision@5 | mrr | ndcg@5 | Latency mean (ms) | p95 (ms) | Errors |
|---|---|---|---|---|---|---|---|---|
| E1 | `dense` | 0.490 | 0.129 | 0.345 | 0.354 | 250 | 777 | 0 |
| E2 | `sparse` | 0.471 | 0.118 | 0.379 | 0.364 | 183 | 194 | 0 |
| E3 | `hybrid` | 0.578 | 0.153 | 0.458 | 0.431 | 244 | 399 | 0 |
| E4 | `hybrid_rerank` | 0.588 | 0.165 | 0.456 | 0.461 | 9,986 | 11,402 | 0 |
| E5 | `hybrid_qt_rerank` | 0.588 | 0.165 | 0.456 | 0.461 | 10,285 | 12,032 | 0 |

### Recall@5 by question category

| Exp | comparison | limitations | multi_document | multi_query | simple_factual |
|---|---|---|---|---|---|
| E1 | 0.333 | 1.000 | 0.250 | 0.417 | 0.500 |
| E2 | 0.667 | 0.500 | 0.250 | 0.250 | 0.500 |
| E3 | 0.667 | 0.750 | 0.750 | 0.417 | 0.500 |
| E4 | 0.833 | 1.000 | 0.750 | 0.000 | 0.500 |
| E5 | 0.833 | 1.000 | 0.750 | 0.000 | 0.500 |

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
