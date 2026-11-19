# Evaluation

How ResearchPilot is measured, and what the measurements say. Implemented in M5.

## Dataset: `rag_papers_v1`

20 questions over the four indexed papers (RAG 2005.11401, DPR 2004.04906, GraphRAG 2404.16130,
Self-RAG 2310.11511), in `app/evaluation/datasets/rag_papers_v1.json`:

| Category | Count | Example |
|---|---|---|
| `simple_factual` | 8 | What generator model does RAG use, and how many parameters does it have? |
| `comparison` | 3 | Compare how RAG and Self-RAG decide when to retrieve passages. |
| `multi_document` | 2 | How does RAG use DPR, and how was the DPR retriever trained? |
| `limitations` | 2 | What limitations of its evaluation does the GraphRAG paper report? |
| `multi_query` | 2 | What retriever, generator and knowledge source does RAG combine? |
| `insufficient_evidence` | 3 | How many parameters does GPT-4 have? (not answerable from the collection) |

### Content-based evidence labels
Each answerable question has 1–3 **evidence labels**. A label names a paper (by a title fragment)
and phrases the supporting passage must contain, e.g.

```json
{"paper": "rag", "contains": ["BART-large", "400M parameters"]}
```

A retrieved chunk is *relevant* to a label if it comes from that paper and contains every phrase
(`"a|b"` = either; case- and whitespace-insensitive). Labels describe the evidence rather than a
chunk id, so they survive re-chunking and re-ingestion.

**How the labels were made.** Every label was written after searching the actual indexed chunk
text (not from memory of the papers), and `validate()` re-checks all labels against the
collection before every run: a label must match at least one chunk (otherwise it could never be
found) and at most four (otherwise it does not identify specific evidence). The validation result
is saved with each run (`label_check.json`). For `rag_papers_v1`, every label matches 1–3 chunks.

Unanswerable questions were checked to have no supporting text: "ColBERT", "late interaction",
"MMLU" and GPT-4's size appear in none of the 243 chunks.

## Metrics

### Retrieval (answerable questions only)
| Metric | Definition |
|---|---|
| Recall@K | Share of the question's evidence labels covered by the top K chunks |
| Precision@K | Share of the top K chunks that satisfy any label |
| MRR | 1 / rank of the first relevant chunk (0 if none in the top K) |
| nDCG@K | Binary-gain DCG of the top K over the ideal DCG with min(K, relevant chunks in the collection) hits |
| Latency | Per query, after model warm-up |

Recall counts *labels*, not chunks: a multi-part question is fully recalled only if every
needed piece of evidence is retrieved.

### End-to-end research workflow
| Metric | Definition | Needs an LLM judge? |
|---|---|---|
| Completion rate | Runs that finished without error | no |
| Abstention accuracy | Unanswerable questions answered with `evidence_sufficient = false` | no |
| False abstention rate | Answerable questions wrongly declared insufficient | no |
| Evidence recall | Labels covered by the evidence the workflow actually used | no |
| Citation label accuracy | Cited references whose chunk satisfies one of the question's labels | no |
| Citation validity | Cited evidence IDs that exist (fabricated citations lower it) | no |
| Citation coverage | Reported claims with at least one valid citation | no |
| Support rate | Claims the verifier judged supported (partial = 0.5) | yes |
| Ragas: faithfulness, answer relevancy, answer correctness, context recall | Ragas 0.4 metrics | yes |
| Latency, node latency, tokens per query, cost per query, retry and repair rates | from run metadata | no |

### Ragas setup
Ragas 0.4 `metrics.collections`, judged by `openai/gpt-oss-20b` through Groq's
OpenAI-compatible endpoint, with the project's BGE embeddings for answer relevancy and
correctness. A metric that fails is recorded as missing (`None`), never as 0. Ragas makes several
LLM calls per metric and question, so on Groq's free tier it runs on a subset.

## Limitations of LLM-as-a-judge
* The judge is itself an LLM: it can be wrong, inconsistent across runs, and sensitive to prompt
  wording.
* It may prefer fluent, longer or more confident answers over correct but hedged ones.
* Here the judge (gpt-oss-20b) is from the same model family as the generator (gpt-oss-120b),
  which risks self-preference bias.
* Judges see only the provided contexts; they cannot tell a true statement that is missing from
  the evidence from a hallucination.

That is why the label-based metrics (recall, citation label accuracy, abstention) are the anchor,
and LLM-judged scores are reported next to them, not instead of them.

## Running

```bash
python scripts/run_evaluation.py --name retrieval                    # E1-E5, cheap
python scripts/run_evaluation.py --name full --experiments none --generation --pace 30
python scripts/run_evaluation.py --name ragas --experiments none --generation --ragas \
    --questions q01,q05,q09,q12,q14,q17 --pace 30
python scripts/run_evaluation.py --name retrieval --langsmith        # publish to LangSmith
```

or `POST /api/v1/evaluations/run` (background), then `GET /api/v1/evaluations/results`.
Each run writes `experiments/results/<timestamp>_<name>/` with `config.json` (options, models,
code version), `label_check.json`, `retrieval.jsonl`, `generation.jsonl`, `summary.json` and
`report.md`.

## Results (2026-11-19)

Measured on the live system: 4 papers / 243 chunks in Qdrant Cloud, Groq free tier, laptop CPU.
Raw data: `experiments/results/20261119T143417Z_final-retrieval/` and
`experiments/results/20261119T144103Z_final-generation/`. The retrieval experiments were run twice
with identical scores (latency aside).

### Retrieval, top-K = 5 (17 answerable questions)

| Exp | Strategy | Recall@5 | Precision@5 | MRR | nDCG@5 | Mean latency |
|---|---|---|---|---|---|---|
| E1 | dense | 0.490 | 0.129 | 0.345 | 0.354 | 0.25 s |
| E2 | sparse (BM25) | 0.471 | 0.118 | 0.379 | 0.364 | 0.18 s |
| E3 | hybrid (RRF) | 0.578 | 0.153 | 0.458 | 0.431 | 0.24 s |
| E4 | hybrid + rerank | **0.588** | **0.165** | 0.456 | **0.461** | 10.0 s |
| E5 | hybrid + query transformation + rerank | 0.588 | 0.165 | 0.456 | 0.461 | 10.3 s |

Recall@5 by category:

| Exp | comparison | limitations | multi_document | multi_query | simple_factual |
|---|---|---|---|---|---|
| E1 | 0.333 | 1.000 | 0.250 | 0.417 | 0.500 |
| E2 | 0.667 | 0.500 | 0.250 | 0.250 | 0.500 |
| E3 | 0.667 | 0.750 | 0.750 | 0.417 | 0.500 |
| E4 | 0.833 | 1.000 | 0.750 | 0.000 | 0.500 |
| E5 | 0.833 | 1.000 | 0.750 | 0.000 | 0.500 |

What this shows (directional; 17 questions):
* **Hybrid beats either retriever alone** (+9–11 recall points over dense/sparse at the same
  latency), and fusion helps most on multi-document questions (0.25 → 0.75).
* **Reranking adds little at top-5** (+1 recall point, +3 nDCG points) for ~10 s of CPU time per
  query. It clearly helps comparison questions (0.67 → 0.83) but **hurts multi-part questions**
  (0.42 → 0.00): scoring every candidate against the whole question favours one broad passage
  over several specific ones.
* **Query transformation (E5) changed nothing measurable here.** It generated 2–4 queries per
  question, but the reranker re-scores against the original question and ends at the same top 5.
* **Simple factual recall is stuck at 0.50 for every strategy.** Questions phrased about "RAG"
  in general (q01 retriever, q02 generator) retrieve the abstract/introduction/method-overview
  chunks; the specific answering chunk never says "RAG". This is the main retrieval weakness.

### End-to-end research workflow (strategy E4, 20 questions)

| Metric | Value |
|---|---|
| Completion rate | 0.90 (18/20) — one schema-validation failure (since fixed), one Groq daily quota stop |
| Abstention accuracy (unanswerable) | **1.00** (2/2 completed) |
| False abstention rate | **0.00** |
| Evidence recall (labels covered by evidence used) | **0.50** |
| Citation label accuracy | **0.30** |
| Citation validity / coverage | 1.00 / 1.00 |
| Support rate (LLM verifier) | 0.97 |
| Repair rate / retrieval retry rate | 0.17 / 0.00 |
| Latency p50 / mean / p95 | 13.5 s / 17.0 s / 72.2 s |
| Tokens per query (mean) | 7,200 |

Mean node latency: rerank 8.2 s, synthesize 3.9 s, verify 2.6 s, retrieve 1.0 s, evaluate 0.9 s,
understand 0.8 s, plan 0.5 s. The p95 outlier (72 s) was a comparison question that needed a
repair pass while Groq was throttling.

**The most important finding is the gap between the two kinds of metric.** The LLM verifier
judged 97% of claims supported by their citations, but the label-based metrics show that only
half of the specifically needed evidence was used and 30% of citations point to the labelled
passages. The answers are faithful to what they cite, but often cite general overview passages
(e.g. "RAG uses BART" from the introduction rather than the section stating BART-large, 400M
parameters), so they are less specific and occasionally incomplete. An LLM judge alone would have
hidden this.

### Ragas (6 saved answers: q01, q05, q10, q12, q14, q17; judge gpt-oss-20b)

| Metric | Mean | Scored |
|---|---|---|
| Answer relevancy | 0.903 | 6/6 |
| Context recall | 0.500 | 5/6 |
| Faithfulness | not measured | 0/6 |
| Answer correctness | not measured | 0/6 |

Context recall (0.50, judged by an LLM against the reference answers) independently agrees with
the label-based evidence recall (0.50). Faithfulness and answer correctness make the most judge
calls; they failed on Groq rate limits in the first attempt and on the judge model's **daily**
token limit in a paced re-run (failures are recorded as missing, not 0). Run on a fresh quota
day: `python scripts/run_evaluation.py --name ragas-faithfulness --ragas-from
experiments/results/20261119T144103Z_final-generation --questions q01,q05,q10,q12,q14,q17
--ragas-metrics faithfulness,answer_correctness --pace 60`. (A single isolated faithfulness call
for q05 succeeded with a score of 1.0, confirming the setup works.)

### Not yet measured
* The generation run reached **Groq's free-tier daily limit (200k tokens/day for
  gpt-oss-120b)** after ~18 questions. A full run of all 20 questions, and runs with other
  strategies (E1–E5 end to end), need either a second day of quota or a paid tier.

### Improvement candidates suggested by the data
1. Rerank per sub-question (or diversify after reranking) for multi-part questions.
2. Add the paper's abbreviation (e.g. "RAG") to the context header of every chunk, so specific
   chunks match questions phrased with the system's name.
3. Consider a smaller/faster reranker: at top-5 it adds ~1 recall point for ~10 s.

