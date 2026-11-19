"""Markdown report comparing the experiments of one evaluation run."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CAVEATS = """## Interpreting these results

* **Small dataset.** {n} questions over {papers} papers; differences of a few points between
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
"""


def _fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3f}" if value <= 1.0 else f"{value:,.0f}"
    return str(value)


def render_report(
    meta: Mapping[str, Any], retrieval: Mapping[str, Any], generation: Mapping[str, Any] | None
) -> str:
    k = meta["top_k"]
    keys = [f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}"]
    lines = [
        f"# Evaluation report: {meta['name']}",
        "",
        f"* Dataset: `{meta['dataset']}` ({meta['questions']} questions, "
        f"{meta['answerable']} answerable)",
        f"* Date (UTC): {meta['created_at']}",
        f"* Code version: `{meta.get('git_commit') or 'unknown'}`",
        f"* Models: main `{meta['models']['main']}`, fast `{meta['models']['fast']}`, "
        f"embeddings `{meta['models']['embedding']}`, reranker `{meta['models']['reranker']}`",
        f"* Retrieval top-K: {k}",
        "",
    ]
    if retrieval:
        lines += [
            "## Retrieval (E1-E5)",
            "",
            "| Exp | Strategy | " + " | ".join(keys) + " | Latency mean (ms) | p95 (ms) | Errors |",
            "|---|---|" + "---|" * len(keys) + "---|---|---|",
        ]
        for exp, s in retrieval.items():
            lines.append(
                f"| {exp} | `{s['strategy']}` | "
                + " | ".join(_fmt(s[m]) for m in keys)
                + f" | {_fmt(s['latency_ms_mean'])} | {_fmt(s['latency_ms_p95'])} | {s['errors']} |"
            )
        cats = sorted({c for s in retrieval.values() for c in s["by_category"]})
        lines += [
            "",
            f"### Recall@{k} by question category",
            "",
            "| Exp | " + " | ".join(cats) + " |",
            "|---|" + "---|" * len(cats),
        ]
        for exp, s in retrieval.items():
            lines.append(
                f"| {exp} | "
                + " | ".join(_fmt(s["by_category"].get(c, {}).get(f"recall@{k}")) for c in cats)
                + " |"
            )
        lines.append("")
    if generation:
        g = generation
        lines += [
            f"## End-to-end research workflow (strategy `{meta.get('generation_strategy')}`)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Completion rate | {_fmt(g['completion_rate'])} "
            f"({g['completed']}/{g['questions']}) |",
            f"| Abstention accuracy (unanswerable) | {_fmt(g['abstention_accuracy'])} |",
            f"| False abstention rate (answerable) | {_fmt(g['false_abstention_rate'])} |",
            f"| Evidence recall (labels covered by used evidence) | {_fmt(g['evidence_recall'])} |",
            f"| Citation label accuracy (cited chunks matching labels) | "
            f"{_fmt(g['citation_label_accuracy'])} |",
            f"| Citation validity (IDs exist) | {_fmt(g['citation_validity'])} |",
            f"| Citation coverage (reported claims cited) | {_fmt(g['citation_coverage'])} |",
            f"| Support rate (LLM verifier) | {_fmt(g['support_rate'])} |",
            f"| Retrieval retry rate | {_fmt(g['retry_rate'])} |",
            f"| Repair rate | {_fmt(g['repair_rate'])} |",
            f"| Latency mean / p50 / p95 (ms) | {_fmt(g['latency_ms_mean'])} / "
            f"{_fmt(g['latency_ms_p50'])} / {_fmt(g['latency_ms_p95'])} |",
            f"| Tokens per query (mean) | {_fmt(g['tokens_per_query_mean'])} |",
            f"| Cost per query (USD) | {_fmt(g['cost_usd_per_query'])} |",
        ]
        for name, value in (g.get("ragas") or {}).items():
            lines.append(f"| Ragas {name} | {_fmt(value)} |")
        lines += ["", "### Mean node latency (ms)", "", "| Node | ms |", "|---|---|"]
        lines += [f"| {n} | {_fmt(v)} |" for n, v in g["node_latency_ms_mean"].items()]
        lines.append("")
    lines.append(CAVEATS.format(n=meta["questions"], papers=meta["papers"]))
    return "\n".join(lines)
