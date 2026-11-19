"""Run the evaluation suite from the command line.

Examples:
    # Retrieval experiments E1-E5 (cheap: only E5 calls the LLM)
    python scripts/run_evaluation.py --name retrieval

    # Add the end-to-end research workflow, paced for Groq's free tier
    python scripts/run_evaluation.py --name full --generation --pace 45

    # Ragas on a small subset, and publish retrieval experiments to LangSmith
    python scripts/run_evaluation.py --name ragas --experiments E4 --generation --ragas \
        --questions q01,q09,q14,q18 --pace 45 --langsmith
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.core.resources import Resources  # noqa: E402
from app.evaluation.runner import (  # noqa: E402
    EvaluationOptions,
    run_evaluation,
    score_saved_generation,
)


def main() -> None:
    p = argparse.ArgumentParser(description="ResearchPilot evaluation")
    p.add_argument("--name", default="eval")
    p.add_argument("--dataset", default="rag_papers_v1")
    p.add_argument(
        "--experiments",
        default="E1,E2,E3,E4,E5",
        help="comma-separated, or 'none' to skip retrieval experiments",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--generation", action="store_true", help="run the research workflow too")
    p.add_argument("--generation-strategy", default="hybrid_rerank")
    p.add_argument("--questions", default="", help="comma-separated question ids")
    p.add_argument("--limit", type=int)
    p.add_argument("--ragas", action="store_true")
    p.add_argument("--ragas-metrics", default="")
    p.add_argument("--pace", type=float, default=0.0, help="seconds between generation runs")
    p.add_argument("--collection")
    p.add_argument("--langsmith", action="store_true")
    p.add_argument("--ragas-from", help="Ragas-score the answers of an earlier results directory")
    a = p.parse_args()

    settings = get_settings()
    configure_logging("WARNING", json_output=False)
    if a.ragas_from:
        resources = Resources.from_settings(settings)
        try:
            out_dir, _ = score_saved_generation(
                resources,
                Path(a.ragas_from),
                question_ids=[q.strip() for q in a.questions.split(",") if q.strip()] or None,
                metrics=[m.strip() for m in a.ragas_metrics.split(",") if m.strip()] or None,
                name=a.name,
                pace_seconds=a.pace,
            )
        finally:
            resources.close()
        print(f"Results: {out_dir}")
        print((out_dir / "report.md").read_text(encoding="utf-8"))
        return
    options = EvaluationOptions(
        dataset=a.dataset,
        experiments=[]
        if a.experiments == "none"
        else [e.strip() for e in a.experiments.split(",")],
        top_k=a.top_k,
        include_generation=a.generation,
        generation_strategy=a.generation_strategy,
        question_ids=[q.strip() for q in a.questions.split(",") if q.strip()] or None,
        limit=a.limit,
        ragas=a.ragas,
        ragas_metrics=[m.strip() for m in a.ragas_metrics.split(",") if m.strip()] or None,
        pace_seconds=a.pace,
        collection=a.collection,
        langsmith=a.langsmith,
        name=a.name,
    )
    resources = Resources.from_settings(settings)
    try:
        out_dir, summary = run_evaluation(
            resources, options, on_progress=lambda s: print(f"  ... {s}", flush=True)
        )
    finally:
        resources.close()
    print(f"\nResults: {out_dir}")
    print((out_dir / "report.md").read_text(encoding="utf-8"))
    if summary.get("generation") is None and not summary.get("retrieval"):
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
