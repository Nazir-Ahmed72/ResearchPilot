"""Runs an evaluation end to end and stores reproducible results.

Output directory ``experiments/results/<UTC timestamp>_<name>/``:
* ``config.json``       options, models, dataset, code version
* ``label_check.json``  every evidence label validated against the indexed collection
* ``retrieval.jsonl``   one row per (experiment, question): ranked chunks + metrics
* ``generation.jsonl``  one row per question: answer, citations, metrics, Ragas scores
* ``summary.json``      aggregated metrics
* ``report.md``         human-readable comparison
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from app.core.config import PROJECT_ROOT
from app.core.resources import Resources
from app.core.types import RetrievalStrategy
from app.database.models import EvaluationRun, JobStatus
from app.database.repositories.papers import CollectionRepository
from app.database.session import session_scope
from app.evaluation import generation_eval, retrieval_eval
from app.evaluation.dataset import load_dataset, validate
from app.evaluation.report import render_report
from app.retrieval.types import RetrievalFilters

logger = logging.getLogger(__name__)
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results"
GENERATION_KEY = "GEN"


@dataclass
class EvaluationOptions:
    dataset: str = "rag_papers_v1"
    experiments: list[str] = field(default_factory=lambda: list(retrieval_eval.EXPERIMENTS))
    top_k: int = 5
    include_generation: bool = False
    generation_strategy: str = RetrievalStrategy.HYBRID_RERANK.value
    question_ids: list[str] | None = None
    limit: int | None = None
    ragas: bool = False
    ragas_metrics: list[str] | None = None
    pace_seconds: float = 0.0
    collection: str | None = None
    langsmith: bool = False
    name: str = "eval"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001 - not a git checkout (e.g. a container image)
        return None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _set_runs(resources: Resources, run_ids: dict[str, UUID] | None, **fields: Any) -> None:
    if not run_ids:
        return
    with session_scope(resources.session_factory) as s:
        for run_id in run_ids.values():
            run = s.get(EvaluationRun, run_id)
            if run is not None:
                for key, value in fields.items():
                    setattr(run, key, value)


def run_evaluation(
    resources: Resources,
    options: EvaluationOptions,
    *,
    run_ids: dict[str, UUID] | None = None,
    results_root: Path | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Returns (results directory, summary). ``run_ids`` maps experiment keys (E1..E5, GEN) to
    ``EvaluationRun`` rows that are updated with status and metrics."""
    # Resolved at call time (not as a default argument) so it can be configured/patched.
    results_root = results_root or RESULTS_ROOT
    now = datetime.now(timezone.utc)
    _set_runs(resources, run_ids, status=JobStatus.RUNNING, started_at=now)
    try:
        return _run(resources, options, run_ids, results_root, now, on_progress)
    except Exception as exc:
        _set_runs(
            resources,
            run_ids,
            status=JobStatus.FAILED,
            error_message=f"{type(exc).__name__}: {exc}"[:1000],
            finished_at=datetime.now(timezone.utc),
        )
        raise


def _run(resources, options, run_ids, results_root, now, on_progress):  # type: ignore[no-untyped-def]
    settings = resources.settings
    dataset = load_dataset(options.dataset).subset(options.question_ids, options.limit)
    filters = RetrievalFilters()
    if options.collection:
        with resources.session_factory() as s:
            coll = CollectionRepository(s).get_by_name(options.collection)
            if coll is None:
                raise ValueError(f"Collection {options.collection!r} not found")
            filters = RetrievalFilters(collection_id=str(coll.id))

    corpus = retrieval_eval.load_corpus(resources.qdrant, settings.qdrant_collection, filters)
    label_check = validate(dataset, corpus)  # fail fast if labels do not fit this collection

    out_dir = results_root / f"{now.strftime('%Y%m%dT%H%M%SZ')}_{options.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": options.name,
        "dataset": dataset.name,
        "questions": len(dataset.questions),
        "answerable": sum(q.answerable for q in dataset.questions),
        "papers": len(dataset.papers),
        "created_at": now.isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "top_k": options.top_k,
        "generation_strategy": options.generation_strategy if options.include_generation else None,
        "options": asdict(options),
        "models": {
            "main": settings.llm_model,
            "fast": settings.llm_fast_model,
            "embedding": settings.embedding_model,
            "reranker": settings.reranker_model,
        },
        "retrieval_settings": {
            "candidate_k": settings.retrieval_candidate_k,
            "rerank_candidate_k": settings.rerank_candidate_k,
            "rrf_k": settings.rrf_k,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "label_check.json").write_text(
        json.dumps([c.model_dump() for c in label_check], indent=2), encoding="utf-8"
    )

    # Load models before anything is timed, so latency reflects retrieval, not model loading.
    resources.dense_embedder.embed_query("warm up")
    if resources.reranker is not None:
        resources.reranker.score("warm up", ["warm up"])

    retrieval_summary: dict[str, Any] = {}
    retrieval_rows: list[dict[str, Any]] = []
    if options.experiments:
        if on_progress:
            on_progress("retrieval")
        retrieval_rows = retrieval_eval.evaluate_retrieval(
            resources.retrieval_engine,
            dataset,
            experiments=options.experiments,
            k=options.top_k,
            corpus=corpus,
            filters=filters,
        )
        _write_jsonl(out_dir / "retrieval.jsonl", retrieval_rows)
        retrieval_summary = retrieval_eval.summarize_retrieval(retrieval_rows, options.top_k)

    generation_summary = None
    if options.include_generation:
        rows = generation_eval.evaluate_generation(
            resources.evaluation_graph,
            dataset,
            strategy=options.generation_strategy,
            filters={k: v for k, v in asdict(filters).items() if v} or None,
            tracer=resources.tracer,
            pace_seconds=options.pace_seconds,
            timeout_seconds=settings.research_timeout_seconds,
            on_progress=on_progress,
        )
        if options.ragas:
            from app.evaluation import ragas_eval

            ragas_eval.score_rows(
                rows,
                settings,
                resources.dense_embedder,
                options.ragas_metrics or ragas_eval.DEFAULT_METRICS,
            )
        _write_jsonl(out_dir / "generation.jsonl", rows)
        generation_summary = generation_eval.summarize_generation(rows, settings.llm_pricing)

    summary = {"retrieval": retrieval_summary, "generation": generation_summary}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(
        render_report(meta, retrieval_summary, generation_summary), encoding="utf-8"
    )

    if options.langsmith:
        from app.evaluation.langsmith_sync import sync_to_langsmith

        meta["langsmith"] = sync_to_langsmith(settings, dataset, retrieval_rows, options.top_k)
        (out_dir / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    finished = datetime.now(timezone.utc)
    if run_ids:
        with session_scope(resources.session_factory) as s:
            for key, run_id in run_ids.items():
                run = s.get(EvaluationRun, run_id)
                if run is None:
                    continue
                metrics = (
                    generation_summary if key == GENERATION_KEY else retrieval_summary.get(key)
                ) or {}
                run.metrics = metrics
                run.results_path = str(
                    out_dir.relative_to(PROJECT_ROOT)
                    if out_dir.is_relative_to(PROJECT_ROOT)
                    else out_dir
                )
                run.status = JobStatus.SUCCEEDED
                run.finished_at = finished
    logger.info("Evaluation finished", extra={"results": str(out_dir)})
    return out_dir, summary


def score_saved_generation(
    resources: Resources,
    source_dir: Path,
    *,
    question_ids: list[str] | None = None,
    metrics: list[str] | None = None,
    name: str = "ragas",
    results_root: Path | None = None,
    pace_seconds: float = 0.0,
) -> tuple[Path, dict[str, Any]]:
    """Ragas-score the answers of an EARLIER generation run instead of re-running the workflow.

    Judges exactly the answers that were measured, and costs only judge-model tokens (useful when
    the main model's quota is exhausted)."""
    from app.evaluation import ragas_eval

    rows = [
        json.loads(line)
        for line in (source_dir / "generation.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if question_ids:
        rows = [r for r in rows if r["question_id"] in set(question_ids)]
    rows = [r for r in rows if r.get("completed")]
    if not rows:
        raise ValueError("No completed generation rows to score")
    results_root = results_root or RESULTS_ROOT
    chosen = list(metrics or ragas_eval.DEFAULT_METRICS)
    ragas_eval.score_rows(
        rows, resources.settings, resources.dense_embedder, chosen, pace_seconds=pace_seconds
    )

    now = datetime.now(timezone.utc)
    out_dir = results_root / f"{now.strftime('%Y%m%dT%H%M%SZ')}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    slim = [
        {
            "question_id": r["question_id"],
            "category": r["category"],
            "answerable": r["answerable"],
            "ragas": r.get("ragas", {}),
        }
        for r in rows
    ]
    _write_jsonl(out_dir / "ragas.jsonl", slim)
    summary = {
        "source": str(source_dir.name),
        "judge_model": resources.settings.llm_fast_model,
        "metrics": chosen,
        "questions": [r["question_id"] for r in rows],
        "mean": generation_eval.summarize_generation(rows)["ragas"],
        "scored": {
            m: sum(1 for r in rows if (r.get("ragas") or {}).get(m) is not None) for m in chosen
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        f"# Ragas scores for `{source_dir.name}`",
        "",
        f"* Judge: `{summary['judge_model']}` via Groq; embeddings "
        f"`{resources.settings.embedding_model}`",
        f"* Questions: {', '.join(summary['questions'])}",
        "",
        "| Question | Category | " + " | ".join(chosen) + " |",
        "|---|---|" + "---|" * len(chosen),
    ]
    for r in slim:
        lines.append(
            f"| {r['question_id']} | {r['category']} | "
            + " | ".join(
                "–" if r["ragas"].get(m) is None else f"{r['ragas'][m]:.3f}" for m in chosen
            )
            + " |"
        )
    lines.append(
        "| **mean** | | "
        + " | ".join(
            "–" if summary["mean"].get(m) is None else f"**{summary['mean'][m]:.3f}**"
            for m in chosen
        )
        + " |"
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir, summary
