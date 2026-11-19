"""Evaluation runs: creation/listing (API side) and batch execution (worker side).

One API request creates one ``EvaluationRun`` row per retrieval experiment (E1..E5) and one for
the end-to-end workflow ("GEN") if requested, sharing a ``batch_id``. The worker executes the
batch with the same ``run_evaluation`` used by ``scripts/run_evaluation.py``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import DependencyUnavailableError, InvalidInputError, NotFoundError
from app.core.resources import Resources
from app.core.types import RetrievalStrategy
from app.database.models import EvaluationRun, JobStatus
from app.evaluation.dataset import load_dataset
from app.evaluation.retrieval_eval import EXPERIMENTS
from app.evaluation.runner import GENERATION_KEY, EvaluationOptions, run_evaluation
from app.workers.dispatch import TaskDispatcher

logger = logging.getLogger(__name__)


class EvaluationService:
    def __init__(self, session: Session, dispatcher: TaskDispatcher) -> None:
        self.session = session
        self.dispatcher = dispatcher

    def create(self, options: EvaluationOptions) -> tuple[UUID, list[EvaluationRun]]:
        unknown = set(options.experiments) - set(EXPERIMENTS)
        if unknown:
            raise InvalidInputError(
                f"Unknown experiments: {sorted(unknown)}", details={"allowed": list(EXPERIMENTS)}
            )
        if not options.experiments and not options.include_generation:
            raise InvalidInputError("Select at least one experiment or include_generation")
        try:
            dataset = load_dataset(options.dataset)
        except FileNotFoundError as exc:
            raise NotFoundError("Dataset not found", details={"dataset": options.dataset}) from exc
        except ValueError as exc:
            raise InvalidInputError(f"Invalid dataset: {exc}") from exc

        batch_id = uuid.uuid4()
        config = {"batch_id": str(batch_id), "options": asdict(options)}
        keys = list(options.experiments) + ([GENERATION_KEY] if options.include_generation else [])
        runs = []
        for key in keys:
            strategy = (
                RetrievalStrategy(options.generation_strategy)
                if key == GENERATION_KEY
                else EXPERIMENTS[key]
            )
            run = EvaluationRun(
                id=uuid.uuid4(),
                name=options.name,
                experiment=key,
                retrieval_strategy=strategy,
                dataset_name=dataset.name,
                status=JobStatus.QUEUED,
                config=config,
            )
            self.session.add(run)
            runs.append(run)
        self.session.commit()
        try:
            task_id = self.dispatcher.enqueue_evaluation(batch_id)
        except DependencyUnavailableError as exc:
            for run in runs:
                run.status = JobStatus.FAILED
                run.error_message = exc.message
            self.session.commit()
            raise
        for run in runs:
            run.config = {**run.config, "task_id": task_id}
        self.session.commit()
        return batch_id, runs

    def list(self, *, limit: int, experiment: str | None = None) -> Sequence[EvaluationRun]:
        stmt = select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit)
        if experiment:
            stmt = stmt.where(EvaluationRun.experiment == experiment)
        return self.session.scalars(stmt).all()

    def get(self, run_id: UUID) -> EvaluationRun:
        run = self.session.get(EvaluationRun, run_id)
        if run is None:
            raise NotFoundError("Evaluation run not found", details={"run_id": str(run_id)})
        return run


def execute_batch(resources: Resources, batch_id: UUID) -> None:
    """Worker entry point: run every experiment of a batch in one evaluation pass."""
    with resources.session_factory() as s:
        runs = [
            r
            for r in s.scalars(select(EvaluationRun)).all()
            if (r.config or {}).get("batch_id") == str(batch_id)
        ]
        if not runs:
            logger.warning("Evaluation batch not found", extra={"batch_id": str(batch_id)})
            return
        options = EvaluationOptions(**runs[0].config["options"])
        run_ids = {r.experiment: r.id for r in runs}
    try:
        run_evaluation(resources, options, run_ids=run_ids)
    except Exception:  # status/error already recorded on the runs by run_evaluation
        logger.error("Evaluation batch failed", extra={"batch_id": str(batch_id)}, exc_info=True)
