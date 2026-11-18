"""Data access for papers, collections and ingestion jobs.

Repositories only query and stage changes; transaction boundaries (commit/rollback) belong to the
calling service, which knows when a unit of work is complete.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.database.models import IngestionJob, Paper, PaperCollection, PaperStatus


class PaperRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, paper_id: UUID) -> Paper | None:
        return self.session.get(Paper, paper_id)

    def get_with_jobs(self, paper_id: UUID) -> Paper | None:
        stmt = (
            select(Paper)
            .where(Paper.id == paper_id)
            .options(selectinload(Paper.ingestion_jobs), selectinload(Paper.collection))
        )
        return self.session.scalars(stmt).first()

    def get_by_hash(self, content_hash: str) -> Paper | None:
        return self.session.scalars(select(Paper).where(Paper.content_hash == content_hash)).first()

    def list(
        self,
        *,
        limit: int,
        offset: int,
        status: PaperStatus | None = None,
        collection_id: UUID | None = None,
    ) -> tuple[Sequence[Paper], int]:
        conditions = []
        if status is not None:
            conditions.append(Paper.status == status)
        if collection_id is not None:
            conditions.append(Paper.collection_id == collection_id)
        total = self.session.scalar(select(func.count()).select_from(Paper).where(*conditions))
        stmt = (
            select(Paper)
            .where(*conditions)
            .options(selectinload(Paper.collection))
            .order_by(Paper.created_at.desc(), Paper.id)
            .limit(limit)
            .offset(offset)
        )
        return self.session.scalars(stmt).all(), int(total or 0)

    def add(self, paper: Paper) -> Paper:
        self.session.add(paper)
        return paper

    def delete(self, paper: Paper) -> None:
        self.session.delete(paper)


class CollectionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_name(self, name: str) -> PaperCollection | None:
        stmt = select(PaperCollection).where(PaperCollection.name == name)
        return self.session.scalars(stmt).first()

    def get_or_create(self, name: str) -> PaperCollection:
        existing = self.get_by_name(name)
        if existing is not None:
            return existing
        collection = PaperCollection(name=name)
        self.session.add(collection)
        self.session.flush()
        return collection


class IngestionJobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, job_id: UUID) -> IngestionJob | None:
        return self.session.get(IngestionJob, job_id)

    def add(self, job: IngestionJob) -> IngestionJob:
        self.session.add(job)
        return job
