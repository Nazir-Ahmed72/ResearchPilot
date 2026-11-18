"""Direct access to the retrieval engine, for inspecting and comparing strategies E1-E5.

Not part of the research workflow itself (that is ``/research``), but invaluable for seeing
exactly which chunks each strategy returns and why (per-stage scores and ranks).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_resources
from app.api.schemas.retrieval import SearchRequest, SearchResponse
from app.core.exceptions import DependencyUnavailableError, ErrorResponse, NotFoundError
from app.core.resources import Resources
from app.database.repositories.papers import CollectionRepository
from app.retrieval.engine import RerankerUnavailableError
from app.retrieval.types import RetrievalFilters

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search the paper collection with a chosen retrieval strategy (E1-E5)",
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def search(
    body: SearchRequest,
    db: Session = Depends(get_db),
    resources: Resources = Depends(get_resources),
) -> SearchResponse:
    collection_id = None
    if body.collection:
        collection = CollectionRepository(db).get_by_name(body.collection)
        if collection is None:
            raise NotFoundError("Collection not found", details={"collection": body.collection})
        collection_id = str(collection.id)
    filters = RetrievalFilters(
        collection_id=collection_id,
        paper_ids=tuple(str(p) for p in body.paper_ids),
        sections=tuple(body.sections),
        year_from=body.year_from,
        year_to=body.year_to,
    )
    try:
        result = resources.retrieval_engine.retrieve(
            body.query, strategy=body.strategy, top_k=body.top_k, filters=filters
        )
    except RerankerUnavailableError as exc:
        raise DependencyUnavailableError(str(exc)) from exc
    return SearchResponse.from_result(result)
