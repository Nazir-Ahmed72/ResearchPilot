"""Celery entry point for the worker CLI: ``celery -A app.workers.celery_app worker ...``.

Kept separate from ``celery_config`` so importing the factory (API, tests) never reads settings
as a side effect.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.workers.celery_config import create_celery_app

celery_app = create_celery_app(get_settings())
