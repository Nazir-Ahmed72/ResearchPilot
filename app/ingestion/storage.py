"""Original-PDF storage.

Files are stored under server-generated names (``<paper uuid>.pdf``), so user input never
reaches a filesystem path. ``_resolve`` still verifies every key stays inside the root, as a
second line of defence. Local disk is shared by the API and worker on one machine. A cloud
deployment would swap in an object-storage implementation with the same three methods.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from uuid import UUID

_SAFE_KEY = re.compile(r"^[0-9a-f-]{36}\.pdf$")


class LocalFileStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _resolve(self, key: str) -> Path:
        if not _SAFE_KEY.match(key):
            raise ValueError(f"Invalid storage key: {key!r}")
        path = (self.root / key).resolve()
        if path.parent != self.root:
            raise ValueError("Storage key escapes the storage root")
        return path

    def save(self, paper_id: UUID, data: bytes) -> str:
        key = f"{paper_id}.pdf"
        path = self._resolve(key)
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pdf.part")
        tmp.write_bytes(data)
        os.replace(tmp, path)  # atomic: readers never see a half-written file
        return key

    def read(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def delete(self, key: str) -> None:
        self._resolve(key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()
