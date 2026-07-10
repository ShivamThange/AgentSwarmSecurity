from __future__ import annotations

from . import replay
from .graph import TwinStore
from .models import WhatIfPreview

def build_preview(store: TwinStore, root_id: str) -> WhatIfPreview | None:
    return replay.build_preview(store, root_id)
