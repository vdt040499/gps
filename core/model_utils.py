"""Resolve Hugging Face model ids vs optional local snapshots under <repo>/models/."""
from __future__ import annotations

import re
from pathlib import Path

_HUB_ID_RE = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")

_REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_model_id(
    model_path: str | None,
    *,
    local_relative: str,
    hub_id: str,
) -> str:
    """
    If ``<repo>/local_relative`` exists and contains ``config.json``, use that folder.
    Else if ``model_path`` looks like a Hub id (``org/name``), use it.
    Otherwise fall back to ``hub_id`` (download from Hub on first load).
    """
    local = _REPO_ROOT / local_relative
    if local.is_dir() and (local / "config.json").exists():
        return str(local)

    if model_path is None:
        return hub_id

    s = model_path.strip()
    candidate = Path(s)
    if not candidate.is_absolute():
        candidate = (_REPO_ROOT / s.lstrip("./")).resolve()
    if candidate.is_dir() and (candidate / "config.json").exists():
        return str(candidate)

    if _HUB_ID_RE.match(s):
        return s

    return hub_id
