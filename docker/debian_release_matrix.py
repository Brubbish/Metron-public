from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


RELEASE_ORDER = [
    "etch",
    "lenny",
    "squeeze",
    "wheezy",
    "jessie",
    "stretch",
    "buster",
    "bullseye",
    "bookworm",
    "trixie",
]


ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE_PATH = ROOT / "debian_repository_profiles.json"


def profile_path() -> Path:
    return Path(os.environ.get("THEAUTO_REPOSITORY_PROFILE", DEFAULT_PROFILE_PATH))


def load_repository_profile(path: Path | None = None) -> dict[str, Any]:
    source = path or profile_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    releases = payload.get("releases")
    if payload.get("schema_version") != 1 or not isinstance(releases, dict):
        raise ValueError(f"invalid repository profile: {source}")
    missing = [release for release in RELEASE_ORDER if release not in releases]
    if missing:
        raise ValueError(f"repository profile misses releases: {', '.join(missing)}")
    for release in RELEASE_ORDER:
        config = releases[release]
        if not all(config.get(key) for key in ("base_image", "base_url", "repositories", "components")):
            raise ValueError(f"incomplete repository profile for {release}")
    return payload


def repository_profile_hash(payload: dict[str, Any] | None = None) -> str:
    canonical = json.dumps(
        payload or load_repository_profile(),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


REPOSITORY_PROFILE = load_repository_profile()
REPOSITORY_PROFILE_HASH = repository_profile_hash(REPOSITORY_PROFILE)
RELEASE_CONFIG = REPOSITORY_PROFILE["releases"]
