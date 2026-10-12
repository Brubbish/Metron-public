"""Load Debian default compiler/header search metadata."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from .metadata import DEFAULT_DEBIAN_VERSION, normalize_debian_version

DEFAULT_HEADER_ARCH = 'x86_64-linux-gnu'
DEFAULT_HEADER_DIR = Path(__file__).resolve().parent / 'data' / 'default_headers'


@lru_cache(maxsize=None)
def load_default_header_filter(
    debian_version: str = DEFAULT_DEBIAN_VERSION,
    arch: str = DEFAULT_HEADER_ARCH,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return default system header path-suffix index and basenames.

    Missing metadata is treated as an empty filter so unsupported suites do not
    change extractor behavior.
    """
    path = os.environ.get('ENHANCED_DEP_DEFAULT_HEADERS_PATH')
    metadata_path = Path(path) if path else default_header_path(debian_version, arch)
    if not metadata_path.exists():
        return frozenset(), frozenset()
    data = json.loads(metadata_path.read_text())
    suffix_index = set()
    for value in data.get('headers_exact', []):
        suffix_index.update(header_path_suffixes(value))
    return (
        frozenset(suffix_index),
        frozenset(_normalize_header(value) for value in data.get('headers_basename', [])),
    )


def default_header_path(
    debian_version: str = DEFAULT_DEBIAN_VERSION,
    arch: str = DEFAULT_HEADER_ARCH,
) -> Path:
    version = normalize_debian_version(debian_version)
    return DEFAULT_HEADER_DIR / f'{version}-{arch}.json'


def _normalize_header(value: str) -> str:
    return str(value).strip().lstrip('/').lower()


def header_path_suffixes(value: str) -> tuple[str, ...]:
    normalized = _normalize_header(value)
    parts = [part for part in normalized.split('/') if part]
    return tuple('/'.join(parts[index:]) for index in range(len(parts)))
