"""Profile-scoped cache for direct dependency mappings.

Accumulates (normalized_matched_text → packages) mappings across projects
to reduce LLM calls. Cache is distro-independent; validation against
valid_packages happens at query time.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass

from .metadata import artifact_cache_path, metadata_profile_hash

_CONFIDENCE_DIRECT = 100
_CONFIDENCE_LLM = 1
_USE_THRESHOLD = 10

_lock = threading.Lock()

_STRIP_KEYWORDS_RE = re.compile(
    r'\b(REQUIRED|QUIET|CONFIG|MODULE|NO_MODULE|EXACT|GLOBAL|'
    r'IMPORTED_TARGET|NO_CMAKE_PATH|NO_CMAKE_ENVIRONMENT_PATH|'
    r'COMPONENTS|OPTIONAL_COMPONENTS)\b',
    re.IGNORECASE,
)
_STRIP_VERSION_RE = re.compile(r'(?<![.\w])\d+\.\d+\.\d+[\d.]*(?![.\w])')
_STRIP_VERSION_OP_RE = re.compile(r'[><=!]+\s*[\d.]+')
_STRIP_MESON_VERSION_RE = re.compile(r",?\s*version\s*:\s*'[^']*'")


def resolve_variables(text: str, build_variables: dict[str, str] | None = None) -> str:
    """Replace ${VAR}, $(VAR), Autoconf $VAR, and @VAR@ with known values."""
    if not build_variables or ('$' not in text and '@' not in text):
        return text
    for var, val in build_variables.items():
        if isinstance(val, (list, tuple)):
            val = next((str(item) for item in reversed(val) if str(item)), '')
        else:
            val = str(val)
        text = text.replace(f'${{{var}}}', val)
        text = text.replace(f'$({var})', val)
        text = text.replace(f'@{var}@', val)
        text = re.sub(rf'\${re.escape(var)}\b', lambda _match, value=val: value, text)
    return text


def normalize_cache_key(text: str, build_variables: dict[str, str] | None = None) -> str:
    """Normalize matched_text for use as cache key.

    First resolves variables, then strips keywords/versions.
    Returns '' if unresolved variables remain (not cacheable).
    """
    text = text.strip()
    text = resolve_variables(text, build_variables)
    # If unresolved variables remain, don't cache
    if (
        '${' in text
        or '$(' in text
        or re.search(r'\$[A-Za-z_][A-Za-z0-9_]*\b', text)
        or re.search(r'@[A-Za-z0-9_]+@', text)
    ):
        return ''
    text = _STRIP_KEYWORDS_RE.sub('', text)
    text = _STRIP_VERSION_RE.sub('', text)
    text = _STRIP_VERSION_OP_RE.sub('', text)
    text = _STRIP_MESON_VERSION_RE.sub('', text)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = ' '.join(text.split())
    return text


@dataclass
class CacheEntry:
    packages: list[str]
    confidence: int
    source: str


class DependencyCache:
    """Thread-safe matched_text → packages cache with persistence."""

    def __init__(self, distro: str = 'trixie'):
        self.distro = distro
        self.profile_hash = metadata_profile_hash(distro)
        self._cache: dict[str, CacheEntry] = {}
        self._dirty = False

    def query(self, matched_text: str, valid_packages: set[str] | None = None,
              build_variables: dict[str, str] | None = None) -> list[str] | None:
        """Query cache. Returns package list or None if miss/low confidence."""
        key = normalize_cache_key(matched_text, build_variables)
        if not key:
            return None
        entry = self._cache.get(key)
        if entry is None or entry.confidence < _USE_THRESHOLD:
            return None
        if valid_packages is not None:
            valid = [p for p in entry.packages if p in valid_packages]
            if not valid:
                return None
            return valid
        return list(entry.packages)

    def update(self, matched_text: str, packages: list[str], source: str,
               build_variables: dict[str, str] | None = None) -> None:
        """Update cache. Same key+packages → confidence++. Direct source always wins."""
        # Broad artifact/LLM guesses are project-local candidates. Persisting
        # them lets repeated false positives become authoritative over time.
        if source not in {'direct'}:
            return
        key = normalize_cache_key(matched_text, build_variables)
        if not key or not packages:
            return
        packages = sorted(set(packages))
        confidence = _source_confidence(source)

        with _lock:
            existing = self._cache.get(key)
            if existing:
                if existing.packages == packages:
                    existing.confidence += 1
                    self._dirty = True
                elif confidence >= existing.confidence or source == 'direct':
                    self._cache[key] = CacheEntry(packages, confidence, source)
                    self._dirty = True
            else:
                self._cache[key] = CacheEntry(packages, confidence, source)
                self._dirty = True

    def prefill_from_direct_matches(self, file_matches: dict, valid_packages: set[str],
                                    build_variables: dict[str, str] | None = None) -> None:
        """Pre-fill cache from scanner's __direct__ matches."""
        for rel_path, matches in file_matches.items():
            if not rel_path.startswith('__direct__/'):
                continue
            for match in matches:
                packages = getattr(match, "packages", None) or []
                if packages:
                    valid = [p for p in packages if p in valid_packages]
                    if valid:
                        self.update(match.matched_text, valid, source='direct',
                                    build_variables=build_variables)

    def validate(self, valid_packages: set[str]) -> int:
        """Remove entries with no valid packages. Returns count removed."""
        removed = 0
        to_delete = []
        for key, entry in self._cache.items():
            valid = [p for p in entry.packages if p in valid_packages]
            if not valid:
                to_delete.append(key)
                removed += 1
            elif valid != entry.packages:
                entry.packages = valid
                self._dirty = True
        for key in to_delete:
            del self._cache[key]
            self._dirty = True
        return removed

    def load(self) -> None:
        """Load cache from disk."""
        path = artifact_cache_path(self.distro)
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get('schema_version') != 2 or data.get('repository_profile_hash') != self.profile_hash:
                return
            for key, entry in (data.get('entries') or {}).items():
                if not isinstance(entry, dict):
                    continue
                self._cache[key] = CacheEntry(
                    packages=entry.get('packages', []),
                    confidence=entry.get('confidence', 1),
                    source=entry.get('source', 'unknown'),
                )
        except Exception:
            pass

    def save(self) -> None:
        """Save cache to disk."""
        if not self._dirty:
            return
        path = artifact_cache_path(self.distro)
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = {}
        for key, entry in sorted(self._cache.items()):
            entries[key] = {
                'packages': entry.packages,
                'confidence': entry.confidence,
                'source': entry.source,
            }
        data = {
            'schema_version': 2,
            'repository_profile_hash': self.profile_hash,
            'distro': self.distro,
            'entries': entries,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._dirty = False

    def stats(self) -> dict:
        """Return cache statistics."""
        total = len(self._cache)
        by_source = {}
        for entry in self._cache.values():
            by_source[entry.source] = by_source.get(entry.source, 0) + 1
        usable = sum(1 for e in self._cache.values() if e.confidence >= _USE_THRESHOLD)
        return {'total': total, 'usable': usable, 'by_source': by_source}


def _source_confidence(source: str) -> int:
    if source == 'direct':
        return _CONFIDENCE_DIRECT
    return _CONFIDENCE_LLM
