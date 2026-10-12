"""Conservative source header scan for header-only library backfill.

Scans #include <...> in source files and resolves via Provider Index.
Does NOT send anything to LLM. Outputs to __header_artifact_backfill__.
"""

import os
import re
from dataclasses import dataclass

from .default_headers import header_path_suffixes, load_default_header_filter
from .scanner import _should_skip_dir, THIRD_PARTY_PREFIXES
from .artifacts import (
    _header_index_suffix_packages,
    _header_stem,
    _provider_index,
    _provider_index_allowed,
    _STANDARD_HEADERS,
    _HEADER_ONLY_STEMS,
)
from .provider_normalization import boost_header_packages

_SOURCE_EXTS = frozenset({
    '.c', '.cc', '.cpp', '.cxx', '.m', '.mm',
    '.h', '.hh', '.hpp', '.hxx', '.ipp', '.tpp',
    '.pyx', '.pxd', '.pxi',
})

_MAX_FILE_SIZE = 1 * 1024 * 1024

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*<([^>]+)>', re.MULTILINE)

_BUILD_DIR_NAMES = frozenset({
    'build', '_build', 'builddir', 'cmake-build-debug', 'cmake-build-release',
    'cmake-build-relwithdebinfo', 'cmake-build-minsizerel',
})

_CXX_STD_HEADERS = frozenset({
    'algorithm', 'any', 'array', 'atomic', 'barrier', 'bit', 'bitset',
    'cassert', 'cctype', 'cerrno', 'cfenv', 'cfloat', 'charconv',
    'chrono', 'cinttypes', 'climits', 'clocale', 'cmath', 'codecvt',
    'compare', 'complex', 'concepts', 'condition_variable', 'coroutine',
    'csetjmp', 'csignal', 'cstdarg', 'cstddef', 'cstdint', 'cstdio',
    'cstdlib', 'cstring', 'ctime', 'cuchar', 'cwchar', 'cwctype',
    'deque', 'exception', 'execution', 'expected', 'filesystem',
    'format', 'forward_list', 'fstream', 'functional', 'future',
    'initializer_list', 'iomanip', 'ios', 'iosfwd', 'iostream',
    'istream', 'iterator', 'latch', 'limits', 'list', 'locale',
    'map', 'memory', 'memory_resource', 'mutex', 'new', 'numbers',
    'numeric', 'optional', 'ostream', 'print', 'queue', 'random',
    'ranges', 'ratio', 'regex', 'scoped_allocator', 'semaphore',
    'set', 'shared_mutex', 'source_location', 'span', 'spanstream',
    'sstream', 'stack', 'stacktrace', 'stdexcept', 'stop_token',
    'streambuf', 'string', 'string_view', 'strstream', 'syncstream',
    'system_error', 'thread', 'tuple', 'type_traits', 'typeindex',
    'typeinfo', 'unordered_map', 'unordered_set', 'utility',
    'valarray', 'variant', 'vector', 'version',
})

_SKIP_PREFIXES = ('linux/', 'asm/', 'sys/', 'net/', 'netinet/', 'arpa/', 'bits/', 'mach/')
_MAX_BASENAME_PROVIDERS = 6
_MAX_STABLE_HEADER_PROVIDERS = 10


@dataclass(frozen=True)
class HeaderCandidate:
    header: str
    package: str
    match_kind: str


def _should_skip_header_dir(name: str) -> bool:
    if _should_skip_dir(name):
        return True
    return name.lower() in _BUILD_DIR_NAMES or name.lower().startswith('cmake-build')


def _is_skip_token(token: str) -> bool:
    if not token:
        return True
    low = token.lower()
    if low in _CXX_STD_HEADERS:
        return True
    if low in _STANDARD_HEADERS:
        return True
    if any(low.startswith(p) for p in _SKIP_PREFIXES):
        return True
    return False


_HEADER_EXTS = frozenset({'.h', '.hpp', '.hxx'})


def _collect_bundled_headers(project_root: str) -> set[str]:
    """Collect include-style paths for headers found in bundled/vendored dirs."""
    bundled: set[str] = set()
    try:
        entries = os.listdir(project_root)
    except OSError:
        return bundled
    for entry in entries:
        if entry.lower() not in THIRD_PARTY_PREFIXES:
            continue
        vendor_dir = os.path.join(project_root, entry)
        if not os.path.isdir(vendor_dir):
            continue
        for dirpath, _dirnames, filenames in os.walk(vendor_dir):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in _HEADER_EXTS:
                    continue
                # Compute relative path from inside the vendor dir
                rel = os.path.relpath(os.path.join(dirpath, fname), vendor_dir)
                # Normalize to forward slashes
                rel = rel.replace(os.sep, '/')
                # Add the full relative path and all suffixes
                # e.g. "zlib/include/zlib.h" -> also "include/zlib.h", "zlib.h"
                parts = rel.split('/')
                for i in range(len(parts)):
                    bundled.add('/'.join(parts[i:]))
    return bundled


def scan_source_headers(project_root: str, valid_packages: set[str]) -> dict[str, set[str]]:
    """Scan source #include <...> and resolve via provider index.

    Returns {header_token: {resolved_packages}}.
    Only returns high-confidence, unique-provider results.
    """
    index = _provider_index()
    basename_index = _build_header_basename_index(index, valid_packages)
    default_header_filter = load_default_header_filter()
    bundled_headers = _collect_bundled_headers(project_root)
    seen_tokens: set[str] = set()
    candidates: list[HeaderCandidate] = []

    for dirpath, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [d for d in dirnames if not _should_skip_header_dir(d)]

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _SOURCE_EXTS:
                continue

            filepath = os.path.join(dirpath, fname)
            try:
                size = os.path.getsize(filepath)
            except OSError:
                continue
            if size > _MAX_FILE_SIZE or size == 0:
                continue

            try:
                content = open(filepath, errors='replace').read()
            except Exception:
                continue

            for token in _INCLUDE_RE.findall(content):
                token = token.strip()
                if token in seen_tokens:
                    continue
                seen_tokens.add(token)

                if _is_skip_token(token):
                    continue

                if token in bundled_headers:
                    continue

                candidates.extend(_header_candidates_for_token(
                    token,
                    index,
                    basename_index,
                    valid_packages,
                    default_header_filter,
                ))

    return _select_header_candidate_packages(candidates)


def _build_header_basename_index(index, valid_packages: set[str]) -> dict[str, set[str]]:
    basename_index: dict[str, set[str]] = {}
    for header, packages in index.headers_exact.items():
        basename = header.rsplit('/', 1)[-1]
        allowed = {
            pkg for pkg in packages
            if _provider_index_allowed(pkg, valid_packages)
        }
        if allowed:
            basename_index.setdefault(basename, set()).update(allowed)
    return basename_index


def _header_candidates_for_token(
    token: str,
    index,
    basename_index: dict[str, set[str]],
    valid_packages: set[str],
    default_header_filter: tuple[frozenset[str], frozenset[str]] | None = None,
) -> list[HeaderCandidate]:
    low = token.lower()
    basename = low.rsplit('/', 1)[-1]
    candidates: list[HeaderCandidate] = []
    default_suffixes, default_basenames = default_header_filter or (frozenset(), frozenset())
    if _matches_default_header(low, default_suffixes):
        return []

    for pkg in boost_header_packages(low, valid_packages):
        candidates.append(HeaderCandidate(low, pkg, 'boost'))
    if candidates:
        return candidates

    exact_pkgs = _bounded_header_packages(
        index.headers_exact.get(low, set()),
        valid_packages,
        _MAX_STABLE_HEADER_PROVIDERS,
    )
    suffix_pkgs = _header_index_suffix_packages(low, index, valid_packages)
    if exact_pkgs is None:
        return []

    for pkg in exact_pkgs:
        candidates.append(HeaderCandidate(low, pkg, 'exact'))

    for pkg in suffix_pkgs:
        candidates.append(HeaderCandidate(low, pkg, 'suffix'))

    stem = _header_stem(low)
    pkg = _HEADER_ONLY_STEMS.get(stem)
    if pkg and pkg in valid_packages:
        candidates.append(HeaderCandidate(low, pkg, 'header_only_stem'))

    if candidates:
        return list(dict.fromkeys(candidates))

    if basename in default_basenames:
        return []

    basename_pkgs = _bounded_header_packages(
        basename_index.get(basename, set()),
        valid_packages,
        _MAX_BASENAME_PROVIDERS,
    )
    if basename_pkgs is None:
        return []
    for pkg in basename_pkgs:
        candidates.append(HeaderCandidate(low, pkg, 'basename'))

    return list(dict.fromkeys(candidates))


def _matches_default_header(header: str, default_suffixes: frozenset[str]) -> bool:
    return any(suffix in default_suffixes for suffix in header_path_suffixes(header))


def _allowed_header_packages(packages: set[str], valid_packages: set[str]) -> set[str]:
    return {
        pkg for pkg in packages
        if _provider_index_allowed(pkg, valid_packages)
    }


def _bounded_header_packages(
    packages: set[str],
    valid_packages: set[str],
    max_providers: int,
) -> set[str] | None:
    allowed = _allowed_header_packages(packages, valid_packages)
    if len(allowed) > max_providers:
        return None
    return allowed


def _select_header_candidate_packages(candidates: list[HeaderCandidate]) -> dict[str, set[str]]:
    by_header: dict[str, set[str]] = {}
    for candidate in candidates:
        by_header.setdefault(candidate.header, set()).add(candidate.package)

    selected: dict[str, set[str]] = {}
    unique_hits: dict[str, int] = {}
    total_hits: dict[str, int] = {}

    for header, packages in by_header.items():
        for package in packages:
            total_hits[package] = total_hits.get(package, 0) + 1
        if len(packages) == 1:
            package = next(iter(packages))
            selected[header] = {package}
            unique_hits[package] = unique_hits.get(package, 0) + 1

    for header, packages in by_header.items():
        if header in selected:
            continue
        ranked = sorted(
            ((unique_hits.get(package, 0), total_hits.get(package, 0), package) for package in packages),
            reverse=True,
        )
        if not ranked:
            continue
        best = ranked[0]
        if len(ranked) > 1 and ranked[1][:2] == best[:2]:
            continue
        selected[header] = {best[2]}

    return selected


def _resolve_header_token(
    token: str,
    index,
    valid_packages: set[str],
    default_header_filter: tuple[frozenset[str], frozenset[str]] | None = None,
) -> set[str]:
    """Resolve a header token to packages via provider index."""
    basename_index = _build_header_basename_index(index, valid_packages)
    candidates = _header_candidates_for_token(
        token,
        index,
        basename_index,
        valid_packages,
        default_header_filter,
    )
    selected = _select_header_candidate_packages(candidates)
    return set().union(*selected.values()) if selected else set()


def backfill_header_scan_packages(
    per_file: dict,
    project_root: str,
    valid_packages: set[str],
) -> None:
    """Add header-only library packages discovered from source includes."""
    existing = {
        pkg
        for deps in per_file.values()
        for dep in deps
        for pkg in dep.get('packages', dep.get('suite_packages', []))
    }

    header_results = scan_source_headers(project_root, valid_packages)

    additions: set[str] = set()
    for _token, pkgs in header_results.items():
        additions.update(pkgs)
    additions -= existing

    if not additions:
        return
    per_file['__header_artifact_backfill__'] = [
        {'software_name': pkg, 'packages': [pkg]}
        for pkg in sorted(additions)
    ]


def collect_header_anchor_packages(project_root: str, valid_packages: set[str]) -> set[str]:
    """Collect deterministic header-derived anchor packages for a project."""
    header_results = scan_source_headers(project_root, valid_packages)
    anchors: set[str] = set()
    for pkgs in header_results.values():
        anchors.update(pkgs)
    return anchors


def inject_header_anchor_packages(
    per_file: dict,
    header_packages: set[str],
) -> None:
    """Inject header anchor packages into result payload in a dedicated bucket."""
    if not header_packages:
        return
    existing = {
        pkg
        for deps in per_file.values()
        for dep in deps
        for pkg in dep.get('packages', dep.get('suite_packages', []))
    }
    additions = sorted(header_packages - existing)
    if not additions:
        return
    per_file['__header_anchor_packages__'] = [
        {'software_name': pkg, 'packages': [pkg], 'suite_packages': [pkg]}
        for pkg in additions
    ]
