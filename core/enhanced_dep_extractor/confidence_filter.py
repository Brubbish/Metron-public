"""Confidence-based FP filter for the enhanced dependency extractor.

Seven combined mechanisms (M1 + M1b + M1c + M-A + M-D + G1 + G4):

- M1: backfill bucket — drop pkgs whose EVERY library token has prior < thresh.
- M1c: backfill bucket — drop pkgs whose BEST library token has prior < backfill_min_prior
       (only applies when the pkg has at least one library token).
- M1b: per-file bucket — drop pkgs whose EVERY library token is BOTH low-prior AND not
       mentioned as a substring of the dep's `software_name` text.
- M-A: cardinality gate — drop backfill pkgs that have HIGH library-token cardinality
       (many .so files ship in one package, making it an ambiguous prediction target)
       UNLESS the project's evidence text mentions the package's stem.
- M-D: project-coherence — drop backfill pkgs whose stem is absent from the project's
       evidence text (per_file software_names + project source files). Generalizes M1b
       from per-dep to project-wide scope.
- G1: Essential/Priority gate — drop pkgs where Debian Essential:yes or Priority:required
       (these are pre-installed on every Debian system and never listed as build-deps).
       Read from Debian Packages metadata, not a hardcoded list.
- G4: token collision detector — drop backfill pkgs whose library token is a short
       pure-alpha string that equals the package name (e.g. token 'and' → pkg 'and',
       token 'file' → pkg 'file'). Detects English-word/package-name collisions
       structurally, replacing the former _SHELL_BUILTINS/_STOPWORDS blocklists.

Library token priors are data-driven (computed by `tools/build_token_priors.py` and stored
in `token_priors/{version}_token_priors.json`). If the priors file is missing, M1/M1b/M1c
silently no-op; M-A, M-D, G1, G4 still run using provider_index + Debian Packages metadata.

No hardcoded package lists, family tables, or tool-name allowlists. All decisions are
based on structural signals (token cardinality, stem presence, Debian metadata, token
linguistic properties).
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Iterable

from .metadata import DEFAULT_DEBIAN_VERSION, token_priors_path, provider_index_path, pkginfo_paths

_LOCK = threading.Lock()
# Version-keyed caches: each Debian version gets its own cached data.
# This is thread-safe for concurrent projects targeting different distros.
_PRIOR_CACHE: dict[str, dict[str, list]] = {}       # {version: {token: [prec, count]}}
_PKG2TOK_CACHE: dict[str, dict[str, list[str]]] = {} # {version: {pkg: [tokens]}}
_CARDINALITY_CACHE: dict[str, dict[str, int]] = {}   # {version: {pkg: card}}
_PKG_META_CACHE: dict[str, dict[str, dict]] = {}     # {version: {pkg: {essential, priority}}}

# File extensions whose contents we scan for project-coherence stem matches.
# Structural (file types), not a package list.
_SOURCE_EXTS = frozenset({'.c', '.h', '.cpp', '.cc', '.hpp', '.cxx', '.cmake', '.am',
                          '.ac', '.pro', '.py', '.build', '.mk', '.m4'})
_CONFIG_FILES = frozenset({'cmakelists.txt', 'meson.build', 'makefile', 'makefile.am',
                           'gnumakefile', 'configure.ac', 'configure.in', 'sconstruct'})
_DOC_EXTS = frozenset({'.md', '.rst', '.txt', '.yml', '.yaml', '.cfg', '.ini'})

# Regex for extracting the family-name stem of a Debian package.
_LIB_PREFIX_RE = re.compile(r'^lib')
_DEV_SUFFIX_RE = re.compile(r'-(?:dev|tools|tool|bin|utils|doc|dbg|common|data)$')
_VERSION_TAIL_RE = re.compile(r'\d+(?:\.\d+)*(?:t64)?$')

# Skip directories that never carry build-dep evidence.
_SKIP_DIRS = frozenset({'.git', '__pycache__', 'node_modules', 'debian', 'build',
                        'builddir', 'third_party', '3rdparty', 'vendor', 'external',
                        'subprojects', 'out'})

# High-cardinality threshold: packages with ≥ this many library tokens are considered
# intrinsically ambiguous (any short English word could resolve to them via .so basename).
_HIGH_CARD_THRESHOLD = 4

# Minimum stem length to require project-mention check.
_STEM_MIN_LEN = 4


def _pkg_stem(pkg: str) -> str:
    """Extract the family-name stem of a Debian package, lowercased.

    Examples: libcamitk-dev → 'camitk', libvtk9-dev → 'vtk',
              qtbase5-dev → 'qtbase', libssl-dev → 'ssl', zlib1g-dev → 'zlib'
    """
    s = pkg.lower()
    s = _LIB_PREFIX_RE.sub('', s, count=1)
    s = _DEV_SUFFIX_RE.sub('', s)
    s = _VERSION_TAIL_RE.sub('', s)
    return s.strip('-')


def _load(version: str | None = None) -> tuple[dict[str, list], dict[str, list[str]]]:
    """Load priors and reverse pkg→tokens map for the active Debian version."""
    version = version or DEFAULT_DEBIAN_VERSION
    with _LOCK:
        if version in _PRIOR_CACHE:
            return _PRIOR_CACHE[version], _PKG2TOK_CACHE[version]
        path = token_priors_path(version)
        priors: dict[str, list] = {}
        pkg2tok: dict[str, list[str]] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
                priors = data.get('priors', {})
                pkg2tok = data.get('pkg2tokens', {})
            except Exception:
                pass
        _PRIOR_CACHE[version] = priors
        _PKG2TOK_CACHE[version] = pkg2tok
    return priors, pkg2tok


def _load_cardinality(version: str | None = None) -> dict[str, int]:
    """Build {pkg: number_of_library_tokens} from provider_index.libraries."""
    version = version or DEFAULT_DEBIAN_VERSION
    with _LOCK:
        if version in _CARDINALITY_CACHE:
            return _CARDINALITY_CACHE[version]
        path = provider_index_path(version)
        card: dict[str, int] = {}
        try:
            with open(path) as f:
                idx = json.load(f)
            for tok, pkgs in idx.get('libraries', {}).items():
                if not pkgs:
                    continue
                for p in pkgs:
                    card[p] = card.get(p, 0) + 1
        except Exception:
            pass
        _CARDINALITY_CACHE[version] = card
    return card


def reset_cache() -> None:
    """Clear all cached version data. Safe to call from any thread."""
    with _LOCK:
        _PRIOR_CACHE.clear()
        _PKG2TOK_CACHE.clear()
        _CARDINALITY_CACHE.clear()
        _PKG_META_CACHE.clear()


# ─── G1: Debian Essential/Priority metadata ──────────────────────────

# _PKG_META_CACHE is version-keyed (defined above with _CARDINALITY_CACHE etc.)


def _load_pkg_meta(version: str | None = None) -> dict[str, dict]:
    """Parse Debian Packages files → {pkg: {essential, priority}}."""
    version = version or DEFAULT_DEBIAN_VERSION
    with _LOCK:
        if version in _PKG_META_CACHE:
            return _PKG_META_CACHE[version]
        meta: dict[str, dict] = {}
        for path in pkginfo_paths(version):
            cur: str | None = None
            try:
                with open(path) as f:
                    for line in f:
                        if line.startswith('Package: '):
                            cur = line.split(':', 1)[1].strip()
                            if cur not in meta:
                                meta[cur] = {'essential': False, 'priority': ''}
                        elif cur is None:
                            continue
                        elif line.startswith('Essential: '):
                            if 'yes' in line.split(':', 1)[1].strip().lower():
                                meta[cur]['essential'] = True
                        elif line.startswith('Priority: '):
                            meta[cur]['priority'] = line.split(':', 1)[1].strip()
            except Exception:
                pass
        _PKG_META_CACHE[version] = meta
    return meta


def _g1_is_pre_installed(pkg: str, meta: dict[str, dict]) -> bool:
    """G1: True if Debian guarantees this package is pre-installed on every system."""
    m = meta.get(pkg)
    if not m:
        return False
    if m.get('essential'):
        return True
    if m.get('priority') in ('required', 'important'):
        return True
    return False


# ─── G4: Token collision detector ────────────────────────────────────

def _g4_is_collision_token(pkg: str, pkg2tok: dict[str, list[str]]) -> bool:
    """G4: detect if pkg's library token is a short pure-alpha word == pkg name.

    This is a structural test: short (≤5 chars), purely alphabetic tokens that
    equal the package name are almost certainly English-word/package-name
    collisions (e.g. token 'and' → pkg 'and', token 'file' → pkg 'file').
    Real library tokens have separators (-, +, .) or digits (libssl, protobuf-c, boost_filesystem).
    """
    tokens = pkg2tok.get(pkg, [])
    if not tokens:
        # No library token — check if pkg name itself is short+alpha
        if len(pkg) <= 5 and pkg.isalpha():
            return True
        return False
    for tok in tokens:
        tok_l = tok.lower()
        pkg_l = pkg.lower()
        if tok_l == pkg_l and len(tok_l) <= 5 and tok.isalpha():
            return True
        if len(tok_l) <= 3 and tok.isalpha():
            return True
    return False


def _sw_mentions(sw: str, tokens: Iterable[str], min_token_len: int = 3) -> bool:
    sw_l = sw.lower()
    return any(len(t) >= min_token_len and t.lower() in sw_l for t in tokens)


def _build_project_blob(per_file: dict, project_path: str | None = None) -> str:
    """Lowercased text blob: per_file software_names + file paths + project source files.

    The blob is used to check whether a predicted package's stem appears anywhere in
    the project's evidence (not just in the single dep entry that produced it).
    """
    parts: list[str] = []
    for bucket, deps in per_file.items():
        if bucket.startswith('__artifact_backfill__') or bucket.startswith('__same_source'):
            continue
        if not isinstance(deps, list):
            continue
        parts.append(bucket)  # file path
        for d in deps:
            if isinstance(d, dict):
                sw = d.get('software_name') or ''
                if sw:
                    parts.append(sw)
    blob = ' '.join(parts).lower()

    if project_path:
        try:
            for dirpath, dirnames, filenames in os.walk(project_path):
                dirnames[:] = [d for d in dirnames
                               if not d.startswith('.')
                               and d.lower() not in _SKIP_DIRS]
                for fn in filenames:
                    fn_l = fn.lower()
                    ext = os.path.splitext(fn_l)[1]
                    # Doc/config files: always scan
                    if (fn_l in _CONFIG_FILES
                            or fn_l.startswith(('readme', 'install', 'requirement', 'depend'))
                            or ext in _DOC_EXTS):
                        pass
                    # Source/config files: scan if small
                    elif ext in _SOURCE_EXTS:
                        p = os.path.join(dirpath, fn)
                        try:
                            if os.path.getsize(p) > 30_000:
                                continue
                        except Exception:
                            continue
                    else:
                        continue
                    p = os.path.join(dirpath, fn)
                    try:
                        if os.path.getsize(p) > 100_000:
                            continue
                        with open(p, errors='replace') as f:
                            blob += ' ' + f.read().lower()
                    except Exception:
                        pass
        except Exception:
            pass
    return blob


def _stem_in_blob(stem: str, blob: str) -> bool:
    """Check if stem appears as a word-boundary match in the lowercased blob."""
    if len(stem) < _STEM_MIN_LEN:
        return True  # too short to be safe; assume present
    return bool(re.search(rf'\b{re.escape(stem)}\b', blob))


def apply_confidence_filter(
    per_file: dict,
    *,
    project_path: str | None = None,
    version: str | None = None,
    token_thresh: float = 0.10,
    backfill_min_prior: float = 0.20,
    min_token_count: int = 3,
) -> None:
    """In-place confidence filter on per_file buckets.

    Combines seven mechanisms (M1/M1b/M1c/M-A/M-D/G1/G4). Each is independently safe:
    - M1/M1b/M1c no-op if the priors file is missing.
    - M-A/M-D use provider_index cardinality + project source scan; no external data.
    - G1 uses Debian Packages metadata (Essential/Priority).
    - G4 uses provider_index token linguistic properties.

    Args:
        per_file: per-file dependency dict (mutated in place).
        project_path: path to the project root (enables M-D source scan).
        version: Debian version to load priors + provider_index + metadata for.
        token_thresh: prior threshold below which a library token is "bad" (M1 + M1b).
        backfill_min_prior: stricter prior threshold for backfill bucket's BEST token (M1c).
        min_token_count: only treat a token as bad if it has fired in at least this many
            projects (avoids treating genuinely rare-but-valid tokens as noise).
    """
    priors, pkg2tok = _load(version)
    card = _load_cardinality(version)
    meta = _load_pkg_meta(version)

    # ── Compute bad tokens (for M1/M1b/M1c) ──
    bad_tokens = set()
    if priors:
        bad_tokens = {
            tok for tok, (prec, n) in priors.items()
            if n >= min_token_count and prec < token_thresh
        }

    # ── Build project evidence blob (for M-A + M-D) ──
    blob = _build_project_blob(per_file, project_path=project_path)

    # ── M1 + M1c + M-A + M-D: __artifact_backfill__ ─────────────────────
    bf = per_file.get('__artifact_backfill__')
    if isinstance(bf, list):
        new_bf = []
        for d in bf:
            if not isinstance(d, dict):
                new_bf.append(d)
                continue
            pkgs = d.get('suite_packages') or d.get('packages', [])
            drop = False
            for p in pkgs:
                toks = pkg2tok.get(p, [])
                # M1: drop if ALL library tokens are bad
                if toks and all(t in bad_tokens for t in toks):
                    drop = True
                    break
                # M1c: drop if has token AND best prior < backfill_min_prior
                if toks and priors:
                    best = max((priors.get(t, (0.0, 0))[0] for t in toks), default=0.0)
                    if best < backfill_min_prior:
                        drop = True
                        break
                # M-A: drop if high-cardinality pkg AND stem not in project blob
                if card.get(p, 0) >= _HIGH_CARD_THRESHOLD:
                    stem = _pkg_stem(p)
                    if not _stem_in_blob(stem, blob):
                        drop = True
                        break
            if drop:
                continue
            # M-D + G4: drop if pkg stem is absent from project evidence entirely,
            # OR if token is a short-alpha collision word.
            kept = []
            for p in pkgs:
                stem = _pkg_stem(p)
                if len(stem) >= _STEM_MIN_LEN and not _stem_in_blob(stem, blob):
                    continue  # M-D: stem not in project evidence
                if _g4_is_collision_token(p, pkg2tok):
                    continue  # G4: English-word/package-name collision
                kept.append(p)
            if kept:
                nd = dict(d)
                nd['suite_packages'] = kept
                if 'packages' in nd:
                    nd['packages'] = [p for p in nd.get('packages', []) if p in kept]
                new_bf.append(nd)
        if new_bf:
            per_file['__artifact_backfill__'] = new_bf
        else:
            per_file.pop('__artifact_backfill__', None)

    # ── M1b: per-file buckets ───────────────────────────────────────────
    for bucket, deps in list(per_file.items()):
        if not isinstance(deps, list) or bucket.startswith('__'):
            continue
        new_deps = []
        for d in deps:
            if not isinstance(d, dict):
                new_deps.append(d)
                continue
            pkgs_field = d.get('suite_packages')
            if pkgs_field is None:
                pkgs_field = d.get('packages', [])
            sw = d.get('software_name') or ''
            kept = []
            for pkg in pkgs_field:
                toks = [t for t in pkg2tok.get(pkg, []) if len(t) >= 3]
                if not toks:
                    kept.append(pkg)
                    continue
                if all(t in bad_tokens for t in toks) and not _sw_mentions(sw, toks):
                    continue
                kept.append(pkg)
            if kept:
                nd = dict(d)
                if 'suite_packages' in nd:
                    nd['suite_packages'] = kept
                if 'packages' in nd:
                    nd['packages'] = [p for p in nd.get('packages', []) if p in kept]
                new_deps.append(nd)
        if new_deps:
            per_file[bucket] = new_deps
        else:
            per_file.pop(bucket, None)

    # ── G1: drop Essential/required/important packages from ALL buckets ──
    # These packages are pre-installed on every Debian system and never appear
    # as Build-Depends. Uses Debian metadata, not a hardcoded list.
    if meta:
        for bucket, deps in list(per_file.items()):
            if not isinstance(deps, list):
                continue
            new_deps = []
            for d in deps:
                if not isinstance(d, dict):
                    new_deps.append(d)
                    continue
                pkgs_field = d.get('suite_packages')
                if pkgs_field is None:
                    pkgs_field = d.get('packages', [])
                kept = [p for p in pkgs_field if not _g1_is_pre_installed(p, meta)]
                if kept:
                    nd = dict(d)
                    if 'suite_packages' in nd:
                        nd['suite_packages'] = kept
                    if 'packages' in nd:
                        nd['packages'] = [p for p in nd.get('packages', []) if p in kept]
                    new_deps.append(nd)
            if new_deps:
                per_file[bucket] = new_deps
            else:
                per_file.pop(bucket, None)
