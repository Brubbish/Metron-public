"""Canonical package-name collapse via regex.

Replaces the hardcoded `PROVIDER_ALIAS_FAMILIES` table in `eval.py` and parts of
`_PROVIDER_FAMILIES`/`_PROVIDER_HINTS` in `main.py`. Recognises common packaging
variations (ABI numbers, time64 transition, SSL flavours, arch prefixes) and collapses
each variant family to a single canonical representative.
"""
from __future__ import annotations

import re
from collections.abc import Iterable


_ARCH_PREFIX_RE = re.compile(r'^lib(?:32|64|x32)(?=[a-z])')
# Note: we deliberately do NOT collapse SSL flavour suffixes (-openssl/-gnutls/-nss)
# because Debian projects legitimately depend on either variant (their build-deps picks
# one specifically). Collapsing them would lose TPs when GT names the non-default flavour.
_FLAVOR_RE = re.compile(r'-(?:openssl|gnutls|nss|mbedtls|wolfssl|gcrypt|nettle)(?=-|$)')


def _stem(pkg: str) -> str:
    """Compute a canonical stem for a package name.

    Strips arch prefixes and trailing version numbers. Short alpha cores (<4 chars
    after `lib`) are protected so names like `liblz4-dev` are not collapsed to
    `liblz-dev`. SSL flavour suffixes are deliberately preserved — Debian projects
    legitimately pick either -openssl or -gnutls variants and collapsing would lose
    TPs.
    """
    s = _ARCH_PREFIX_RE.sub('lib', pkg)
    parts = s.split('-')
    new_parts: list[str] = []
    for i, part in enumerate(parts):
        # Pure version component (possibly with t64): drop
        if i > 0 and re.fullmatch(r'\d+(?:\.\d+)*(?:t64)?', part):
            continue
        # Embedded ABI in a name segment: libfoo17t64 → libfoo
        # Protect short alpha cores: lz4, bz2, ml4 should keep their digit suffix.
        m = re.fullmatch(r'(lib[a-z]{3,})(\d{1,3}(?:t64)?)(.*)', part)
        if m and not m.group(3):
            new_parts.append(m.group(1))
        else:
            new_parts.append(part)
    return '-'.join(p for p in new_parts if p)


def _rank(pkg: str) -> tuple:
    """Preference key for picking the canonical member of a stem family (lower = better)."""
    has_dev = '-dev' in pkg
    has_version = bool(re.search(r'-\d|t64', pkg))
    no_arch = not pkg.startswith(('lib32', 'lib64', 'libx32'))
    return (
        0 if has_dev else 1,
        0 if has_dev and not has_version else 1,
        0 if no_arch else 1,
        len(pkg),
        pkg,
    )


def canonicalize(pkgs: Iterable[str]) -> list[str]:
    """Collapse package variants to one canonical representative per stem family.

    Examples:
      {'libc++-17-dev', 'libc++-18-dev', 'libc++-dev'} → {'libc++-dev'}
      {'libcurl4-openssl-dev', 'libcurl4-gnutls-dev'} → {'libcurl4-gnutls-dev'}
      {'lib32ncurses-dev', 'libncurses-dev'} → {'libncurses-dev'}
      {'libssl3t64'} → {'libssl3t64'}  (no -dev sibling to collapse to)
    """
    groups: dict[str, list[str]] = {}
    for p in pkgs:
        groups.setdefault(_stem(p), []).append(p)
    out: list[str] = []
    for members in groups.values():
        members.sort(key=_rank)
        out.append(members[0])
    return sorted(set(out))


def dropped_candidates(pkgs: Iterable[str]) -> set[str]:
    """Return the set of packages that canonicalize() would drop."""
    pkg_set = set(pkgs)
    return pkg_set - set(canonicalize(pkg_set))


def canonicalize_per_file(per_file: dict) -> None:
    """In-place: apply canonicalize to the union of packages across all buckets.

    For each bucket dep entry, remove any package that is dropped by canonicalize.
    Empty buckets / dep entries are pruned.
    """
    all_pkgs: set[str] = set()
    for deps in per_file.values():
        if not isinstance(deps, list):
            continue
        for d in deps:
            if isinstance(d, dict):
                all_pkgs.update(d.get('suite_packages', []))
                all_pkgs.update(d.get('packages', []))
    dropped = dropped_candidates(all_pkgs)
    if not dropped:
        return
    for bucket, deps in list(per_file.items()):
        if not isinstance(deps, list):
            continue
        for d in deps:
            if not isinstance(d, dict):
                continue
            new_t = [p for p in d.get('suite_packages', []) if p not in dropped]
            d['suite_packages'] = new_t
            if 'packages' in d:
                d['packages'] = [p for p in d.get('packages', []) if p not in dropped]
        per_file[bucket] = [d for d in deps if not isinstance(d, dict) or d.get('suite_packages')]
        if not per_file[bucket]:
            del per_file[bucket]
