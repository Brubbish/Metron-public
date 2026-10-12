"""Deterministic conflict filtering helpers.

This module intentionally stays small and standalone so it can be enabled,
disabled, or swapped out independently during ablation experiments.
"""

from __future__ import annotations


SOURCE_STRENGTH = {
    'direct_rule': 4,
    'match_artifact': 4,
    'header_anchor': 4,
    'file_artifact_backfill': 3,
    'build_driver': 3,
    'llm': 2,
    'same_source_expansion': 1,
    'cache': 1,
}


def source_strength(sources: list[str] | set[str]) -> int:
    """Return the strongest evidence level represented by a source set."""
    return max((SOURCE_STRENGTH.get(source, 0) for source in sources), default=0)


def filter_against_header_conflicts(
    header_packages: list[str] | set[str],
    llm_candidates: list[str],
    conflicts_by_package: dict[str, list[str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """Phase 1 direct-conflict filter using header-derived anchor packages.

    Policy:
    - drop duplicates already fixed by header evidence
    - drop if candidate conflicts with any header package
    - drop if any header package explicitly conflicts with candidate
    - ignore self-conflicts on either side
    - do not inspect dependency closure or Provides groups
    """
    header = set(header_packages)
    kept: list[str] = []
    dropped: list[dict[str, str]] = []

    for candidate in llm_candidates:
        if candidate in header:
            dropped.append(
                {
                    'candidate': candidate,
                    'reason': 'already_resolved',
                    'matched_header_pkg': candidate,
                    'evidence_side': 'header',
                }
            )
            continue

        candidate_hit = None
        for target in conflicts_by_package.get(candidate, ()):
            if target == candidate:
                continue
            if target in header:
                candidate_hit = target
                break
        if candidate_hit is not None:
            dropped.append(
                {
                    'candidate': candidate,
                    'reason': 'direct_conflict',
                    'matched_header_pkg': candidate_hit,
                    'evidence_side': 'candidate',
                }
            )
            continue

        reverse_hit = None
        for header_pkg in header:
            if candidate in (
                target
                for target in conflicts_by_package.get(header_pkg, ())
                if target != header_pkg
            ):
                reverse_hit = header_pkg
                break
        if reverse_hit is not None:
            dropped.append(
                {
                    'candidate': candidate,
                    'reason': 'direct_conflict',
                    'matched_header_pkg': reverse_hit,
                    'evidence_side': 'header',
                }
            )
            continue

        kept.append(candidate)

    return kept, dropped


def filter_project_conflicts(
    package_sources: dict[str, set[str]],
    conflicts_by_package: dict[str, list[str]],
) -> tuple[set[str], list[dict[str, object]]]:
    """Choose weak packages to drop from project-wide direct conflicts.

    Packages with equally strong evidence are kept and logged as ambiguous. The
    filter only considers explicit package-to-package Conflicts metadata.
    """
    dropped: set[str] = set()
    log: list[dict[str, object]] = []
    packages = sorted(package_sources)

    for package in packages:
        if package in dropped:
            continue
        for other in conflicts_by_package.get(package, ()):
            if other == package or other not in package_sources or other in dropped:
                continue

            package_strength = source_strength(package_sources[package])
            other_strength = source_strength(package_sources[other])
            if package_strength == other_strength:
                log.append(
                    {
                        'reason': 'ambiguous_conflict',
                        'package': package,
                        'conflicts_with': other,
                        'package_sources': sorted(package_sources[package]),
                        'conflicts_with_sources': sorted(package_sources[other]),
                    }
                )
                continue

            remove, keep = (
                (package, other)
                if package_strength < other_strength
                else (other, package)
            )
            dropped.add(remove)
            log.append(
                {
                    'reason': 'weaker_conflict_evidence',
                    'dropped': remove,
                    'kept': keep,
                    'dropped_sources': sorted(package_sources[remove]),
                    'kept_sources': sorted(package_sources[keep]),
                }
            )
            if remove == package:
                break

    return dropped, log


def filter_conflicting_candidates(
    candidates: list[str],
    resolved_packages: list[str] | set[str],
    conflicts_by_package: dict[str, list[str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """Drop candidates whose own Conflicts entries hit an already-resolved package.

    The policy is intentionally conservative:
    - only the candidate package's own `Conflicts` list is consulted
    - self-conflicts are ignored
    - no reverse inference is attempted
    - `Provides` groups are not considered here
    """
    resolved = set(resolved_packages)
    kept: list[str] = []
    dropped: list[dict[str, str]] = []

    for candidate in candidates:
        hit = None
        for target in conflicts_by_package.get(candidate, ()):
            if target == candidate:
                continue
            if target in resolved:
                hit = target
                break
        if hit is None:
            kept.append(candidate)
        else:
            dropped.append(
                {
                    'candidate': candidate,
                    'conflicts_with': hit,
                }
            )
    return kept, dropped
