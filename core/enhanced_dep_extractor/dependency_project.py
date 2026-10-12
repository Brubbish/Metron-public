"""Dependency project abstraction for TheAuto's enhanced dependency extractor.

A dependency project merges related dependency clues (headers, library names,
package names, pkg-config modules, CMake packages, version predicates) into a
single semantic entity.  This aligns the implementation with the paper's
description of "grouping different dependency clues into dependency projects"
(Section~\\ref{sec:pkb}).

Key design decisions:
- Clues retain their original kind, evidence text, file location, and the
  KB lookup path that resolved them (headers_exact, pkgconfig, libraries, etc.).
- Package references record whether they are primary or alternative providers.
- Unmatched clues (those that could not be mapped to any package) are retained
  as ``DependencyProject(unmatched=True, packages=[])`` so that later build
  diagnosis can reconsider them — satisfying the paper's claim that "unmatched
  clues are retained in the Build Plan".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from .metadata import pkglist_path


@dataclass
class Clue:
    """A single piece of dependency evidence from repository artifacts."""
    kind: str            # header | library | pkgconfig | cmake_package | executable | version_predicate | direct_rule | build_driver
    evidence: str        # "openssl/ssl.h" / "ssl" / "openssl >= 1.1" / package name
    file_path: str
    line_num: int
    source_type: str     # direct_rule | match_artifact | file_artifact_backfill | cache | build_driver
    artifact_path: str = ""  # headers_exact | headers_suffix | pkgconfig | libraries | cmake | executables | distro_mapping | fallback


@dataclass
class PackageRef:
    """A Debian package that provides one or more clues."""
    name: str
    suite: str
    role: str            # primary | alternative
    artifact_path: str = ""  # which KB lookup path found this package
    source_repo: str = ""
    component: str = ""
    channel: str = ""
    version: str = ""
    availability: str = "indexed"


@dataclass
class VersionPredicate:
    """A version constraint attached to a dependency project."""
    op: str              # >= | = | <= | <
    version: str
    source_clue_idx: int = -1  # index into DependencyProject.clues


@dataclass
class DependencyProject:
    """A semantic dependency entity grouping related clues and packages."""
    project_id: str
    clues: list[Clue] = field(default_factory=list)
    packages: list[PackageRef] = field(default_factory=list)
    version_predicates: list[VersionPredicate] = field(default_factory=list)
    unmatched: bool = False
    confidence: str = "medium"     # high | medium | low | unknown
    requiredness: str = "conditional"  # required | conditional | unknown
    condition: str = ""
    scope_paths: list[str] = field(default_factory=list)
    local_or_external: str = "external"

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "clues": [
                {
                    "kind": c.kind,
                    "evidence": c.evidence,
                    "file_path": c.file_path,
                    "line_num": c.line_num,
                    "source_type": c.source_type,
                    "artifact_path": c.artifact_path,
                }
                for c in self.clues
            ],
            "packages": [
                {
                    "name": p.name,
                    "suite": p.suite,
                    "role": p.role,
                    "artifact_path": p.artifact_path,
                    "source_repo": p.source_repo,
                    "component": p.component,
                    "channel": p.channel,
                    "version": p.version,
                    "availability": p.availability,
                }
                for p in self.packages
            ],
            "version_predicates": [
                {"op": v.op, "version": v.version, "source_clue_idx": v.source_clue_idx}
                for v in self.version_predicates
            ],
            "unmatched": self.unmatched,
            "confidence": self.confidence,
            "requiredness": self.requiredness,
            "condition": self.condition,
            "scope": {"evidence_paths": self.scope_paths},
            "local_or_external": self.local_or_external,
        }


# ─── Assembly: per_file → list[DependencyProject] ───────────────────────────

# Map per_file bucket / source_type to clue kind.
_BUCKET_KIND_MAP = {
    'debian/control': 'direct_rule',
    '__auto_build_tools__': 'build_driver',
    '__artifact_backfill__': 'file_artifact_backfill',
    '__same_source_dev_expansion__': 'same_source_expansion',
    '__header_anchor_packages__': 'header_anchor',
}

_SOURCE_KIND_MAP = {
    'direct_rule': 'direct_rule',
    'build_driver': 'build_driver',
    'match_artifact': 'cmake_package',
    'file_artifact_backfill': 'header',
    'cache': 'cmake_package',
    'same_source_expansion': 'library',
    'header_anchor': 'header',
    'llm': 'cmake_package',
}


def _infer_clue_kind(rel_path: str, source_types: list[str]) -> str:
    """Infer clue kind from bucket name and source type tags."""
    if rel_path in _BUCKET_KIND_MAP:
        kind = _BUCKET_KIND_MAP[rel_path]
        if kind == 'file_artifact_backfill':
            return 'header'  # backfill is mostly header/include based
        return kind
    for st in source_types:
        if st in _SOURCE_KIND_MAP:
            return _SOURCE_KIND_MAP[st]
    return 'cmake_package'


def _best_project_id(clues: list[Clue], fallback: str) -> str:
    """Pick the most descriptive evidence as project_id."""
    # Prefer header evidence (most specific), then cmake_package, then library
    priority = {'header': 0, 'cmake_package': 1, 'pkgconfig': 2,
                'library': 3, 'executable': 4, 'direct_rule': 5,
                'build_driver': 6, 'version_predicate': 7,
                'file_artifact_backfill': 8, 'same_source_expansion': 9,
                'header_anchor': 10}
    best = fallback
    best_score = 99
    for clue in clues:
        score = priority.get(clue.kind, 50)
        if score < best_score:
            best_score = score
            best = clue.evidence[:80]
    return best


def assemble_dependency_projects(
    per_file: dict[str, list[dict[str, Any]]],
    unmatched_clues: list[dict[str, Any]] | None = None,
    suite: str = "",
    version_constraints: list[dict[str, Any]] | None = None,
    local_names: set[str] | None = None,
) -> list[DependencyProject]:
    """Convert per_file dict + unmatched clues into list[DependencyProject].

    Args:
        per_file: the per_file dict from process_project (rel_path → list[dep]).
        unmatched_clues: serialized unmatched RuleMatch dicts from
            __unmatched_clues__ bucket (may be None).
        suite: current Debian suite (for PackageRef).
        version_constraints: list of version constraint dicts (from
            suite_compat), each with dep_name, op, version.

    Returns:
        Ordered list of DependencyProject.
    """
    # Step 1: collect clues grouped by package name
    package_to_clues: dict[str, list[Clue]] = {}

    for rel_path, deps in per_file.items():
        if rel_path.startswith('__') and rel_path not in _BUCKET_KIND_MAP:
            # Skip internal buckets like __filter_log__, __suite_resolution_log__
            continue
        if not isinstance(deps, list):
            continue
        for dep in deps:
            if not isinstance(dep, dict):
                continue
            packages = dep.get('suite_packages') or dep.get('packages') or []
            if not packages:
                continue
            source_types = dep.get('sources') or []
            evidence = str(dep.get('software_name') or '')[:200]
            clue_kind = _infer_clue_kind(rel_path, source_types)
            source_type = source_types[0] if source_types else 'unknown'
            clue = Clue(
                kind=clue_kind,
                evidence=evidence,
                file_path=rel_path,
                line_num=dep.get('line_num', 0),
                source_type=source_type,
            )
            for pkg in packages:
                package_to_clues.setdefault(pkg, []).append(clue)

    # Step 2: create one DependencyProject per package
    projects: list[DependencyProject] = []
    for pkg_name in sorted(package_to_clues.keys()):
        clues = package_to_clues[pkg_name]
        # Deduplicate clues by (kind, evidence, file_path)
        seen: set[tuple] = set()
        deduped: list[Clue] = []
        for c in clues:
            key = (c.kind, c.evidence, c.file_path)
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        project_id = _best_project_id(deduped, pkg_name)
        role = 'primary'  # all packages from per_file are primary (filtering already chose)
        is_local = any(_semantic_name(c.evidence) in (local_names or set()) for c in deduped)
        project = DependencyProject(
            project_id=project_id,
            clues=deduped,
            packages=[] if is_local else [_package_ref(pkg_name, suite, role)],
            confidence='high' if len(deduped) > 1 else 'medium',
            requiredness=(
                'required'
                if any(c.source_type in {'direct_rule', 'build_driver'} for c in deduped)
                else 'conditional'
            ),
            condition=(
                ''
                if any(c.source_type in {'direct_rule', 'build_driver'} for c in deduped)
                else 'selected_entry_or_failure_evidence'
            ),
            scope_paths=sorted({c.file_path for c in deduped if c.file_path and not c.file_path.startswith('__')}),
            local_or_external='local' if is_local else 'external',
        )
        projects.append(project)

    # Step 3: attach version predicates
    if version_constraints:
        _attach_version_predicates(projects, version_constraints)

    # Step 4: add unmatched clues as separate projects
    if unmatched_clues:
        for uc in unmatched_clues:
            if not isinstance(uc, dict):
                continue
            evidence = str(uc.get('matched_text') or uc.get('extracted_name') or '')[:200]
            rule_type = str(uc.get('rule_type') or '')
            kind = 'cmake_package'
            if 'header' in rule_type:
                kind = 'header'
            elif 'library' in rule_type or 'l_flag' in rule_type:
                kind = 'library'
            elif 'pkgconfig' in rule_type or 'pkg_check' in rule_type:
                kind = 'pkgconfig'
            elif 'program' in rule_type or 'executable' in rule_type:
                kind = 'executable'

            clue = Clue(
                kind=kind,
                evidence=evidence,
                file_path=str(uc.get('file_path') or ''),
                line_num=int(uc.get('line_num') or 0),
                source_type='unmatched',
            )
            artifact_names = uc.get('artifact_names') or []
            project = DependencyProject(
                project_id=str(uc.get('extracted_name') or evidence[:60]),
                clues=[clue],
                packages=[],
                unmatched=True,
                confidence='low',
            )
            # If there are artifact names, add them as additional clue evidence
            for an in artifact_names[:3]:
                project.clues.append(Clue(
                    kind=kind,
                    evidence=str(an),
                    file_path=str(uc.get('file_path') or ''),
                    line_num=int(uc.get('line_num') or 0),
                    source_type='unmatched_artifact',
                ))
            projects.append(project)

    return projects


def _semantic_name(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(value or '').lower())


@lru_cache(maxsize=16)
def _package_metadata(suite: str) -> dict[str, Any]:
    try:
        payload = json.loads(pkglist_path(suite).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _package_ref(name: str, suite: str, role: str) -> PackageRef:
    record = _package_metadata(suite).get(name) or {}
    versions = [item for item in record.get('version') or [] if isinstance(item, dict)]
    provider = versions[0] if versions else {}
    return PackageRef(
        name=name,
        suite=suite,
        role=role,
        source_repo=str(provider.get('source_repo') or ''),
        component=str(provider.get('component') or ''),
        channel=str(provider.get('channel') or ''),
        version=str(provider.get('version') or ''),
        availability='indexed' if record else 'unknown',
    )


def _attach_version_predicates(
    projects: list[DependencyProject],
    constraints: list[dict[str, Any]],
) -> None:
    """Attach version predicates to matching projects by package name.

    constraints: list of dicts with keys dep_name, op, version (from
    suite_compat VersionConstraint serialization).
    """
    # Build package → predicates index
    pkg_predicates: dict[str, list[VersionPredicate]] = {}
    for c in constraints:
        dep_name = str(c.get('dep_name') or '').lower()
        op = str(c.get('op') or '>=')
        version = str(c.get('version') or '')
        if not dep_name or not version:
            continue
        pkg_predicates.setdefault(dep_name, []).append(
            VersionPredicate(op=op, version=version)
        )

    for project in projects:
        for pkg in project.packages:
            preds = pkg_predicates.get(pkg.name.lower())
            if preds:
                project.version_predicates.extend(preds)
