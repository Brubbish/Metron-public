#!/usr/bin/env python3
"""Enhanced rule+LLM dependency extractor for C/C++ projects.

Usage:
    python3 -m enhanced_dep_extractor.main <input_txt> -o results.json [options]
"""

import argparse
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass

import openai

from .scanner import scan_project, ProjectScan
from .llm_mapper import map_file_deps, merge_dependencies
from .openrouter_compat import get_client_max_retries
from .rules import CONTEXT_LINES
from . import artifacts
from .artifacts import backfill_artifact_packages
from .artifact_cache import DependencyCache, normalize_cache_key
from .metadata import (
    DEFAULT_DEBIAN_VERSION,
    chromadb_collection,
    chromadb_path,
    normalize_debian_version,
    pkginfo_paths,
    pkglist_path,
    provider_index_path,
    metadata_profile_hash,
)
from .conflict_filter import filter_project_conflicts
from .distro_select import DistroDecision, fixed_decision, select_debian_version

DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1/'
DEFAULT_MODEL = 'google/gemma-4-31b-it'
EXTRACTOR_SCHEMA_VERSION = 2

_print_lock = threading.Lock()
_kb_lock = threading.Lock()

_pkglist_cache = None
_package_source_cache = None
_source_dev_packages_cache = None
_source_packages_cache = None
_chromadb_cache = None
_chromadb_collection_cache = None
_valid_package_cache = None
_artifact_cache: DependencyCache | None = None

DEBIAN_VERSION = DEFAULT_DEBIAN_VERSION
PKGLIST_PATH = str(pkglist_path(DEBIAN_VERSION))
PACKAGES_PATH = ''
CHROMADB_PATH = str(chromadb_path(DEBIAN_VERSION))
EMBEDDING_API_BASE = 'https://api.siliconflow.cn/v1/'
EMBEDDING_API_KEY = os.environ.get('EMBEDDING_API_KEY', '')
EMBEDDING_MODEL = 'Qwen/Qwen3-Embedding-4B'
CHROMADB_COLLECTION = chromadb_collection(DEBIAN_VERSION)

_RUNTIME_ABI_RE = re.compile(r'^lib.*\d(?:[a-z]*\d*)?(?:t64)?$')
_EXPANSION_DENY_RE = re.compile(
    r'(?:^|-)doc(?:-|$)|(?:^|-)dbg(?:-|$)|(?:^|-)debug(?:-|$)|'
    r'(?:^|-)examples?(?:-|$)|(?:^|-)tests?(?:-|$)|(?:^|-)data(?:-|$)'
)
_EXPANSION_TOOL_ALLOWLIST = frozenset({
    'qtbase5-dev-tools',
    'qttools5-dev-tools',
    'qt6-base-dev-tools',
    'qt6-tools-dev',
    'qt6-tools-dev-tools',
    'qt5-qmake',
    'qt6-qmake',
})

_PROVIDER_FAMILIES = (
    frozenset({'libjack-dev', 'libjack-jackd2-dev', 'jackd2'}),
    frozenset({'libcurl4-openssl-dev', 'libcurl4-gnutls-dev'}),
    frozenset({'ocaml-dune', 'libdune-ocaml-dev'}),
)

_PROVIDER_HINTS = (
    (re.compile(r'(?i)\b(?:jack|libjack|jackd2)\b'), {'libjack-dev', 'libjack-jackd2-dev'}),
    (re.compile(r'(?i)\b(?:libcurl|curl)\b'), {'libcurl4-openssl-dev', 'libcurl4-gnutls-dev'}),
    (re.compile(r'(?i)\bdune(?:-configurator)?\b'), {'ocaml-dune', 'libdune-ocaml-dev'}),
)

_PROVIDER_PRIORITY = (
    ('pkg-config', 'pkgconf'),
    ('libkrb5-dev', 'heimdal-dev'),
    ('libcurl4-gnutls-dev', 'libcurl4-openssl-dev'),
    ('libgl1-mesa-dev', 'libgl1-mesa-swx11-dev', 'nvidia-glx-dev', 'nvidia-glx-legacy-96xx-dev'),
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    enable_header_anchors_in_llm: bool = False
    enable_header_conflict_filter: bool = False
    enable_header_result_injection: bool = False


EXPERIMENTS = {
    'extract_only': ExperimentConfig(name='extract_only'),
    'extract_plus_header': ExperimentConfig(
        name='extract_plus_header',
        enable_header_result_injection=True,
    ),
    'extract_plus_filter': ExperimentConfig(
        name='extract_plus_filter',
        enable_header_anchors_in_llm=True,
        enable_header_conflict_filter=False,
        enable_header_result_injection=True,
    ),
    'extract_plus_conflict_filter': ExperimentConfig(
        name='extract_plus_conflict_filter',
        enable_header_conflict_filter=True,
        enable_header_result_injection=True,
    ),
}

_conflicts_index_cache = None

ALL_SOURCE_FLAGS = frozenset({
    'build_driver',
    'direct_rule',
    'match_artifact',
    'file_artifact_backfill',
    'header_anchor',
})

OPTIONAL_SOURCE_FLAGS = frozenset({
    'llm',
    'same_source_expansion',
})

SOURCE_FLAGS = ALL_SOURCE_FLAGS | OPTIONAL_SOURCE_FLAGS


def effective_distro_selection(requested: str | None, test_mode: bool) -> str:
    """Resolve distro selection mode after test-mode defaults are applied."""
    if requested:
        return requested
    return 'fixed' if test_mode else 'auto'


def should_prescan_for_distro_selection(distro_selection: str) -> bool:
    """Only auto-selection needs the expensive up-front scan/select pass."""
    return distro_selection == 'auto'


def configure_metadata(debian_version: str | None = None) -> None:
    """Switch all local Debian metadata used by a run."""
    global DEBIAN_VERSION, PKGLIST_PATH, PACKAGES_PATH, CHROMADB_PATH, CHROMADB_COLLECTION
    global _pkglist_cache, _package_source_cache, _source_dev_packages_cache, _source_packages_cache
    global _chromadb_cache, _chromadb_collection_cache, _valid_package_cache, _artifact_cache

    DEBIAN_VERSION = normalize_debian_version(debian_version)
    PKGLIST_PATH = str(pkglist_path(DEBIAN_VERSION))
    PACKAGES_PATH = ''
    CHROMADB_PATH = str(chromadb_path(DEBIAN_VERSION))
    CHROMADB_COLLECTION = chromadb_collection(DEBIAN_VERSION)
    artifacts.configure_metadata(
        debian_version=DEBIAN_VERSION,
        provider_index=str(provider_index_path(DEBIAN_VERSION)),
    )
    _pkglist_cache = None
    _package_source_cache = None
    _source_dev_packages_cache = None
    _source_packages_cache = None
    _chromadb_cache = None
    _chromadb_collection_cache = None
    _valid_package_cache = None
    # Save old cache before switching, then reset
    if _artifact_cache is not None:
        _artifact_cache.save()
    _artifact_cache = None
    # Reset confidence_filter caches so they reload with the new version's data
    from .confidence_filter import reset_cache as _reset_confidence_cache
    _reset_confidence_cache()


def _log(msg: str):
    with _print_lock:
        print(msg, file=sys.stderr, flush=True)


def _looks_like_code_evidence(software_name: str) -> bool:
    """Check if LLM's software_name looks like a code snippet, not a project name."""
    return bool(re.search(
        r'[(\']|^-l|dependency\s*\(|find_package|pkg_check|check_lib|check_header|'
        r'find_program|find_library|target_link|QT\s*[+=]|LIBS\s*[+=]|PKGCONFIG\s*[+=]',
        software_name, re.IGNORECASE,
    ))


def _project_identity(project_path: str) -> tuple[str, str]:
    repo_name = os.path.basename(os.path.normpath(project_path))
    project_name = repo_name
    if '_' in repo_name:
        project_name = repo_name.split('_', 1)[1]
    return project_name, repo_name


def process_project(
    client: openai.OpenAI,
    model: str,
    project_path: str,
    context_lines: int = CONTEXT_LINES,
    max_retries: int = 3,
    llm_timeout: float | None = None,
    disable_llm: bool = False,
    experiment: ExperimentConfig | None = None,
    scan: ProjectScan | None = None,
    enabled_sources: set[str] | None = None,
    use_llm_judge: bool = False,
) -> dict:
    """Process a single project: scan + LLM map."""
    experiment = experiment or EXPERIMENTS['extract_only']
    enabled_sources = set(enabled_sources or ALL_SOURCE_FLAGS)
    scan = scan or scan_project(project_path, context_lines=context_lines)
    project_name, repo_name = _project_identity(project_path)

    per_file = {}
    artifacts_by_file = {
        rel_path: list(file_artifacts)
        for rel_path, file_artifacts in getattr(scan, 'file_artifacts', {}).items()
    }
    from .header_scan import collect_header_anchor_packages, inject_header_anchor_packages
    need_header_anchor_packages = (
        'header_anchor' in enabled_sources
        and (
            experiment.enable_header_anchors_in_llm
            or experiment.enable_header_result_injection
            or experiment.enable_header_conflict_filter
        )
    )
    header_anchor_packages = (
        collect_header_anchor_packages(project_path, _get_pkglist())
        if need_header_anchor_packages
        else set()
    )

    # Direct packages from debian/control
    if scan.direct_packages and 'direct_rule' in enabled_sources:
        per_file['debian/control'] = [
            {'software_name': pkg, 'packages': [pkg], 'sources': ['direct_rule']}
            for pkg in sorted(scan.direct_packages)
        ]

    # Auto-detected build tool packages
    if scan.build_driver_packages and 'build_driver' in enabled_sources:
        per_file['__auto_build_tools__'] = [
            {'software_name': pkg, 'packages': [pkg], 'sources': ['build_driver']}
            for pkg in sorted(scan.build_driver_packages)
        ]

    # Per-file LLM mapping
    files_processed = 0
    cache = _get_artifact_cache() if enabled_sources == set(ALL_SOURCE_FLAGS) else None
    valid_pkgs = _get_pkglist()
    bvars = scan.build_variables

    # Pre-fill cache from direct matches
    if cache is not None:
        cache.prefill_from_direct_matches(scan.file_matches, valid_pkgs, build_variables=bvars)

    for rel_path, matches in scan.file_matches.items():
        if not matches:
            continue

        # Resolve deterministic direct/package/artifact evidence before LLM fallback.
        cached_deps = []
        remaining_matches = []
        for match in matches:
            deterministic_pkgs, deterministic_sources = _deterministic_packages_for_match(
                match,
                valid_pkgs,
                enabled_sources,
            )
            cached = cache.query(match.matched_text, valid_pkgs, build_variables=bvars) if cache is not None else None
            all_pkgs, _ = _resolve_suite_package_set(
                sorted(set(deterministic_pkgs + (cached or []))),
                valid_pkgs,
                _get_conflicts_for_suite(),
            )
            if all_pkgs:
                dep = {'software_name': match.matched_text[:100], 'packages': all_pkgs}
                if deterministic_sources:
                    dep['sources'] = sorted(set(deterministic_sources) | ({'cache'} if cached else set()))
                elif cached:
                    dep['sources'] = ['cache']
                cached_deps.append(dep)
                # Artifact-derived mappings remain local to this project. Only
                # direct package declarations are persisted by DependencyCache.
            else:
                remaining_matches.append(match)

        # Direct matches: just process, no LLM needed
        if rel_path.startswith('__direct__/'):
            extra_deps = []
            if 'llm' in enabled_sources:
                llm_direct = map_file_deps(
                    client, model, rel_path, remaining_matches, scan.build_systems,
                    project_name=project_name,
                    repo_name=repo_name,
                    max_retries=max_retries,
                    timeout=llm_timeout,
                    header_anchor_packages=sorted(header_anchor_packages) if experiment.enable_header_anchors_in_llm else None,
                )
                for dep in llm_direct:
                    dep['sources'] = sorted(set(dep.get('sources', [])) | {'llm'})
                extra_deps = llm_direct
            deps = merge_dependencies(cached_deps + extra_deps)
            if deps:
                per_file[rel_path] = deps
            files_processed += 1
            continue

        # Collect already-resolved packages as context for LLM
        resolved_pkgs = sorted({
            pkg
            for deps in per_file.values()
            for dep in deps
            for pkg in dep.get('packages', [])
        } | {
            pkg
            for dep in cached_deps
            for pkg in dep.get('packages', [])
        } | (
            set(header_anchor_packages) if experiment.enable_header_result_injection else set()
        ))

        # Only call LLM for uncached matches
        if remaining_matches and not disable_llm and 'llm' in enabled_sources:
            llm_deps = map_file_deps(
                client, model, rel_path, remaining_matches, scan.build_systems,
                project_name=project_name,
                repo_name=repo_name,
                max_retries=max_retries,
                timeout=llm_timeout,
                resolved_packages=resolved_pkgs,
                header_anchor_packages=sorted(header_anchor_packages) if experiment.enable_header_anchors_in_llm else None,
            )
            # Match LLM output back to matched_text for cache update
            match_norm_map = {}
            for match in remaining_matches:
                mn = normalize_cache_key(match.matched_text, bvars)
                if mn:
                    match_norm_map[mn] = match

            for dep in llm_deps:
                pkgs = dep.get('packages', [])
                sw = dep.get('software_name', '')
                if not pkgs or not sw:
                    continue
                dep['sources'] = sorted(set(dep.get('sources', [])) | {'llm'})
                # Validate: software_name must look like code, not a project name
                if not _looks_like_code_evidence(sw):
                    continue
                sw_norm = normalize_cache_key(sw, bvars)
                if not sw_norm:
                    continue
                # Exact normalized match → use matched_text as cache key
                # LLM mappings are bounded candidates, not global cache facts.
        else:
            llm_deps = []
            # Capture clues that deterministic paths could not resolve, so they
            # survive into the Build Plan for later failure diagnosis (paper L77).
            for m in remaining_matches:
                per_file.setdefault('__unmatched_clues__', []).append({
                    'matched_text': (m.matched_text or '')[:200],
                    'extracted_name': m.extracted_name,
                    'rule_type': m.rule_type,
                    'file_path': m.file_path,
                    'line_num': m.line_num,
                    'artifact_kinds': [a.kind for a in (m.artifacts or [])],
                    'artifact_names': [a.name for a in (m.artifacts or [])],
                })

        # Merge cached + LLM results
        all_deps = merge_dependencies(cached_deps + llm_deps)
        if all_deps:
            per_file[rel_path] = all_deps
        files_processed += 1

    if 'file_artifact_backfill' in enabled_sources:
        backfill_artifact_packages(per_file, artifacts_by_file, valid_pkgs)
        _tag_bucket_sources(per_file, '__artifact_backfill__', 'file_artifact_backfill')

    # Confidence-based FP filter (M1+M1b+M1c+M-A+M-D). Silently no-ops if the priors
    # file is missing, so it is safe to call unconditionally.
    from .confidence_filter import apply_confidence_filter
    apply_confidence_filter(per_file, project_path=project_path, version=DEBIAN_VERSION)

    if 'same_source_expansion' in enabled_sources:
        expand_same_source_dev_packages(per_file, artifacts_by_file)
        _tag_bucket_sources(per_file, '__same_source_dev_expansion__', 'same_source_expansion')

    if experiment.enable_header_result_injection and 'header_anchor' in enabled_sources:
        inject_header_anchor_packages(per_file, header_anchor_packages)
        _tag_bucket_sources(per_file, '__header_anchor_packages__', 'header_anchor')

    if experiment.enable_header_conflict_filter:
        apply_header_conflict_filter(per_file, header_anchor_packages)

    _ensure_suite_package_fields(per_file)
    resolve_suite_packages(per_file)

    # Canonicalise package variants (libc++-17/18/19-dev → libc++-dev, etc.). Done
    # last so it sees the union of all buckets.
    from .canonicalize import canonicalize_per_file
    canonicalize_per_file(per_file)
    _ensure_suite_package_fields(per_file)
    resolve_suite_packages(per_file)

    # LLM judge: one batched KEEP/DROP call per project to filter false positives.
    # Only runs when explicitly enabled. Skips if client is None (test mode).
    if use_llm_judge and client is not None:
        from .llm_judge import apply_llm_judge
        apply_llm_judge(
            per_file, project_path, client, model,
            max_retries=max_retries, timeout=llm_timeout, version=DEBIAN_VERSION,
        )
        _ensure_suite_package_fields(per_file)
        resolve_suite_packages(per_file)

    return per_file

def _deterministic_packages_for_match(
    match,
    valid_pkgs: set[str],
    enabled_sources: set[str],
) -> tuple[list[str], set[str]]:
    """Resolve a single rule match without LLM using direct and artifact evidence."""
    packages: set[str] = set()
    sources: set[str] = set()
    if 'direct_rule' in enabled_sources:
        direct = {pkg for pkg in (match.packages or []) if pkg in valid_pkgs}
        if direct:
            packages.update(direct)
            sources.add('direct_rule')
    if 'match_artifact' in enabled_sources:
        artifact_pkgs = artifacts.packages_from_artifacts(getattr(match, 'artifacts', []) or [], valid_pkgs)
        if artifact_pkgs:
            packages.update(artifact_pkgs)
            sources.add('match_artifact')
    return sorted(packages), sources


def _tag_bucket_sources(per_file: dict, bucket: str, source_name: str) -> None:
    deps = per_file.get(bucket)
    if not isinstance(deps, list):
        return
    for dep in deps:
        if not isinstance(dep, dict):
            continue
        dep['sources'] = sorted(set(dep.get('sources', [])) | {source_name})


def _get_pkglist():
    global _pkglist_cache
    if _pkglist_cache is None:
        with _kb_lock:
            if _pkglist_cache is None:
                with open(PKGLIST_PATH) as f:
                    _pkglist_cache = set(json.load(f).keys())
    return _pkglist_cache


def _get_conflicts_for_suite() -> dict[str, list[str]]:
    global _conflicts_index_cache
    if _conflicts_index_cache is None:
        with _kb_lock:
            if _conflicts_index_cache is None:
                from .build_conflicts_index import OUTPUT_PATH
                with open(OUTPUT_PATH) as f:
                    payload = json.load(f)
                _conflicts_index_cache = payload.get('conflicts', {}).get(DEBIAN_VERSION, {})
    return _conflicts_index_cache


def apply_header_conflict_filter(per_file: dict, header_anchor_packages: set[str]) -> list[dict]:
    """Apply project-wide direct conflict filtering in-place and return a log.

    `header_anchor_packages` is kept for the existing call surface; filtering now
    compares all final packages by evidence source instead of treating headers as
    the only trusted side.
    """
    conflicts = _get_conflicts_for_suite()
    package_sources: dict[str, set[str]] = {}
    for package in header_anchor_packages:
        if isinstance(package, str) and package:
            package_sources.setdefault(package, set()).add('header_anchor')
    for deps in per_file.values():
        if not isinstance(deps, list):
            continue
        for dep in deps:
            if not isinstance(dep, dict):
                continue
            sources = set(dep.get('sources') or [])
            for package in dep.get('packages', []):
                if isinstance(package, str) and package:
                    package_sources.setdefault(package, set()).update(sources)

    dropped_packages, filter_log = filter_project_conflicts(package_sources, conflicts)
    if not dropped_packages:
        return filter_log

    for file_key, deps in list(per_file.items()):
        if not isinstance(deps, list):
            continue
        filtered_deps = []
        for dep in deps:
            if not isinstance(dep, dict):
                continue
            kept = [pkg for pkg in dep.get('packages', []) if pkg not in dropped_packages]
            if not kept:
                continue
            dep['packages'] = kept
            if 'suite_packages' in dep:
                dep['suite_packages'] = [pkg for pkg in dep.get('suite_packages', []) if pkg not in dropped_packages]
            filtered_deps.append(dep)
        if filtered_deps:
            per_file[file_key] = filtered_deps
        else:
            per_file.pop(file_key, None)

    if filter_log:
        per_file['__filter_log__'] = filter_log
    return filter_log


def _package_field(dep: dict) -> list[str]:
    packages = dep.get('suite_packages')
    if packages is None:
        packages = dep.get('packages', [])
    if isinstance(packages, str):
        return [packages]
    if not isinstance(packages, list):
        return []
    return [str(pkg).strip() for pkg in packages if str(pkg).strip()]


def _set_package_fields(dep: dict, packages: list[str]) -> None:
    unique = sorted(dict.fromkeys(str(pkg).strip() for pkg in packages if str(pkg).strip()))
    dep['packages'] = list(unique)
    dep['suite_packages'] = list(unique)


def _normalize_suite_package(pkg: str, valid_pkgs: set[str]) -> str:
    if pkg == 'pkgconf' and pkg not in valid_pkgs and 'pkg-config' in valid_pkgs:
        return 'pkg-config'
    return pkg


def _provider_rank(pkg: str) -> tuple[int, int]:
    for family_index, family in enumerate(_PROVIDER_PRIORITY):
        if pkg in family:
            return family_index, family.index(pkg)
    return (10_000, 10_000)


def _resolve_provider_families(packages: set[str], valid_pkgs: set[str]) -> tuple[set[str], list[dict]]:
    resolved = set(packages)
    log: list[dict] = []
    for family in _PROVIDER_PRIORITY:
        present = [pkg for pkg in family if pkg in resolved]
        if len(present) <= 1:
            continue
        valid_present = [pkg for pkg in present if pkg in valid_pkgs]
        keep = valid_present[0] if valid_present else present[0]
        for pkg in present:
            if pkg == keep:
                continue
            resolved.discard(pkg)
            log.append({'reason': 'provider_family_choice', 'dropped': pkg, 'kept': keep})
    return resolved, log


def _resolve_suite_package_set(packages: list[str], valid_pkgs: set[str], conflicts: dict[str, list[str]]) -> tuple[list[str], list[dict]]:
    normalized = {
        _normalize_suite_package(pkg, valid_pkgs)
        for pkg in packages
        if isinstance(pkg, str) and pkg
    }
    normalized = {pkg for pkg in normalized if pkg in valid_pkgs}
    resolved, log = _resolve_provider_families(normalized, valid_pkgs)

    changed = True
    while changed:
        changed = False
        for pkg in sorted(resolved):
            conflicts_with = [other for other in conflicts.get(pkg, []) if other in resolved and other != pkg]
            if not conflicts_with:
                continue
            for other in conflicts_with:
                keep, drop = sorted([pkg, other], key=_provider_rank)
                if drop in resolved:
                    resolved.discard(drop)
                    log.append({'reason': 'suite_conflict_choice', 'dropped': drop, 'kept': keep})
                    changed = True
                    break
            if changed:
                break
    return sorted(resolved), log


def resolve_suite_packages(per_file: dict) -> list[dict]:
    """Normalize package fields, apply suite availability, and resolve common conflicts."""
    valid_pkgs = _get_pkglist()
    conflicts = _get_conflicts_for_suite()
    filter_log: list[dict] = []
    for file_key, deps in list(per_file.items()):
        if not isinstance(deps, list):
            continue
        kept_deps = []
        for dep in deps:
            if not isinstance(dep, dict):
                kept_deps.append(dep)
                continue
            resolved, dep_log = _resolve_suite_package_set(_package_field(dep), valid_pkgs, conflicts)
            if not resolved:
                continue
            _set_package_fields(dep, resolved)
            for item in dep_log:
                item.setdefault('file_key', file_key)
                item.setdefault('software_name', dep.get('software_name') or '')
            filter_log.extend(dep_log)
            kept_deps.append(dep)
        if kept_deps:
            per_file[file_key] = kept_deps
        else:
            per_file.pop(file_key, None)

    project_packages = {
        pkg
        for deps in per_file.values()
        if isinstance(deps, list)
        for dep in deps
        if isinstance(dep, dict)
        for pkg in _package_field(dep)
    }
    project_resolved, project_log = _resolve_suite_package_set(sorted(project_packages), valid_pkgs, conflicts)
    project_keep = set(project_resolved)
    if project_keep != project_packages:
        for file_key, deps in list(per_file.items()):
            if not isinstance(deps, list):
                continue
            kept_deps = []
            for dep in deps:
                if not isinstance(dep, dict):
                    kept_deps.append(dep)
                    continue
                kept = [pkg for pkg in _package_field(dep) if pkg in project_keep]
                if not kept:
                    continue
                _set_package_fields(dep, kept)
                kept_deps.append(dep)
            if kept_deps:
                per_file[file_key] = kept_deps
            else:
                per_file.pop(file_key, None)
        filter_log.extend(project_log)
    if filter_log:
        per_file['__suite_resolution_log__'] = filter_log
    return filter_log


def _ensure_suite_package_fields(per_file: dict) -> None:
    """Populate both `packages` and `suite_packages` with the active suite package names."""
    for deps in per_file.values():
        if not isinstance(deps, list):
            continue
        for dep in deps:
            if not isinstance(dep, dict):
                continue
            packages = _package_field(dep)
            if packages:
                _set_package_fields(dep, packages)


def _get_artifact_cache() -> DependencyCache:
    global _artifact_cache
    if _artifact_cache is None:
        with _kb_lock:
            if _artifact_cache is None:
                _artifact_cache = DependencyCache(distro=DEBIAN_VERSION)
                _artifact_cache.load()
    return _artifact_cache


def _get_package_source_maps():
    global _package_source_cache, _source_dev_packages_cache, _source_packages_cache
    if (
        _package_source_cache is not None
        and _source_dev_packages_cache is not None
        and _source_packages_cache is not None
    ):
        return _package_source_cache, _source_dev_packages_cache, _source_packages_cache

    with _kb_lock:
        if (
            _package_source_cache is not None
            and _source_dev_packages_cache is not None
            and _source_packages_cache is not None
        ):
            return _package_source_cache, _source_dev_packages_cache, _source_packages_cache

        package_source = {}
        source_dev_packages = {}
        source_packages = {}
        blocks = []
        paths = [PACKAGES_PATH] if PACKAGES_PATH else [str(path) for path in pkginfo_paths(DEBIAN_VERSION)]
        for path in paths:
            try:
                with open(path, errors='replace') as f:
                    blocks.extend(f.read().split('\n\n'))
            except Exception as e:
                _log(f'  Packages metadata load failed for {path}: {e}')

        for block in blocks:
            pkg_match = re.search(r'^Package:\s*(\S+)', block, re.MULTILINE)
            if not pkg_match:
                continue
            pkg = pkg_match.group(1)
            source_match = re.search(r'^Source:\s*(\S+)', block, re.MULTILINE)
            source = source_match.group(1).split('(')[0].strip() if source_match else pkg
            package_source[pkg] = source
            source_packages.setdefault(source, set()).add(pkg)
            pkg_l = pkg.lower()
            if (
                pkg_l.endswith('-dev')
                or '-dev-' in pkg_l
                or pkg_l.endswith('-tools')
                or pkg_l.endswith('-tool')
                or pkg_l.endswith('-bin')
            ):
                source_dev_packages.setdefault(source, set()).add(pkg)

        _package_source_cache = package_source
        _source_dev_packages_cache = source_dev_packages
        _source_packages_cache = source_packages
        return _package_source_cache, _source_dev_packages_cache, _source_packages_cache


def _is_runtime_expansion_seed(pkg: str) -> bool:
    pkg_l = pkg.lower()
    if pkg_l in {'gcc', 'g++', 'gfortran'}:
        return False
    if pkg_l.endswith(('-dev', '-tools', '-tool', '-bin', '-doc')):
        return False
    if '-dev-' in pkg_l:
        return False
    if _RUNTIME_ABI_RE.match(pkg_l):
        return True
    if pkg_l in {'systemd', 'jackd2'}:
        return True
    return False


def _is_same_source_expansion_candidate(pkg: str) -> bool:
    pkg_l = pkg.lower()
    if _EXPANSION_DENY_RE.search(pkg_l):
        return False
    if re.match(r'lib(?:32|x32)', pkg_l):
        return False
    if re.search(r'(?:^|-)gcc(?:jit)?-\d+-dev$', pkg_l):
        return False
    if re.search(r'(?:^|-)g(?:fortran|m2|o|phobos)-\d+-dev$', pkg_l):
        return False
    if re.search(r'(?:^|-)objc-\d+-dev$', pkg_l):
        return False
    if re.search(r'(?:^|-)stdc\+\+-\d+-dev$', pkg_l):
        return False
    if pkg_l.endswith('-doc') or '-doc-' in pkg_l:
        return False
    if pkg_l in _EXPANSION_TOOL_ALLOWLIST:
        return True
    if pkg_l.endswith('-dev') and '-private-dev' not in pkg_l:
        return True
    return False


def _is_runtime_sibling_seed(pkg: str) -> bool:
    pkg_l = pkg.lower()
    return (
        pkg_l.endswith(('-dev', '-tools', '-tool', '-bin'))
        or '-dev-' in pkg_l
        or pkg_l in _EXPANSION_TOOL_ALLOWLIST
    )


def _is_same_source_runtime_candidate(pkg: str) -> bool:
    pkg_l = pkg.lower()
    if _EXPANSION_DENY_RE.search(pkg_l):
        return False
    if pkg_l.endswith(('-dev', '-tools', '-tool', '-bin', '-doc')):
        return False
    if '-dev-' in pkg_l:
        return False
    if re.match(r'lib(?:32|x32)', pkg_l):
        return False
    return True


def _artifact_text(artifacts_by_file) -> str:
    if not artifacts_by_file:
        return ''
    parts = []
    for artifacts in artifacts_by_file.values():
        for artifact in artifacts:
            parts.append(artifact.name)
            parts.extend(artifact.components)
    return ' '.join(parts)


def _artifact_has_feature(artifacts_by_file, feature: str) -> bool:
    if not artifacts_by_file:
        return False
    return any(
        artifact.kind == 'feature' and artifact.name == feature
        for artifacts in artifacts_by_file.values()
        for artifact in artifacts
    )


def _package_artifact_names(artifacts_by_file) -> set[str]:
    if not artifacts_by_file:
        return set()
    return {
        artifact.name.lower()
        for artifacts_for_file in artifacts_by_file.values()
        for artifact in artifacts_for_file
        if artifact.kind == 'package'
    }


def _expansion_evidence_allows(pkg: str, dep_text: str, artifact_text: str) -> bool:
    evidence = f'{dep_text} {artifact_text}'
    pkg_l = pkg.lower()
    if pkg_l.startswith(('libqt5', 'libqt6', 'qtbase5', 'qt6-')):
        return bool(re.search(r'(?i)\b(?:qt|qml|quick|widgets|linguist|qdoc|moc|uic|rcc)\b', evidence))
    if pkg_l in {'systemd', 'libsystemd0'}:
        return bool(re.search(r'(?i)\b(?:systemd|libsystemd|journald|sd-)\b', evidence))
    if pkg_l in {'jackd2'} or 'jack' in pkg_l:
        return bool(re.search(r'(?i)\b(?:jack|libjack|jackd2)\b', evidence))
    if 'curl' in pkg_l:
        return bool(re.search(r'(?i)\b(?:curl|libcurl)\b', evidence))
    if pkg_l.startswith('lib') and re.search(r'\d', pkg_l):
        stem = re.sub(r'(?:t64)?\d.*$', '', pkg_l)
        stem = stem[3:] if stem.startswith('lib') else stem
        return bool(stem and re.search(rf'(?i)\b{re.escape(stem)}\b', evidence))
    return False


def _qt_source_sibling_allowed(pkg: str, artifacts_by_file) -> bool:
    pkg_l = pkg.lower()
    if pkg_l in {'qtbase5-dev-tools', 'qttools5-dev-tools', 'qt6-tools-dev-tools'}:
        return True
    if pkg_l in {'qt5-qmake', 'qt6-qmake'}:
        return True
    if pkg_l in {'qttools5-dev', 'qt6-tools-dev', 'qt6-l10n-tools'}:
        return _artifact_has_feature(artifacts_by_file, 'qt_translation')
    return False


def expand_same_source_dev_packages(per_file, artifacts_by_file=None):
    if not per_file:
        return

    package_source, source_dev_packages, source_packages = _get_package_source_maps()
    if not package_source or not source_dev_packages:
        return

    additions = set()
    artifact_text = _artifact_text(artifacts_by_file)
    package_artifact_seeds = _package_artifact_names(artifacts_by_file)
    for file_deps in per_file.values():
        for dep in file_deps:
            dep_text = ' '.join([str(dep.get('software_name', ''))] + dep.get('packages', []))
            for regex, family in _PROVIDER_HINTS:
                if regex.search(dep_text):
                    additions.update(family)
            for pkg in dep.get('packages', []):
                for family in _PROVIDER_FAMILIES:
                    if pkg in family:
                        additions.update(family)
                # Runtime sibling expansion disabled — it leaks runtime packages
                # (libssl3t64, openssl-provider-legacy, jackd2, ...) that are not valid
                # build-deps. Same-source expansion below (-dev siblings only) covers
                # legitimate cases.
                if not _is_runtime_expansion_seed(pkg):
                    continue
                if not _expansion_evidence_allows(pkg, dep_text, artifact_text):
                    continue
                source = package_source.get(pkg)
                if source:
                    additions.update(
                        candidate for candidate in source_dev_packages.get(source, set())
                        if _is_same_source_expansion_candidate(candidate)
                        and (
                            not (pkg.startswith('libqt5') or pkg.startswith('qtbase5') or pkg.startswith('libqt6') or pkg.startswith('qt6-'))
                            or _qt_source_sibling_allowed(candidate, artifacts_by_file)
                        )
                    )
                if pkg.startswith('libqt5') or pkg.startswith('qtbase5'):
                    additions.add('qt5-qmake')
                if pkg.startswith('libqt6') or pkg.startswith('qt6-'):
                    additions.add('qt6-tools-dev-tools')

    existing = set()
    for file_deps in per_file.values():
        for dep in file_deps:
            existing.update(dep.get('packages', []))

    additions -= existing
    if not additions:
        return

    per_file['__same_source_dev_expansion__'] = [
        {'software_name': pkg, 'packages': [pkg]}
        for pkg in sorted(additions)
    ]


def _get_chromadb_collection():
    global _chromadb_cache, _chromadb_collection_cache
    if _chromadb_collection_cache is None:
        with _kb_lock:
            if _chromadb_collection_cache is None:
                import chromadb
                from chromadb.config import Settings
                from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

                embedding_fn = OpenAIEmbeddingFunction(
                    api_base=EMBEDDING_API_BASE,
                    api_key=EMBEDDING_API_KEY,
                    model_name=EMBEDDING_MODEL,
                )
                _chromadb_cache = chromadb.PersistentClient(
                    path=CHROMADB_PATH,
                    settings=Settings(anonymized_telemetry=False),
                )
                _chromadb_collection_cache = _chromadb_cache.get_collection(
                    name=CHROMADB_COLLECTION,
                    embedding_function=embedding_fn,
                )
    return _chromadb_collection_cache


def verify_and_correct_per_project(per_file):
    if not per_file:
        return

    pkglist = _get_pkglist()
    collection = _get_chromadb_collection()

    all_pkgs = set()
    for file_deps in per_file.values():
        for dep in file_deps:
            if isinstance(dep, dict):
                all_pkgs.update(_package_field(dep))

    if not all_pkgs:
        return

    invalid = [p for p in all_pkgs if p not in pkglist]
    if not invalid:
        return

    with _kb_lock:
        try:
            resp = collection.query(query_texts=invalid, n_results=3)
            vec_map = {}
            for i, p in enumerate(invalid):
                ids = resp['ids'][i]
                for id_ in ids:
                    if id_ and id_ in pkglist:
                        vec_map[p] = id_
                        break
        except Exception as e:
            _log(f'  KB vector search failed: {e}')
            return

    if not vec_map:
        return

    for file_deps in per_file.values():
        for dep in file_deps:
            if not isinstance(dep, dict):
                continue
            pkgs = _package_field(dep)
            if not pkgs:
                continue
            corrected = []
            for p in pkgs:
                if p in pkglist:
                    corrected.append(p)
                elif p in vec_map:
                    corrected.append(vec_map[p])
            _set_package_fields(dep, corrected)
    resolve_suite_packages(per_file)


def collect_all_packages(per_project: dict) -> list[str]:
    """Collect unique package names across all projects."""
    pkgs = set()
    for proj_payload in per_project.values():
        # New structure: {'dependency_projects': [...], 'all_packages': [...]}
        if isinstance(proj_payload, dict) and 'all_packages' in proj_payload:
            pkgs.update(proj_payload['all_packages'])
            continue
        # Legacy structure: {rel_path: list[dep]}
        for file_deps in proj_payload.values():
            if isinstance(file_deps, list):
                for dep in file_deps:
                    if isinstance(dep, dict):
                        pkgs.update(_package_field(dep))
    return sorted(pkgs)


def _project_local_names(project_path: str) -> set[str]:
    """Return semantic names proven to be repository-local build entities."""
    root = Path(project_path)
    values = {root.name}
    try:
        values.update(item.name for item in root.iterdir() if item.is_dir() and not item.name.startswith('.'))
        for pattern in ('*Config.cmake.in', '*-config.cmake.in', '*.pc.in'):
            values.update(path.name.split('.', 1)[0].removesuffix('-config').removesuffix('Config') for path in root.glob(pattern))
    except OSError:
        pass
    return {re.sub(r'[^a-z0-9]+', '', value.lower()) for value in values if value}


def main():
    parser = argparse.ArgumentParser(description='Enhanced rule+LLM dependency extractor')
    parser.add_argument('input', help='Text file with one project path per line')
    parser.add_argument('-o', '--output', required=True, help='Output JSON file')
    parser.add_argument('--model', default=DEFAULT_MODEL, help=f'LLM model (default: {DEFAULT_MODEL})')
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL, help='OpenAI-compatible API base URL')
    parser.add_argument('--api-key', default=None, help='API key (default: $ARK_API_KEY or $OPENAI_API_KEY)')
    parser.add_argument('--workers', type=int, default=8, help='Project-level parallelism')
    parser.add_argument('--context-lines', type=int, default=CONTEXT_LINES, help='±N context lines around matches')
    parser.add_argument('--max-retries', type=int, default=3, help='Max LLM retries per chunk')
    parser.add_argument('--llm-timeout', type=float, default=None, help='Timeout seconds per LLM chunk')
    parser.add_argument('--disable-llm', action='store_true', help='Skip LLM mapping for uncached rule matches')
    parser.add_argument('--llm-judge', action='store_true',
                        help='Enable LLM judge: one batched KEEP/DROP call per project to filter false positives')
    parser.add_argument(
        '--source-mask',
        default=None,
        help='Comma-separated source set to enable: '
             'build_driver,direct_rule,match_artifact,llm,file_artifact_backfill,same_source_expansion,header_anchor',
    )
    parser.add_argument('--save-interval', type=int, default=10, help='Save checkpoint every N completed projects')
    parser.add_argument('--skip-existing', action='store_true', help='Skip projects already in output file')
    parser.add_argument('--debian-version', default=DEFAULT_DEBIAN_VERSION, help='Debian metadata version to use')
    parser.add_argument(
        '--scan-third-party',
        action='store_true',
        help='Include thirdparty/vendor/external/deps/subprojects/bundled directories in project scans',
    )
    parser.add_argument(
        '--scan-debian-rules',
        action='store_true',
        help='Parse debian/rules for extra -l<lib> and pkg-config hints '
             'beyond what debian/control provides. Off by default.',
    )
    parser.add_argument(
        '--distro-selection',
        choices=('auto', 'fixed'),
        default=None,
        help='Validate/correct the baseline Debian version from project constraints',
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Use testing defaults; currently skips distro selection unless explicitly overridden',
    )
    parser.add_argument(
        '--skip-distro-selection',
        action='store_true',
        help='Testing shortcut: use fixed baseline Debian metadata without distro selection',
    )
    parser.add_argument(
        '--experiment',
        choices=sorted(EXPERIMENTS),
        default='extract_only',
        help='Experiment preset controlling header injection/filtering behavior',
    )
    args = parser.parse_args()
    args.test_mode = args.test_mode or args.skip_distro_selection
    args.distro_selection = effective_distro_selection(args.distro_selection, args.test_mode)
    if args.source_mask:
        enabled_sources = {
            token.strip() for token in args.source_mask.split(',')
            if token.strip()
        }
        unknown_sources = sorted(enabled_sources - SOURCE_FLAGS)
        if unknown_sources:
            raise SystemExit(f'Unknown --source-mask entries: {", ".join(unknown_sources)}')
    else:
        enabled_sources = set(ALL_SOURCE_FLAGS)
    baseline_debian_version = normalize_debian_version(args.debian_version)
    configure_metadata(baseline_debian_version)
    experiment = EXPERIMENTS[args.experiment]

    api_key = args.api_key or os.environ.get('ARK_API_KEY') or os.environ.get('OPENAI_API_KEY', os.environ.get('OPENROUTER_API_KEY', ''))
    if not api_key:
        print('Error: set ARK_API_KEY or OPENAI_API_KEY environment variable', file=sys.stderr)
        sys.exit(1)

    with open(args.input) as f:
        project_paths = [line.strip() for line in f if line.strip()]

    _log(f'Loaded {len(project_paths)} project paths')
    _log(f'Using baseline Debian metadata: {baseline_debian_version}')

    # Load existing results if skip-existing
    existing = {}
    existing_metadata = {}
    existing_constraint_details = {}
    if args.skip_existing and os.path.exists(args.output):
        try:
            with open(args.output) as f:
                payload = json.load(f)
                existing = payload.get('per_project', {})
                existing_metadata = payload.get('per_project_metadata', {})
            details_path = _constraint_details_path(args.output)
            if os.path.exists(details_path):
                with open(details_path) as f:
                    details_payload = json.load(f)
                    existing_constraint_details = details_payload.get('per_project_constraint_details', {})
            _log(f'Loaded {len(existing)} existing results')
        except Exception:
            pass

    client = openai.OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        max_retries=get_client_max_retries(args.base_url),
    )
    per_project = dict(existing)
    per_project_metadata = dict(existing_metadata)
    per_project_constraint_details = dict(existing_constraint_details)
    per_project_status: dict[str, dict] = {}

    to_process = [p for p in project_paths if p not in per_project]
    _log(f'Processing {len(to_process)} projects ({len(existing)} skipped)')

    t0 = time.time()
    done = 0
    total = len(to_process)

    scan_by_project: dict[str, ProjectScan] = {}
    decision_by_project: dict[str, DistroDecision] = {}

    def _scan_and_select(proj_path: str):
        if not os.path.isdir(proj_path):
            raise FileNotFoundError(f'project directory does not exist: {proj_path}')
        scan = scan_project(
            proj_path,
            context_lines=args.context_lines,
            scan_third_party=args.scan_third_party,
            scan_debian_rules=args.scan_debian_rules,
        )
        if args.distro_selection == 'fixed':
            decision = fixed_decision(baseline_debian_version)
        else:
            decision = select_debian_version(proj_path, scan, baseline_debian_version)
        return proj_path, scan, decision

    if to_process and should_prescan_for_distro_selection(args.distro_selection):
        _log(f'Scanning {len(to_process)} projects for distro selection')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_scan_and_select, p): p for p in to_process}
            for future in as_completed(futures):
                proj_path = futures[future]
                try:
                    proj_path, scan, decision = future.result()
                    scan_by_project[proj_path] = scan
                    decision_by_project[proj_path] = decision
                    per_project_metadata[proj_path] = decision.as_metadata()
                    per_project_constraint_details[proj_path] = decision.as_constraint_details()
                    if decision.selected_debian_version != decision.baseline_debian_version:
                        _log(
                            f'  Distro correction {os.path.basename(proj_path)}: '
                            f'{decision.baseline_debian_version} -> '
                            f'{decision.selected_debian_version} '
                            f'({decision.selection_source})'
                        )
                except Exception as e:
                    per_project[proj_path] = {}
                    per_project_status[proj_path] = {
                        'status': 'error',
                        'error': f'{type(e).__name__}: {e}',
                        'stage': 'scan_and_select',
                    }
                    per_project_metadata[proj_path] = {
                        'baseline_debian_version': baseline_debian_version,
                        'selected_debian_version': baseline_debian_version,
                        'selection_source': f'selection_error: {e}',
                        'constraints': [],
                    }
                    per_project_constraint_details[proj_path] = {
                        'baseline_debian_version': baseline_debian_version,
                        'selected_debian_version': baseline_debian_version,
                        'selection_source': f'selection_error: {e}',
                        'constraints': [],
                        'constraint_details': [],
                        'unknown_constraints': [],
                        'unknown_constraint_details': [],
                    }
                    done += 1
                    _log(f'[{done}/{total}] ERROR selecting distro for {os.path.basename(proj_path)}: {e}')

    grouped_projects: dict[str, list[str]] = {}
    for proj_path in to_process:
        if proj_path in per_project:
            continue
        decision = decision_by_project.get(proj_path) or fixed_decision(baseline_debian_version)
        grouped_projects.setdefault(decision.selected_debian_version, []).append(proj_path)

    def _worker(proj_path):
        scan = scan_by_project.get(proj_path)
        result = process_project(
            client, args.model, proj_path,
            context_lines=args.context_lines,
            max_retries=args.max_retries,
            llm_timeout=args.llm_timeout,
            disable_llm=args.disable_llm,
            experiment=experiment,
            scan=scan,
            enabled_sources=enabled_sources,
            use_llm_judge=args.llm_judge,
        )
        verify_and_correct_per_project(result)
        # Convert per_file to dependency projects (paper §4.2 grouping clues
        # into dependency projects, retaining unmatched clues).
        from .dependency_project import assemble_dependency_projects
        unmatched = result.pop('__unmatched_clues__', [])
        decision = decision_by_project.get(proj_path)
        suite_name = decision.selected_debian_version if decision else baseline_debian_version
        constraints: list[dict] = []
        try:
            from .suite_compat import extract_version_constraints
            if scan:
                for vc in extract_version_constraints(scan):
                    for clause in vc.clauses:
                        constraints.append({
                            'dep_name': vc.dep_name,
                            'op': clause.op,
                            'version': clause.version,
                        })
        except Exception:
            pass
        dep_projects = assemble_dependency_projects(
            result,
            unmatched,
            suite_name,
            constraints,
            local_names=_project_local_names(proj_path),
        )
        scan_stats = {
            'file_match_count': sum(len(items) for items in (scan.file_matches or {}).values()) if scan else 0,
            'file_artifact_count': sum(len(items) for items in (scan.file_artifacts or {}).values()) if scan else 0,
            'direct_package_count': len(scan.direct_packages or ()) if scan else 0,
            'build_driver_package_count': len(scan.build_driver_packages or ()) if scan else 0,
            'constraint_count': len(constraints),
            'dependency_project_count': len(dep_projects),
            'unmatched_project_count': sum(1 for dp in dep_projects if dp.unmatched),
        }
        signal_count = sum(
            scan_stats[key]
            for key in (
                'file_match_count',
                'direct_package_count',
                'build_driver_package_count',
                'constraint_count',
            )
        )
        status = 'degraded' if signal_count and not dep_projects else 'ok'
        return proj_path, {
            'dependency_projects': [dp.to_dict() for dp in dep_projects],
            'all_packages': sorted({package.name for dp in dep_projects for package in dp.packages}),
        }, {
            'status': status,
            'scan_stats': scan_stats,
            'error': 'dependency signals were found but no dependency project was materialized'
            if status == 'degraded' else '',
        }

    for distro, group in sorted(grouped_projects.items()):
        configure_metadata(distro)
        _log(f'Processing {len(group)} projects with Debian metadata: {distro}')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, p): p for p in group}
            for future in as_completed(futures):
                proj_path = futures[future]
                try:
                    proj_path, result, project_status = future.result()
                    per_project[proj_path] = result
                    per_project_status[proj_path] = project_status
                    done += 1
                    n_files = int((project_status or {}).get('scan_stats', {}).get('file_match_count') or 0)
                    n_deps = len(result.get('all_packages') or []) if isinstance(result, dict) else 0
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    _log(f'[{done}/{total}] {os.path.basename(proj_path)}: '
                         f'{n_files} files, {n_deps} deps ({rate:.1f} proj/s)')
                except Exception as e:
                    done += 1
                    _log(f'[{done}/{total}] ERROR {os.path.basename(proj_path)}: {e}')
                    per_project[futures[future]] = {}
                    per_project_status[futures[future]] = {
                        'status': 'error',
                        'error': f'{type(e).__name__}: {e}',
                        'stage': 'dependency_materialization',
                    }

                if args.save_interval > 0 and done % args.save_interval == 0:
                    _save_output(
                        args.output,
                        args,
                        per_project,
                        per_project_metadata,
                        per_project_constraint_details,
                        baseline_debian_version,
                        enabled_sources=enabled_sources,
                        per_project_status=per_project_status,
                    )

    _save_output(
        args.output,
        args,
        per_project,
        per_project_metadata,
        per_project_constraint_details,
        baseline_debian_version,
        enabled_sources=enabled_sources,
        per_project_status=per_project_status,
    )

    # Save artifact cache
    if _artifact_cache is not None:
        _artifact_cache.save()
        stats = _artifact_cache.stats()
        _log(f'  Cache: {stats["total"]} entries, {stats["usable"]} usable')

    # Report LLM usage
    from .llm_mapper import get_llm_usage
    llm_usage = get_llm_usage()
    if llm_usage["calls"] > 0:
        _log(f'  LLM usage: {llm_usage["calls"]} calls, '
             f'{llm_usage["input_tokens"]+llm_usage["output_tokens"]} tokens, '
             f'{llm_usage["elapsed"]:.1f}s')
        usage_path = args.output.replace('.json', '_llm_usage.json')
        with open(usage_path, 'w') as f:
            json.dump(llm_usage, f, indent=2)

    elapsed = time.time() - t0
    all_pkgs = collect_all_packages(per_project)
    _log(f'\nDone. {len(per_project)} projects, {len(all_pkgs)} unique packages, {elapsed:.1f}s')
    if any(
        str(item.get('status') or '') in {'error', 'degraded'}
        for item in per_project_status.values()
        if isinstance(item, dict)
    ):
        raise SystemExit(2)


def _save_output(
    output_path: str,
    args,
    per_project: dict,
    per_project_metadata: dict | None = None,
    per_project_constraint_details: dict | None = None,
    requested_debian_version: str | None = None,
    enabled_sources: set[str] | None = None,
    per_project_status: dict | None = None,
):
    requested_debian_version = requested_debian_version or DEBIAN_VERSION
    metadata = per_project_metadata or {}
    constraint_details = per_project_constraint_details or {}
    details_path = _constraint_details_path(output_path)
    for meta in metadata.values():
        if isinstance(meta, dict):
            meta['constraint_details_path'] = details_path
    enabled_sources = set(enabled_sources or ALL_SOURCE_FLAGS)
    selected_versions = sorted({
        str(meta.get('selected_debian_version'))
        for meta in metadata.values()
        if isinstance(meta, dict) and meta.get('selected_debian_version')
    })
    output_debian_version = selected_versions[0] if len(selected_versions) == 1 else requested_debian_version
    statuses = per_project_status or {}
    status_values = {
        str(item.get('status') or '')
        for item in statuses.values()
        if isinstance(item, dict)
    }
    aggregate_status = (
        'error' if status_values == {'error'}
        else 'partial' if status_values & {'error', 'degraded'}
        else 'ok'
    )
    output = {
        'schema_version': EXTRACTOR_SCHEMA_VERSION,
        'status': aggregate_status,
        'run_config': {
            'model': args.model,
            'base_url': args.base_url,
            'context_lines': args.context_lines,
            'save_interval': args.save_interval,
            'llm_timeout': args.llm_timeout,
            'requested_debian_version': requested_debian_version,
            'debian_version': output_debian_version,
            'selected_debian_versions': selected_versions,
            'distro_selection': getattr(args, 'distro_selection', 'fixed'),
            'pkglist_path': str(pkglist_path(output_debian_version)),
            'provider_index_path': str(provider_index_path(output_debian_version)),
            'repository_profile_hash': metadata_profile_hash(output_debian_version),
            'method': 'enhanced_rule_llm',
            'experiment': args.experiment,
            'disable_llm': getattr(args, 'disable_llm', False),
            'source_mask': sorted(enabled_sources),
        },
        'per_project_metadata': metadata,
        'per_project_status': statuses,
        'per_project': per_project,
        'all_packages': collect_all_packages(per_project),
    }
    tmp = output_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    os.replace(tmp, output_path)

    details_output = {
        'run_config': {
            'requested_debian_version': requested_debian_version,
            'debian_version': output_debian_version,
            'selected_debian_versions': selected_versions,
            'distro_selection': getattr(args, 'distro_selection', 'fixed'),
        },
        'per_project_constraint_details': constraint_details,
    }
    details_tmp = details_path + '.tmp'
    with open(details_tmp, 'w') as f:
        json.dump(details_output, f, ensure_ascii=False, indent=2)
    os.replace(details_tmp, details_path)


def _constraint_details_path(output_path: str) -> str:
    if output_path.endswith('.json'):
        return output_path[:-5] + '_constraint_details.json'
    return output_path + '_constraint_details.json'


if __name__ == '__main__':
    main()
