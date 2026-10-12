"""Persistent build-plan state and compact prompt projections."""

from __future__ import annotations

import json
import os
import re
import hashlib
import shlex
from copy import deepcopy
from typing import Any, Optional

from diagnosis.diagnosis_signals import (
    COMPILER_POLICY_PATTERN,
    INFRASTRUCTURE_FAILURE_PATTERN,
    MISSING_HEADER_PATTERN,
    MISSING_TOOL_PATTERN,
    PACKAGE_MANAGER_FAILURE_PATTERN,
    TOOL_CONFIGURATION_PATTERN,
    VERSION_REQUIREMENT_PATTERN,
    extract_missing_headers,
    extract_missing_tools,
    has_compiler_policy,
    has_tool_configuration_failure,
    is_wrong_entry_or_target,
)
from core.debian_sysuse import normalize_sysuse
from core.repository_environment import best_satisfying_package_version


MAX_PROMPT_ITEMS = 6
BUILD_PLAN_SCHEMA_VERSION = 3


def build_initial_plan(
    project_info: dict[str, Any],
    current_sysuse: str,
    existing_plan: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create or refresh the persistent plan state from Phase 2 metadata."""
    plan = deepcopy(existing_plan or {})
    plan["schema_version"] = BUILD_PLAN_SCHEMA_VERSION
    plan.setdefault("selected", {})
    plan.setdefault("alternatives", {})
    plan.setdefault("dependency_state", {})
    plan.setdefault("dependency_inventory", {})
    plan.setdefault("attempts", [])
    plan.setdefault("pending_fix_plan", [])
    plan.setdefault("entrypoint_candidates", [])
    plan.setdefault("sysuse_candidates", [])
    plan.setdefault("system_dep_candidates", [])
    plan.setdefault("source_dep_candidates", [])
    plan.setdefault("bootstrap_candidates", [])
    plan.setdefault("generator_candidates", [])
    plan.setdefault("target_candidates", [])
    plan.setdefault("provider_evidence", [])
    plan.setdefault("repair_decisions", [])
    plan.setdefault("repair_outcomes", [])
    plan.setdefault("candidate_outcomes", [])
    plan.setdefault("repair_overrides", {})
    plan.setdefault("retired_system_packages", [])
    plan.setdefault("failed_attempts", [])
    plan.setdefault("next_action_hints", [])
    # Older persisted plans may predate stable candidate IDs. Normalize those
    # IDs before merging new source/provider evidence so empty IDs cannot
    # collapse unrelated candidates into one row.
    _ensure_candidate_ids(plan)

    _merge_by_key(
        plan["entrypoint_candidates"],
        [
            {
                "entry_id": str(item.get("entry_id") or "").strip(),
                "command": str((item.get("command_sequence") or [item.get("command") or ""])[0]).strip(),
                "command_sequence": _string_list(item.get("command_sequence")) or _string_list([item.get("command")]),
                "command_steps": deepcopy(item.get("command_steps") or []),
                "workdir": item.get("workdir") or project_info.get("build_workdir") or ".",
                "status": item.get("status") or "candidate",
                "source": item.get("source") or "phase1_candidate",
                "confidence": item.get("confidence") or "unknown",
                "path": item.get("entry_path") or item.get("path") or "",
                "evidence_paths": _string_list(item.get("evidence_paths")),
                "build_system": item.get("build_system") or item.get("family") or "unknown",
                "scope": item.get("scope") or "unknown",
                "root_evidence": _string_list(item.get("root_evidence")),
                "entry_preflight": deepcopy(item.get("entry_preflight") or {}),
                "rank": item.get("rank"),
                "rank_reason": item.get("rank_reason") or "",
                "eligible": bool(item.get("eligible", True)),
                "eligibility_reasons": _string_list(item.get("eligibility_reasons")),
                "purposes": _string_list(item.get("purposes")),
                "target_candidates": deepcopy(
                    item.get("target_candidates")
                    or item.get("build_target_candidates")
                    or []
                ),
            }
            for item in (project_info.get("build_entry_candidates") or [])
            if isinstance(item, dict)
            and (_string_list(item.get("command_sequence")) or str(item.get("command") or "").strip())
        ],
        "entry_id",
    )
    main_sequence = _string_list(project_info.get("main_build_plan"))
    if main_sequence and not project_info.get("build_entry_candidates"):
        _merge_by_key(
            plan["entrypoint_candidates"],
            [
                {
                    "entry_id": "entry_main_plan",
                    "command": main_sequence[0],
                    "command_sequence": main_sequence,
                    "workdir": project_info.get("build_workdir") or ".",
                    "status": "selected",
                    "source": "phase1",
                    "confidence": "main",
                    "path": project_info.get("build_file_path") or "",
                    "target_candidates": [],
                }
            ],
            "entry_id",
        )
    if not project_info.get("build_entry_candidates"):
        _merge_by_key(
            plan["entrypoint_candidates"],
            [
                {
                    "entry_id": str(item.get("entry_id") or _command_entry_id(item)).strip(),
                    "command": str(item.get("command") or "").strip(),
                    "command_sequence": _string_list(item.get("command_sequence")) or [str(item.get("command") or "").strip()],
                    "command_steps": deepcopy(item.get("command_steps") or []),
                    "workdir": project_info.get("build_workdir") or ".",
                    "status": "planned",
                    "source": item.get("source") or "phase1_candidate",
                    "confidence": item.get("confidence") or "unknown",
                    "path": item.get("path") or "",
                    "target_candidates": deepcopy(
                        item.get("target_candidates")
                        or item.get("build_target_candidates")
                        or []
                    ),
                }
                for item in (project_info.get("command_candidates") or [])
                if isinstance(item, dict) and str(item.get("command") or "").strip()
            ],
            "entry_id",
        )
    _merge_by_key(
        plan["system_dep_candidates"],
        [
            {"name": dep, "kind": "apt_package", "status": "planned", "source": "phase1"}
            for dep in _string_list(project_info.get("apt_install_plan"))
        ],
        "name",
    )
    _merge_by_key(
        plan["source_dep_candidates"],
        [
            {
                "candidate_id": _stable_candidate_id(
                    "source",
                    f"{str(dep.get('name') or '').strip()}\x1f{dep.get('repo_url') or dep.get('url') or dep.get('source_path') or dep.get('submodule_path') or ''}",
                ),
                "name": str(dep.get("name") or "").strip(),
                "kind": "source_dependency",
                # Entries in Phase 1's source_dep_plan are already part of
                # the executable baseline, not merely discovery hints.
                "status": "selected",
                "source": "phase1",
                "source_path": dep.get("source_path") or dep.get("submodule_path") or "",
                "container_source_path": dep.get("container_source_path") or dep.get("source_path") or dep.get("submodule_path") or "",
                "install_prefix": dep.get("install_prefix") or "",
                "build_order": _limit(dep.get("build_order"), 3),
                "env_exports": _limit(dep.get("env_exports"), 4),
                "setup_commands": _limit(dep.get("setup_commands"), 4),
                "fetch_commands": _limit(dep.get("fetch_commands"), 4),
                "repo_url": dep.get("repo_url") or dep.get("url") or "",
                "submodule_path": dep.get("submodule_path") or "",
                "metadata_path": dep.get("metadata_path") or "",
                "clone_status": dep.get("clone_status") or "",
                "apt_packages": _string_list(dep.get("apt_packages") or dep.get("apt_install_plan")),
            }
            for dep in (project_info.get("source_dep_plan") or [])
            if isinstance(dep, dict) and str(dep.get("name") or "").strip()
        ],
        "candidate_id",
    )
    _merge_by_key(
        plan["sysuse_candidates"],
        _sysuse_candidates(project_info.get("sysuse_candidate_plan"), current_sysuse),
        "sysuse",
    )
    _merge_by_key(
        plan["bootstrap_candidates"],
        [
            {"command": command, "status": "planned", "source": "repo_wrapper"}
            for command in _bootstrap_commands(project_info)
        ],
        "command",
    )
    _merge_by_key(
        plan["generator_candidates"],
        [
            {
                "generator_id": _stable_candidate_id("generator", command),
                "command": command,
                "status": "candidate",
                "source": "repo_wrapper",
            }
            for command in _bootstrap_commands(project_info)
        ],
        "generator_id",
    )
    _merge_by_key(
        plan["target_candidates"],
        _target_candidates(plan["entrypoint_candidates"]),
        "target_id",
    )
    _ensure_candidate_ids(plan)
    _set_current_sysuse(plan, current_sysuse)
    _refresh_selected_and_alternatives(plan, project_info, current_sysuse)
    _refresh_dependency_state(plan, project_info)
    if isinstance(project_info.get("dependency_inventory"), dict):
        plan["dependency_inventory"] = deepcopy(project_info["dependency_inventory"])
    return plan


def save_build_plan(plan: dict[str, Any], path: str) -> None:
    """Atomically persist a build plan for replay and audit."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def set_pending_fix_plan(plan: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
    updated = deepcopy(plan or {})
    updated["pending_fix_plan"] = deepcopy(actions or [])
    return updated


def select_build_entry_in_plan(plan: dict[str, Any], entry_id: str) -> dict[str, Any]:
    updated = deepcopy(plan or {})
    selected_entry = None
    for item in updated.get("entrypoint_candidates") or []:
        if str(item.get("entry_id") or "") == str(entry_id or ""):
            if item.get("eligible", True) and item.get("status") != "ineligible":
                item["status"] = "selected"
                selected_entry = item
        elif item.get("status") == "selected":
            item["status"] = "candidate"
    if selected_entry:
        updated.setdefault("selected", {})["build_entry_id"] = selected_entry.get("entry_id") or ""
        updated["selected"]["command_sequence"] = list(selected_entry.get("command_sequence") or [])
        updated["selected"]["command_steps"] = deepcopy(selected_entry.get("command_steps") or [])
        updated["selected"]["build_entry_path"] = selected_entry.get("path") or ""
        _refresh_entry_alternatives(updated)
    return updated


def record_build_attempt(plan: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(plan or {})
    attempts = list(updated.get("attempts") or [])
    attempts.append(deepcopy(record))
    updated["attempts"] = attempts[-32:]
    return updated


def register_provider_evidence(plan: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(plan or {})
    updated.setdefault("provider_evidence", [])
    evidence_id = str(evidence.get("evidence_id") or "").strip()
    if evidence_id:
        _merge_by_key(updated["provider_evidence"], [deepcopy(evidence)], "evidence_id")
    return updated


def rebind_environment(
    plan: dict[str, Any],
    *,
    sysuse: str,
    container_generation: int,
    packages: list[dict[str, Any]],
    repository_profile_hash: str,
) -> dict[str, Any]:
    """Invalidate environment facts and install the new required package closure."""
    updated = deepcopy(plan or {})
    _set_current_sysuse(updated, sysuse)
    updated.setdefault("selected", {})["repository_profile_hash"] = repository_profile_hash
    updated["provider_evidence"] = [
        item for item in updated.get("provider_evidence") or []
        if isinstance(item, dict)
        and (item.get("environment") or {}).get("container_generation") == container_generation
        and normalize_sysuse(str((item.get("environment") or {}).get("sysuse") or ""))
        == normalize_sysuse(sysuse)
    ]
    # Every package candidate is environment-bound.  Keeping a Phase 3
    # provider from the old suite here would let the next Phase 2 payload
    # silently reinstall an unverified package.  Unresolved/optional packages
    # are rediscovered by the current-suite probe instead of being copied
    # across the container boundary.
    retained: list[dict[str, Any]] = []
    retained.extend({
        "candidate_id": _stable_candidate_id("package", str(item.get("name") or "")),
        "name": str(item.get("name") or ""),
        "kind": "apt_package",
        # The resolver already proved this provider in the new suite.  It is
        # therefore part of the executable package set, not an unbound hint.
        "status": "selected",
        "source": "environment_rebind",
        "suite": sysuse,
        "version": item.get("version") or "",
        "source_repo": item.get("source_repo") or "",
        "dependency_id": item.get("dependency_id") or "",
    } for item in packages if item.get("name"))
    # Make the new suite's base closure visible before resolving packages
    # contributed by source dependencies; otherwise the resolver would see
    # stale selected candidates from the previous container generation.
    updated["system_dep_candidates"] = retained
    source_packages, unresolved_source_packages = _bound_source_dependency_packages(
        updated,
        sysuse,
    )
    retained.extend(source_packages)
    updated["system_dep_candidates"] = retained
    updated["unresolved_source_packages"] = unresolved_source_packages
    updated.setdefault("alternatives", {})["sysuses"] = [
        deepcopy(item) for item in updated.get("sysuse_candidates") or []
        if normalize_sysuse(str(item.get("sysuse") or "")) != normalize_sysuse(sysuse)
    ]
    return updated


def _bound_source_dependency_packages(
    plan: dict[str, Any],
    sysuse: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve explicitly declared source-dependency packages in one suite."""
    selected_names = {
        str(item.get("name") or "").strip().lower()
        for item in plan.get("system_dep_candidates") or []
        if isinstance(item, dict) and item.get("status") == "selected"
    }
    bound: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for dependency in plan.get("source_dep_candidates") or []:
        if not isinstance(dependency, dict) or dependency.get("status") != "selected":
            continue
        dependency_id = str(dependency.get("candidate_id") or "")
        for raw_name in _string_list(dependency.get("apt_packages")):
            package = raw_name.strip().lower()
            if not package or package in selected_names:
                continue
            version = best_satisfying_package_version(package, sysuse)
            if not version:
                unresolved.append({
                    "package": package,
                    "source_dependency_id": dependency_id,
                    "suite": normalize_sysuse(sysuse),
                })
                continue
            selected_names.add(package)
            bound.append({
                "candidate_id": _stable_candidate_id("package", package),
                "name": package,
                "kind": "apt_package",
                "status": "selected",
                "source": "source_dependency",
                "suite": normalize_sysuse(sysuse),
                "version": str(version.get("version") or ""),
                "source_repo": str(version.get("source_repo") or ""),
                "component": str(version.get("component") or ""),
                "dependency_id": dependency_id,
            })
    return bound, unresolved


def record_repair_decision(
    plan: dict[str, Any],
    decision: dict[str, Any],
    *,
    state: str = "selected",
) -> dict[str, Any]:
    updated = deepcopy(plan or {})
    record = deepcopy(decision or {})
    if not record:
        return updated
    records = list(updated.get("repair_decisions") or [])
    fingerprint = str(record.get("fingerprint") or "")
    existing = next(
        (
            item for item in reversed(records)
            if fingerprint and str(item.get("fingerprint") or "") == fingerprint
        ),
        None,
    )
    if existing is not None:
        existing.update(record)
        existing["state"] = state
    else:
        record["state"] = state
        records.append(record)
    updated["repair_decisions"] = records[-24:]
    return updated


def record_repair_outcome(
    plan: dict[str, Any],
    *,
    fingerprint: str,
    failure_identity: str,
    outcome: str,
    next_failure_identity: str = "",
    operation: Optional[dict[str, Any]] = None,
    details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    updated = deepcopy(plan or {})
    records = list(updated.get("repair_outcomes") or [])
    records.append({
        "fingerprint": str(fingerprint or ""),
        "failure_identity": str(failure_identity or ""),
        "outcome": str(outcome or "unknown"),
        "next_failure_identity": str(next_failure_identity or ""),
        "operation": deepcopy(operation or {}),
        "details": deepcopy(details or {}),
    })
    updated["repair_outcomes"] = records[-32:]
    return updated


def record_candidate_outcome(
    plan: dict[str, Any],
    *,
    operation: dict[str, Any],
    outcome: str,
    failure_identity: str = "",
) -> dict[str, Any]:
    """Record candidate-level lifecycle without retiring its Matrix action."""
    updated = deepcopy(plan or {})
    record = {
        "operation": deepcopy(operation or {}),
        "outcome": str(outcome or "unknown"),
        "failure_identity": str(failure_identity or ""),
    }
    updated["candidate_outcomes"] = list(updated.get("candidate_outcomes") or [])[-31:] + [record]
    return updated


def apply_repair_operation_to_plan(
    plan: dict[str, Any],
    operation: dict[str, Any],
) -> dict[str, Any]:
    """Apply one validated typed operation to persistent controller state."""
    updated = deepcopy(plan or {})
    kind = str(operation.get("kind") or "")
    if kind == "system_package_selection":
        package = str(operation.get("package") or "")
        candidates = updated.setdefault("system_dep_candidates", [])
        _merge_by_key(candidates, [{
            "candidate_id": _stable_candidate_id("package", package),
            "name": package,
            "kind": "apt_package",
            "status": "selected",
            "source": "provider_evidence",
            "provider_evidence_id": operation.get("provider_evidence_id") or "",
        }], "name")
        # Packages form a closure. Selecting a newly proven package must not
        # demote unrelated providers already bound for this environment.
        for item in candidates:
            if isinstance(item, dict) and str(item.get("name") or "") == package:
                item["status"] = "selected"
    elif kind == "source_dependency_selection":
        source_id = str(operation.get("source_dependency_id") or "")
        selected_source = next(
            (
                item for item in updated.get("source_dep_candidates") or []
                if isinstance(item, dict)
                and source_id in {
                    str(item.get("candidate_id") or ""),
                    str(item.get("source_dep_id") or ""),
                    str(item.get("name") or ""),
                }
                and item.get("status") not in {"failed", "exhausted", "ineligible"}
            ),
            None,
        )
        # Source dependencies form a set, not a single-choice provider
        # family. Selecting one Phase 3 candidate must not demote the
        # executable Phase 1 dependencies already in the baseline plan.
        if selected_source:
            selected_source["status"] = "selected"
        if selected_source:
            updated.setdefault("selected", {})["source_dependency_id"] = selected_source.get("candidate_id") or ""
    elif kind == "generator_selection":
        _select_candidate(updated.get("generator_candidates") or [], operation.get("generator_id"), ("generator_id", "candidate_id"))
        updated.setdefault("selected", {})["generator_id"] = operation.get("generator_id") or ""
    elif kind == "build_entry_selection":
        return select_build_entry_in_plan(updated, str(operation.get("entry_id") or ""))
    elif kind == "build_target_selection":
        selected_target = _select_candidate(updated.get("target_candidates") or [], operation.get("target_id"), ("target_id", "candidate_id"))
        if selected_target and selected_target.get("entry_id"):
            updated = select_build_entry_in_plan(updated, str(selected_target.get("entry_id") or ""))
        if selected_target and selected_target.get("command_sequence"):
            updated.setdefault("selected", {})["command_sequence"] = list(selected_target.get("command_sequence") or [])
        updated.setdefault("selected", {})["target_id"] = operation.get("target_id") or ""
    elif kind == "system_release_selection":
        release_id = str(operation.get("release_id") or "")
        selected = _select_candidate(updated.get("sysuse_candidates") or [], release_id, ("candidate_id", "sysuse"))
        if selected:
            updated.setdefault("selected", {})["requested_sysuse"] = selected.get("sysuse") or ""
    elif kind == "build_plan_patch":
        changes = deepcopy(operation.get("changes") or {})
        updated.setdefault("repair_overrides", {}).update(changes)
    elif kind == "tool_wrapper":
        wrappers = updated.setdefault("repair_overrides", {}).setdefault("tool_wrappers", {})
        wrappers[str(operation.get("tool") or "")] = deepcopy(operation.get("wrapper") or {})
    return updated


def extract_failure_frame(error_text: str, failed_command: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Compatibility entry point for callers without command-record slices."""
    record = dict(failed_command or {})
    record.setdefault("command", "")
    record.setdefault("status", "failed")
    record.setdefault("exit_code", 1)
    record["stderr"] = error_text or ""
    return extract_failure_frame_from_records([record])


def _extract_failure_signals(text: str, failed_command: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    dependency_names = _unique(
        next((value for value in match.groups() if value), "")
        for match in _DEPENDENCY_RE.finditer(text)
    )
    frame = {
        "error_class": "unknown",
        "failed_command": _compact_failed_command(failed_command),
        "missing_headers": extract_missing_headers(text),
        "missing_libraries": _unique(_LIB_RE.findall(text)),
        "missing_pkg_configs": _unique(
            [m.group(1) or m.group(2) for m in _PKG_RE.finditer(text)]
            + dependency_names
        ),
        "missing_cmake_packages": _unique(
            next((value for value in match.groups() if value), "")
            for match in _CMAKE_RE.finditer(text)
        ),
        "missing_paths": _unique(_PATH_RE.findall(text)),
        "missing_tools": extract_missing_tools(text),
        "toolchain_errors": _toolchain_signals(text),
        "bootstrap_errors": _bootstrap_signals(text),
        "evidence": _evidence_lines(text),
    }
    frame["diagnose_tags"] = _diagnose_tags(text, frame, failed_command)
    frame["error_class"] = _classify_failure(frame)
    return frame


def extract_failure_frame_from_records(
    command_records: list[dict[str, Any]],
    *,
    fallback_text: str = "",
    failed_command: Optional[dict[str, Any]] = None,
    selected_command_steps: Optional[list[dict[str, Any]]] = None,
    attempt_start: int = 0,
    attempt_end: Optional[int] = None,
) -> dict[str, Any]:
    """Select a primary diagnostic fragment from the current attempt."""
    records = list(command_records or [])
    if attempt_start or attempt_end is not None:
        records = records[attempt_start:attempt_end]
    if not records:
        fallback_record = dict(failed_command or {})
        fallback_record.setdefault("command", "")
        fallback_record.setdefault("status", "failed")
        fallback_record.setdefault("exit_code", 1)
        fallback_record["stderr"] = fallback_text or ""
        records = [fallback_record]

    all_candidates: list[dict[str, Any]] = []
    candidates_by_record: dict[int, list[dict[str, Any]]] = {}
    for record_index, record in enumerate(records):
        record_sequence_index, _ = _match_command_step(record, selected_command_steps or [])
        fragments = _extract_failure_fragments(record, record_index, record_sequence_index)
        candidates_by_record[record_index] = fragments
        all_candidates.extend(fragments)

    failed_indices = []
    for index, record in enumerate(records):
        if not _record_looks_failed(record):
            continue
        command_key = _command_record_key(record)
        superseded = bool(command_key) and any(
            _command_record_key(later) == command_key and not _record_looks_failed(later)
            and str(later.get("status") or "").lower() in {"success", "done"}
            for later in records[index + 1:]
        )
        if not superseded:
            failed_indices.append(index)
    if not failed_indices:
        primary_record = dict(failed_command or records[-1])
        primary_index = max(0, len(records) - 1)
        sequence_index, stage = _match_command_step(primary_record, selected_command_steps or [])
        return {
            "attempt_status": "incomplete_attempt",
            "error_class": "incomplete_attempt",
            "primary_error_class": "incomplete_attempt",
            "attempt_record_count": len(records),
            "primary_command": _compact_command_record(primary_record),
            "primary_failure": {},
            "primary_failure_text": "",
            "primary_failure_index": primary_index,
            "primary_command_index": primary_index,
            "failure_candidates": all_candidates[-12:],
            "supporting_evidence": [],
            "evidence_bundle": [],
            "evidence_requests": ["verify build artifact or compilation/link postcondition"],
            "failure_confidence": "none",
        }

    # Failure fragments are ranked only inside the newest unresolved causal
    # record. Cross-record scoring used to resurrect older, more verbose
    # diagnostics after the build had already progressed.
    primary_index = failed_indices[-1]
    primary_record = records[primary_index]
    sequence_index, stage = _match_command_step(primary_record, selected_command_steps or [])

    primary_candidates = candidates_by_record.get(primary_index, [])
    primary_failure = _select_primary_fragment(primary_candidates)
    if not primary_failure:
        fallback_fragments = _extract_text_fragments(
            fallback_text,
            stream="agent_output",
            command=str(primary_record.get("command") or ""),
            record_index=primary_index,
            sequence_index=sequence_index,
        )
        all_candidates.extend(fallback_fragments)
        primary_failure = _select_primary_fragment(fallback_fragments)
    if not primary_failure:
        raw_text = _record_failure_text(primary_record) or fallback_text
        indexed_lines = [
            (index, line.strip())
            for index, line in enumerate(raw_text.splitlines())
            if line.strip() and not _is_noise_line(line)
        ]
        tail = indexed_lines[-8:]
        fallback_diagnostic = "\n".join(line for _, line in tail) or "unknown failure"
        fallback_line_index = tail[0][0] if tail else max(0, len(raw_text.splitlines()) - 1)
        primary_failure = _make_fragment(
            fallback_diagnostic,
            kind="other",
            stream="fallback",
            line_index=fallback_line_index,
            line_count=max(1, len(raw_text.splitlines())),
            command=str(primary_record.get("command") or ""),
            record_index=primary_index,
            sequence_index=sequence_index,
        )
        all_candidates.append(primary_failure)

    primary_record_candidates = [
        item for item in all_candidates
        if item.get("record_index") == primary_index
    ]
    supporting_evidence = _supporting_fragments(primary_record_candidates, primary_failure)
    primary_command = _compact_command_record(primary_record)
    primary_command.update({
        "record_index": primary_index,
        "sequence_index": sequence_index,
        "stage": stage,
    })
    signal_text = primary_failure.get("text") or ""
    context_text = "\n".join(
        part for part in (
            _record_failure_text(primary_record),
            "\n".join(supporting_evidence),
        )
        if part
    )
    primary_signals = _extract_failure_signals(signal_text, primary_command)
    frame = _extract_failure_signals(context_text or signal_text, primary_command)
    frame.update({
        "attempt_status": "failed",
        "attempt_record_count": len(records),
        "primary_command": primary_command,
        "primary_failure": primary_failure,
        "primary_failure_text": signal_text[:2000],
        "primary_failure_index": primary_index,
        "primary_command_index": primary_index,
        "failure_candidates": all_candidates[-12:],
        "supporting_evidence": supporting_evidence[:8],
        "evidence_bundle": _build_evidence_bundle(
            primary_failure,
            primary_record,
            supporting_evidence,
        ),
        "primary_signals": {
            key: deepcopy(primary_signals.get(key) or [])
            for key in (
                "missing_headers",
                "missing_libraries",
                "missing_pkg_configs",
                "missing_cmake_packages",
                "missing_paths",
                "missing_tools",
                "toolchain_errors",
                "bootstrap_errors",
            )
        },
        "primary_diagnose_tags": deepcopy(primary_signals.get("diagnose_tags") or []),
        "diagnostic_spans": deepcopy(all_candidates[-20:]),
        "failure_confidence": _failure_confidence(primary_failure, all_candidates),
        "evidence_requests": _failure_evidence_requests(primary_failure, all_candidates),
    })
    frame["failure_signature"] = _failure_signature(primary_command, primary_failure)
    frame["primary_error_class"] = _classify_failure(primary_signals)
    return frame


def render_failure_frame_text(frame: dict[str, Any], fallback_text: str = "") -> str:
    primary = str(frame.get("primary_failure_text") or "").strip()
    if not primary:
        primary = str((frame.get("primary_failure") or {}).get("text") or "").strip()
    bundle = [
        str(item.get("text") or "").strip()
        for item in frame.get("evidence_bundle") or []
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]
    if bundle:
        return "\n".join(bundle)[:4000]
    supporting = [str(item).strip() for item in frame.get("supporting_evidence") or [] if str(item).strip()]
    if not primary:
        return fallback_text[-4000:] if fallback_text else ""
    return "\n".join([primary] + supporting)[:4000]


def update_build_plan_from_failure(
    plan: dict[str, Any],
    failure_frame: dict[str, Any],
    failed_command: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return plan state updated with current-attempt signals."""
    updated = deepcopy(plan or {})
    updated.setdefault("system_dep_candidates", [])
    updated.setdefault("source_dep_candidates", [])
    updated.setdefault("failed_attempts", [])
    updated.setdefault("next_action_hints", [])
    updated.setdefault("retired_system_packages", [])

    for name in failure_frame.get("missing_pkg_configs") or []:
        _merge_by_key(
            updated["system_dep_candidates"],
            [{"name": name, "kind": "pkg_config", "status": "unresolved", "source": "current_failure"}],
            "name",
        )
    for name in failure_frame.get("missing_cmake_packages") or []:
        _merge_by_key(
            updated["system_dep_candidates"],
            [{"name": name, "kind": "cmake_package", "status": "unresolved", "source": "current_failure"}],
            "name",
        )
    for name in failure_frame.get("missing_libraries") or []:
        _merge_by_key(
            updated["system_dep_candidates"],
            [{"name": name, "kind": "link_library", "status": "unresolved", "source": "current_failure"}],
            "name",
        )
    for name in failure_frame.get("missing_tools") or []:
        _merge_by_key(
            updated["system_dep_candidates"],
            [{"name": name, "kind": "build_tool", "status": "unresolved", "source": "current_failure"}],
            "name",
        )

    for package in _failed_package_names(failure_frame):
        if package not in updated["retired_system_packages"]:
            updated["retired_system_packages"].append(package)
        for candidate in updated.get("system_dep_candidates") or []:
            if str(candidate.get("name") or "").lower() == package.lower():
                candidate["status"] = "failed"

    source_candidates = _source_candidates_from_failure(failure_frame, failed_command)
    if source_candidates:
        _merge_by_key(updated["source_dep_candidates"], source_candidates, "candidate_id")
    _ensure_candidate_ids(updated)

    failed_record = {
        "command": (failed_command or failure_frame.get("primary_command") or failure_frame.get("failed_command") or {}).get("command", ""),
        "error_class": failure_frame.get("primary_error_class") or failure_frame.get("error_class") or "unknown",
        "aggregate_error_class": failure_frame.get("error_class") or "unknown",
        "primary_error_class": failure_frame.get("primary_error_class") or failure_frame.get("error_class") or "unknown",
        "failure_signature": failure_frame.get("failure_signature") or "",
        "signals": _primary_signals(failure_frame.get("primary_signals") or failure_frame),
        "diagnose_tags": _limit(failure_frame.get("primary_diagnose_tags") or failure_frame.get("diagnose_tags"), 4),
        # Keep diagnosis text separate from command-record metadata.  The rich
        # fragment remains available in the current failure frame, while the
        # persistent compact plan stores only stable diagnostic fields.
        "primary_failure_text": str(
            failure_frame.get("primary_failure_text")
            or (failure_frame.get("primary_failure") or {}).get("text")
            or ""
        )[:2000],
        "primary_failure_kind": str(
            (failure_frame.get("primary_failure") or {}).get("kind") or "unknown"
        ),
        "primary_failure_stream": str(
            (failure_frame.get("primary_failure") or {}).get("stream") or "unknown"
        ),
        "supporting_evidence": _limit(failure_frame.get("supporting_evidence"), 4),
    }
    if failed_record["command"] or failed_record["signals"]:
        updated["failed_attempts"] = (updated["failed_attempts"] + [failed_record])[-8:]
        _mark_entrypoint_failed(updated, failed_record["command"])
        _refresh_entry_alternatives(updated)

    hints = _next_hints(failure_frame)
    for hint in hints:
        if hint not in updated["next_action_hints"]:
            updated["next_action_hints"].append(hint)
    updated["next_action_hints"] = updated["next_action_hints"][-8:]
    return updated


def add_source_dependency_candidate(plan: dict[str, Any], source_dependency: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(plan or {})
    updated.setdefault("source_dep_candidates", [])
    name = str(source_dependency.get("name") or "").strip()
    if not name:
        return updated
    candidate = {
        "candidate_id": _stable_candidate_id(
            "source",
            f"{name}\x1f{source_dependency.get('repo_url') or source_dependency.get('source_path') or source_dependency.get('submodule_path') or ''}",
        ),
        "name": name,
        "kind": "source_dependency",
        "status": "registered",
        "source": "phase3",
        "repo_url": source_dependency.get("repo_url") or source_dependency.get("url") or "",
        "source_path": source_dependency.get("source_path") or source_dependency.get("submodule_path") or "",
        "submodule_path": source_dependency.get("submodule_path") or "",
        "container_source_path": source_dependency.get("container_source_path") or source_dependency.get("source_path") or source_dependency.get("submodule_path") or "",
        "install_prefix": source_dependency.get("install_prefix") or "",
        "build_order": _limit(source_dependency.get("build_order"), 6),
        "setup_commands": _limit(source_dependency.get("setup_commands"), 6),
        "fetch_commands": _limit(source_dependency.get("fetch_commands"), 6),
        "env_exports": _limit(source_dependency.get("env_exports"), 6),
        "clone_status": source_dependency.get("clone_status") or "",
        "metadata_path": source_dependency.get("metadata_path") or "",
        "apt_packages": _string_list(
            source_dependency.get("apt_packages")
            or source_dependency.get("apt_install_plan")
        ),
    }
    _merge_by_key(
        updated["source_dep_candidates"],
        [candidate],
        "candidate_id",
    )
    return updated


def compact_build_plan_for_prompt(plan: dict[str, Any], max_items: int = MAX_PROMPT_ITEMS) -> dict[str, Any]:
    """Short plan view for project_info/build_context prompts."""
    return {
        "schema_version": plan.get("schema_version") or 1,
        "selected": deepcopy(plan.get("selected") or {}),
        "alternatives": {
            "build_entries": _head_limit((plan.get("alternatives") or {}).get("build_entries"), max_items),
            "sysuses": _head_limit((plan.get("alternatives") or {}).get("sysuses"), max_items),
        },
        "entrypoints": _ordered_candidate_limit(plan.get("entrypoint_candidates"), max_items),
        "sysuse": _ordered_candidate_limit(plan.get("sysuse_candidates"), max_items),
        "system_deps": _limit(plan.get("system_dep_candidates"), max_items),
        "source_deps": _limit(plan.get("source_dep_candidates"), max_items),
        "generators": _limit(plan.get("generator_candidates"), max_items),
        "targets": _limit(plan.get("target_candidates"), max_items),
        "provider_evidence": _limit(plan.get("provider_evidence"), max_items),
        "dependency_state": {
            "projects": _limit((plan.get("dependency_state") or {}).get("projects"), max_items),
            "unresolved_clues": _limit((plan.get("dependency_state") or {}).get("unresolved_clues"), max_items),
        },
        "dependency_inventory": {
            "schema_version": (plan.get("dependency_inventory") or {}).get("schema_version", 1),
            "dependencies": _limit(
                (plan.get("dependency_inventory") or {}).get("dependencies"),
                max_items,
            ),
        },
        "bootstrap": _limit(plan.get("bootstrap_candidates"), max_items),
        "failed_attempts": _limit(plan.get("failed_attempts"), max_items),
        "attempts": _limit(plan.get("attempts"), max_items),
        "pending_fix_plan": _limit(plan.get("pending_fix_plan"), max_items),
        "repair_outcomes": _limit(plan.get("repair_outcomes"), max_items),
        "repair_overrides": deepcopy(plan.get("repair_overrides") or {}),
        "retired_system_packages": _limit(plan.get("retired_system_packages"), max_items),
        "next_hints": _limit(plan.get("next_action_hints"), max_items),
    }


def compact_failure_frame_for_prompt(frame: dict[str, Any], max_items: int = MAX_PROMPT_ITEMS) -> dict[str, Any]:
    if not frame:
        return {}
    primary_failure = frame.get("primary_failure") or {}
    return {
        "error_class": frame.get("error_class") or "unknown",
        "primary_error_class": frame.get("primary_error_class") or frame.get("error_class") or "unknown",
        "failed_command": frame.get("failed_command") or {},
        "primary_command": deepcopy(frame.get("primary_command") or {}),
        "primary_failure_text": frame.get("primary_failure_text") or primary_failure.get("text") or "",
        "primary_failure_kind": primary_failure.get("kind") or "unknown",
        "primary_failure_stream": primary_failure.get("stream") or "unknown",
        "primary_failure_index": frame.get("primary_failure_index") if frame.get("primary_failure_index") is not None else 0,
        "failure_signature": frame.get("failure_signature") or "",
        "primary_signals": deepcopy(frame.get("primary_signals") or {}),
        "primary_diagnose_tags": _limit(frame.get("primary_diagnose_tags"), max_items),
        "missing_headers": _limit(frame.get("missing_headers"), max_items),
        "missing_libraries": _limit(frame.get("missing_libraries"), max_items),
        "missing_pkg_configs": _limit(frame.get("missing_pkg_configs"), max_items),
        "missing_cmake_packages": _limit(frame.get("missing_cmake_packages"), max_items),
        "missing_paths": _limit(frame.get("missing_paths"), max_items),
        "missing_tools": _limit(frame.get("missing_tools"), max_items),
        "toolchain_errors": _limit(frame.get("toolchain_errors"), max_items),
        "bootstrap_errors": _limit(frame.get("bootstrap_errors"), max_items),
        "diagnose_tags": _limit(frame.get("diagnose_tags"), max_items),
        "supporting_evidence": _limit(frame.get("supporting_evidence"), max_items),
        "evidence_bundle": _limit(frame.get("evidence_bundle"), max_items),
        "failure_candidates": _limit(frame.get("failure_candidates"), max_items),
        "evidence": _limit(frame.get("evidence"), max_items),
    }


def compact_attempt_history(history: list[dict[str, Any]], max_items: int = 6) -> list[dict[str, Any]]:
    return _limit(history, max_items)


def make_attempt_history_record(
    *,
    solution_kind: str,
    result: str,
    hypothesis: str = "",
    failed_command: Optional[dict[str, Any]] = None,
    sysuse: str = "",
    action_signature: str = "",
) -> dict[str, Any]:
    return {
        "kind": solution_kind,
        "result": result,
        "hypothesis": hypothesis or "",
        "command": (failed_command or {}).get("command", ""),
        "sysuse": sysuse or "",
        "action_signature": action_signature or "",
    }


_LIB_RE = re.compile(r"(?:cannot find|unable to find)\s+-l([A-Za-z0-9_+.-]+)", re.I)
_PKG_RE = re.compile(
    r"No package '([^']+)' found|Package ['\"]([^'\"]+)['\"](?:,[^\n]*)?(?:\s+was)? not found",
    re.I,
)
_DEPENDENCY_RE = re.compile(
    r"Dependency\s+[\"']([^\"']+)[\"']\s+(?:was\s+)?not found|"
    r"do not have the\s+([A-Za-z0-9_.+-]+)\s+development package|"
    r"Could not find\s+([A-Za-z0-9_.+-]+)\s+include director(?:y|ies)|"
    r"Unable to find development files for\s+([A-Za-z0-9_.+-]+)|"
    r"Checking for\s+[\"']?([A-Za-z0-9_.+-]+)[\"']?(?:\s*[<>=][^:\n]*)?\s*:\s*not found|"
    r"configure:\s*error:\s*(?:\*+\s*)?([A-Za-z0-9_.+-]+)(?:\s+(?:library|configuration script))?\s+not found|"
    r"(?:error:\s*)?([A-Za-z0-9_.+-]+)\s+not found on system or in local libs|"
    r"Could not find prerequisite package\s+[\"']([A-Za-z0-9_.+-]+)[\"']|"
    r"\b([A-Za-z0-9_.+-]+)\s+could not be found automatically|"
    r"configure:\s*error:\s*(?:\*+\s*)?[^\n]*\brequires\s+([A-Za-z0-9_.+-]+)",
    re.I,
)
_CMAKE_RE = re.compile(
    r"Could not find a package configuration file provided by\s+\"([^\"]+)\"|"
    r"Could not find\s+([A-Za-z0-9_.+-]+)Config\.cmake|"
    r"Could NOT find\s+([A-Za-z0-9_.+-]+)(?:\s|$)",
    re.I,
)
_PATH_RE = re.compile(r"(?:No such file or directory|cannot find source file):\s*([A-Za-z0-9_./+-]+)", re.I)


def _classify_failure(frame: dict[str, Any]) -> str:
    tags = set(frame.get("diagnose_tags") or [])
    if "runtime_package_conflict" in tags or "package_not_found" in tags:
        return "package_install_failure"
    if frame.get("missing_tools"):
        return "missing_build_tool"
    if frame.get("bootstrap_errors"):
        return "bootstrap"
    if frame.get("missing_paths") and not (frame.get("missing_headers") or frame.get("missing_libraries")):
        return "missing_source_path"
    if frame.get("missing_pkg_configs") or frame.get("missing_cmake_packages"):
        return "missing_package_metadata"
    if frame.get("missing_headers") or frame.get("missing_libraries"):
        return "missing_dependency"
    if frame.get("toolchain_errors") or _has_linker_signals(frame):
        return "toolchain_or_api"
    return "unknown"


def _source_candidates_from_failure(
    frame: dict[str, Any],
    failed_command: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    command = str((failed_command or {}).get("command") or "")
    source_paths = list(frame.get("missing_paths") or []) + list(frame.get("missing_headers") or [])
    text = " ".join([command] + source_paths)
    if not re.search(r"(?:^|/|\b)(third_party|3rdparty|deps|vendor|external|submodules?)(?:/|\b)", text, re.I):
        return []
    names = []
    for path in source_paths:
        parts = [p for p in str(path).split("/") if p and p not in {".", ".."}]
        if parts:
            names.append(parts[0] if parts[0] not in {"third_party", "3rdparty", "deps", "vendor", "external"} else (parts[1] if len(parts) > 1 else parts[0]))
    if not names:
        names = ["external_source_dependency"]
    return [
        {
            "candidate_id": _stable_candidate_id("source", f"{name}\x1f"),
            "name": name,
            "kind": "source_dependency_signal",
            "status": "unresolved",
            "source": "current_failure",
            "source_path": "",
        }
        for name in _unique(names)
    ]


def _failed_package_names(frame: dict[str, Any]) -> list[str]:
    text = "\n".join(
        [str(frame.get("primary_failure_text") or "")]
        + [str(item) for item in frame.get("supporting_evidence") or []]
        + [str(item) for item in frame.get("evidence") or []]
    )
    patterns = (
        r"Unable to locate package\s+([a-z0-9][a-z0-9+.-]*(?::[a-z0-9-]+)?)",
        r"Package\s+['`]?([a-z0-9][a-z0-9+.-]*(?::[a-z0-9-]+)?)['`]?(?:\s+has no installation candidate|\s+is not available)",
    )
    return _unique(
        match.group(1)
        for pattern in patterns
        for match in re.finditer(pattern, text, re.I)
    )


def _toolchain_signals(text: str) -> list[str]:
    patterns = [
        "implicit declaration",
        "no member named",
        "multiple definition",
        "not declared in this scope",
        "incomplete type",
        "storage size of",
        "undefined reference",
        "cannot find -l",
        "too few arguments",
        "too many arguments",
        "-Werror",
    ]
    lowered = text.lower()
    signals = [p for p in patterns if p.lower() in lowered]
    if re.search(VERSION_REQUIREMENT_PATTERN, text, re.I):
        signals.append("version requirement")
    if has_compiler_policy(text):
        signals.append("compiler policy")
    if has_tool_configuration_failure(text):
        signals.append("toolchain option")
    return _unique(signals)


def _has_linker_signals(frame: dict[str, Any]) -> bool:
    evidence = "\n".join(str(line) for line in frame.get("evidence") or [])
    return bool(re.search(r"undefined reference|multiple definition|cannot find -l|ld returned|collect2:", evidence, re.I))


def _record_looks_failed(record: dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False
    status = str(record.get("status") or "").lower()
    if status in {"failed", "timeout", "error"}:
        return True
    if status in {"control", "skipped", "blocked"}:
        return False
    exit_code = record.get("exit_code")
    if exit_code in (0, "0", None, "None", "none"):
        return False
    try:
        return int(exit_code) != 0
    except (TypeError, ValueError):
        return bool(str(exit_code).strip())


def _command_record_key(record: dict[str, Any]) -> tuple[str, str]:
    command = re.sub(r"\s+", " ", str(record.get("command") or record.get("input") or "").strip())
    workdir = os.path.normpath(str(record.get("workdir") or record.get("cwd") or "."))
    return (command, workdir) if command else ("", "")


def _record_failure_text(record: Optional[dict[str, Any]]) -> str:
    if not record:
        return ""
    pieces = []
    for key in ("stderr", "output", "stdout", "stderr_tail", "output_tail", "stdout_tail"):
        value = str((record or {}).get(key) or "").strip()
        if value:
            pieces.append(value)
    return "\n".join(pieces)


def _failed_record_score(
    record: dict[str, Any],
    fragments: list[dict[str, Any]],
    index: int,
    total: int,
) -> tuple[int, int, int, int]:
    """Prefer the causal command in a chain, not cleanup or wrapper output."""
    command = str(record.get("command") or "").lower()
    role = str(record.get("role") or record.get("command_role") or "").lower()
    score = 0
    if role in {"build", "compile", "configure", "bootstrap"}:
        score += 40
    if role in {"cleanup", "probe", "validation", "install"}:
        score -= 30
    if re.search(r"\b(?:make|ninja|cmake|meson|configure|autoreconf|gcc|g\+\+|clang)\b", command):
        score += 10
    causal = [item for item in fragments if item.get("kind") not in {"wrapper_error", "other"}]
    score += min(50, len(causal) * 12)
    return score, index, -int(record.get("distance_to_end") or max(0, total - index - 1)), index


def _failure_confidence(primary: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    if not primary:
        return "none"
    kind = str(primary.get("kind") or "other")
    if kind in {"tool_missing", "package_manager_error", "header_not_found", "library_not_found", "unresolved_symbol"}:
        return "high"
    if len(candidates) > 1:
        return "medium"
    return "low"


def _failure_evidence_requests(primary: dict[str, Any], candidates: list[dict[str, Any]]) -> list[str]:
    if not primary:
        return ["verify build artifact or compilation/link postcondition"]
    if str(primary.get("kind") or "") in {"configure_error", "wrapper_error", "other"}:
        return ["inspect the tool-specific nested diagnostic log or nearest causal child line"]
    return []


def _build_evidence_bundle(
    primary_failure: dict[str, Any],
    primary_record: dict[str, Any],
    supporting_evidence: list[str],
) -> list[dict[str, str]]:
    """Keep compact, attributable diagnostics from the failed command."""
    entries: list[tuple[str, str]] = []
    primary_text = str(primary_failure.get("text") or "").strip()
    if primary_text:
        entries.append(("primary", primary_text))
    entries.extend(("failed_command", line) for line in _diagnostic_lines(_record_failure_text(primary_record)))
    entries.extend(("supporting", str(line).strip()) for line in supporting_evidence if str(line).strip())

    seen: set[str] = set()
    compact: list[dict[str, str]] = []
    for source, text in entries:
        normalized = _normalise_failure_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        compact.append({
            "id": f"E{len(compact) + 1}",
            "source": source,
            "text": text[:500],
        })
        if len(compact) >= 12:
            break
    return compact


def _diagnostic_lines(text: str) -> list[str]:
    pattern = re.compile(
        r"error|fatal|failed|failure|not found|no such file|cannot |could not |"
        r"command not found|cannot continue|missing required|too old|unsupported|undefined|"
        r"unknown (?:argument|option)|unrecognized (?:argument|option)|"
        r"no rule to make target|does not appear to contain",
        re.I,
    )
    values = [
        line.strip()[:500]
        for line in str(text or "").splitlines()
        if pattern.search(line) and not _is_noise_line(line)
    ]
    return _unique(values)[:20]


_NOISE_LINE_RE = re.compile(
    r"(?:^\s*[-*]?\s*\[[^\]]+\]\([^)]+\)\s*$|"
    r"nothing to be done for|is up to date\.?$|already (?:installed|the newest version)|"
    r"reading package lists|building dependency tree|reading state information|"
    r"^\s*\[cwd is now [^]]+\]\s*$|^\s*\d+\s+gems? installed\s*$)",
    re.I,
)


def _is_noise_line(value: Any) -> bool:
    return bool(_NOISE_LINE_RE.search(str(value or "").strip()))


_FRAGMENT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("package_manager_error", re.compile(PACKAGE_MANAGER_FAILURE_PATTERN, re.I)),
    ("tool_missing", re.compile(MISSING_TOOL_PATTERN, re.I)),
    ("compiler_policy", re.compile(COMPILER_POLICY_PATTERN, re.I)),
    ("tool_configuration", re.compile(
        rf"{TOOL_CONFIGURATION_PATTERN}|{VERSION_REQUIREMENT_PATTERN}",
        re.I,
    )),
    ("infrastructure", re.compile(INFRASTRUCTURE_FAILURE_PATTERN, re.I)),
    ("fatal_error", re.compile(r"\bfatal error\b|\bfatal:\s", re.I)),
    ("header_not_found", re.compile(MISSING_HEADER_PATTERN, re.I)),
    ("unresolved_symbol", re.compile(r"undefined reference|unresolved external|undefined symbol|multiple definition", re.I)),
    ("library_not_found", re.compile(r"cannot find\s+-l|library not found", re.I)),
    ("package_metadata", re.compile(
        r"No package ['\"]|Package ['\"].*(?:not found|was not found)|"
        r"Could not find a package configuration|"
        r"Dependency\s+['\"][^'\"]+['\"]\s+(?:was\s+)?not found|"
        r"Unable to find development files for\s+[A-Za-z0-9_.+\-]+|"
        r"Checking for\s+['\"]?[A-Za-z0-9_.+\-]+['\"]?(?:\s*[<>=][^:\n]*)?\s*:\s*not found|"
        r"Could not find prerequisite package\s+['\"][A-Za-z0-9_.+\-]+['\"]|"
        r"\b[A-Za-z0-9_.+\-]+\s+could not be found automatically|"
        r"[A-Za-z0-9_.+\-]+\s+not found on system or in local libs|"
        r"development package (?:installed|not found)|"
        r"configure:\s*error:[^\n]*(?:headers? (?:were |was )?not found|"
        r"not found|is required|missing required)|"
        r"(?:perl|python|ruby) module\s+[A-Za-z0-9_.+\-]+\s+not found|"
        r"No\s+[A-Za-z0-9_.+\-]+(?:\s+or\s+[A-Za-z0-9_.+\-]+)+\s+present|"
        r"Neither a subproject directory nor a [A-Za-z0-9_.+\-]+\.wrap file was found",
        re.I,
    )),
    ("configure_error", re.compile(r"configure:\s*error|CMake Error|meson\.build.*ERROR", re.I)),
    ("wrapper_error", re.compile(r"make(?:\[\d+\])?: \*\*\*|ninja: build stopped|collect2: error|ld returned \d+ exit status|error: command .* failed", re.I)),
    ("compiler_error", re.compile(r"(?:^|\s)error:\s", re.I)),
    ("missing_path", re.compile(r"No such file or directory|file not found", re.I)),
    ("timeout", re.compile(r"timed out|timeout after", re.I)),
)


def _match_command_step(
    record: dict[str, Any],
    command_steps: list[dict[str, Any]],
) -> tuple[Optional[int], str]:
    record_command = _normalise_command_text(record.get("command"))
    best: tuple[Optional[int], str] = (None, "unknown")
    for index, step in enumerate(command_steps):
        if not isinstance(step, dict):
            continue
        step_command = _normalise_command_text(step.get("command"))
        if not step_command:
            continue
        if record_command == step_command or record_command.endswith(step_command) or step_command.endswith(record_command):
            return index, str(step.get("stage") or "unknown")
    return best


def _extract_failure_fragments(
    record: dict[str, Any],
    record_index: int,
    sequence_index: Optional[int],
) -> list[dict[str, Any]]:
    command = str(record.get("command") or "")
    all_fragments: list[dict[str, Any]] = []
    for stream in ("stderr", "stdout", "output"):
        text = str(record.get(stream) or "")
        fragments = _extract_text_fragments(
            text,
            stream=stream,
            command=command,
            record_index=record_index,
            sequence_index=sequence_index,
        )
        all_fragments.extend(fragments)
    return all_fragments


def _extract_text_fragments(
    text: str,
    *,
    stream: str,
    command: str,
    record_index: int,
    sequence_index: Optional[int],
) -> list[dict[str, Any]]:
    lines = str(text or "").splitlines()
    fragments: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for line_index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or _is_noise_line(line):
            continue
        for kind, pattern in _FRAGMENT_RULES:
            if not pattern.search(line):
                continue
            key = (kind, line)
            if key not in seen:
                fragments.append(_make_fragment(
                    _diagnostic_span(lines, line_index),
                    kind=kind,
                    stream=stream,
                    line_index=line_index,
                    line_count=len(lines),
                    command=command,
                    record_index=record_index,
                    sequence_index=sequence_index,
                ))
                seen.add(key)
            break
    return fragments


def _diagnostic_span(lines: list[str], line_index: int) -> str:
    """Keep a causal line with its short indented continuation."""
    selected = [str(lines[line_index]).strip()]
    for index in range(line_index + 1, min(len(lines), line_index + 13)):
        line = str(lines[index]).strip()
        if not line or _is_noise_line(line):
            continue
        if lines[index][:1].isspace() or re.search(
            r"(?:required|not found|cannot|could not|can't|does not exist|"
            r"not a full path|unknown .* command|aborting)",
            line,
            re.I,
        ):
            selected.append(line)
        elif len(selected) > 1:
            break
        if len(selected) >= 8:
            break
    return "\n".join(selected)[:1000]


def _make_fragment(
    text: str,
    *,
    kind: str,
    stream: str,
    line_index: int,
    line_count: int,
    command: str,
    record_index: int,
    sequence_index: Optional[int],
) -> dict[str, Any]:
    return {
        "text": str(text or "")[:1000],
        "normalized_text": _normalise_failure_text(text),
        "kind": kind,
        "stream": stream,
        "line_start": line_index,
        "line_end": line_index,
        "distance_to_end": max(0, line_count - line_index - 1),
        "causal": kind not in {"wrapper_error", "configure_error"},
        "command": command,
        "record_index": record_index,
        "sequence_index": sequence_index,
    }


def _select_primary_fragment(fragments: list[dict[str, Any]]) -> dict[str, Any]:
    if not fragments:
        return {}
    substantive = [item for item in fragments if item.get("kind") != "wrapper_error"]
    pool = substantive or fragments
    kind_rank = {
        "package_manager_error": 0,
        "tool_missing": 0,
        "compiler_policy": 0,
        "tool_configuration": 0,
        "infrastructure": 0,
        "header_not_found": 0,
        "unresolved_symbol": 0,
        "library_not_found": 0,
        "package_metadata": 0,
        "configure_error": 1,
        "fatal_error": 1,
        "compiler_error": 1,
        "missing_path": 2,
        "timeout": 2,
        "other": 3,
        "wrapper_error": 4,
    }
    return deepcopy(min(
        pool,
        key=lambda item: (
            kind_rank.get(str(item.get("kind") or "other"), 3),
            not bool(item.get("causal", False)),
            int(item.get("distance_to_end", 10**9)),
            -int(item.get("line_end", 0)),
        ),
    ))


def _supporting_fragments(
    candidates: list[dict[str, Any]],
    primary: dict[str, Any],
) -> list[str]:
    ordered = sorted(
        candidates,
        key=lambda item: (int(item.get("record_index", 0)), int(item.get("line_start", 0))),
    )
    values = []
    for item in ordered:
        if (
            item.get("record_index") == primary.get("record_index")
            and item.get("line_start") == primary.get("line_start")
            and item.get("text") == primary.get("text")
        ):
            continue
        values.append(str(item.get("text") or ""))
    return _unique(values)[-8:]


def _normalise_command_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"(?:^|\s)cd\s+(?:['\"]?/work(?:/[^'\";&]+)?['\"]?)\s*&&\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalise_failure_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"/work/", "<repo>/", text)
    text = re.sub(r":\d+(?::\d+)?(?=[:\s])", ":<line>", text)
    return re.sub(r"\s+", " ", text)


def _failure_signature(command: dict[str, Any], fragment: dict[str, Any]) -> str:
    payload = {
        "command": _normalise_command_text(command.get("command")),
        "sequence_index": command.get("sequence_index"),
        "kind": fragment.get("kind") or "other",
        "failure": fragment.get("normalized_text") or "",
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _bootstrap_signals(text: str) -> list[str]:
    patterns = ["autoreconf", "aclocal", "undefined macro", "config.guess", "config.sub", "autogen.sh", "bootstrap"]
    lowered = text.lower()
    return [p for p in patterns if p.lower() in lowered]


def _diagnose_tags(
    text: str,
    frame: dict[str, Any],
    failed_command: Optional[dict[str, Any]],
) -> list[str]:
    lowered = (text or "").lower()
    command = str((failed_command or {}).get("command") or "").lower()
    tags = []
    if re.search(r"\bapt(?:-get)?(?:\s+[^\s;&|]+)*\s+install\b", command):
        if re.search(r"unable to locate package|has no installation candidate|couldn't find any package|e: package", lowered):
            tags.append("package_not_found")
        if re.search(
            r"held broken packages|conflicts:|unable to correct problems|not going to be installed|breaks:|"
            r"packages were downgraded|would be downgraded|allow-downgrades",
            lowered,
        ):
            tags.append("runtime_package_conflict")
    if re.search(r"failed to fetch|could not resolve|temporary failure resolving|gnutls_handshake|certificate|connection timed out|could not connect|hash sum mismatch", lowered):
        tags.append("dependency_fetch_failure")
    if re.search(r"(?:^|/|\b)(third_party|3rdparty|deps|vendor|external|submodules?)(?:/|\b)", " ".join(frame.get("missing_paths") or []) + " " + " ".join(frame.get("missing_headers") or ""), re.I):
        tags.append("source_dependency_needed")
    if is_wrong_entry_or_target(text):
        tags.append("wrong_entrypoint_or_partial_target")
    if frame.get("missing_tools"):
        tags.append("build_tool_provider_missing")
    if frame.get("toolchain_errors") or re.search(
        r"-werror|treated as error|undefined reference|no member named|"
        r"unknown type name|implicit declaration|multiple definition",
        lowered,
    ):
        tags.append("build_layer_compat_needed")
    return _unique(tags)


def _evidence_lines(text: str) -> list[str]:
    lines = []
    for line in (text or "").splitlines():
        if re.search(
            r"error:|fatal error|not found|no such file|cannot |could not |"
            r"can't find|unable to find|does not exist|not a full path|aborting|"
            r"undefined reference|required|requires|too old|command not found|"
            r"unknown (?:argument|option|(?:cmake|meson) command)|"
            r"packages were downgraded|would be downgraded|no PAL version enabled|"
            r"unrecognized (?:command[- ]line )?(?:argument|option)|"
            r"externally-managed-environment|treated as error",
            line,
            re.I,
        ):
            lines.append(line.strip()[:240])
    return _unique(lines)[-6:]


def _primary_signals(frame: dict[str, Any]) -> list[str]:
    signals = []
    for key in (
        "missing_headers", "missing_libraries", "missing_pkg_configs",
        "missing_cmake_packages", "missing_paths", "missing_tools",
        "toolchain_errors", "bootstrap_errors",
    ):
        for value in frame.get(key) or []:
            signals.append(f"{key}:{value}")
    return signals[:6]


def _next_hints(frame: dict[str, Any]) -> list[str]:
    tags = set(frame.get("diagnose_tags") or [])
    hints = []
    if "runtime_package_conflict" in tags:
        hints.append("shrink apt install to one provider per conflict family before retrying")
    if "package_not_found" in tags:
        hints.append("map package names against the current Debian suite before apt retry")
    if "source_dependency_needed" in tags:
        hints.append("register an executable source_dependency plan with build_order and exports")
    if "build_tool_provider_missing" in tags:
        hints.append("resolve the missing build tool to a verified system or source provider")
    if "wrong_entrypoint_or_partial_target" in tags:
        hints.append("switch to a high-confidence command_candidate or component target instead of repeating")
    if "build_layer_compat_needed" in tags:
        hints.append("try build flags/tool shims or dependency-version alignment before source edits")
    error_class = frame.get("primary_error_class") or frame.get("error_class")
    if error_class == "missing_source_path":
        hints.append("inspect source dependency or generated-file path before installing packages")
    if error_class == "missing_package_metadata":
        hints.append("verify package metadata name, version, and search paths before apt/install retry")
    if error_class == "missing_build_tool":
        hints.append("resolve the missing build tool to a verified provider before retrying the build step")
    if error_class == "missing_dependency":
        hints.append("cross-check missing artifact against repo paths before choosing system package vs source dependency")
    if error_class == "bootstrap":
        hints.append("try project bootstrap/autogen wrapper or generated-file recovery before normal rebuild")
    if error_class == "toolchain_or_api":
        hints.append("check sysuse/toolchain/API mismatch and avoid repeated package installs")
    return _unique(hints)


def _sysuse_candidates(raw_plan: Any, current_sysuse: str) -> list[dict[str, Any]]:
    out = [{
        "candidate_id": f"release:{current_sysuse}",
        "sysuse": current_sysuse,
        "status": "current",
        "source": "runtime",
        "reason": "current container release",
        "probes": [],
    }] if current_sysuse else []
    if isinstance(raw_plan, list):
        for item in raw_plan:
            if not isinstance(item, dict):
                continue
            suite = str(item.get("sysuse") or item.get("suite") or item.get("codename") or "").strip()
            if suite:
                candidate = deepcopy(item)
                candidate.update({
                    "candidate_id": str(item.get("candidate_id") or f"release:{suite}"),
                    "sysuse": suite,
                    "status": str(item.get("status") or "candidate"),
                    "source": str(item.get("source") or "sysuse_probe"),
                    "reason": str(item.get("reason") or item.get("rationale") or ""),
                    "probes": deepcopy(item.get("probes") or item.get("probe_plan") or []),
                })
                out.append(candidate)
    return out


def _bootstrap_commands(project_info: dict[str, Any]) -> list[str]:
    raw_commands = _string_list(project_info.get("main_build_plan"))
    raw_commands.extend(_string_list(project_info.get("bootstrap_plan")))
    for entry in project_info.get("build_entry_candidates") or []:
        if isinstance(entry, dict):
            raw_commands.extend(_string_list(entry.get("command_sequence")))
    commands = []
    for raw in raw_commands:
        segments = [item.strip() for item in re.split(r"\s*&&\s*", raw) if item.strip()]
        prefix = []
        for segment in segments:
            if re.fullmatch(r"cd\s+.+", segment):
                prefix.append(segment)
                continue
            if re.search(
                r"\b(autogen\.sh|bootstrap(?:\.sh)?|buildconf|autoreconf|autoconf|automake|"
                r"aclocal|autoheader|libtoolize|bison|flex|protoc|moc|uic)\b",
                segment,
            ):
                candidate = " && ".join(prefix + [segment])
                if not re.search(
                    r"\b(?:apt(?:-get)?|dpkg|pip3?|npm|yarn|pnpm|gem|cargo)\s+(?:install|i|add)\b",
                    candidate,
                    re.I,
                ):
                    commands.append(candidate)
                break
    return _unique(commands)


def _target_candidates(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Materialize only scanner-provided targets with executable commands.

    Target names inferred from an existing command are not alternatives: they
    would select the same command again.  A target can enter the repair pool
    only when upstream discovery explicitly supplies a different command
    sequence, keeping target switching fail-closed when evidence is absent.
    """
    candidates: list[dict[str, Any]] = []
    for entry in entries or []:
        if not isinstance(entry, dict) or entry.get("status") == "ineligible":
            continue
        base_sequence = _string_list(entry.get("command_sequence"))
        for raw in entry.get("target_candidates") or []:
            if not isinstance(raw, dict):
                continue
            target = str(
                raw.get("target_id")
                or raw.get("target")
                or raw.get("name")
                or ""
            ).strip()
            sequence = _string_list(raw.get("command_sequence") or raw.get("commands"))
            if not target or not sequence or sequence == base_sequence:
                continue
            payload = f"{entry.get('entry_id') or ''}\x1f{target}\x1f{' && '.join(sequence)}"
            candidates.append({
                "target_id": str(raw.get("target_id") or _stable_candidate_id("target", payload)),
                "name": target,
                "entry_id": entry.get("entry_id") or "",
                "command_sequence": sequence,
                "workdir": raw.get("workdir") or entry.get("workdir") or ".",
                "status": str(raw.get("status") or "candidate"),
                "source": str(raw.get("source") or "scanner_target"),
                "evidence": str(raw.get("evidence") or raw.get("reason") or "")[:500],
            })
    return candidates


def _legacy_target_candidates(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for entry in entries or []:
        if not isinstance(entry, dict) or entry.get("status") == "ineligible":
            continue
        for command in entry.get("command_sequence") or []:
            for target in _targets_from_command(str(command or "")):
                payload = f"{entry.get('entry_id') or ''}\x1f{target}"
                candidates.append({
                    "target_id": _stable_candidate_id("target", payload),
                    "name": target,
                    "entry_id": entry.get("entry_id") or "",
                    "status": "candidate",
                    "source": "entry_command",
                    "evidence": str(command)[:500],
                })
    return candidates


def _targets_from_command(command: str) -> list[str]:
    targets = []
    for segment in re.split(r"&&|\|\||[;|]", command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            continue
        if not tokens:
            continue
        executable = os.path.basename(tokens[0]).lower()
        if executable == "cmake" and "--build" in tokens and "--target" in tokens:
            index = tokens.index("--target")
            if index + 1 < len(tokens):
                targets.append(tokens[index + 1])
        elif executable == "meson" and len(tokens) > 1 and tokens[1] == "compile":
            targets.extend(_positional_targets(tokens[2:], options_with_value={"-C", "--jobs"}))
        elif executable in {"make", "gmake", "ninja"}:
            targets.extend(_positional_targets(tokens[1:], options_with_value={"-C", "-f", "-j", "--jobs"}))
    return [item for item in _unique(targets) if item not in {"all", "install", "test", "check"}]


def _positional_targets(tokens: list[str], *, options_with_value: set[str]) -> list[str]:
    targets = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in options_with_value:
            skip_next = True
            continue
        if token.startswith("-") or "=" in token:
            continue
        if re.fullmatch(r"[A-Za-z0-9_.+-]+", token):
            targets.append(token)
    return targets


def _stable_candidate_id(kind: str, value: str) -> str:
    digest = hashlib.sha256(str(value or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{kind}:{digest}"


def _ensure_candidate_ids(plan: dict[str, Any]) -> None:
    specs = (
        ("system_dep_candidates", "candidate_id", "package", ("name", "kind")),
        ("source_dep_candidates", "candidate_id", "source", ("name", "source_path")),
        ("bootstrap_candidates", "candidate_id", "bootstrap", ("command",)),
        ("generator_candidates", "generator_id", "generator", ("command",)),
        ("target_candidates", "target_id", "target", ("entry_id", "name")),
    )
    for plan_key, id_key, kind, value_keys in specs:
        for item in plan.get(plan_key) or []:
            if not isinstance(item, dict) or item.get(id_key):
                continue
            value = "\x1f".join(str(item.get(key) or "") for key in value_keys)
            item[id_key] = _stable_candidate_id(kind, value)


def _select_candidate(
    candidates: list[dict[str, Any]],
    candidate_id: Any,
    id_fields: tuple[str, ...],
) -> Optional[dict[str, Any]]:
    selected = None
    needle = str(candidate_id or "")
    for item in candidates:
        if not isinstance(item, dict):
            continue
        matches = any(str(item.get(field) or "") == needle for field in id_fields)
        if matches and item.get("status") not in {"failed", "exhausted", "ineligible"}:
            item["status"] = "selected"
            selected = item
        elif item.get("status") == "selected":
            item["status"] = "candidate"
    return selected


def _mark_entrypoint_failed(plan: dict[str, Any], command: str) -> None:
    if not command:
        return
    current_sysuse = str((plan.get("selected") or {}).get("sysuse") or "").strip()
    selected_entry_id = str((plan.get("selected") or {}).get("build_entry_id") or "").strip()
    for item in plan.get("entrypoint_candidates") or []:
        if not isinstance(item, dict):
            continue
        if selected_entry_id and str(item.get("entry_id") or "") != selected_entry_id:
            continue
        planned = str(item.get("command") or "")
        sequence = [str(value or "") for value in item.get("command_sequence") or []]
        matched = planned and (planned == command or planned in command or command in planned)
        if not matched:
            matched = any(value and (value == command or value in command or command in value) for value in sequence)
        if matched:
            failed_sysuses = _string_list(item.get("failed_sysuses"))
            if current_sysuse and current_sysuse not in failed_sysuses:
                failed_sysuses.append(current_sysuse)
            item["failed_sysuses"] = failed_sysuses
            item["status"] = "failed"


def _merge_by_key(target: list[dict[str, Any]], additions: list[dict[str, Any]], key: str) -> None:
    seen = {str(item.get(key) or "").lower(): item for item in target if isinstance(item, dict)}
    for item in additions:
        value = str(item.get(key) or "").strip()
        if not value:
            continue
        existing = seen.get(value.lower())
        if existing:
            for k, v in item.items():
                if v not in ("", None, [], {}) and not existing.get(k):
                    existing[k] = v
        else:
            target.append(item)
            seen[value.lower()] = item


def _command_entry_id(item: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "path": item.get("path") or "",
            "command": item.get("command") or "",
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return "entry_command_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _compact_failed_command(failed_command: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not failed_command:
        return {}
    return {
        "command": failed_command.get("command", ""),
        "status": failed_command.get("status", ""),
        "exit_code": failed_command.get("exit_code"),
        "log_path": failed_command.get("log_path", ""),
    }


def _compact_command_record(record: Optional[dict[str, Any]]) -> dict[str, Any]:
    compact = _compact_failed_command(record)
    if not record:
        return compact
    compact["stderr_tail"] = str(record.get("stderr") or record.get("stderr_tail") or "")[-800:]
    compact["output_tail"] = str(record.get("output") or record.get("output_tail") or "")[-800:]
    return compact


def _limit(value: Any, max_items: int) -> Any:
    if not isinstance(value, list):
        return [] if value is None else value
    return value[-max_items:]


def _head_limit(value: Any, max_items: int) -> Any:
    if not isinstance(value, list):
        return [] if value is None else value
    return value[:max_items]


def _ordered_candidate_limit(value: Any, max_items: int) -> Any:
    if not isinstance(value, list):
        return [] if value is None else value
    if any(isinstance(item, dict) and item.get("rank") is not None for item in value):
        return value[:max_items]
    return value[-max_items:]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _unique(values: Any) -> list[str]:
    seen = set()
    out = []
    for value in values:
        value = str(value or "").strip()
        if not value or value.lower() in seen:
            continue
        seen.add(value.lower())
        out.append(value)
    return out


def _set_current_sysuse(plan: dict[str, Any], current_sysuse: str) -> None:
    current = str(current_sysuse or "").strip()
    if not current:
        return
    plan.setdefault("selected", {})["sysuse"] = current
    for item in plan.get("sysuse_candidates") or []:
        suite = str(item.get("sysuse") or "").strip()
        if normalize_sysuse(suite) == normalize_sysuse(current):
            item["status"] = "current"
        elif item.get("status") == "current":
            item["status"] = "candidate"


def _refresh_entry_alternatives(plan: dict[str, Any]) -> None:
    entries = plan.get("entrypoint_candidates") or []
    current_sysuse = str((plan.get("selected") or {}).get("sysuse") or "").strip()
    plan.setdefault("alternatives", {})["build_entries"] = [
        deepcopy(item)
        for item in entries
        if item.get("status") not in {"selected", "ineligible", "failed", "exhausted"}
        and item.get("eligible", True)
        and current_sysuse not in _string_list(item.get("failed_sysuses"))
    ]


def _refresh_selected_and_alternatives(
    plan: dict[str, Any],
    project_info: dict[str, Any],
    current_sysuse: str,
) -> None:
    selected = plan.setdefault("selected", {})
    selected["sysuse"] = str(current_sysuse or selected.get("sysuse") or "")

    selected_entry = None
    requested_entry = project_info.get("selected_build_entry")
    requested_id = ""
    if isinstance(requested_entry, dict):
        requested_id = str(requested_entry.get("entry_id") or "")
    elif requested_entry:
        requested_id = str(requested_entry)

    entries = plan.get("entrypoint_candidates") or []
    for item in entries:
        if requested_id and str(item.get("entry_id") or "") == requested_id and item.get("eligible", True) and item.get("status") != "ineligible":
            selected_entry = item
            break
    if selected_entry is None:
        selected_entry = next(
            (item for item in entries if item.get("status") == "selected" and item.get("eligible", True)),
            None,
        )
    if selected_entry is None:
        selected_entry = next(
            (
                item for item in entries
                if item.get("status") != "ineligible"
                and item.get("eligible", True)
                and str(current_sysuse or "") not in _string_list(item.get("failed_sysuses"))
            ),
            None,
        )
    if selected_entry is not None:
        for item in entries:
            if item is not selected_entry and item.get("status") == "selected":
                item["status"] = "candidate"
        selected_entry["status"] = "selected"
        selected.update(
            {
                "build_entry_id": selected_entry.get("entry_id") or "",
                "build_entry_path": selected_entry.get("path") or "",
                "command_sequence": list(selected_entry.get("command_sequence") or []),
                "command_steps": deepcopy(selected_entry.get("command_steps") or []),
            }
        )
    _refresh_entry_alternatives(plan)

    realization_methods = project_info.get("selected_realization_methods")
    if isinstance(realization_methods, dict):
        selected["realization_methods"] = deepcopy(realization_methods)
    plan.setdefault("alternatives", {})["sysuses"] = [
        deepcopy(item)
        for item in plan.get("sysuse_candidates") or []
        if item.get("status") != "current"
    ]


def _refresh_dependency_state(plan: dict[str, Any], project_info: dict[str, Any]) -> None:
    existing = deepcopy(plan.get("dependency_state") or {})
    projects = project_info.get("dependency_projects")
    if not isinstance(projects, list):
        projects = project_info.get("enhanced_dependency_projects")
    if isinstance(projects, list):
        existing["projects"] = deepcopy(projects)
    unresolved = project_info.get("unresolved_dependency_clues")
    if isinstance(unresolved, list):
        existing["unresolved_clues"] = deepcopy(unresolved)
    existing.setdefault("projects", [])
    existing.setdefault("unresolved_clues", [])
    plan["dependency_state"] = existing
