"""Low-cost provenance probes for deterministic build-failure routing.

The probes are intentionally local and synchronous: they inspect the checked
out repository, compact build plan, failure frame, and already-recorded command
history. They do not run Docker commands or package-manager queries.
"""

from __future__ import annotations

import os
import re
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from core.debian_sysuse import normalize_sysuse
from diagnosis.diagnosis_router import DiagnosisRoute
from diagnosis.fix_decision_matrix import canonical_fix_direction, compatibility_directions, evidence_gate_satisfied


VENDORED_PATH_RE = re.compile(r"(?:^|/|\b)(third_party|3rdparty|deps|vendor|external|submodules?)(?:/|\b)", re.I)
GENERATION_HINT_RE = re.compile(
    r"(configure_file|add_custom_command|add_custom_target|protobuf|protoc|"
    r"\.(?:proto|y|l|ui|in)\b|moc\b|uic\b|bison\b|flex\b|AC_CONFIG_HEADERS)",
    re.I,
)
SOURCE_EXTS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".ipp"}
BUILD_FILE_NAMES = {
    "CMakeLists.txt",
    "configure.ac",
    "configure.in",
    "Makefile",
    "Makefile.am",
    "Makefile.in",
    "meson.build",
    "qmake.pro",
}


def run_provenance_probes(
    route: DiagnosisRoute,
    failure_frame: dict[str, Any],
    build_plan: Optional[dict[str, Any]],
    repo_path: str,
    command_records: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Return a compact provenance/ownership probe for the current route."""
    symptom = route.symptom_family
    entity = route.entity or {}
    name = entity.get("name") or _fallback_object_name(symptom, failure_frame)
    base = {
        "object_kind": entity.get("kind") or _object_kind_for_symptom(symptom),
        "object_name": name,
        "provenance": "unknown",
        "confidence": "low",
        "evidence": [],
        "preferred_decisions": [],
        "discouraged_decisions": [],
        "failure_mechanism": route.failure_mechanism,
        "version_relation": dict(route.version_relation or {}),
    }

    if str(entity.get("kind") or "") == "build_macro":
        result = _probe_build_macro(base, name, build_plan or {})
    elif route.classification_status == "out_of_matrix" and route.failure_domain == "source":
        result = _probe_source_compile(base, failure_frame, repo_path)
    elif symptom == "header_not_found":
        result = _probe_header(base, name, repo_path)
    elif symptom == "library_not_found":
        result = _probe_library(base, name, repo_path, command_records or [])
    elif symptom == "unresolved_symbol":
        result = _probe_symbol(base, name, repo_path, command_records or [])
    elif symptom in {"pkg_config_failure", "cmake_package_failure"}:
        result = _probe_package_metadata(base, name, build_plan or {})
    elif symptom == "build_tool_not_found":
        result = _probe_build_tool(base, name, build_plan or {})
    elif symptom == "api_or_toolchain_mismatch" and route.failure_mechanism == "version_shortfall":
        relation = route.version_relation or {}
        name = str(relation.get("subject") or route.failed_tool or name or "").strip()
        base["object_kind"] = "build_tool"
        base["object_name"] = name
        result = _probe_build_tool(base, name, build_plan or {})
    elif symptom == "source_path_missing":
        result = _probe_build_input(base, build_plan or {})
    elif symptom in {"wrong_entrypoint", "generated_file_missing"}:
        result = _probe_entry_or_generation(base, build_plan or {})
    else:
        result = base
    return _attach_candidate_facts(result, route, build_plan or {})


def apply_probe_refinement(route: DiagnosisRoute, probe: dict[str, Any]) -> DiagnosisRoute:
    """Return a route with only evidence-supported conditional actions.

    Direct actions come from the offline Matrix prior. Conditional actions are
    admitted only when the structured probe satisfies their gate; fallback
    actions stay separate for Build Plan exhaustion handling.
    """
    if not probe:
        return route
    if route.classification_status == "out_of_matrix":
        repair_scope = dict(route.repair_scope or {})
        repair_scope["provenance_probe"] = _compact_probe(probe)
        if route.failure_domain == "source" and probe.get("provenance") != "project_source":
            repair_scope["matrix_actions"] = []
            repair_scope["selected_matrix_action"] = None
            repair_scope["matrix_empty"] = True
            return replace(
                route,
                classification_status="insufficient_evidence",
                failure_domain="unknown",
                classification_source="source_provenance_unresolved",
                classification_confidence="low",
                repair_scope=repair_scope,
            )
        return replace(route, repair_scope=repair_scope)
    preferred = [str(item) for item in probe.get("preferred_decisions") or [] if item]
    preferred_output = []
    for decision in preferred:
        preferred_output.extend(compatibility_directions(decision))
    preferred = _unique(preferred_output)
    discouraged = {canonical_fix_direction(item) for item in probe.get("discouraged_decisions") or [] if item}
    repair_scope = dict(route.repair_scope or {})
    direct_actions = [
        dict(item) for item in repair_scope.get("direct_matrix_actions")
        or repair_scope.get("matrix_actions") or []
        if isinstance(item, dict) and not item.get("fallback_only")
    ]
    relation_subject = str((route.version_relation or {}).get("subject") or "").strip().lower()
    compiler_shortfall = route.failure_mechanism == "version_shortfall" and (
        route.failed_tool == "compiler"
        or relation_subject in {"cc", "c++", "gcc", "g++", "clang", "clang++"}
    )
    if route.failure_mechanism == "version_shortfall" and not compiler_shortfall:
        direct_actions = [
            item for item in direct_actions
            if str(item.get("action_id") or "") != "compiler_config"
        ]
    conditional_actions = [
        dict(item) for item in repair_scope.get("conditional_actions") or []
        if isinstance(item, dict) and _conditional_action_supported(item, route, probe)
    ]
    direct_directions = []
    for item in direct_actions:
        direct_directions.extend(compatibility_directions(item.get("fix_direction") or ""))
    conditional_directions = []
    for item in conditional_actions:
        conditional_directions.extend(compatibility_directions(item.get("fix_direction") or ""))
    allowed_directions = _unique(direct_directions + conditional_directions)
    ordered = []
    for decision in preferred:
        if decision in allowed_directions and decision not in discouraged and decision not in ordered:
            ordered.append(decision)
    for decision in allowed_directions:
        if decision not in ordered and decision not in discouraged:
            ordered.append(decision)
    # Preserve a discouraged direction only as a diagnostic compatibility
    # projection when it is the sole direct matrix action; it is never promoted
    # from a conditional or fallback pool.
    if not ordered:
        ordered = [item for item in allowed_directions if item not in discouraged]
    repair_scope["target_decisions"] = ordered
    repair_scope["allowed_revision_types"] = _allowed_revisions_for_decisions(
        ordered,
        route.repair_scope.get("allowed_revision_types") if route.repair_scope else [],
    )
    repair_scope["provenance_probe"] = _compact_probe(probe)
    matrix_actions = direct_actions + conditional_actions
    direction_rank = {
        canonical_fix_direction(direction): index
        for index, direction in enumerate(ordered)
    }
    matrix_actions.sort(
        key=lambda item: (
            direction_rank.get(str(item.get("fix_direction") or ""), len(direction_rank)),
            int(item.get("matrix_rank") or 99),
        )
    )
    for index, item in enumerate(matrix_actions, 1):
        item["route_rank"] = index
    repair_scope["matrix_actions"] = matrix_actions
    repair_scope["selected_matrix_action"] = matrix_actions[0] if matrix_actions else None
    repair_scope["matrix_empty"] = not bool(matrix_actions)
    repair_scope["direct_matrix_actions"] = direct_actions
    repair_scope["conditional_actions"] = conditional_actions
    repair_scope.setdefault("fallback_actions", [])
    checks = list(route.next_checks or [])
    if probe.get("provenance") and probe.get("provenance") != "unknown":
        checks.insert(0, f"provenance probe: {probe.get('provenance')}")
    return replace(
        route,
        candidate_decisions=ordered,
        next_checks=_unique(checks),
        repair_scope=repair_scope,
    )


def _probe_source_compile(
    base: dict[str, Any],
    failure_frame: dict[str, Any],
    repo_path: str,
) -> dict[str, Any]:
    text = str(failure_frame.get("primary_failure_text") or "")
    match = re.search(
        r"(?:^|\n)((?:[^\n:]+/)?[^\n:]+\.(?:c|cc|cpp|cxx|h|hh|hpp|hxx)):\d+(?::\d+)?:\s*(?:fatal\s+)?error:",
        text,
        re.I,
    )
    source_path = str(match.group(1) if match else "").strip()
    source_path = re.sub(r"^/work/?", "", source_path)
    exact_path = _repo_relative_file_exists(repo_path, source_path) if source_path else ""
    resolved_path = exact_path
    if source_path and not resolved_path:
        basename_matches = _find_repo_files(repo_path, os.path.basename(source_path), max_results=6)
        suffix_matches = [
            candidate
            for candidate in basename_matches
            if candidate == source_path or candidate.endswith(f"/{source_path}")
        ]
        if len(suffix_matches) == 1:
            resolved_path = suffix_matches[0]
        elif len(basename_matches) == 1:
            resolved_path = basename_matches[0]
    base.update(
        object_kind="source_file",
        object_name=source_path,
    )
    if resolved_path:
        base.update(
            provenance="project_source",
            confidence="high",
            discouraged_decisions=[
                "system_package_provider",
                "source_dependency",
                "generation",
                "system_release",
                "build_entry",
                "build_target",
            ],
        )
        base["evidence"].append(f"compiler diagnostic points to checked-out source {resolved_path}")
    else:
        base.update(
            provenance="generated_or_external",
            confidence="low",
        )
        base["evidence"].append("compiler diagnostic source is not present at the reported repository path")
    return base


def _conditional_action_supported(
    action: dict[str, Any],
    route: DiagnosisRoute,
    probe: dict[str, Any],
) -> bool:
    if action.get("fallback_only"):
        return False
    gate = str(action.get("evidence_gate") or "").strip().lower()
    provenance = str(probe.get("provenance") or "").strip().lower()
    direction = canonical_fix_direction(action.get("fix_direction") or "")
    if gate in {"provider_candidate", "source_dependency_candidate", "generator_candidate"}:
        if not str(probe.get("object_name") or "").strip():
            return False
    discouraged = {canonical_fix_direction(item) for item in probe.get("discouraged_decisions") or [] if item}
    if direction in discouraged:
        return False
    supported = evidence_gate_satisfied(
        gate,
        provenance=provenance,
        failed_phase=route.failed_phase,
        preferred_decisions=probe.get("preferred_decisions") or [],
        fix_direction=direction,
        evidence_facts=probe,
    )
    return bool(supported)


def _probe_header(base: dict[str, Any], header: str, repo_path: str) -> dict[str, Any]:
    exact_path = _repo_relative_file_exists(repo_path, header)
    paths = _find_repo_files(repo_path, os.path.basename(header), max_results=6)
    if exact_path:
        base.update(
            provenance="repo_local_invisible",
            confidence="high",
            preferred_decisions=["dependency_discovery", "build_entry"],
            discouraged_decisions=["system_release"],
        )
        base["evidence"].append(f"repo contains requested header path {exact_path}")
        return base

    if VENDORED_PATH_RE.search(header or ""):
        base.update(
            provenance="project_managed",
            confidence="high",
            preferred_decisions=["source_dependency", "build_entry", "generation"],
            discouraged_decisions=["system_release"],
        )
        base["evidence"].append("missing header path points at vendored/project-managed dependency")
        if paths:
            base["evidence"].append("repo contains same-basename header elsewhere, but not at requested vendored path")
        return base

    if paths:
        base.update(
            provenance="repo_local_invisible",
            confidence="high",
            preferred_decisions=["dependency_discovery", "build_entry"],
            discouraged_decisions=["system_release"],
        )
        base["evidence"].append(f"repo contains {len(paths)} matching header path(s)")
        base["evidence"].extend(paths[:3])
        return base

    if _repo_has_generation_hint(repo_path, header):
        base.update(
            provenance="generated",
            confidence="medium",
            preferred_decisions=["generation", "build_entry", "build_target"],
            discouraged_decisions=["system_release"],
        )
        base["evidence"].append("build files contain generated-file rules or generator inputs")
        return base

    base.update(
        provenance="system_or_external",
        confidence="medium",
        preferred_decisions=["system_package_provider", "dependency_discovery", "system_release"],
    )
    base["evidence"].append("header absent from repository; no vendored/generated signal found")
    return base


def _probe_library(
    base: dict[str, Any],
    library: str,
    repo_path: str,
    command_records: list[dict[str, Any]],
) -> dict[str, Any]:
    lib_names = [f"lib{library}.so", f"lib{library}.a"]
    matches = []
    for lib_name in lib_names:
        matches.extend(_find_repo_files(repo_path, lib_name, max_results=3))
    latest_link = _latest_command_text(command_records)
    if matches:
        has_link_path = any("-L" in latest_link and _dirname_fragment(path) in latest_link for path in matches)
        base.update(
            provenance="repo_local_invisible" if not has_link_path else "link_composition",
            confidence="high",
            preferred_decisions=["linker_configuration", "build_target"],
            discouraged_decisions=["system_release"],
        )
        base["evidence"].append(f"repo/build tree contains candidate library {matches[0]}")
        if not has_link_path:
            base["evidence"].append("latest command does not appear to expose the library directory")
        return base

    if _build_files_reference(repo_path, library) and VENDORED_PATH_RE.search(_read_build_file_snippets(repo_path, library)):
        base.update(
            provenance="project_managed",
            confidence="medium",
            preferred_decisions=["source_dependency", "linker_configuration"],
            discouraged_decisions=["system_release"],
        )
        base["evidence"].append("build files reference library near vendored/project-managed paths")
        return base

    base.update(
        provenance="system_or_external",
        confidence="medium",
        preferred_decisions=["system_package_provider", "linker_configuration", "system_release"],
    )
    base["evidence"].append("library absent from repository/build tree")
    return base


def _probe_symbol(
    base: dict[str, Any],
    symbol: str,
    repo_path: str,
    command_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if re.search(r"GLIBCXX_|CXXABI_|@GLIBC", symbol or ""):
        base.update(
            provenance="system_or_external",
            confidence="high",
            preferred_decisions=["system_release", "compiler_configuration", "linker_configuration"],
            discouraged_decisions=["system_package_provider", "source_dependency"],
        )
        base["evidence"].append("symbol contains C/C++ runtime ABI tag")
        return base

    if _symbol_defined_in_repo(repo_path, symbol):
        base.update(
            provenance="project_source",
            confidence="medium",
            preferred_decisions=["build_target", "linker_configuration", "build_entry"],
            discouraged_decisions=["system_release"],
        )
        base["evidence"].append("repository source appears to define the unresolved symbol")
        return base

    latest_link = _latest_command_text(command_records)
    if re.search(r"\b(?:ld|gcc|g\+\+|clang|clang\+\+|cc|c\+\+)\b", latest_link) and "-l" in latest_link:
        base.update(
            provenance="link_composition",
            confidence="medium",
            preferred_decisions=["linker_configuration", "system_package_provider", "build_target"],
        )
        base["evidence"].append("latest command looks like a link invocation")
        return base

    base.update(
        provenance="unknown",
        confidence="low",
        preferred_decisions=["linker_configuration", "system_package_provider", "build_target"],
    )
    base["evidence"].append("symbol owner not resolved by local probes")
    return base


def _probe_package_metadata(base: dict[str, Any], name: str, build_plan: dict[str, Any]) -> dict[str, Any]:
    source_text = " ".join(str(item) for item in build_plan.get("source_deps") or [])
    system_text = " ".join(str(item) for item in build_plan.get("system_deps") or [])
    if name and name.lower() in source_text.lower():
        base.update(
            provenance="project_managed",
            confidence="medium",
            preferred_decisions=["source_dependency", "dependency_discovery", "linker_configuration"],
            discouraged_decisions=["system_release"],
        )
        base["evidence"].append("build plan source dependency candidates mention this metadata name")
        return base
    if not name:
        base["evidence"].append("dependency metadata diagnostic did not identify a concrete entity")
        return base
    base.update(
        provenance="system_or_external",
        confidence="medium",
        preferred_decisions=["system_package_provider", "dependency_discovery", "system_release"],
    )
    if name and name.lower() in system_text.lower():
        base["evidence"].append("build plan system dependency candidates mention this metadata name")
    else:
        base["evidence"].append("metadata not tied to registered source dependency")
    return base


def _probe_build_tool(base: dict[str, Any], name: str, build_plan: dict[str, Any]) -> dict[str, Any]:
    source_text = " ".join(
        str(item)
        for key in ("source_deps", "source_dep_candidates")
        for item in build_plan.get(key) or []
    )
    system_text = " ".join(
        str(item)
        for key in ("system_deps", "system_dep_candidates")
        for item in build_plan.get(key) or []
    )
    if name and name.lower() in source_text.lower():
        base.update(
            provenance="project_managed",
            confidence="medium",
            preferred_decisions=["source_dependency", "dependency_discovery"],
            discouraged_decisions=["system_release"],
        )
        base["evidence"].append("build plan ties the missing tool to a source dependency candidate")
        return base
    if not name:
        base["evidence"].append("missing-tool diagnostic did not identify a concrete executable")
        return base
    base.update(
        provenance="system_or_external",
        confidence="medium",
        preferred_decisions=["system_package_provider", "system_release"],
    )
    if name and name.lower() in system_text.lower():
        base["evidence"].append("build plan contains a system-provider candidate for the missing tool")
    else:
        base["evidence"].append("missing tool has no project-managed provider in the compact build plan")
    return base


def _probe_build_input(base: dict[str, Any], build_plan: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        item
        for key in ("generator_candidates", "generators", "bootstrap_candidates", "bootstrap", "entrypoint_candidates", "entrypoints")
        for item in build_plan.get(key) or []
        if isinstance(item, dict) and item.get("status") != "failed"
    ]
    generator_candidates = [
        item
        for item in candidates
        if re.search(
            r"(?:^|\s|/)(?:bootstrap|autogen\.sh|autoreconf|generate)(?:\s|$)",
            " ".join([
                str(item.get("command") or ""),
                " ".join(str(command) for command in item.get("command_sequence") or []),
                str(item.get("path") or ""),
            ]),
            re.I,
        )
    ]
    if generator_candidates:
        base.update(
            provenance="generated",
            confidence="medium",
            preferred_decisions=["generation", "build_entry", "build_target"],
            discouraged_decisions=["system_release"],
        )
        base["evidence"].append("build plan contains an unfailed bootstrap or generator candidate")
        return base
    base.update(
        provenance="unknown",
        confidence="low",
        preferred_decisions=["build_entry", "build_target"],
    )
    base["evidence"].append("no unfailed bootstrap or generator candidate found in the build plan")
    return base


def _probe_entry_or_generation(base: dict[str, Any], build_plan: dict[str, Any]) -> dict[str, Any]:
    entrypoints = build_plan.get("entrypoints") or []
    planned = [item for item in entrypoints if isinstance(item, dict) and item.get("status") != "failed"]
    if planned:
        base.update(
            provenance="build_plan_candidate",
            confidence="medium",
            preferred_decisions=["build_entry", "generation", "build_target"],
            discouraged_decisions=["system_package_provider", "source_dependency"],
        )
        base["evidence"].append("build plan contains unfailed entrypoint/generation candidates")
        return base
    base.update(
        provenance="unknown",
        confidence="low",
        preferred_decisions=["build_entry", "generation", "build_target"],
    )
    base["evidence"].append("no unfailed entrypoint candidate found in compact build plan")
    return base


def _probe_build_macro(base: dict[str, Any], macro: str, build_plan: dict[str, Any]) -> dict[str, Any]:
    """Classify Autotools macro diagnostics as provider-backed bootstrap inputs."""
    base.update(
        object_kind="build_macro",
        object_name=macro,
        provenance="system_or_external",
        confidence="medium",
        preferred_decisions=["system_package_provider", "source_dependency", "generation"],
        discouraged_decisions=["system_release"],
    )
    base["evidence"].append("Autotools macro is missing from the active bootstrap/configure environment")
    generator_candidates = [
        item for item in (
            build_plan.get("generator_candidates")
            or build_plan.get("generators")
            or build_plan.get("bootstrap_candidates")
            or build_plan.get("bootstrap")
            or []
        )
        if isinstance(item, dict) and item.get("status") not in {"failed", "exhausted", "ineligible"}
    ]
    if generator_candidates:
        base["evidence"].append("build plan also contains an unfailed bootstrap/generator candidate")
    return base


def _find_repo_files(repo_path: str, basename: str, max_results: int = 5) -> list[str]:
    if not repo_path or not basename:
        return []
    root = Path(repo_path)
    if not root.is_dir():
        return []
    results = []
    skip_dirs = {".git", ".venv", "__pycache__", "node_modules"}
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        if basename in files:
            results.append(os.path.relpath(os.path.join(current, basename), root))
            if len(results) >= max_results:
                break
    return results


def _repo_relative_file_exists(repo_path: str, file_path: str) -> str:
    if not repo_path or not file_path or os.path.isabs(file_path):
        return ""
    relative = Path(file_path)
    if any(part == ".." for part in relative.parts):
        return ""
    candidate = Path(repo_path) / relative
    if candidate.is_file():
        return str(relative)
    return ""


def _repo_has_generation_hint(repo_path: str, object_name: str) -> bool:
    if not repo_path:
        return False
    basename = os.path.basename(object_name or "").lower()
    needle = os.path.splitext(basename)[0]
    if not needle:
        return False
    template_names = {
        f"{basename}.in",
        f"{needle}.in",
        f"{needle}.proto",
        f"{needle}.y",
        f"{needle}.l",
        f"{needle}.ui",
    }
    if any(_find_repo_files(repo_path, candidate, max_results=1) for candidate in template_names):
        return True
    for path in _iter_build_files(repo_path):
        text = _read_text(path)
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if basename not in line.lower() and needle not in line.lower():
                continue
            context = "\n".join(lines[max(0, index - 1):index + 2])
            if GENERATION_HINT_RE.search(context):
                return True
    return False


def _build_files_reference(repo_path: str, needle: str) -> bool:
    if not needle:
        return False
    return bool(_read_build_file_snippets(repo_path, needle))


def _read_build_file_snippets(repo_path: str, needle: str) -> str:
    chunks = []
    needle_l = str(needle or "").lower()
    for path in _iter_build_files(repo_path):
        text = _read_text(path)
        if needle_l in text.lower():
            chunks.append(text[:4000])
    return "\n".join(chunks)


def _iter_build_files(repo_path: str):
    if not repo_path:
        return
    root = Path(repo_path)
    if not root.is_dir():
        return
    skip_dirs = {".git", ".venv", "__pycache__", "node_modules"}
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for filename in files:
            if filename in BUILD_FILE_NAMES or filename.endswith((".cmake", ".mk")):
                yield Path(current) / filename


def _read_text(path: Path, limit: int = 200_000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return handle.read(limit)
    except OSError:
        return ""


def _symbol_defined_in_repo(repo_path: str, symbol: str) -> bool:
    symbol = _clean_symbol(symbol)
    if not symbol or not repo_path:
        return False
    root = Path(repo_path)
    if not root.is_dir():
        return False
    pattern = re.compile(rf"\b{re.escape(symbol)}\b")
    skip_dirs = {".git", ".venv", "__pycache__", "node_modules", "build"}
    checked = 0
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for filename in files:
            if Path(filename).suffix not in SOURCE_EXTS:
                continue
            checked += 1
            if checked > 800:
                return False
            text = _read_text(Path(current) / filename, limit=80_000)
            if pattern.search(text):
                return True
    return False


def _clean_symbol(symbol: str) -> str:
    symbol = str(symbol or "").strip()
    symbol = re.sub(r"\(.*\)$", "", symbol).strip()
    symbol = symbol.split("@", 1)[0]
    if "::" in symbol:
        symbol = symbol.rsplit("::", 1)[-1]
    return re.sub(r"[^A-Za-z0-9_~]", "", symbol)


def _latest_command_text(command_records: list[dict[str, Any]]) -> str:
    for item in reversed(command_records or []):
        command = str(item.get("command") or item.get("input") or "")
        if command:
            return command
    return ""


def _dirname_fragment(path: str) -> str:
    dirname = os.path.dirname(path)
    return os.path.basename(dirname) if dirname else ""


def _fallback_object_name(symptom: str, frame: dict[str, Any]) -> str:
    keys = {
        "header_not_found": "missing_headers",
        "library_not_found": "missing_libraries",
        "pkg_config_failure": "missing_pkg_configs",
        "cmake_package_failure": "missing_cmake_packages",
        "build_tool_not_found": "missing_tools",
        "source_path_missing": "missing_paths",
    }
    values = frame.get(keys.get(symptom, "")) or []
    return str(values[0]) if values else ""


def _object_kind_for_symptom(symptom: str) -> str:
    return {
        "header_not_found": "header",
        "library_not_found": "library",
        "unresolved_symbol": "function_symbol",
        "pkg_config_failure": "pkg_config_module",
        "cmake_package_failure": "cmake_package",
        "build_tool_not_found": "build_tool",
        "source_path_missing": "path",
    }.get(symptom, "")


def _allowed_revisions_for_decisions(decisions: list[str], existing: list[str]) -> list[str]:
    revisions = list(existing or [])
    mapping = {
        "system_package_provider": ["system_package_provider_switch", "apt_single_provider_reduction"],
        "source_dependency": ["source_dependency_register"],
        "dependency_discovery": ["dependency_discovery_adjustment", "include_path_adjustment"],
        "system_release": ["candidate_release_switch"],
        "build_entry": ["entrypoint_candidate_switch"],
        "build_target": ["target_candidate_switch"],
        "generation": ["known_generation_command"],
        "compiler_configuration": ["build_flag_adjustment", "tool_shim"],
        "linker_configuration": ["link_flag_adjustment", "link_order_adjustment"],
    }
    for decision in decisions:
        revisions.extend(mapping.get(decision, []))
    return _unique(revisions)


def _compact_probe(probe: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "object_kind": probe.get("object_kind") or "",
        "object_name": probe.get("object_name") or "",
        "provenance": probe.get("provenance") or "unknown",
        "confidence": probe.get("confidence") or "low",
        "evidence": list(probe.get("evidence") or [])[:5],
        "preferred_decisions": list(probe.get("preferred_decisions") or [])[:5],
        "discouraged_decisions": list(probe.get("discouraged_decisions") or [])[:5],
    }
    for key in (
        "provider_lookup_required",
        "provider_candidate_ids",
        "source_dependency_candidate_ids",
        "generator_candidate_ids",
        "build_entry_candidate_ids",
        "build_target_candidate_ids",
        "release_candidate_ids",
        "discovery_paths",
        "discovery_flags",
        "link_signal",
        "tool_configuration_signal",
        "version_shortfall_signal",
        "version_relation",
    ):
        value = probe.get(key)
        if value not in (None, False, [], ""):
            compact[key] = value
    return compact


def _attach_candidate_facts(
    probe: dict[str, Any],
    route: DiagnosisRoute,
    build_plan: dict[str, Any],
) -> dict[str, Any]:
    """Attach structured facts consumed by Matrix evidence gates."""
    result = dict(probe or {})
    object_name = str(result.get("object_name") or "").strip()
    provenance = str(result.get("provenance") or "unknown").strip().lower()
    result["provider_lookup_required"] = bool(
        object_name and provenance == "system_or_external"
    )
    result["provider_candidate_ids"] = _verified_provider_candidate_ids(
        build_plan,
        object_name,
    )
    result["source_dependency_candidate_ids"] = _executable_source_candidate_ids(
        build_plan.get("source_dep_candidates") or build_plan.get("source_deps") or [], object_name,
    )
    result["generator_candidate_ids"] = _candidate_ids(
        build_plan.get("generator_candidates") or build_plan.get("generators") or build_plan.get("bootstrap_candidates") or build_plan.get("bootstrap") or [],
        ("generator_id", "candidate_id", "command"),
    )
    result["build_entry_candidate_ids"] = _candidate_ids(
        build_plan.get("entrypoint_candidates") or build_plan.get("entrypoints") or [],
        ("entry_id", "candidate_id", "command"),
    )
    result["build_target_candidate_ids"] = _candidate_ids(
        build_plan.get("target_candidates") or build_plan.get("targets") or [],
        ("target_id", "candidate_id", "name"),
    )
    result["release_candidate_ids"] = _candidate_ids(
        build_plan.get("sysuse_candidates") or build_plan.get("sysuse") or [],
        ("candidate_id", "sysuse"),
    )
    evidence = [str(item) for item in result.get("evidence") or []]
    if provenance == "repo_local_invisible":
        result["discovery_paths"] = [
            item for item in evidence
            if re.search(r"(?:^|\s)(?:[A-Za-z0-9_.+-]+/)+[A-Za-z0-9_.+-]+", item)
        ][:4]
        result["discovery_flags"] = _unique(
            re.findall(r"(?:-I|-L|PKG_CONFIG_PATH|CMAKE_PREFIX_PATH)[^\s]*", " ".join(evidence))
        )[:4]
    else:
        result["discovery_paths"] = []
        result["discovery_flags"] = []
    result["link_signal"] = bool(
        route.failed_phase == "link"
        or provenance == "link_composition"
        or route.symptom_family in {"library_not_found", "unresolved_symbol"}
    )
    result["version_shortfall_signal"] = route.failure_mechanism == "version_shortfall"
    result["version_relation"] = dict(route.version_relation or {})
    relation_subject = str((route.version_relation or {}).get("subject") or "").strip().lower()
    compiler_shortfall = route.failure_mechanism == "version_shortfall" and (
        route.failed_tool == "compiler"
        or relation_subject in {"cc", "c++", "gcc", "g++", "clang", "clang++"}
    )
    result["tool_configuration_signal"] = bool(
        route.symptom_family in {"compiler_policy_error", "api_or_toolchain_mismatch"}
        and route.failed_tool not in {"", "unknown"}
        and (route.failure_mechanism != "version_shortfall" or compiler_shortfall)
    )
    return result


def _verified_provider_candidate_ids(
    build_plan: dict[str, Any],
    object_name: str,
) -> list[str]:
    """Return candidates backed by provider evidence for the active suite."""
    needle = _normalise_candidate_name(object_name)
    current_sysuse = normalize_sysuse(
        str((build_plan.get("selected") or {}).get("sysuse") or "")
    )
    evidence_ids = {
        str(item.get("evidence_id") or "")
        for item in build_plan.get("provider_evidence") or []
        if isinstance(item, dict)
        and item.get("available")
        and (item.get("availability_probe") or {}).get("status") == "available"
        and (item.get("availability_probe") or {}).get("exit_code") in {0, "0"}
        and (
            (item.get("precondition_probe") or {}).get("observed_missing")
            or (item.get("precondition_probe") or {}).get("observed_unsatisfied")
        )
        and normalize_sysuse(str((item.get("environment") or {}).get("sysuse") or "")) == current_sysuse
        and _candidate_name_matches(
            needle,
            _normalise_candidate_name((item.get("entity") or {}).get("name")),
        )
        and item.get("evidence_id")
    }
    candidates = [
        item
        for item in build_plan.get("system_dep_candidates") or []
        if isinstance(item, dict)
        and str(item.get("provider_evidence_id") or "") in evidence_ids
    ]
    return _candidate_ids(candidates, ("candidate_id", "provider_id", "name"))


def _matching_candidate_ids(
    candidates: list[Any],
    object_name: str,
    keys: tuple[str, ...],
) -> list[str]:
    if not object_name:
        return []
    needle = _normalise_candidate_name(object_name)
    matched = []
    for item in candidates:
        if not isinstance(item, dict) or item.get("status") in {"failed", "exhausted", "selected", "current", "ineligible"}:
            continue
        candidate_names = [
            item.get("name"),
            item.get("package"),
            item.get("provider"),
            (item.get("provider") or {}).get("package") if isinstance(item.get("provider"), dict) else "",
        ]
        if any(_candidate_name_matches(needle, _normalise_candidate_name(value)) for value in candidate_names):
            matched.extend(_candidate_ids([item], keys))
    return _unique(matched)


def _normalise_candidate_name(value: Any) -> str:
    value = str(value or "").strip().lower()
    value = os.path.basename(value.replace("\\", "/"))
    return re.sub(r"[^a-z0-9+.-]+", "", value)


def _candidate_name_matches(needle: str, candidate: str) -> bool:
    if not needle or not candidate:
        return False
    if needle == candidate:
        return True
    needle_core = needle[3:] if needle.startswith("lib") else needle
    candidate_core = candidate[3:] if candidate.startswith("lib") else candidate
    if needle_core == candidate_core:
        return True
    # Match provider suffixes such as zlib -> zlib1g-dev or ssl -> libssl-dev,
    # but never arbitrary substrings in a free-form reason field.
    return bool(
        candidate_core.startswith(needle_core + "-")
        or candidate_core.startswith(needle_core + ".")
        or candidate_core.startswith(needle_core + "[")
    )


def _executable_source_candidate_ids(candidates: list[Any], object_name: str) -> list[str]:
    if not object_name:
        return []
    parts = [part.lower() for part in str(object_name).replace("\\", "/").split("/") if part]
    owner = ""
    for index, part in enumerate(parts[:-1]):
        if part in {"third_party", "3rdparty", "deps", "vendor", "external", "submodule", "submodules"}:
            owner = parts[index + 1]
            break
    needles = {
        _normalise_candidate_name(str(object_name)),
        _normalise_candidate_name(os.path.basename(str(object_name))),
        _normalise_candidate_name(owner),
    } - {""}
    eligible = [
        item for item in candidates
        if isinstance(item, dict)
        and item.get("build_order")
        and (item.get("container_source_path") or item.get("source_path") or item.get("submodule_path"))
        and any(
            _candidate_name_matches(needle, _normalise_candidate_name(item.get("name")))
            or _candidate_name_matches(needle, _normalise_candidate_name(item.get("source_path")))
            or _candidate_name_matches(needle, _normalise_candidate_name(item.get("submodule_path")))
            for needle in needles
        )
    ]
    return _candidate_ids(eligible, ("candidate_id", "source_dep_id", "name"))


def _candidate_ids(candidates: list[Any], keys: tuple[str, ...]) -> list[str]:
    result = []
    for item in candidates:
        if not isinstance(item, dict) or item.get("status") in {"failed", "exhausted", "selected", "current", "ineligible"}:
            continue
        value = next((str(item.get(key) or "").strip() for key in keys if item.get(key)), "")
        if not value:
            continue
        if not re.match(r"^[a-z][a-z0-9_:-]{2,80}$", value, re.I):
            digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:12]
            value = f"candidate:{digest}"
        result.append(value)
    return _unique(result)[:8]


def _unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
