"""Deterministic build-entry candidate ranking.

The repository scanners produce evidence from several sources.  This module
keeps each source's command sequence intact and ranks the resulting entries
before Phase 2 is created.  The LLM may explain or refine a selected entry,
but it does not decide the initial ordering.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional


_PROVENANCE_RANK = {
    "cicd": 0,
    "ci_explicit": 0,
    "doc_explicit": 1,
    "readrepo_explicit": 1,
    "debian_rules": 1,
    "script_detected": 2,
    "project_script": 2,
    "family_resolver": 2,
    "structural": 3,
    "inferred": 3,
}
_CONFIDENCE_RANK = {"main": 4, "high": 3, "medium": 2, "low": 1}
_BUILD_FILE_BY_FAMILY = {
    "cmake": "CMakeLists.txt",
    "meson": "meson.build",
    "autotools": "configure",
    "make": "Makefile",
    "scons": "SConstruct",
    "waf": "wscript",
    "xmake": "xmake.lua",
}
_DOC_EVIDENCE_NAMES = {
    "readme", "install", "installing", "building", "build", "build.md", "contributing",
}
_PLACEHOLDER_RE = re.compile(r"<[^>]+>|\{\{?[^{}]+\}?\}|\$\([^)]+\)")
_BUILD_COMMAND_RE = re.compile(
    r"\b(?:make|gmake|ninja)\b|\bcmake\s+--build\b|\bmeson\s+compile\b|"
    r"\brake\s+(?:compile|build|native|extension)\b|\bgem\s+build\b|"
    r"\b(?:python|python3)\s+setup\.py\s+build\b|"
    r"\bscons\b|(?:^|\s)(?:\./)?waf\s+build\b|\bxmake(?:\s|$)|"
    r"\b(?:gcc|g\+\+|clang|clang\+\+|cc|c\+\+)\b|"
    r"(?:^|\s)(?:(?:bash|sh)\s+)?\./[^\s]*(?:build|compile)[^\s]*",
    re.I,
)
_TEST_COMMAND_RE = re.compile(r"\b(?:ctest|pytest|make\s+(?:check|test)|ninja\s+test)\b", re.I)
_INSTALL_COMMAND_RE = re.compile(r"\b(?:make|ninja)\s+install\b|\bcmake\s+--install\b", re.I)
_TEST_ONLY_COMMAND_RE = re.compile(
    r"^(?:(?:cd\s+[^;&]+\s*&&\s*)*)(?:ctest|pytest|(?:make|gmake|ninja)\s+(?:check|test))(?:\s|$)",
    re.I,
)
_INSTALL_ONLY_COMMAND_RE = re.compile(
    r"^(?:(?:cd\s+[^;&]+\s*&&\s*)*)(?:(?:make|gmake|ninja)\s+install|cmake\s+--install)(?:\s|$)",
    re.I,
)
_ENV_REFERENCE_RE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
_KNOWN_RUNTIME_ENV = frozenset({"CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "MAKEFLAGS", "PATH"})
_ROOT_PROJECT_MARKERS = frozenset({
    "CMakeLists.txt", "meson.build", "configure", "configure.ac", "configure.in",
    "Makefile", "Makefile.am", "SConstruct", "wscript", "xmake.lua",
    "README", "README.md", "README.rst", "pyproject.toml", "package.json",
    ".qmake.conf", "sync.profile", ".gn", "BUILD.gn", "DEPS",
    "configure.py", "Makefile.master", "usr/src/Makefile",
})
_NON_ROOT_DIRS = frozenset({
    "third_party", "third-party", "3rdparty", "vendor", "vendors", "external",
    "deps", "dependencies", "submodules", "test", "tests", "benchmark", "benchmarks",
    "examples", "example", "ci", ".github", ".circleci", "freebsd", "openwrt", "ports",
    "package", "packages",
})


def _command_sequence(candidate: dict[str, Any]) -> list[str]:
    sequence = candidate.get("command_sequence")
    if isinstance(sequence, list):
        commands = [str(item) for item in sequence if str(item).strip()]
    else:
        commands = []
    if not commands:
        command = str(candidate.get("command") or "")
        if command.strip():
            commands = [command]
    return commands


def _normalise_workdir(value: Any) -> str:
    workdir = str(value or ".").strip().replace("\\", "/") or "."
    if workdir.startswith("/work/"):
        workdir = workdir[len("/work/"):]
    elif workdir == "/work":
        workdir = "."
    workdir = os.path.normpath(workdir).replace("\\", "/")
    if workdir == ".." or workdir.startswith("../") or os.path.isabs(workdir):
        return "."
    return workdir


def _stage_for_command(command: str) -> str:
    lowered = str(command or "").lower()
    if re.search(r"\bapt(?:-get)?\s+install\b|\bdpkg\b|\bpip\s+install\b|\bnpm\s+install\b", lowered):
        return "setup"
    if re.search(r"autoreconf|autogen\.sh|bootstrap(?:\.sh)?|buildconf|aclocal|autoheader", lowered):
        return "bootstrap"
    if re.search(r"\./configure|cmake\s+(?:-s|-b|\.|\.\.)|meson\s+setup|\bqmake\b", lowered):
        return "configure"
    if re.search(r"\b(?:make|ninja|cmake\s+--build|meson\s+compile|gcc|g\+\+|clang|clang\+\+)\b", lowered):
        return "compile"
    return "unknown"


def _command_steps(candidate: dict[str, Any]) -> list[dict[str, str]]:
    raw_steps = candidate.get("command_steps")
    steps: list[dict[str, str]] = []
    if isinstance(raw_steps, list):
        current_workdir = _normalise_workdir(candidate.get("workdir") or ".")
        for raw in raw_steps:
            if not isinstance(raw, dict):
                continue
            command = str(raw.get("command") or "")
            if command.strip():
                workdir = _normalise_workdir(
                    raw.get("workdir") if raw.get("workdir") is not None else current_workdir
                )
                steps.append({
                    "command": command,
                    "workdir": workdir,
                    "stage": str(raw.get("stage") or _stage_for_command(command)),
                })
                current_workdir = _workdir_after_command(command, workdir)
    if steps:
        return steps
    current_workdir = _normalise_workdir(candidate.get("workdir") or ".")
    for command in _command_sequence(candidate):
        steps.append({"command": command, "workdir": current_workdir, "stage": _stage_for_command(command)})
        current_workdir = _workdir_after_command(command, current_workdir)
    return steps


def _workdir_after_command(command: str, initial_workdir: str) -> str:
    workdir = initial_workdir
    for match in re.finditer(r"(?:^|&&)\s*cd\s+((?:'[^']*'|\"[^\"]*\"|[^&;\s]+))", command):
        try:
            parts = shlex.split(match.group(1))
        except ValueError:
            parts = []
        if len(parts) != 1:
            continue
        if parts[0].startswith("/work/") or parts[0] == "/work":
            workdir = _normalise_workdir(parts[0])
        elif not os.path.isabs(parts[0]):
            workdir = _normalise_workdir(os.path.join(workdir, parts[0]))
    return workdir


def _build_file_path(directory: str, filename: str) -> str:
    return filename if directory in {"", "."} else f"{directory.rstrip('/')}/{filename}"


def _cmake_positional_source(tokens: list[str]) -> str:
    if not tokens or "--build" in tokens or "--install" in tokens:
        return ""
    options_with_value = {
        "-B", "-G", "-A", "-T", "-C", "-U",
        "--preset", "--toolchain", "--install-prefix",
    }
    positional: list[str] = []
    skip_next = False
    for token in tokens[1:]:
        if skip_next:
            skip_next = False
            continue
        if token in options_with_value:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        positional.append(token)
    return positional[-1] if positional else ""


def _explicit_path_from_command(command: str, workdir: str) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    if not tokens:
        return ""
    for index, token in enumerate(tokens):
        if token == "-S" and index + 1 < len(tokens) and tokens[0].endswith("cmake"):
            source = _normalise_workdir(os.path.join(workdir, tokens[index + 1]))
            return _build_file_path(source, "CMakeLists.txt")
        if token.startswith("-S") and len(token) > 2 and tokens[0].endswith("cmake"):
            source = _normalise_workdir(os.path.join(workdir, token[2:]))
            return _build_file_path(source, "CMakeLists.txt")
        if token == "-C" and index + 1 < len(tokens) and tokens[0].endswith("make"):
            directory = _normalise_workdir(os.path.join(workdir, tokens[index + 1]))
            return _build_file_path(directory, "Makefile")
    if tokens[0].endswith("cmake"):
        positional_source = _cmake_positional_source(tokens)
        if positional_source:
            source = _normalise_workdir(os.path.join(workdir, positional_source))
            return _build_file_path(source, "CMakeLists.txt")
    script_match = re.search(
        r"(?:^|\s)(?:(?:bash|sh)\s+)?(?:\./)?([A-Za-z0-9_./+-]+\.(?:sh|build))(?:\s|$)",
        command,
    )
    if script_match:
        return _normalise_workdir(os.path.join(workdir, script_match.group(1)))
    if re.search(r"(?:^|\s)\./configure(?:\s|$)", command):
        return _build_file_path(workdir, "configure")
    executable_match = re.match(r"^(?:(?:bash|sh)\s+)?\./([A-Za-z0-9_./+-]+)(?:\s|$)", command)
    if executable_match:
        return _normalise_workdir(os.path.join(workdir, executable_match.group(1)))
    return ""


def _path_from_commands(steps: list[dict[str, str]], family: str) -> str:
    for step in steps:
        workdir = step["workdir"]
        for command in re.split(r"\s*&&\s*", step["command"]):
            cd_match = re.match(r"^cd\s+(.+?)\s*$", command)
            if cd_match:
                try:
                    cd_parts = shlex.split(cd_match.group(1))
                except ValueError:
                    cd_parts = []
                if len(cd_parts) == 1:
                    if cd_parts[0].startswith("/work/") or cd_parts[0] == "/work":
                        workdir = _normalise_workdir(cd_parts[0])
                    elif not os.path.isabs(cd_parts[0]):
                        workdir = _normalise_workdir(os.path.join(workdir, cd_parts[0]))
                continue
            entry_path = _explicit_path_from_command(command.strip(), workdir)
            if entry_path:
                return entry_path
    filename = _BUILD_FILE_BY_FAMILY.get(family)
    return _build_file_path(steps[0]["workdir"], filename) if filename and steps else ""


def _entry_id(candidate: dict[str, Any]) -> str:
    payload = "\x1f".join(
        [
            str(candidate.get("entry_path") or candidate.get("path") or ""),
            "\x1e".join(
                f"{item.get('workdir', '.')}\x1d{item.get('command', '')}"
                for item in candidate.get("command_steps") or []
            ),
        ]
    )
    return "entry_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _infer_build_system(steps: list[dict[str, str]], declared: str) -> str:
    family = str(declared or "").strip().lower()
    if family:
        return family
    command_text = "\n".join(str(item.get("command") or "") for item in steps).lower()
    if re.search(r"\bcmake\b", command_text):
        return "cmake"
    if re.search(r"\bmeson\b", command_text):
        return "meson"
    if re.search(r"autoreconf|autogen\.sh|\./configure|\bconfigure\b", command_text):
        return "autotools"
    if re.search(r"\bg?make\b", command_text):
        return "make"
    if re.search(r"\bscons\b", command_text):
        return "scons"
    if re.search(r"(?:^|\s)(?:\./)?waf\b", command_text):
        return "waf"
    if re.search(r"\bxmake\b", command_text):
        return "xmake"
    if re.search(r"(?:^|\s)(?:(?:bash|sh)\s+)?\./", command_text):
        return "script"
    return "unknown"


def _is_evidence_path(path: str, source: str) -> bool:
    normalized = str(path or "").strip().replace("\\", "/").lower()
    basename = os.path.basename(normalized)
    if source in {"cicd", "ci_explicit"}:
        return bool(
            normalized.startswith((".github/", ".circleci/"))
            or basename in {".gitlab-ci.yml", ".travis.yml", "appveyor.yml"}
            or basename.endswith((".yml", ".yaml"))
        )
    if source == "doc_explicit":
        stem = basename.rsplit(".", 1)[0]
        return bool(
            normalized.startswith(("doc/", "docs/"))
            or basename.endswith((".md", ".rst", ".txt"))
            or basename in _DOC_EVIDENCE_NAMES
            or stem in _DOC_EVIDENCE_NAMES
        )
    return False


def _path_depth(path: str) -> int:
    value = str(path or "").replace("\\", "/").strip("/")
    if not value or value == ".":
        return 0
    # A file at src/CMakeLists.txt is a depth-1 entry.
    return max(0, len(value.split("/")) - 1)


def _path_parts(path: str) -> list[str]:
    value = str(path or "").replace("\\", "/").strip("/")
    return [] if not value or value == "." else [part for part in value.split("/") if part]


def _entry_semantics(candidate: dict[str, Any], repo_path: str = "") -> dict[str, Any]:
    """Annotate ownership without rejecting valid nested project entries."""
    entry_path = str(candidate.get("entry_path") or candidate.get("path") or "").strip()
    parts = _path_parts(entry_path)
    source = str(candidate.get("source") or "").lower()
    evidence_paths = [str(item).replace("\\", "/") for item in candidate.get("evidence_paths") or []]
    basename = Path(entry_path).name
    scope = str(candidate.get("scope") or "").strip().lower()
    if scope not in {"root", "workspace_root", "module_root", "subproject", "third_party", "test", "ci"}:
        scope = "root" if len(parts) <= 1 else "subproject"
        lowered_parts = {part.lower() for part in parts[:-1]}
        if lowered_parts & _NON_ROOT_DIRS:
            scope = "third_party" if lowered_parts & {
                "third_party", "third-party", "3rdparty", "vendor", "vendors", "external",
                "deps", "dependencies", "freebsd", "openwrt", "ports", "package", "packages",
            } else "test" if lowered_parts & {
                "test", "tests", "benchmark", "benchmarks", "examples", "example",
            } else "ci"
    if source in {"cicd", "ci_explicit"}:
        scope = "ci"
    root_evidence = [str(item) for item in candidate.get("root_evidence") or [] if str(item).strip()]
    if len(parts) <= 1 and basename:
        root_evidence.append(f"entry_at_root:{basename}")
    if source in {"readrepo_explicit", "doc_explicit", "debian_rules", "project_script", "family_resolver"}:
        root_evidence.append(f"declared_source:{source}")
    root_markers = []
    subproject_markers = []
    if repo_path:
        root = Path(repo_path)
        for marker in sorted(_ROOT_PROJECT_MARKERS):
            if (root / marker).exists():
                root_markers.append(marker)
        root_evidence.extend(f"root_marker:{marker}" for marker in root_markers[:8])
        if len(parts) > 1:
            entry_dir = root.joinpath(*parts[:-1])
            for marker in sorted(_ROOT_PROJECT_MARKERS):
                if (entry_dir / marker).exists():
                    subproject_markers.append(marker)
            root_evidence.extend(f"subproject_marker:{marker}" for marker in subproject_markers[:6])
    build_step = any(
        (
            str(item.get("stage") or "").strip().lower() in {"build", "compile", "link"}
            or _BUILD_COMMAND_RE.search(str(item.get("command") or ""))
        )
        and not _TEST_ONLY_COMMAND_RE.search(str(item.get("command") or ""))
        and not _INSTALL_ONLY_COMMAND_RE.search(str(item.get("command") or ""))
        for item in candidate.get("command_steps") or []
    )
    prepare_step = any(str(item.get("stage") or "") in {"bootstrap", "configure"} for item in candidate.get("command_steps") or [])
    explicit_independent = bool(
        candidate.get("independent_project")
        or candidate.get("subproject")
        or subproject_markers
        or any("subproject" in item.lower() or "standalone" in item.lower() for item in evidence_paths)
    )
    evidence_backed_entry = bool(
        build_step
        and evidence_paths
        and source in {"readrepo_explicit", "doc_explicit", "debian_rules", "project_script", "family_resolver"}
    )
    return {
        "scope": scope,
        "root_evidence": list(dict.fromkeys(root_evidence))[:12],
        "entry_preflight": {
            "project_marker": bool(root_markers) or len(parts) <= 1 or explicit_independent or evidence_backed_entry,
            "build_step": bool(build_step),
            "prepare_step": bool(prepare_step),
            "independent_project": explicit_independent,
            "evidence_backed_entry": evidence_backed_entry,
        },
    }


def _last_update_at_commit(repo_path: str, commit: str, path: str) -> int:
    if not repo_path or not path:
        return 0
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ct", commit or "HEAD", "--", path],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    if result.returncode != 0:
        return 0
    try:
        return int(result.stdout.strip() or 0)
    except ValueError:
        return 0


def _normalise_candidate(raw: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    steps = _command_steps(raw)
    if not steps:
        return None
    candidate = dict(raw)
    candidate["command_steps"] = steps
    candidate["command_sequence"] = [item["command"] for item in steps]
    candidate["command"] = steps[0]["command"]
    candidate["source"] = str(
        candidate.get("source") or candidate.get("provenance") or "inferred"
    ).strip().lower()
    family = _infer_build_system(
        steps,
        str(candidate.get("build_system") or candidate.get("family") or ""),
    )
    evidence_paths = candidate.get("evidence_paths") or []
    if not isinstance(evidence_paths, list):
        evidence_paths = [evidence_paths]
    evidence_path = str(candidate.get("evidence_path") or "").strip()
    if evidence_path:
        evidence_paths.append(evidence_path)
    candidate["evidence_paths"] = list(dict.fromkeys(str(item).strip() for item in evidence_paths if str(item).strip()))
    candidate["entry_path"] = str(candidate.get("entry_path") or "").strip()
    evidence_only_entry = _is_evidence_path(candidate["entry_path"], candidate["source"])
    if evidence_only_entry:
        candidate["evidence_paths"] = list(dict.fromkeys(candidate["evidence_paths"] + [candidate["entry_path"]]))
        candidate["entry_path"] = ""
    if not candidate["entry_path"]:
        candidate["entry_path"] = _path_from_commands(steps, family)
    if not candidate["entry_path"] and not evidence_only_entry:
        candidate["entry_path"] = str(candidate.get("path") or "").strip()
    candidate["build_system"] = family or "unknown"
    candidate["confidence"] = str(candidate.get("confidence") or "unknown").lower()
    candidate["entry_id"] = _entry_id(candidate)
    candidate["path_depth"] = int(candidate.get("path_depth") or _path_depth(candidate["entry_path"]))
    purposes, eligibility_reasons = _candidate_purpose(steps)
    candidate["purposes"] = purposes
    candidate["required_env"] = [
        name
        for reason in eligibility_reasons
        if reason.startswith("requires_unbound_env:")
        for name in reason.split(":", 1)[1].split(",")
        if name
    ]
    candidate["eligible"] = not eligibility_reasons
    candidate["eligibility_reasons"] = eligibility_reasons
    semantics = _entry_semantics(candidate)
    candidate.update(semantics)
    return candidate


def _candidate_purpose(steps: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    commands = [str(item.get("command") or "").strip() for item in steps]
    text = "\n".join(commands)
    purposes = []
    if any(str(item.get("stage") or "") == "setup" for item in steps):
        purposes.append("setup")
    if any(str(item.get("stage") or "") in {"bootstrap", "configure"} for item in steps):
        purposes.append("prepare")
    has_build_step = any(
        (
            str(step.get("stage") or "").strip().lower() in {"build", "compile", "link"}
            or _BUILD_COMMAND_RE.search(command)
        )
        and not _TEST_ONLY_COMMAND_RE.search(command)
        and not _INSTALL_ONLY_COMMAND_RE.search(command)
        for step, command in zip(steps, commands)
    )
    if has_build_step:
        purposes.append("build")
    if _TEST_COMMAND_RE.search(text):
        purposes.append("test")
    if _INSTALL_COMMAND_RE.search(text):
        purposes.append("install")
    reasons = []
    if _PLACEHOLDER_RE.search(text):
        reasons.append("contains_placeholder")
    if re.search(r"\bgit\s+clone\b", text, re.I):
        reasons.append("contains_repository_clone")
    if "build" not in purposes:
        reasons.append("no_build_step")
    defined_env = {
        match.group(1)
        for command in commands
        for match in re.finditer(r"(?:^|[;&]\s)([A-Za-z_][A-Za-z0-9_]*)=", command)
    }
    referenced_env = {
        match.group(1) or match.group(2)
        for command in commands
        for match in _ENV_REFERENCE_RE.finditer(command)
    }
    required_env = sorted(referenced_env - defined_env - _KNOWN_RUNTIME_ENV)
    if required_env:
        reasons.append("requires_unbound_env:" + ",".join(required_env[:8]))
    return purposes, reasons


def _merge_candidate(existing: dict[str, Any], candidate: dict[str, Any]) -> None:
    existing["evidence_paths"] = list(dict.fromkeys(
        list(existing.get("evidence_paths") or []) + list(candidate.get("evidence_paths") or [])
    ))
    if _PROVENANCE_RANK.get(candidate["source"], 4) < _PROVENANCE_RANK.get(existing["source"], 4):
        existing["source"] = candidate["source"]
    if _CONFIDENCE_RANK.get(candidate["confidence"], 0) > _CONFIDENCE_RANK.get(existing["confidence"], 0):
        existing["confidence"] = candidate["confidence"]


def rank_build_entry_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    repo_path: str = "",
    target_commit: str = "HEAD",
) -> list[dict[str, Any]]:
    """Return stable candidates ordered according to the paper's policy."""
    by_id: dict[str, dict[str, Any]] = {}
    for raw in candidates:
        candidate = _normalise_candidate(raw)
        if candidate is None:
            continue
        existing = by_id.get(candidate["entry_id"])
        if existing is not None:
            _merge_candidate(existing, candidate)
            continue
        by_id[candidate["entry_id"]] = candidate
    normalised = list(by_id.values())
    for candidate in normalised:
        candidate.update(_entry_semantics(candidate, repo_path))
        candidate["source_rank"] = _PROVENANCE_RANK.get(candidate["source"], 4)
        candidate["last_update_at_commit"] = int(
            candidate.get("last_update_at_commit")
            or _last_update_at_commit(repo_path, target_commit, candidate["entry_path"])
        )
        preflight = candidate.get("entry_preflight") or {}
        scope = str(candidate.get("scope") or "unknown")
        if scope == "root" and preflight.get("build_step") and preflight.get("project_marker"):
            candidate["semantic_rank"] = 0
        elif scope == "root" and preflight.get("build_step"):
            candidate["semantic_rank"] = 1
        elif scope in {"workspace_root", "module_root"} and preflight.get("build_step"):
            candidate["semantic_rank"] = 2
        elif scope == "subproject" and preflight.get("independent_project") and preflight.get("build_step"):
            candidate["semantic_rank"] = 2
        elif scope == "subproject" and preflight.get("evidence_backed_entry") and preflight.get("build_step"):
            candidate["semantic_rank"] = 2
        else:
            candidate["semantic_rank"] = 3
        if scope in {"ci", "test", "third_party"} or (
            scope == "subproject"
            and not preflight.get("independent_project")
            and not preflight.get("evidence_backed_entry")
        ):
            candidate["eligible"] = False
            reasons = list(candidate.get("eligibility_reasons") or [])
            reason = f"non_main_entry_scope:{scope}"
            if reason not in reasons:
                reasons.append(reason)
            candidate["eligibility_reasons"] = reasons
        candidate["rank_reason"] = (
            f"semantic_rank={candidate['semantic_rank']}; "
            f"scope={candidate.get('scope', 'unknown')}; "
            f"source_rank={candidate['source_rank']}; "
            f"path_depth={candidate['path_depth']}; "
            f"last_update_at_commit={candidate['last_update_at_commit']}"
        )

    normalised.sort(
        key=lambda item: (
            not bool(item.get("eligible")),
            item.get("semantic_rank", 9),
            item["source_rank"],
            item["path_depth"],
            -item["last_update_at_commit"],
            -_CONFIDENCE_RANK.get(item["confidence"], 0),
            item["entry_path"],
            item["entry_id"],
        )
    )
    selected = False
    for index, candidate in enumerate(normalised):
        candidate["rank"] = index
        if candidate.get("eligible") and not selected:
            candidate["status"] = "selected"
            selected = True
        else:
            candidate["status"] = "candidate" if candidate.get("eligible") else "ineligible"
    return normalised


def materialize_build_entry(candidate: dict[str, Any], repo_root: str = "/work") -> dict[str, Any]:
    """Prepare command steps without changing their shell text.

    ``workdir`` is a repository-relative execution attribute.  The Docker
    executor owns mounting ``/work`` and applying that attribute, so adding a
    ``cd /work`` prefix here would make a later workdir application ambiguous.
    ``repo_root`` remains part of the public signature for existing callers.
    """
    del repo_root
    materialized = dict(candidate or {})
    materialized_steps: list[dict[str, str]] = []
    for raw_step in candidate.get("command_steps") or []:
        if not isinstance(raw_step, dict):
            continue
        command = str(raw_step.get("command") or "")
        if not command.strip():
            continue
        materialized_steps.append({
            "command": command,
            "workdir": _normalise_workdir(raw_step.get("workdir") or "."),
            "stage": str(raw_step.get("stage") or _stage_for_command(command)),
        })
    materialized["command_steps"] = materialized_steps
    materialized["command_sequence"] = [item["command"] for item in materialized_steps]
    materialized["command"] = materialized["command_sequence"][0] if materialized_steps else ""
    return materialized


def select_build_entry(
    candidates: Iterable[dict[str, Any]],
    *,
    repo_path: str = "",
    target_commit: str = "HEAD",
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    ranked = rank_build_entry_candidates(
        candidates,
        repo_path=repo_path,
        target_commit=target_commit,
    )
    selected = next((item for item in ranked if item.get("status") == "selected"), None)
    alternatives = [item for item in ranked if item is not selected]
    return selected, alternatives
