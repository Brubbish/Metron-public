"""Evidence-backed build-family candidates for sparse Phase-1 scans.

The deterministic scanners usually find standard autotools/cmake/meson/make
entries.  This module covers a small set of generic build families that are
common in C/C++ repositories but are not represented by a single conventional
root marker.  It only emits existing build-entry candidate dictionaries; the
normal ranker still owns eligibility and ordering.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable


_THIRD_PARTY_PARTS = {
    "third_party",
    "third-party",
    "3rdparty",
    "vendor",
    "vendors",
    "external",
    "deps",
    "dependencies",
    "submodules",
}
_TEST_PARTS = {"test", "tests", "benchmark", "benchmarks", "example", "examples"}
_CI_PARTS = {".github", ".circleci", "ci"}


def _clean_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").removeprefix("./")


def _dirname(path: str) -> str:
    directory = os.path.dirname(path)
    return directory or "."


def _basename(path: str) -> str:
    return os.path.basename(path.rstrip("/"))


def _join(directory: str, name: str) -> str:
    return name if directory in {"", "."} else f"{directory.rstrip('/')}/{name}"


def _path_parts(path: str) -> list[str]:
    return [part for part in _clean_path(path).split("/") if part]


def _scope_for_path(path: str, preferred: str = "module_root") -> str:
    parts = [part.lower() for part in _path_parts(path)]
    directory_parts = parts[:-1] if "." in _basename(path) else parts
    lowered = set(directory_parts)
    if lowered & _THIRD_PARTY_PARTS:
        return "third_party"
    if lowered & _TEST_PARTS:
        return "test"
    if lowered & _CI_PARTS:
        return "ci"
    if len(directory_parts) == 0:
        return "root"
    return preferred


def _inventory_files(manifest_inventory: dict[str, Any]) -> set[str]:
    files: set[str] = set()
    for item in manifest_inventory.get("files") or []:
        if not isinstance(item, dict):
            continue
        path = _clean_path(item.get("path"))
        if path:
            files.add(path)
    for path in manifest_inventory.get("manifest_paths") or []:
        cleaned = _clean_path(path)
        if cleaned:
            files.add(cleaned)
    return files


def _existing_files(repo_root: str | Path, files: Iterable[str]) -> set[str]:
    root = Path(repo_root)
    existing: set[str] = set()
    for path in files:
        cleaned = _clean_path(path)
        if cleaned and (root / cleaned).is_file():
            existing.add(cleaned)
    return existing


def _candidate(
    *,
    family: str,
    entry_path: str,
    workdir: str,
    scope: str,
    evidence_paths: Iterable[str],
    command_steps: list[dict[str, str]],
    notes: Iterable[str] = (),
    confidence: str = "medium",
) -> dict[str, Any]:
    evidence = list(dict.fromkeys(_clean_path(path) for path in evidence_paths if _clean_path(path)))
    return {
        "entry_path": _clean_path(entry_path),
        "evidence_paths": evidence,
        "source": "family_resolver",
        "confidence": confidence,
        "family": family,
        "build_system": family,
        "scope": scope,
        "independent_project": scope in {"workspace_root", "module_root"},
        "command_steps": command_steps,
        "candidate_notes": list(dict.fromkeys(str(note) for note in notes if str(note).strip())),
        "root_evidence": [f"family:{family}", *(f"evidence:{path}" for path in evidence[:4])],
    }


def _qmake_candidates(files: set[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    pro_files = sorted(path for path in files if path.endswith(".pro"))
    for pro_path in pro_files[:8]:
        directory = _dirname(pro_path)
        evidence = [pro_path]
        for marker in (".qmake.conf", "sync.profile"):
            marker_path = _join(directory, marker)
            if marker_path in files:
                evidence.append(marker_path)
        command = f"qmake {_basename(pro_path)}"
        candidates.append(
            _candidate(
                family="qmake",
                entry_path=pro_path,
                workdir=directory,
                scope=_scope_for_path(pro_path, "module_root"),
                evidence_paths=evidence,
                command_steps=[
                    {"command": command, "workdir": directory, "stage": "configure"},
                    {"command": "make", "workdir": directory, "stage": "compile"},
                ],
                confidence="high" if directory == "." else "medium",
            )
        )
    return candidates


def _python_configure_candidates(files: set[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in sorted(item for item in files if _basename(item) == "configure.py")[:6]:
        directory = _dirname(path)
        candidates.append(
            _candidate(
                family="python_configure",
                entry_path=path,
                workdir=directory,
                scope=_scope_for_path(path, "module_root"),
                evidence_paths=[path],
                command_steps=[
                    {"command": "python3 configure.py", "workdir": directory, "stage": "configure"},
                    {"command": "make", "workdir": directory, "stage": "compile"},
                ],
                confidence="high" if directory == "." else "medium",
            )
        )
    return candidates


def _gn_candidates(files: set[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    directories = sorted({
        _dirname(path)
        for path in files
        if path in {".gn", "BUILD.gn"} or path.endswith("/BUILD.gn")
    }, key=lambda item: (0 if item == "." else item.count("/") + 1, item))
    for directory in directories[:6]:
        entry_path = _join(directory, ".gn") if _join(directory, ".gn") in files else _join(directory, "BUILD.gn")
        evidence = [entry_path]
        for sibling in (".gn", "DEPS", "script/bootstrap.py", "script/build.py"):
            sibling_path = _join(directory, sibling)
            if sibling_path in files:
                evidence.append(sibling_path)
        build_gn = _join(directory, "BUILD.gn")
        if build_gn in files:
            evidence.append(build_gn)
        steps: list[dict[str, str]] = []
        if _join(directory, "script/bootstrap.py") in files:
            steps.append({"command": "python3 script/bootstrap.py", "workdir": directory, "stage": "bootstrap"})
        if _join(directory, "script/build.py") in files:
            steps.append({"command": "python3 script/build.py", "workdir": directory, "stage": "compile"})
        else:
            steps.extend([
                {"command": "gn gen out/Default", "workdir": directory, "stage": "configure"},
                {"command": "ninja -C out/Default", "workdir": directory, "stage": "compile"},
            ])
        notes = ["requires_external_workspace_deps"] if _join(directory, "DEPS") in files else []
        candidates.append(
            _candidate(
                family="gn_ninja",
                entry_path=entry_path,
                workdir=directory,
                scope=_scope_for_path(entry_path, "module_root"),
                evidence_paths=evidence,
                command_steps=steps,
                notes=notes,
                confidence="high" if directory == "." else "medium",
            )
        )
    return candidates


def _workspace_make_candidates(files: set[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    workspace_markers = [
        path
        for path in sorted(files)
        if _basename(path) == "Makefile.master" or path == "usr/src/Makefile"
    ]
    for path in workspace_markers[:6]:
        directory = _dirname(path)
        candidates.append(
            _candidate(
                family="workspace_make",
                entry_path=path,
                workdir=directory,
                scope="workspace_root" if _scope_for_path(path) not in {"test", "third_party", "ci"} else _scope_for_path(path),
                evidence_paths=[path],
                command_steps=[{"command": "make", "workdir": directory, "stage": "compile"}],
                confidence="medium",
            )
        )
    return candidates


def resolve_build_families(
    repo_root: str | Path,
    manifest_inventory: dict[str, Any],
    docs_summary: Any = None,
) -> list[dict[str, Any]]:
    """Return generic family candidates backed by repository evidence.

    ``docs_summary`` is accepted for the public Phase-1 call signature; this
    resolver intentionally keeps v1 decisions based on concrete repository
    files so it cannot invent project-specific commands from prose.
    """
    del docs_summary
    files = _existing_files(repo_root, _inventory_files(manifest_inventory))
    candidates: list[dict[str, Any]] = []
    candidates.extend(_qmake_candidates(files))
    candidates.extend(_python_configure_candidates(files))
    candidates.extend(_gn_candidates(files))
    candidates.extend(_workspace_make_candidates(files))

    by_key: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    for item in candidates:
        key = (
            str(item.get("family") or ""),
            str(item.get("entry_path") or ""),
            tuple(str(step.get("command") or "") for step in item.get("command_steps") or []),
        )
        by_key.setdefault(key, item)
    return list(by_key.values())
