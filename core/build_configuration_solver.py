"""Deterministic system-release and realization selection.

The solver accepts normalized realization candidates from the dependency
extractor.  It is intentionally independent of Docker and LLMs so the
feasibility and ranking policy can be tested with synthetic release data.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional


def _versioned_date(value: Any) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _release_distance(release: dict[str, Any], commit_date: str) -> int:
    release_date = _versioned_date(release.get("release_date") or release.get("date"))
    target = _versioned_date(commit_date)
    if release_date is None or target is None:
        return 10**9
    return abs((release_date - target).days)


def _project_id(project: dict[str, Any]) -> str:
    return str(
        project.get("project_id")
        or project.get("id")
        or project.get("name")
        or project.get("unified_name")
        or ""
    ).strip()


def _methods_for_project(release: dict[str, Any], project_id: str) -> list[dict[str, Any]]:
    mappings = (
        release.get("realizations")
        or release.get("realization_methods")
        or release.get("dependencies")
        or {}
    )
    raw = mappings.get(project_id) if isinstance(mappings, dict) else None
    if raw is None and isinstance(mappings, dict):
        raw = mappings.get(project_id.lower())
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("methods") or raw.get("realizations") or [raw]
    return [dict(item) for item in raw if isinstance(item, dict)]


def _method_feasible(method: dict[str, Any], entry_id: str) -> bool:
    if method.get("feasible") is False or method.get("available") is False:
        return False
    allowed_entries = method.get("build_entry_ids") or method.get("entry_ids")
    if allowed_entries and entry_id and entry_id not in {str(item) for item in allowed_entries}:
        return False
    if method.get("required_artifacts") and method.get("provides") is not None:
        required = {str(item).lower() for item in method.get("required_artifacts") or []}
        provided = {str(item).lower() for item in method.get("provides") or []}
        if not required.issubset(provided):
            return False
    return True


def _choose_method(project: dict[str, Any], methods: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    requires_source = bool(
        project.get("requires_source")
        or project.get("source_required")
        or project.get("explicit_source")
    )
    feasible = [item for item in methods if _method_feasible(item, "")]
    if requires_source:
        source = [
            item for item in feasible
            if str(item.get("kind") or item.get("type") or "").lower()
            in {"source", "source_dependency", "vendored"}
        ]
        if source:
            return source[0]
    package = [
        item for item in feasible
        if str(item.get("kind") or item.get("type") or "").lower() in {"package", "apt", "system_package"}
    ]
    if package:
        return package[0]
    return feasible[0] if feasible else None


def select_build_configuration(
    dependency_projects: list[dict[str, Any]],
    candidate_releases: list[dict[str, Any]],
    *,
    build_entry_id: str = "",
    commit_date: str = "",
) -> dict[str, Any]:
    """Return the best feasible release and selected method per project.

    A release is feasible only when every required dependency project has one
    usable realization.  Package-backed projects are preferred within each
    release; release ranking then maximizes package coverage and minimizes
    distance to the target commit date.
    """
    evaluated: list[dict[str, Any]] = []
    projects = [item for item in dependency_projects or [] if isinstance(item, dict)]
    for release in candidate_releases or []:
        if not isinstance(release, dict):
            continue
        selected: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        package_coverage = 0
        for project in projects:
            project_id = _project_id(project)
            if not project_id:
                continue
            methods = [
                method for method in _methods_for_project(release, project_id)
                if _method_feasible(method, build_entry_id)
            ]
            method = _choose_method(project, methods)
            if method is None:
                missing.append(project_id)
                continue
            selected[project_id] = method
            kind = str(method.get("kind") or method.get("type") or "").lower()
            if kind in {"package", "apt", "system_package"}:
                package_coverage += 1
        if missing:
            continue
        evaluated.append(
            {
                "sysuse": release.get("sysuse") or release.get("suite") or release.get("codename") or "",
                "release": release,
                "selected_realization_methods": selected,
                "package_coverage": package_coverage,
                "release_distance": _release_distance(release, commit_date),
                "missing_projects": [],
            }
        )

    evaluated.sort(
        key=lambda item: (
            -int(item["package_coverage"]),
            int(item["release_distance"]),
            str(item["sysuse"]),
        )
    )
    if not evaluated:
        return {
            "status": "infeasible",
            "selected": None,
            "alternatives": [],
        }
    selected = evaluated[0]
    return {
        "status": "feasible",
        "selected": selected,
        "alternatives": evaluated[1:],
    }
