"""Repository-profile and Debian package metadata queries.

This module is the runtime seam between Phase 1/3 and the generated Debian
metadata. Callers ask for one environment-bound package fact instead of
opening pkglist/profile files independently.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from core.debian_sysuse import normalize_sysuse


ROOT = Path(__file__).resolve().parent
BUILD_RAG_INFO = ROOT / "Build_RAG_Info"
PROFILE_PATH = ROOT.parent / "docker" / "debian_repository_profiles.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def repository_profile_hash() -> str:
    payload = _load_json(PROFILE_PATH)
    if not payload:
        return ""
    import hashlib
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def metadata_manifest(sysuse: str, arch: str = "amd64") -> dict[str, Any]:
    suite = normalize_sysuse(sysuse)
    return _load_json(BUILD_RAG_INFO / "metadata_manifests" / f"{suite}_{arch}.json")


def metadata_matches_profile(sysuse: str, arch: str = "amd64") -> bool:
    manifest = metadata_manifest(sysuse, arch)
    return bool(
        manifest
        and manifest.get("repository_profile_hash")
        and manifest.get("repository_profile_hash") == repository_profile_hash()
        and manifest.get("resolved_profile_hash")
    )


def resolved_repository_profile_hash(sysuse: str, arch: str = "amd64") -> str:
    manifest = metadata_manifest(sysuse, arch)
    return str(manifest.get("resolved_profile_hash") or "")


def package_record(package: str, sysuse: str, arch: str = "amd64") -> dict[str, Any]:
    suite = normalize_sysuse(sysuse)
    payload = _load_json(BUILD_RAG_INFO / "pkglist" / f"{suite}_pkglist_{arch}.json")
    record = payload.get(str(package or "").strip().lower())
    return dict(record) if isinstance(record, dict) else {}


def package_versions(package: str, sysuse: str, arch: str = "amd64") -> list[dict[str, Any]]:
    record = package_record(package, sysuse, arch)
    raw = record.get("version")
    if isinstance(raw, list):
        result = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("version"):
                continue
            normalized = dict(item)
            source_repo = str(normalized.get("source_repo") or "")
            if source_repo and not normalized.get("repo_name"):
                repo_name, component = source_repo.split("/", 1) if "/" in source_repo else (source_repo, "main")
                normalized.setdefault("repo_name", repo_name)
                normalized.setdefault("component", component)
                normalized.setdefault("suite", normalize_sysuse(sysuse))
            result.append(normalized)
        return result
    if isinstance(raw, str) and raw:
        return [{"version": raw, "suite": normalize_sysuse(sysuse)}]
    return []


def _version_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.findall(r"\d+|[A-Za-z]+|~", str(value or "").split(":", 1)[-1])
    return tuple((0, -1) if part == "~" else (1, int(part)) if part.isdigit() else (2, part.lower()) for part in parts)


def version_satisfies(observed: str, operator: str, required: str) -> bool:
    if not observed or not required:
        return False
    op = str(operator or ">=").strip()
    dpkg_operator = {"==": "eq", "=": "eq", ">=": "ge", ">": "gt", "<=": "le", "<": "lt"}.get(op)
    if dpkg_operator:
        try:
            result = subprocess.run(
                ["dpkg", "--compare-versions", str(observed), dpkg_operator, str(required)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            pass
    left, right = _version_key(observed), _version_key(required)
    return {
        ">=": left >= right,
        ">": left > right,
        "=": left == right,
        "==": left == right,
        "<=": left <= right,
        "<": left < right,
    }.get(op, False)


def best_satisfying_package_version(
    package: str,
    sysuse: str,
    operator: str = "",
    required: str = "",
) -> dict[str, Any]:
    versions = package_versions(package, sysuse)
    if required:
        versions = [
            item for item in versions
            if version_satisfies(str(item.get("version") or ""), operator or ">=", required)
        ]
    if not versions:
        return {}
    return max(versions, key=lambda item: _version_key(str(item.get("version") or "")))


def required_packages_for_environment(
    dependency_inventory: dict[str, Any],
    sysuse: str,
    selected_entry_id: str = "",
) -> list[dict[str, Any]]:
    """Bind the required dependency closure to one suite and selected entry."""
    return bind_required_packages(
        dependency_inventory,
        sysuse,
        selected_entry_id,
    )["packages"]


def bind_required_packages(
    dependency_inventory: dict[str, Any],
    sysuse: str,
    selected_entry_id: str = "",
) -> dict[str, Any]:
    """Resolve required apt provider groups for one concrete environment.

    Each dependency is one provider group.  Alternatives such as ``a | b``
    are tried in declared order, but a dependency is unresolved only after no
    provider is available with a satisfying version in this suite.
    """
    bound: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dependency in dependency_inventory.get("dependencies") or []:
        if not isinstance(dependency, dict):
            continue
        requirement = dependency.get("requirement") or {}
        if str(requirement.get("requiredness") or "") != "required":
            continue
        scope_entries = {
            str(item) for item in (dependency.get("scope") or {}).get("entry_ids") or [] if item
        }
        if scope_entries and selected_entry_id and selected_entry_id not in scope_entries:
            continue
        predicates = [item for item in requirement.get("version_predicates") or [] if isinstance(item, dict)]
        apt_providers = [
            provider for provider in dependency.get("providers") or []
            if isinstance(provider, dict) and provider.get("kind") == "apt"
        ]
        if not apt_providers:
            continue
        selected_provider = False
        for provider in apt_providers:
            package = str(provider.get("name") or "").strip().lower()
            if not package or package in seen:
                continue
            predicate = predicates[0] if predicates else {}
            version = best_satisfying_package_version(
                package,
                sysuse,
                str(predicate.get("op") or ""),
                str(predicate.get("version") or ""),
            )
            if not version:
                continue
            seen.add(package)
            bound.append({
                "name": package,
                "dependency_id": str(dependency.get("dependency_id") or ""),
                "suite": normalize_sysuse(sysuse),
                "version": str(version.get("version") or ""),
                "source_repo": str(version.get("source_repo") or ""),
                "component": str(version.get("component") or ""),
                "channel": str(version.get("channel") or ""),
            })
            selected_provider = True
            break
        if not selected_provider:
            unresolved.append({
                "dependency_id": str(dependency.get("dependency_id") or ""),
                "providers": [
                    str(provider.get("name") or "").strip().lower()
                    for provider in apt_providers
                    if provider.get("name")
                ],
                "version_predicates": predicates,
                "suite": normalize_sysuse(sysuse),
            })
    return {
        "suite": normalize_sysuse(sysuse),
        "packages": bound,
        "unresolved_dependencies": unresolved,
        "complete": not unresolved,
    }


def release_candidates_satisfying_package(
    package: str,
    suites: Iterable[str],
    operator: str,
    required: str,
) -> list[dict[str, Any]]:
    result = []
    for suite in suites:
        version = best_satisfying_package_version(package, suite, operator, required)
        if version:
            result.append({"sysuse": normalize_sysuse(suite), **version})
    return result
