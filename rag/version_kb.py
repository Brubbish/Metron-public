#!/usr/bin/env python3
"""Utilities for building and querying the Debian source-version knowledge base."""

from __future__ import annotations

import bz2
import gzip
import hashlib
import json
import lzma
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

DEFAULT_ARCH = "amd64"
DEBIAN_SUITE_ORDER = [
    "etch",
    "lenny",
    "squeeze",
    "wheezy",
    "jessie",
    "stretch",
    "buster",
    "bullseye",
    "bookworm",
    "trixie",
]
SUITE_ORDER_INDEX = {suite: index for index, suite in enumerate(DEBIAN_SUITE_ORDER)}
CHANNEL_PRIORITY = {
    "main": 0,
    "updates": 1,
    "proposed-updates": 2,
    "backports": 3,
    "backports-sloppy": 4,
}
REPO_NAME_SUFFIXES: Sequence[Tuple[str, str]] = (
    ("-backports-sloppy", "backports-sloppy"),
    ("-proposed-updates", "proposed-updates"),
    ("-backports", "backports"),
    ("-updates", "updates"),
)
THEAUTO_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_PROFILE_PATH = Path(
    os.environ.get(
        "THEAUTO_REPOSITORY_PROFILE",
        THEAUTO_ROOT / "docker" / "debian_repository_profiles.json",
    )
)


def load_repository_profile(path: Path = REPOSITORY_PROFILE_PATH) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    releases = payload.get("releases")
    if payload.get("schema_version") != 1 or not isinstance(releases, dict):
        raise ValueError(f"invalid repository profile: {path}")
    missing = [suite for suite in DEBIAN_SUITE_ORDER if suite not in releases]
    if missing:
        raise ValueError(f"repository profile misses releases: {', '.join(missing)}")
    return payload


def repository_profile_hash(payload: Optional[Dict[str, Any]] = None) -> str:
    canonical = json.dumps(
        payload or REPOSITORY_PROFILE,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def repositories_from_profile(payload: Dict[str, Any]) -> Dict[str, Dict[str, List[List[str]]]]:
    repositories: Dict[str, Dict[str, List[List[str]]]] = {}
    for suite in DEBIAN_SUITE_ORDER:
        config = payload["releases"][suite]
        repositories.setdefault(str(config["base_url"]), {})[suite] = [
            list(config["repositories"]),
            list(config["components"]),
        ]
    return repositories


REPOSITORY_PROFILE = load_repository_profile()
REPOSITORY_PROFILE_HASH = repository_profile_hash(REPOSITORY_PROFILE)
DEBIAN_REPOSITORIES = repositories_from_profile(REPOSITORY_PROFILE)
_COMPRESSED_EXTENSIONS = (
    ("xz", lzma.decompress),
    ("gz", gzip.decompress),
    ("bz2", bz2.decompress),
)
_SOURCE_RE = re.compile(r"^(?P<name>.+?)\s+\((?P<version>[^()]+)\)$")


@dataclass(frozen=True)
class RepoTarget:
    base_url: str
    suite: str
    repo_name: str
    channel: str
    component: str
    kind: str
    arch: Optional[str] = None

    @property
    def source_label(self) -> str:
        return f"{self.repo_name}/{self.component}"

    @property
    def is_primary(self) -> bool:
        return self.repo_name == self.suite and self.channel == "main"

    @property
    def cache_name(self) -> str:
        parts = [self.repo_name, self.component]
        if self.kind == "binary":
            parts.extend([self.arch or DEFAULT_ARCH, "pkginfo"])
        else:
            parts.append("sourceinfo")
        return "_".join(parts)

    @property
    def output_name(self) -> str:
        if self.kind == "binary":
            return f"{self.suite}_pkglist_{self.arch or DEFAULT_ARCH}.json"
        return f"{self.suite}_sourcelist.json"


def suite_sort_key(suite: str) -> Tuple[int, str]:
    return (SUITE_ORDER_INDEX.get(suite, len(SUITE_ORDER_INDEX)), suite)


def repo_priority(channel: str, is_primary: bool = False) -> int:
    if is_primary:
        return -1
    return CHANNEL_PRIORITY.get(channel, 99)


def record_priority(record: Dict[str, Any]) -> Tuple[int, int, str, str]:
    return (
        0 if record.get("is_primary") else 1,
        repo_priority(str(record.get("channel", "")), bool(record.get("is_primary"))),
        str(record.get("repo_name", "")),
        str(record.get("component", "")),
    )


def split_repo_name(repo_name: str) -> Tuple[str, str]:
    for suffix, channel in REPO_NAME_SUFFIXES:
        if repo_name.endswith(suffix):
            return repo_name[: -len(suffix)], channel
    return repo_name, "main"


def parse_source_label(source_repo: str, fallback_suite: str) -> Dict[str, Any]:
    if "/" in source_repo:
        repo_name, component = source_repo.split("/", 1)
    else:
        repo_name, component = source_repo, "main"
    suite, channel = split_repo_name(repo_name)
    suite = suite or fallback_suite
    return {
        "suite": suite,
        "repo_name": repo_name,
        "component": component,
        "channel": channel,
        "is_primary": repo_name == suite and channel == "main",
        "source_repo": f"{repo_name}/{component}",
    }


def normalize_source_field(source_value: Optional[str], fallback_name: str) -> Tuple[str, str, str]:
    raw_value = (source_value or "").strip()
    if not raw_value:
        return fallback_name, "", ""
    match = _SOURCE_RE.match(raw_value)
    if match:
        return match.group("name").strip(), match.group("version").strip(), raw_value
    return raw_value, "", raw_value


def iter_repo_targets(kind: str, arch: str = DEFAULT_ARCH, suites: Optional[Iterable[str]] = None) -> Iterator[RepoTarget]:
    requested_suites = set(suites or [])
    for base_url, suite_map in DEBIAN_REPOSITORIES.items():
        for suite in DEBIAN_SUITE_ORDER:
            if suite not in suite_map:
                continue
            if requested_suites and suite not in requested_suites:
                continue
            repo_names, components = suite_map[suite]
            for repo_name in repo_names:
                parsed_suite, channel = split_repo_name(repo_name)
                for component in components:
                    yield RepoTarget(
                        base_url=base_url,
                        suite=parsed_suite or suite,
                        repo_name=repo_name,
                        channel=channel,
                        component=component,
                        kind=kind,
                        arch=arch if kind == "binary" else None,
                    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


def metadata_manifest_path(base_dir: Path, suite: str, arch: str = DEFAULT_ARCH) -> Path:
    return base_dir / "metadata_manifests" / f"{suite}_{arch}.json"


def suite_repository_profile(suite: str) -> Dict[str, Any]:
    config = REPOSITORY_PROFILE["releases"][suite]
    return {
        "schema_version": REPOSITORY_PROFILE["schema_version"],
        "repository_profile_hash": REPOSITORY_PROFILE_HASH,
        "suite": suite,
        "architecture": REPOSITORY_PROFILE.get("architecture", DEFAULT_ARCH),
        "base_url": config["base_url"],
        "repositories": list(config["repositories"]),
        "components": list(config["components"]),
    }


def resolved_suite_repository_profile(
    base_dir: Path,
    suite: str,
    arch: str = DEFAULT_ARCH,
) -> Dict[str, Any]:
    """Derive the actually populated repo/component targets from pkglist.

    The configured profile is the source of truth for what may be fetched;
    this resolved view records what the offline metadata actually contains.
    Empty targets are omitted so generated images do not enable repositories
    that the package database never proved usable.
    """
    suite = str(suite).strip()
    declared = suite_repository_profile(suite)
    targets: Dict[Tuple[str, str], Dict[str, Any]] = {}
    pkg_path = output_path(base_dir, kind="binary", suite=suite, arch=arch)
    if pkg_path.is_file():
        try:
            payload = load_json(pkg_path)
        except (OSError, ValueError):
            payload = {}
        for record in payload.values():
            if not isinstance(record, dict):
                continue
            versions = record.get("version")
            if isinstance(versions, dict):
                versions = [versions]
            for version in versions or []:
                if not isinstance(version, dict):
                    continue
                source_label = str(version.get("source_repo") or "").strip()
                parsed = parse_source_label(source_label, suite)
                repo_name = str(version.get("repo_name") or parsed.get("repo_name") or "").strip()
                component = str(version.get("component") or parsed.get("component") or "").strip()
                if repo_name and component:
                    targets[(repo_name, component)] = {
                        "repo_name": repo_name,
                        "component": component,
                        "channel": str(version.get("channel") or parsed.get("channel") or "main"),
                    }
    resolved = sorted(targets.values(), key=lambda item: (item["repo_name"], item["component"]))
    resolved_canonical = json.dumps(resolved, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    resolved_hash = hashlib.sha256(resolved_canonical.encode("utf-8")).hexdigest()[:16]
    declared["resolved_repositories"] = resolved
    declared["resolved_profile_hash"] = resolved_hash
    declared["resolved_from"] = str(pkg_path)
    return declared


def iter_deb822_paragraphs(text: str) -> Iterator[Dict[str, str]]:
    paragraph: Dict[str, str] = {}
    current_key: Optional[str] = None
    for raw_line in text.splitlines():
        if not raw_line.strip():
            if paragraph:
                yield paragraph
                paragraph = {}
            current_key = None
            continue
        if raw_line.startswith((" ", "\t")) and current_key:
            paragraph[current_key] += "\n" + raw_line[1:]
            continue
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        current_key = key.strip()
        paragraph[current_key] = value.strip()
    if paragraph:
        yield paragraph


def parse_binary_packages_text(text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for paragraph in iter_deb822_paragraphs(text):
        binary_pkg = paragraph.get("Package", "").strip()
        version = paragraph.get("Version", "").strip()
        if not binary_pkg or not version:
            continue
        source_pkg, source_version_hint, source_raw = normalize_source_field(paragraph.get("Source"), binary_pkg)
        records.append(
            {
                "package": binary_pkg,
                "binary_pkg": binary_pkg,
                "description": paragraph.get("Description", "").strip(),
                "source": source_pkg,
                "source_pkg": source_pkg,
                "source_raw": source_raw,
                "source_version_hint": source_version_hint,
                "version": version,
            }
        )
    return records


def parse_source_packages_text(text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for paragraph in iter_deb822_paragraphs(text):
        source_pkg = paragraph.get("Package", "").strip()
        version = paragraph.get("Version", "").strip()
        if not source_pkg or not version:
            continue
        binaries = [item.strip() for item in paragraph.get("Binary", "").split(",") if item.strip()]
        records.append(
            {
                "package": source_pkg,
                "source_pkg": source_pkg,
                "version": version,
                "binary_packages": sorted(set(binaries)),
            }
        )
    return records


def _version_entry_key(version_entry: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(version_entry.get("version", "")),
        str(version_entry.get("repo_name", "")),
        str(version_entry.get("component", "")),
    )


def _sorted_version_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        entries,
        key=lambda item: (
            repo_priority(str(item.get("channel", "")), bool(item.get("is_primary"))),
            str(item.get("repo_name", "")),
            str(item.get("component", "")),
            str(item.get("version", "")),
        ),
    )


def merge_binary_records(store: Dict[str, Dict[str, Any]], records: List[Dict[str, Any]], target: RepoTarget) -> None:
    for record in records:
        binary_pkg = record["package"]
        source_pkg = record.get("source_pkg") or record.get("source") or binary_pkg
        entry = store.setdefault(
            binary_pkg,
            {
                "binary_pkg": binary_pkg,
                "description": record.get("description", ""),
                "source": source_pkg,
                "source_pkg": source_pkg,
                "source_raw": record.get("source_raw", ""),
                "version": [],
            },
        )
        if not entry.get("description") and record.get("description"):
            entry["description"] = record["description"]
        if source_pkg:
            entry["source"] = source_pkg
            entry["source_pkg"] = source_pkg
        if not entry.get("source_raw") and record.get("source_raw"):
            entry["source_raw"] = record["source_raw"]

        version_entry = {
            "version": record["version"],
            "source_repo": target.source_label,
            "suite": target.suite,
            "repo_name": target.repo_name,
            "channel": target.channel,
            "component": target.component,
            "is_primary": target.is_primary,
            "source_version_hint": record.get("source_version_hint", ""),
        }
        existing_keys = {_version_entry_key(item) for item in entry["version"]}
        if _version_entry_key(version_entry) not in existing_keys:
            entry["version"].append(version_entry)
            entry["version"] = _sorted_version_entries(entry["version"])


def merge_source_records(store: Dict[str, Dict[str, Any]], records: List[Dict[str, Any]], target: RepoTarget) -> None:
    for record in records:
        source_pkg = record["package"]
        entry = store.setdefault(source_pkg, {"source_pkg": source_pkg, "version": []})
        version_entry = {
            "version": record["version"],
            "source_repo": target.source_label,
            "suite": target.suite,
            "repo_name": target.repo_name,
            "channel": target.channel,
            "component": target.component,
            "is_primary": target.is_primary,
            "binary_packages": record.get("binary_packages", []),
        }
        existing_keys = {_version_entry_key(item) for item in entry["version"]}
        if _version_entry_key(version_entry) not in existing_keys:
            entry["version"].append(version_entry)
            entry["version"] = _sorted_version_entries(entry["version"])


def decompress_payload(payload: bytes, suffix: str) -> str:
    for extension, decompressor in _COMPRESSED_EXTENSIONS:
        if suffix.endswith(extension):
            return decompressor(payload).decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported compressed file suffix: {suffix}")


def download_text_candidates(urls: Sequence[str], timeout: int = 60) -> Tuple[str, str]:
    last_error: Optional[Exception] = None
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = response.read()
            return decompress_payload(payload, url), url
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as error:
            last_error = error
    if last_error is None:
        raise RuntimeError("No download candidates were provided")
    raise RuntimeError(str(last_error))


def build_binary_urls(target: RepoTarget) -> List[str]:
    if target.kind != "binary":
        raise ValueError("build_binary_urls requires a binary RepoTarget")
    prefix = f"{target.base_url}/dists/{target.repo_name}/{target.component}/binary-{target.arch}"
    return [f"{prefix}/Packages.{extension}" for extension, _ in _COMPRESSED_EXTENSIONS]


def build_source_urls(target: RepoTarget) -> List[str]:
    if target.kind != "source":
        raise ValueError("build_source_urls requires a source RepoTarget")
    prefix = f"{target.base_url}/dists/{target.repo_name}/{target.component}/source"
    return [f"{prefix}/Sources.{extension}" for extension, _ in _COMPRESSED_EXTENSIONS]


def output_path(base_dir: Path, kind: str, suite: str, arch: str = DEFAULT_ARCH) -> Path:
    folder_name = "pkglist" if kind == "binary" else "sourcelist"
    ensure_dir(base_dir / folder_name)
    if kind == "binary":
        return base_dir / folder_name / f"{suite}_pkglist_{arch}.json"
    return base_dir / folder_name / f"{suite}_sourcelist.json"


def load_suite_json_map(directory: Path, pattern: str) -> Dict[str, Dict[str, Any]]:
    if not directory.exists():
        return {}
    results: Dict[str, Dict[str, Any]] = {}
    for json_file in sorted(directory.glob(pattern)):
        suite = json_file.name.split("_", 1)[0]
        results[suite] = load_json(json_file)
    return results


def coerce_version_entries(version_value: Any, fallback_suite: str) -> List[Dict[str, Any]]:
    if isinstance(version_value, str):
        parsed = parse_source_label(f"{fallback_suite}/main", fallback_suite)
        parsed.update({"version": version_value, "source_version_hint": "", "binary_packages": []})
        return [parsed]
    if not isinstance(version_value, list):
        return []

    entries: List[Dict[str, Any]] = []
    for item in version_value:
        if isinstance(item, str):
            parsed = parse_source_label(f"{fallback_suite}/main", fallback_suite)
            parsed.update({"version": item, "source_version_hint": "", "binary_packages": []})
            entries.append(parsed)
            continue
        source_repo = str(item.get("source_repo") or f"{fallback_suite}/main")
        parsed = parse_source_label(source_repo, fallback_suite)
        parsed.update(
            {
                "version": str(item.get("version", "")).strip(),
                "source_version_hint": str(item.get("source_version_hint", "")).strip(),
                "binary_packages": sorted(set(item.get("binary_packages") or [])),
            }
        )
        parsed["channel"] = str(item.get("channel") or parsed["channel"])
        parsed["component"] = str(item.get("component") or parsed["component"])
        parsed["repo_name"] = str(item.get("repo_name") or parsed["repo_name"])
        parsed["suite"] = str(item.get("suite") or parsed["suite"] or fallback_suite)
        parsed["is_primary"] = bool(item.get("is_primary", parsed["repo_name"] == parsed["suite"] and parsed["channel"] == "main"))
        if not parsed["source_repo"]:
            parsed["source_repo"] = f"{parsed['repo_name']}/{parsed['component']}"
        if parsed["version"]:
            entries.append(parsed)
    return _sorted_version_entries(entries)


def build_source_version_rows(source_payload_by_suite: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for suite, sourcelist_data in source_payload_by_suite.items():
        for source_pkg, source_info in sourcelist_data.items():
            normalized_source_pkg = str(source_info.get("source_pkg") or source_pkg).strip() or source_pkg
            top_level_binaries = sorted(set(source_info.get("binary_packages") or []))
            for version_entry in coerce_version_entries(source_info.get("version"), suite):
                binary_packages = version_entry.get("binary_packages") or top_level_binaries
                rows.append(
                    {
                        "suite": version_entry["suite"],
                        "repo_name": version_entry["repo_name"],
                        "channel": version_entry["channel"],
                        "component": version_entry["component"],
                        "source_pkg": normalized_source_pkg,
                        "source_version": version_entry["version"],
                        "binary_packages": list(binary_packages),
                        "is_primary": version_entry["is_primary"],
                        "version_origin": "sources",
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            suite_sort_key(str(row["suite"])),
            str(row["source_pkg"]),
            record_priority(row),
            str(row["source_version"]),
        ),
    )


def build_binary_source_rows(
    pkg_payload_by_suite: Dict[str, Dict[str, Any]],
    arch: str = DEFAULT_ARCH,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_keys = set()
    for suite, pkglist_data in pkg_payload_by_suite.items():
        for binary_pkg, pkg_info in pkglist_data.items():
            normalized_binary_pkg = str(pkg_info.get("binary_pkg") or binary_pkg).strip() or binary_pkg
            raw_source_pkg = str(pkg_info.get("source_pkg") or pkg_info.get("source") or normalized_binary_pkg).strip()
            source_pkg, _, _ = normalize_source_field(raw_source_pkg, normalized_binary_pkg)
            description = str(pkg_info.get("description", "")).strip()
            for version_entry in coerce_version_entries(pkg_info.get("version"), suite):
                row = {
                    "suite": version_entry["suite"],
                    "repo_name": version_entry["repo_name"],
                    "channel": version_entry["channel"],
                    "component": version_entry["component"],
                    "arch": arch,
                    "binary_pkg": normalized_binary_pkg,
                    "source_pkg": source_pkg,
                    "description": description,
                    "is_primary": version_entry["is_primary"],
                }
                key = (
                    row["suite"],
                    row["repo_name"],
                    row["component"],
                    row["arch"],
                    row["binary_pkg"],
                    row["source_pkg"],
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            suite_sort_key(str(row["suite"])),
            str(row["binary_pkg"]),
            record_priority(row),
            str(row["source_pkg"]),
        ),
    )


def group_source_versions_by_distro(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            suite_sort_key(str(row.get("suite", ""))),
            record_priority(dict(row)),
            str(row.get("repo_name", "")),
            str(row.get("component", "")),
            str(row.get("source_version", row.get("version", ""))),
        ),
    )
    for row in ordered_rows:
        suite = str(row.get("suite", ""))
        item = {
            "repo_name": str(row.get("repo_name", "")),
            "component": str(row.get("component", "")),
            "channel": str(row.get("channel", "main")),
            "version": str(row.get("source_version", row.get("version", ""))),
        }
        binary_packages = list(row.get("binary_packages") or [])
        if binary_packages:
            item["binary_packages"] = binary_packages
        grouped.setdefault(suite, []).append(item)
    return {
        suite: grouped[suite]
        for suite in sorted(grouped, key=suite_sort_key)
    }


def match_priority(matched_kind: str, exact_match: bool) -> int:
    if matched_kind == "source" and exact_match:
        return 0
    if matched_kind == "binary" and exact_match:
        return 1
    return 2


def normalize_match_candidate(candidate: Dict[str, Any], matched_kind: str) -> Dict[str, Any]:
    exact_match = bool(candidate.get("exact_match"))
    score = 1.0 if exact_match else float(candidate.get("score") or 0.0)
    return {
        "resolved_source_pkg": str(candidate.get("resolved_source_pkg", "")).strip(),
        "matched_name": str(candidate.get("matched_name", "")).strip(),
        "matched_kind": matched_kind,
        "match_type": "exact" if exact_match else "fuzzy",
        "exact_match": exact_match,
        "score": round(score, 4),
        "has_distro_hit": bool(candidate.get("has_distro_hit")),
    }


def candidate_sort_key(candidate: Dict[str, Any], distro_filter: Optional[str]) -> Tuple[int, int, float, int, str, str]:
    distro_rank = 0
    if distro_filter:
        distro_rank = 0 if candidate.get("has_distro_hit") else 1
    kind_rank = 0 if candidate.get("matched_kind") == "source" else 1
    return (
        match_priority(str(candidate.get("matched_kind", "binary")), bool(candidate.get("exact_match"))),
        distro_rank,
        -float(candidate.get("score") or 0.0),
        kind_rank,
        str(candidate.get("matched_name", "")),
        str(candidate.get("resolved_source_pkg", "")),
    )


def resolve_query_candidates(
    source_candidates: Sequence[Dict[str, Any]],
    binary_candidates: Sequence[Dict[str, Any]],
    distro_filter: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    candidates_by_source: Dict[str, Dict[str, Any]] = {}
    for matched_kind, raw_candidates in (("source", source_candidates), ("binary", binary_candidates)):
        for raw_candidate in raw_candidates:
            candidate = normalize_match_candidate(dict(raw_candidate), matched_kind)
            resolved_source_pkg = candidate["resolved_source_pkg"]
            if not resolved_source_pkg:
                continue
            current = candidates_by_source.get(resolved_source_pkg)
            if current is None or candidate_sort_key(candidate, distro_filter) < candidate_sort_key(current, distro_filter):
                candidates_by_source[resolved_source_pkg] = candidate

    if not candidates_by_source:
        return None

    selected = min(candidates_by_source.values(), key=lambda item: candidate_sort_key(item, distro_filter))
    return {
        "resolved_source_pkg": selected["resolved_source_pkg"],
        "match": {
            "matched_name": selected["matched_name"],
            "matched_kind": selected["matched_kind"],
            "match_type": selected["match_type"],
            "score": selected["score"],
        },
    }


def collapse_records_by_suite(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen_suites = set()
    ordered = sorted(
        records,
        key=lambda item: (
            suite_sort_key(str(item.get("suite", ""))),
            record_priority(item),
            str(item.get("source_version", item.get("binary_version", item.get("version", "")))),
        ),
    )
    for record in ordered:
        suite = str(record.get("suite", ""))
        if suite in seen_suites:
            continue
        seen_suites.add(suite)
        selected.append(record)
    return selected


def sort_group_records(records: List[Dict[str, Any]], include_repos: bool) -> List[Dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda item: (
            suite_sort_key(str(item.get("suite", ""))),
            record_priority(item),
            str(item.get("repo_name", "")),
            str(item.get("component", "")),
            str(item.get("source_version", item.get("binary_version", item.get("version", "")))),
        ),
    )
    if include_repos:
        return ordered
    return collapse_records_by_suite(ordered)


def summarize_suite_records(records: List[Dict[str, Any]], include_repos: bool) -> List[Dict[str, Any]]:
    return [dict(record) for record in sort_group_records(records, include_repos)]
