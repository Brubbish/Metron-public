"""Sysuse candidate planning and cheap probe commands.

The build still starts from one Debian suite.  This module adds a small,
ordered set of fallback suites plus commands that cheaply reveal whether a
suite has the expected compiler and package surface before spending a long
build attempt there.
"""

from __future__ import annotations

from typing import Iterable, Optional

from core.debian_sysuse import normalize_sysuse
from core.source_dep_registry import registry_sysuse_hints


CODENAME_ORDER = [
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


def _add_candidate(out: list[dict], sysuse: str, reason: str) -> None:
    codename = normalize_sysuse(sysuse)
    if codename not in CODENAME_ORDER:
        return
    if any(item["sysuse"] == codename for item in out):
        return
    out.append({"sysuse": codename, "reason": reason})


def _neighbors(codename: str) -> Iterable[tuple[str, str]]:
    try:
        idx = CODENAME_ORDER.index(codename)
    except ValueError:
        return []
    out: list[tuple[str, str]] = []
    if idx + 1 < len(CODENAME_ORDER):
        out.append((CODENAME_ORDER[idx + 1], "one suite newer than current"))
    if idx - 1 >= 0:
        out.append((CODENAME_ORDER[idx - 1], "one suite older than current"))
    return out


def build_sysuse_candidate_plan(
    *,
    repo_path: str,
    current_sysuse: str,
    commit_date: str = "",
    selected_distro: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """Return ordered sysuse fallback candidates with probe commands."""
    current = normalize_sysuse(current_sysuse)
    candidates: list[dict] = []
    _add_candidate(candidates, current, "current selected sysuse")

    if selected_distro:
        _add_candidate(
            candidates,
            selected_distro,
            "dependency extractor selected this sysuse",
        )

    for hint in registry_sysuse_hints(repo_path):
        _add_candidate(candidates, hint, "source dependency registry hint")

    for neighbor, reason in _neighbors(current):
        _add_candidate(candidates, neighbor, reason)

    probe_commands = [
        "printf 'sysuse probe\\n'",
        "gcc --version | head -n1 || true",
        "g++ --version | head -n1 || true",
        "cmake --version 2>/dev/null | head -n1 || true",
        "meson --version 2>/dev/null || true",
        "pkg-config --version 2>/dev/null || true",
        "apt-cache policy gcc g++ cmake meson pkg-config 2>/dev/null | sed -n '1,80p' || true",
    ]
    out: list[dict] = []
    for item in candidates[:limit]:
        out.append({
            "sysuse": item["sysuse"],
            "reason": item["reason"],
            "commit_date": commit_date,
            "probe_commands": probe_commands,
        })
    return out
