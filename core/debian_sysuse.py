"""Debian release selection helpers shared by batch and agent workflows."""

from __future__ import annotations

import os
import subprocess
from datetime import date, datetime
from typing import Optional, Union


DEBIAN_RELEASE_WINDOWS = [
    (date.min, date(2007, 4, 7), "etch"),
    (date(2007, 4, 8), date(2009, 2, 13), "etch"),
    (date(2009, 2, 14), date(2011, 2, 5), "lenny"),
    (date(2011, 2, 6), date(2013, 5, 3), "squeeze"),
    (date(2013, 5, 4), date(2015, 4, 24), "wheezy"),
    (date(2015, 4, 25), date(2017, 6, 16), "jessie"),
    (date(2017, 6, 17), date(2019, 7, 5), "stretch"),
    (date(2019, 7, 6), date(2021, 8, 13), "buster"),
    (date(2021, 8, 14), date(2023, 6, 9), "bullseye"),
    (date(2023, 6, 10), date(2025, 8, 8), "bookworm"),
    (date(2025, 8, 9), None, "trixie"),
]

VALID_DEBIAN_CODENAMES = {codename for _, _, codename in DEBIAN_RELEASE_WINDOWS}


DateLike = Union[date, datetime, str]


def normalize_sysuse(sysuse: str) -> str:
    """Return a bare Debian codename from sysuse/image strings."""
    normalized = str(sysuse or "").strip()
    for prefix in ("theauto/debian:", "debian:"):
        if normalized.startswith(prefix):
            return normalized.split(":", 1)[1]
    return normalized


def parse_date(value: DateLike) -> date:
    """Parse ``YYYY-MM-DD`` strings or date/datetime objects."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("empty date")
    return datetime.strptime(text[:10], "%Y-%m-%d").date()


CODENAME_ORDER = [codename for _, _, codename in DEBIAN_RELEASE_WINDOWS]

# Hard floor for sysuse selection. Anything older than stretch is risky for
# building MODERN C/C++ projects: toolchain (autoconf/automake/cmake/meson)
# is too old, common -dev packages (libglib2.0-dev, libssl-dev, libyang2-dev,
# libqb-dev, etc.) either don't exist or are far below the version floors
# required by current upstream sources. Observed batch failures: squeeze
# lacks libqb-dev and AM_PROG_AR; wheezy/jessie ship GLib 2.42 below the
# g_autoptr floor (2.48, stretch+); automake 1.11 in squeeze cannot parse
# modern configure.ac.
#
# BUT: this rationale only holds when the source being built is itself
# "modern". A 2015 commit was almost certainly written against the jessie-era
# ABI (GnuTLS 3.3, GLib 2.42), and forcing stretch on it produces exactly the
# API-mismatch failures we saw in batch (libsoup, gvfs, pango, etc.). So the
# floor is now commit-date aware: a legacy suite is allowed when the commit
# falls inside (or near) that suite's release window. See
# `_commit_date_allows_legacy` below for the exact window logic.
SYSUSE_FLOOR_CODENAME = "stretch"
_LEGACY_FLOOR_BLOCKLIST = {"etch", "lenny", "squeeze", "wheezy", "jessie"}

# Plausible commit-date windows for legacy suites. Each window covers the
# suite's official release period (DEBIAN_RELEASE_WINDOWS) plus a buffer:
#   - 1 year of lead time (commits made during the suite's testing/freeze)
#   - LTS tail (jessie/wheezy had LTS until ~2020/2018; squeeze/lenny/etch
#     had none, so they only get a 1-year post-EOL grace).
# A commit inside this window is assumed to target the suite's ABI.
_LEGACY_SUITE_WINDOW_END = {
    "etch":    date(2010, 6, 30),    # EOL 2010-06, no LTS
    "lenny":   date(2012, 6, 30),    # EOL 2012-03, no LTS
    "squeeze": date(2014, 6, 30),    # EOL 2014-05, LTS to 2016-02 (be conservative)
    "wheezy":  date(2018, 5, 31),    # EOL 2016-04, LTS to 2018-05
    "jessie":  date(2020, 6, 30),    # EOL 2017-06, LTS to 2020-06
}


def _legacy_window_for(codename: str) -> tuple[Optional[date], Optional[date]]:
    """Return the (start, end) commit-date window inside which `codename` is a
    plausible sysuse. Returns (None, None) for non-legacy codenames."""
    if codename not in _LEGACY_FLOOR_BLOCKLIST:
        return (None, None)
    # Start = 1 year before the suite's official release date (dev freeze).
    for start, _end, name in DEBIAN_RELEASE_WINDOWS:
        if name == codename:
            try:
                window_start = date(start.year - 1, start.month, start.day)
            except ValueError:
                # start.year == 1 (date.min sentinel) — clamp to date.min.
                window_start = date.min
            window_end = _LEGACY_SUITE_WINDOW_END[codename]
            return (window_start, window_end)
    return (None, None)


def _commit_date_allows_legacy(commit_date: Optional[date], codename: str) -> bool:
    """Return True if `commit_date` falls inside the plausible window for the
    legacy `codename`. None/missing commit_date -> False (preserve default floor).
    """
    if commit_date is None:
        return False
    start, end = _legacy_window_for(codename)
    if start is None or end is None:
        return False
    return start <= commit_date <= end


def _next_codename(codename: str) -> str:
    """Return the next Debian release after the given one, or the same if already latest."""
    try:
        idx = CODENAME_ORDER.index(codename)
        if idx + 1 < len(CODENAME_ORDER):
            return CODENAME_ORDER[idx + 1]
    except ValueError:
        pass
    return codename


def apply_sysuse_floor(
    codename: str,
    *,
    allow_legacy: bool = False,
    commit_date: Optional[date] = None,
) -> str:
    """Bump legacy Debian codenames to the configured floor.

    Default floor is `stretch` (Debian 9, GLib 2.54, automake 1.15, gcc 6).
    Opt-out paths (in priority order):
      - `allow_legacy=True`: opt out completely, return the codename as-is.
        Used by `THEAUTO_ALLOW_LEGACY_SYSUSE=1` for projects that genuinely
        target historical Debian.
      - `commit_date=<date>`: keep a legacy suite when the commit falls inside
        that suite's plausible window (release period + buffer). This is the
        default path used by `debian_codename_for_date` so that e.g. a 2015
        commit gets `jessie` (matching its ABI) instead of being bumped to
        `stretch`.

    With neither opt-out, legacy suites are bumped to `SYSUSE_FLOOR_CODENAME`.
    """
    normalized = normalize_sysuse(codename)
    if allow_legacy:
        return normalized
    if normalized in _LEGACY_FLOOR_BLOCKLIST:
        if _commit_date_allows_legacy(commit_date, normalized):
            return normalized
        return SYSUSE_FLOOR_CODENAME
    return normalized


def debian_codename_for_date(
    value: DateLike,
    *,
    bump: bool = False,
    floor: bool = True,
) -> str:
    """Map a commit date to the Debian release active at that date.

    With bump=True, returns one release newer than the strict date match.
    With floor=True (default), legacy releases (jessie and older) are bumped
    up to `stretch` UNLESS the commit date falls inside that suite's plausible
    window — in which case the legacy suite is kept, because the source was
    likely written against its ABI.
    """
    commit_date = parse_date(value)
    for start, end, codename in DEBIAN_RELEASE_WINDOWS:
        if commit_date >= start and (end is None or commit_date <= end):
            resolved = _next_codename(codename) if bump else codename
            if floor:
                return apply_sysuse_floor(resolved, commit_date=commit_date)
            return resolved
    raise ValueError(f"No Debian release window found for {commit_date.isoformat()}")


def sysuse_for_date(value: DateLike, *, prefix: bool = True, floor: bool = True) -> str:
    """Return the workflow sysuse string for a commit date."""
    codename = debian_codename_for_date(value, floor=floor)
    return f"debian:{codename}" if prefix else codename


def git_commit_date(repo_path: str, commit: str) -> Optional[date]:
    """Read the commit date from git and return it as a date object."""
    commit_ref = str(commit or "").strip()
    if not commit_ref:
        return None
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit_ref}^{{commit}}"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    resolved_commit = resolved.stdout.strip()
    if resolved.returncode != 0 or not resolved_commit:
        return None
    result = subprocess.run(
        ["git", "show", "-s", "--format=%ci", resolved_commit, "--"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    output = result.stdout.strip()
    if result.returncode != 0 or not output:
        return None
    try:
        return parse_date(output.split()[0])
    except ValueError:
        return None


def sysuse_for_git_commit(repo_path: str, commit: str, *, prefix: bool = True) -> Optional[str]:
    commit_date = git_commit_date(repo_path, commit)
    if commit_date is None:
        return None
    return sysuse_for_date(commit_date, prefix=prefix)
