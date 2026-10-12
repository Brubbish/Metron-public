"""Version-aware Debian sysuse selection for Phase 1.

Scans build files to extract toolchain version requirements, then selects
the minimum Debian release that satisfies all constraints. Falls back to
commit-date heuristic when no version signals are found.

Usage:
    from sysuse_select import select_sysuse
    sysuse = select_sysuse("/path/to/repo", commit="abc123")
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional


# ─── Debian toolchain version table ───────────────────────────────────────────
# Each entry: (codename, gcc_major, cmake_version, meson_version, cpp_std_max)
# cpp_std_max: highest C++ standard fully supported by that GCC version

@dataclass(frozen=True)
class DebianToolchain:
    codename: str
    gcc_major: int
    cmake: tuple[int, int]       # (major, minor)
    meson: tuple[int, int] | None  # None = not available
    cpp_std_max: int             # e.g. 14, 17, 20, 23

TOOLCHAINS: list[DebianToolchain] = [
    # Legacy suites — included so toolchain queries (gcc/cmake/cpp_std) cover
    # the full etch→trixie range, matching the suite coverage of
    # distro_select._minimum_satisfying_sysuse (which handles package version
    # constraints via suite_compat + pkglist). meson=None because meson is not
    # packaged pre-stretch.
    DebianToolchain("etch",     4,  (2, 6),   None,      98),
    DebianToolchain("lenny",    4,  (2, 6),   None,      98),
    DebianToolchain("squeeze",  4,  (2, 8),   None,      3),   # gcc 4.4: partial c++0x, treat as c++03
    DebianToolchain("wheezy",   4,  (2, 8),   None,      11),  # gcc 4.7: full c++11
    DebianToolchain("jessie",   4,  (3, 0),   None,      11),
    DebianToolchain("stretch",  6,  (3, 7),   (0, 37),   14),
    DebianToolchain("buster",   8,  (3, 13),  (0, 49),   17),
    DebianToolchain("bullseye", 10, (3, 18),  (0, 56),   20),
    DebianToolchain("bookworm", 12, (3, 25),  (1, 0),    20),
    DebianToolchain("trixie",   14, (3, 31),  (1, 3),    23),
]

# Canonical 10-codename order (etch→trixie), independent of TOOLCHAINS.
# Package-version queries are handled by distro_select._minimum_satisfying_sysuse
# (via suite_compat + pkglist), not by this module; this list is kept in sync so
# that any future toolchain query can iterate all suites consistently.
ALL_CODENAMES_IN_ORDER: list[str] = [tc.codename for tc in TOOLCHAINS]

CODENAME_ORDER = [tc.codename for tc in TOOLCHAINS]


# ─── Commit-date fallback (same table as debian_sysuse.py) ────────────────────

DEBIAN_RELEASE_WINDOWS = [
    (date.min,          date(2007, 4, 7), "etch"),  # pre-etch era falls here too
    (date(2007, 4, 8),  date(2009, 2, 13), "etch"),
    (date(2009, 2, 14), date(2011, 2, 5),  "lenny"),
    (date(2011, 2, 6),  date(2013, 5, 3),  "squeeze"),
    (date(2013, 5, 4),  date(2015, 4, 24), "wheezy"),
    (date(2015, 4, 25), date(2017, 6, 16), "jessie"),
    (date(2017, 6, 17), date(2019, 7, 5),  "stretch"),
    (date(2019, 7, 6),  date(2021, 8, 13), "buster"),
    (date(2021, 8, 14), date(2023, 6, 9),  "bullseye"),
    (date(2023, 6, 10), date(2025, 8, 8),  "bookworm"),
    (date(2025, 8, 9),  None,              "trixie"),
]


def commit_date_to_sysuse(repo_path: str, commit: str) -> str:
    """Original date-based fallback."""
    result = subprocess.run(
        ["git", "show", "-s", "--format=%ci", commit],
        cwd=repo_path, capture_output=True, text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode != 0:
        return "bookworm"
    date_str = result.stdout.strip().split()[0]
    try:
        y, m, d = date_str.split("-")
        commit_d = date(int(y), int(m), int(d))
    except (ValueError, IndexError):
        return "bookworm"
    for start, end, codename in DEBIAN_RELEASE_WINDOWS:
        if commit_d >= start and (end is None or commit_d <= end):
            return codename
    return "bookworm"


# ─── Version signal extraction ────────────────────────────────────────────────

@dataclass
class VersionSignals:
    """Collected version requirements from build files."""
    cmake_minimum: tuple[int, int] | None = None       # cmake_minimum_required
    cmake_minimum_root: tuple[int, int] | None = None
    cmake_cxx_standard: int | None = None              # CMAKE_CXX_STANDARD
    cmake_cxx_standard_root: int | None = None
    cmake_cxx_standard_hard: bool = False
    meson_version: tuple[int, int] | None = None       # meson_version: '>= X.Y'
    meson_version_root: tuple[int, int] | None = None
    meson_cpp_std: int | None = None                   # cpp_std in default_options
    meson_cpp_std_root: int | None = None
    meson_cpp_std_hard: bool = False
    autotools_cpp_std: int | None = None               # AX_CXX_COMPILE_STDCXX(N)
    autotools_cpp_std_root: int | None = None
    makefile_cpp_std: int | None = None                # -std=c++N in CXXFLAGS
    makefile_cpp_std_root: int | None = None
    needs_meson: bool = False                          # meson.build present
    needs_meson_hard: bool = False
    needs_cmake: bool = False                          # CMakeLists.txt present
    needs_cmake_hard: bool = False
    needs_cargo: bool = False                          # Cargo.toml present
    needs_cargo_hard: bool = False
    gcc_minimum: int | None = None                     # explicit GCC version req
    gcc_minimum_root: int | None = None
    cmake_minimum_hard: bool = False
    meson_version_hard: bool = False
    autotools_cpp_std_hard: bool = False
    makefile_cpp_std_hard: bool = False
    gcc_minimum_hard: bool = False
    build_system: str = ""                             # detected primary build system
    signals_found: list[str] = field(default_factory=list)  # audit trail


# Regex patterns for signal extraction

_CMAKE_MIN_RE = re.compile(
    r'cmake_minimum_required\s*\(\s*VERSION\s+(\d+)\.(\d+)',
    re.IGNORECASE,
)
_CMAKE_CXX_STD_RE = re.compile(
    r'(?:set\s*\(\s*CMAKE_CXX_STANDARD\s+|'
    r'CMAKE_CXX_STANDARD\s+)(\d+)',
    re.IGNORECASE,
)
_CMAKE_CXX_STD_TARGET_RE = re.compile(
    r'(?:CXX_STANDARD\s+|cxx_std_)(\d+)',
    re.IGNORECASE,
)

_MESON_VERSION_RE = re.compile(
    r"meson_version\s*:\s*['\"]?\s*>=?\s*(\d+)\.(\d+)",
    re.IGNORECASE,
)
_MESON_CPP_STD_RE = re.compile(
    r"(?:cpp_std\s*[=:]\s*['\"]?c\+\+|default_options\s*:.*?['\"]cpp_std=c\+\+)(\d+)",
    re.IGNORECASE,
)

_AX_CXX_STDCXX_RE = re.compile(
    r'AX_CXX_COMPILE_STDCXX\s*\(\s*\[?(\d+)\]?',
)
_AC_CXX_STD_RE = re.compile(
    r'-std=c\+\+(\d+)',
)

_MAKEFILE_STD_RE = re.compile(
    r'(?:CXXFLAGS|CPPFLAGS|CXX|FLAGS)\s*[+:?]?=.*-std=c\+\+(\d+)',
    re.IGNORECASE,
)
_MAKEFILE_STD_INLINE_RE = re.compile(
    r'-std=c\+\+(\d+)',
)

_GCC_VERSION_RE = re.compile(
    r'(?:gcc|g\+\+|GCC)\s*(?:>=?\s*|version\s+)(\d+)',
    re.IGNORECASE,
)

_ROOT_SIGNAL_FILES = frozenset({
    'cmakelists.txt', 'meson.build', 'meson_options.txt', 'meson.options',
    'configure.ac', 'configure.in', 'makefile', 'gnumakefile',
    'makefile.am', 'cargo.toml', 'configure',
})


def should_extract_signal_file(rel_path: str) -> bool:
    """Return whether a repo-relative path participates in toolchain signal extraction."""
    rel = rel_path.replace('\\', '/').lstrip('./')
    basename = os.path.basename(rel).lower()
    depth = 0 if '/' not in rel else rel.count('/')

    if depth == 0:
        return basename in _ROOT_SIGNAL_FILES
    if depth > 2:
        return False
    return (
        basename in _ROOT_SIGNAL_FILES - {'configure'}
        or basename.endswith('.cmake')
        or basename.endswith('.mk')
    )


def _parse_cpp_std(value: str) -> int | None:
    """Normalize C++ standard values: '2a' -> 20, '1z' -> 17, etc."""
    mapping = {
        '98': 98, '03': 3, '0x': 11, '11': 11,
        '1y': 14, '14': 14, '1z': 17, '17': 17,
        '2a': 20, '20': 20, '2b': 23, '23': 23,
        '2c': 26, '26': 26,
    }
    return mapping.get(value)


def _max_cpp_std(*values: int | None) -> int | None:
    """Return the highest C++ standard from a set of values."""
    valid = [v for v in values if v is not None and v > 3]
    return max(valid) if valid else None


def extract_signals_from_file(
    filepath: str,
    content: str,
    signals: VersionSignals,
) -> None:
    """Extract version signals from a single build file into signals object."""
    basename = os.path.basename(filepath).lower()
    root_entry = "/" not in filepath.replace("\\", "/").lstrip("./")

    # ── CMake ──
    if basename == 'cmakelists.txt' or filepath.lower().endswith('.cmake'):
        signals.needs_cmake = True
        signals.needs_cmake_hard = signals.needs_cmake_hard or root_entry
        if not signals.build_system:
            signals.build_system = 'cmake'

        for m in _CMAKE_MIN_RE.finditer(content):
            ver = (int(m.group(1)), int(m.group(2)))
            if signals.cmake_minimum is None or ver > signals.cmake_minimum:
                signals.cmake_minimum = ver
                signals.cmake_minimum_hard = signals.cmake_minimum_hard or root_entry
            if root_entry and (signals.cmake_minimum_root is None or ver > signals.cmake_minimum_root):
                signals.cmake_minimum_root = ver
                signals.signals_found.append(
                    f"cmake_minimum_required({ver[0]}.{ver[1]}) in {filepath}"
                )

        for m in _CMAKE_CXX_STD_RE.finditer(content):
            std = _parse_cpp_std(m.group(1))
            if std and (signals.cmake_cxx_standard is None or std > signals.cmake_cxx_standard):
                signals.cmake_cxx_standard = std
                signals.cmake_cxx_standard_hard = signals.cmake_cxx_standard_hard or root_entry
                if root_entry:
                    signals.cmake_cxx_standard_root = max(signals.cmake_cxx_standard_root or 0, std)
                signals.signals_found.append(
                    f"CMAKE_CXX_STANDARD {std} in {filepath}"
                )

        for m in _CMAKE_CXX_STD_TARGET_RE.finditer(content):
            std = _parse_cpp_std(m.group(1))
            if std and (signals.cmake_cxx_standard is None or std > signals.cmake_cxx_standard):
                signals.cmake_cxx_standard = std
                signals.cmake_cxx_standard_hard = signals.cmake_cxx_standard_hard or root_entry
                if root_entry:
                    signals.cmake_cxx_standard_root = max(signals.cmake_cxx_standard_root or 0, std)
                signals.signals_found.append(
                    f"cxx_std_{std} target feature in {filepath}"
                )

    # ── Meson ──
    elif basename == 'meson.build' or basename == 'meson_options.txt' or basename == 'meson.options':
        signals.needs_meson = True
        signals.needs_meson_hard = signals.needs_meson_hard or root_entry
        if not signals.build_system:
            signals.build_system = 'meson'

        for m in _MESON_VERSION_RE.finditer(content):
            ver = (int(m.group(1)), int(m.group(2)))
            if signals.meson_version is None or ver > signals.meson_version:
                signals.meson_version = ver
                signals.meson_version_hard = signals.meson_version_hard or root_entry
            if root_entry and (signals.meson_version_root is None or ver > signals.meson_version_root):
                signals.meson_version_root = ver
                signals.signals_found.append(
                    f"meson_version >= {ver[0]}.{ver[1]} in {filepath}"
                )

        for m in _MESON_CPP_STD_RE.finditer(content):
            std = _parse_cpp_std(m.group(1))
            if std and (signals.meson_cpp_std is None or std > signals.meson_cpp_std):
                signals.meson_cpp_std = std
                signals.meson_cpp_std_hard = signals.meson_cpp_std_hard or root_entry
                if root_entry:
                    signals.meson_cpp_std_root = max(signals.meson_cpp_std_root or 0, std)
                signals.signals_found.append(
                    f"cpp_std=c++{std} in {filepath}"
                )

    # ── Autotools (configure.ac / configure.in) ──
    elif basename in ('configure.ac', 'configure.in'):
        if not signals.build_system:
            signals.build_system = 'autotools'

        for m in _AX_CXX_STDCXX_RE.finditer(content):
            std = _parse_cpp_std(m.group(1))
            if std and (signals.autotools_cpp_std is None or std > signals.autotools_cpp_std):
                signals.autotools_cpp_std = std
                signals.autotools_cpp_std_hard = signals.autotools_cpp_std_hard or root_entry
                if root_entry:
                    signals.autotools_cpp_std_root = max(signals.autotools_cpp_std_root or 0, std)
                signals.signals_found.append(
                    f"AX_CXX_COMPILE_STDCXX({std}) in {filepath}"
                )

        for m in _AC_CXX_STD_RE.finditer(content):
            std = _parse_cpp_std(m.group(1))
            if std and (signals.autotools_cpp_std is None or std > signals.autotools_cpp_std):
                signals.autotools_cpp_std = std
                signals.autotools_cpp_std_hard = signals.autotools_cpp_std_hard or root_entry
                if root_entry:
                    signals.autotools_cpp_std_root = max(signals.autotools_cpp_std_root or 0, std)
                signals.signals_found.append(
                    f"-std=c++{std} in {filepath}"
                )

    # ── Makefile / GNUmakefile ──
    elif basename in ('makefile', 'gnumakefile') or basename.endswith('.mk') or basename == 'makefile.am':
        if not signals.build_system:
            signals.build_system = 'make'

        for m in _MAKEFILE_STD_RE.finditer(content):
            std = _parse_cpp_std(m.group(1))
            if std and (signals.makefile_cpp_std is None or std > signals.makefile_cpp_std):
                signals.makefile_cpp_std = std
                signals.makefile_cpp_std_hard = signals.makefile_cpp_std_hard or root_entry
                if root_entry:
                    signals.makefile_cpp_std_root = max(signals.makefile_cpp_std_root or 0, std)
                signals.signals_found.append(
                    f"-std=c++{std} in CXXFLAGS in {filepath}"
                )

        # Also check inline -std= in any context (link commands, etc.)
        if signals.makefile_cpp_std is None:
            for m in _MAKEFILE_STD_INLINE_RE.finditer(content):
                std = _parse_cpp_std(m.group(1))
                if std and (signals.makefile_cpp_std is None or std > signals.makefile_cpp_std):
                    signals.makefile_cpp_std = std
                    signals.makefile_cpp_std_hard = signals.makefile_cpp_std_hard or root_entry
                    if root_entry:
                        signals.makefile_cpp_std_root = max(signals.makefile_cpp_std_root or 0, std)
                    signals.signals_found.append(
                        f"-std=c++{std} inline in {filepath}"
                    )

    # ── Cargo.toml ──
    elif basename == 'cargo.toml':
        signals.needs_cargo = True
        signals.needs_cargo_hard = signals.needs_cargo_hard or root_entry
        if not signals.build_system:
            signals.build_system = 'cargo'
        signals.signals_found.append(f"Cargo.toml found: {filepath}")

    # ── configure script (generated) ──
    elif basename == 'configure':
        for m in _AC_CXX_STD_RE.finditer(content[:50000]):  # limit scan size
            std = _parse_cpp_std(m.group(1))
            if std and (signals.autotools_cpp_std is None or std > signals.autotools_cpp_std):
                signals.autotools_cpp_std = std
                signals.autotools_cpp_std_hard = signals.autotools_cpp_std_hard or root_entry
                if root_entry:
                    signals.autotools_cpp_std_root = max(signals.autotools_cpp_std_root or 0, std)
                signals.signals_found.append(
                    f"-std=c++{std} in configure script {filepath}"
                )

    # ── GCC version requirements (any file) ──
    for m in _GCC_VERSION_RE.finditer(content[:20000]):
        gcc_ver = int(m.group(1))
        if gcc_ver >= 4 and (signals.gcc_minimum is None or gcc_ver > signals.gcc_minimum):
            signals.gcc_minimum = gcc_ver
            signals.gcc_minimum_hard = signals.gcc_minimum_hard or root_entry
            if root_entry:
                signals.gcc_minimum_root = max(signals.gcc_minimum_root or 0, gcc_ver)
            signals.signals_found.append(
                f"GCC >= {gcc_ver} in {filepath}"
            )


def extract_signals(repo_path: str) -> VersionSignals:
    """Walk the repo and extract all version signals from build files."""
    signals = VersionSignals()
    root = str(Path(repo_path).resolve())

    SKIP_DIRS = {
        '.git', 'node_modules', '__pycache__', 'debian', '.svn',
        '3rdparty', 'thirdparty', 'vendor', 'external', 'deps',
        'subprojects', 'bundled', 'third_party',
    }

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        depth = 0 if rel_dir == '.' else rel_dir.count(os.sep) + 1
        if depth > 2:
            dirnames.clear()
            continue

        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS and not d.startswith('.')]

        if depth == 0:
            rel_dir = ''

        for fname in filenames:
            rel_path = fname if not rel_dir else os.path.join(rel_dir, fname)
            if not should_extract_signal_file(rel_path):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                content = open(fpath, errors='replace').read()
                extract_signals_from_file(rel_path, content, signals)
            except (OSError, IOError):
                pass

    return signals


# ─── Decision logic: signals → minimum Debian codename ────────────────────────

def _codename_index(codename: str) -> int:
    """Return index in CODENAME_ORDER, or -1 if not found."""
    try:
        return CODENAME_ORDER.index(codename)
    except ValueError:
        return -1


def _min_debian_for_cmake(cmake_ver: tuple[int, int]) -> str | None:
    """Find minimum Debian that ships cmake >= cmake_ver."""
    for tc in TOOLCHAINS:
        if tc.cmake >= cmake_ver:
            return tc.codename
    return None  # requirement exceeds all known Debian versions


def _min_debian_for_meson(meson_ver: tuple[int, int]) -> str | None:
    """Find minimum Debian that ships meson >= meson_ver."""
    for tc in TOOLCHAINS:
        if tc.meson is not None and tc.meson >= meson_ver:
            return tc.codename
    return None


def _min_debian_for_cpp_std(std: int) -> str | None:
    """Find minimum Debian whose GCC fully supports the given C++ standard."""
    for tc in TOOLCHAINS:
        if tc.cpp_std_max >= std:
            return tc.codename
    return None


def _min_debian_for_gcc(gcc_major: int) -> str | None:
    """Find minimum Debian that ships GCC >= gcc_major."""
    for tc in TOOLCHAINS:
        if tc.gcc_major >= gcc_major:
            return tc.codename
    return None


def _min_debian_for_meson_available() -> str:
    """Minimum Debian that has meson packaged at all."""
    for tc in TOOLCHAINS:
        if tc.meson is not None:
            return tc.codename
    return "trixie"


def _highest_codename(*codenames: str | None) -> str:
    """Return the most recent (highest-index) codename from the arguments."""
    best_idx = -1
    best = "bookworm"  # safe default
    for cn in codenames:
        if cn is None:
            continue
        idx = _codename_index(cn)
        if idx > best_idx:
            best_idx = idx
            best = cn
    return best


@dataclass
class SysuseDecision:
    """Result of sysuse selection with full audit trail."""
    codename: str
    source: str                    # 'version_signals' | 'commit_date' | 'default'
    constraints: list[str] = field(default_factory=list)
    signals: VersionSignals | None = None
    commit_date_codename: str | None = None


def decide_sysuse(signals: VersionSignals) -> SysuseDecision:
    """Given extracted signals, determine the minimum Debian codename.

    Returns the highest (most recent) Debian that satisfies ALL constraints.
    If no signals found, returns None codename so caller can fall back.

    Constraint strength (only hard constraints push the sysuse upward):
      - HARD:  C++ standard, explicit GCC minimum, Cargo/Rust, meson.build present
               without a version pin (meson itself only exists stretch+).
               These can't be worked around — older toolchains genuinely lack
               the feature or package.
      - SOFT:  cmake_minimum_required, explicit meson_version. These are
               advisory minimums ("tested with X.Y") and cmake/meson are
               backwards compatible, so an older cmake/meson usually still
               builds the project. More importantly, CMakeLists.txt and
               meson.build in old commits are frequently added by later
               maintainers and do NOT reflect the era the source was written
               for, so trusting them blindly pushes the sysuse to a much
               newer Debian than the source's ABI era. We record them as
               audit-only constraints but do not let them raise the floor.
    """
    constraints: list[str] = []
    candidates: list[str | None] = []

    # 1. CMake version requirement. Root-entry constraints are hard; nested
    # project constraints remain audit-only until that entry is selected.
    cmake_minimum = signals.cmake_minimum_root or signals.cmake_minimum
    if cmake_minimum:
        cn = _min_debian_for_cmake(cmake_minimum)
        ver_str = f"{cmake_minimum[0]}.{cmake_minimum[1]}"
        strength = "hard" if signals.cmake_minimum_hard else "soft"
        constraints.append(f"cmake >= {ver_str} -> {cn or 'UNSATISFIABLE'} ({strength})")
        if signals.cmake_minimum_hard:
            candidates.append(cn)

    # 2. Meson version requirement follows the same root-entry strength rule.
    meson_version = signals.meson_version_root or signals.meson_version
    if meson_version:
        cn = _min_debian_for_meson(meson_version)
        ver_str = f"{meson_version[0]}.{meson_version[1]}"
        strength = "hard" if signals.meson_version_hard else "soft"
        constraints.append(f"meson >= {ver_str} -> {cn or 'UNSATISFIABLE'} ({strength})")
        if signals.meson_version_hard:
            candidates.append(cn)
    elif signals.needs_meson:
        cn = _min_debian_for_meson_available()
        strength = "hard" if signals.needs_meson_hard else "soft"
        if signals.needs_meson_hard:
            candidates.append(cn)
        constraints.append(f"meson required (no version) -> {cn} ({strength})")

    # 3. C++ standard requirement (take the max across all build systems) — HARD
    cpp_std = _max_cpp_std(
        signals.cmake_cxx_standard_root or signals.cmake_cxx_standard,
        signals.meson_cpp_std_root or signals.meson_cpp_std,
        signals.autotools_cpp_std_root or signals.autotools_cpp_std,
        signals.makefile_cpp_std_root or signals.makefile_cpp_std,
    )
    cpp_hard = bool(
        signals.cmake_cxx_standard_hard
        or signals.meson_cpp_std_hard
        or signals.autotools_cpp_std_hard
        or signals.makefile_cpp_std_hard
    )
    if cpp_std:
        cn = _min_debian_for_cpp_std(cpp_std)
        if cpp_hard:
            candidates.append(cn)
        constraints.append(f"C++{cpp_std} -> {cn or 'UNSATISFIABLE'} ({'hard' if cpp_hard else 'soft'})")

    # 4. Explicit GCC version requirement — HARD
    gcc_minimum = signals.gcc_minimum_root or signals.gcc_minimum
    if gcc_minimum:
        cn = _min_debian_for_gcc(gcc_minimum)
        if signals.gcc_minimum_hard:
            candidates.append(cn)
        constraints.append(f"GCC >= {gcc_minimum} -> {cn or 'UNSATISFIABLE'} ({'hard' if signals.gcc_minimum_hard else 'soft'})")

    # 5. Cargo/Rust — HARD
    if signals.needs_cargo:
        cn = "buster"  # Rust 1.34+ in buster, sufficient for most crates
        if signals.needs_cargo_hard:
            candidates.append(cn)
        constraints.append(f"Cargo/Rust project -> {cn} ({'hard' if signals.needs_cargo_hard else 'soft'})")

    # 6. CMake project without explicit version — HARD (need cmake to exist)
    if signals.needs_cmake and not signals.cmake_minimum:
        # CMake 3.0 is in jessie, but most real projects need at least 3.5+
        # Use stretch as a safe floor for cmake projects
        cn = "stretch"
        if signals.needs_cmake_hard:
            candidates.append(cn)
        constraints.append(f"cmake required (no version) -> {cn} ({'hard' if signals.needs_cmake_hard else 'soft'})")

    if not candidates:
        return SysuseDecision(
            codename="",
            source="no_signals",
            constraints=constraints,
            signals=signals,
        )

    # If any hard constraint is unsatisfiable, use trixie (newest available)
    if None in candidates:
        return SysuseDecision(
            codename="trixie",
            source="version_signals",
            constraints=constraints + ["WARNING: some constraints exceed trixie"],
            signals=signals,
        )

    chosen = _highest_codename(*candidates)
    return SysuseDecision(
        codename=chosen,
        source="version_signals",
        constraints=constraints,
        signals=signals,
    )


# ─── Main entry point ────────────────────────────────────────────────────────

def select_sysuse(
    repo_path: str,
    commit: str = "HEAD",
    *,
    prefer_newer: bool = True,
) -> SysuseDecision:
    """Select the best Debian sysuse for building a project.

    Strategy:
    1. Extract version signals from build files (cmake, meson, autotools, make, cargo)
    2. Compute minimum Debian that satisfies all constraints
    3. Compare with commit-date heuristic
    4. Return the NEWER of the two (prefer_newer=True) or the version-signal
       result if it's more recent than the date-based one

    Args:
        repo_path: Path to the source repository
        commit: Git commit to build (used for date fallback)
        prefer_newer: If True, always pick the newer of version-signal vs date-based.
                      If False, trust version signals when available, else date.

    Returns:
        SysuseDecision with chosen codename and audit trail
    """
    # Step 1: Extract version signals
    signals = extract_signals(repo_path)

    # Step 2: Decide based on signals
    decision = decide_sysuse(signals)

    # Step 3: Get commit-date fallback
    date_codename = commit_date_to_sysuse(repo_path, commit)
    decision.commit_date_codename = date_codename

    # Step 4: Merge strategies
    if not decision.codename:
        # No version signals found, use date-based
        decision.codename = date_codename
        decision.source = "commit_date"
        decision.constraints.append(f"no version signals, using commit date -> {date_codename}")
        return decision

    # When version signals exist, decide whether to bump up
    signal_idx = _codename_index(decision.codename)
    date_idx = _codename_index(date_codename)

    if prefer_newer:
        # Use whichever is newer - the project needs at least the version-signal
        # minimum, but if the commit is from a newer era, the newer Debian is
        # more likely to have the right ecosystem
        if date_idx > signal_idx:
            decision.constraints.append(
                f"commit_date suggests {date_codename} (newer than signal-based {decision.codename}), upgrading"
            )
            decision.codename = date_codename
            decision.source = "version_signals+date_upgrade"
    else:
        # Trust version signals strictly; only use date if signals say nothing
        if signal_idx < date_idx:
            decision.constraints.append(
                f"version signals say {decision.codename}, commit date says {date_codename}; keeping signal-based"
            )

    return decision


def select_sysuse_codename(
    repo_path: str,
    commit: str = "HEAD",
) -> str:
    """Convenience wrapper returning just the codename string."""
    return select_sysuse(repo_path, commit).codename


# ─── CLI for testing ──────────────────────────────────────────────────────────

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Version-aware Debian sysuse selection"
    )
    parser.add_argument("repo_path", help="Path to source repository")
    parser.add_argument("--commit", default="HEAD", help="Git commit (default: HEAD)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all signals")
    args = parser.parse_args()

    decision = select_sysuse(args.repo_path, args.commit)

    if args.json:
        out = {
            "codename": decision.codename,
            "sysuse": f"debian:{decision.codename}",
            "source": decision.source,
            "constraints": decision.constraints,
            "commit_date_codename": decision.commit_date_codename,
            "build_system": decision.signals.build_system if decision.signals else "",
            "signals_found": decision.signals.signals_found if decision.signals else [],
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"sysuse: debian:{decision.codename}")
        print(f"source: {decision.source}")
        if decision.commit_date_codename:
            print(f"commit_date_fallback: {decision.commit_date_codename}")
        if decision.constraints:
            print("constraints:")
            for c in decision.constraints:
                print(f"  - {c}")
        if args.verbose and decision.signals and decision.signals.signals_found:
            print("signals:")
            for s in decision.signals.signals_found:
                print(f"  - {s}")


if __name__ == "__main__":
    main()
