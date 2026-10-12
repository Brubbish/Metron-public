"""Phase-1 source dependency resolver.

Identifies, clones, and Phase-1 analyzes external source dependencies that the
main project requires but apt cannot satisfy (version-floor mismatch, missing
package, or explicit doc-provided git URL).

The resolver is **clone-only**: it never builds anything itself. Phase 2 is
responsible for executing each dep's ``build_order`` inside the container.

Pipeline
--------
1. ``collect_source_dep_candidates`` merges 3 sources of candidates:
   - ``Cargo.lock`` git refs (already extracted by ``build_hints``)
   - Doc-extracted source deps (from ``summarize_artifact`` LLM output)
   - ``configure.ac`` PKG_CHECK_MODULES version constraints that the current
     sysuse's apt cannot satisfy (best-effort; needs a URL hint from docs)
2. ``recursive_clone_and_phase1`` clones each candidate (max_depth=2),
   runs mini-Phase-1 (``discover_build_artifacts`` + ``summarize_all_artifacts_async``)
   on each clone to extract apt_packages and build_commands, and recurses
   one level for transitive source deps.
3. Result is a list of ``ClonedSourceDep`` dicts merged into Phase 2's
   ``source_dep_plan``.

Clone layout (host)
-------------------
source deps live in a sibling directory of the main worktree so they do not
pollute the main repo's working tree::

    <batch output_root>/
    ├── worktrees/<index>_<owner>__<repo>__<commit12>/    ← main repo (compile_path)
    └── source_deps/<index>_<owner>__<repo>__<commit12>/  ← clone root
        └── libyang/                                       ← one dep
            ├── CMakeLists.txt, ...
            └── _phase1/
                ├── readrepo.txt        (mini-Phase-1 output)
                └── artifact_summaries.json

The clone root is mounted into the container at ``/source_deps`` (read-write),
so Phase 2 sees each dep at ``/source_deps/<dep_name>/``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

# Make sibling modules importable when this file is run from agent_build_main
_CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from core.source_dep_registry import match_source_dep_registry


# ---------------------------------------------------------------------------
# Version constraint parsing
# ---------------------------------------------------------------------------

# PKG_CHECK_MODULES([LIBYANG], [libyang >= 2.1.128])
# PKG_CHECK_MODULES([GLIB], [glib-2.0 >= 2.48.0 gobject-2.0 >= 2.48.0])
# PKG_CHECK_MODULES([FOO], [foo])
_PKG_CHECK_MODULES_RE = re.compile(
    r'PKG_CHECK_MODULES\s*\(\s*\[?(\w+)\]?\s*,\s*\[?([^\],]+(?:,[^\],]+)*?)\]?\s*[,)]',
    re.MULTILINE,
)

# meson dependency('glib-2.0', version: '>= 2.48.0')
_MESON_DEPENDENCY_RE = re.compile(
    r"dependency\s*\(\s*['\"]([\w.-]+)['\"]\s*,\s*version\s*:\s*['\"]([^'\"]+)['\"]",
    re.MULTILINE,
)

# CMake find_package(libyang 2.1.128 REQUIRED)
_FIND_PACKAGE_RE = re.compile(
    r'find_package\s*\(\s*([\w.-]+)\s+([0-9][0-9.]*[0-9])\s+(?:REQUIRED|COMPONENTS)',
    re.MULTILINE,
)

# Single constraint: "libyang >= 2.1.128"
# Captures pkg name + operator + version
_VERSION_OP_RE = re.compile(r'^\s*([\w.-]+)\s*(>=|<=|==|>|<|=)?\s*([0-9][0-9.]*[0-9])?\s*$')


@dataclass
class VersionConstraint:
    """One pkg-config name + version floor/ceiling."""

    pkg: str
    operator: str = '>='
    version: Optional[str] = None

    def display(self) -> str:
        if self.version:
            return f"{self.pkg} {self.operator} {self.version}"
        return self.pkg


def parse_version_constraint(text: str) -> Optional[VersionConstraint]:
    """Parse "libyang >= 2.1.128" → VersionConstraint.

    Returns None if text is empty or cannot be parsed.
    """
    text = (text or '').strip()
    if not text:
        return None
    # Some callers pass "libyang-2.0 >= 2.1.128"; split on first operator
    m = re.match(
        r'^\s*([\w.,+-]+?)\s*(>=|<=|==|>|<|=)?\s*([0-9][0-9.]*[0-9])?\s*$',
        text,
    )
    if not m:
        return None
    pkg = m.group(1).strip()
    op = (m.group(2) or '>=').strip()
    ver = m.group(3)
    if not pkg:
        return None
    return VersionConstraint(pkg=pkg, operator=op, version=ver)


def parse_configure_ac_constraints(content: str) -> list[VersionConstraint]:
    """Extract version-floor constraints from configure.ac / configure.in.

    Recognizes:
      PKG_CHECK_MODULES([LIBYANG], [libyang >= 2.1.128])
      PKG_CHECK_MODULES([GLIB], [glib-2.0 >= 2.48.0 gobject-2.0 >= 2.48.0])
      PKG_CHECK_MODULES([FOO], [foo])
    """
    constraints: list[VersionConstraint] = []
    seen_keys: set[tuple[str, str]] = set()
    for m in _PKG_CHECK_MODULES_RE.finditer(content):
        deps_text = m.group(2).strip('[] ')
        # Find all "pkg op version" or "pkg" chunks within the deps_text.
        # We use a tolerant regex that matches pkg-config names like
        # "glib-2.0" or "libyang-cplusplus".
        for chunk_match in re.finditer(
            r'([\w.+-]+)\s*(>=|<=|==|>|<|=)?\s*([0-9][0-9.]*[0-9])?',
            deps_text,
        ):
            pkg = chunk_match.group(1)
            op = (chunk_match.group(2) or '').strip()
            ver = chunk_match.group(3)
            # Skip when this match is just the version number continuing
            # from the previous match (regex backtracking artifact).
            if pkg.isdigit():
                continue
            vc = VersionConstraint(
                pkg=pkg,
                operator=op or '>=',
                version=ver if op else None,
            )
            key = (vc.pkg.lower(), vc.version or '')
            if key in seen_keys:
                continue
            seen_keys.add(key)
            constraints.append(vc)
    return constraints


def parse_meson_constraints(content: str) -> list[VersionConstraint]:
    """Extract version constraints from meson.build / meson_options.txt."""
    constraints: list[VersionConstraint] = []
    seen_keys: set[tuple[str, str]] = set()
    for m in _MESON_DEPENDENCY_RE.finditer(content):
        vc = VersionConstraint(pkg=m.group(1), operator='>=', version=m.group(2))
        key = (vc.pkg.lower(), vc.version or '')
        if key in seen_keys:
            continue
        seen_keys.add(key)
        constraints.append(vc)
    return constraints


def parse_cmake_constraints(content: str) -> list[VersionConstraint]:
    """Extract version constraints from CMakeLists.txt find_package(...)."""
    constraints: list[VersionConstraint] = []
    seen_keys: set[tuple[str, str]] = set()
    for m in _FIND_PACKAGE_RE.finditer(content):
        vc = VersionConstraint(pkg=m.group(1), operator='>=', version=m.group(2))
        key = (vc.pkg.lower(), vc.version or '')
        if key in seen_keys:
            continue
        seen_keys.add(key)
        constraints.append(vc)
    return constraints


def extract_version_constraints(repo_path: str) -> list[VersionConstraint]:
    """Scan configure.ac + meson.build + CMakeLists.txt for version constraints."""
    constraints: list[VersionConstraint] = []
    for fname in ('configure.ac', 'configure.in'):
        path = os.path.join(repo_path, fname)
        if os.path.isfile(path):
            try:
                with open(path, errors='replace') as f:
                    constraints.extend(parse_configure_ac_constraints(f.read()))
            except OSError:
                pass
    for root, _, files in os.walk(repo_path):
        # only top-level + depth-1 meson.build
        depth = os.path.relpath(root, repo_path).count(os.sep)
        if depth > 1:
            continue
        for fname in files:
            if fname == 'meson.build':
                try:
                    with open(os.path.join(root, fname), errors='replace') as f:
                        constraints.extend(parse_meson_constraints(f.read()))
                except OSError:
                    pass
            if fname == 'CMakeLists.txt':
                try:
                    with open(os.path.join(root, fname), errors='replace') as f:
                        constraints.extend(parse_cmake_constraints(f.read()))
                except OSError:
                    pass
    # Dedupe by (pkg, op, version)
    seen: set[tuple[str, str, str]] = set()
    deduped: list[VersionConstraint] = []
    for vc in constraints:
        key = (vc.pkg.lower(), vc.operator, vc.version or '')
        if key in seen:
            continue
        seen.add(key)
        deduped.append(vc)
    return deduped


# ---------------------------------------------------------------------------
# Semver helpers (minimal — no pre-release / build suffix support)
# ---------------------------------------------------------------------------

def _parse_semver(text: str) -> Optional[tuple[int, ...]]:
    """Parse "v2.1.128" / "2.1.128" / "v2.1" → (2, 1, 128)."""
    cleaned = re.sub(r'^[vV]', '', str(text).strip())
    parts = re.split(r'[._-]', cleaned)
    nums: list[int] = []
    for p in parts:
        if p.isdigit():
            nums.append(int(p))
        else:
            # stop at first non-numeric chunk ("2.1.128-rc1" → (2,1,128))
            break
    return tuple(nums) if nums else None


def _satisfies(version: tuple[int, ...], op: str, floor: tuple[int, ...]) -> bool:
    """Compare two semver tuples under op."""
    # Pad to equal length
    a = version + (0,) * (max(len(version), len(floor)) - len(version))
    b = floor + (0,) * (max(len(version), len(floor)) - len(floor))
    if op == '>=':
        return a >= b
    if op == '>':
        return a > b
    if op == '<=':
        return a <= b
    if op == '<':
        return a < b
    if op in ('==', '='):
        return a == b
    return a >= b  # default to >=


def select_min_satisfying_tag(
    tags: list[str],
    op: str,
    floor_version: str,
) -> Optional[str]:
    """Given a list of tags, return the smallest one that satisfies op/version.

    Picks minimum to avoid unintended major-version bumps (libyang 2.x vs 3.x).
    """
    floor_tuple = _parse_semver(floor_version)
    if floor_tuple is None:
        return None
    candidates: list[tuple[tuple[int, ...], str]] = []
    for tag in tags:
        v = _parse_semver(tag)
        if v is None:
            continue
        if _satisfies(v, op, floor_tuple):
            candidates.append((v, tag))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Candidate collection
# ---------------------------------------------------------------------------

@dataclass
class SourceDepCandidate:
    """A potential source dep to clone + Phase-1 analyze."""

    name: str
    clone_url: str
    origin: str  # "cargo_lock" | "doc_extracted" | "configure_ac"
    version_constraint: Optional[VersionConstraint] = None
    notes: str = ''
    setup_commands: list[str] = field(default_factory=list)
    build_order: list[str] = field(default_factory=list)
    instructions: str = ''
    checkout_ref: str = ''
    checkout_before_project_date: bool = False
    checkout_branch: str = ''

    def key(self) -> str:
        return self.name.lower()


def _collect_cargo_candidates(cargo_refs: list[dict]) -> list[SourceDepCandidate]:
    """Cargo.lock git refs → candidates."""
    out: list[SourceDepCandidate] = []
    for ref in cargo_refs or []:
        url = str(ref.get('url') or '').strip()
        name = str(ref.get('name') or '').strip()
        if not url or not name:
            continue
        out.append(SourceDepCandidate(
            name=name,
            clone_url=url,
            origin='cargo_lock',
            notes=f'pinned commit {ref.get("commit", "")[:12]}',
            checkout_ref=str(ref.get('commit') or '').strip(),
        ))
    return out


# Names that should never be treated as source deps because they are either
# (a) standard build tools provided by every Debian release (apt is always
#     preferred over source build for these), or
# (b) Python/Ruby/Node ecosystem packages handled by their own package
#     managers, not by source-cloning into the build tree.
_APT_PREFERRED_TOOLCHAIN_NAMES = frozenset({
    'autoconf', 'automake', 'libtool', 'bison', 'flex', 'yacc',
    'make', 'cmake', 'ninja', 'meson', 'pkg-config', 'gcc', 'g++',
    'clang', 'llvm', 'gdb', 'binutils', 'gettext', 'texinfo',
    'doxygen', 'sphinx', 'perl', 'python', 'python3', 'ruby', 'node',
    'swig', 'ragel', 'gperf', 'lex', 'yacc',
    'm4', 'dpkg-dev', 'debhelper',
})

# URL patterns that indicate the dep is for a different platform (not Debian).
# We skip those candidates because they cannot be consumed by our Debian-based
# build flow.
_NON_DEBIAN_URL_RE = re.compile(
    r'(?:yum|\.rpm|centos|fedora|redhat|rhel|suse|opensuse|'
    r'macports|brew\.sh|homebrew|cygwin|msys2|mingw|'
    r'pypi\.org|pythonhosted|rubygems|npmjs)',
    re.IGNORECASE,
)


def _is_apt_preferred_toolchain(name: str) -> bool:
    """Return True if this name refers to a standard build tool that Debian
    apt always provides — we should never source-build these."""
    return name.lower() in _APT_PREFERRED_TOOLCHAIN_NAMES


def _is_non_debian_url(url: str) -> bool:
    """Return True if the URL points to a non-Debian package source."""
    return bool(_NON_DEBIAN_URL_RE.search(url))


def _collect_doc_candidates(doc_source_deps: list[dict]) -> list[SourceDepCandidate]:
    """LLM-extracted source_deps from artifact summarizers → candidates."""
    out: list[SourceDepCandidate] = []
    for entry in doc_source_deps or []:
        if isinstance(entry, dict):
            name = str(entry.get('name') or '').strip()
            url = str(entry.get('url') or '').strip()
            reason = str(entry.get('reason') or '').strip()
            build_commands = [str(item).strip() for item in (entry.get('build_commands') or []) if str(item).strip()]
            setup_commands = [str(item).strip() for item in (entry.get('setup_commands') or []) if str(item).strip()]
            checkout_ref = str(entry.get('revision') or entry.get('commit') or entry.get('ref') or '').strip()
        else:
            name = str(entry).strip()
            url = ''
            reason = ''
            build_commands = []
            setup_commands = []
            checkout_ref = ''
        if not name:
            continue
        if not url:
            # Skip if URL not provided; we cannot clone without one
            continue
        # Filter: standard toolchain — apt is always preferred
        if _is_apt_preferred_toolchain(name):
            continue
        # Filter: URLs for other platforms (rpm/yum/centos/brew/etc.)
        if _is_non_debian_url(url):
            continue
        out.append(SourceDepCandidate(
            name=name,
            clone_url=url,
            origin='doc_extracted',
            notes=reason,
            setup_commands=setup_commands,
            build_order=build_commands,
            instructions=str(entry.get('instructions') or '').strip() if isinstance(entry, dict) else '',
            checkout_ref=checkout_ref,
        ))
    return out


def _collect_registry_candidates(repo_path: str) -> list[SourceDepCandidate]:
    """Known project-specific source deps from the local registry."""
    out: list[SourceDepCandidate] = []
    for entry in match_source_dep_registry(repo_path):
        out.append(SourceDepCandidate(
            name=entry.name,
            clone_url=entry.clone_url,
            origin='registry',
            notes=entry.reason,
            setup_commands=list(entry.setup_commands),
            build_order=list(entry.build_order),
            instructions=entry.instructions,
            checkout_ref=entry.checkout_ref,
            checkout_before_project_date=entry.checkout_before_project_date,
            checkout_branch=entry.checkout_branch,
        ))
    return out


def _collect_configure_ac_candidates(
    constraints: list[VersionConstraint],
    doc_source_deps: list[dict],
) -> list[SourceDepCandidate]:
    """configure.ac version-floor constraints → candidates (URL from docs).

    We only emit a candidate if the same dep name appears in doc_source_deps
    with a URL — we cannot invent clone URLs from pkg-config names alone.
    """
    # Build name → url index from doc_source_deps
    doc_url_by_name: dict[str, str] = {}
    for entry in doc_source_deps or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get('name') or '').strip().lower()
        url = str(entry.get('url') or '').strip()
        if name and url:
            doc_url_by_name[name] = url
            # Also index common pkg-config aliases (libfoo → foo, libfoo-2.0 → libfoo)
            if name.startswith('lib'):
                doc_url_by_name[name[3:]] = url
            base = re.sub(r'-[\d.]+$', '', name)
            if base != name:
                doc_url_by_name[base] = url

    out: list[SourceDepCandidate] = []
    for vc in constraints:
        pkg_lower = vc.pkg.lower()
        url = doc_url_by_name.get(pkg_lower) or doc_url_by_name.get(
            pkg_lower.replace('-2.0', '').replace('-2', '')
        )
        if not url:
            continue
        out.append(SourceDepCandidate(
            name=vc.pkg,
            clone_url=url,
            origin='configure_ac',
            version_constraint=vc,
            notes=vc.display(),
        ))
    return out


def collect_source_dep_candidates(
    repo_path: str,
    doc_source_deps: list[dict],
    cargo_refs: list[dict],
) -> list[SourceDepCandidate]:
    """Merge 3 sources of source-dep candidates and dedupe by name.

    Submodule deps are intentionally excluded — those are handled separately by
    ``preflight_git_submodules`` and surface via ``submodule_plan``.
    """
    registry = _collect_registry_candidates(repo_path)
    cargo = _collect_cargo_candidates(cargo_refs)
    docs = _collect_doc_candidates(doc_source_deps)
    constraints = extract_version_constraints(repo_path)
    configure_ac = _collect_configure_ac_candidates(constraints, doc_source_deps)

    # Filter out the project itself — LLM summarizers sometimes extract the
    # main repo's "how to build me" snippet as a source_dep.
    repo_basename = os.path.basename(os.path.normpath(repo_path)).lower()
    # Strip the index prefix and commit hash from worktree names
    # (e.g. "00027_frrouting__frr__b0800bfdf04b" → "frr")
    repo_realname = ''
    if '__' in repo_basename:
        parts = repo_basename.split('__')
        if len(parts) >= 2:
            repo_realname = parts[1].lower()
    elif '_' in repo_basename:
        repo_realname = repo_basename.split('_', 1)[1].lower()
    project_aliases = {repo_basename, repo_realname} if repo_realname else {repo_basename}

    def _is_self_reference(name: str, url: str) -> bool:
        name_lower = name.lower()
        if name_lower in project_aliases:
            return True
        # URL heuristic: if URL ends with "/<repo_name>.git"
        url_lower = url.lower()
        for alias in project_aliases:
            if alias and (url_lower.endswith(f'/{alias}.git') or f'/{alias}.' in url_lower):
                return True
        return False

    def _is_python_pip_dep(url: str, build_commands: list) -> bool:
        """Skip Python pip-installed deps — they go through pip, not source build."""
        if 'pypi.org' in url.lower() or 'pythonhosted' in url.lower():
            return True
        for cmd in (build_commands or []):
            cmd_lower = str(cmd).lower()
            if cmd_lower.startswith('pip install') or 'pip install git+' in cmd_lower:
                return True
        return False

    seen: set[str] = set()
    merged: list[SourceDepCandidate] = []
    # registry first (known successful/manual patterns), then configure_ac
    # (specific version constraint), docs, cargo.
    for c in registry + configure_ac + docs + cargo:
        if c.key() in seen:
            continue
        if _is_self_reference(c.name, c.clone_url):
            continue
        # exabgp-style: Python pip deps are handled by pip, not source build
        # (we don't have a clean way to source-build Python packages into /usr)
        build_cmds_for_check = []
        for entry in (doc_source_deps or []):
            if isinstance(entry, dict) and entry.get('name', '').lower() == c.name.lower():
                build_cmds_for_check = entry.get('build_commands') or []
                break
        if _is_python_pip_dep(c.clone_url, build_cmds_for_check):
            continue
        seen.add(c.key())
        merged.append(c)
    return merged


# ---------------------------------------------------------------------------
# Cloning
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: str, timeout: int = 120) -> tuple[int, str, str]:
    """Run git with GIT_TERMINAL_PROMPT=0; return (rc, stdout, stderr)."""
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}
    try:
        result = subprocess.run(
            ['git', *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return (
            int(result.returncode or 0),
            result.stdout or '',
            result.stderr or '',
        )
    except subprocess.TimeoutExpired as e:
        return 124, '', f'git timeout after {timeout}s: {e}'
    except Exception as e:
        return 1, '', f'git exception: {e}'


def _git_ls_remote_tags(url: str, timeout: int = 30) -> list[str]:
    """Return list of tag refs (without the refs/tags/ prefix)."""
    rc, stdout, stderr = _git(['ls-remote', '--tags', url], cwd='.', timeout=timeout)
    if rc != 0:
        return []
    tags: list[str] = []
    for line in stdout.splitlines():
        parts = line.split('\t', 1)
        if len(parts) != 2:
            continue
        ref = parts[1]
        if ref.startswith('refs/tags/'):
            tag = ref[len('refs/tags/'):]
            # Skip dereferenced commit pointers (refs/tags/X^{})
            if tag.endswith('^{}'):
                tag = tag[:-3]
            if tag not in tags:
                tags.append(tag)
    return tags


def clone_one_dep(
    candidate: SourceDepCandidate,
    dest_path: str,
    timeout: int = 180,
    project_commit_date: str = "",
) -> tuple[bool, str]:
    """Clone a single source dep to ``dest_path``.

    Returns (success, message). On success, also checks out the tag/commit
    that best satisfies ``candidate.version_constraint``.
    """
    # "Already cloned" = has a .git dir. Just having files (e.g. a `_phase1/`
    # subdir created by the orchestrator before clone) is NOT enough.
    if os.path.isdir(os.path.join(dest_path, '.git')):
        return True, 'already present'
    # Clean any stale non-git content (e.g. partial state from a prior failed run)
    if os.path.exists(dest_path):
        import shutil
        try:
            shutil.rmtree(dest_path)
        except OSError:
            pass

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    parent = os.path.dirname(dest_path)
    base = os.path.basename(dest_path)
    rc, _, stderr = _git(
        ['clone', '--no-checkout', candidate.clone_url, base],
        cwd=parent,
        timeout=timeout,
    )
    if rc != 0:
        return False, f'git clone failed: {stderr.strip()[:200]}'

    # Determine ref to checkout
    ref_to_checkout: Optional[str] = None
    if candidate.checkout_ref:
        ref_to_checkout = candidate.checkout_ref
    elif candidate.checkout_before_project_date and project_commit_date:
        rc, stdout, _ = _git(
            ['rev-list', '-n', '1', f'--before={project_commit_date}', '--all'],
            cwd=dest_path,
            timeout=60,
        )
        ref_to_checkout = stdout.strip() if rc == 0 and stdout.strip() else 'HEAD'
    elif candidate.version_constraint and candidate.version_constraint.version:
        tags = _git_ls_remote_tags(candidate.clone_url, timeout=30)
        ref_to_checkout = select_min_satisfying_tag(
            tags,
            candidate.version_constraint.operator,
            candidate.version_constraint.version,
        )
        if ref_to_checkout is None:
            # No tag satisfies — fall back to default branch HEAD
            ref_to_checkout = 'HEAD'
    elif project_commit_date:
        # Prefer the newest reachable source revision no later than the target
        # project commit when the repository did not provide an explicit pin.
        rc, stdout, _ = _git(
            ['rev-list', '-n', '1', f'--before={project_commit_date}', '--all'],
            cwd=dest_path,
            timeout=60,
        )
        ref_to_checkout = stdout.strip() if rc == 0 and stdout.strip() else 'HEAD'
    else:
        # No project date is available; retain the repository default.
        ref_to_checkout = 'HEAD'

    rc, _, stderr = _git(['checkout', ref_to_checkout], cwd=dest_path, timeout=60)
    if rc != 0:
        return False, f'git checkout {ref_to_checkout} failed: {stderr.strip()[:200]}'

    return True, f'checked out {ref_to_checkout}'


# ---------------------------------------------------------------------------
# Mini Phase-1 for cloned deps
# ---------------------------------------------------------------------------

async def _mini_phase1(
    dep_path: str,
    sysuse: str,
    log_dir: str,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> dict:
    """Run mini-Phase-1 on a cloned source dep.

    Returns dict with keys: apt_packages, source_deps, setup_commands,
    build_commands, notes.
    """
    from core.build_artifact_discovery import discover_build_artifacts
    from core.build_artifact_discovery.summarizers import summarize_all_artifacts_async

    async def _do() -> dict:
        artifacts = discover_build_artifacts(dep_path)
        if not artifacts:
            return {'apt_packages': [], 'source_deps': [], 'setup_commands': [], 'build_commands': [], 'notes': 'no build artifacts found'}
        try:
            summaries = await summarize_all_artifacts_async(
                dep_path, artifacts, semaphore=semaphore,
            )
        except Exception as e:
            return {'apt_packages': [], 'source_deps': [], 'setup_commands': [], 'build_commands': [], 'notes': f'summarize failed: {e}'}

        apt_packages: list[str] = []
        build_commands: list[str] = []
        source_deps: list[dict] = []
        setup_commands: list[dict] = []
        notes_parts: list[str] = []
        for s in summaries:
            for pkg in (s.apt_packages or []):
                if pkg and pkg not in apt_packages:
                    apt_packages.append(pkg)
            for cmd in (s.build_commands or []):
                cmd = str(cmd).strip()
                # Skip CI-template lines like `CC=${{ matrix.config.cc }}` —
                # these are not real shell commands and would confuse Phase 2.
                if '${{' in cmd or 'matrix.' in cmd or 'env.' in cmd and '${' in cmd:
                    continue
                if cmd and cmd not in build_commands:
                    build_commands.append(cmd)
            for sd in (s.source_deps or []):
                source_deps.append(sd)
            for sc in (s.setup_commands or []):
                setup_commands.append(sc)
            if s.notes:
                notes_parts.append(f'{os.path.basename(s.path)}: {s.notes}')

        return {
            'apt_packages': apt_packages,
            'source_deps': source_deps,
            'setup_commands': setup_commands,
            'build_commands': build_commands,
            'notes': '; '.join(notes_parts) if notes_parts else '',
        }

    if semaphore is not None:
        async with semaphore:
            return await _do()
    return await _do()


# ---------------------------------------------------------------------------
# ClonedSourceDep output type
# ---------------------------------------------------------------------------

def _make_cloned_dep(
    candidate: SourceDepCandidate,
    host_source_path: str,
    container_source_path: str,
    clone_status: str,
    phase1: dict,
    selected_ref: str = '',
    transitive: list[dict] = None,
    depth: int = 0,
    error_msg: str = '',
) -> dict:
    """Build a ClonedSourceDep dict (the source_dep_plan entry)."""
    build_order = list(phase1.get('build_commands', [])) if phase1 else []
    setup_commands = list(phase1.get('setup_commands', [])) if phase1 else []
    if clone_status == 'cloned':
        build_order = list(dict.fromkeys(list(candidate.build_order) + build_order))
        setup_commands = list(dict.fromkeys(list(candidate.setup_commands) + setup_commands))
    return {
        'name': candidate.name,
        'clone_url': candidate.clone_url,
        'selected_ref': selected_ref,
        'host_source_path': host_source_path,
        'source_path': container_source_path,
        'clone_status': clone_status,  # cloned | failed
        'phase1_status': 'ok' if clone_status == 'cloned' and build_order else 'failed',
        'origin': candidate.origin,
        'version_constraint': (
            candidate.version_constraint.display()
            if candidate.version_constraint else ''
        ),
        'apt_packages': phase1.get('apt_packages', []) if phase1 else [],
        'build_order': build_order,
        'setup_commands': setup_commands,
        'notes': (phase1.get('notes', '') if phase1 else '') + (f' | {error_msg}' if error_msg else ''),
        'registry_notes': candidate.notes,
        'instructions': candidate.instructions,
        'checkout_before_project_date': bool(candidate.checkout_before_project_date),
        'checkout_branch': candidate.checkout_branch,
        'transitive_deps': transitive or [],
        'depth': depth,
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

DEFAULT_CONTAINER_SOURCE_ROOT = '/source_deps'


async def recursive_clone_and_phase1(
    candidates: list[SourceDepCandidate],
    host_clone_root: str,
    sysuse: str,
    max_depth: int = 2,
    timeout_per_dep: int = 180,
    container_source_root: str = DEFAULT_CONTAINER_SOURCE_ROOT,
    summarize_semaphore: Optional[asyncio.Semaphore] = None,
    project_commit_date: str = "",
) -> list[dict]:
    """Clone + mini-Phase-1 each candidate, recurse one level for transitive deps.

    Args:
        candidates: Top-level candidates from ``collect_source_dep_candidates``.
        host_clone_root: Host dir to clone into (e.g.
            ``<output_root>/source_deps/<worktree_basename>/``).
        sysuse: Debian codename for the main project (deps inherit it).
        max_depth: Recursion depth. 0 = no recursion; 2 = top + 1 nested level.
        timeout_per_dep: Per-dep clone timeout.
        container_source_root: Container path that ``host_clone_root`` will be
            mounted at (e.g. ``/source_deps``).
        summarize_semaphore: Optional concurrency limit for LLM summarize calls.

    Returns:
        List of ClonedSourceDep dicts ready for ``source_dep_plan``.
    """
    os.makedirs(host_clone_root, exist_ok=True)
    results: list[dict] = []

    # Cap at 5 top-level candidates to avoid excessive clone time
    # (neuron had 8 candidates, each needing clone + mini-phase1)
    capped = candidates[:5]
    if len(candidates) > 5:
        print(
            f"[SourceDepResolver] capping {len(candidates)} candidates to 5 "
            f"(skipped: {[c.name for c in candidates[5:]]})"
        )

    for candidate in capped:
        dep_dir_name = re.sub(r'[^A-Za-z0-9._-]', '_', candidate.name) or 'dep'
        host_dep_path = os.path.join(host_clone_root, dep_dir_name)
        container_dep_path = f'{container_source_root}/{dep_dir_name}'
        dep_log_dir = os.path.join(host_dep_path, '_phase1')
        os.makedirs(dep_log_dir, exist_ok=True)

        # Clone (sync — no concurrency per design)
        ok, msg = clone_one_dep(
            candidate,
            host_dep_path,
            timeout=timeout_per_dep,
            project_commit_date=project_commit_date,
        )
        if not ok:
            print(f'[SourceDepResolver] CLONE FAILED for {candidate.name}: {msg}')
            results.append(_make_cloned_dep(
                candidate,
                host_source_path=host_dep_path,
                container_source_path=container_dep_path,
                clone_status='failed',
                phase1={},
                error_msg=msg,
            ))
            continue

        # Determine selected_ref for record (read from git)
        rc, stdout, _ = _git(['rev-parse', 'HEAD'], cwd=host_dep_path, timeout=10)
        selected_ref = stdout.strip()[:12] if rc == 0 else ''

        # Mini phase1 (sync — no concurrency per design)
        print(f'[SourceDepResolver] mini phase1 for {candidate.name} (depth=0)')
        phase1 = await _mini_phase1(
            host_dep_path, sysuse, dep_log_dir, semaphore=summarize_semaphore,
        )
        # Save phase1 output for debugging
        try:
            with open(os.path.join(dep_log_dir, 'phase1.json'), 'w', encoding='utf-8') as f:
                json.dump(phase1, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

        # Recurse for transitive deps
        transitive: list[dict] = []
        if max_depth >= 1:
            transitive_candidates = collect_source_dep_candidates(
                host_dep_path,
                doc_source_deps=phase1.get('source_deps', []),
                cargo_refs=[],  # we don't re-extract Cargo.lock for the dep
            )
            # Filter out ones already processed at this level
            existing_names = {c.key() for c in candidates}
            transitive_candidates = [
                c for c in transitive_candidates if c.key() not in existing_names
            ]
            if transitive_candidates:
                print(
                    f'[SourceDepResolver] {candidate.name} has {len(transitive_candidates)} '
                    f'transitive candidates; recursing (depth=1)'
                )
                # Run transitive clone inline (one more level only — avoid deep stacks)
                for tc in transitive_candidates[:5]:  # cap at 5 transitives
                    t_dir_name = re.sub(r'[^A-Za-z0-9._-]', '_', tc.name) or 'tdep'
                    t_host = os.path.join(host_dep_path, '_transitive', t_dir_name)
                    t_container = f'{container_dep_path}/_transitive/{t_dir_name}'
                    t_log_dir = os.path.join(t_host, '_phase1')
                    os.makedirs(t_log_dir, exist_ok=True)
                    t_ok, t_msg = clone_one_dep(
                        tc,
                        t_host,
                        timeout=timeout_per_dep,
                        project_commit_date=project_commit_date,
                    )
                    if not t_ok:
                        print(f'[SourceDepResolver] transitive CLONE FAILED for {tc.name}: {t_msg}')
                        transitive.append(_make_cloned_dep(
                            tc,
                            host_source_path=t_host,
                            container_source_path=t_container,
                            clone_status='failed',
                            phase1={},
                            depth=1,
                            error_msg=t_msg,
                        ))
                        continue
                    t_phase1 = await _mini_phase1(
                        t_host, sysuse, t_log_dir, semaphore=summarize_semaphore,
                    )
                    t_rc, t_stdout, _ = _git(['rev-parse', 'HEAD'], cwd=t_host, timeout=10)
                    t_ref = t_stdout.strip()[:12] if t_rc == 0 else ''
                    transitive.append(_make_cloned_dep(
                        tc,
                        host_source_path=t_host,
                        container_source_path=t_container,
                        clone_status='cloned',
                        phase1=t_phase1,
                        selected_ref=t_ref,
                        depth=1,
                    ))

        results.append(_make_cloned_dep(
            candidate,
            host_source_path=host_dep_path,
            container_source_path=container_dep_path,
            clone_status='cloned',
            phase1=phase1,
            selected_ref=selected_ref,
            transitive=transitive,
            depth=0,
        ))

    return results


# ---------------------------------------------------------------------------
# Host clone-root path helper
# ---------------------------------------------------------------------------

def compute_host_clone_root(compile_path: str) -> str:
    """Compute the host source-dep clone root from the main repo path.

    Main repo (compile_path) lives at:
        <output_root>/worktrees/<index>_<owner>__<repo>__<commit12>/

    Source deps go to:
        <output_root>/source_deps/<index>_<owner>__<repo>__<commit12>/

    The output_root is computed as compile_path.parent.parent.
    """
    compile_path = os.path.abspath(compile_path)
    worktree_name = os.path.basename(compile_path.rstrip(os.sep))
    output_root = os.path.dirname(os.path.dirname(compile_path.rstrip(os.sep)))
    return os.path.join(output_root, 'source_deps', worktree_name)
