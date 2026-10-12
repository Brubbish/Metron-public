"""
Precision-focused build artifact discovery for C/C++ projects.

Discovers build-relevant documents (README, INSTALL, CI/CD, Dockerfile,
debian packaging) that contribute build steps and dependency info.
Returns 2-6 high-value files per project, not 20-200+.
"""

import os
import re
from dataclasses import dataclass


@dataclass
class DiscoveredArtifact:
    path: str       # relative path from project root
    kind: str       # "root_doc" | "debian_control" | "debian_rules" | "cicd" | "dockerfile"
    priority: int   # lower = more important


# ============================================================
# Root document matching (depth=0 only)
# ============================================================

_ROOT_DOC_RE = re.compile(
    r'^(readme|install|building)'
    r'(\.(md|rst|txt|html|htm|adoc|org|markdown))?$',
    re.IGNORECASE,
)

_ROOT_DOC_DENY_RE = re.compile(
    r'^install-sh$|^install_dep$|^installer$',
    re.IGNORECASE,
)

_BUILD_DOC_RE = re.compile(
    r'^build\.(md|rst|txt)$',
    re.IGNORECASE,
)

_LOCAL_BUILD_DOC_REF_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])((?:doc|docs|documentation)/"
    r"[A-Za-z0-9_./+-]*(?:build|install|depend|require|setup)"
    r"[A-Za-z0-9_./+-]*\.(?:md|rst|txt|adoc|org))",
    re.IGNORECASE,
)


# ============================================================
# CI/CD selection
# ============================================================

_CICD_PRIORITY_LIST = [
    'ci.yml', 'ci.yaml',
    'build.yml', 'build.yaml',
    'main.yml', 'main.yaml',
    'linux.yml', 'linux.yaml',
    'ubuntu.yml', 'ubuntu.yaml',
    'ccpp.yml', 'c-cpp.yml',
    'cmake.yml', 'cmake.yaml',
    'meson.yml', 'meson.yaml',
    'check.yml', 'check.yaml',
    'test.yml', 'test.yaml',
    'tests.yml', 'tests.yaml',
]

_CICD_DENY_RE = re.compile(
    r'(?:release|deploy|codeql|coverage|coverity|lint|format|'
    r'docs|doxygen|mirror|backup|windows|macos|mac-|'
    r'nightly|weekly|scorecard|license|chatops|commitlint|'
    r'hexpm|call-|android|mingw|fuzz)',
    re.IGNORECASE,
)

_CICD_BUILD_CONTENT_RE = re.compile(
    r'apt-get\s+install|apt\s+install|cmake|make\b|meson|'
    r'configure|ninja|autoreconf|autoconf',
)


# ============================================================
# Dockerfile matching
# ============================================================

_DOCKERFILE_RE = re.compile(
    r'^dockerfile(?:[.\-].*)?$',
    re.IGNORECASE,
)

_DOCKERFILE_DENY_RE = re.compile(
    r'(?:alpine|centos|fedora|windows|arm|aarch)',
    re.IGNORECASE,
)

_DOCKERFILE_PREFER_RE = re.compile(
    r'(?:debian|ubuntu|build|ci|linux)',
    re.IGNORECASE,
)


# ============================================================
# Main discovery function
# ============================================================

def discover_build_artifacts(
    project_root: str,
    *,
    include_report: bool = False,
) -> list[DiscoveredArtifact] | tuple[list[DiscoveredArtifact], dict]:
    """Discover build-relevant documents in a project.

    Returns a small set (typically 2-6) of high-value files:
    root docs, debian packaging, one CI workflow, one Dockerfile.
    """
    project_root = os.path.abspath(project_root)
    artifacts: list[DiscoveredArtifact] = []
    omitted: list[dict] = []
    root_docs = _find_root_docs(project_root)
    artifacts.extend(root_docs)
    artifacts.extend(_find_referenced_build_docs(project_root, root_docs))
    artifacts.extend(_find_debian(project_root))
    cicd = _find_cicd(project_root)
    dockerfiles = _find_dockerfile(project_root)
    dependency_docs = _find_dep_install_docs(project_root)
    artifacts.extend(cicd[:3])
    artifacts.extend(dockerfiles[:3])
    artifacts.extend(dependency_docs[:8])
    omitted.extend(
        {'path': item.path, 'kind': item.kind, 'reason': 'cicd_limit'}
        for item in cicd[3:]
    )
    omitted.extend(
        {'path': item.path, 'kind': item.kind, 'reason': 'dockerfile_limit'}
        for item in dockerfiles[3:]
    )
    omitted.extend(
        {'path': item.path, 'kind': item.kind, 'reason': 'dependency_doc_limit'}
        for item in dependency_docs[8:]
    )

    artifacts.sort(key=lambda a: a.priority)
    deduped = {}
    for artifact in artifacts:
        deduped.setdefault(artifact.path, artifact)
    ranked = list(deduped.values())
    selected = ranked[:24]
    report = {
        'candidate_count': len(ranked),
        'selected_count': len(selected),
        'limits': {'total': 24, 'cicd': 3, 'dockerfile': 3, 'dependency_docs': 8},
        'omitted': omitted + [
            {'path': item.path, 'kind': item.kind, 'reason': 'total_limit'}
            for item in ranked[24:]
        ],
    }
    return (selected, report) if include_report else selected


# ============================================================
# Root docs
# ============================================================

def _find_root_docs(root: str) -> list[DiscoveredArtifact]:
    results = []
    try:
        entries = os.listdir(root)
    except OSError:
        return results

    for name in entries:
        if not os.path.isfile(os.path.join(root, name)):
            continue
        if _ROOT_DOC_DENY_RE.match(name):
            continue
        if _ROOT_DOC_RE.match(name) or _BUILD_DOC_RE.match(name):
            priority = _root_doc_priority(name)
            results.append(DiscoveredArtifact(path=name, kind='root_doc', priority=priority))

    return results


def _root_doc_priority(name: str) -> int:
    lower = name.lower()
    if lower.startswith('install'):
        return 10
    if lower.startswith('building'):
        return 11
    if lower.startswith('build'):
        return 12
    # README last among docs — often has build info but mixed with other content
    return 15


def _find_referenced_build_docs(
    root: str,
    initial_docs: list[DiscoveredArtifact],
    *,
    max_depth: int = 2,
    max_files: int = 16,
) -> list[DiscoveredArtifact]:
    """Follow bounded repository-local build-document references."""
    root_real = os.path.realpath(root)
    queued = [(item.path, 0) for item in initial_docs]
    seen = {item.path for item in initial_docs}
    results: list[DiscoveredArtifact] = []
    while queued and len(results) < max_files:
        rel_path, depth = queued.pop(0)
        try:
            content = open(os.path.join(root, rel_path), encoding='utf-8', errors='ignore').read(128 * 1024)
        except OSError:
            continue
        if depth >= max_depth:
            continue
        for match in _LOCAL_BUILD_DOC_REF_RE.finditer(content):
            candidate = os.path.normpath(match.group(1)).replace('\\', '/')
            if candidate in seen or candidate.startswith('../'):
                continue
            full_path = os.path.realpath(os.path.join(root, candidate))
            if not full_path.startswith(root_real + os.sep) or not os.path.isfile(full_path):
                continue
            seen.add(candidate)
            artifact = DiscoveredArtifact(path=candidate, kind='root_doc', priority=16 + depth)
            results.append(artifact)
            queued.append((candidate, depth + 1))
            if len(results) >= max_files:
                break
    return results


# ============================================================
# debian packaging
# ============================================================

def _find_debian(root: str) -> list[DiscoveredArtifact]:
    results = []
    debian_dir = os.path.join(root, 'debian')
    if not os.path.isdir(debian_dir):
        return results

    control = os.path.join(debian_dir, 'control')
    if os.path.isfile(control):
        results.append(DiscoveredArtifact(path='debian/control', kind='debian_control', priority=1))

    rules = os.path.join(debian_dir, 'rules')
    if os.path.isfile(rules):
        results.append(DiscoveredArtifact(path='debian/rules', kind='debian_rules', priority=2))

    return results


# ============================================================
# CI/CD — select a bounded set of useful workflows
# ============================================================

def _find_cicd(root: str) -> list[DiscoveredArtifact]:
    # Try GitHub Actions first (most common)
    workflows_dir = os.path.join(root, '.github', 'workflows')
    if os.path.isdir(workflows_dir):
        results = _pick_github_workflows(root, workflows_dir)
        if results:
            return results

    # Fallback: .gitlab-ci.yml
    gitlab_ci = os.path.join(root, '.gitlab-ci.yml')
    if os.path.isfile(gitlab_ci):
        return [DiscoveredArtifact(path='.gitlab-ci.yml', kind='cicd', priority=5)]

    # Fallback: .travis.yml
    travis = os.path.join(root, '.travis.yml')
    if os.path.isfile(travis):
        return [DiscoveredArtifact(path='.travis.yml', kind='cicd', priority=6)]

    return []


def _pick_github_workflows(root: str, workflows_dir: str) -> list[DiscoveredArtifact]:
    try:
        all_files = [f for f in os.listdir(workflows_dir)
                     if os.path.isfile(os.path.join(workflows_dir, f))
                     and f.endswith(('.yml', '.yaml'))]
    except OSError:
        return []

    if not all_files:
        return []

    selected: list[DiscoveredArtifact] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        rel = os.path.join('.github', 'workflows', name)
        if rel in seen or not _cicd_has_build_content(os.path.join(root, rel)):
            return
        seen.add(rel)
        selected.append(DiscoveredArtifact(path=rel, kind='cicd', priority=5 + len(selected)))

    # Phase 1: try priority list (exact match)
    lower_map = {f.lower(): f for f in all_files}
    for candidate in _CICD_PRIORITY_LIST:
        if candidate in lower_map:
            add(lower_map[candidate])

    # Phase 2: prefer files with linux/ubuntu in name (non-denied)
    linux_prefer = re.compile(r'(?:linux|ubuntu|unix|debian)', re.IGNORECASE)
    non_denied = [f for f in all_files if not _CICD_DENY_RE.search(f)]
    preferred = [f for f in non_denied if linux_prefer.search(f)]
    for name in sorted(preferred):
        add(name)

    # Phase 3: fallback — any non-denied file with build content
    for name in sorted(non_denied):
        add(name)

    return selected


def _cicd_has_build_content(filepath: str) -> bool:
    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            content = f.read(8192)  # first 8KB is enough
        return bool(_CICD_BUILD_CONTENT_RE.search(content))
    except OSError:
        return False


# ============================================================
# Dockerfile — pick ONE best
# ============================================================

def _find_dockerfile(root: str) -> list[DiscoveredArtifact]:
    try:
        entries = os.listdir(root)
    except OSError:
        return []

    candidates = []
    for name in entries:
        if not os.path.isfile(os.path.join(root, name)):
            continue
        if not _DOCKERFILE_RE.match(name):
            continue
        if _DOCKERFILE_DENY_RE.search(name):
            continue
        candidates.append(name)

    if not candidates:
        return []

    candidates.sort(key=lambda item: (not bool(_DOCKERFILE_PREFER_RE.search(item)), item.lower()))
    return [
        DiscoveredArtifact(path=name, kind='dockerfile', priority=7 + index)
        for index, name in enumerate(candidates)
    ]


# ============================================================
# Dependency install docs (doc/, docs/, sub-INSTALL files)
# ============================================================

# Files in `doc/`, `docs/`, or named `INSTALL-*` / `BUILD-*` sub-documents
# that explicitly mention `git clone` (a strong signal they document how to
# build/install an external source dependency).
_DEP_DOC_DIRS = ('doc', 'docs', 'documentation')
_DEP_DOC_FILE_RE = re.compile(
    r'(?:BUILD|INSTALL|DEPENDENC|REQUIRE|LIBRARIES|LIBS|SETUP)',
    re.IGNORECASE,
)
_GIT_CLONE_RE = re.compile(r'git\s+clone\s+https?://', re.IGNORECASE)


def _find_dep_install_docs(root: str) -> list[DiscoveredArtifact]:
    """Find docs in `doc/` or `docs/` that mention `git clone https://...`.

    These are typically project-specific install instructions for source-built
    dependencies (e.g. FRR's `doc/developer/building-libyang.rst`). Without
    this scan, source_dep_resolver cannot discover the GitHub URL for deps
    that are referenced by configure.ac but not by the root README.
    """
    results: list[DiscoveredArtifact] = []
    project_basename = os.path.basename(os.path.normpath(root)).lower()
    for dirname in _DEP_DOC_DIRS:
        dep_dir = os.path.join(root, dirname)
        if not os.path.isdir(dep_dir):
            continue
        # Walk up to 2 levels deep to catch doc/developer/X.rst
        try:
            for dirpath, _, filenames in os.walk(dep_dir):
                rel_depth = os.path.relpath(dirpath, dep_dir).count(os.sep)
                if rel_depth > 2:
                    continue
                for fname in filenames:
                    if not fname.lower().endswith(('.md', '.rst', '.txt', '.adoc')):
                        continue
                    # Skip files whose name contains the project name itself
                    # (e.g. building-frr-for-centos8.rst) — these are about
                    # building the project, not its deps.
                    if project_basename and project_basename in fname.lower():
                        continue
                    fpath = os.path.join(dirpath, fname)
                    try:
                        with open(fpath, errors='ignore') as f:
                            content = f.read(64 * 1024)  # first 64KB
                    except OSError:
                        continue
                    # Require `git clone` in content — filename alone is not
                    # enough (too many false positives like building-frr-for-X.rst).
                    if not _GIT_CLONE_RE.search(content):
                        continue
                    rel = os.path.relpath(fpath, root)
                    results.append(
                        DiscoveredArtifact(path=rel, kind='root_doc', priority=20)
                    )
        except OSError:
            continue
    # Also scan root-level `INSTALL-*` and `BUILDING-*` files that we may have
    # missed because _ROOT_DOC_RE only matches the bare names.
    try:
        for name in os.listdir(root):
            full = os.path.join(root, name)
            if not os.path.isfile(full):
                continue
            if not _DEP_DOC_FILE_RE.search(name):
                continue
            try:
                with open(full, errors='ignore') as f:
                    content = f.read(64 * 1024)
            except OSError:
                continue
            if _GIT_CLONE_RE.search(content):
                if not any(a.path == name for a in results):
                    results.append(
                        DiscoveredArtifact(path=name, kind='root_doc', priority=18)
                    )
    except OSError:
        pass
    # Dedupe + bounded recall so dependency-specific docs do not crowd out
    # build entries while still covering more than the previous six.
    seen: set[str] = set()
    deduped: list[DiscoveredArtifact] = []
    for a in results:
        if a.path in seen:
            continue
        seen.add(a.path)
        deduped.append(a)
    return deduped
