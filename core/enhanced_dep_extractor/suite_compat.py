"""Extract version constraints from build metadata and rank Debian suites."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

from .artifact_cache import resolve_variables
from .metadata import DISTRO_MAPPING_PATH, SUPPORTED_DEBIAN_VERSIONS, pkglist_path
from .rules import RuleMatch
from .scanner import ProjectScan


_WORD_RE = re.compile(r'"[^"]*"|\'[^\']*\'|\$\{[^}]+\}|\$\([^)]*\)|@[A-Za-z0-9_]+@|[A-Za-z0-9_+./:~<>=!-]+')
_VERSION_RE = re.compile(r'^[0-9][A-Za-z0-9.+:~_-]*$')
_PKGCONFIG_SPEC_RE = re.compile(
    r'(?P<name>[A-Za-z0-9_+.-]+)\s*(?P<op>>=|<=|==|=|>|<)\s*(?P<version>[0-9][A-Za-z0-9.+:~_-]*)'
)
_PYTHON_REQ_RE = re.compile(
    r'(?P<name>[A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*(?P<op>~=|==|!=|>=|<=|>|<)\s*(?P<version>[0-9][A-Za-z0-9.+:~_-]*)'
)
_OCAML_DEP_RE = re.compile(
    r'["\']?(?P<name>[A-Za-z0-9_.+-]+)["\']?\s*\{\s*(?P<op>>=|<=|=|>|<)\s*["\'](?P<version>[^"\']+)["\']\s*\}'
)
_OCAML_FLATTENED_DEP_RE = re.compile(
    r'\bdepends\s*=\s*(?P<name>[A-Za-z0-9_.+-]+)\s*,\s*(?P<version>[0-9][A-Za-z0-9.+:~_-]*)',
    re.IGNORECASE,
)
_MESON_VERSION_RE = re.compile(r'\bversion\s*:\s*(?P<value>\[[^\]]+\]|"[^"]+"|\'[^\']*\')', re.IGNORECASE | re.DOTALL)
_MESON_BARE_VERSION_VAR_RE = re.compile(
    r'(?P<prefix>\bversion\s*:\s*)(?P<var>[A-Za-z_][A-Za-z0-9_]*)\b',
    re.IGNORECASE,
)
_MACRO_VERSION_RE = re.compile(r'\b(?P<name>[A-Z][A-Z0-9_]+)\s*\(\s*\[?(?P<version>[0-9][A-Za-z0-9.+:~_-]*)\]?', re.MULTILINE)
_PKGCONFIG_CARRIER_RULES = frozenset({
    'cmake:pkg_check_modules',
    'cmake:pkg_get_variable',
    'ac:PKG_CHECK_MODULES',
    'ac:PKG_CHECK_EXISTS',
    'ac:PKG_CHECK_VAR',
    'ac:pkg-config',
    'make:pkg-config',
    'configure:pkg-config',
    'bootstrap:pkg-config',
    'qmake:PKGCONFIG',
})
_MACRO_PACKAGE_WHITELIST = frozenset({
    'AM_PATH_PYTHON',
    'GTK_DOC_CHECK',
    'GOBJECT_INTROSPECTION_CHECK',
    'VAPIGEN_CHECK',
})
_SKIP_VERSION_RULE_PREFIXES = (
    'cmake_minimum_required',
    'project',
    'meson_version',
    'AC_PREREQ',
    'PKG_PREREQ',
)
_PKGLIST_CACHE: dict[str, dict] = {}
_DISTRO_MAPPING_CACHE: dict | None = None


@dataclass(frozen=True)
class VersionClause:
    op: str
    version: str


@dataclass(frozen=True)
class ComparePlan:
    op: str
    rhs: str
    supported: bool = True


@dataclass(frozen=True)
class VersionConstraint:
    source_rule: str
    file_path: str
    line_num: int
    dep_system: str
    dep_name: str
    clauses: tuple[VersionClause, ...]
    raw: str
    artifact_kind: str = ''
    packages_hint: tuple[str, ...] = ()
    confidence: str = 'medium'


@dataclass
class ConstraintEvaluation:
    constraint: VersionConstraint
    status: str
    packages: tuple[str, ...] = ()
    matched_package: str = ''
    matched_version: str = ''
    source_repo: str = ''


@dataclass
class SuiteCompatibility:
    suite: str
    hard_fail_count: int = 0
    hard_pass_count: int = 0
    unknown_count: int = 0
    non_main_hits: int = 0
    evaluations: list[ConstraintEvaluation] = field(default_factory=list)


def extract_version_constraints(scan: ProjectScan) -> list[VersionConstraint]:
    constraints: list[VersionConstraint] = []
    seen: set[tuple] = set()
    for matches in scan.file_matches.values():
        for match in matches:
            for constraint in _extract_from_match(match, scan.build_variables):
                key = (
                    constraint.file_path,
                    constraint.line_num,
                    constraint.dep_system,
                    constraint.dep_name.lower(),
                    tuple((clause.op, clause.version) for clause in constraint.clauses),
                )
                if key in seen:
                    continue
                seen.add(key)
                constraints.append(constraint)
    return constraints


def rank_debian_suites(constraints: Iterable[VersionConstraint]) -> list[SuiteCompatibility]:
    constraints = list(constraints)
    results: list[SuiteCompatibility] = []
    for suite in SUPPORTED_DEBIAN_VERSIONS:
        result = SuiteCompatibility(suite=suite)
        for constraint in constraints:
            evaluation = _evaluate_constraint_for_suite(constraint, suite)
            result.evaluations.append(evaluation)
            if evaluation.status == 'hard_pass':
                result.hard_pass_count += 1
                if evaluation.source_repo and not evaluation.source_repo.startswith(f'{suite}/'):
                    result.non_main_hits += 1
            elif evaluation.status == 'hard_fail':
                result.hard_fail_count += 1
            else:
                result.unknown_count += 1
        results.append(result)
    results.sort(key=lambda item: (
        item.hard_fail_count,
        -item.hard_pass_count,
        item.unknown_count,
        item.non_main_hits,
        SUPPORTED_DEBIAN_VERSIONS.index(item.suite),
    ))
    return results


def best_debian_suites(scan: ProjectScan, limit: int = 3) -> list[SuiteCompatibility]:
    constraints = extract_version_constraints(scan)
    return rank_debian_suites(constraints)[:limit]


def _extract_from_match(match: RuleMatch, build_variables: dict[str, str] | None) -> list[VersionConstraint]:
    text = _constraint_text(match, build_variables)
    if not text:
        return []
    match_identity = ' '.join(
        part for part in (match.matched_text, match.extracted_name) if part
    )
    if any(prefix in match_identity for prefix in _SKIP_VERSION_RULE_PREFIXES):
        return []

    constraints: list[VersionConstraint] = []
    if match.rule_type in {'cmake:find_package_call', 'cmake:find_dependency_call'}:
        constraints.extend(_extract_cmake_constraints(match, text, build_variables))
    if match.rule_type == 'meson:dependency':
        constraints.extend(_extract_meson_constraints(match, text))
    if match.rule_type in _PKGCONFIG_CARRIER_RULES or ':dependency-list' in match.rule_type:
        constraints.extend(_extract_pkgconfig_constraints(match, text))
    if match.rule_type.startswith('python:'):
        constraints.extend(_extract_python_constraints(match, text))
    if match.rule_type.startswith('ocaml:'):
        constraints.extend(_extract_ocaml_constraints(match, text))
    if match.rule_type in {'ac:known_macro', 'ac:AM_PATH'}:
        constraints.extend(_extract_macro_constraints(match, text))
    return _attach_package_hints(constraints, match)


def _constraint_text(match: RuleMatch, build_variables: dict[str, str] | None) -> str:
    parts = []
    seen: set[str] = set()
    if match.context:
        stripped_lines = []
        for line in match.context.splitlines():
            if line == '...':
                continue
            stripped_lines.append(re.sub(r'^\d+:\s*', '', line))
        context_text = '\n'.join(stripped_lines).strip()
        if context_text and context_text not in seen:
            seen.add(context_text)
            parts.append(context_text)
    for text in (match.matched_text, match.extracted_name):
        text = (text or '').strip()
        if text and text not in seen:
            seen.add(text)
            parts.append(text)
    text = '\n'.join(part for part in parts if part)
    text = resolve_variables(text, build_variables)
    if match.rule_type.startswith('meson:'):
        text = _resolve_meson_bare_version_variables(text, build_variables)
    return text.strip()


def _resolve_meson_bare_version_variables(text: str, build_variables: dict[str, str] | None) -> str:
    if not build_variables:
        return text

    def replace(match: re.Match) -> str:
        value = build_variables.get(match.group('var'))
        if isinstance(value, (list, tuple)):
            value = next((str(item) for item in reversed(value) if str(item)), '')
        elif value is not None:
            value = str(value)
        if not value:
            return match.group(0)
        escaped = value.replace('\\', '\\\\').replace("'", "\\'")
        return f"{match.group('prefix')}'{escaped}'"

    return _MESON_BARE_VERSION_VAR_RE.sub(replace, text)


def _extract_cmake_constraints(
    match: RuleMatch,
    text: str,
    build_variables: dict[str, str] | None = None,
) -> list[VersionConstraint]:
    # Extract the argument body from the actual find_package / find_dependency
    # call in matched_text, NOT the surrounding context. A greedy `\((.*)\)`
    # over the full text (context + matched_text) crosses statement boundaries
    # and misattributes versions — e.g. an earlier `find_package(Libtorrent
    # REQUIRED)` borrowing Qt5's `5.2.0` from a later call, or a `set(...)` body
    # leaking a version into an unrelated find_package.
    #
    # Variables (${FOO_VERSION}) are resolved against build_variables here so
    # that version predicates supplied via CMake variables are captured.
    body_source = resolve_variables(match.matched_text or text, build_variables)
    body_match = re.search(
        r'(?:find_package|find_dependency)\s*\(([^)]*)\)', body_source, re.IGNORECASE,
    )
    if not body_match:
        return []
    tokens = [_clean_token(token) for token in _WORD_RE.findall(body_match.group(1))]
    tokens = [token for token in tokens if token and not _is_variable_token(token)]
    if not tokens:
        return []

    # Detect NAMES multi-name form: find_package(Foo NAMES foo1 foo2 1.2)
    # In this case the real cmake package names are foo1/foo2, not Foo.
    names_idx = next(
        (i for i, t in enumerate(tokens) if t.lower() == 'names'),
        None,
    )
    if names_idx is not None and names_idx + 1 < len(tokens):
        dep_names: list[str] = []
        for token in tokens[names_idx + 1:]:
            low = token.lower()
            if low in {'required', 'quiet', 'components', 'optional_components',
                       'module', 'config', 'names', 'exact', 'no_default_path',
                       'no_module', 'global', 'no_package_root_path', 'no_cmake_path'}:
                break
            if _VERSION_RE.match(token):
                break
            dep_names.append(token)
        if not dep_names:
            dep_names = [tokens[0]]
    else:
        dep_names = [tokens[0]]

    clauses: list[VersionClause] = []
    confidence = 'medium'
    exact = any(token.lower() == 'exact' for token in tokens[1:])
    for token in tokens[1:]:
        low = token.lower()
        if low in {'required', 'quiet', 'components', 'optional_components',
                   'module', 'config', 'names', 'exact', 'no_default_path',
                   'no_module', 'global', 'no_package_root_path', 'no_cmake_path'}:
            continue
        if '...' in token and token[0].isdigit():
            lower, upper = token.split('...', 1)
            if lower and _VERSION_RE.match(lower):
                clauses.append(VersionClause('>=', lower))
            if upper:
                if upper.startswith('<') and _VERSION_RE.match(upper[1:]):
                    clauses.append(VersionClause('<', upper[1:]))
                elif _VERSION_RE.match(upper):
                    clauses.append(VersionClause('<=', upper))
            confidence = 'medium'
            break
        if _VERSION_RE.match(token):
            clauses.append(VersionClause('=' if exact else '>=', token))
            confidence = 'high' if exact else 'medium'
            break
    if not clauses:
        return []

    # For NAMES form, lower confidence since we're guessing which name resolves
    if names_idx is not None:
        confidence = 'low'

    return [
        VersionConstraint(
            source_rule=match.rule_type,
            file_path=match.file_path,
            line_num=match.line_num,
            dep_system='cmake',
            dep_name=dep_name,
            clauses=tuple(clauses),
            raw=text,
            artifact_kind='cmake',
            confidence=confidence,
        )
        for dep_name in dep_names
    ]


def _extract_meson_constraints(match: RuleMatch, text: str) -> list[VersionConstraint]:
    constraints: list[VersionConstraint] = []
    wanted_name = (match.extracted_name or '').strip().lower()
    for dep_match in re.finditer(
        r'dependency\s*\(\s*(?P<name>\[[^\]]+\]|"[^"]+"|\'[^\']*\')(?P<body>.*?)\)',
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        names = _parse_literal_string_list(dep_match.group('name'))
        if wanted_name:
            names = [name for name in names if name.lower() == wanted_name]
        if not names:
            continue
        call_text = dep_match.group(0)
        version_match = _MESON_VERSION_RE.search(call_text)
        if not version_match:
            continue
        clauses = _parse_clause_value(version_match.group('value'))
        if not clauses:
            continue
        constraints.extend(
            VersionConstraint(
                source_rule=match.rule_type,
                file_path=match.file_path,
                line_num=match.line_num,
                dep_system='meson',
                dep_name=name,
                clauses=tuple(clauses),
                raw=call_text,
                artifact_kind='pkgconfig',
                confidence='high',
            )
            for name in names
        )
    return constraints


def _extract_pkgconfig_constraints(match: RuleMatch, text: str) -> list[VersionConstraint]:
    constraints: list[VersionConstraint] = []
    for spec in _PKGCONFIG_SPEC_RE.finditer(text):
        name = spec.group('name')
        if _skip_pkgconfig_name(name):
            continue
        constraints.append(
            VersionConstraint(
                source_rule=match.rule_type,
                file_path=match.file_path,
                line_num=match.line_num,
                dep_system='pkgconfig',
                dep_name=name,
                clauses=(VersionClause(_normalize_op(spec.group('op')), spec.group('version')),),
                raw=text,
                artifact_kind='pkgconfig',
                confidence='high',
            )
        )
    return constraints


def _extract_python_constraints(match: RuleMatch, text: str) -> list[VersionConstraint]:
    constraints: list[VersionConstraint] = []
    for req in _PYTHON_REQ_RE.finditer(text):
        constraints.append(
            VersionConstraint(
                source_rule=match.rule_type,
                file_path=match.file_path,
                line_num=match.line_num,
                dep_system='python',
                dep_name=req.group('name').replace('_', '-').lower(),
                clauses=(VersionClause(_normalize_op(req.group('op')), req.group('version')),),
                raw=text,
                confidence='high',
            )
        )
    return constraints


def _extract_ocaml_constraints(match: RuleMatch, text: str) -> list[VersionConstraint]:
    constraints: list[VersionConstraint] = []
    for dep in _OCAML_DEP_RE.finditer(text):
        constraints.append(
            VersionConstraint(
                source_rule=match.rule_type,
                file_path=match.file_path,
                line_num=match.line_num,
                dep_system='ocaml',
                dep_name=dep.group('name').lower(),
                clauses=(VersionClause(_normalize_op(dep.group('op')), dep.group('version').strip()),),
                raw=text,
                confidence='high',
            )
        )
    if constraints:
        return constraints
    for dep in _OCAML_FLATTENED_DEP_RE.finditer(text):
        constraints.append(
            VersionConstraint(
                source_rule=match.rule_type,
                file_path=match.file_path,
                line_num=match.line_num,
                dep_system='ocaml',
                dep_name=dep.group('name').lower(),
                clauses=(VersionClause('>=', dep.group('version').strip()),),
                raw=text,
                confidence='medium',
            )
        )
    return constraints


def _extract_macro_constraints(match: RuleMatch, text: str) -> list[VersionConstraint]:
    constraints: list[VersionConstraint] = []
    for macro in _MACRO_VERSION_RE.finditer(text):
        name = macro.group('name')
        if name not in _MACRO_PACKAGE_WHITELIST:
            continue
        constraints.append(
            VersionConstraint(
                source_rule=match.rule_type,
                file_path=match.file_path,
                line_num=match.line_num,
                dep_system='autoconf-macro',
                dep_name=name,
                clauses=(VersionClause('>=', macro.group('version')),),
                raw=text,
                confidence='medium',
            )
        )
    return constraints


def _attach_package_hints(constraints: list[VersionConstraint], match: RuleMatch) -> list[VersionConstraint]:
    packages = tuple(sorted(set(getattr(match, 'packages', ()) or ())))
    if not packages or len(constraints) != 1:
        return constraints
    constraint = constraints[0]
    return [
        VersionConstraint(
            source_rule=constraint.source_rule,
            file_path=constraint.file_path,
            line_num=constraint.line_num,
            dep_system=constraint.dep_system,
            dep_name=constraint.dep_name,
            clauses=constraint.clauses,
            raw=constraint.raw,
            artifact_kind=constraint.artifact_kind,
            packages_hint=packages,
            confidence=constraint.confidence,
        )
    ]


def _evaluate_constraint_for_suite(constraint: VersionConstraint, suite: str) -> ConstraintEvaluation:
    packages, mapping_known = _candidate_packages_for_suite(constraint, suite)
    if not packages:
        return ConstraintEvaluation(
            constraint=constraint,
            status='hard_fail' if mapping_known else 'unknown',
            packages=tuple(packages),
        )
    entries = _package_versions_for_suite(suite)
    best_match: tuple[str, str, str] | None = None
    for package in packages:
        pkg_entry = entries.get(package)
        if not pkg_entry:
            continue
        for version_entry in _coerce_version_entries(pkg_entry.get('version')):
            version = version_entry.get('version', '')
            if version and _clauses_satisfied(version, constraint.clauses):
                source_repo = version_entry.get('source_repo', '')
                candidate = (
                    _source_repo_priority(source_repo, suite),
                    package,
                    version,
                    source_repo,
                )
                if best_match is None or candidate < best_match:
                    best_match = candidate
    if best_match is None:
        return ConstraintEvaluation(constraint=constraint, status='hard_fail', packages=tuple(packages))
    return ConstraintEvaluation(
        constraint=constraint,
        status='hard_pass',
        packages=tuple(packages),
        matched_package=best_match[1],
        matched_version=best_match[2],
        source_repo=best_match[3],
    )


def _candidate_packages_for_suite(constraint: VersionConstraint, suite: str) -> tuple[list[str], bool]:
    if constraint.packages_hint:
        pkglist = _package_versions_for_suite(suite)
        packages = [package for package in constraint.packages_hint if package in pkglist]
        return packages, True
    if not constraint.artifact_kind:
        return [], False
    mapping = _distro_mapping().get(constraint.artifact_kind, {}).get(constraint.dep_name.lower())
    if mapping:
        return list(mapping.get(suite, [])), True
    # Fallback for cmake packages missing from distro_mapping: try naming
    # heuristics (lib<name>-dev / <name>-cmake) so they get a hard_pass/hard_fail
    # verdict instead of "unknown".
    if constraint.artifact_kind == 'cmake':
        dep_lower = constraint.dep_name.lower()
        pkglist = _package_versions_for_suite(suite)
        compact = dep_lower.replace('-', '').replace('_', '')
        candidates = []
        for cand in (
            f'lib{dep_lower}-dev',
            f'lib{compact}-dev',
            f'{dep_lower}-cmake',
            f'{compact}-cmake',
        ):
            if cand in pkglist and cand not in candidates:
                candidates.append(cand)
        if candidates:
            return candidates, True
    return [], False


def _package_versions_for_suite(suite: str) -> dict:
    if suite not in _PKGLIST_CACHE:
        with open(pkglist_path(suite)) as f:
            _PKGLIST_CACHE[suite] = json.load(f)
    return _PKGLIST_CACHE[suite]


def _distro_mapping() -> dict:
    global _DISTRO_MAPPING_CACHE
    if _DISTRO_MAPPING_CACHE is None:
        with open(DISTRO_MAPPING_PATH) as f:
            _DISTRO_MAPPING_CACHE = json.load(f)
    return _DISTRO_MAPPING_CACHE


def _coerce_version_entries(value) -> list[dict]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, str):
        return [{'version': value, 'source_repo': ''}]
    return []


def _clauses_satisfied(version: str, clauses: tuple[VersionClause, ...]) -> bool:
    return all(_compare_versions(version, clause.op, clause.version) for clause in clauses)


@lru_cache(maxsize=8192)
def _compare_versions(lhs: str, op: str, rhs: str) -> bool:
    plan = _comparison_plan(op, rhs)
    if not plan.supported:
        return False
    if plan.op == '!=':
        return not _compare_versions(lhs, '=', rhs)
    result = subprocess.run(
        ['dpkg', '--compare-versions', lhs, plan.op, plan.rhs],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _parse_literal_string_list(value: str) -> list[str]:
    value = value.strip()
    if value.startswith('[') and value.endswith(']'):
        items = re.findall(r'"([^"]+)"|\'([^\']+)\'', value)
        return [left or right for left, right in items if (left or right)]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return [value[1:-1]]
    return []


def _parse_clause_value(value: str) -> list[VersionClause]:
    clauses: list[VersionClause] = []
    for item in _parse_literal_string_list(value):
        spec = item.strip()
        match = re.match(r'^(>=|<=|==|=|>|<|!=|~=)\s*(.+)$', spec)
        if match:
            clauses.append(VersionClause(_normalize_op(match.group(1)), match.group(2).strip()))
        elif _VERSION_RE.match(spec):
            clauses.append(VersionClause('=', spec))
    if clauses:
        return clauses
    spec = value.strip().strip('"\'')
    match = re.match(r'^(>=|<=|==|=|>|<|!=|~=)\s*(.+)$', spec)
    if match:
        return [VersionClause(_normalize_op(match.group(1)), match.group(2).strip())]
    if _VERSION_RE.match(spec):
        return [VersionClause('=', spec)]
    return []


def _skip_pkgconfig_name(name: str) -> bool:
    low = name.lower()
    return (
        name.isupper()
        or name.startswith('-')
        or low in {
            'action-if-found', 'action-if-not-found', 'required', 'quiet', 'modules', 'version',
            'atleast-version', 'exact-version', 'max-version',
        }
        or low.startswith(('${', '$(', '@'))
    )


def _clean_token(token: str) -> str:
    return token.strip().strip('"\'`()[]{}<>,;')


def _is_variable_token(token: str) -> bool:
    return '$' in token or token.startswith('@') or token.endswith('@')


def _normalize_op(op: str) -> str:
    if op == '==':
        return '='
    return op


def _source_repo_priority(source_repo: str, suite: str) -> tuple[int, int, str]:
    if not source_repo:
        return (99, 99, '')
    if '/' in source_repo:
        repo_name, component = source_repo.split('/', 1)
    else:
        repo_name, component = source_repo, 'main'
    if repo_name == suite:
        return (0, 0, component)
    if repo_name == f'{suite}-updates':
        return (1, 0, component)
    if repo_name == f'{suite}-proposed-updates':
        return (2, 0, component)
    if repo_name == f'{suite}-backports':
        return (3, 0, component)
    if repo_name == f'{suite}-backports-sloppy':
        return (4, 0, component)
    if repo_name == f'{suite}-lts':
        return (5, 0, component)
    return (98, 0, source_repo)


def _comparison_plan(op: str, rhs: str) -> ComparePlan:
    normalized_op = _normalize_op(op)
    rhs = rhs.strip()
    if not rhs:
        return ComparePlan(normalized_op, rhs, supported=False)
    if normalized_op == '>':
        return ComparePlan('>>', rhs)
    if normalized_op == '<':
        return ComparePlan('<<', rhs)
    if normalized_op == '~=':
        # PEP 440 compatible-release semantics are not equivalent to Debian
        # comparisons; keep extraction, but make evaluation conservative.
        return ComparePlan(normalized_op, rhs, supported=False)
    if '*' in rhs:
        return ComparePlan(normalized_op, rhs, supported=False)
    return ComparePlan(normalized_op, rhs)
