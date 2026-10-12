"""Non-LLM Debian distro correction from build/version constraints."""

from __future__ import annotations

from dataclasses import dataclass, field

from .metadata import SUPPORTED_DEBIAN_VERSIONS, normalize_debian_version
from .scanner import ProjectScan
from . import suite_compat
from . import sysuse_select


@dataclass(frozen=True)
class DistroDecision:
    baseline_debian_version: str
    selected_debian_version: str
    selection_source: str
    constraints: tuple[str, ...] = ()
    satisfied_suites: tuple[str, ...] = ()
    unknown_constraints: tuple[str, ...] = ()
    constraint_details: tuple[dict, ...] = ()
    unknown_constraint_details: tuple[dict, ...] = ()
    minimum_satisfying_sysuse: str = ''
    next_satisfying_sysuse_candidates: tuple[dict, ...] = ()

    def as_metadata(self) -> dict:
        return {
            'baseline_debian_version': self.baseline_debian_version,
            'selected_debian_version': self.selected_debian_version,
            'selection_source': self.selection_source,
            'constraints': list(self.constraints),
            'satisfied_suites': list(self.satisfied_suites),
            'unknown_constraints': list(self.unknown_constraints),
            'minimum_satisfying_sysuse': self.minimum_satisfying_sysuse,
            'next_satisfying_sysuse_candidates': list(self.next_satisfying_sysuse_candidates),
        }

    def as_constraint_details(self) -> dict:
        return {
            'baseline_debian_version': self.baseline_debian_version,
            'selected_debian_version': self.selected_debian_version,
            'selection_source': self.selection_source,
            'constraints': list(self.constraints),
            'constraint_details': list(self.constraint_details),
            'unknown_constraints': list(self.unknown_constraints),
            'unknown_constraint_details': list(self.unknown_constraint_details),
            'satisfied_suites': list(self.satisfied_suites),
            'minimum_satisfying_sysuse': self.minimum_satisfying_sysuse,
            'next_satisfying_sysuse_candidates': list(self.next_satisfying_sysuse_candidates),
        }


@dataclass
class _SuiteStatus:
    hard_fail_count: int = 0
    hard_pass_count: int = 0
    unknown_count: int = 0
    notes: list[str] = field(default_factory=list)


def fixed_decision(baseline_debian_version: str) -> DistroDecision:
    baseline = normalize_debian_version(baseline_debian_version)
    return DistroDecision(
        baseline_debian_version=baseline,
        selected_debian_version=baseline,
        selection_source='fixed',
    )


def select_debian_version(
    project_path: str,
    scan: ProjectScan,
    baseline_debian_version: str,
) -> DistroDecision:
    """Validate/correct commit-date baseline using hard non-LLM constraints."""
    baseline = normalize_debian_version(baseline_debian_version)
    statuses = {suite: _SuiteStatus() for suite in SUPPORTED_DEBIAN_VERSIONS}
    constraints: list[str] = []
    unknown_constraints: list[str] = []
    constraint_details: list[dict] = []
    unknown_constraint_details: list[dict] = []

    _apply_toolchain_constraints(project_path, scan, statuses, constraints, constraint_details)
    _apply_package_constraints(scan, statuses, constraints, unknown_constraints, constraint_details, unknown_constraint_details)

    baseline_status = statuses[baseline]
    if baseline_status.hard_fail_count == 0:
        source = 'baseline_no_constraints' if not constraints else 'baseline_satisfies_constraints'
        satisfied = _satisfied_suites(statuses)
        minimum = _minimum_satisfying_sysuse(constraints, satisfied)
        return DistroDecision(
            baseline_debian_version=baseline,
            selected_debian_version=baseline,
            selection_source=source,
            constraints=tuple(constraints),
            satisfied_suites=satisfied,
            unknown_constraints=tuple(unknown_constraints),
            constraint_details=tuple(constraint_details),
            unknown_constraint_details=tuple(unknown_constraint_details),
            minimum_satisfying_sysuse=minimum,
            next_satisfying_sysuse_candidates=tuple(
                _next_satisfying_sysuse_candidates(minimum, satisfied, constraint_details)
            ),
        )

    satisfying = _satisfied_suites(statuses, require_pass=True)
    if not satisfying:
        return DistroDecision(
            baseline_debian_version=baseline,
            selected_debian_version=baseline,
            selection_source='baseline_after_unsatisfied_constraints',
            constraints=tuple(constraints),
            satisfied_suites=(),
            unknown_constraints=tuple(unknown_constraints),
            constraint_details=tuple(constraint_details),
            unknown_constraint_details=tuple(unknown_constraint_details),
        )

    selected = _nearest_suite(baseline, satisfying)
    baseline_idx = SUPPORTED_DEBIAN_VERSIONS.index(baseline)
    selected_idx = SUPPORTED_DEBIAN_VERSIONS.index(selected)
    if selected_idx > baseline_idx:
        source = 'constraints_upgrade'
    elif selected_idx < baseline_idx:
        source = 'constraints_downgrade'
    else:
        source = 'baseline_satisfies_constraints'
    minimum = _minimum_satisfying_sysuse(constraints, satisfying)
    return DistroDecision(
        baseline_debian_version=baseline,
        selected_debian_version=selected,
        selection_source=source,
        constraints=tuple(constraints),
        satisfied_suites=satisfying,
        unknown_constraints=tuple(unknown_constraints),
        constraint_details=tuple(constraint_details),
        unknown_constraint_details=tuple(unknown_constraint_details),
        minimum_satisfying_sysuse=minimum,
        next_satisfying_sysuse_candidates=tuple(
            _next_satisfying_sysuse_candidates(minimum, satisfying, constraint_details)
        ),
    )


def _apply_toolchain_constraints(
    project_path: str,
    scan: ProjectScan,
    statuses: dict[str, _SuiteStatus],
    constraints: list[str],
    constraint_details: list[dict],
) -> None:
    signals = scan.version_signals or sysuse_select.VersionSignals()
    decision = sysuse_select.decide_sysuse(signals)
    if not decision.codename:
        return

    min_suite = normalize_debian_version(decision.codename)
    min_idx = SUPPORTED_DEBIAN_VERSIONS.index(min_suite)
    for note in decision.constraints:
        label = f'toolchain: {note}'
        constraints.append(label)
        constraint_details.append({
            'kind': 'toolchain',
            'label': label,
            'dep_system': 'toolchain',
            'dep_name': 'toolchain',
            'clauses': [],
            'file_path': '',
            'line_num': 0,
            'source_rule': 'version_signals',
            'raw': note,
            'artifact_kind': '',
            'packages_hint': [],
            'confidence': 'medium',
        })
    for suite in SUPPORTED_DEBIAN_VERSIONS:
        status = statuses[suite]
        if SUPPORTED_DEBIAN_VERSIONS.index(suite) < min_idx:
            status.hard_fail_count += 1
        else:
            status.hard_pass_count += 1


def _apply_package_constraints(
    scan: ProjectScan,
    statuses: dict[str, _SuiteStatus],
    constraints: list[str],
    unknown_constraints: list[str],
    constraint_details: list[dict],
    unknown_constraint_details: list[dict],
) -> None:
    for constraint in suite_compat.extract_version_constraints(scan):
        label = _constraint_label(constraint)
        constraints.append(label)
        detail = _constraint_detail(constraint, label)
        constraint_details.append(detail)
        suite_versions = {}
        any_known = False
        for suite in SUPPORTED_DEBIAN_VERSIONS:
            evaluation = suite_compat._evaluate_constraint_for_suite(constraint, suite)
            status = statuses[suite]
            if evaluation.status == 'hard_fail':
                status.hard_fail_count += 1
                any_known = True
            elif evaluation.status == 'hard_pass':
                status.hard_pass_count += 1
                any_known = True
                suite_versions[suite] = {
                    'package': getattr(evaluation, 'matched_package', ''),
                    'version': getattr(evaluation, 'matched_version', ''),
                    'source_repo': getattr(evaluation, 'source_repo', ''),
                }
            else:
                status.unknown_count += 1
        if suite_versions:
            detail['suite_versions'] = suite_versions
        if not any_known:
            unknown_constraints.append(label)
            unknown_constraint_details.append(detail)


def _constraint_label(constraint: suite_compat.VersionConstraint) -> str:
    clauses = ' '.join(f'{clause.op} {clause.version}' for clause in constraint.clauses)
    location = f'{constraint.file_path}:{constraint.line_num}'
    return f'{constraint.dep_system}: {constraint.dep_name} {clauses} ({location})'


def _constraint_detail(constraint: suite_compat.VersionConstraint, label: str) -> dict:
    return {
        'kind': 'package',
        'label': label,
        'dep_system': constraint.dep_system,
        'dep_name': constraint.dep_name,
        'clauses': [
            {'op': clause.op, 'version': clause.version}
            for clause in constraint.clauses
        ],
        'file_path': constraint.file_path,
        'line_num': constraint.line_num,
        'source_rule': constraint.source_rule,
        'raw': constraint.raw,
        'artifact_kind': constraint.artifact_kind,
        'packages_hint': list(constraint.packages_hint),
        'confidence': constraint.confidence,
    }


def _satisfied_suites(
    statuses: dict[str, _SuiteStatus],
    require_pass: bool = False,
) -> tuple[str, ...]:
    suites = []
    for suite in SUPPORTED_DEBIAN_VERSIONS:
        status = statuses[suite]
        if status.hard_fail_count == 0 and (not require_pass or status.hard_pass_count > 0):
            suites.append(suite)
    return tuple(suites)


def _minimum_satisfying_sysuse(
    constraints: list[str],
    satisfied_suites: tuple[str, ...],
) -> str:
    if not constraints or not satisfied_suites:
        return ''
    return satisfied_suites[0]


def _next_satisfying_sysuse_candidates(
    minimum: str,
    satisfied_suites: tuple[str, ...],
    constraint_details: list[dict],
    limit: int = 3,
) -> list[dict]:
    if not minimum or minimum not in satisfied_suites:
        return []
    min_idx = SUPPORTED_DEBIAN_VERSIONS.index(minimum)
    candidates = [
        suite for suite in satisfied_suites
        if SUPPORTED_DEBIAN_VERSIONS.index(suite) > min_idx
    ][:limit]
    return [
        {
            'sysuse': suite,
            'package_version_differences': _package_version_differences(
                minimum,
                suite,
                constraint_details,
            ),
        }
        for suite in candidates
    ]


def _package_version_differences(
    minimum_suite: str,
    candidate_suite: str,
    constraint_details: list[dict],
    limit: int = 8,
) -> list[dict]:
    differences: list[dict] = []
    for detail in constraint_details:
        if detail.get('kind') != 'package':
            continue
        versions = detail.get('suite_versions') or {}
        minimum = versions.get(minimum_suite) or {}
        candidate = versions.get(candidate_suite) or {}
        if not candidate:
            continue
        if minimum and minimum.get('version') == candidate.get('version'):
            continue
        differences.append({
            'dep_system': detail.get('dep_system', ''),
            'dep_name': detail.get('dep_name', ''),
            'package': candidate.get('package', ''),
            'minimum_sysuse': minimum_suite,
            'minimum_version': minimum.get('version', ''),
            'candidate_sysuse': candidate_suite,
            'candidate_version': candidate.get('version', ''),
            'source_repo': candidate.get('source_repo', ''),
        })
        if len(differences) >= limit:
            break
    return differences


def _nearest_suite(baseline: str, candidates: tuple[str, ...]) -> str:
    baseline_idx = SUPPORTED_DEBIAN_VERSIONS.index(baseline)
    return min(
        candidates,
        key=lambda suite: (
            abs(SUPPORTED_DEBIAN_VERSIONS.index(suite) - baseline_idx),
            -SUPPORTED_DEBIAN_VERSIONS.index(suite),
        ),
    )
