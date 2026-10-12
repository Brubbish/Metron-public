#!/usr/bin/env python3
"""Evaluate enhanced_dep_extractor results against ground truth.

Usage:
    python3 enhanced_dep_extractor/eval.py --input results.json [--gt path]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    from .metadata import DEFAULT_DEBIAN_VERSION, normalize_debian_version, pkginfo_glob, pkglist_path
except ImportError:
    from metadata import DEFAULT_DEBIAN_VERSION, normalize_debian_version, pkginfo_glob, pkglist_path

GT_PATH = Path(__file__).resolve().parent.parent / 'apt_download/final_deps_1000.json'
PKGINFO_DIR = Path(os.environ.get('METRON_PKGINFO_DIR', 'Build_RAG_Info/Download'))
NESTED_CLOSURE_DEPTH = 10

DEBIAN_PACKAGING_HELPERS = frozenset({
    'architecture-is-64-bit',
    'architecture-is-little-endian',
    'architecture-properties',
    'debhelper',
    'debhelper-compat',
    'debputy',
    'dh-debputy',
    'dh-sequence-bash-completion',
    'dh-sequence-gir',
    'dh-sequence-installsysusers',
    'dh-sequence-kf5',
    'dh-sequence-movetousr',
    'dh-sequence-python3',
    'dh-sequence-sphinxdoc',
    'dh-sequence-systemd',
    'dh-systemd',
    'dh-exec',
    'dh-autoreconf',
    'dpkg-build-api',
    'dpkg-dev',
    'pkg-kde-tools',
    'quilt',
})

TEST_ENVIRONMENT_HELPERS = frozenset({
    'xauth',
    'xvfb',
    'xvfb-run',
})

PROVIDER_ALIAS_FAMILIES = (
    frozenset({'architecture-is-64-bit', 'architecture-is-little-endian', 'architecture-properties'}),
    frozenset({'googletest', 'libgtest-dev', 'libgmock-dev'}),
    frozenset({'catch', 'catch2', 'libcatch2-dev'}),
    frozenset({'libjack-dev', 'libjack-jackd2-dev', 'jackd2'}),
    frozenset({'libcurl4-openssl-dev', 'libcurl4-gnutls-dev', 'libcurl4-nss-dev'}),
    frozenset({'pkg-config', 'pkgconf'}),
    frozenset({'systemd-dev', 'libsystemd-dev', 'systemd'}),
    frozenset({'libncurses-dev', 'libncurses5-dev', 'libncursesw5-dev', 'libtinfo-dev'}),
    frozenset({'zlib1g-dev', 'libz-dev'}),
    frozenset({'libpam-dev', 'libpam0g-dev'}),
    frozenset({'gnutls-dev', 'libgnutls28-dev', 'libgnutls-openssl-dev'}),
    frozenset({'libcairo-dev', 'libcairo2-dev'}),
    frozenset({'libfreetype6-dev', 'libfreetype-dev'}),
    frozenset({'libgcrypt-dev', 'libgcrypt20-dev'}),
    frozenset({'libglu-dev', 'libglu1-mesa-dev'}),
    frozenset({'libgpgme11-dev', 'libgpgme-dev'}),
    frozenset({'libltdl3-dev', 'libltdl-dev'}),
    frozenset({'libsamplerate-dev', 'libsamplerate0-dev'}),
    frozenset({'libsndfile-dev', 'libsndfile1-dev'}),
    frozenset({'libsoapysdr0.8-dev', 'libsoapysdr-dev'}),
    frozenset({'opencl-dev', 'ocl-icd-opencl-dev'}),
    frozenset({'git-core', 'git'}),
    frozenset({'libbfd-dev', 'binutils-dev'}),
    frozenset({'liblcms-dev', 'liblcms2-dev'}),
    frozenset({'liblua5.1-dev', 'liblua5.1-0-dev'}),
    frozenset({'libpinyin-dev', 'libpinyin15-dev'}),
    frozenset({'libqt5opengl5-desktop-dev', 'libqt5opengl5-dev'}),
    frozenset({'tex4ht', 'texlive-plain-generic'}),
    frozenset({'transfig', 'fig2dev'}),
    frozenset({'libjpeg-dev', 'libjpeg62-turbo-dev'}),
    frozenset({'libldap2-dev', 'libldap-dev'}),
    frozenset({'libsdl2-dev', 'libsdl2-compat-dev'}),
    frozenset({'libunwind-dev', 'libunwind-19-dev'}),
    frozenset({'x11proto-core-dev', 'x11proto-dev'}),
    frozenset({'x11proto-scrnsaver-dev', 'x11proto-dev'}),
    frozenset({'x11proto-xinerama-dev', 'x11proto-dev'}),
    frozenset({'qt5keychain-dev', 'qtkeychain-qt5-dev'}),
    frozenset({'postgresql-client', 'postgresql-client-17'}),
)


def clean_pkg(value) -> str:
    if not isinstance(value, str):
        return ''
    return re.sub(r'\s+', '', value.strip())


def sorted_pkgs(values) -> list[str]:
    if not isinstance(values, (list, set, tuple)):
        return []
    return sorted({pkg for value in values if (pkg := clean_pkg(value))})


def parse_provides_expr(expr: str) -> list[str]:
    names = []
    for part in expr.split(','):
        match = re.match(r'\s*([a-z0-9][a-z0-9.+-]*)', part)
        if match:
            names.append(match.group(1))
    return names


def load_binary_package_metadata(packages_path: Path | None = None, debian_version: str = DEFAULT_DEBIAN_VERSION):
    """Load binary packages and Provides relationships from a Packages file."""
    real_packages: set[str] = set()
    provides: dict[str, set[str]] = {}
    paths = [packages_path] if packages_path else sorted(PKGINFO_DIR.glob(pkginfo_glob(debian_version)))
    for path in paths:
        if not path or not path.exists():
            continue
        for block in path.read_text(errors='replace').split('\n\n'):
            pkg_match = re.search(r'^Package:\s*(\S+)', block, re.MULTILINE)
            if not pkg_match:
                continue
            pkg = pkg_match.group(1)
            real_packages.add(pkg)

            provides_match = re.search(
                r'^Provides:\s*(.*?)(?:\n[A-Z][A-Za-z0-9-]*:|\Z)',
                block,
                re.MULTILINE | re.DOTALL,
            )
            if not provides_match:
                continue
            provides_text = ' '.join(line.strip() for line in provides_match.group(1).splitlines())
            for virtual in parse_provides_expr(provides_text):
                provides.setdefault(virtual, set()).add(pkg)
    return real_packages, provides


def build_alias_map(provides: dict[str, set[str]] | None = None) -> dict[str, set[str]]:
    """Build package-equivalence groups used only for evaluation matching."""
    alias_map: dict[str, set[str]] = {}

    def add_family(family):
        family = {pkg for pkg in family if pkg}
        for pkg in family:
            alias_map.setdefault(pkg, set()).update(family)

    for family in PROVIDER_ALIAS_FAMILIES:
        add_family(family)
    if provides:
        for virtual, providers in provides.items():
            add_family({virtual, *providers})
    for pkg, aliases in list(alias_map.items()):
        aliases.add(pkg)
    return alias_map


def expand_with_aliases(pkgs: set[str], alias_map: dict[str, set[str]]) -> set[str]:
    expanded = set(pkgs)
    for pkg in pkgs:
        expanded.update(alias_map.get(pkg, ()))
    return expanded


def apply_exclusions(
    gt_by_project: dict[str, set[str]],
    pred_by_project: dict[str, set[str]],
    exclude_debian_helpers: bool = False,
    exclude_test_helpers: bool = False,
) -> tuple[dict[str, set[str]], dict[str, set[str]], set[str]]:
    excluded: set[str] = set()
    if exclude_debian_helpers:
        excluded.update(DEBIAN_PACKAGING_HELPERS)
    if exclude_test_helpers:
        excluded.update(TEST_ENVIRONMENT_HELPERS)
    if not excluded:
        return gt_by_project, pred_by_project, excluded
    gt_filtered = {project: (pkgs - excluded) for project, pkgs in gt_by_project.items()}
    pred_filtered = {project: (pkgs - excluded) for project, pkgs in pred_by_project.items()}
    return gt_filtered, pred_filtered, excluded


def parse_depends_expr(expr: str) -> list[str]:
    if not expr:
        return []
    parts, buf, depth = [], [], 0
    for ch in expr:
        if ch == '(':
            depth += 1
        elif ch == ')' and depth:
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(''.join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append(''.join(buf))

    deps = []
    for part in parts:
        for alt in part.split('|'):
            alt = re.sub(r'\([^)]*\)', '', alt)
            alt = re.sub(r'\[[^]]*\]', '', alt)
            alt = re.sub(r'<[^>]*>', '', alt)
            alt = alt.strip()
            if ':' in alt:
                alt = alt.split(':', 1)[0]
            pkg = clean_pkg(alt)
            if pkg and not pkg.startswith('<'):
                deps.append(pkg)
                break
    return deps


def load_dependency_graph(valid_packages: set[str], debian_version: str = DEFAULT_DEBIAN_VERSION) -> dict[str, list[str]]:
    field_re = re.compile(r'^(\S[^:]*):\s*(.*)$')
    deps_by_pkg: dict[str, list[str]] = {}

    for path in sorted(PKGINFO_DIR.glob(pkginfo_glob(debian_version))):
        if not path.exists() or path.stat().st_size == 0:
            continue
        current: dict[str, str] = {}
        field = None
        for line in path.read_text(errors='replace').splitlines():
            if not line:
                pkg = current.get('Package')
                if pkg:
                    expr = ' '.join(
                        p for p in (current.get('Pre-Depends', ''), current.get('Depends', '')) if p
                    )
                    deps_by_pkg[pkg] = [d for d in parse_depends_expr(expr) if d in valid_packages]
                current = {}
                field = None
                continue
            if line.startswith(' ') and field:
                current[field] = current.get(field, '') + ' ' + line.strip()
                continue
            match = field_re.match(line)
            if match:
                field = match.group(1)
                current[field] = match.group(2).strip()
        pkg = current.get('Package')
        if pkg:
            expr = ' '.join(
                p for p in (current.get('Pre-Depends', ''), current.get('Depends', '')) if p
            )
            deps_by_pkg[pkg] = [d for d in parse_depends_expr(expr) if d in valid_packages]

    return deps_by_pkg


def load_gt(gt_path: Path) -> dict[str, set[str]]:
    data = json.loads(gt_path.read_text())
    return {
        project: set(sorted_pkgs(entry.get('dependencies', [])))
        for project, entry in data.items()
        if isinstance(entry, dict)
    }


def load_predictions(input_path: Path) -> dict[str, set[str]]:
    data = json.loads(input_path.read_text())
    pred_by_project: dict[str, set[str]] = {}
    for project_path, files in data.get('per_project', {}).items():
        project = Path(project_path).name
        pkgs: set[str] = set()
        if isinstance(files, dict):
            for deps in files.values():
                if not isinstance(deps, list):
                    continue
                for dep in deps:
                    if isinstance(dep, dict):
                        pkgs.update(sorted_pkgs(dep.get('suite_packages', [])))
        pred_by_project[project] = pkgs
    return pred_by_project


def evaluate(
    gt_by_project: dict[str, set[str]],
    pred_by_project: dict[str, set[str]],
    dep_closure=None,
    alias_map: dict[str, set[str]] | None = None,
    label: str = 'Enhanced Rule+LLM',
    excluded_packages: set[str] | None = None,
):
    rows = []
    alias_map = alias_map or {}
    excluded_packages = excluded_packages or set()
    for project in sorted(gt_by_project.keys()):
        gt_pkgs = gt_by_project.get(project, set())
        if not gt_pkgs:
            continue
        if project not in pred_by_project:
            continue

        pred_pkgs = pred_by_project.get(project, set())
        gt_expanded = {pkg: expand_with_aliases({pkg}, alias_map) for pkg in gt_pkgs}
        pred_expanded = {pkg: expand_with_aliases({pkg}, alias_map) for pkg in pred_pkgs}
        hit_gt = {
            gt_pkg for gt_pkg, aliases in gt_expanded.items()
            if aliases & pred_pkgs or any(aliases & pred_aliases for pred_aliases in pred_expanded.values())
        }
        hit_pred = {
            pred_pkg for pred_pkg, aliases in pred_expanded.items()
            if aliases & gt_pkgs or any(aliases & gt_aliases for gt_aliases in gt_expanded.values())
        }
        exact_hit = gt_pkgs & pred_pkgs
        alias_hit_gt = hit_gt - exact_hit
        hit = hit_gt
        fp_pkgs = pred_pkgs - hit_pred
        fn_pkgs = gt_pkgs - hit_gt

        nested_gt_covered = 0
        nested_pred_redundant = 0
        if dep_closure:
            for gt_pkg in fn_pkgs:
                gt_aliases = expand_with_aliases({gt_pkg}, alias_map)
                if any(gt_aliases & dep_closure(p) for p in pred_pkgs):
                    nested_gt_covered += 1
            for pred_pkg in fp_pkgs:
                pred_aliases = expand_with_aliases({pred_pkg}, alias_map)
                if any(pred_aliases & dep_closure(g) for g in gt_pkgs):
                    nested_pred_redundant += 1

        recall = len(hit) / len(gt_pkgs) if gt_pkgs else 1.0
        precision = len(hit_pred) / len(pred_pkgs) if pred_pkgs else (1.0 if not gt_pkgs else 0.0)
        nested_recall = (len(hit) + nested_gt_covered) / len(gt_pkgs) if gt_pkgs else 1.0
        nested_precision = (
            (len(hit_pred) + nested_pred_redundant) / len(pred_pkgs) if pred_pkgs
            else (1.0 if not gt_pkgs else 0.0)
        )

        rows.append({
            'project': project,
            'gt': len(gt_pkgs),
            'pred': len(pred_pkgs),
            'tp': len(hit),
            'pred_tp': len(hit_pred),
            'alias_tp': len(alias_hit_gt),
            'fp': len(fp_pkgs),
            'fn': len(fn_pkgs),
            'recall': recall,
            'precision': precision,
            'nested_gt_covered': nested_gt_covered,
            'nested_pred_redundant': nested_pred_redundant,
            'nested_adjusted_recall': nested_recall,
            'nested_adjusted_precision': nested_precision,
        })

    if not rows:
        print('No projects evaluated.')
        return

    n = len(rows)
    total_gt = sum(r['gt'] for r in rows)
    total_pred = sum(r['pred'] for r in rows)
    tp = sum(r['tp'] for r in rows)
    pred_tp = sum(r['pred_tp'] for r in rows)
    alias_tp = sum(r['alias_tp'] for r in rows)
    nested_gt = sum(r['nested_gt_covered'] for r in rows)
    nested_pred = sum(r['nested_pred_redundant'] for r in rows)

    macro_recall = sum(r['recall'] for r in rows) / n
    macro_precision = sum(r['precision'] for r in rows) / n
    micro_recall = tp / total_gt if total_gt else 0
    micro_precision = pred_tp / total_pred if total_pred else 0

    macro_nested_recall = sum(r['nested_adjusted_recall'] for r in rows) / n
    macro_nested_precision = sum(r['nested_adjusted_precision'] for r in rows) / n
    micro_nested_recall = (tp + nested_gt) / total_gt if total_gt else 0
    micro_nested_precision = (pred_tp + nested_pred) / total_pred if total_pred else 0

    print(f'╔══════════════════════════════════════════════════╗')
    print(f'║  Enhanced Rule+LLM Extractor Evaluation         ║')
    print(f'╠══════════════════════════════════════════════════╣')
    print(f'║  Projects evaluated:     {n:>6}                  ║')
    print(f'║  Total GT deps:          {total_gt:>6}                  ║')
    print(f'║  Total predicted deps:   {total_pred:>6}                  ║')
    print(f'║  True positives:         {tp:>6}                  ║')
    print(f'║  Alias positives:        {alias_tp:>6}                  ║')
    print(f'╠══════════════════════════════════════════════════╣')
    print(f'║  Macro recall:           {macro_recall*100:>6.2f}%                 ║')
    print(f'║  Macro precision:        {macro_precision*100:>6.2f}%                 ║')
    print(f'║  Micro recall:           {micro_recall*100:>6.2f}%                 ║')
    print(f'║  Micro precision:        {micro_precision*100:>6.2f}%                 ║')
    print(f'╠══════════════════════════════════════════════════╣')
    print(f'║  Nested macro recall:    {macro_nested_recall*100:>6.2f}%                 ║')
    print(f'║  Nested macro precision: {macro_nested_precision*100:>6.2f}%                 ║')
    print(f'║  Nested micro recall:    {micro_nested_recall*100:>6.2f}%                 ║')
    print(f'║  Nested micro precision: {micro_nested_precision*100:>6.2f}%                 ║')
    print(f'╠══════════════════════════════════════════════════╣')
    print(f'║  100% recall projects:   {sum(1 for r in rows if r["recall"] >= 1.0):>6}                  ║')
    print(f'║  0% recall projects:     {sum(1 for r in rows if r["recall"] == 0):>6}                  ║')
    print(f'║  Empty predictions:      {sum(1 for r in rows if r["pred"] == 0):>6}                  ║')
    print(f'╚══════════════════════════════════════════════════╝')

    if alias_map or excluded_packages:
        print('\nEvaluation adjustments:')
        if alias_map:
            print('  - Provider/virtual aliases enabled')
        if excluded_packages:
            print(f'  - Excluded packages: {len(excluded_packages)}')

    # Recall distribution
    print('\nRecall distribution:')
    bins = [(0, 0, '0%'), (0, 0.2, '(0,20%)'), (0.2, 0.4, '[20,40%)'),
            (0.4, 0.6, '[40,60%)'), (0.6, 0.8, '[60,80%)'),
            (0.8, 1.0, '[80,100%)'), (1.0, 1.01, '100%')]
    for lo, hi, bin_label in bins:
        if lo == 0 and hi == 0:
            cnt = sum(1 for r in rows if r['recall'] == 0)
        elif lo == 1.0:
            cnt = sum(1 for r in rows if r['recall'] >= 1.0)
        elif lo == 0:
            cnt = sum(1 for r in rows if 0 < r['recall'] < hi)
        else:
            cnt = sum(1 for r in rows if lo <= r['recall'] < hi)
        bar = '█' * (cnt // 2)
        print(f'  {bin_label:>10s}: {cnt:>4} {bar}')

    # Comparison table
    print('\nComparison with other methods (macro, KB-verified):')
    print(f'  {"Method":<40s} {"Recall":>8s} {"Precision":>10s}')
    print(f'  {"-"*40} {"-"*8} {"-"*10}')
    methods = [
        ('CCScanner+LLM(KB)', 43.42, 35.01),
        ('regex+LLM(KB)', 58.84, 41.51),
        ('LLM直接读(KB,384)', 66.52, 38.37),
        ('CCScanner+regex+LLM(KB) union', 72.79, 41.01),
        ('readoc385(原始)', 72.46, 45.73),
        (f'** {label} **', macro_recall*100, macro_precision*100),
    ]
    for name, r, p in methods:
        marker = '  >>> ' if '**' in name else '      '
        print(f'{marker}{name:<40s} {r:>7.2f}% {p:>9.2f}%')

    return {
        'projects': n,
        'macro_recall': macro_recall,
        'macro_precision': macro_precision,
        'micro_recall': micro_recall,
        'micro_precision': micro_precision,
        'macro_nested_recall': macro_nested_recall,
        'macro_nested_precision': macro_nested_precision,
        'micro_nested_recall': micro_nested_recall,
        'micro_nested_precision': micro_nested_precision,
        'alias_true_positives': alias_tp,
        'excluded_packages': sorted(excluded_packages),
        'per_project': rows,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate enhanced extractor results')
    parser.add_argument('--input', required=True, help='Results JSON from main.py or verify_and_fix.py')
    parser.add_argument('--gt', default=str(GT_PATH), help='Ground truth JSON')
    parser.add_argument('--debian-version', default=DEFAULT_DEBIAN_VERSION, help='Debian metadata version to use')
    parser.add_argument('--packages', default=None, help='Optional Packages/pkginfo file for Provides aliases')
    parser.add_argument('--no-provider-aliases', action='store_true', help='Disable Provides/provider-family matching')
    parser.add_argument('--exclude-debian-helpers', action='store_true', help='Exclude Debian packaging helper/constraint packages from GT and predictions')
    parser.add_argument('--exclude-test-helpers', action='store_true', help='Exclude test-environment helper packages such as xauth/xvfb')
    parser.add_argument('--no-nested', action='store_true', help='Skip nested dep analysis')
    parser.add_argument('--save', default=None, help='Save detailed results to JSON')
    args = parser.parse_args()
    debian_version = normalize_debian_version(args.debian_version)

    gt_by_project = load_gt(Path(args.gt))
    pred_by_project = load_predictions(Path(args.input))
    gt_by_project, pred_by_project, excluded_packages = apply_exclusions(
        gt_by_project,
        pred_by_project,
        exclude_debian_helpers=args.exclude_debian_helpers,
        exclude_test_helpers=args.exclude_test_helpers,
    )

    alias_map = {}
    if not args.no_provider_aliases:
        try:
            real_packages, provides = load_binary_package_metadata(
                Path(args.packages) if args.packages else None,
                debian_version=debian_version,
            )
            alias_map = build_alias_map(provides)
            print(
                f'Loaded package metadata: {len(real_packages)} real packages, '
                f'{len(provides)} provided names',
                file=sys.stderr,
            )
        except Exception as e:
            print(f'Warning: could not load package Provides aliases: {e}', file=sys.stderr)
            alias_map = build_alias_map()

    print(f'GT projects: {len(gt_by_project)}', file=sys.stderr)
    print(f'Predicted projects: {len(pred_by_project)}', file=sys.stderr)
    overlap = set(gt_by_project.keys()) & set(pred_by_project.keys())
    print(f'Overlap: {len(overlap)}', file=sys.stderr)

    dep_closure_fn = None
    if not args.no_nested:
        try:
            valid_packages = set(json.loads(pkglist_path(debian_version).read_text()).keys())
            dependency_graph = load_dependency_graph(valid_packages, debian_version=debian_version)
            from functools import lru_cache

            @lru_cache(maxsize=None)
            def dep_closure(pkg: str) -> frozenset[str]:
                seen: set[str] = set()
                def rec(current: str, depth: int):
                    if depth >= NESTED_CLOSURE_DEPTH:
                        return
                    for dep in dependency_graph.get(current, []):
                        if dep not in seen:
                            seen.add(dep)
                            rec(dep, depth + 1)
                rec(pkg, 0)
                return frozenset(seen)

            dep_closure_fn = dep_closure
            print(f'Loaded {debian_version} dep graph: {len(dependency_graph)} packages', file=sys.stderr)
        except Exception as e:
            print(f'Warning: could not load nested deps: {e}', file=sys.stderr)

    label_parts = ['Enhanced Rule+LLM']
    if alias_map:
        label_parts.append('provider aliases')
    if excluded_packages:
        label_parts.append('filtered')
    result = evaluate(
        gt_by_project,
        pred_by_project,
        dep_closure_fn,
        alias_map=alias_map,
        label=' + '.join(label_parts),
        excluded_packages=excluded_packages,
    )

    if args.save and result:
        with open(args.save, 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f'\nDetailed results saved to {args.save}')


if __name__ == '__main__':
    main()
