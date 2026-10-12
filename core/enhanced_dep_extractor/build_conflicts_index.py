#!/usr/bin/env python3
"""Offline generator: build cross-distro package conflicts index.

Produces a JSON mapping each package to its conflicts and provides groups,
used at runtime to deduplicate LLM output against already-resolved packages.

For versioned Conflicts (e.g. "Conflicts: foo (<< 1.2)"), we resolve them
offline by checking the actual version of foo in the same distro. If the
constraint is satisfied (i.e. the conflict is active), we include it;
otherwise we discard it.

Usage:
    python -m enhanced_dep_extractor.build_conflicts_index [-o output.json]
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '/usr/lib/python3/dist-packages')
from debian.debian_support import version_compare

from .metadata import BUILD_RAG_INFO_DIR, SUPPORTED_DEBIAN_VERSIONS, pkginfo_paths

OUTPUT_PATH = BUILD_RAG_INFO_DIR / 'conflicts_index.json'
PER_DISTRO_OUTPUT_DIR = BUILD_RAG_INFO_DIR / 'conflicts_by_distro'
_VERSION_RE = re.compile(r'^([a-z0-9][a-z0-9.+\-:]+)\s*\(([<>=!]+)\s*([^)]+)\)$')
_NAME_RE = re.compile(r'^([a-z0-9][a-z0-9.+\-:]+)', re.IGNORECASE)
_REL_OP_ALIASES = {
    '<': '<<',
    '>': '>>',
    '==': '=',
}


def _available_distros() -> list[str]:
    return [d for d in SUPPORTED_DEBIAN_VERSIONS if pkginfo_paths(d)]


def _normalize_rel_op(op: str) -> str:
    return _REL_OP_ALIASES.get(op, op)


def _iter_deb822_paragraphs(text: str):
    paragraph: dict[str, str] = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip():
            if paragraph:
                yield paragraph
                paragraph = {}
            current_key = None
            continue
        if raw_line.startswith((' ', '\t')) and current_key:
            paragraph[current_key] += '\n' + raw_line[1:]
            continue
        if ':' not in raw_line:
            continue
        key, value = raw_line.split(':', 1)
        current_key = key.strip()
        paragraph[current_key] = value.strip()
    if paragraph:
        yield paragraph


def _split_relationship_field(raw_field: str) -> list[str]:
    parts: list[str] = []
    for chunk in raw_field.replace('\n', ' ').split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        for alt in chunk.split('|'):
            alt = alt.strip()
            if alt:
                parts.append(alt)
    return parts


def _extract_relationship_name(part: str) -> str:
    match = _NAME_RE.match(part)
    return match.group(1).strip().lower() if match else ''


def _conflict_is_active(
    target_pkg: str,
    op: str,
    constraint_ver: str,
    versions: dict[str, set[str]],
) -> bool:
    """Check whether any known version of target_pkg activates the conflict."""
    normalized_op = _normalize_rel_op(op)
    if normalized_op not in {'<<', '<=', '>>', '>=', '='}:
        return False
    for actual in versions.get(target_pkg, ()):
        try:
            cmp = version_compare(actual, constraint_ver)
        except Exception:
            continue
        if normalized_op == '<<' and cmp < 0:
            return True
        if normalized_op == '<=' and cmp <= 0:
            return True
        if normalized_op == '>>' and cmp > 0:
            return True
        if normalized_op == '>=' and cmp >= 0:
            return True
        if normalized_op == '=' and cmp == 0:
            return True
    return False


def _parse_pkginfo(
    distro: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, set[str]]]:
    """Parse suite-wide Conflicts, Provides, and Versions from all pkginfo channels."""
    paths = pkginfo_paths(distro)
    if not paths:
        return {}, {}, {}

    conflicts_raw: dict[str, list[str]] = defaultdict(list)
    provides_raw: dict[str, set[str]] = defaultdict(set)
    versions: dict[str, set[str]] = defaultdict(set)

    for path in paths:
        with open(path, errors='replace') as f:
            for paragraph in _iter_deb822_paragraphs(f.read()):
                current_pkg = paragraph.get('Package', '').strip().lower()
                if not current_pkg:
                    continue
                version = paragraph.get('Version', '').strip()
                if version:
                    versions[current_pkg].add(version)

                raw_conflicts = paragraph.get('Conflicts', '').strip()
                if raw_conflicts:
                    conflicts_raw[current_pkg].append(raw_conflicts)

                raw_provides = paragraph.get('Provides', '').strip()
                if raw_provides:
                    for part in _split_relationship_field(raw_provides):
                        name = _extract_relationship_name(part)
                        if name:
                            provides_raw[current_pkg].add(name)

    conflicts: dict[str, list[str]] = {}
    for pkg, raw_fields in conflicts_raw.items():
        resolved: set[str] = set()
        for raw_field in raw_fields:
            for part in _split_relationship_field(raw_field):
                match = _VERSION_RE.match(part)
                if match:
                    target, op, ver = match.group(1).lower(), match.group(2), match.group(3)
                    if target != pkg and _conflict_is_active(target, op, ver, versions):
                        resolved.add(target)
                    continue
                name = _extract_relationship_name(part)
                if name and name != pkg:
                    resolved.add(name)
        if resolved:
            conflicts[pkg] = sorted(resolved)

    provides = {
        pkg: sorted(names)
        for pkg, names in provides_raw.items()
        if names
    }
    return conflicts, provides, versions


def _build_provides_groups(provides: dict[str, list[str]]) -> dict[str, list[str]]:
    """Group virtual package providers without treating them as hard conflicts."""
    virtual_to_real: dict[str, set[str]] = defaultdict(set)
    for pkg, virt_list in provides.items():
        for virt in virt_list:
            if virt != pkg:
                virtual_to_real[virt].add(pkg)

    return {
        virt: sorted(real_pkgs)
        for virt, real_pkgs in virtual_to_real.items()
        if len(real_pkgs) > 1
    }


def build_conflicts_index() -> dict:
    """Build cross-distro conflicts index.

    Structure:
    {
        "conflicts": {
            "distro": {
                "pkg": ["conflicting_pkg1", "conflicting_pkg2"]
            }
        },
        "provides_groups": {
            "distro": {
                "virtual_pkg": ["real_pkg1", "real_pkg2"]
            }
        }
    }
    """
    result = {
        'conflicts': {},
        'provides_groups': {},
    }

    for distro in _available_distros():
        conflicts, provides, _ = _parse_pkginfo(distro)
        provides_groups = _build_provides_groups(provides)

        if conflicts:
            result['conflicts'][distro] = conflicts
        if provides_groups:
            result['provides_groups'][distro] = provides_groups

    return result


def build_per_distro_conflicts_index() -> dict[str, dict]:
    """Build one conflicts payload per distro to support ablation-friendly use."""
    payloads: dict[str, dict] = {}
    for distro in _available_distros():
        conflicts, provides, _ = _parse_pkginfo(distro)
        payloads[distro] = {
            'conflicts': conflicts,
            'provides_groups': _build_provides_groups(provides),
        }
    return payloads


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build cross-distro conflicts index')
    parser.add_argument('-o', '--output', default=str(OUTPUT_PATH), help='Output JSON path')
    parser.add_argument(
        '--per-distro-dir',
        default=str(PER_DISTRO_OUTPUT_DIR),
        help='Directory for per-distro conflicts JSON payloads',
    )
    args = parser.parse_args()

    distros = _available_distros()
    print(f'Building conflicts index for: {distros}')
    index = build_conflicts_index()
    per_distro = build_per_distro_conflicts_index()

    for distro in distros:
        n_conf = len(index['conflicts'].get(distro, {}))
        n_prov = len(index['provides_groups'].get(distro, {}))
        print(f'  {distro:10s} conflicts={n_conf:4d}  provides_groups={n_prov:4d}')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(index, f, separators=(',', ':'))

    per_distro_dir = Path(args.per_distro_dir)
    per_distro_dir.mkdir(parents=True, exist_ok=True)
    for distro, payload in per_distro.items():
        with open(per_distro_dir / f'{distro}_conflicts.json', 'w') as f:
            json.dump(payload, f, separators=(',', ':'))

    size_kb = output_path.stat().st_size / 1024
    print(f'Written to {output_path} ({size_kb:.0f} KB)')
    print(f'Per-distro payloads written to {per_distro_dir}')


if __name__ == '__main__':
    main()
