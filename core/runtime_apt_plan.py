"""Runtime normalization for apt install commands.

The extractor can still over-include mutually exclusive Debian provider
packages.  This module keeps the last line of defense close to command
execution, so Phase 2 and Phase 3 both get the same conservative correction.
"""

from __future__ import annotations

import re
import shlex
from typing import Iterable, Optional

from core.debian_sysuse import normalize_sysuse


LEGACY_SUITES = {"etch", "lenny", "squeeze", "wheezy", "jessie"}

PROVIDER_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("pkg-config", "pkgconf"),
    ("libkrb5-dev", "heimdal-dev"),
    ("libcurl4-gnutls-dev", "libcurl4-openssl-dev"),
    ("libgl1-mesa-dev", "libgl1-mesa-swx11-dev", "nvidia-glx-dev", "nvidia-glx-legacy-96xx-dev"),
)


def normalize_apt_install_command(command: str, sysuse: Optional[str] = None) -> str:
    """Normalize simple apt install package lists.

    The function intentionally avoids shell evaluation and only rewrites the
    package-list segment after `apt install` / `apt-get install`.  Commands that
    rely on command substitution are returned unchanged.
    """
    command = (command or "").strip()
    if not command:
        return command
    if "$(" in command or "`" in command:
        return command

    pattern = re.compile(r"\b(?P<apt>apt(?:-get)?)\s+install(?P<body>[^;&|]*)", re.IGNORECASE)
    return pattern.sub(lambda match: _rewrite_install_match(match, sysuse), command, count=1)


def normalize_apt_packages(packages: Iterable[str], sysuse: Optional[str] = None) -> list[str]:
    suite = _normalized_suite(sysuse)
    normalized = []
    for package in packages:
        pkg = str(package or "").strip()
        if not pkg:
            continue
        pkg = _suite_alias(pkg, suite)
        if _looks_like_package_token(pkg):
            normalized.append(pkg)
    normalized = _unique_preserve_order(normalized)
    return _choose_provider_family_members(normalized)


def _rewrite_install_match(match: re.Match[str], sysuse: Optional[str]) -> str:
    body = match.group("body") or ""
    try:
        tokens = shlex.split(body)
    except ValueError:
        return match.group(0)
    if not tokens:
        return match.group(0)

    options: list[str] = []
    packages: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("-"):
            options.append(token)
            if token in {"-o", "-c"} and index + 1 < len(tokens):
                index += 1
                options.append(tokens[index])
        else:
            packages.append(token)
        index += 1

    if not any(opt == "-y" or opt.startswith("--assume-yes") for opt in options):
        options.insert(0, "-y")
    normalized_packages = normalize_apt_packages(packages, sysuse=sysuse)
    if not normalized_packages:
        return match.group(0)
    # An explicit package version is an intentional request for a non-default
    # candidate. Apt refuses a real downgrade under -y unless this option is
    # present; adding it only for version-pinned installs preserves the normal
    # conservative path for unpinned dependency installation.
    if (
        any("=" in package for package in normalized_packages)
        and "--allow-downgrades" not in options
    ):
        options.append("--allow-downgrades")
    return f"{match.group('apt')} install {' '.join(options + normalized_packages)}"


def _suite_alias(package: str, suite: str) -> str:
    if package == "pkgconf" and suite in LEGACY_SUITES:
        return "pkg-config"
    return package


def _choose_provider_family_members(packages: list[str]) -> list[str]:
    package_set = set(packages)
    chosen_by_family: dict[str, str] = {}
    for family in PROVIDER_FAMILIES:
        present = [pkg for pkg in family if pkg in package_set]
        if present:
            chosen_by_family.update({pkg: present[0] for pkg in present})

    out = []
    emitted = set()
    for package in packages:
        chosen = chosen_by_family.get(package, package)
        if chosen in emitted:
            continue
        emitted.add(chosen)
        out.append(chosen)
    return out


def _looks_like_package_token(token: str) -> bool:
    if not token or token in {"&&", "|", ";"}:
        return False
    if token.startswith(("/", "$", "<", ">")):
        return False
    return bool(re.match(r"^[A-Za-z0-9][A-Za-z0-9+_.:-]*(?:=[^\s;&|]+)?$", token))


def _normalized_suite(sysuse: Optional[str]) -> str:
    if not sysuse:
        return ""
    try:
        return normalize_sysuse(sysuse)
    except Exception:
        return str(sysuse).replace("debian:", "").strip().lower()


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out
