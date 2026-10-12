#!/usr/bin/env python3
from __future__ import annotations

import shlex


PACKAGE_COMPAT_CANDIDATES = {
    "autoconf": ("autoconf",),
    "automake": ("automake",),
    "bison": ("bison",),
    "build-essential": ("build-essential",),
    "ca-certificates": ("ca-certificates",),
    "cmake": ("cmake", "cmake3"),
    "flex": ("flex",),
    "git": ("git", "git-core"),
    "libssl-dev": ("libssl-dev", "libssl1.0-dev", "libssl0.9.8-dev"),
    "libtool": ("libtool", "libtool-bin"),
    "make": ("make",),
    "meson": ("meson",),
    "ninja": ("ninja-build", "ninja"),
    "ninja-build": ("ninja-build", "ninja"),
    "patch": ("patch",),
    "pkg-config": ("pkg-config", "pkgconf"),
    "pkgconf": ("pkgconf", "pkg-config"),
    "python": ("python3", "python2", "python", "python2.7", "python2.6", "python2.5"),
    "python-dev": (
        "python3-dev",
        "python-dev",
        "python2-dev",
        "python2.7-dev",
        "python2.6-dev",
        "python2.5-dev",
    ),
    "python-pip": ("python3-pip", "python-pip"),
    "python-setuptools": ("python3-setuptools", "python-setuptools"),
    "python3": ("python3", "python2", "python", "python2.7", "python2.6", "python2.5"),
    "python3-dev": (
        "python3-dev",
        "python-dev",
        "python2-dev",
        "python2.7-dev",
        "python2.6-dev",
        "python2.5-dev",
    ),
    "python3-pip": ("python3-pip", "python-pip"),
    "python3-setuptools": ("python3-setuptools", "python-setuptools"),
    "python3-venv": ("python3-venv", "python-virtualenv", "virtualenv"),
}


COMMON_BUILD_DEPENDENCIES = (
    "build-essential",
    "pkg-config",
    "autoconf",
    "automake",
    "libtool",
    "cmake",
    "ninja-build",
    "meson",
    "git",
    "ca-certificates",
    "python3",
    "python3-dev",
    "libssl-dev",
)


def get_candidate_packages(requested_package: str) -> tuple[str, ...]:
    return PACKAGE_COMPAT_CANDIDATES.get(requested_package, (requested_package,))


def get_common_dependency_spec() -> list[dict[str, object]]:
    return [
        {
            "requested_package": requested_package,
            "candidate_packages": list(get_candidate_packages(requested_package)),
        }
        for requested_package in COMMON_BUILD_DEPENDENCIES
    ]


def render_install_script(allow_archive_downgrades: bool = False) -> str:
    downgrade_flag = " --force-yes" if allow_archive_downgrades else ""
    lines = [
        "#!/bin/sh",
        "set -eu",
        "",
        "export DEBIAN_FRONTEND=noninteractive",
        'apt_update_cmd="apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 -o Acquire::http::Pipeline-Depth=0 -o Acquire::https::Pipeline-Depth=0 -o Acquire::http::Dl-Limit=0 -o Acquire::https::Dl-Limit=0 -o Acquire::AllowInsecureRepositories=true -o APT::Get::AllowUnauthenticated=true update"',
        '$apt_update_cmd',
        'install_pkgs=""',
        "",
        "append_pkg() {",
        '  pkg="$1"',
        '  case " $install_pkgs " in',
        '    *" $pkg "*) ;;',
        '    *) install_pkgs="${install_pkgs} ${pkg}" ;;',
        "  esac",
        "}",
        "",
        "probe_first_available() {",
        '  logical_name="$1"',
        "  shift",
        '  resolved=""',
        '  for candidate in "$@"; do',
        "    if apt-cache show \"$candidate\" 2>/dev/null | grep -q '^Package:'; then",
        '      resolved="$candidate"',
        "      break",
        "    fi",
        "  done",
        "",
        '  if [ -n "$resolved" ]; then',
        '    append_pkg "$resolved"',
        '    echo "resolved: $logical_name -> $resolved"',
        "    return 0",
        "  fi",
        "",
        '  echo "missing: $logical_name" >&2',
        "  return 1",
        "}",
        "",
    ]

    for requested_package in COMMON_BUILD_DEPENDENCIES:
        quoted_args = " ".join(
            shlex.quote(candidate)
            for candidate in (requested_package, *get_candidate_packages(requested_package))
        )
        lines.append(f"probe_first_available {quoted_args} || true")

    lines.extend(
        [
            "",
            'if [ -z "$install_pkgs" ]; then',
            '  echo "no common build dependencies could be resolved" >&2',
            "  exit 1",
            "fi",
            "",
            'attempt=1',
            'max_attempts=5',
            "while :; do",
            "  # shellcheck disable=SC2086",
            f'  if apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 -o Acquire::http::Pipeline-Depth=0 -o Acquire::https::Pipeline-Depth=0 -o Acquire::http::Dl-Limit=0 -o Acquire::https::Dl-Limit=0 -o APT::Get::AllowUnauthenticated=true install -y --allow-unauthenticated{downgrade_flag} --fix-missing $install_pkgs; then',
            "    break",
            "  fi",
            '  if [ "$attempt" -ge "$max_attempts" ]; then',
            '    echo "apt-get install failed after ${max_attempts} attempts" >&2',
            "    exit 1",
            "  fi",
            '  attempt=$((attempt + 1))',
            '  echo "retrying apt-get install (${attempt}/${max_attempts})" >&2',
            "  $apt_update_cmd",
            "  sleep 5",
            "done",
        ]
    )
    return "\n".join(lines) + "\n"


def render_docker_run_block(allow_archive_downgrades: bool = False) -> str:
    script = render_install_script(allow_archive_downgrades).rstrip().splitlines()
    body = [line for line in script if line != "#!/bin/sh"]
    return "RUN /bin/sh -eu <<'EOF'\n" + "\n".join(body) + "\nEOF\n"


if __name__ == "__main__":
    print(render_docker_run_block())
