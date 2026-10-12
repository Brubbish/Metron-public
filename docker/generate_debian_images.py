#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

from common_build_deps import render_docker_run_block
from debian_release_matrix import (
    RELEASE_CONFIG,
    RELEASE_ORDER,
    REPOSITORY_PROFILE,
    REPOSITORY_PROFILE_HASH,
)


ROOT = Path(__file__).resolve().parent
GENERATED_DIR = ROOT / "generated"
DEFAULT_MANIFEST_DIR = ROOT / "metadata_manifests"


DOCKERFILE_TEMPLATE = """\
FROM {base_image}

LABEL org.theauto.repository-profile="{profile_hash}" \
      org.theauto.debian-release="{release_name}"

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ARG NO_PROXY
ARG no_proxy

ENV HTTP_PROXY=${{HTTP_PROXY}}
ENV HTTPS_PROXY=${{HTTPS_PROXY}}
ENV http_proxy=${{http_proxy}}
ENV https_proxy=${{https_proxy}}
ENV NO_PROXY=${{NO_PROXY}}
ENV no_proxy=${{no_proxy}}

RUN rm -f /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/lib/apt/lists/partial \
    && mkdir -p /etc/apt/preferences.d \
    && printf '%s\\n' {source_lines} > /etc/apt/sources.list \
    && printf '%s\\n' {apt_preference_lines} > /etc/apt/preferences.d/theauto-repositories \
    && printf '%s\\n' '{profile_json}' > /etc/theauto-repository-profile.json

{install_run_block}

CMD ["/bin/bash"]
"""


def _source_targets(config: dict, manifest: dict | None = None) -> list[tuple[str, list[str]]]:
    resolved = (manifest or {}).get("resolved_repositories") or []
    if resolved:
        by_repo: dict[str, list[str]] = {}
        for target in resolved:
            if not isinstance(target, dict):
                continue
            repo_name = str(target.get("repo_name") or "").strip()
            component = str(target.get("component") or "").strip()
            if repo_name and component:
                by_repo.setdefault(repo_name, []).append(component)
        source_targets = sorted(by_repo.items())
    else:
        source_targets = [(repo_name, list(config["components"])) for repo_name in config["repositories"]]
    return [(repo_name, sorted(set(components))) for repo_name, components in source_targets if components]


def _source_lines(config: dict, source_targets: list[tuple[str, list[str]]]) -> list[str]:
    options = " [check-valid-until=no trusted=yes]" if config.get("archive") else ""
    return [
        f"deb{options} {config['base_url']} {repo_name} {' '.join(components)}"
        for repo_name, components in source_targets
    ]


def _apt_preference_lines(config: dict, source_targets: list[tuple[str, list[str]]]) -> list[str]:
    lines: list[str] = []
    for repo_name, _components in source_targets:
        if repo_name.endswith(("-backports", "-backports-sloppy", "-proposed-updates")):
            priority = 100
        elif config.get("archive"):
            priority = 1001
        else:
            priority = 990
        lines.extend(
            [
                "Package: *",
                f"Pin: release n={repo_name}",
                f"Pin-Priority: {priority}",
                "",
            ]
        )
    return lines


def write_release_context(release_name: str, manifest_dir: Path, allow_declared_only: bool = False) -> None:
    release_dir = GENERATED_DIR / release_name
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    config = RELEASE_CONFIG[release_name]
    manifest_path = manifest_dir / f"{release_name}_{REPOSITORY_PROFILE.get('architecture', 'amd64')}.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    if not manifest and not allow_declared_only:
        raise RuntimeError(f"missing resolved metadata manifest: {manifest_path}")
    manifest_profile_hash = manifest.get("profile_hash") or manifest.get("repository_profile_hash")
    if manifest and manifest_profile_hash != REPOSITORY_PROFILE_HASH:
        raise RuntimeError(f"metadata manifest profile mismatch: {manifest_path}")

    profile_record = {
        "schema_version": REPOSITORY_PROFILE["schema_version"],
        "profile_hash": REPOSITORY_PROFILE_HASH,
        "resolved_profile_hash": manifest.get("resolved_profile_hash", ""),
        "release": release_name,
        "architecture": REPOSITORY_PROFILE.get("architecture", "amd64"),
        **config,
    }
    source_targets = _source_targets(config, manifest)
    source_lines = _source_lines(config, source_targets)
    apt_preference_lines = _apt_preference_lines(config, source_targets)
    (release_dir / "Dockerfile").write_text(
        DOCKERFILE_TEMPLATE.format(
            base_image=config["base_image"],
            release_name=release_name,
            profile_hash=REPOSITORY_PROFILE_HASH,
            source_lines=" ".join(json.dumps(line) for line in source_lines),
            apt_preference_lines=" ".join(json.dumps(line) for line in apt_preference_lines),
            profile_json=json.dumps(profile_record, sort_keys=True, separators=(",", ":")),
            install_run_block=render_docker_run_block(bool(config.get("archive"))).rstrip(),
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="append", choices=RELEASE_ORDER)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument(
        "--allow-declared-only",
        action="store_true",
        help="Allow generation without a resolved pkglist manifest (legacy fallback).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    releases = args.release or RELEASE_ORDER
    for release_name in releases:
        write_release_context(release_name, args.manifest_dir, args.allow_declared_only)
    print(
        f"generated {len(releases)} release contexts under {GENERATED_DIR} "
        f"(profile {REPOSITORY_PROFILE_HASH})"
    )


if __name__ == "__main__":
    main()
