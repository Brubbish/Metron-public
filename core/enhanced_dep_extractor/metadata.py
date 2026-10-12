"""Versioned Debian metadata paths for the enhanced dependency extractor."""

from __future__ import annotations

import os
import json
from pathlib import Path


DEFAULT_DEBIAN_VERSION = 'trixie'
SUPPORTED_DEBIAN_VERSIONS = (
    'etch', 'lenny', 'squeeze', 'wheezy', 'jessie',
    'stretch', 'buster', 'bullseye', 'bookworm', 'trixie',
)

BUILD_RAG_INFO_DIR = Path(os.environ.get('METRON_RAG_DIR', './rag'))
DOWNLOAD_DIR = BUILD_RAG_INFO_DIR / 'Download'
PKGLIST_DIR = BUILD_RAG_INFO_DIR / 'pkglist'
PKGPATH_DIR = BUILD_RAG_INFO_DIR / 'pkgpath'
PROVIDER_INDEX_DIR = BUILD_RAG_INFO_DIR / 'provider_index'
PKGLISTDB_DIR = BUILD_RAG_INFO_DIR / 'pkglistdb'
METADATA_MANIFEST_DIR = BUILD_RAG_INFO_DIR / 'metadata_manifests'


def normalize_debian_version(version: str | None) -> str:
    version = (version or DEFAULT_DEBIAN_VERSION).strip().lower()
    if version not in SUPPORTED_DEBIAN_VERSIONS:
        allowed = ', '.join(SUPPORTED_DEBIAN_VERSIONS)
        raise ValueError(f'Unsupported Debian version {version!r}; expected one of: {allowed}')
    return version


def pkglist_path(version: str | None = None) -> Path:
    version = normalize_debian_version(version)
    return PKGLIST_DIR / f'{version}_pkglist_amd64.json'


def pkgpath_path(version: str | None = None) -> Path:
    version = normalize_debian_version(version)
    return PKGPATH_DIR / f'{version}_pkgpath_amd64.json'


def provider_index_path(version: str | None = None) -> Path:
    version = normalize_debian_version(version)
    return PROVIDER_INDEX_DIR / f'{version}_provider_index_amd64.json'


def chromadb_path(version: str | None = None) -> Path:
    version = normalize_debian_version(version)
    return PKGLISTDB_DIR / f'{version}_pkglist_amd64_db'


def chromadb_collection(version: str | None = None) -> str:
    version = normalize_debian_version(version)
    return f'{version}_db'


def pkginfo_glob(version: str | None = None) -> str:
    version = normalize_debian_version(version)
    return f'{version}*_amd64_pkginfo'


def pkginfo_paths(version: str | None = None) -> list[Path]:
    return sorted(path for path in DOWNLOAD_DIR.glob(pkginfo_glob(version)) if path.is_file())


ARTIFACT_CACHE_DIR = BUILD_RAG_INFO_DIR / 'artifact_cache'
DISTRO_MAPPING_PATH = BUILD_RAG_INFO_DIR / 'distro_mapping.json'
TOKEN_PRIORS_DIR = BUILD_RAG_INFO_DIR / 'token_priors'


def metadata_manifest_path(version: str | None = None) -> Path:
    version = normalize_debian_version(version)
    return METADATA_MANIFEST_DIR / f'{version}_amd64.json'


def metadata_profile_hash(version: str | None = None) -> str:
    path = metadata_manifest_path(version)
    if not path.is_file():
        return 'unversioned'
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return 'invalid'
    return str(payload.get('repository_profile_hash') or 'unversioned')


def artifact_cache_path(version: str | None = None) -> Path:
    version = normalize_debian_version(version)
    override = os.environ.get('ENHANCED_DEP_ARTIFACT_CACHE_PATH')
    if override:
        return Path(override)
    profile_hash = metadata_profile_hash(version)
    return ARTIFACT_CACHE_DIR / 'v2' / profile_hash / f'{version}_artifact_cache.json'


def token_priors_path(version: str | None = None) -> Path:
    """Path to the data-driven library token priors file.

    The file is built by `tools/build_token_priors.py` from any historical baseline +
    GT pair. If it does not exist, `confidence_filter` falls back to a silent no-op.
    """
    version = normalize_debian_version(version)
    return TOKEN_PRIORS_DIR / f'{version}_token_priors.json'


def env_path(name: str, default: Path) -> str:
    return os.environ.get(name, str(default))
