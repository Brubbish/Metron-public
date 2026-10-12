#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${THEAUTO_PYTHON_BIN:-python3}"
TAG_SUFFIX="${1:-}"

# Resolved package-list manifests are optional; without them the generated
# contexts fall back to the declared repositories/components of the profile.
MANIFEST_ARGS=()
if [ ! -d "${SCRIPT_DIR}/metadata_manifests" ]; then
  MANIFEST_ARGS+=(--allow-declared-only)
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_debian_images.py" "${MANIFEST_ARGS[@]}"

for release in etch lenny squeeze wheezy jessie stretch buster bullseye bookworm trixie; do
  bash "${SCRIPT_DIR}/build_one_image.sh" "${release}" "${TAG_SUFFIX}"
done
