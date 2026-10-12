#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 <release> [tag-suffix]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE="$1"
TAG_SUFFIX="${2:-}"
IMAGE_TAG="${RELEASE}${TAG_SUFFIX:+-${TAG_SUFFIX}}"
CONTEXT_DIR="${SCRIPT_DIR}/generated/${RELEASE}"

if [ ! -d "${CONTEXT_DIR}" ]; then
  echo "missing build context: ${CONTEXT_DIR}" >&2
  echo "run: python3 ${SCRIPT_DIR}/generate_debian_images.py --release ${RELEASE} --allow-declared-only" >&2
  exit 1
fi

HTTP_PROXY_VALUE="${HTTP_PROXY:-${http_proxy:-}}"
HTTPS_PROXY_VALUE="${HTTPS_PROXY:-${https_proxy:-}}"
NO_PROXY_VALUE="${NO_PROXY:-${no_proxy:-}}"

docker build \
  --build-arg HTTP_PROXY="${HTTP_PROXY_VALUE}" \
  --build-arg HTTPS_PROXY="${HTTPS_PROXY_VALUE}" \
  --build-arg http_proxy="${HTTP_PROXY_VALUE}" \
  --build-arg https_proxy="${HTTPS_PROXY_VALUE}" \
  --build-arg NO_PROXY="${NO_PROXY_VALUE}" \
  --build-arg no_proxy="${NO_PROXY_VALUE}" \
  -t "theauto/debian:${IMAGE_TAG}" \
  "${CONTEXT_DIR}"
