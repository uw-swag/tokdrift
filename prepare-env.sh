#!/bin/bash

set -euo pipefail

PROJECT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
PYTHON_VERSION=${1:-3.11}

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH." >&2
    echo "Install it from https://github.com/astral-sh/uv/releases and re-run this script." >&2
    exit 1
fi

printf "%s\n" "${PYTHON_VERSION}" > "${PROJECT_DIR}/.python-version"

echo ">>> Syncing dependencies with uv (Python ${PYTHON_VERSION})"
(
    cd "${PROJECT_DIR}"
    uv sync --dev
)
