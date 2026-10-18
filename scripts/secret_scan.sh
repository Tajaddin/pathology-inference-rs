#!/usr/bin/env bash
# Scan the whole git history and the working tree for secrets before a push.
# Uses a local gitleaks if there is one, else the official container image.
set -euo pipefail

version=8.30.1
repo=$(git rev-parse --show-toplevel)

if command -v gitleaks >/dev/null 2>&1; then
  gitleaks git --redact --exit-code 1 "$repo"
  gitleaks dir --redact --exit-code 1 "$repo"
else
  docker run --rm -v "$repo:/repo" "ghcr.io/gitleaks/gitleaks:v$version" git --redact --exit-code 1 /repo
  docker run --rm -v "$repo:/repo" "ghcr.io/gitleaks/gitleaks:v$version" dir --redact --exit-code 1 /repo
fi
echo "secret scan clean"
