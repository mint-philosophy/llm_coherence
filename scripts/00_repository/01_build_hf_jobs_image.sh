#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-${IMAGE:-}}"
PLATFORM="${PLATFORM:-linux/amd64}"

if [[ -z "$IMAGE" ]]; then
  echo "Usage: $0 <docker-image-tag>" >&2
  echo "Example: $0 elenaajayi/llm-coherence-vllm:glm-base-20260618" >&2
  echo "You can also set IMAGE=<docker-image-tag>." >&2
  exit 1
fi

docker buildx build \
  --platform "$PLATFORM" \
  -f Dockerfile.hf_jobs \
  -t "$IMAGE" \
  --push \
  .
