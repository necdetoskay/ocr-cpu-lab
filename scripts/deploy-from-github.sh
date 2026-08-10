#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/necdetoskay/ocr-cpu-lab.git#main:docker-compose.cpu.yml"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed or not in PATH." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: Docker Compose v2 plugin is not available." >&2
  exit 1
fi

echo "[ocr-cpu-lab] Pulling/building and starting CPU-only stack from GitHub..."
docker compose -f "$REPO_URL" pull || true
docker compose -f "$REPO_URL" up -d --build

echo
 echo "[ocr-cpu-lab] Containers:"
docker compose -f "$REPO_URL" ps

echo
 echo "[ocr-cpu-lab] UI should become available at: http://<SERVER_IP>:7861"
echo "[ocr-cpu-lab] Model server logs:"
echo "docker logs -f ocr-cpu-llama-server"
echo "[ocr-cpu-lab] Resource usage:"
echo "docker stats ocr-cpu-llama-server ocr-cpu-ui"
