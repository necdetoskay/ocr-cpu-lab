# Docker CPU Benchmark

This stack runs the OvisOCR2 GGUF test on a real server with Docker Compose so CPU-only measurements are reproducible and are not distorted by the developer workstation workload.

## Architecture

- `llama-server`: official `ghcr.io/ggml-org/llama.cpp:server` image
- Model: `bartowski/ATH-MaaS_OvisOCR2-GGUF:Q4_K_M`
- GPU layers: `0`
- multimodal projector GPU offload: disabled
- `ocr-ui`: lightweight Gradio application
- llama.cpp cache persisted in a named Docker volume
- host UI port: `44387`
- host llama.cpp API port: `43721`

## Requirements

- Docker Engine
- Docker Compose v2 with Git-resource support
- Linux amd64 or arm64 host
- Internet access for the first image/model download
- enough free disk space for the model/cache

## Preflight

Check Docker/Compose and make sure the chosen host ports are not already listening:

```bash
docker --version
docker compose version
ss -ltn | grep -E ':(43721|44387)\b' || true
```

No output from the `ss` command means both host ports are currently free.

## Preferred deployment — no manual clone

Docker Compose can load the Compose project directly from the public GitHub repository. The repository is fetched by Compose/BuildKit as needed; no manual `git clone` is required.

```bash
docker compose -p ocr-cpu-lab -f "https://github.com/necdetoskay/ocr-cpu-lab.git#main:docker-compose.cpu.yml" up -d --build
```

Keep the URL quoted because `#` has shell meaning.

Check status:

```bash
docker ps --filter name=ocr-cpu
```

Watch the model server:

```bash
docker logs -f ocr-cpu-llama-server
```

Watch UI:

```bash
docker logs -f ocr-cpu-ui
```

When the server is ready, open:

```text
http://SERVER_IP:44387
```

The llama.cpp API is exposed on:

```text
http://SERVER_IP:43721
```

Container-internal ports remain 7861 and 8080; only the host mappings use the high ports above.

## Traditional cloned-repository deployment

If remote Compose is unavailable on an older Compose release, clone the repository and run:

```bash
git clone https://github.com/necdetoskay/ocr-cpu-lab.git
cd ocr-cpu-lab
docker compose -p ocr-cpu-lab -f docker-compose.cpu.yml up -d --build
```

## Verify CPU-only configuration

```bash
docker stats ocr-cpu-llama-server ocr-cpu-ui
```

Inspect the llama-server logs and confirm the command includes:

```text
--n-gpu-layers 0
--no-mmproj-offload
```

This Compose file uses the non-CUDA llama.cpp server image.

## Record the server before benchmarking

```bash
uname -a
lscpu
free -h
docker version
docker compose version
```

If available, also record:

```bash
numactl --hardware
```

## Benchmark sequence

Use the exact same test PDF used on the workstation first.

1. One clean page — comparison baseline.
2. Three-page PDF — sequential stability.
3. Ten-page PDF — throughput.
4. Resolution benchmark.
5. Complex layout/table/formula/handwriting benchmark.

Do not compare two hosts using different input PDFs or different render settings.

## Useful monitoring

```bash
docker stats ocr-cpu-llama-server
```

Host-level monitoring:

```bash
top
```

or, when installed:

```bash
htop
```

## Stop

For remote-Compose deployment:

```bash
docker compose -p ocr-cpu-lab -f "https://github.com/necdetoskay/ocr-cpu-lab.git#main:docker-compose.cpu.yml" down
```

The downloaded model cache remains in the `llama-cache` named volume.

To remove the cache intentionally, use the same command with `-v`.

## Interpretation

The workstation GGUF baseline measured 64.90 seconds for a 1489x2105 clean one-page PDF with 1393 completion tokens. The three-page workstation run completed in 253.63 seconds (84.54 seconds/page average) while the workstation was under concurrent load. The server test should use the same source documents so host-hardware effects can be measured directly.
