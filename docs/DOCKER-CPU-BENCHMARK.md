# Docker CPU Benchmark

This stack runs the OvisOCR2 GGUF test on a real server with Docker Compose so CPU-only measurements are reproducible and are not distorted by the developer workstation workload.

## Architecture

- `llama-server`: official `ghcr.io/ggml-org/llama.cpp:server` image
- Model: `bartowski/ATH-MaaS_OvisOCR2-GGUF:Q4_K_M`
- GPU layers: `0`
- multimodal projector GPU offload: disabled
- `ocr-ui`: lightweight Gradio application
- llama.cpp cache persisted in a named Docker volume

## Requirements

- Docker Engine with Docker Compose plugin
- Linux amd64 or arm64 host
- Internet access for the first model download
- enough free disk space for the model/cache

## Start

```bash
git pull
docker compose -f docker-compose.cpu.yml pull
docker compose -f docker-compose.cpu.yml up -d --build
```

Watch model startup/download:

```bash
docker compose -f docker-compose.cpu.yml logs -f llama-server
```

Watch both services:

```bash
docker compose -f docker-compose.cpu.yml logs -f
```

When the server is ready, open:

```text
http://SERVER_IP:7861
```

The llama.cpp API is also exposed on port 8080 for benchmark inspection.

## Verify CPU-only configuration

```bash
docker compose -f docker-compose.cpu.yml ps
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

```bash
docker compose -f docker-compose.cpu.yml down
```

The downloaded model cache remains in the `llama-cache` named volume.

To remove the cache intentionally:

```bash
docker compose -f docker-compose.cpu.yml down -v
```

## Interpretation

The workstation GGUF baseline currently measured approximately 64.9 seconds for a 1489x2105 clean one-page PDF with 1393 completion tokens. The server test should use the same document so the host hardware effect can be measured directly.
