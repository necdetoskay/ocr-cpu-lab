$ErrorActionPreference = "Stop"

Write-Host "[ocr-cpu-lab] Starting OvisOCR2 GGUF with llama.cpp (CPU only)..."
Write-Host "[ocr-cpu-lab] Model: bartowski/ATH-MaaS_OvisOCR2-GGUF:Q4_K_M"
Write-Host "[ocr-cpu-lab] GPU layers: 0"
Write-Host "[ocr-cpu-lab] Multimodal projector GPU offload: disabled"

llama-server `
  -hf "bartowski/ATH-MaaS_OvisOCR2-GGUF:Q4_K_M" `
  --n-gpu-layers 0 `
  --no-mmproj-offload `
  --host 127.0.0.1 `
  --port 8080
