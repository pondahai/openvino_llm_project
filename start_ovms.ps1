# 啟動 OpenVINO Model Server (OVMS) 並載入 Qwen3.8-27B-int4-ov
# 需先下載 OVMS Windows 套件並解壓 (詳見 docs/Intel_IrisXe_OpenVINO_Notes.md 第十節)
# 用法: .\start_ovms.ps1 [-OvmsDir C:\Users\USER\ovms_pkg\ovms] [-Port 8000]
param(
    [string]$OvmsDir = $(if ($env:OVMS_DIR) { $env:OVMS_DIR } else { "C:\Users\USER\ovms_pkg\ovms" }),
    [string]$ModelPath = (Join-Path $PSScriptRoot "models\Qwen3.8-27B-int4-ov"),
    [int]$Port = 8000
)

if (-not (Test-Path (Join-Path $OvmsDir "setupvars.ps1"))) {
    Write-Error "找不到 OVMS：$OvmsDir (可用 -OvmsDir 或環境變數 OVMS_DIR 指定)"
    exit 1
}
. (Join-Path $OvmsDir "setupvars.ps1") | Out-Null

# 模型資料夾含視覺元件，必須使用 VLM_CB
# max_num_batched_tokens=128：分段預填充，避免 Iris Xe 觸發 TDR
# cache_size=2：KV cache 2GB (權重約 26GB，GPU 共享記憶體池約 29.5GB)
ovms --rest_port $Port `
  --model_name qwen3.8-27b-ovms `
  --model_path $ModelPath `
  --task text_generation `
  --target_device GPU `
  --pipeline_type VLM_CB `
  --cache_size 2 `
  --max_num_batched_tokens 128 `
  --max_num_seqs 1 `
  --enable_prefix_caching true `
  --tool_parser qwen3coder `
  --reasoning_parser qwen3 `
  --plugin_config '{\"GPU_ENABLE_LARGE_ALLOCATIONS\": true, \"GPU_QUEUE_PRIORITY\": \"LOW\", \"GPU_QUEUE_THROTTLE\": \"LOW\", \"MODEL_PRIORITY\": \"LOW\"}' `
  --log_level INFO
