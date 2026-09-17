# 啟動 OpenVINO Model Server (OVMS) 並載入 Qwen3.8-27B-int4-ov 或 Qwen3.6-35B-A3B-int4-ov
# 需先下載 OVMS Windows 套件並解壓 (詳見 docs/Intel_IrisXe_OpenVINO_Notes.md 第十節)
# 用法: .\start_ovms.ps1 [-Model qwen3.8-27b|qwen3.6-35b-a3b] [-Device GPU|CPU] [-OvmsDir C:\Users\USER\ovms_pkg\ovms] [-Port 8000]
param(
    [ValidateSet("qwen3.8-27b", "qwen3.6-35b-a3b")]
    [string]$Model = "qwen3.8-27b",
    [string]$OvmsDir = $(if ($env:OVMS_DIR) { $env:OVMS_DIR } else { "C:\Users\USER\ovms_pkg\ovms" }),
    [int]$Port = 8000,
    # 預設：27B 用 GPU，35B-A3B 用 CPU
    [ValidateSet("GPU", "CPU")]
    [string]$Device = $(if ($Model -eq "qwen3.6-35b-a3b") { "CPU" } else { "GPU" })
)

# 兩個模型無法同時載入 (GPU 共享記憶體池約 29.5GB)，API 名稱為 "<Model>-ovms"
$ModelDirs = @{
    "qwen3.8-27b"     = "Qwen3.8-27B-int4-ov"
    "qwen3.6-35b-a3b" = "Qwen3.6-35B-A3B-int4-ov"
}
$ModelPath = Join-Path $PSScriptRoot "models\$($ModelDirs[$Model])"
if (-not (Test-Path (Join-Path $ModelPath "openvino_language_model.xml"))) {
    Write-Error "找不到模型：$ModelPath"
    exit 1
}

if (-not (Test-Path (Join-Path $OvmsDir "setupvars.ps1"))) {
    Write-Error "找不到 OVMS：$OvmsDir (可用 -OvmsDir 或環境變數 OVMS_DIR 指定)"
    exit 1
}
. (Join-Path $OvmsDir "setupvars.ps1") | Out-Null

# 模型資料夾含視覺元件，必須使用 VLM_CB
# GPU：max_num_batched_tokens=128 分段預填充，避免 Iris Xe 觸發 TDR
#      cache_size=2：KV cache 2GB (權重約 26GB，GPU 共享記憶體池約 29.5GB)
# CPU：Qwen3.6-35B-A3B 在 GPU 編譯時超出記憶體池，改用 CPU (系統 RAM 64GB)
if ($Device -eq "GPU") {
    $DeviceArgs = @(
        "--cache_size", "2",
        "--max_num_batched_tokens", "128",
        "--plugin_config", '{\"GPU_ENABLE_LARGE_ALLOCATIONS\": true, \"GPU_QUEUE_PRIORITY\": \"LOW\", \"GPU_QUEUE_THROTTLE\": \"LOW\", \"MODEL_PRIORITY\": \"LOW\"}'
    )
} else {
    $DeviceArgs = @("--cache_size", "4")
}

ovms --rest_port $Port `
  --model_name "$Model-ovms" `
  --model_path $ModelPath `
  --task text_generation `
  --target_device $Device `
  --pipeline_type VLM_CB `
  --max_num_seqs 1 `
  --enable_prefix_caching true `
  --tool_parser qwen3coder `
  --reasoning_parser qwen3 `
  @DeviceArgs `
  --log_level INFO
