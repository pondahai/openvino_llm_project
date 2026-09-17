# ⚡ Intel Iris Xe OpenVINO LLM 推論與 Agent API 專案

本專案實現在 **Intel 第 11 代 Core i5 整合式繪圖核心（Iris Xe 80 EUs iGPU）** 與 **64GB 雙通道 DDR4 記憶體** 上，透過 **OpenVINO 原生加速引擎** 部署旗艦級 **Qwen3.8-27B (INT4)** 大語言模型。

專案提供**互動式 Web 對話介面**以及**完全相容 OpenAI API 協議的微服務**，可無縫對接 LangChain、AutoGen、CrewAI 等各類 Agent 框架。

---

## 📸 開版實測展示 (GUI Showcase)

以下為本機 Intel Iris Xe GPU 實際進行經典對話的即時截圖：

![Intel Iris Xe 對話展示](docs/images/gui_showcase.png)

* **核心硬體**：Intel 11th Gen Core i5-1135G7 @ 2.40GHz + Iris Xe Graphics (80 EUs)
* **記憶體**：64 GB DDR4-3200 雙通道
* **推論核心**：`OpenVINO/Qwen3.8-27B-int4-ov` (14.87 GB)
* **實測表現**：首字延遲 (TTFT) ~2.58s，生成速度 **1.72 ~ 1.87 tokens/s**（達到硬體頻寬極限約 84% 實用率）

---

## 📊 理論速度推導 vs 實際對話速度

自回歸 LLM 生成階段為典型的記憶體頻寬受限任務（Memory-Bound）：

| 評估維度 / 項目 | 理論極限 | 實際實測數值 (Native Benchmark) | 實際對話介面 (GUI 對話情境) |
| :--- | :--- | :--- | :--- |
| **記憶體頻寬** | 51.2 GB/s (DDR4-3200 雙通道) | ~25.6 GB/s 有效利用 | ~27.8 GB/s 有效利用 |
| **流式產字速度 (TPS)** | **2.23 ~ 3.44 tokens/s** | **1.72 tokens/s** | **1.87 tokens/s** |
| **首字延遲 (TTFT)** | < 3.0 秒 | **2.58 秒** | **6.51 秒 (含網頁通訊)** |
| **記憶體佔用 (RAM)** | 14.87 GB (基礎權重) | **26.18 GB** (含 KV Cache) | **26.50 GB** (含網頁緩衝) |

> 完整理論推導過程與深度分析請參閱：[`docs/Intel_IrisXe_OpenVINO_Notes.md`](docs/Intel_IrisXe_OpenVINO_Notes.md)，所有筆記的導覽見下方 [📚 研究筆記導覽](#-研究筆記導覽)。

---

## 🚀 快速上手指南

### 1. 安裝環境依賴
```powershell
pip install -r requirements.txt
```

### 2. 啟動互動 Web 介面 (Streamlit GUI)
雙擊執行 `run_gui.bat` 或在終端機執行：
```powershell
streamlit run app.py
```
開啟瀏覽器前往 `http://localhost:8501`。

### 3. 啟動 OpenAI 相容 API 伺服器 (供 Agent 使用)
雙擊執行 `run_api_server.bat` 或在終端機執行：
```powershell
python -u openai_server.py
```
改用 Qwen3.6-35B-A3B (CPU 推論，模型名稱 `qwen3.6-35b-a3b-int4-ov`)：執行 `run_api_server.bat qwen3.6-35b-a3b`，或先設定 `$env:LLM_MODEL = "qwen3.6-35b-a3b"`。

伺服器將在 `http://127.0.0.1:1234` 啟動，完全相容 OpenAI SDK：
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="qwen3.8-27b-int4-ov",
    messages=[{"role": "user", "content": "你好！"}],
    stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

### 4. (推薦) 使用 OpenVINO Model Server (OVMS) 作為後端
OVMS 為 Intel 官方推論伺服器，內建工具呼叫解析與 **prefix caching** (可同時快取多段對話)。自製伺服器也支援 prefix caching，但只保留最近一段對話的狀態。

1. 下載 OVMS Windows 套件 (weekly 版，正式版 2026.3.1 無法載入本模型的 main 版本)：
   `https://storage.openvinotoolkit.org/repositories/openvino_model_server/packages/weekly/latest/`
2. 解壓後以 `-OvmsDir` 或環境變數 `OVMS_DIR` 指定 `ovms` 資料夾，雙擊 `run_ovms.bat` 或執行：
```powershell
.\start_ovms.ps1 -OvmsDir C:\path\to\ovms
```
3. API 位址為 `http://127.0.0.1:8000/v3`，模型名稱 `qwen3.8-27b-ovms`；GUI 側邊欄可切換後端。
4. 關閉思考模式時請在請求加上 `stop: ["</think>"]` (模型偶爾在回答後輸出 `</think>` 並重複回答)：
```python
client = OpenAI(base_url="http://127.0.0.1:8000/v3", api_key="not-needed")
client.chat.completions.create(
    model="qwen3.8-27b-ovms",
    messages=[{"role": "user", "content": "你好！"}],
    stop=["</think>"],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```
詳細評估與實測數據見 [研究筆記第十、十一節](docs/Intel_IrisXe_OpenVINO_Notes.md)。

---

## 📚 研究筆記導覽

所有筆記位於 [`docs/`](docs/)，建議閱讀順序：歷程日誌 → 測試計畫 → 研究筆記。

### 1. [📜 實戰完整歷程日誌](docs/Intel_IrisXe_Technical_Chronicle.md)
從「這台筆電能不能跑大模型」出發的時間線紀錄：
* 硬體體檢、LM Studio 與 OpenVINO 後端比較
* 驗證「能否共用 LM Studio 的 GGUF 模型」與相容性掃描失敗原因
* Intel AI Playground 的限制、突破下載限速、Qwen3.8-27B 首次實測
* ChatML 對話模板演進與 OpenAI 相容 Agent Server 的誕生

### 2. [🧪 推論測試計畫與首發成果](docs/Qwen3.8_27B_IrisXe_Test_Plan.md)
* 測試目標、環境基準、指標 (TTFT / TPS / 記憶體) 與三種 Prompt 情境
* 自動化基準測試腳本與驗收標準
* 首發實測：峰值記憶體 26.18 GB、生成約 1.7~1.9 tokens/s

### 3. [📓 完整研究筆記](docs/Intel_IrisXe_OpenVINO_Notes.md)
專案的核心技術文件，依章節整理：

| 章節 | 內容 |
| :--- | :--- |
| 一 ~ 三 | 硬體規格、DDR4 頻寬推導理論速度、理論 vs 三種實測模式對比 |
| 四 ~ 六 | OpenVINO 環境偵測、GGUF 格式不相容分析、與 LM Studio 對話行為對齊 |
| 七 | 專案展示 (前後端分離 Client 模式 / 獨立 GUI 模式) |
| 八 | 除錯實錄：Jinja2 500 錯誤、GPU 4GB 單塊記憶體上限、OpenCL `-14` 與 TDR 螢幕閃爍 |
| 九 | 硬體極限分析 (29.5GB GPU 記憶體池、TDR、頻寬)、KV Cache 估算 (為何 128k 不可行)、分段預填充與 16k 上下文 |
| 十 | OpenVINO Model Server (OVMS) 評估：版本相容性、量化 bug 風險、設定對照與實測比較 |
| 十一 | 自製伺服器 Prefix Caching 設計與實測 (多輪長文由 ~250 秒降到 5 秒) |
| 十二 | Qwen3.6-35B-A3B (MoE) 部署：GPU 記憶體池不足改用 CPU (生成約 4 tokens/s)、自製伺服器預填充約 13 tokens/s 並修正 prefix caching、HETERO 失敗 |

### 4. [📄 GGUF 相容性掃描原始紀錄](docs/openvino_test_results.txt)
掃描 LM Studio 模型資料夾 (共 13 個 GGUF 檔) 並以 OpenVINO GenAI 嘗試直接載入的原始輸出，記錄各模型失敗原因 (對應歷程日誌第四階段)。

---

## 📂 專案結構

```text
openvino_llm_project/
│
├── app.py                     # Streamlit 網頁互動對話介面 (支援串流打字機效果)
├── openai_server.py           # 相容 OpenAI API (Port 1234) 之微服務
├── test_openai_client.py      # OpenAI 官方 SDK 測試腳本
├── run_benchmark.py           # 原生 OpenVINO 推論效能基準測試
├── native_infer_qwen27b.py    # 終端機純 Python 推論測試腳本
├── fast_resumable_downloader.py # 高速分塊斷點續傳下載器
├── requirements.txt           # 專案套件依賴清單
├── run_gui.bat                # 一鍵啟動 GUI 對話介面批次檔
├── run_api_server.bat         # 一鍵啟動 API 伺服器批次檔
├── run_ovms.bat               # 一鍵啟動 OVMS 批次檔
├── start_ovms.ps1             # OVMS 啟動參數 (VLM_CB / GPU / prefix caching)
├── .gitignore                 # Git 忽略設定 (排除巨大權重與快取)
└── docs/                      # 開發筆記、理論推導與測試報告
    ├── Intel_IrisXe_OpenVINO_Notes.md    # 核心硬體規格與頻寬理論推導筆記
    ├── Intel_IrisXe_Technical_Chronicle.md # 完整踩坑歷程與技術紀要
    ├── Qwen3.8_27B_IrisXe_Test_Plan.md   # 測試計劃與驗收標準
    ├── openvino_test_results.txt         # GGUF 相容性掃描原始紀錄
    └── images/
        └── gui_showcase.png              # GUI 實測開版成果截圖
```

---

## 🛡️ 授權與宣告
本專案僅供學術研究與邊緣運算硬體評估使用。模型權重來自 Hugging Face 官方發佈之 `OpenVINO/Qwen3.8-27B-int4-ov`。
