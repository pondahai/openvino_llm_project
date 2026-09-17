# 📜 本機 Intel Iris Xe 與 OpenVINO LLM 實戰完整歷程日誌

本文依時間序列詳細記錄本次技術探索的完整來龍去脈、背景脈絡、實測步驟與排錯細節。

---

## 🕒 第一階段：緣起與硬體體檢

### 1. 問題發起
* **提問**：「本機整合式 GPU（iGPU）可以用在 LLM 推論嗎？」
* **理論解答**：可以。iGPU 具備與系統主記憶體共享的「統一記憶體（Unified Memory）」特性，能突破一般獨立顯卡固定顯存（如 8GB/16GB VRAM）的限制，只要主機板 RAM 夠大，就能裝載更大參數量（如 14B~70B）的模型，但缺點在於一般 DDR 記憶體頻寬（約 40~60 GB/s）不及獨立顯卡的 GDDR6（300~1000 GB/s）。

### 2. 本機實體硬體檢測（PowerShell 深入查詢）
透過 WMI 指令對系統進行即時檢測：
* **處理器**：`Intel(R) Core(TM) i5-1135G7 @ 2.40GHz` (4C8T, Tiger Lake)
* **顯示核心**：`Intel(R) Iris(R) Xe Graphics (iGPU)`
* **記憶體規格**：
  * 插槽 1：`32GB DDR4-3200`
  * 插槽 2：`32GB DDR4-3200`
  * **總容量**：**64 GB（雙通道運作）**
  * 理論記憶體頻寬：$3200 \times 8 \times 2 \div 1000 \approx \mathbf{51.2\text{ GB/s}}$

> **結論**：本機具備罕見的「超大 64GB 記憶體」，容量上甚至能吃下 27B~32B 等級的大型模型。

---

## 🕒 第二階段：生態現況與後端比較

### 1. LM Studio 與 OpenVINO 的整合現況
* 經查詢，**LM Studio 官方並沒有原生整合 OpenVINO**。
* 在 Windows 平台上，LM Studio 調用 Intel 內顯唯一可行且開箱即用的方案是 **Vulkan 後端**。

### 2. 後端效能對比研究
根據 Intel 官方與開發者社群的公開 Benchmark 比較：
* **Intel SYCL (oneAPI)**：算力釋放最高 (100%)，但安裝複雜（需裝整套數十 GB 的 oneAPI Toolkit）。
* **Intel OpenVINO**：推論速度次高 (~85-95%)，針對 Intel EU 算力有深度優化，安裝輕量。
* **Vulkan (LM Studio 預設)**：通用圖形 API 模擬 (~60-75%)，雖然能跑，但無法發揮 Intel 專屬指令集最佳效率。
* **純 CPU (AVX-512)**：最慢 (~30-50%)，且 CPU 負載滿載發熱。

---

## 🕒 第三階段：環境搭建與「模型共用」假說驗證

### 1. 安裝 OpenVINO 原生環境
在終端機安裝最新套件：
* `openvino` (2026.3.1)
* `openvino-genai` (2026.3.1.0)
* `openvino_tokenizers`

**執行 Python 硬體偵測確認：**
```python
import openvino as ov
core = ov.Core()
print(core.available_devices)
# 回傳: ['CPU', 'GPU'] -> 成功識別 Intel(R) Iris(R) Xe Graphics (iGPU)
```
**證明：OpenVINO 底層 Runtime 100% 原生支援本機的 11 代 Iris Xe。**

### 2. 測試「是否能直接共用 LM Studio 的現成 GGUF 模型」
* **出發點**：使用者本機在 `C:\Users\USER\.lmstudio\models\` 已經下載了許多大型模型（包含 `Qwen3.8-27B-Q4_K_M.gguf`、`Bonsai-27B`、`gemma-4` 等共 13 個檔案）。Intel 宣稱 OpenVINO GenAI 支援 GGUF 直接讀取，若能直接共用，便可免去動輒十幾 GB 的重複下載。
* **初次嘗試載入 `Qwen3.8-27B`**：
  呼叫 `ov_genai.LLMPipeline(path, "GPU")` 時拋出異常：
  ```text
  IndexError: invalid unordered_map<K, T> key
  ```

---

## 🕒 第四階段：全自動化相容性掃描與錯誤排查

為避免單一模型特例，撰寫自動化測試腳本 `test_all_gguf.py`，逐一載入本機 LM Studio 內的所有模型：

### 測試輸出日誌彙總：
1. `Bonsai-27B-Q1_0.gguf` 👉 **FAILED**: `invalid unordered_map<K, T> key`
2. `gemma-4-12B-it-QAT-Q4_0.gguf` 👉 **FAILED**: `invalid unordered_map<K, T> key`
3. `gemma-4-26B-A4B-it-QAT-Q4_0.gguf` 👉 **FAILED**: `invalid unordered_map<K, T> key`
4. `gemma-4-E4B-it-Q4_K_M.gguf` 👉 **FAILED**: `Unsupported model architecture 'gemma4'`
5. `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` 👉 **FAILED**: `[load_gguf] gguf_tensor_to_f16 failed`
6. `NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf` 👉 **FAILED**: `[load_gguf] gguf_tensor_to_f16 failed`
7. `Qwen3.8-27B-Q4_K_M.gguf` 👉 **FAILED**: `invalid unordered_map<K, T> key`

### 技術原因深究：
1. **GGUF Reader 尚在 Preview**：OpenVINO 核心的 GGUF 映射表只支援早期標準的 Llama 3.1 和純文字版 Qwen 2.5。
2. **Qwen3.8 是多模態架構 (VLM)**：自帶視覺編碼器，與純文字 `LLMPipeline` 的映射規則衝突，導致查找鍵值失敗。
3. **結論**：OpenVINO 必須吃官方標準的 **IR 格式（`.xml` + `.bin`）**。

---

## 🕒 第五階段：官方 GUI（Intel AI Playground）限制的真相

使用者提出關鍵疑問：
> *「關於 GUI，你剛剛說官方就有開發 GUI，為什麼要自己寫？」*
> *「你看一遍官方說明，然後跟我說能不能裝？」*

### 研讀官方文檔與 GitHub Issue 後的關鍵發現：
Intel 官方確實推出了自家的桌上型軟體 **Intel AI Playground**，但它並非為所有 Intel 硬體設計：
1. **強制性的硬體白名單**：最低要求 Core Ultra 或 Arc 獨顯，排除了第 11 代 CPU。
2. **軟體過度綑綁**：強制綑綁 82GB+ 的 Stable Diffusion 生圖套件。
3. **自己兜程式的必要性**：繞過官方 GUI 外殼限制，直接調用原生的 OpenVINO 核心。

---

## 🕒 第六階段：突破下載限速與 Qwen3.8-27B 實測

### 1. 突破下載瓶頸
* **踩坑**：匿名下載超大檔案（`openvino_language_model.bin` 13.9GB）時，Python 串流緩衝區出現掛死問題。
* **解決**：
  1. 使用者提供 Hugging Face Access Token 取得高速 CDN 權限。
  2. 自行開發 `fast_resumable_downloader.py`，採用 **4MB HTTP Range 分塊斷點續傳機制**，順利完成 14.87GB 全模型下載。

### 2. 基準推論實測突破 (2026-09-16)
* **GPU 載入耗時**：**40.42 秒**（將 14.87GB 權重與圖譜編譯至 Iris Xe 80 EUs）。
* **首字延遲 (TTFT)**：**2.58 秒**。
* **產字速率 (TPS)**：**1.72 tokens/s**（符合 51.2 GB/s 雙通道記憶體架構極限）。
* **記憶體峰值**：**26.18 GB**（本機 64GB RAM 輕鬆承載，零 OOM）。

---

## 🕒 第七階段：對話模板 (ChatML) 演進與 Agent Server 實作

### 1. 文字接龍 vs 對話助理
* **現象**：使用者輸入「你好」，模型卻輸出「，我是負責管理我們社區圖書館的志願者...」。
* **原因**：缺少對話標籤，模型以「文字接龍（Text Completion）」模式續寫散文。
* **對策**：實作 Qwen 規範的 **ChatML 對話模板**（`<|im_start|>system...<|im_end|>`），成功引導模型進入 AI 助理角色。

### 2. 專為 Agent 打造：OpenAI 相容伺服器 (Port 1234)
為了讓 LangChain、AutoGen、CrewAI、Dify 等 Agent 框架直接調用：
* 開發 `openai_server.py`，監聽 `http://127.0.0.1:1234/v1`。
* 實作標準的 **Server-Sent Events (SSE) 串流協定**（發送 `data: {...}\n\n` 與 `data: [DONE]\n\n`）。
* 透過官方 `openai` Python SDK 測試呼叫成功，正式成為可供 Agent 系統無縫調用的本地大腦！

---

## 🕒 第八階段：部署 MoE 模型 Qwen3.6-35B-A3B (2026-09-17)

### 1. 下載
* 官方轉換版 `OpenVINO/Qwen3.6-35B-A3B-int4-ov` 約 19.7 GB (語言模型 18.65 GB)。
* **踩坑**：`huggingface_hub` 預設的 Xet 下載方式卡住不動；設定 `HF_HUB_DISABLE_XET=1` 改走 HTTP 後正常，約 2 小時 40 分完成。

### 2. GPU 放不下，改用 CPU
* 權重比 27B 還小，但 GPU 編譯時超出 Iris Xe 約 29.5 GB 的共享記憶體池 (`Can not allocate 536870912 bytes for USM Device`)。
* 改用 CPU：每個 token 只啟用約 3B 參數，生成約 **4 tokens/s**，是 27B GPU 的兩倍。

### 3. OVMS vs 自製伺服器
* OVMS (CPU) 長文預填充約 6.8 tokens/s，2,500 tokens 首字延遲 368 秒。
* 讓 `openai_server.py` 支援選模型與裝置 (`LLM_MODEL` / `LLM_DEVICE`)，CPU 分段預填充 512 tokens：同長度只要 **196 秒**，快過 27B GPU 的約 250 秒。
* **踩坑**：Qwen3.6 對話模板重新渲染歷史時預設刪除 `<think>` 區塊，與已送入模型的 token 對不上，prefix caching 每輪失效；伺服器改為固定傳 `preserve_thinking=True` 後，多輪後續問題從約 200 秒降到 **約 3 秒**。
* 另修正自製伺服器串流不回傳 `usage`、不讀 `chat_template_kwargs.enable_thinking` 兩個相容性問題。

### 4. CPU + GPU 分工的嘗試 (失敗)
* OVMS 的 `HETERO:GPU,CPU` 自動分配仍把太多層放上 GPU 而記憶體不足。
* 自製伺服器手動指定各層裝置：GPU 編譯時權重膨脹約 10 倍 (1.69 GB 權重佔用 17.3 GB)；狀態節點放 GPU 會讓 GPU plugin 崩潰，放 CPU 又無法編譯。
* **結論**：本機跑此模型的最佳方案是自製伺服器全 CPU 執行。詳細數據見研究筆記第十二節。
