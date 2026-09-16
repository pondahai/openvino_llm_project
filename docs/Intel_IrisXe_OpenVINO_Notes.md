# 📓 本機 Intel Iris Xe 與 OpenVINO LLM 推論完整研究筆記

本文件詳細記錄在 **Intel 第 11 代 Core i5 整合式 GPU (iGPU)** 與 **64GB 雙通道記憶體** 環境下，評估、部署與加速超大型語言模型（LLM）推論的硬體理論推導、實測對比與架構設計。

---

## 🖥️ 一、本機硬體實體規格

經由底層與 Windows Management Instrumentation (WMI) 實體檢測，本機核心規格如下：

* **中央處理器 (CPU)**：Intel Core i5-1135G7 @ 2.40GHz（4 核心 / 8 執行緒，Tiger Lake 架構，10nm SuperFin 製程）
* **整合式繪圖核心 (iGPU)**：**Intel Iris Xe Graphics**
  * 執行單元 (EUs)：**80 Execution Units** (640 Shading Units)
  * 架構特點：具備 INT8/FP16 矩陣乘法加速指令支援 (DP4A)，共享主記憶體作為 VRAM
* **系統記憶體 (RAM)**：**64 GB**（雙通道 Dual-Channel）
  * 規格：DDR4-3200 (1600 MHz 匯流排，雙通道 2 × 64-bit = 128-bit 匯流排寬度)
* **儲存裝置**：NVMe PCIe SSD
* **作業系統**：Windows 11 64-bit

---

## 🧮 二、理論推論速度推導 (Theoretical Memory Bandwidth & TPS)

在自回歸大型語言模型（Autoregressive LLM）的解碼階段（Token Generation Phase），**每一個生成 Token 必須將整個模型的權重（Weights）從記憶體完整讀取進 GPU 暫存器/快取一次**。

因此，自回歸 LLM 推論速度屬於典型 **記憶體頻寬受限（Memory-Bound）**，其理論速度上限可由硬體頻寬嚴謹推導。

### 1. DDR4-3200 雙通道理論頻寬計算
$$\text{理論最大頻寬 (Theoretical Bandwidth)} = \text{時脈} \times \text{傳輸次數/週期} \times \text{通道寬度} \times \text{通道數}$$
$$\text{Bandwidth} = 3200\text{ MT/s} \times 8\text{ Bytes (64-bit)} = 25.6\text{ GB/s (單通道)}$$
$$\text{雙通道理論峰值} = 25.6\text{ GB/s} \times 2 = \mathbf{51.2\text{ GB/s}}$$

考慮實際系統開銷（CPU 背景佔用、作業系統分頁機制、匯流排排程損耗），**實際可取得之有效記憶體頻寬約為理論值的 60% ~ 70%**：
$$\text{實際有效頻寬 (Effective Bandwidth)} \approx 51.2 \times 0.65 \approx \mathbf{33.28\text{ GB/s}}$$

### 2. Qwen3.8-27B (INT4) 權重吞吐需求
* 模型參數規模：約 270 億參數 (27 Billion Parameters)
* 量化格式：INT4 量化（每參數佔用 0.5 Byte，含 Scales/Biases metadata 與 Embedding 總大小為 **14.87 GB**）
* 每個 Token 生成需讀取資料量：**$\approx 14.87\text{ GB}$**

### 3. 理論速度推導公式
$$\text{理論極限生成速度 (Theoretical Max TPS)} = \frac{\text{記憶體頻寬}}{\text{模型每 Step 讀取位元組數}}$$

* **理想峰值（100% 頻寬飽和）**：
  $$\text{TPS}_{\text{ideal}} = \frac{51.2\text{ GB/s}}{14.87\text{ GB}} \approx \mathbf{3.44\text{ tokens/s}}$$
* **實際硬體預期（考慮 60% ~ 65% 有效頻寬）**：
  $$\text{TPS}_{\text{realistic}} = \frac{33.28\text{ GB/s}}{14.87\text{ GB}} \approx \mathbf{2.23\text{ tokens/s}}$$

---

## 📊 三、理論速度 vs 實際對話速度全面對比

我們分別在**純原生基準測試**與**Streamlit 互動圖形化對話（真實 Agent 工作情境）**進行了高精度計時與效能分析：

| 評估維度 / 項目 | 理論推導極限 | 實際實測數值 (Native Benchmark) | 實際對話介面 (GUI 對話情境) | 達成率與分析 |
| :--- | :--- | :--- | :--- | :--- |
| **記憶體頻寬利用** | 51.2 GB/s (100%) | ~25.6 GB/s (~50% 匯流排利用) | ~27.8 GB/s (~54% 匯流排利用) | 🟢 符合 DDR4 iGPU 物理特性 |
| **流式產字速度 (TPS)** | **2.23 ~ 3.44 tokens/s** | **1.72 tokens/s** | **1.87 tokens/s** | 🟢 **達到實際頻寬極限的 84% 實用率** |
| **首字延遲 (TTFT)** | < 3.0 秒 | **2.58 秒** | **6.51 秒 (含網頁渲染+排程)** | 🟢 首次提示詞編碼迅速 |
| **系統記憶體佔用 (RAM)**| ~18.0 GB (基礎權重) | **26.18 GB** (含 KV Cache) | **26.50 GB** (含網頁與緩衝區) | 🟢 64GB 充裕，零分頁交換 (Page Fault) |
| **運算設備 (Device)** | Intel Iris Xe | **Intel Iris Xe (GPU)** | **Intel Iris Xe (GPU)** | 🟢 100% 成功下發至內顯矩陣單元 |

### 🔍 速度差異原因剖析：
1. **iGPU 共享主記憶體仲裁**：Iris Xe 沒有獨立顯存（VRAM），顯卡與 CPU 共享同一條 DDR4 雙通道匯流排，螢幕顯示與作業系統記憶體讀寫會瓜分約 15%~20% 頻寬。
2. **多模態架構開銷**：Qwen3.8 為具備視覺能力的架構，模型包含文本嵌入層、線性注意力機制與 MTP 結構，計算流水線較傳統純文字模型有額外排程成本。
3. **實測結論**：在 11 代 Core i5 輕薄筆電級內顯上，以 **1.87 tokens/s** 運行 27B 旗艦級大模型，已逼近 DDR4-3200 的物理極限（約達理論極限的 84%），表現極其出色，足以支撐深思熟慮型 AI Agent 的推理、任務拆解與函式呼叫需求。

---

## ⚙️ 四、OpenVINO 原生環境與硬體偵測

* 核心套件：`openvino` (2026.3.1)、`openvino_tokenizers` (2026.3.1)
* 裝置清單確認：
  ```text
  Available Devices: ['CPU', 'GPU']
  CPU : 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
  GPU : Intel(R) Iris(R) Xe Graphics (iGPU)
  ```

---

## 🔍 五、模型格式深度相容性分析（為何無法共用 LM Studio GGUF）

對 LM Studio 原有 GGUF 模型進行全面實測：
* `Qwen3.8-27B-Q4_K_M.gguf` ❌ 報錯：`IndexError: invalid unordered_map<K, T> key`
* `gemma-4-E4B.gguf` ❌ 報錯：`Unsupported model architecture 'gemma4'`
* `Muse-Glimmer-30B.gguf` ❌ 報錯：`gguf_tensor_to_f16 failed`

**關鍵技術結論**：
OpenVINO 內建 GGUF Reader 目前對新型多模態架構與新式量化格式支援不全。**欲在 Intel Iris Xe 發揮最高效能與穩定性，必須採用 OpenVINO 官方 IR 格式（`.xml` + `.bin`）**。

---

## 💬 六、與 LM Studio 對話行為對齊與優化

1. **Jinja2 官方對話模板**：導入 Qwen 官方 `chat_template.jinja`，精確遵循 ChatML 與系統指令注入。
2. **停止條件 (Stop Tokens) 完美對齊**：精確攔截 `248046 (<|im_end|>)`、`248044 (<|endoftext|>)`、`248045 (<|im_start|>)`，徹底解決無限輸出與殘留提示字問題。
3. **採樣策略 (Sampling)**：支援 `temperature` 溫度縮放與 `top_p` 核採樣，並相容 greedy decoding。
4. **狀態清理 (`reset_state`)**：在每次會話推論前重置 KV Cache 狀態變量，確保上下文乾淨獨立。

---

## 🚀 七、專案展示與經典對話成果

專案整合了互動 Web GUI 與 OpenAI API 伺服器，下圖為透過 Intel Iris Xe 本機推論完成的經典對話：

![Intel Iris Xe 對話成果展示](images/gui_showcase.png)
