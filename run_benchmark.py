import time
import os
import sys
import psutil
import openvino_genai as ov_genai

MODEL_DIR = r"C:\Users\USER\openvino_llm_project\models\Qwen3.8-27B-int4-ov"

print("==========================================================")
print("  Qwen3.8-27B (INT4) on Intel Iris Xe Benchmark Test")
print("  Device: Intel Iris Xe Graphics (iGPU) + 64GB RAM")
print("==========================================================")

# 1. 測量模型載入時間
print(f"\n[Step 1] 正在將 27B 模型載入至 Intel Iris Xe GPU...")
t0 = time.time()
try:
    pipe = ov_genai.LLMPipeline(MODEL_DIR, "GPU")
    load_time = time.time() - t0
    ram_gb = psutil.Process().memory_info().rss / (1024**3)
    print(f"✅ 模型成功載入至 Iris Xe GPU！")
    print(f"   - 載入耗時: {load_time:.2f} 秒")
    print(f"   - 記憶體佔用: {ram_gb:.2f} GB")
except Exception as e:
    print(f"❌ GPU 載入失敗: {e}")
    sys.exit(1)

# 2. 測試推論生成
print(f"\n[Step 2] 正在執行推論生成測試...")
prompt = "請用繁體中文簡要說明量子電腦與傳統電腦的主要差異。"
print(f"Prompt: {prompt}\n")
print("--- [模型輸出開始] ---")

config = ov_genai.GenerationConfig()
config.max_new_tokens = 128
config.temperature = 0.7

tokens = []
first_token_time = None
start_infer = time.time()

def streamer(subword):
    global first_token_time
    if first_token_time is None:
        first_token_time = time.time()
    tokens.append(subword)
    # 避開特殊 emoji 編碼報錯
    try:
        sys.stdout.write(subword)
        sys.stdout.flush()
    except Exception:
        pass
    return False

pipe.generate(prompt, config, streamer)
total_time = time.time() - start_infer
print("\n--- [模型輸出結束] ---")

# 3. 數據量化與輸出
ttft = (first_token_time - start_infer) if first_token_time else 0
gen_time = total_time - ttft
num_tokens = len(tokens)
tps = num_tokens / gen_time if gen_time > 0 else 0

print(f"\n==========================================================")
print(f"📊 實測基準效能數據報告:")
print(f"  - 首字響應時間 (TTFT): {ttft:.2f} 秒")
print(f"  - 生成 Token 數: {num_tokens} tokens")
print(f"  - 純生成耗時: {gen_time:.2f} 秒")
print(f"  - ⚡ 產字速率 (TPS): {tps:.2f} tokens/秒")
print(f"  - 總端到端耗時: {total_time:.2f} 秒")
print(f"==========================================================")
