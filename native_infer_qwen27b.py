import os
import sys
import time
import numpy as np
import openvino as ov
import openvino_tokenizers

# 強制設定 UTF-8 輸出避免 Windows cp950 編碼報錯
sys.stdout.reconfigure(encoding='utf-8')

MODEL_DIR = r"C:\Users\USER\openvino_llm_project\models\Qwen3.8-27B-int4-ov"

print("==========================================================")
print("  Qwen3.8-27B Native OpenVINO Inference Test on Iris Xe")
print("  Hardware: Intel Iris Xe Graphics (iGPU) + 64GB RAM")
print("==========================================================")

core = ov.Core()

# 1. 載入 Tokenizer 與 Detokenizer
print("[Step 1] 正在載入 Tokenizer / Detokenizer...")
tok_model = core.read_model(os.path.join(MODEL_DIR, "openvino_tokenizer.xml"))
detok_model = core.read_model(os.path.join(MODEL_DIR, "openvino_detokenizer.xml"))
compiled_tok = core.compile_model(tok_model, "CPU")
compiled_detok = core.compile_model(detok_model, "CPU")
print("[Step 1 完成] Tokenizer 載入就緒！")

# 2. 載入文字 Embeddings 模型
print("[Step 2] 正在載入 Text Embeddings 模組至 Iris Xe GPU...")
embed_model = core.read_model(os.path.join(MODEL_DIR, "openvino_text_embeddings_model.xml"))
compiled_embed = core.compile_model(embed_model, "GPU")
print("[Step 2 完成] Embeddings 模組載入就緒！")

# 3. 載入主語言模型核心至 Iris Xe GPU
print("[Step 3] 正在載入 27B 語言模型至 Intel Iris Xe GPU (約需 20~30 秒)...")
t0 = time.time()
lm_model = core.read_model(os.path.join(MODEL_DIR, "openvino_language_model.xml"))
compiled_lm = core.compile_model(lm_model, "GPU")
load_time = time.time() - t0
print(f"[Step 3 完成] 27B 語言模型核心成功載入至 Intel Iris Xe GPU！耗時: {load_time:.2f} 秒")

# 4. 準備對話測試
prompt = "請用繁體中文簡要介紹量子運算的基本原理。"
print(f"\n[Step 4] 開始推論生成...\nPrompt: {prompt}\n")

# Tokenize
tok_res = compiled_tok([prompt])
input_ids = tok_res["input_ids"] # shape: [1, seq_len]
seq_len = input_ids.shape[1]

# Embeddings
embed_res = compiled_embed([input_ids])
inputs_embeds = embed_res[compiled_embed.outputs[0]] # shape: [1, seq_len, 5120]

# 初始 Attention mask 與 Position IDs
attention_mask = np.ones((1, seq_len), dtype=np.int64)
# Qwen position_ids shape: [4, 1, seq_len]
position_ids = np.zeros((4, 1, seq_len), dtype=np.int64)
for i in range(4):
    position_ids[i, 0, :] = np.arange(seq_len, dtype=np.int64)
beam_idx = np.zeros((1,), dtype=np.int32)

print("--- [模型生成輸出開始] ---")
generated_tokens = []
start_infer_time = time.time()
first_token_time = None

infer_request = compiled_lm.create_infer_request()

for step in range(64):
    inputs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "beam_idx": beam_idx
    }
    infer_request.infer(inputs)
    logits = infer_request.get_output_tensor(0).data # [1, seq_len, vocab_size]
    next_token_id = int(np.argmax(logits[0, -1, :]))
    
    if first_token_time is None:
        first_token_time = time.time()
        
    generated_tokens.append(next_token_id)
    
    # 逐字解碼輸出
    decoded_word = compiled_detok([np.array([[next_token_id]], dtype=np.int64)])["string_output"][0]
    sys.stdout.write(decoded_word)
    sys.stdout.flush()
    
    if next_token_id in [151643, 151645]: # EOS tokens
        break
        
    # 準備下一步輸入
    next_input_ids = np.array([[next_token_id]], dtype=np.int64)
    inputs_embeds = compiled_embed([next_input_ids])[compiled_embed.outputs[0]]
    attention_mask = np.ones((1, attention_mask.shape[1] + 1), dtype=np.int64)
    new_pos = np.zeros((4, 1, 1), dtype=np.int64)
    for i in range(4):
        new_pos[i, 0, 0] = attention_mask.shape[1] - 1
    position_ids = new_pos

print("\n--- [模型生成輸出結束] ---")

total_infer_time = time.time() - start_infer_time
ttft = first_token_time - start_infer_time if first_token_time else 0
gen_time = total_infer_time - ttft
tps = len(generated_tokens) / gen_time if gen_time > 0 else 0

print(f"\n==========================================================")
print(f"實測推論基準指標:")
print(f"  - 首字延遲 (TTFT): {ttft:.2f} 秒")
print(f"  - 產出 Token 數: {len(generated_tokens)} tokens")
print(f"  - 純生成耗時: {gen_time:.2f} 秒")
print(f"  - 產字速率 (TPS): {tps:.2f} tokens/s")
print(f"  - 總端到端耗時: {total_infer_time:.2f} 秒")
print(f"==========================================================")
