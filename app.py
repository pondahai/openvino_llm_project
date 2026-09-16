import os
import sys
import time
import streamlit as st
import numpy as np
import openvino as ov
import openvino_tokenizers
from jinja2 import Environment, FileSystemLoader

# 強制 UTF-8 輸出
sys.stdout.reconfigure(encoding='utf-8')

st.set_page_config(
    page_title="Intel Iris Xe - Qwen3.8 27B Chat",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Intel Iris Xe OpenVINO 對話介面")
st.caption("硬體：Intel 11th Gen Core i5-1135G7 + Iris Xe Graphics (80 EUs) + 64GB DDR4-3200 | 核心模型：Qwen3.8-27B-int4-ov (Instruct)")

MODEL_DIR = r"C:\Users\USER\openvino_llm_project\models\Qwen3.8-27B-int4-ov"

# Qwen3.8 精確停止 Token ID (248046: <|im_end|>, 248044: <|endoftext|>, 248045: <|im_start|>)
STOP_TOKEN_IDS = {248046, 248044, 248045, 151643, 151645}

# 側邊欄控制
with st.sidebar:
    st.header("⚙️ 硬體與模型設置")
    st.success("🟢 核心已就緒：Qwen3.8-27B (INT4)")
    st.info("💻 運算設備：Intel Iris Xe (GPU)")
    
    device = st.selectbox("執行設備 (Device)", ["GPU", "CPU"], index=0)
    max_tokens = st.slider("最大輸出長度 (Max Tokens)", 64, 512, 180, step=32)
    enable_thinking = st.checkbox("啟用深層思考 (<think> 模式)", value=False, help="關閉時將直接輸出回答，與一般對話一致；開啟時會展現逐步推演過程")
    system_prompt = st.text_area(
        "系統指令 (System Prompt)",
        value="你是由阿里巴巴開發的 Qwen 人工智慧助手。請一律使用繁體中文，友善、準確、清晰地回答使用者的問題。",
        height=90
    )
    
    st.markdown("---")
    st.markdown("""
    **實測效能指標：**
    * 模型大小：14.87 GB
    * 首次載入：~40 秒 (GPU 編譯)
    * 首字延遲 (TTFT)：~2.58 秒
    * 流式產字速度：~1.72 tokens/s
    * 記憶體佔用：~26 GB (64GB 充裕)
    """)
    
    if st.button("🧹 清空對話記錄"):
        st.session_state.messages = []
        st.rerun()

# 載入 Jinja2 模板環境
@st.cache_resource
def get_jinja_template():
    env = Environment(loader=FileSystemLoader(MODEL_DIR))
    return env.get_template("chat_template.jinja")

# 載入 OpenVINO 模型管道 (全域快取)
@st.cache_resource(show_spinner=False)
def load_models(target_device):
    core = ov.Core()
    
    # 1. Tokenizer & Detokenizer
    tok_m = core.read_model(os.path.join(MODEL_DIR, "openvino_tokenizer.xml"))
    detok_m = core.read_model(os.path.join(MODEL_DIR, "openvino_detokenizer.xml"))
    c_tok = core.compile_model(tok_m, "CPU")
    c_detok = core.compile_model(detok_m, "CPU")
    
    # 2. Text Embeddings (執行於 GPU)
    embed_m = core.read_model(os.path.join(MODEL_DIR, "openvino_text_embeddings_model.xml"))
    c_embed = core.compile_model(embed_m, target_device)
    
    # 3. 主 27B 語言模型 (執行於 GPU)
    lm_m = core.read_model(os.path.join(MODEL_DIR, "openvino_language_model.xml"))
    c_lm = core.compile_model(lm_m, target_device)
    
    return c_tok, c_detok, c_embed, c_lm

# 初始化對話歷史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染歷史對話
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 處理用戶輸入
if user_prompt := st.chat_input("請輸入訊息，與 Qwen3.8-27B 開始對話..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        status_box = st.empty()
        status_box.info(f"⚡ 正在透過 Intel Iris Xe ({device}) 處理中，請稍候...")
        
        try:
            c_tok, c_detok, c_embed, c_lm = load_models(device)
            template = get_jinja_template()
            status_box.empty()
            
            # 準備對話歷史送入 Jinja 官方模板
            jinja_msgs = [{"role": "system", "content": system_prompt}]
            for m in st.session_state.messages:
                jinja_msgs.append({"role": m["role"], "content": m["content"]})
            
            formatted_prompt = template.render(
                messages=jinja_msgs,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            
            tok_res = c_tok([formatted_prompt])
            input_ids = tok_res["input_ids"]
            seq_len = input_ids.shape[1]
            
            embed_res = c_embed([input_ids])
            cur_embeds = embed_res[c_embed.outputs[0]]
            
            cur_mask = np.ones((1, seq_len), dtype=np.int64)
            cur_pos = np.zeros((4, 1, seq_len), dtype=np.int64)
            for i in range(4):
                cur_pos[i, 0, :] = np.arange(seq_len, dtype=np.int64)
            b_idx = np.zeros((1,), dtype=np.int32)
            
            infer_req = c_lm.create_infer_request()
            infer_req.reset_state()
            start_t = time.time()
            metrics = {"first_token_time": None, "count": 0}
            
            # 定義即時流式生成器
            def token_generator(inputs_embeds, attention_mask, position_ids, beam_idx):
                for step in range(max_tokens):
                    inputs = {
                        "inputs_embeds": inputs_embeds,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "beam_idx": beam_idx
                    }
                    infer_req.infer(inputs)
                    logits = infer_req.get_output_tensor(0).data
                    next_token = int(np.argmax(logits[0, -1, :]))
                    
                    if metrics["first_token_time"] is None:
                        metrics["first_token_time"] = time.time()
                    
                    # 1. 精確比對 Token ID 是否為停止符號 (<|im_end|>, <|endoftext|>, <|im_start|>)
                    if next_token in STOP_TOKEN_IDS:
                        break
                    
                    # 2. 解碼字串
                    decoded = c_detok([np.array([[next_token]], dtype=np.int64)])["string_output"][0]
                    
                    # 3. 雙重保險：字串層級檢查是否出現特殊標籤或換角標籤
                    if any(tag in decoded for tag in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]):
                        break
                    
                    metrics["count"] += 1
                    yield decoded
                        
                    next_in_ids = np.array([[next_token]], dtype=np.int64)
                    inputs_embeds = c_embed([next_in_ids])[c_embed.outputs[0]]
                    attention_mask = np.ones((1, attention_mask.shape[1] + 1), dtype=np.int64)
                    new_pos = np.zeros((4, 1, 1), dtype=np.int64)
                    for i in range(4):
                        new_pos[i, 0, 0] = attention_mask.shape[1] - 1
                    position_ids = new_pos

            # 使用 Streamlit 官方原生 st.write_stream 實現流暢打字機效果
            final_content = st.write_stream(token_generator(cur_embeds, cur_mask, cur_pos, b_idx))
            
            elapsed = time.time() - start_t
            ttft = metrics["first_token_time"] - start_t if metrics["first_token_time"] else 0
            gen_time = elapsed - ttft
            tps = metrics["count"] / gen_time if gen_time > 0 else 0
            
            st.caption(f"⚡ 首字延遲 (TTFT): {ttft:.2f}s | 純生成耗時: {gen_time:.2f}s | 速度: {tps:.2f} tokens/s | 硬體: Intel Iris Xe")
            st.session_state.messages.append({"role": "assistant", "content": final_content})
            
        except Exception as e:
            status_box.empty()
            st.error(f"❌ 推論出錯: {str(e)}")
