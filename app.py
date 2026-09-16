import os
import sys
import time
import streamlit as st
from openai import OpenAI

# 強制 UTF-8 輸出
sys.stdout.reconfigure(encoding='utf-8')

st.set_page_config(
    page_title="Intel Iris Xe - Qwen3.8 27B Chat (Client Mode)",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Intel Iris Xe OpenVINO 對話介面 (Client 模式)")
st.caption("前端：Streamlit Client | 後端：OpenVINO API Server (Port 1234) | 硬體：Intel Iris Xe Graphics (80 EUs) + 64GB RAM")

API_BASE_URL = "http://127.0.0.1:1234/v1"
MODEL_NAME = "qwen3.8-27b-int4-ov"

# 初始化 OpenAI 客戶端 (連線本地 OpenVINO 伺服器)
@st.cache_resource
def get_client():
    return OpenAI(base_url=API_BASE_URL, api_key="not-needed")

client = get_client()

# 側邊欄控制
with st.sidebar:
    st.header("⚙️ 伺服器與模型設置")
    
    # 檢查後端伺服器連線狀態
    server_online = False
    try:
        models = client.models.list()
        server_online = True
        st.success(f"🟢 後端伺服器連線正常 (Port 1234)\n模型: {models.data[0].id}")
    except Exception as e:
        st.error(f"🔴 無法連線至後端伺服器 (http://127.0.0.1:1234/v1)\n請先執行 `run_api_server.bat` 啟動伺服器！")
        
    st.info("💻 運算設備：Intel Iris Xe (已配置 Priority.LOW 保護螢幕不閃爍)")
    
    max_tokens = st.slider("最大輸出長度 (Max Tokens)", 64, 384, 160, step=32)
    history_turns = st.slider("歷史對話保留輪數", 1, 10, 3, step=1, help="限制送入 GPU 的對話輪數，避免超長歷史導致內顯 TDR 驅動重置或顯存超限")
    temperature = st.slider("溫度 (Temperature)", 0.0, 1.5, 0.7, step=0.1)
    top_p = st.slider("Top P", 0.1, 1.0, 0.9, step=0.05)
    enable_thinking = st.checkbox("啟用深層思考 (<think> 模式)", value=False)
    system_prompt = st.text_area(
        "系統指令 (System Prompt)",
        value="你是由阿里巴巴開發的 Qwen 人工智慧助手。請一律使用繁體中文，友善、準確、清晰地回答使用者的問題。",
        height=90
    )
    
    st.markdown("---")
    st.markdown("""
    **前後端分離優勢：**
    * 記憶體單一實例：僅佔用約 26 GB（不重複載入）
    * Agent 與 GUI 共用同一個後台
    * 標準 OpenAI API 串流通訊
    * GPU 優先級自動調節，保護桌面流暢
    """)
    
    if st.button("🧹 清空對話記錄"):
        st.session_state.messages = []
        st.rerun()

# 初始化對話歷史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染歷史對話
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 處理用戶輸入
if user_prompt := st.chat_input("請輸入訊息，透過 OpenVINO Server 開始對話..."):
    if not server_online:
        st.error("伺服器未連線，請先啟動 `run_api_server.bat`！")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        status_box = st.empty()
        status_box.info("⚡ 透過 OpenAI API 串流要求 OpenVINO Server 生成中...")
        
        try:
            # 滑動視窗截取最近 N 輪對話（避免過多歷史造成 GPU TDR 閃爍與崩潰）
            recent_msgs = st.session_state.messages[-(history_turns * 2):]
            req_messages = [{"role": "system", "content": system_prompt}]
            for m in recent_msgs:
                req_messages.append({"role": m["role"], "content": m["content"]})
            
            start_t = time.time()
            metrics = {"first_token_time": None, "count": 0}
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=req_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                extra_body={"enable_thinking": enable_thinking}
            )
            
            status_box.empty()

            def sse_stream_generator():
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        if metrics["first_token_time"] is None:
                            metrics["first_token_time"] = time.time()
                        metrics["count"] += 1
                        yield content

            final_content = st.write_stream(sse_stream_generator())
            
            elapsed = time.time() - start_t
            ttft = metrics["first_token_time"] - start_t if metrics["first_token_time"] else 0
            gen_time = elapsed - ttft
            tps = metrics["count"] / gen_time if gen_time > 0 else 0
            
            st.caption(f"⚡ 首字延遲 (TTFT): {ttft:.2f}s | 純生成耗時: {gen_time:.2f}s | 速度: {tps:.2f} tokens/s | 架構: Client -> OpenAI API Server -> Iris Xe")
            st.session_state.messages.append({"role": "assistant", "content": final_content})
            
        except Exception as e:
            status_box.empty()
            st.error(f"❌ 伺服器通訊錯誤: {str(e)}")
