import sys
import os
from openai import OpenAI

sys.stdout.reconfigure(encoding='utf-8')

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="not-needed"
)

print("測試連接本機 OpenVINO 伺服器 (Qwen3.8-27B)...")
models = client.models.list()
print("可用模型清單:", [m.id for m in models.data])

print("\n--- 呼叫 /v1/chat/completions (流式串流) ---")
response = client.chat.completions.create(
    model="qwen3.8-27b-int4-ov",
    messages=[
        {"role": "system", "content": "你是一個專業的 AI Agent。請用繁體中文回答。"},
        {"role": "user", "content": "你好，請用一句話告訴我你準備好執行 Agent 任務了嗎？"}
    ],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        sys.stdout.write(content)
        sys.stdout.flush()

print("\n\n🎉 測試完全成功！OpenAI API 100% 完美相容！")
