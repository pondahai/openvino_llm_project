import sys
import json
from openai import OpenAI

sys.stdout.reconfigure(encoding='utf-8')

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")

print("測試修復後的 tool_calls 與 Agent 對話流程...")
messages = [
    {"role": "user", "content": "台北現在天氣如何？"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "Taipei"})
                }
            }
        ]
    },
    {
        "role": "tool",
        "content": json.dumps({"city": "Taipei", "temperature": 26, "condition": "Sunny"})
    }
]

resp = client.chat.completions.create(
    model="qwen3.8-27b-int4-ov",
    messages=messages,
    max_tokens=50,
    stream=False
)

print("測試成功！Server 回應:")
print(resp.choices[0].message.content)
