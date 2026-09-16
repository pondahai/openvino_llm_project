import os
import sys
import time
import json
import uuid
import asyncio
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment, FileSystemLoader

import numpy as np
import openvino as ov
import openvino_tokenizers

sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="OpenVINO OpenAI-Compatible Server for Agents")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = r"C:\Users\USER\openvino_llm_project\models\Qwen3.8-27B-int4-ov"
MODEL_NAME = "qwen3.8-27b-int4-ov"

# Qwen3.8 官方停止標記與特殊 Token ID
# 248046: <|im_end|>, 248044: <|endoftext|>, 248045: <|im_start|>
STOP_TOKEN_IDS = {248046, 248044, 248045, 151643, 151645}

print("==========================================================")
print("  啟動 OpenVINO OpenAI-Compatible Server (Port 1234)")
print("  完整支援 Agent 框架 (LangChain/AutoGen/CrewAI/LM Studio)")
print("==========================================================")

# Jinja2 模板
jinja_env = Environment(loader=FileSystemLoader(MODEL_DIR))
chat_template = jinja_env.get_template("chat_template.jinja")

core = ov.Core()

# 關鍵配置：
# 1. 解鎖 Intel GPU 單次大塊記憶體分配限制 (突破 4GB 預設限制，充分發揮 64GB 記憶體能力)
# 2. 降低 GPU 隊列與模型優先級，讓出 Windows 桌面渲染與合成器資源，徹底杜絕螢幕閃爍與 TDR 驅動重設
gpu_config = {
    "GPU_ENABLE_LARGE_ALLOCATIONS": True,
    "GPU_QUEUE_THROTTLE": "LOW",
    "GPU_QUEUE_PRIORITY": "LOW",
    "MODEL_PRIORITY": "LOW"
}
try:
    core.set_property("GPU", gpu_config)
    print("⚡ 已成功開啟大塊記憶體支援並設定 GPU 優先級為 LOW (讓出桌面渲染，防止螢幕閃爍)！")
except Exception as e:
    print(f"⚠️ 設定 GPU 屬性警告: {e}")

print("[1/3] 載入 Tokenizer & Detokenizer...")
tok_m = core.read_model(os.path.join(MODEL_DIR, "openvino_tokenizer.xml"))
detok_m = core.read_model(os.path.join(MODEL_DIR, "openvino_detokenizer.xml"))
c_tok = core.compile_model(tok_m, "CPU")
c_detok = core.compile_model(detok_m, "CPU")

print("[2/3] 載入 Text Embeddings 至 CPU (節省 GPU 顯存與減輕驅動壓力)...")
embed_m = core.read_model(os.path.join(MODEL_DIR, "openvino_text_embeddings_model.xml"))
c_embed = core.compile_model(embed_m, "CPU")

print("[3/3] 載入 27B 語言模型至 Intel Iris Xe GPU...")
lm_m = core.read_model(os.path.join(MODEL_DIR, "openvino_language_model.xml"))
c_lm = core.compile_model(lm_m, "GPU")
# 重複使用固定的 infer_request 物件，每次呼叫前 reset_state()
infer_req = c_lm.create_infer_request()
print("✅ 模型全數載入完畢！API 伺服器就緒！\n")

# Pydantic 數據結構嚴格相容 OpenAI
class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Any]]] = ""
    name: Optional[str] = None
    tool_calls: Optional[List[Any]] = None

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = MODEL_NAME
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=512, alias="max_completion_tokens")
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    enable_thinking: Optional[bool] = False

def render_prompt(req: ChatCompletionRequest) -> str:
    msgs_data = []
    # 為避免 Intel Iris Xe 內顯在 Prefill 階段超過 Windows 2秒 TDR 逾時門檻 (引發螢幕閃爍與 OpenCL -14)
    # 保留 System 提示詞，對話歷史採取滑動窗口機制 (只保留最近 1-2 輪，最多約 3 則訊息)
    system_msgs = [m for m in req.messages if m.role == "system"]
    conversation_msgs = [m for m in req.messages if m.role != "system"]
    
    # 取最近 2 則訊息 (例如上一輪 Assistant 與當前 User)
    truncated_conv = conversation_msgs[-2:] if len(conversation_msgs) > 2 else conversation_msgs
    active_msgs = system_msgs + truncated_conv

    for m in active_msgs:
        item = {"role": m.role, "content": m.content if isinstance(m.content, str) else str(m.content)}
        if m.tool_calls:
            normalized_tool_calls = []
            for tc in m.tool_calls:
                # 兼容 dict 與物件結構
                if isinstance(tc, dict):
                    tc_copy = dict(tc)
                    fn = tc_copy.get("function")
                    if isinstance(fn, dict) and "arguments" in fn and isinstance(fn["arguments"], str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"])
                        except Exception:
                            pass
                    normalized_tool_calls.append(tc_copy)
                else:
                    normalized_tool_calls.append(tc)
            item["tool_calls"] = normalized_tool_calls
        msgs_data.append(item)
        
    return chat_template.render(
        messages=msgs_data,
        tools=req.tools,
        add_generation_prompt=True,
        enable_thinking=req.enable_thinking
    )

def sample_token(logits_vec: np.ndarray, temperature: float = 0.7, top_p: float = 0.9) -> int:
    if temperature is None or temperature <= 0.01:
        return int(np.argmax(logits_vec))
    
    # 溫度縮放
    scaled = logits_vec / max(temperature, 1e-4)
    # 數值穩定性防溢位
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    probs /= np.sum(probs)
    
    # Top-P 核採樣 (Nucleus Sampling)
    if top_p is not None and top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = cumulative_probs > top_p
        if np.any(cutoff):
            first_cutoff = np.argmax(cutoff)
            if first_cutoff > 0:
                sorted_probs[first_cutoff + 1:] = 0.0
                probs = sorted_probs / np.sum(sorted_probs)
                sorted_indices = sorted_indices[:len(sorted_probs)]
                return int(np.random.choice(sorted_indices, p=probs))
                
    return int(np.random.choice(len(probs), p=probs))

@app.get("/v1/models")
@app.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "openvino",
                "permission": [],
                "root": MODEL_NAME,
                "parent": None
            }
        ]
    }

# 兼容 LM Studio / Ollama / LocalAI 的端點
@app.get("/api/v0/models")
@app.get("/api/v0/models/{model_id:path}")
def get_v0_models(model_id: Optional[str] = None):
    return {
        "id": MODEL_NAME,
        "object": "model",
        "name": MODEL_NAME,
        "max_context_length": 262144
    }

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    formatted_prompt = render_prompt(req)
    
    # 輸入編碼
    tok_res = c_tok([formatted_prompt])
    input_ids = tok_res["input_ids"]
    
    # 內顯安全防護：若 Prompt 超過 200 tokens，截取最近 200 tokens，防止 Prefill 超過 2 秒引發 Windows TDR 斷線
    MAX_PROMPT_TOKENS = 200
    if input_ids.shape[1] > MAX_PROMPT_TOKENS:
        input_ids = input_ids[:, -MAX_PROMPT_TOKENS:]
        
    seq_len = input_ids.shape[1]
    
    embed_res = c_embed([input_ids])
    cur_embeds = embed_res[c_embed.outputs[0]]
    
    cur_mask = np.ones((1, seq_len), dtype=np.int64)
    cur_pos = np.zeros((4, 1, seq_len), dtype=np.int64)
    for i in range(4):
        cur_pos[i, 0, :] = np.arange(seq_len, dtype=np.int64)
    b_idx = np.zeros((1,), dtype=np.int32)
    
    infer_req.reset_state()
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())
    max_steps = req.max_tokens if req.max_tokens else 512

    # 使用者自訂 stop 字串處理
    user_stops = []
    if req.stop:
        if isinstance(req.stop, str):
            user_stops = [req.stop]
        elif isinstance(req.stop, list):
            user_stops = req.stop

    # --- 1. 標準 SSE 串流協議 (OpenAI Streaming) ---
    if req.stream:
        async def openai_sse_generator():
            nonlocal cur_embeds, cur_mask, cur_pos, b_idx
            
            # 第一包：建立角色區塊
            first_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": MODEL_NAME,
                "system_fingerprint": "fp_openvino_irisxe",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "logprobs": None,
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

            accumulated_text = ""
            stopped = False
            for step in range(max_steps):
                inputs = {
                    "inputs_embeds": cur_embeds,
                    "attention_mask": cur_mask,
                    "position_ids": cur_pos,
                    "beam_idx": b_idx
                }
                try:
                    infer_req.infer(inputs)
                except Exception as e:
                    # 優雅降級回傳錯誤訊息，防止連線被強行中斷
                    err_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": MODEL_NAME,
                        "system_fingerprint": "fp_openvino_irisxe",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"\n\n[系統錯誤: 推論記憶體超限 - {str(e)}]"},
                            "logprobs": None,
                            "finish_reason": "error"
                        }]
                    }
                    yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
                    stopped = True
                    break

                logits = infer_req.get_output_tensor(0).data
                next_token = sample_token(logits[0, -1, :], req.temperature, req.top_p)

                # 1. 精確比對 Token ID
                if next_token in STOP_TOKEN_IDS:
                    finish_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": MODEL_NAME,
                        "system_fingerprint": "fp_openvino_irisxe",
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "logprobs": None,
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
                    stopped = True
                    break

                decoded_word = c_detok([np.array([[next_token]], dtype=np.int64)])["string_output"][0]

                # 2. 特殊標籤字串過濾
                if any(tag in decoded_word for tag in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]):
                    finish_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": MODEL_NAME,
                        "system_fingerprint": "fp_openvino_irisxe",
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "logprobs": None,
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
                    stopped = True
                    break

                # 3. 使用者自訂 stop 比對
                accumulated_text += decoded_word
                should_stop_user = False
                for s in user_stops:
                    if s in accumulated_text:
                        should_stop_user = True
                        break
                if should_stop_user:
                    finish_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": MODEL_NAME,
                        "system_fingerprint": "fp_openvino_irisxe",
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "logprobs": None,
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
                    stopped = True
                    break

                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": MODEL_NAME,
                    "system_fingerprint": "fp_openvino_irisxe",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": decoded_word},
                        "logprobs": None,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.001)

                next_in_ids = np.array([[next_token]], dtype=np.int64)
                cur_embeds = c_embed([next_in_ids])[c_embed.outputs[0]]
                cur_mask = np.ones((1, cur_mask.shape[1] + 1), dtype=np.int64)
                new_pos = np.zeros((4, 1, 1), dtype=np.int64)
                for i in range(4):
                    new_pos[i, 0, 0] = cur_mask.shape[1] - 1
                cur_pos = new_pos

            if not stopped:
                finish_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": MODEL_NAME,
                    "system_fingerprint": "fp_openvino_irisxe",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "logprobs": None,
                        "finish_reason": "length"
                    }]
                }
                yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(openai_sse_generator(), media_type="text/event-stream")

    # --- 2. 標準非串流 JSON 回應 ---
    else:
        full_text = []
        finish_reason = "length"
        for step in range(max_steps):
            inputs = {
                "inputs_embeds": cur_embeds,
                "attention_mask": cur_mask,
                "position_ids": cur_pos,
                "beam_idx": b_idx
            }
            try:
                infer_req.infer(inputs)
            except Exception as e:
                full_text.append(f"\n[系統錯誤: 推論記憶體超限 - {str(e)}]")
                finish_reason = "error"
                break

            logits = infer_req.get_output_tensor(0).data
            next_token = sample_token(logits[0, -1, :], req.temperature, req.top_p)

            if next_token in STOP_TOKEN_IDS:
                finish_reason = "stop"
                break

            decoded_word = c_detok([np.array([[next_token]], dtype=np.int64)])["string_output"][0]
            if any(tag in decoded_word for tag in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]):
                finish_reason = "stop"
                break

            full_text.append(decoded_word)
            current_so_far = "".join(full_text)
            if any(s in current_so_far for s in user_stops):
                finish_reason = "stop"
                break

            next_in_ids = np.array([[next_token]], dtype=np.int64)
            cur_embeds = c_embed([next_in_ids])[c_embed.outputs[0]]
            cur_mask = np.ones((1, cur_mask.shape[1] + 1), dtype=np.int64)
            new_pos = np.zeros((4, 1, 1), dtype=np.int64)
            for i in range(4):
                new_pos[i, 0, 0] = cur_mask.shape[1] - 1
            cur_pos = new_pos

        content_str = "".join(full_text)
        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": MODEL_NAME,
            "system_fingerprint": "fp_openvino_irisxe",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content_str
                    },
                    "logprobs": None,
                    "finish_reason": finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": seq_len,
                "completion_tokens": len(full_text),
                "total_tokens": seq_len + len(full_text)
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=1234)
