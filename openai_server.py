import os
import sys
import time
import json
import uuid
import asyncio
import re
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, AliasChoices, model_validator
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
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

# 模型選擇 (LLM_MODEL)：(資料夾, API 名稱, 預設裝置)
# Qwen3.6-35B-A3B 在 GPU 編譯時超出 Iris Xe 共享記憶體池，預設用 CPU
MODELS = {
    "qwen3.8-27b": ("Qwen3.8-27B-int4-ov", "qwen3.8-27b-int4-ov", "GPU"),
    "qwen3.6-35b-a3b": ("Qwen3.6-35B-A3B-int4-ov", "qwen3.6-35b-a3b-int4-ov", "CPU"),
}
MODEL_KEY = os.environ.get("LLM_MODEL", "qwen3.8-27b")
if MODEL_KEY not in MODELS:
    sys.exit(f"LLM_MODEL 必須是 {list(MODELS)} 之一，目前為 {MODEL_KEY!r}")
_model_folder, MODEL_NAME, _default_device = MODELS[MODEL_KEY]
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", _model_folder)
LM_DEVICE = os.environ.get("LLM_DEVICE", _default_device).upper()

# Qwen3.8 / Qwen3.6 官方停止標記與特殊 Token ID (兩者 tokenizer 相同)
# 248046: <|im_end|>, 248044: <|endoftext|>, 248045: <|im_start|>
# 上下文上限 (prompt + 生成)：27B 為 16 層 full attention × 4 KV heads × 256 dim，16k tokens 的 KV cache 約 2GB
# (35B-A3B 為 10 層 × 2 KV heads，約 0.6GB)
MAX_CONTEXT_TOKENS = 16384
# 分段預填充：每段 token 數。GPU 需確保單次推論遠低於 Windows TDR 2 秒門檻；CPU 無此限制，用較大分段
PREFILL_CHUNK_TOKENS = int(os.environ.get("PREFILL_CHUNK_TOKENS", "128" if LM_DEVICE == "GPU" else "512"))
# Prefix caching：多輪對話沿用上一個請求的模型狀態，只預填充新增的 token (PREFIX_CACHE=0 可關閉)
ENABLE_PREFIX_CACHE = os.environ.get("PREFIX_CACHE", "1") != "0"

STOP_TOKEN_IDS = {248046, 248044, 248045, 151643, 151645}

print("==========================================================")
print("  啟動 OpenVINO OpenAI-Compatible Server (Port 1234)")
print("  完整支援 Agent 框架 (LangChain/AutoGen/CrewAI/LM Studio)")
print("==========================================================")

# Jinja2 模板與自定義例外輔助函式
def _jinja_raise_exception(msg: str):
    raise ValueError(msg)

jinja_env = Environment(loader=FileSystemLoader(MODEL_DIR))
jinja_env.globals["raise_exception"] = _jinja_raise_exception
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
    print(f"⚠️ 設定 GPU 屬性時發生提示: {e}")

print("\n[1/3] 載入 Tokenizer & Detokenizer...")
c_tok = core.compile_model(os.path.join(MODEL_DIR, "openvino_tokenizer.xml"), "CPU")
c_detok = core.compile_model(os.path.join(MODEL_DIR, "openvino_detokenizer.xml"), "CPU")

print("[2/3] 載入 Text Embeddings 至 CPU (節省 GPU 顯存與輕量化預填充)...")
c_embed = core.compile_model(os.path.join(MODEL_DIR, "openvino_text_embeddings_model.xml"), "CPU")

print(f"[3/3] 載入語言模型 {_model_folder} 至 {LM_DEVICE}...")
c_lm = core.compile_model(os.path.join(MODEL_DIR, "openvino_language_model.xml"), LM_DEVICE)
infer_req = c_lm.create_infer_request()
# 全域只有一個 infer_req (KV cache 狀態)，同一時間只能服務一個請求
infer_lock = asyncio.Lock()
# 目前模型狀態 (KV cache 與 linear attention state) 所對應的 token 序列
kv_cache_ids: List[int] = []
print("🎉 模型參數載入完畢！API 伺服器就緒。\n")

# Pydantic 數據結構嚴格相容 OpenAI
class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Any]]] = ""
    name: Optional[str] = None
    tool_calls: Optional[List[Any]] = None

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = MODEL_NAME
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=512, validation_alias=AliasChoices("max_tokens", "max_completion_tokens"))
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    # {"include_usage": true}：串流結束前多送一個只含 usage 的 chunk
    stream_options: Optional[Dict[str, Any]] = None
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    # 思考模式：最上層 enable_thinking 優先，其次 chat_template_kwargs.enable_thinking (OVMS / vLLM 慣用)，預設關閉
    enable_thinking: Optional[bool] = None
    # 其餘鍵值 (如 reasoning_effort) 直接傳給對話模板
    chat_template_kwargs: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _resolve_enable_thinking(self):
        if self.enable_thinking is None:
            self.enable_thinking = bool((self.chat_template_kwargs or {}).get("enable_thinking", False))
        return self

def has_tool_context(req: ChatCompletionRequest) -> bool:
    return bool(req.tools or any(m.role == "tool" or m.tool_calls for m in req.messages))

def render_prompt(req: ChatCompletionRequest, messages: List[ChatMessage]) -> str:
    # 上下文長度由 build_input_ids 依 token 數裁剪，搭配分段預填充避免 TDR
    msgs_data = []
    for m in messages:
        item = {"role": m.role, "content": m.content if isinstance(m.content, str) else str(m.content)}
        if m.tool_calls:
            normalized_tool_calls = []
            for tc in m.tool_calls:
                # 兼容 dict 與物件結構
                if isinstance(tc, dict):
                    tc_copy = dict(tc)
                    fn = tc_copy.get("function")
                    if isinstance(fn, dict):
                        fn = tc_copy["function"] = dict(fn)
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
        
    template_kwargs = dict(req.chat_template_kwargs or {})
    template_kwargs.update(
        messages=msgs_data,
        tools=req.tools,
        add_generation_prompt=True,
        enable_thinking=req.enable_thinking,
        # 保留歷史回答的 <think> 區塊，使重新渲染的歷史與已送入模型的 token 一致 (prefix caching 才能沿用)
        # Qwen3.8 模板預設即保留；Qwen3.6 模板預設會刪除，需明確開啟 (固定開啟，不接受客戶端覆寫)
        preserve_thinking=True
    )
    return chat_template.render(**template_kwargs)

def build_input_ids(req: ChatCompletionRequest, max_tokens: int) -> np.ndarray:
    # 以「訊息」為單位從最舊的對話開始丟棄，保留 system 與 tools 定義，
    # 避免直接截斷 token 把 chat template 開頭 (system / tools / <|im_start|>) 切壞
    system_msgs = [m for m in req.messages if m.role == "system"]
    conv = [m for m in req.messages if m.role != "system"]
    while True:
        input_ids = c_tok([render_prompt(req, system_msgs + conv)])["input_ids"]
        if input_ids.shape[1] <= max_tokens or len(conv) <= 1:
            break
        # 丟掉最舊的一則，並持續丟到以 user 訊息開頭，避免留下失去對應 tool_calls 的 tool 回應
        conv = conv[1:]
        while len(conv) > 1 and conv[0].role != "user":
            conv = conv[1:]
    if input_ids.shape[1] > max_tokens:
        # 僅剩最後一則仍超長：保留開頭 (system/tools) 與結尾 (最新內容與生成提示)
        head = max_tokens // 2
        print(f"⚠️ Prompt 仍超過上限 ({input_ids.shape[1]} > {max_tokens})，截除中段 token")
        input_ids = np.concatenate([input_ids[:, :head], input_ids[:, -(max_tokens - head):]], axis=1)
    return input_ids

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
            first_cutoff = int(np.argmax(cutoff))
            kept_probs = sorted_probs[:first_cutoff + 1]
            kept_probs = kept_probs / np.sum(kept_probs)
            return int(np.random.choice(sorted_indices[:first_cutoff + 1], p=kept_probs))
                
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
        "max_context_length": MAX_CONTEXT_TOKENS
    }

SPECIAL_TAGS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)

def decode_ids(ids: List[int]) -> str:
    if not ids:
        return ""
    return str(c_detok([np.array([ids], dtype=np.int64)])["string_output"][0])

class IncrementalDecoder:
    """以視窗重新解碼新 token，避免中文 UTF-8 位元組被拆在兩個 token 時輸出 �"""
    def __init__(self):
        self.ids: List[int] = []
        self.prefix_offset = 0
        self.read_offset = 0

    def add(self, token_id: int) -> str:
        self.ids.append(token_id)
        prefix_text = decode_ids(self.ids[self.prefix_offset:self.read_offset])
        new_text = decode_ids(self.ids[self.prefix_offset:])
        if new_text.endswith("�") or len(new_text) <= len(prefix_text):
            # 字元尚未完整 (或為不輸出的特殊 token)，先暫緩
            return ""
        self.prefix_offset = self.read_offset
        self.read_offset = len(self.ids)
        return new_text[len(prefix_text):]

    def flush(self) -> str:
        prefix_text = decode_ids(self.ids[self.prefix_offset:self.read_offset])
        new_text = decode_ids(self.ids[self.prefix_offset:])
        self.prefix_offset = self.read_offset = len(self.ids)
        # 生成結束時仍不完整的字元直接捨棄
        return new_text[len(prefix_text):].rstrip("�")

def _coerce_param(value: str, param_type: Optional[str]) -> Any:
    if param_type == "string":
        return value
    try:
        return json.loads(value)
    except Exception:
        return value

def parse_tool_calls(text: str, tools: Optional[List[Dict[str, Any]]]):
    """解析 Qwen XML 格式的 <tool_call>，回傳 (前置文字, OpenAI 格式 tool_calls)"""
    matches = list(TOOL_CALL_RE.finditer(text))
    if not matches:
        return text, []
    schemas = {}
    for t in tools or []:
        fn = t.get("function", t)
        if isinstance(fn, dict) and "name" in fn:
            schemas[fn["name"]] = (fn.get("parameters") or {}).get("properties") or {}
    calls = []
    for m in matches:
        name = m.group(1)
        props = schemas.get(name, {})
        args = {}
        for pm in PARAM_RE.finditer(m.group(2)):
            key = pm.group(1)
            args[key] = _coerce_param(pm.group(2), (props.get(key) or {}).get("type"))
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}
        })
    return text[:matches[0].start()].strip(), calls

def partial_tag_len(text: str, tag: str) -> int:
    # text 結尾可能是 tag 前綴的最長長度，這段需暫緩輸出
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0

def common_prefix_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def generate_text(input_ids: np.ndarray, max_steps: int, temperature, top_p, user_stops: List[str], stats: Dict[str, int]):
    """同步產生器 (呼叫端須持有 infer_lock，結束後必須 close())：逐步 yield (文字片段, finish_reason 或 None)"""
    global kv_cache_ids
    ids = input_ids[0].tolist()
    seq_len = len(ids)
    b_idx = np.zeros((1,), dtype=np.int32)

    # Prefix caching：linear attention 狀態無法回退，只有上次處理過的完整序列是本次 prompt 的開頭時才能沿用；
    # 至少需留 1 個新 token 以取得 logits
    reuse = common_prefix_len(kv_cache_ids, ids) if ENABLE_PREFIX_CACHE else 0
    if reuse != len(kv_cache_ids) or reuse >= seq_len:
        reuse = 0
    stats["cached_tokens"] = reuse
    kv_cache_ids = []          # 狀態即將變動，完成前先視為無效
    fed: List[int] = ids[:reuse]   # 已送入模型 (存在於 KV cache / linear state) 的 token

    try:
        # 分段預填充：除最後一段外逐段送入 GPU 累積狀態，最後一段留給生成迴圈的第一次 infer 以取得 logits
        try:
            if reuse == 0:
                infer_req.reset_state()
            last_start = reuse + ((seq_len - reuse - 1) // PREFILL_CHUNK_TOKENS) * PREFILL_CHUNK_TOKENS
            for start in range(reuse, seq_len, PREFILL_CHUNK_TOKENS):
                end = min(start + PREFILL_CHUNK_TOKENS, seq_len)
                cur_embeds = c_embed([input_ids[:, start:end]])[c_embed.outputs[0]]
                cur_mask = np.ones((1, end), dtype=np.int64)
                cur_pos = np.tile(np.arange(start, end, dtype=np.int64), (4, 1, 1))
                pending_ids = ids[start:end]
                if start == last_start:
                    break
                infer_req.infer({
                    "inputs_embeds": cur_embeds,
                    "attention_mask": cur_mask,
                    "position_ids": cur_pos,
                    "beam_idx": b_idx
                })
                fed += pending_ids
        except Exception as e:
            fed = []
            yield f"\n\n[系統錯誤: 預填充失敗 - {e}]", "error"
            return
        if reuse:
            print(f"♻️ Prefix cache 命中: 沿用 {reuse} tokens，新增預填充 {seq_len - reuse} tokens")
        elif seq_len > PREFILL_CHUNK_TOKENS:
            print(f"🧩 分段預填充完成: {seq_len} tokens / 每段 {PREFILL_CHUNK_TOKENS}")

        decoder = IncrementalDecoder()
        stop_strings = list(SPECIAL_TAGS) + user_stops
        text = ""
        emitted = 0
        for _ in range(max_steps):
            try:
                infer_req.infer({
                    "inputs_embeds": cur_embeds,
                    "attention_mask": cur_mask,
                    "position_ids": cur_pos,
                    "beam_idx": b_idx
                })
            except Exception as e:
                fed = []
                yield f"\n\n[系統錯誤: 推論記憶體超限 - {e}]", "error"
                return
            fed += pending_ids

            logits = infer_req.get_output_tensor(0).data
            next_token = sample_token(logits[0, -1, :], temperature, top_p)
            if next_token in STOP_TOKEN_IDS:
                yield text[emitted:] + decoder.flush(), "stop"
                return
            stats["completion_tokens"] += 1

            text += decoder.add(next_token)
            # 特殊標籤與使用者 stop 字串：截在最早出現的位置
            hits = [i for i in (text.find(t) for t in stop_strings) if i >= 0]
            if hits:
                yield text[emitted:max(min(hits), emitted)], "stop"
                return
            # 結尾可能是跨 token 的 stop 字串開頭，先暫緩輸出
            safe = len(text) - max(partial_tag_len(text, t) for t in stop_strings)
            if safe > emitted:
                yield text[emitted:safe], None
                emitted = safe
            else:
                yield "", None

            cur_embeds = c_embed([np.array([[next_token]], dtype=np.int64)])[c_embed.outputs[0]]
            cur_mask = np.ones((1, cur_mask.shape[1] + 1), dtype=np.int64)
            cur_pos = np.full((4, 1, 1), cur_mask.shape[1] - 1, dtype=np.int64)
            pending_ids = [next_token]

        yield text[emitted:] + decoder.flush(), "length"
    finally:
        # 記錄模型狀態實際包含的 token，供下一個請求比對
        kv_cache_ids = fed

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    has_tools = has_tool_context(req)
    max_steps = min(req.max_tokens or 512, MAX_CONTEXT_TOKENS // 2)
    # prompt 與生成共用 MAX_CONTEXT_TOKENS 的 KV cache 空間
    input_ids = build_input_ids(req, MAX_CONTEXT_TOKENS - max_steps)

    seq_len = input_ids.shape[1]
    print(f"📥 接收請求: Prompt Token 數 = {seq_len} | Tools 啟用: {has_tools}")

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())
    user_stops = [req.stop] if isinstance(req.stop, str) else list(req.stop or [])
    if not req.enable_thinking:
        # 思考模式關閉時 prompt 已帶空的 <think></think>；模型在工具結果後偶爾仍會輸出 </think> 並重複回答，視為結束
        user_stops.append("</think>")
    parse_tools = bool(req.tools)

    def run_generation():
        stats = {"completion_tokens": 0, "cached_tokens": 0}
        return generate_text(input_ids, max_steps, req.temperature, req.top_p, user_stops, stats), stats

    def make_usage(stats: Dict[str, int]) -> Dict[str, Any]:
        return {
            "prompt_tokens": seq_len,
            "completion_tokens": stats["completion_tokens"],
            "total_tokens": seq_len + stats["completion_tokens"],
            "prompt_tokens_details": {"cached_tokens": stats["cached_tokens"]}
        }

    # --- 1. 標準 SSE 串流協議 (OpenAI Streaming) ---
    if req.stream:
        include_usage = bool((req.stream_options or {}).get("include_usage"))

        def make_chunk(delta: Optional[Dict[str, Any]], finish_reason: Optional[str] = None,
                       usage: Optional[Dict[str, Any]] = None) -> str:
            chunk: Dict[str, Any] = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": MODEL_NAME,
                "system_fingerprint": "fp_openvino_irisxe",
                # delta 為 None 時是 include_usage 的最後一個 chunk，choices 依規格為空陣列
                "choices": [] if delta is None else [{
                    "index": 0,
                    "delta": delta,
                    "logprobs": None,
                    "finish_reason": finish_reason
                }]
            }
            if include_usage:
                chunk["usage"] = usage
            return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        async def openai_sse_generator():
            async with infer_lock:
                yield make_chunk({"role": "assistant", "content": ""})
                gen, stats = run_generation()
                full_text = ""
                pending = ""       # 暫緩送出、可能是 <tool_call> 開頭的文字
                tool_start = -1    # full_text 中 <tool_call> 的位置
                finish_reason = "length"
                try:
                    while True:
                        item = await run_in_threadpool(next, gen, None)
                        if item is None:
                            break
                        delta, finish = item
                        full_text += delta
                        if tool_start < 0 and delta:
                            if parse_tools:
                                pending += delta
                                idx = pending.find(TOOL_CALL_OPEN)
                                if idx >= 0:
                                    tool_start = len(full_text) - len(pending) + idx
                                    out, pending = pending[:idx], ""
                                else:
                                    keep = partial_tag_len(pending, TOOL_CALL_OPEN)
                                    out, pending = pending[:len(pending) - keep], pending[len(pending) - keep:]
                            else:
                                out = delta
                            if out:
                                yield make_chunk({"content": out})
                        if finish:
                            finish_reason = finish
                            break
                finally:
                    # 在持有鎖期間關閉產生器 (含客戶端中斷)，確保 kv_cache_ids 與模型狀態一致
                    gen.close()

                if tool_start >= 0:
                    _, calls = parse_tool_calls(full_text[tool_start:], req.tools)
                    if calls:
                        yield make_chunk({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
                        finish_reason = "tool_calls"
                    else:
                        # 工具呼叫格式不完整 (例如被 max_tokens 截斷)，原樣當作文字送出
                        yield make_chunk({"content": full_text[tool_start:]})
                elif pending:
                    yield make_chunk({"content": pending})

                yield make_chunk({}, finish_reason)
                if include_usage:
                    yield make_chunk(None, usage=make_usage(stats))
                yield "data: [DONE]\n\n"

        return StreamingResponse(openai_sse_generator(), media_type="text/event-stream")

    # --- 2. 標準非串流 JSON 回應 ---
    def generate_all():
        gen, stats = run_generation()
        parts, finish_reason = [], "length"
        try:
            for delta, finish in gen:
                parts.append(delta)
                if finish:
                    finish_reason = finish
                    break
        finally:
            gen.close()
        return "".join(parts), finish_reason, stats

    async with infer_lock:
        content_str, finish_reason, stats = await run_in_threadpool(generate_all)

    message: Dict[str, Any] = {"role": "assistant", "content": content_str}
    if parse_tools:
        content, calls = parse_tool_calls(content_str, req.tools)
        if calls:
            message = {"role": "assistant", "content": content or None, "tool_calls": calls}
            finish_reason = "tool_calls"

    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": MODEL_NAME,
        "system_fingerprint": "fp_openvino_irisxe",
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason
            }
        ],
        "usage": make_usage(stats)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=1234)
