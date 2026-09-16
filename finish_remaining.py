import os
import sys
import time
import requests

# 請先設定環境變數 HF_TOKEN (Hugging Face Access Token)
TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    sys.exit("請先設定環境變數 HF_TOKEN，例如 PowerShell: $env:HF_TOKEN = \"hf_...\"")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

FILES = [
    "openvino_text_embeddings_model.bin",
    "openvino_vision_embeddings_pos_model.bin",
    "tokenizer.json"
]

BASE_URL = "https://huggingface.co/OpenVINO/Qwen3.8-27B-int4-ov/resolve/main"
TARGET_DIR = r"C:\Users\USER\openvino_llm_project\models\Qwen3.8-27B-int4-ov"
os.makedirs(TARGET_DIR, exist_ok=True)

def download_file_resumable(filename):
    url = f"{BASE_URL}/{filename}"
    file_path = os.path.join(TARGET_DIR, filename)
    part_path = file_path + ".part"

    head = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=15)
    if head.status_code != 200:
        print(f"[{filename}] HTTP {head.status_code}")
        return False
    total_size = int(head.headers.get("content-length", 0))

    if os.path.exists(file_path):
        if os.path.getsize(file_path) == total_size:
            print(f"[{filename}] 已存在且完整 ({total_size / (1024**2):.2f} MB)")
            return True

    initial_byte = os.path.getsize(part_path) if os.path.exists(part_path) else 0
    print(f"\n下載: {filename} ({total_size / (1024**2):.2f} MB)")

    chunk_size = 2 * 1024 * 1024
    retries = 0
    while initial_byte < total_size and retries < 50:
        headers = HEADERS.copy()
        headers["Range"] = f"bytes={initial_byte}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=20, allow_redirects=True) as r:
                if r.status_code not in (200, 206):
                    time.sleep(2)
                    retries += 1
                    continue
                mode = "ab" if initial_byte > 0 else "wb"
                with open(part_path, mode) as f:
                    last_p = time.time()
                    downloaded_chunk = 0
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            initial_byte += len(chunk)
                            downloaded_chunk += len(chunk)
                            now = time.time()
                            if now - last_p >= 2.0:
                                spd = (downloaded_chunk / (1024**2)) / (now - last_p)
                                print(f"[{filename}] {(initial_byte/total_size)*100:.1f}% | {initial_byte/(1024**2):.1f}/{total_size/(1024**2):.1f} MB | {spd:.2f} MB/s", flush=True)
                                last_p = now
                                downloaded_chunk = 0
                                retries = 0
        except Exception as e:
            retries += 1
            time.sleep(2)

    if initial_byte >= total_size:
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rename(part_path, file_path)
        print(f"[{filename}] 完成！")
        return True
    return False

if __name__ == "__main__":
    for f in FILES:
        download_file_resumable(f)
    print("ALL_DONE")
