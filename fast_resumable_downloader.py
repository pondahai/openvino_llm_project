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
    "openvino_language_model.bin",
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

    # 1. 取得遠端檔案總大小
    head = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=15)
    if head.status_code != 200:
        print(f"[{filename}] 無法取得檔案資訊，HTTP: {head.status_code}")
        return False
    total_size = int(head.headers.get("content-length", 0))

    # 檢查是否已下載完成
    if os.path.exists(file_path):
        if os.path.getsize(file_path) == total_size:
            print(f"[{filename}] 已經完整下載過 ({total_size / (1024**3):.2f} GB)，跳過！")
            return True

    # 檢查現有已下載位元組 (斷點續傳)
    initial_byte = os.path.getsize(part_path) if os.path.exists(part_path) else 0

    print(f"\n==================================================")
    print(f"下載: {filename}")
    print(f"大小: {total_size / (1024**3):.2f} GB (已下載: {initial_byte / (1024**3):.2f} GB)")
    print(f"==================================================")

    chunk_size = 4 * 1024 * 1024  # 4MB 分塊
    retries = 0
    max_retries = 100

    while initial_byte < total_size and retries < max_retries:
        headers = HEADERS.copy()
        headers["Range"] = f"bytes={initial_byte}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=20, allow_redirects=True) as r:
                if r.status_code not in (200, 206):
                    print(f"伺服器回應異常: {r.status_code}，重試中...")
                    time.sleep(3)
                    retries += 1
                    continue

                mode = "ab" if initial_byte > 0 else "wb"
                with open(part_path, mode) as f:
                    last_print = time.time()
                    downloaded_since_print = 0
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            initial_byte += len(chunk)
                            downloaded_since_print += len(chunk)

                            now = time.time()
                            if now - last_print >= 2.0:
                                speed = (downloaded_since_print / (1024**2)) / (now - last_print)
                                progress = (initial_byte / total_size) * 100
                                print(f"[{filename}] {progress:.1f}% | {initial_byte / (1024**3):.2f}/{total_size / (1024**3):.2f} GB | 速度: {speed:.2f} MB/s", flush=True)
                                last_print = now
                                downloaded_since_print = 0
                                retries = 0  # 成功寫入就重設重試次數

        except Exception as e:
            retries += 1
            print(f"\n連線瞬斷 ({e})，2 秒後自動斷點續傳... (重試第 {retries} 次)")
            time.sleep(2)

    if initial_byte >= total_size:
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rename(part_path, file_path)
        print(f"\n🎉 [{filename}] 下載完成！")
        return True
    else:
        print(f"\n❌ [{filename}] 未能完成下載。")
        return False

if __name__ == "__main__":
    print("啟動專用斷點續傳多線程穩健下載器...")
    for f in FILES:
        download_file_resumable(f)
    print("\n==================================================")
    print("所有核心模型權重檔下載作業完畢！")
