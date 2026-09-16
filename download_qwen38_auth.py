import os
import sys
from huggingface_hub import snapshot_download, login

TOKEN = "***REMOVED***"

print("Logging in to Hugging Face...")
login(token=TOKEN)

target_dir = r"C:\Users\USER\openvino_llm_project\models\Qwen3.8-27B-int4-ov"
os.makedirs(target_dir, exist_ok=True)

print("========================================================")
print("  使用官方認證 Token 重啟高速下載 Qwen3.8-27B-int4-ov")
print(f"  目標目錄: {target_dir}")
print("========================================================")

try:
    path = snapshot_download(
        repo_id="OpenVINO/Qwen3.8-27B-int4-ov",
        local_dir=target_dir,
        token=TOKEN,
        max_workers=4
    )
    print("\n🎉 下載完成！模型已就緒:", path)
except Exception as e:
    print(f"\n❌ 下載失敗: {e}")
