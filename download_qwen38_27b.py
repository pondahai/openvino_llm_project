import os
import sys
from huggingface_hub import snapshot_download

target_dir = r"C:\Users\USER\openvino_llm_project\models\Qwen3.8-27B-int4-ov"
os.makedirs(target_dir, exist_ok=True)

print(f"========================================================")
print(f"  開始下載 Qwen3.8-27B-int4-ov (約 15.2 GB)")
print(f"  目標目錄: {target_dir}")
print(f"========================================================")

try:
    path = snapshot_download(
        repo_id="OpenVINO/Qwen3.8-27B-int4-ov",
        local_dir=target_dir,
        resume_download=True,
        max_workers=4
    )
    print("\n🎉 下載完成！模型已就緒:", path)
except Exception as e:
    print(f"\n❌ 下載失敗: {e}")
