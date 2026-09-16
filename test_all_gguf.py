import openvino_genai as ov_genai
from pathlib import Path
import sys

LM_STUDIO_MODELS_DIR = Path.home() / ".lmstudio" / "models"
models = list(LM_STUDIO_MODELS_DIR.glob("**/*.gguf"))

out_log = Path.home() / "openvino_test_results.txt"
with open(out_log, "w", encoding="utf-8") as log_file:
    log_file.write(f"Total GGUF files found: {len(models)}\n")
    log_file.flush()

    for m in models:
        if "mmproj" in m.name.lower():
            continue
        log_file.write(f"\n==========================================\n")
        log_file.write(f"Testing model: {m.name}\n")
        log_file.flush()
        try:
            pipe = ov_genai.LLMPipeline(str(m), "CPU")
            log_file.write("Result: SUCCESS on CPU!\n")
            log_file.flush()
            try:
                pipe_gpu = ov_genai.LLMPipeline(str(m), "GPU")
                log_file.write("Result: SUCCESS on GPU (Iris Xe)!\n")
                log_file.flush()
            except Exception as egpu:
                log_file.write(f"GPU Failed: {egpu}\n")
                log_file.flush()
        except Exception as e:
            log_file.write(f"Result: FAILED ({type(e).__name__}: {e})\n")
            log_file.flush()

print("ALL TESTS FINISHED!")
