import time
import os
from playwright.sync_api import sync_playwright

def run_chat_and_screenshot():
    print("啟動 Playwright 瀏覽器...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # 設定 1280x800 高畫質視窗
        context = browser.new_context(viewport={"width": 1280, "height": 850})
        page = context.new_page()
        
        print("前往 http://localhost:8501 ...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
        time.sleep(3)
        
        # 尋找輸入框
        print("尋找輸入框...")
        chat_input = page.locator("textarea[data-testid='stChatInputTextArea']")
        chat_input.wait_for(state="visible", timeout=30000)
        
        # 輸入經典問題
        prompt_text = "請用簡潔的繁體中文介紹你自己，並說明你能如何協助我的 AI Agent 工作流？"
        print(f"輸入對話: {prompt_text}")
        chat_input.fill(prompt_text)
        time.sleep(1)
        chat_input.press("Enter")
        
        print("訊息已送出，等待模型推論 (Iris Xe GPU 首次編譯與推理約需 45~90 秒)...")
        # 等待包含助手回答完成的標籤
        # 推論完成後會有 st.caption: "⚡ 首字延遲 (TTFT):"
        page.wait_for_selector("text=⚡ 首字延遲", timeout=180000)
        print("推論完成！正在進行頁面調整以取得最佳截圖...")
        time.sleep(2)
        
        screenshot_dir = r"C:\Users\USER\openvino_llm_project\docs\images"
        os.makedirs(screenshot_dir, exist_ok=True)
        screenshot_path = os.path.join(screenshot_dir, "gui_showcase.png")
        
        page.screenshot(path=screenshot_path, full_page=False)
        print(f"✅ 截圖成功儲存至: {screenshot_path}")
        
        browser.close()

if __name__ == "__main__":
    run_chat_and_screenshot()
