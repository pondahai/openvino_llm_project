import sys
import time
import os
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

def test_and_capture_client_gui():
    print("啟動 Playwright 瀏覽器...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 850})
        
        print("連線至 http://localhost:8501 ...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
        time.sleep(2)
        
        chat_box = page.locator("textarea[data-testid='stChatInputTextArea']")
        chat_box.wait_for(state="visible", timeout=30000)
        
        prompt = "請用簡短繁體中文說明你準備好執行 AI Agent 任務了嗎？"
        print(f"輸入問題: {prompt}")
        chat_box.fill(prompt)
        time.sleep(1)
        chat_box.press("Enter")
        
        print("送出成功，等待後端 API Server 回應中...")
        page.wait_for_selector("text=架構: Client", timeout=60000)
        print("推論完成！儲存截圖...")
        time.sleep(2)
        
        out_path = r"C:\Users\USER\openvino_llm_project\docs\images\client_gui_showcase.png"
        page.screenshot(path=out_path)
        print("截圖成功保存至:", out_path)
        browser.close()

if __name__ == "__main__":
    test_and_capture_client_gui()
