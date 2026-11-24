"""
OpenRouter 整合使用範例 (2025-11 更新版)

運行前請設定環境變數:
    export OPENROUTER_API_KEY="sk-or-xxxx"
    export OPENROUTER_APP_NAME="MyApp"
    export OPENROUTER_APP_URL="https://myapp.com"
"""
import asyncio
import json
import os
from pydantic import BaseModel, Field
from typing import List, Optional

from openrouter_integration import (
    OpenRouterConfig,
    OpenRouterClient,
    SyncOpenRouterClient,
    create_builtin_registry,
    create_coding_agent_registry,
    ToolRegistry,
    Tool,
)
from openrouter_integration.config import MODEL_REGISTRY


# ============================================================
# 範例 1: 基礎用法
# ============================================================
async def example_basic():
    print("=== 範例 1: 基礎對話 ===")
    config = OpenRouterConfig.from_env()
    async with OpenRouterClient(config) as client:
        # 簡單單輪 (預設用 deepseek-v3.2)
        reply = await client.chat_simple("你好，請用繁體中文介紹你自己")
        print(f"回覆: {reply[:200]}...")

        # 多輪對話 + 指定模型
        messages = [
            {"role": "system", "content": "你是代碼專家，用繁體中文回答"},
            {"role": "user", "content": "解釋什麼是裝飾器"},
        ]
        resp = await client.chat(messages, model="claude-sonnet-5.5", temperature=0.3)
        print(f"Tokens: {resp.usage.total_tokens}, 成本: ${resp.usage.cost:.6f}")


# ============================================================
# 範例 2: 串流輸出
# ============================================================
async def example_streaming():
    print("\n=== 範例 2: 串流輸出 ===")
    config = OpenRouterConfig.from_env()
    async with OpenRouterClient(config) as client:
        print("串流回覆: ", end="", flush=True)
        async for chunk in await client.chat(
            [{"role": "user", "content": "寫一首關於 AI 的短詩"}],
            model="gemini-2.5-flash-lite",  # 免費/極便宜
            stream=True,
            task_type="fast_cheap",
        ):
            content = chunk.choices[0].get("delta", {}).get("content", "")
            print(content, end="", flush=True)
        print()


# ============================================================
# 範例 3: 結構化輸出 (JSON Schema / Pydantic)
# ============================================================
class CodeReview(BaseModel):
    score: int = Field(ge=1, le=10, description="代碼品質評分 1-10")
    issues: List[str] = Field(description="發現的問題列表")
    suggestions: List[str] = Field(description="改進建議")
    security_concerns: List[str] = Field(default=[], description="安全疑慮")


async def example_structured_output():
    print("\n=== 範例 3: 結構化輸出 ===")
    config = OpenRouterConfig.from_env()
    async with OpenRouterClient(config) as client:
        code = '''
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
'''

        result = await client.structured_output(
            messages=[
                {"role": "system", "content": "你是資深代碼審查專家"},
                {"role": "user", "content": f"請審查這段代碼：\n{code}"},
            ],
            schema=CodeReview.model_json_schema(),
            model="claude-sonnet-5.5",  # 代碼審查用最強模型
            temperature=0.1,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # 驗證 Pydantic
        review = CodeReview(**result)
        print(f"\n評分: {review.score}/10")
        print(f"問題數: {len(review.issues)}")


# ============================================================
# 範例 4: Function Calling 自動循環
# ============================================================
async def example_function_calling():
    print("\n=== 範例 4: Function Calling (計算器) ===")
    config = OpenRouterConfig.from_env()
    async with OpenRouterClient(config) as client:
        # 定義工具
        tools = [{
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "執行數學計算",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "數學表達式，如：(3+5)*2"},
                    },
                    "required": ["expression"],
                },
            },
        }]

        # 處理器
        async def handler(name: str, args: dict):
            if name == "calculate":
                try:
                    # 簡單 eval (生產環境請用安全解析器)
                    result = eval(args["expression"], {"__builtins__": {}}, {})
                    return {"result": result}
                except Exception as e:
                    return {"error": str(e)}
            return {"error": f"未知工具: {name}"}

        resp = await client.chat_with_tools(
            messages=[{"role": "user", "content": "幫我算一下 (15+27)*3 - 100/4 等於多少"}],
            tools=tools,
            tool_handler=handler,
            model="deepseek-v3.2",  # 支援 tools 的模型
            temperature=0,
        )
        print(f"最終回覆: {resp.choices[0].message.content}")


# ============================================================
# 範例 5: 代碼 Agent (內建工具集)
# ============================================================
async def example_coding_agent():
    print("\n=== 範例 5: 代碼 Agent ===")
    config = OpenRouterConfig.from_env()

    # 建立工作區工具集
    workspace = "/tmp/openrouter_demo"
    os.makedirs(workspace, exist_ok=True)
    tool_registry = create_coding_agent_registry(workspace)

    async def tool_handler(name: str, args: dict):
        return await tool_registry.execute(name, args)

    async with OpenRouterClient(config) as client:
        # 讓 AI 建立一個簡單的專案
        task = """
在工作區建立一個簡單的 Python 專案：
1. 建立 main.py 包含一個計算費波納契數列的函數
2. 建立 test_main.py 測試該函數
3. 運行測試確保通過
"""

        tools = tool_registry.list_for_model()
        resp = await client.chat_with_tools(
            messages=[
                {"role": "system", "content": "你是資深 Python 開發者，使用工具完成任務"},
                {"role": "user", "content": task},
            ],
            tools=tools,
            tool_handler=tool_handler,
            model="claude-sonnet-5.5",  # 代碼任務用最強
            max_rounds=10,
            temperature=0.1,
        )
        print(f"完成: {resp.choices[0].message.content}")


# ============================================================
# 範例 6: 多模型備援路由
# ============================================================
async def example_fallback_routing():
    print("\n=== 範例 6: 智能備援路由 ===")
    config = OpenRouterConfig.from_env()
    async with OpenRouterClient(config) as client:
        # 不指定模型，由 task_type 自動選擇並備援
        for task_type in ["coding", "reasoning", "vision", "fast_cheap", "chinese", "agent"]:
            try:
                resp = await client.chat_simple(
                    f"這是 {task_type} 任務測試，請簡單回覆 OK",
                    task_type=task_type,
                    fallback=True,
                )
                print(f"  {task_type}: ✓ 使用模型 (自動選擇)")
            except Exception as e:
                print(f"  {task_type}: ✗ {e}")


# ============================================================
# 範例 7: 成本監控與預算控制
# ============================================================
async def example_cost_tracking():
    print("\n=== 範例 7: 成本追蹤 ===")
    config = OpenRouterConfig.from_env(daily_budget=0.50, max_cost_per_request=0.10)
    async with OpenRouterClient(config) as client:
        for i in range(3):
            resp = await client.chat_simple(
                f"請數到 100 (第 {i+1} 次)",
                model="gemini-2.5-flash-lite",  # 最便宜模型
            )
            print(f"  請求 {i+1}: Tokens={resp.usage.total_tokens}, 成本=${resp.usage.cost:.6f}")

        print(f"今日累計: ${client.cost_tracker.get_today_spend():.4f}")
        print(f"請求數: {client.cost_tracker.get_today_requests()}")


# ============================================================
# 範例 8: 同步介面 (給非 async 代碼用)
# ============================================================
def example_sync_interface():
    print("\n=== 範例 8: 同步介面 ===")
    config = OpenRouterConfig.from_env()
    with SyncOpenRouterClient(config) as client:
        reply = client.chat_simple("同步介面測試，請回覆 OK")
        print(f"回覆: {reply}")
        print(f"成本: ${client.cost_tracker.get_today_spend():.6f}")


# ============================================================
# 範例 9: 推理模型 (DeepSeek R1, o3, GLM) 獲取思考過程
# ============================================================
async def example_reasoning_models():
    print("\n=== 範例 9: 推理模型思考過程 ===")
    config = OpenRouterConfig.from_env()
    async with OpenRouterClient(config) as client:
        # DeepSeek R1 - 開源推理模型
        resp = await client.chat(
            [{"role": "user", "content": "證明：根號 2 是無理數"}],
            model="deepseek-r1",
            temperature=0.3,
        )
        msg = resp.choices[0].message
        print(f"推理過程:\n{msg.reasoning or '模型未返回 reasoning 字段'}")
        print(f"\n最終答案:\n{msg.content}")

        # o3-mini - OpenAI 推理模型
        resp2 = await client.chat(
            [{"role": "user", "content": "解釋 P vs NP 問題"}],
            model="o3-mini",
        )
        print(f"\no3-mini 回覆:\n{resp2.choices[0].message.content[:300]}...")


# ============================================================
# 範例 10: 視覺模型 (圖片理解)
# ============================================================
async def example_vision():
    print("\n=== 範例 10: 視覺模型 ===")
    config = OpenRouterConfig.from_env()
    async with OpenRouterClient(config) as client:
        # 使用 base64 圖片或 URL
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "這張圖片裡有什麼？請用繁體中文詳細描述"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
            ],
        }]
        resp = await client.chat(messages, model="gemini-4-carbon", task_type="vision")
        print(f"視覺分析: {resp.choices[0].message.content}")


# ============================================================
# 範例 11: 長上下文處理 (1M+ tokens)
# ============================================================
async def example_long_context():
    print("\n=== 範例 11: 長上下文 (1M+ tokens) ===")
    config = OpenRouterConfig.from_env()
    async with OpenRouterClient(config) as client:
        # 模擬長文檔
        long_doc = "這是一個很長的文檔。\n" * 50000  # 約 100k tokens

        resp = await client.chat(
            [
                {"role": "system", "content": "你是文檔分析專家"},
                {"role": "user", "content": f"請總結這份文檔的重點：\n{long_doc}"},
            ],
            model="gemini-pro-1.5",  # 2M context
            task_type="long_context",
            max_tokens=2000,
        )
        print(f"總結: {resp.choices[0].message.content}")
        print(f"輸入 Tokens: {resp.usage.prompt_tokens}")


# ============================================================
# 範例 12: 最新旗艦模型對比
# ============================================================
async def example_flagship_comparison():
    print("\n=== 範例 12: 旗艦模型對比 ===")
    config = OpenRouterConfig.from_env()
    async with OpenRouterClient(config) as client:
        prompt = "用一句話解釋量子糾纏，並給出一個生活化的比喻"
        
        models_to_test = [
            "claude-sonnet-5.5",
            "gpt-5",
            "gemini-4-carbon",
            "deepseek-v3.2",
            "glm-5.3",
            "qwen4-max",
        ]
        
        for model_key in models_to_test:
            try:
                resp = await client.chat_simple(prompt, model=model_key, temperature=0.7)
                model_name = MODEL_REGISTRY[model_key].model_id
                print(f"\n[{model_name}]")
                print(f"  {resp.choices[0].message.content[:200]}...")
                print(f"  成本: ${resp.usage.cost:.6f}")
            except Exception as e:
                print(f"  {model_key}: 錯誤 - {e}")


# ============================================================
# 範例 13: 自訂模型註冊
# ============================================================
async def example_custom_models():
    print("\n=== 範例 13: 自訂模型 ===")
    from openrouter_integration.config import ModelConfig, ModelTier, MODEL_REGISTRY

    # 註冊新模型 (例如微調後的專屬模型)
    MODEL_REGISTRY["my-finetuned"] = ModelConfig(
        model_id="my-org/my-finetuned-model",
        tier=ModelTier.SPECIALIST,
        max_tokens=4096,
        supports_tools=True,
        input_cost_per_1m=0.50,
        output_cost_per_1m=1.50,
        context_window=32768,
        aliases=["my-model", "finetuned"],
    )

    config = OpenRouterConfig.from_env()
    async with OpenRouterClient(config) as client:
        # 現在可以用別名直接呼叫
        resp = await client.chat_simple("測試自訂模型", model="my-model")
        print(f"回覆: {resp}")


# ============================================================
# 主程序
# ============================================================
async def main():
    # 執行所有範例 (按需取消註釋)
    await example_basic()
    # await example_streaming()
    # await example_structured_output()
    # await example_function_calling()
    # await example_coding_agent()
    # await example_fallback_routing()
    # await example_cost_tracking()
    # example_sync_interface()
    # await example_reasoning_models()
    # await example_vision()
    # await example_long_context()
    # await example_flagship_comparison()
    # await example_custom_models()

    print("\n✅ 所有範例執行完成")


if __name__ == "__main__":
    # 設定日誌
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    asyncio.run(main())