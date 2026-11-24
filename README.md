# OpenRouter Integration - 生產級 Python 客戶端

功能完整、類型安全、支援非同步/同步雙模式的 OpenRouter Python 整合庫。

## ✨ 特性

- 🚀 **統一介面** - 一套 API 呼叫 300+ 模型
- 🔄 **智能備援** - 任務型路由 + 自動故障轉移
- 💰 **成本控制** - 請求級/日級預算、即時成本追蹤
- 🌊 **串流支援** - 非同步生成器逐token輸出
- 🛠 **Function Calling** - 內建工具循環處理器
- 📋 **結構化輸出** - JSON Schema / Pydantic 驗證
- 🧠 **推理模型** - 支援 DeepSeek R1、o1 系列思考鏈輸出
- 👁 **多模態** - 圖片/文檔理解 (Gemini, GPT-4o, Claude)
- 📝 **長上下文** - 最高 2M tokens (Gemini 1.5 Pro/Flash)
- 🔌 **擴展性** - 請求/響應鉤子、自訂模型註冊

## 📦 安裝

```bash
cd openrouter_integration
pip install -r requirements.txt
cp .env.example .env
# 編輯 .env 填入你的 OPENROUTER_API_KEY
```

## 🚀 快速開始

### 非同步 (推薦)

```python
import asyncio
from openrouter_integration import OpenRouterConfig, OpenRouterClient

async def main():
    config = OpenRouterConfig.from_env()
    async with OpenRouterClient(config) as client:
        # 簡單對話
        reply = await client.chat_simple("你好，請用繁體中文介紹你自己")
        print(reply)

        # 多輪對話 + 指定模型
        resp = await client.chat(
            messages=[
                {"role": "system", "content": "你是代碼專家"},
                {"role": "user", "content": "寫一個 Python 裝飾器範例"},
            ],
            model="deepseek-v3",  # 使用別名
            temperature=0.3,
        )
        print(resp.choices[0].message.content)
        print(f"成本: ${resp.usage.cost:.6f}")

asyncio.run(main())
```

### 同步介面 (給舊代碼/腳本用)

```python
from openrouter_integration import OpenRouterConfig, SyncOpenRouterClient

config = OpenRouterConfig.from_env()
with SyncOpenRouterClient(config) as client:
    reply = client.chat_simple("同步測試")
    print(reply)
```

## 🎯 核心功能範例

### 1. 智能任務路由 + 自動備援

```python
# 不指定模型，由 task_type 自動選最適合的並備援
resp = await client.chat_simple(
    "幫我重構這段代碼...",
    task_type="coding",        # 會依序嘗試: sonnet-3.5 → deepseek-v3 → qwen-coder-32b → ...
    fallback=True,
)
```

**內建任務類型**: `general` | `coding` | `reasoning` | `vision` | `long_context` | `fast_cheap` | `chinese`

### 2. 結構化輸出 (Pydantic 驗證)

```python
from pydantic import BaseModel, Field
from typing import List

class BugReport(BaseModel):
    severity: str  # low/medium/high/critical
    title: str
    description: str
    fix_suggestion: str
    affected_files: List[str]

report = await client.structured_output(
    messages=[{"role": "user", "content": "代碼審查: ..."}],
    schema=BugReport.model_json_schema(),
    model="sonnet-3.5",
    temperature=0.1,
)
# report 是 dict，可直接 BugReport(**report)
```

### 3. Function Calling 自動循環

```python
tools = [{
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "查詢資料庫",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string"},
            },
            "required": ["sql"],
        },
    },
}]

async def handler(name: str, args: dict):
    if name == "query_database":
        return await db.execute(args["sql"])
    return {"error": "unknown tool"}

resp = await client.chat_with_tools(
    messages=[{"role": "user", "content": "查一下上月銷售額"}],
    tools=tools,
    tool_handler=handler,
    model="deepseek-v3",
    max_rounds=5,
)
```

### 4. 代碼 Agent (內建工具集)

```python
from openrouter_integration import create_coding_agent_registry

tool_registry = create_coding_agent_registry(workspace="/path/to/project")

async def tool_handler(name, args):
    return await tool_registry.execute(name, args)

resp = await client.chat_with_tools(
    messages=[{"role": "user", "content": "建立一個 FastAPI 專案並加上測試"}],
    tools=tool_registry.list_for_model(),
    tool_handler=tool_handler,
    model="sonnet-3.5",
    max_rounds=15,
)
```

**內建工具**: `execute_python` | `execute_shell` | `read_file` | `write_file` | `list_files` | `http_request` | `get_current_time`

### 5. 串流輸出

```python
async for chunk in await client.chat(
    messages=[{"role": "user", "content": "寫一個長故事"}],
    model="gemini-1.5-flash",
    stream=True,
):
    content = chunk.choices[0].get("delta", {}).get("content", "")
    print(content, end="", flush=True)
```

### 6. 推理模型 (DeepSeek R1, o1) - 獲取思考過程

```python
resp = await client.chat(
    [{"role": "user", "content": "證明根號2是無理數"}],
    model="deepseek-r1",  # 或 o1-preview, o1-mini
)

msg = resp.choices[0].message
print("思考過程:", msg.reasoning)  # OpenRouter 專有欄位
print("最終答案:", msg.content)
```

### 7. 視覺/多模態

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "分析這張架構圖"},
        {"type": "image_url", "image_url": {"url": "https://example.com/arch.png"}},
    ],
}]
resp = await client.chat(messages, model="gemini-2.0-flash", task_type="vision")
```

### 8. 成本監控與預算控制

```python
config = OpenRouterConfig.from_env(
    daily_budget=5.0,           # 每日 $5 上限
    max_cost_per_request=0.50,  # 單請求 $0.5 上限
)

async with OpenRouterClient(config) as client:
    resp = await client.chat_simple("...")
    print(f"本次: ${resp.usage.cost:.6f}")
    print(f"今日累計: ${client.cost_tracker.get_today_spend():.4f}")
    print(f"請求數: {client.cost_tracker.get_today_requests()}")
```

超支會拋出 `BudgetExceededError`。

## 📚 模型別名速查

| 別名 | 完整模型 ID | 定位 |
|------|------------|------|
| `sonnet-3.5`, `best-coder` | `anthropic/claude-3.5-sonnet` | **代碼/綜合最強** |
| `opus-4` | `anthropic/claude-4-opus` | 旗艦推理 |
| `gpt-4o` | `openai/gpt-4o` | 多模態旗艦 |
| `o1-preview`, `o1-mini` | `openai/o1-preview` | 推理專精 |
| `gemini-2.0-flash` | `google/gemini-2.0-flash-exp` | **2M Context、多模態、極快** |
| `gemini-1.5-pro` | `google/gemini-pro-1.5` | 2M Context |
| `gemini-1.5-flash`, `free-tier` | `google/gemini-flash-1.5` | **免費額度大、極快** |
| `deepseek-v3`, `ds-v3` | `deepseek/deepseek-chat` | **開源最強、中文佳、高性價比** |
| `deepseek-r1`, `ds-r1` | `deepseek/deepseek-r1` | **開源推理、可看思考鏈** |
| `llama-3.3-70b` | `meta-llama/llama-3.3-70b-instruct` | 開源旗艦、極便宜 |
| `qwen-coder-32b` | `qwen/qwen-2.5-coder-32b-instruct` | 代碼專精開源 |
| `nemotron-ultra` | `nvidia/nemotron-3-ultra` | 極便宜、NVIDIA 優化 |

完整列表見 `config.py` 的 `MODEL_REGISTRY`。

## 🔧 進階配置

```python
config = OpenRouterConfig(
    api_key="sk-or-xxx",
    app_name="MyApp",
    app_url="https://myapp.com",
    default_model="deepseek-v3",
    max_retries=3,
    timeout=60.0,
    daily_budget=10.0,
    max_cost_per_request=1.0,
    log_requests=True,
    log_responses=False,
)

# 自訂模型註冊
from openrouter_integration.config import MODEL_REGISTRY, ModelConfig, ModelTier
MODEL_REGISTRY["my-model"] = ModelConfig(
    model_id="my-org/my-finetuned",
    tier=ModelTier.SPECIALIST,
    aliases=["finetuned"],
)
```

## 📁 專案結構

```
openrouter_integration/
├── __init__.by          # 匯出介面
├── config.py            # 配置管理、模型註冊表、路由策略
├── client.py            # 核心客戶端 (非同步/同步)
├── tools.py             # 內建工具集、代碼 Agent 工具
├── examples.py          # 完整使用範例
├── requirements.txt     # 依賴
├── .env.example         # 環境變數範本
└── README.md            # 本文件
```

## 🧪 執行範例

```bash
# 設定 API Key
export OPENROUTER_API_KEY=sk-or-xxxx

# 運行所有範例
python -m openrouter_integration.examples
```

## 📝 遷移指南 (從官方 OpenAI SDK)

```python
# 原本 (OpenAI)
from openai import AsyncOpenAI
client = AsyncOpenAI(api_key=OPENAI_KEY)
resp = await client.chat.completions.create(model="gpt-4o", messages=[...])

# 遷移到 OpenRouter (零代碼修改，只換 base_url)
from openai import AsyncOpenAI
client = AsyncOpenAI(
    api_key=OPENROUTER_KEY,
    base_url="https://openrouter.ai/api/v1",
)
resp = await client.chat.completions.create(model="anthropic/claude-3.5-sonnet", messages=[...])

# 或使用本庫 (更多功能)
from openrouter_integration import OpenRouterConfig, OpenRouterClient
config = OpenRouterConfig.from_env()
async with OpenRouterClient(config) as client:
    resp = await client.chat_simple("...", model="sonnet-3.5")
```

## 📄 授權

MIT License - 自由使用、修改、分發。

---

**需要更多功能？** 歡迎提 Issue 或 PR！
- WebSocket 即時對話
- 批次處理
- 微調模型管理
- 更多內建工具 (瀏覽器、資料庫、Git)