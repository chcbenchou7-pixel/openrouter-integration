"""
OpenRouter Integration - 生產級 Python 客戶端

快速開始:
    from openrouter_integration import OpenRouterClient, OpenRouterConfig

    config = OpenRouterConfig.from_env()
    async with OpenRouterClient(config) as client:
        response = await client.chat_simple("你好，請用繁體中文介紹你自己")
        print(response)
"""

from .config import (
    OpenRouterConfig,
    ModelConfig,
    ModelTier,
    MODEL_REGISTRY,
    DEFAULT_ROUTING,
)
from .client import (
    OpenRouterClient,
    SyncOpenRouterClient,
    ChatCompletionResponse,
    StreamChunk,
    RequestFailedError,
    BudgetExceededError,
)
from .tools import (
    Tool,
    ToolRegistry,
    create_builtin_registry,
    create_coding_agent_registry,
)

__version__ = "1.0.0"
__all__ = [
    # Config
    "OpenRouterConfig",
    "ModelConfig",
    "ModelTier",
    "MODEL_REGISTRY",
    "DEFAULT_ROUTING",
    # Client
    "OpenRouterClient",
    "SyncOpenRouterClient",
    "ChatCompletionResponse",
    "StreamChunk",
    "RequestFailedError",
    "BudgetExceededError",
    # Tools
    "Tool",
    "ToolRegistry",
    "create_builtin_registry",
    "create_coding_agent_registry",
]