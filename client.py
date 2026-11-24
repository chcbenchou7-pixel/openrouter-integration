"""OpenRouter 核心客戶端 - 生產級封裝"""
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, Callable
from collections import defaultdict

import httpx
from pydantic import BaseModel, Field

from .config import (
    ModelConfig,
    ModelTier,
    MODEL_REGISTRY,
    DEFAULT_ROUTING,
    OpenRouterConfig,
)

logger = logging.getLogger(__name__)


# ============================================================
# 資料模型
# ============================================================
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


class ToolDefinition(BaseModel):
    type: str = "function"
    function: Dict[str, Any]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict]] = None
    response_format: Optional[Dict] = None
    seed: Optional[int] = None
    # OpenRouter 特有
    route: Optional[str] = None  # "fallback" 等
    max_price: Optional[Dict[str, float]] = None  # {"prompt": 1, "completion": 3}
    provider: Optional[Dict] = None  # 指定供應商偏好


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # OpenRouter 額外欄位
    cost: float = 0.0


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None
    # 推理模型特有
    reasoning: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    model: str
    choices: List[Choice]
    usage: Usage
    created: int = Field(default_factory=lambda: int(time.time()))


class StreamChunk(BaseModel):
    id: str
    model: str
    choices: List[Dict]
    usage: Optional[Usage] = None


# ============================================================
# 成本追蹤器
# ============================================================
class CostTracker:
    """請求級與日級成本追蹤"""

    def __init__(self, daily_budget: float = 10.0, max_per_request: float = 1.0):
        self.daily_budget = daily_budget
        self.max_per_request = max_per_request
        self._daily_spend: Dict[date, float] = defaultdict(float)
        self._request_count: Dict[date, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def check_budget(self, estimated_cost: float = 0.0) -> None:
        """檢查預算，超支拋出異常"""
        async with self._lock:
            today = date.today()
            if self._daily_spend[today] + estimated_cost > self.daily_budget:
                raise BudgetExceededError(
                    f"每日預算超限: ${self._daily_spend[today]:.4f} / ${self.daily_budget}"
                )
            if estimated_cost > self.max_per_request:
                raise BudgetExceededError(
                    f"單請求成本超限: ${estimated_cost:.4f} > ${self.max_per_request}"
                )

    async def record(self, cost: float, model: str) -> None:
        """記錄實際成本"""
        async with self._lock:
            today = date.today()
            self._daily_spend[today] += cost
            self._request_count[today] += 1
            logger.info(f"成本記錄: ${cost:.6f} (模型: {model}), 今日累計: ${self._daily_spend[today]:.4f}")

    def get_today_spend(self) -> float:
        return self._daily_spend.get(date.today(), 0.0)

    def get_today_requests(self) -> int:
        return self._request_count.get(date.today(), 0)


class BudgetExceededError(Exception):
    pass


# ============================================================
# 模型解析器
# ============================================================
class ModelResolver:
    """別名解析、路由選擇、能力檢查"""

    def __init__(self, registry: Dict[str, ModelConfig] = None):
        self.registry = registry or MODEL_REGISTRY
        self._alias_map = self._build_alias_map()

    def _build_alias_map(self) -> Dict[str, str]:
        m = {}
        for key, cfg in self.registry.items():
            m[key] = key
            for alias in cfg.aliases:
                m[alias.lower()] = key
        return m

    def resolve(self, model_ref: str) -> ModelConfig:
        """解析模型引用 (別名或完整 ID) -> ModelConfig"""
        key = self._alias_map.get(model_ref.lower())
        if key:
            return self.registry[key]
        # 嘗試直接用 OpenRouter ID
        for cfg in self.registry.values():
            if cfg.model_id == model_ref:
                return cfg
        # 找不到，回傳一個動態配置
        logger.warning(f"未知模型引用: {model_ref}，使用動態配置")
        return ModelConfig(model_id=model_ref, tier=ModelTier.BALANCED)

    def get_routing_chain(self, task_type: str) -> List[ModelConfig]:
        """獲取任務類型的備援鏈"""
        aliases = DEFAULT_ROUTING.get(task_type, DEFAULT_ROUTING["general"])
        return [self.resolve(a) for a in aliases]

    def filter_by_capability(
        self,
        models: List[ModelConfig],
        need_tools: bool = False,
        need_vision: bool = False,
        need_reasoning: bool = False,
        max_cost_per_1m: Optional[float] = None,
    ) -> List[ModelConfig]:
        """按能力過濾模型"""
        filtered = []
        for m in models:
            if need_tools and not m.supports_tools:
                continue
            if need_vision and not m.supports_vision:
                continue
            if need_reasoning and not m.supports_reasoning:
                continue
            if max_cost_per_1m and (m.input_cost_per_1m + m.output_cost_per_1m) > max_cost_per_1m:
                continue
            filtered.append(m)
        return filtered


# ============================================================
# 核心客戶端
# ============================================================
class OpenRouterClient:
    """
    生產級 OpenRouter 客戶端

    特性:
    - 自動重試 + 指數退避
    - 多模型備援路由
    - 成本追蹤與預算控制
    - 串流支援
    - 結構化輸出 (JSON Schema)
    - Function Calling 便利封裝
    - 請求/響應日誌
    """

    def __init__(self, config: OpenRouterConfig):
        self.config = config
        self.resolver = ModelResolver()
        self.cost_tracker = CostTracker(config.daily_budget, config.max_cost_per_request)
        self._client: Optional[httpx.AsyncClient] = None
        self._request_hooks: List[Callable] = []
        self._response_hooks: List[Callable] = []

    async def __aenter__(self):
        await self._ensure_client()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _ensure_client(self):
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": self.config.app_url,
                    "X-Title": self.config.app_name,
                },
                timeout=httpx.Timeout(self.config.timeout),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
            )

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # --------------------------------------------------------
    # 核心請求方法
    # --------------------------------------------------------
    async def _request_with_retry(
        self,
        method: str,
        path: str,
        json_data: Dict,
        stream: bool = False,
    ) -> httpx.Response:
        """帶重試的 HTTP 請求"""
        await self._ensure_client()

        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                # 請求鉤子
                for hook in self._request_hooks:
                    await hook(method, path, json_data)

                if stream:
                    # 串流請求不在這裡處理重試，由調用方處理
                    return await self._client.request(method, path, json=json_data, stream=True)

                response = await self._client.request(method, path, json=json_data)

                # 響應鉤子
                for hook in self._response_hooks:
                    await hook(response)

                # 成功狀態碼
                if response.status_code < 400:
                    return response

                # 可重試的錯誤
                if response.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                # 不可重試的錯誤 (4xx 除了 429)
                response.raise_for_status()

            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                last_error = e
                if attempt < self.config.max_retries:
                    wait = (2 ** attempt) * 0.5 + (attempt * 0.1)  # 指數退避 + 抖動
                    logger.warning(f"請求失敗 (嘗試 {attempt+1}/{self.config.max_retries+1}): {e}, {wait:.1f}s 後重試")
                    await asyncio.sleep(wait)
                    continue
                break

        # 所有重試失敗
        raise RequestFailedError(f"請求失敗經過 {self.config.max_retries+1} 次嘗試: {last_error}") from last_error

    # --------------------------------------------------------
    # 公開 API
    # --------------------------------------------------------
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        response_format: Optional[Dict] = None,
        stream: bool = False,
        task_type: str = "general",
        fallback: bool = True,
        **kwargs,
    ) -> Union[ChatCompletionResponse, AsyncGenerator[StreamChunk, None]]:
        """
        統一聊天接口

        Args:
            messages: 消息列表
            model: 模型別名或 ID (None 時使用 task_type 路由)
            temperature: 溫度
            max_tokens: 最大輸出 tokens
            tools: Function Calling 定義
            tool_choice: 工具選擇策略
            response_format: 結構化輸出格式 {"type": "json_object"} 或 {"type": "json_schema", "json_schema": {...}}
            stream: 是否串流
            task_type: 任務類型 (用於自動路由)
            fallback: 失敗時是否自動嘗試備援模型
        """
        # 解析模型鏈
        if model:
            primary = self.resolver.resolve(model)
            chain = [primary]
        else:
            chain = self.resolver.get_routing_chain(task_type)

        last_error = None
        for i, model_cfg in enumerate(chain):
            try:
                return await self._execute_chat(
                    model_cfg=model_cfg,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens or model_cfg.max_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    stream=stream,
                    **kwargs,
                )
            except Exception as e:
                last_error = e
                logger.warning(f"模型 {model_cfg.model_id} 失敗: {e}")
                if not fallback or i == len(chain) - 1:
                    break
                logger.info(f"嘗試備援模型 ({i+2}/{len(chain)})...")
                continue

        raise RequestFailedError(f"所有模型均失敗，最後錯誤: {last_error}") from last_error

    async def _execute_chat(
        self,
        model_cfg: ModelConfig,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        tools: Optional[List],
        tool_choice: Optional[Union[str, Dict]],
        response_format: Optional[Dict],
        stream: bool,
        **kwargs,
    ):
        """執行單次聊天請求"""
        # 預估成本檢查
        estimated_input = sum(len(str(m.get("content", ""))) for m in messages) / 4  # 粗略估算
        estimated_cost = (estimated_input / 1_000_000) * model_cfg.input_cost_per_1m
        await self.cost_tracker.check_budget(estimated_cost)

        request = ChatCompletionRequest(
            model=model_cfg.model_id,
            messages=[Message(**m) for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            tools=[ToolDefinition(**t) for t in tools] if tools else None,
            tool_choice=tool_choice,
            response_format=response_format,
            stream=stream,
            **kwargs,
        )

        if stream:
            return self._stream_chat(request, model_cfg)

        response = await self._request_with_retry("POST", "/chat/completions", request.model_dump(exclude_none=True))
        data = response.json()
        completion = ChatCompletionResponse(**data)

        # 計算實際成本
        actual_cost = self._calculate_cost(completion.usage, model_cfg)
        await self.cost_tracker.record(actual_cost, model_cfg.model_id)

        if self.config.log_requests:
            logger.info(f"完成: {model_cfg.model_id} | tokens: {completion.usage.total_tokens} | 成本: ${actual_cost:.6f}")

        return completion

    async def _stream_chat(
        self,
        request: ChatCompletionRequest,
        model_cfg: ModelConfig,
    ) -> AsyncGenerator[StreamChunk, None]:
        """串流聊天"""
        response = await self._request_with_retry("POST", "/chat/completions", request.model_dump(exclude_none=True), stream=True)

        async for line in response.aiter_lines():
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                try:
                    chunk_data = json.loads(line[6:])
                    chunk = StreamChunk(**chunk_data)
                    yield chunk
                except json.JSONDecodeError:
                    continue

        # 串流結束後記錄成本 (需從 usage 字段獲取，最後一個 chunk 可能包含)
        # 這裡簡化處理，實際可累積 tokens

    def _calculate_cost(self, usage: Usage, model_cfg: ModelConfig) -> float:
        """計算實際成本"""
        input_cost = (usage.prompt_tokens / 1_000_000) * model_cfg.input_cost_per_1m
        output_cost = (usage.completion_tokens / 1_000_000) * model_cfg.output_cost_per_1m
        return input_cost + output_cost

    # --------------------------------------------------------
    # 便利方法
    # --------------------------------------------------------
    async def chat_simple(
        self,
        prompt: str,
        system: str = "你是一個有幫助的 AI 助手，使用繁體中文回答。",
        model: Optional[str] = None,
        **kwargs,
    ) -> str:
        """簡單單輪對話"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        resp = await self.chat(messages, model=model, **kwargs)
        return resp.choices[0].message.content

    async def chat_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        tool_handler: Callable[[str, Dict], Any],
        model: Optional[str] = None,
        max_rounds: int = 5,
        **kwargs,
    ) -> ChatCompletionResponse:
        """
        自動處理 Function Calling 循環

        Args:
            messages: 初始消息
            tools: 工具定義
            tool_handler: 異步函數 (tool_name, arguments) -> result
            max_rounds: 最大工具調用輪數
        """
        current_messages = messages.copy()

        for round_num in range(max_rounds):
            resp = await self.chat(
                current_messages,
                model=model,
                tools=tools,
                tool_choice="auto",
                **kwargs,
            )

            msg = resp.choices[0].message
            current_messages.append(msg.model_dump())

            if not msg.tool_calls:
                return resp

            # 執行工具
            for tc in msg.tool_calls:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                try:
                    result = await tool_handler(name, args)
                except Exception as e:
                    result = {"error": str(e)}

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        # 超過最大輪數
        return await self.chat(current_messages, model=model, tools=tools, **kwargs)

    async def structured_output(
        self,
        messages: List[Dict],
        schema: Dict,
        model: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        結構化輸出 (JSON Schema)

        Args:
            schema: JSON Schema 字典 或 Pydantic model.model_json_schema()
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": schema,
                "strict": True,
            },
        }
        resp = await self.chat(messages, model=model, response_format=response_format, **kwargs)
        return json.loads(resp.choices[0].message.content)

    # --------------------------------------------------------
    # 鉤子系統
    # --------------------------------------------------------
    def add_request_hook(self, hook: Callable):
        self._request_hooks.append(hook)

    def add_response_hook(self, hook: Callable):
        self._response_hooks.append(hook)


class RequestFailedError(Exception):
    pass


# ============================================================
# 同步包裝器 (給非 async 代碼用)
# ============================================================
class SyncOpenRouterClient:
    """同步介面包裝器"""

    def __init__(self, config: OpenRouterConfig):
        self._async_client = OpenRouterClient(config)
        self._loop = None

    def _get_loop(self):
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop

    def _run(self, coro):
        loop = self._get_loop()
        if loop.is_running():
            # 在運行的循環中，創建任務
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)

    def __enter__(self):
        self._run(self._async_client.__aenter__())
        return self

    def __exit__(self, *args):
        self._run(self._async_client.__aexit__(*args))

    def chat(self, *args, **kwargs):
        return self._run(self._async_client.chat(*args, **kwargs))

    def chat_simple(self, *args, **kwargs):
        return self._run(self._async_client.chat_simple(*args, **kwargs))

    def chat_with_tools(self, *args, **kwargs):
        return self._run(self._async_client.chat_with_tools(*args, **kwargs))

    def structured_output(self, *args, **kwargs):
        return self._run(self._async_client.structured_output(*args, **kwargs))

    @property
    def cost_tracker(self):
        return self._async_client.cost_tracker