"""常用工具定義與處理器"""
import json
import asyncio
import subprocess
import sys
from typing import Any, Dict, List, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime


# ============================================================
# 工具定義結構
# ============================================================
@dataclass
class Tool:
    """工具元數據"""
    name: str
    description: str
    parameters: Dict  # JSON Schema
    handler: Callable[[Dict], Awaitable[Any]]

    def to_openrouter_format(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具註冊表"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools.get(name)

    def list_for_model(self) -> List[Dict]:
        return [t.to_openrouter_format() for t in self._tools.values()]

    async def execute(self, name: str, args: Dict) -> Any:
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"工具不存在: {name}"}
        try:
            return await tool.handler(args)
        except Exception as e:
            return {"error": f"執行失敗: {str(e)}"}


# ============================================================
# 內建工具集
# ============================================================
def create_builtin_registry() -> ToolRegistry:
    registry = ToolRegistry()

    # --- 代碼執行 ---
    registry.register(Tool(
        name="execute_python",
        description="執行 Python 代碼並返回結果。支援 print、返回值、異常捕獲。",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要執行的 Python 代碼"},
                "timeout": {"type": "integer", "description": "超時秒數", "default": 30},
            },
            "required": ["code"],
        },
        handler=_execute_python,
    ))

    registry.register(Tool(
        name="execute_shell",
        description="執行 Shell 命令並返回 stdout/stderr。",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell 命令"},
                "cwd": {"type": "string", "description": "工作目錄"},
                "timeout": {"type": "integer", "default": 60},
            },
            "required": ["command"],
        },
        handler=_execute_shell,
    ))

    # --- 文件操作 ---
    registry.register(Tool(
        name="read_file",
        description="讀取文件內容。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路徑"},
                "encoding": {"type": "string", "default": "utf-8"},
            },
            "required": ["path"],
        },
        handler=_read_file,
    ))

    registry.register(Tool(
        name="write_file",
        description="寫入文件內容。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路徑"},
                "content": {"type": "string", "description": "文件內容"},
                "encoding": {"type": "string", "default": "utf-8"},
            },
            "required": ["path", "content"],
        },
        handler=_write_file,
    ))

    registry.register(Tool(
        name="list_files",
        description="列出目錄文件。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目錄路徑", "default": "."},
                "pattern": {"type": "string", "description": "glob 模式", "default": "**/*"},
            },
            "required": [],
        },
        handler=_list_files,
    ))

    # --- 網絡請求 ---
    registry.register(Tool(
        name="http_request",
        description="發送 HTTP 請求。",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "請求 URL"},
                "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"], "default": "GET"},
                "headers": {"type": "object"},
                "json_body": {"type": "object"},
                "timeout": {"type": "integer", "default": 30},
            },
            "required": ["url"],
        },
        handler=_http_request,
    ))

    # --- 時間 ---
    registry.register(Tool(
        name="get_current_time",
        description="獲取當前時間。",
        parameters={"type": "object", "properties": {}},
        handler=_get_current_time,
    ))

    return registry


# ============================================================
# 工具處理器實現
# ============================================================
async def _execute_python(args: Dict) -> Dict:
    code = args["code"]
    timeout = args.get("timeout", 30)

    # 建立臨時文件執行 (比 exec 更安全，支援導入)
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, temp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {"error": f"執行超時 ({timeout}s)", "stdout": "", "stderr": "Timeout"}

        return {
            "returncode": proc.returncode,
            "stdout": stdout.decode('utf-8', errors='replace'),
            "stderr": stderr.decode('utf-8', errors='replace'),
        }
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass


async def _execute_shell(args: Dict) -> Dict:
    command = args["command"]
    cwd = args.get("cwd")
    timeout = args.get("timeout", 60)

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return {"error": f"執行超時 ({timeout}s)"}

    return {
        "returncode": proc.returncode,
        "stdout": stdout.decode('utf-8', errors='replace'),
        "stderr": stderr.decode('utf-8', errors='replace'),
    }


async def _read_file(args: Dict) -> Dict:
    path = args["path"]
    encoding = args.get("encoding", "utf-8")
    try:
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()
        return {"content": content, "size": len(content)}
    except Exception as e:
        return {"error": str(e)}


async def _write_file(args: Dict) -> Dict:
    path = args["path"]
    content = args["content"]
    encoding = args.get("encoding", "utf-8")
    try:
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
        return {"success": True, "bytes": len(content.encode(encoding))}
    except Exception as e:
        return {"error": str(e)}


async def _list_files(args: Dict) -> Dict:
    path = args.get("path", ".")
    pattern = args.get("pattern", "**/*")
    try:
        from pathlib import Path
        files = list(Path(path).glob(pattern))
        return {
            "files": [
                {"name": f.name, "path": str(f), "is_dir": f.is_dir(), "size": f.stat().st_size if f.is_file() else 0}
                for f in files
            ],
            "count": len(files),
        }
    except Exception as e:
        return {"error": str(e)}


async def _http_request(args: Dict) -> Dict:
    import httpx
    url = args["url"]
    method = args.get("method", "GET")
    headers = args.get("headers", {})
    json_body = args.get("json_body")
    timeout = args.get("timeout", 30)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, url, headers=headers, json=json_body)
            return {
                "status": resp.status_code,
                "headers": dict(resp.headers),
                "json": resp.json() if resp.headers.get("content-type", "").startswith("application/json") else None,
                "text": resp.text,
            }
    except Exception as e:
        return {"error": str(e)}


async def _get_current_time(args: Dict) -> Dict:
    now = datetime.now()
    return {
        "iso": now.isoformat(),
        "timestamp": now.timestamp(),
        "timezone": str(now.astimezone().tzinfo),
    }


# ============================================================
# 組合工具：代碼 Agent 專用
# ============================================================
def create_coding_agent_registry(workspace: str = ".") -> ToolRegistry:
    """創建針對代碼任務的工具集"""
    registry = create_builtin_registry()

    # 限制工作目錄
    original_read = registry.get("read_file").handler
    original_write = registry.get("write_file").handler
    original_list = registry.get("list_files").handler
    original_shell = registry.get("execute_shell").handler

    async def scoped_read(args):
        args["path"] = str(Path(workspace) / args["path"])
        return await original_read(args)

    async def scoped_write(args):
        args["path"] = str(Path(workspace) / args["path"])
        return await original_write(args)

    async def scoped_list(args):
        args["path"] = str(Path(workspace) / args.get("path", "."))
        return await original_list(args)

    async def scoped_shell(args):
        args["cwd"] = workspace
        return await original_shell(args)

    from pathlib import Path
    registry.get("read_file").handler = scoped_read
    registry.get("write_file").handler = scoped_write
    registry.get("list_files").handler = scoped_list
    registry.get("execute_shell").handler = scoped_shell

    # 添加代碼專用工具
    registry.register(Tool(
        name="run_tests",
        description="運行項目測試 (pytest, npm test, go test 等自動檢測)",
        parameters={
            "type": "object",
            "properties": {
                "test_path": {"type": "string", "description": "測試路徑", "default": ""},
                "extra_args": {"type": "string", "description": "額外參數"},
            },
        },
        handler=_run_tests,
    ))

    registry.register(Tool(
        name="lint_code",
        description="代碼靜態檢查 (ruff, eslint, go vet 等)",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
            },
        },
        handler=_lint_code,
    ))

    return registry


async def _run_tests(args: Dict) -> Dict:
    test_path = args.get("test_path", "")
    extra = args.get("extra_args", "")
    cmds = [
        f"pytest {test_path} {extra} -v",
        f"npm test -- {extra}",
        f"go test ./... {extra}",
    ]
    for cmd in cmds:
        result = await _execute_shell({"command": cmd, "timeout": 120})
        if result["returncode"] == 0 or "not found" not in result["stderr"]:
            return result
    return {"error": "未找到測試框架"}


async def _lint_code(args: Dict) -> Dict:
    path = args.get("path", ".")
    cmds = [
        f"ruff check {path}",
        f"eslint {path}",
        f"go vet {path}/...",
    ]
    results = []
    for cmd in cmds:
        result = await _execute_shell({"command": cmd, "timeout": 60})
        results.append({"cmd": cmd, **result})
    return {"results": results}