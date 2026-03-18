import asyncio
import logging
import time
from typing import Callable, Any, Dict, List, Optional
from pydantic import BaseModel
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)

class ToolResult(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    fallback: Optional[str] = None

class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name: str, func: Callable, description: str, params_schema: Dict[str, Any], backup_func: Optional[Callable] = None):
        """注册工具"""
        self.tools[name] = {
            "func": func,
            "description": description,
            "params_schema": params_schema,
            "backup_func": backup_func
        }
        logger.info(f"Tool registered: {name}")

    async def call(self, name: str, params: Dict[str, Any], max_retries: int = 3) -> ToolResult:
        """统一调用入口，包含重试和容错逻辑"""
        
        start_time = time.time()

        if name not in self.tools:
            return ToolResult(success=False, error=f"Tool {name} not found")

        tool_info = self.tools[name]
        func = tool_info["func"]
        backup_func = tool_info["backup_func"]
        
        attempt_count = 0

        @retry(
            wait=wait_exponential(multiplier=1, min=1, max=5),
            stop=stop_after_attempt(max_retries),
            retry=retry_if_exception_type((Exception, httpx.HTTPStatusError, httpx.TimeoutException, asyncio.TimeoutError)),
            reraise=True
        )
        async def _execute_with_retry():
            nonlocal attempt_count
            attempt_count += 1
            # 5秒超时
            async with asyncio.timeout(5.0):
                if asyncio.iscoroutinefunction(func):
                    return await func(**params)
                else:
                    return func(**params)

        try:
            data = await _execute_with_retry()
            return ToolResult(success=True, data=data, retries=attempt_count-1)
        except Exception as e:
            logger.error(f"Tool {name} failed after {max_retries} retries: {e}")
            
            # 尝试备份函数
            if backup_func:
                logger.info(f"Triggering backup_func for {name}")
                try:
                    data = await backup_func(**params) if asyncio.iscoroutinefunction(backup_func) else backup_func(**params)
                    return ToolResult(success=True, data=data, retries=attempt_count-1, fallback="Backup Function")
                except Exception as backup_err:
                    logger.error(f"Backup function failed: {backup_err}")

            return ToolResult(
                success=False, 
                error=str(e), 
                retries=attempt_count-1, 
                fallback="规则引擎兜底"
            )

    def list_tools(self) -> List[Dict[str, str]]:
        """列出所有工具供 LLM 选择"""
        return [
            {"name": name, "description": info["description"]}
            for name, info in self.tools.items()
        ]
