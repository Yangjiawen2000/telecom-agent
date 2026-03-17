import httpx
import json
import asyncio
from typing import AsyncGenerator, List, Dict, Any, Union
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings

class LLMClient:
    def __init__(self):
        self.api_key = settings.KIMI_API_KEY
        self.base_url = settings.KIMI_BASE_URL
        self.provider = settings.LLM_PROVIDER
        
        # Model mapping based on provider
        self.model_map = {
            "kimi": settings.KIMI_MODEL,
            "qwen": settings.QWEN_MODEL
        }
    
    @property
    def current_model(self):
        return self.model_map.get(self.provider, settings.KIMI_MODEL)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def chat(self, messages: List[Dict[str, str]], stream: bool = False, temperature: float = 1.0) -> Union[str, AsyncGenerator[str, None]]:
        # For kimi-k2.5, temperature must be 1.0
        final_temp = 1.0 if "kimi-k2.5" in self.current_model else temperature
        
        payload = {
            "model": self.current_model,
            "messages": messages,
            "stream": stream,
            "temperature": final_temp
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            if not stream:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return self._stream_chat(headers, payload)

    async def _stream_chat(self, headers: dict, payload: dict) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            chunk = data["choices"][0]["delta"].get("content", "")
                            if chunk:
                                yield chunk
                        except Exception:
                            continue

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def embed(self, text: str) -> List[float]:
        """使用 DashScope (Qwen) 进行向量化"""
        payload = {
            "model": settings.EMBEDDING_MODEL,
            "input": text,
            "dimensions": 1024  # v3/v4 均支持定制维度
        }
        
        headers = {
            "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            # 使用 DashScope 的 OpenAI 兼容接口
            response = await client.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result["data"][0]["embedding"]

client = LLMClient()

async def chat(messages: List[Dict[str, str]], stream: bool = False, temperature: float = 1.0):
    return await client.chat(messages, stream, temperature)

async def embed(text: str):
    return await client.embed(text)
