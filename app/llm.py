import httpx
import json
import asyncio
import logging
from typing import AsyncGenerator, List, Dict, Any, Union, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings
import time

logger = logging.getLogger(__name__)

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
    
    def _get_provider_params(self, provider: str) -> Dict[str, Any]:
        if provider == "qwen":
            return {
                "api_key": settings.QWEN_API_KEY,
                "base_url": settings.QWEN_BASE_URL,
                "model": settings.QWEN_MODEL
            }
        else:
            return {
                "api_key": settings.KIMI_API_KEY,
                "base_url": settings.KIMI_BASE_URL,
                "model": settings.KIMI_MODEL
            }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def chat(self, 
                   messages: List[Dict[str, str]], 
                   stream: bool = False, 
                   temperature: float = 1.0,
                   tools: Optional[List[Dict[str, Any]]] = None,
                   model: Optional[str] = None,
                   max_tokens: Optional[int] = None) -> Union[str, AsyncGenerator[str, None], Dict[str, Any]]:
        
        providers = [self.provider]
        # 如果当前是 kimi，把 qwen 作为备选
        if self.provider == "kimi":
            providers.append("qwen")
        
        last_error = None
        for p in providers:
            params = self._get_provider_params(p)
            final_model = model or params["model"]
            # For kimi-k2.5, temperature must be 1.0
            final_temp = 1.0 if "kimi-k2.5" in final_model else temperature
            
            payload = {
                "model": final_model,
                "messages": messages,
                "stream": stream,
                "temperature": final_temp
            }
            if tools:
                payload["tools"] = tools
            if max_tokens:
                payload["max_tokens"] = max_tokens
            
            headers = {
                "Authorization": f"Bearer {params['api_key']}",
                "Content-Type": "application/json"
            }

            try:
                if not stream:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        response = await client.post(
                            f"{params['base_url']}/chat/completions",
                            headers=headers,
                            json=payload
                        )
                        # 如果触发 429 且有备选供应商，则尝试下一个
                        if response.status_code == 429 and p != providers[-1]:
                            logger.warning(f"Provider {p} rate limited (429). Falling back to {providers[providers.index(p)+1]}")
                            continue
                            
                        response.raise_for_status()
                        result = response.json()
                        message = result["choices"][0]["message"]
                        
                        return message
                else:
                    return self._stream_chat(headers, payload, params['base_url'], providers, p)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and p != providers[-1]:
                    logger.warning(f"Provider {p} rate limited (429). Falling back to {providers[providers.index(p)+1]}")
                    continue
                last_error = e
                break
            except Exception as e:
                logger.error(f"LLM call error with {p}: {e}")
                if p != providers[-1]:
                    continue
                last_error = e
                break
        
        if last_error is not None:
            raise last_error
        return {"role": "assistant", "content": "抱歉，系统暂时无法响应，请稍后再试。"}

    async def _stream_chat(self, headers: dict, payload: dict, base_url: str, providers: list, current_p: str) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status_code == 429 and current_p != providers[-1]:
                         next_p = providers[providers.index(current_p)+1]
                         logger.warning(f"Stream: Provider {current_p} rate limited. Falling back to {next_p}")
                         params = self._get_provider_params(next_p)
                         new_headers = {"Authorization": f"Bearer {params['api_key']}", "Content-Type": "application/json"}
                         new_payload = payload.copy()
                         new_payload["model"] = params["model"]
                         if "kimi-k2.5" in params["model"]:
                             new_payload["temperature"] = 1.0
                         
                         async for token in self._stream_chat_raw(new_headers, new_payload, params['base_url']):
                             yield token
                         return

                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices", [])
                                if choices:
                                    token = choices[0].get("delta", {}).get("content", "")
                                    if token:
                                        yield token # 立即 yield
                            except:
                                continue
            except Exception as e:
                logger.error(f"Stream error with {current_p}: {e}")
                if current_p != providers[-1]:
                    next_p = providers[providers.index(current_p)+1]
                    params = self._get_provider_params(next_p)
                    new_headers = {"Authorization": f"Bearer {params['api_key']}", "Content-Type": "application/json"}
                    new_payload = payload.copy()
                    new_payload["model"] = params["model"]
                    async for token in self._stream_chat_raw(new_headers, new_payload, params['base_url']):
                        yield token
                else:
                    raise

    async def _stream_chat_raw(self, headers: dict, payload: dict, base_url: str) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{base_url}/chat/completions", headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]": break
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                token = choices[0].get("delta", {}).get("content", "")
                                if token:
                                    yield token
                        except: continue

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
            "Authorization": f"Bearer {settings.QWEN_API_KEY}",
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
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0]["embedding"]
            raise ValueError(f"Unexpected embedding response: {result}")

client = LLMClient()

async def chat(messages: List[Dict[str, str]], 
             stream: bool = False, 
             temperature: float = 1.0,
             tools: Optional[List[Dict[str, Any]]] = None,
             model: Optional[str] = None,
             max_tokens: Optional[int] = None):
    return await client.chat(messages, stream, temperature, tools, model, max_tokens)

async def embed(text: str):
    return await client.embed(text)
