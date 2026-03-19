import asyncio
import httpx
import json

async def run_memory_test():
    session_id = "test_memory_session_001"
    user_id = "user_test_001"
    url = "http://localhost:8000/chat/message"
    clear_url = f"http://localhost:8000/chat/session/{session_id}?user_id={user_id}"

    async with httpx.AsyncClient() as client:
        # 1. Clear previous session first
        await client.delete(clear_url)

        queries = [
            "你好，我叫张三，我的手机号是18612345678。",
            "麻烦帮我查下我的话费。",
            "我的这个大流量卡套餐具体包含多少流量和语音？",
            "顺便问下如果在国外上网怎么收费？",
            "聊了这么半天，你难道忘记我是谁了吗？我叫什么名字？手机号是多少？"
        ]

        print("==================================================")
        print("MULTI-TURN MULTI-HOP MEMORY TEST")
        print("==================================================")

        for i, query in enumerate(queries, 1):
            print(f"\n[Turn {i}] USER: {query}")
            
            payload = {
                "session_id": session_id,
                "user_id": user_id,
                "message": query
            }
            
            # Send message and read streaming response
            full_response = ""
            async with client.stream("POST", url, json=payload, timeout=60.0) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data["type"] == "token":
                            full_response += data["content"]
            
            print(f"[Turn {i}] AGENT: {full_response}")

if __name__ == "__main__":
    asyncio.run(run_memory_test())
