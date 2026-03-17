import httpx
import json
import asyncio

async def chat_test():
    url = "http://127.0.0.1:8000/api/v1/chat/message"
    payload = {
        "session_id": "manual_test_001",
        "user_id": "user_186",
        "message": "你好，我想查一下我的套餐，手机号是 18612345678"
    }
    
    print(f"Sending message: {payload['message']}\n")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    return
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        event_type = data.get("type")
                        content = data.get("content", "")
                        
                        if event_type == "thinking":
                            print(f"[Thinking]: {content}")
                        elif event_type == "token":
                            print(content, end="", flush=True)
                        elif event_type == "done":
                            print(f"\n\n[Done] Intent: {data.get('intent')}")
    except Exception as e:
        print(f"\nConnection failed: {e}")
        print("Please ensure 'python -m app.main' is running on port 8000.")

if __name__ == "__main__":
    asyncio.run(chat_test())
