import httpx
import json
import asyncio
import time

BASE_URL = "http://localhost:8000"

async def test_broadband_flow():
    session_id = f"test_broadband_{int(time.time())}"
    user_id = "user_broadband_test"

    # Step 0: Clear session
    async with httpx.AsyncClient() as client:
        await client.delete(f"{BASE_URL}/chat/session/{session_id}?user_id={user_id}")

    turns = [
        "我想办宽带",
        "我叫小杨，身份证 1238128382147，手机号 123782147921412，帮我推荐个套餐",
        "选那个39元的套餐吧",
        "确认"
    ]

    print("\n" + "="*50)
    print("BROADBAND APPLICATION FLOW TEST")
    print("="*50 + "\n")

    async with httpx.AsyncClient(timeout=300.0) as client:
        for i, user_input in enumerate(turns):
            print(f"[Turn {i+1}] USER: {user_input}")
            
            payload = {
                "session_id": session_id,
                "user_id": user_id,
                "message": user_input
            }
            
            response_text = ""
            async with client.stream("POST", f"{BASE_URL}/chat/message", json=payload) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data["type"] == "token":
                            response_text += data["content"]
            
            print(f"[Turn {i+1}] AGENT: {response_text}\n")
            
            # 关键验证点：在第三轮，Agent 不应该再问姓名/身份证
            if i == 2:
                forbidden = ["请提供您的姓名", "提供您的身份证号", "手机号码"]
                for word in forbidden:
                    if word in response_text:
                        print(f"❌ FAILED: Agent requested redundant information: '{word}'")
                        return
                
                success_markers = ["提交", "订单", "确认", "办理成功", "24小时内"]
                if any(m in response_text for m in success_markers):
                    print("✅ SUCCESS: Agent remembered info and submitted order!")
                else:
                    print("⚠️ WARNING: Agent didn't ask redundant info, but haven't seen 'order' markers yet.")

if __name__ == "__main__":
    asyncio.run(test_broadband_flow())
