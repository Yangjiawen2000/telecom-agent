import asyncio
import httpx
import json
from app.tools.registry import ToolRegistry

# 我们先写一个简单的测试函数，用来真实请求你跑在 8001 端口的 mock_api
async def mock_get_user(phone: str, simulate_error: bool = False):
    async with httpx.AsyncClient() as client:
        url = f"http://localhost:8001/user/{phone}"
        # 如果传入 simulate_error，就告诉 mock_api 随机抛出 500 错误
        params = {"simulate_error": "true"} if simulate_error else {}
        response = await client.get(url, params=params)
        response.raise_for_status()  # 如果是 500 错误，这里会抛出异常，触发 Registry 的重试机制
        return response.json()

async def main():
    print("初始化 ToolRegistry...\n")
    registry = ToolRegistry()

    # 1. 注册我们的测试工具
    registry.register(
        name="get_user_info",
        func=mock_get_user,
        description="查询用户基本信息（状态/套餐/欠费）",
        params_schema={"phone": "str", "simulate_error": "bool"}
    )

    print("=== 验收项 1: list_tools() 格式清晰 ===")
    tools_list = registry.list_tools()
    print(json.dumps(tools_list, indent=2, ensure_ascii=False))
    print("--------------------------------------------------\n")

    print("=== 验收项 2 & 3: simulate_error=True 容错与重试机制 ===")
    print("👉 正在调用工具，并开启 simulate_error=True 逼迫接口报错...")
    
    # 我们调用刚才注册的工具，故意开启错误模拟
    result = await registry.call(
        name="get_user_info", 
        params={"phone": "18612345678", "simulate_error": True}
    )

    print("\n📦 返回的 ToolResult 对象:")
    print(f"成功状态 (success): {result.success}")
    print(f"重试次数 (retries): {result.retries}")  # 看看是不是重试了
    
    if result.success:
        print(f"最终获取数据: {result.data}")
        print("✅ 结论：虽然遇到了偶发错误，但重试机制成功拿到了数据！")
    else:
        print(f"报错信息 (error): {result.error}")
        print(f"触发兜底 (fallback): {result.fallback}")
        print("✅ 结论：重试耗尽，但系统没有崩溃，成功触发了兜底返回！")

if __name__ == "__main__":
    asyncio.run(main())