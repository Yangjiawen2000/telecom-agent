import asyncio
import httpx
from app.tools.registry import ToolRegistry
from app.tools.init_tools import register_all_tools

async def test_new_tools():
    registry = ToolRegistry()
    register_all_tools(registry)
    
    test_phone = "18612345678"
    
    print("\n" + "="*50)
    print("MOCK BUSINESS INTERFACE VERIFICATION")
    print("="*50)
    
    # 1. 测试查询用量
    print("\n[1] Testing get_usage_info...")
    res = await registry.call("get_usage_info", {"phone": test_phone})
    print(f"Success: {res.success}")
    if res.success:
        print(f"Data: {res.data}")
    else:
        print(f"Error: {res.error}")
        
    # 2. 测试充值
    print("\n[2] Testing recharge_phone...")
    res = await registry.call("recharge_phone", {"phone": test_phone, "amount": 50.0})
    print(f"Success: {res.success}")
    if res.success:
        print(f"Data: {res.data}")
    else:
        print(f"Error: {res.error}")
        
    # 3. 测试查询账单历史
    print("\n[3] Testing get_billing_history...")
    res = await registry.call("get_billing_history", {"phone": test_phone})
    print(f"Success: {res.success}")
    if res.success:
        print(f"Data: {len(res.data)} items found")
        for bill in res.data:
            print(f"  - {bill['month']}: {bill['amount']} ({bill['status']})")
    else:
        print(f"Error: {res.error}")

    print("\n" + "="*50)

if __name__ == "__main__":
    asyncio.run(test_new_tools())
