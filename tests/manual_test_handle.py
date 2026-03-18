import asyncio
import json
import logging
from app.agents.handle_agent import HandleAgent
from app.tools.registry import ToolRegistry

# 设置日志级别，避免干扰
logging.basicConfig(level=logging.ERROR)

# 复用之前写的极简 MockSTM
class MockSTM:
    def __init__(self):
        self.history = []
        
    async def get_history(self):
        return self.history
        
    async def add_message(self, role, content, metadata=None):
        self.history.append({
            "role": role, 
            "content": content, 
            "metadata": metadata or {}
        })

async def main():
    agent = HandleAgent()
    stm = MockSTM()
    
    # 注入一个 Mock 的工具注册表，模拟下单成功，避免报错
    registry = ToolRegistry()
    async def mock_create_order(**params):
        print(f"\n[系统底层] 正在调用 create_order API，参数: {params}")
        # 模拟网络延迟
        await asyncio.sleep(1)
        from app.tools.registry import ToolResult
        return ToolResult(success=True, data={"order_id": "ORD-20260318-8888"})
        
    registry.register("create_order", mock_create_order, "创建订单", {})
    agent.tool_registry = registry

    print("=== 🚀 办理专家交互式手动测试环境已启动 ===")
    print("你可以开始输入需求了 (输入 'quit' 退出)\n")
    
    while True:
        try:
            user_input = input("👤 用户: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            print("🤖 Agent思考中...")
            result = await agent.run(user_input, "sess_manual", "user_01", stm)
            
            # 处理转向 QA Agent 的情况
            if "need_switch" in result:
                print(f"⚠️ 触发转接: {result['reason']}")
                break

            state = result.get('state')
            message = result.get('message')
            form_data = result.get('form_data', {})
            
            print(f"✅ 状态流转至: [{state}]")
            print(f"📦 当前收集表单: {form_data}")
            
            if state == "CONFIRMING" and result.get("card"):
                print(f"🃏 前端渲染卡片: {json.dumps(result['card'], ensure_ascii=False)}")
                
            print(f"💬 回复: {message}\n")
            print("-" * 50)
            
            # 将本轮状态写入历史，供下一轮使用
            await stm.add_message("assistant", message, {
                "handle_state": {
                    "state": state, 
                    "form_data": form_data
                }
            })
            
            if state == "DONE":
                print("🎉 办理流程结束！测试通过。")
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 运行出错: {e}")

if __name__ == "__main__":
    asyncio.run(main())