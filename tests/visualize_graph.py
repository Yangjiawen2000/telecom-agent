import asyncio
import os
import sys
from unittest.mock import MagicMock

# 确保能导入 app
sys.path.append(os.getcwd())

from app.agents.orchestrator import Orchestrator
from app.tools.registry import ToolRegistry

async def main():
    try:
        # 1. 模拟 ToolRegistry 依赖
        mock_registry = MagicMock(spec=ToolRegistry)
        
        # 2. 实例化 Orchestrator 并获取编译后的图
        orchestrator = Orchestrator(registry=mock_registry)
        orchestrator_graph = orchestrator.graph
        
        # 3. 生成 Mermaid 代码
        mermaid_code = orchestrator_graph.get_graph().draw_mermaid()
        
        print("✅ 成功生成 Mermaid 状态图代码！\n")
        
        # 4. 写入 Markdown 文件，方便在 IDE 中预览
        output_file = "graph_view.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# 🤖 智能客服总控路由图\n\n")
            f.write("此图由 `tests/visualize_graph.py` 自动生成。\n\n")
            f.write("```mermaid\n")
            f.write(mermaid_code)
            f.write("\n```\n")
            
        print(f"👉 已保存到根目录的 {output_file} 文件中，你可以在 IDE 中直接打开并预览图形。")
    except Exception as e:
        print(f"❌ 生成失败，请检查 Orchestrator 类是否已正确实现: {e}")

if __name__ == "__main__":
    asyncio.run(main())