import json
import asyncio
import os
import re
import logging
from typing import List, Dict, Any
from app.llm import chat

logger = logging.getLogger(__name__)

# Prompt for entity-relationship extraction with causal awareness
EXTRACTION_PROMPT = """
你是一个电信业务知识专家。请从以下文本中提取实体（Entity）以及它们之间的关系（Relationship），特别关注**因果关系**和**触发条件**。

输出格式必须是严格的 JSON 格式，包含以下字段：
1. "entities": 实体列表，每个实体包含 "name" (名称) 和 "type" (类型，如：套餐、费用、业务、规则、用户类型、条件、状态、操作)。
2. "relationships": 关系列表，每个关系包含 "source" (源实体名称), "target" (目标实体名称), "relation" (关系类型)。
   特别增加以下关系类型用于因果推理：
   - "导致" (Cause -> Effect): A 现象或操作直接引发 B 结果。
   - "触发" (Trigger): A 条件满足时，B 动作被执行。
   - "前提" (Precondition): A 是 B 发生或执行的必要条件。
   - "由于" (Reason): B 的发生是因为 A。
   - 以及常规关系：包含, 要求, 冲突, 适用于, 属于, 赠送, 计费。

文本内容：
{text}

注意：
- 识别隐含的因果逻辑（如“若...则...”，“由于...导致...”，“发生...是因为...”）。
- 确保实体名称在 entities 和 relationships 中保持一致。
- 如果没有提取到任何内容，返回空的 entities 和 relationships 列表。
- 不要输出 Markdown 块，只输出原始 JSON 字符串。
"""

class GraphExtractor:
    def __init__(self):
        pass

    async def extract_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取实体和关系"""
        prompt = EXTRACTION_PROMPT.format(text=text)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await chat(messages, temperature=0.1)
            # 找到 JSON 部分
            match = re.search(r'(\{.*\})', response, re.DOTALL)
            if match:
                json_str = match.group(1)
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in LLM response")
                return {"entities": [], "relationships": []}
        except Exception as e:
            logger.error(f"Error extracting graph from text: {e}")
            return {"entities": [], "relationships": []}

    async def process_knowledge_base(self, kb_dir: str) -> Dict[str, Any]:
        """处理整个知识库目录"""
        all_entities = {}
        all_relationships = []
        
        if not os.path.exists(kb_dir):
            logger.error(f"Knowledge directory {kb_dir} does not exist.")
            return {"entities": [], "relationships": []}

        for filename in os.listdir(kb_dir):
            if filename.endswith(".md"):
                file_path = os.path.join(kb_dir, filename)
                logger.info(f"Processing file: {filename}")
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # 分块逻辑：按二级或三级标题切分
                chunks = re.split(r'\n(?=#{2,3} )', content)
                
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    
                    logger.info(f"  Processing chunk {i+1}/{len(chunks)}...")
                    result = await self.extract_from_text(chunk)
                    
                    # 合并实体（去重）
                    for entity in result.get("entities", []):
                        name = entity["name"]
                        if name not in all_entities:
                            all_entities[name] = entity
                        
                    # 合并关系
                    all_relationships.extend(result.get("relationships", []))
                    
        return {
            "entities": list(all_entities.values()),
            "relationships": all_relationships
        }

async def main():
    # 方便测试的入口
    extractor = GraphExtractor()
    kb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../knowledge"))
    print(f"Starting extraction from {kb_path}...")
    graph = await extractor.process_knowledge_base(kb_path)
    
    output_path = os.path.join(kb_path, "graph.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    
    print(f"Extracted {len(graph['entities'])} entities and {len(graph['relationships'])} relationships.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
