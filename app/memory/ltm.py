import time
import logging
from typing import List, Dict, Any, Optional
from pymilvus import (
    connections, 
    Collection, 
    FieldSchema, 
    CollectionSchema, 
    DataType, 
    utility
)
from app.config import settings
from app.llm import embed

logger = logging.getLogger(__name__)

class LongTermMemory:
    """长期记忆模块：基于 Milvus 存储业务知识库和用户画像"""
    
    def __init__(self):
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self._connect()
        
    def _connect(self):
        if not connections.has_connection("default"):
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")

    async def init_collections(self):
        """初始化知识库和用户画像 Collection"""
        # 1. Knowledge Base Collection
        kb_fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="created_at", dtype=DataType.INT64)
        ]
        kb_schema = CollectionSchema(kb_fields, "Telecom business knowledge base")
        
        if not utility.has_collection("knowledge_base"):
            kb_col = Collection("knowledge_base", kb_schema)
            # 创建索引
            index_params = {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            kb_col.create_index("embedding", index_params)
            logger.info("Created collection: knowledge_base")
        
        # 2. User Profile Collection
        up_fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="updated_at", dtype=DataType.INT64)
        ]
        up_schema = CollectionSchema(up_fields, "User profile and long-term preferences")
        
        if not utility.has_collection("user_profile"):
            up_col = Collection("user_profile", up_schema)
            index_params = {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            up_col.create_index("embedding", index_params)
            logger.info("Created collection: user_profile")

        # 3. Graph Relationships Collection (New for GraphRAG)
        rel_fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="target", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="relation", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024), # Source+Relation+Target embedding
        ]
        rel_schema = CollectionSchema(rel_fields, "Knowledge graph relationships")
        
        if not utility.has_collection("graph_relationships"):
            rel_col = Collection("graph_relationships", rel_schema)
            index_params = {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            rel_col.create_index("embedding", index_params)
            logger.info("Created collection: graph_relationships")

    async def upsert_knowledge(self, docs: List[Dict[str, Any]]):
        """批量写入知识点"""
        col = Collection("knowledge_base")
        col.load()  # Ensure loaded for consistency or if required by some ops
        # docs format: [{"content": "...", "source": "...", "doc_type": "...", "embedding": [...]}]
        entities = [
            [d["content"] for d in docs],
            [d["embedding"] for d in docs],
            [d.get("source", "unknown") for d in docs],
            [d.get("doc_type", "text") for d in docs],
            [int(time.time()) for _ in docs]
        ]
        col.insert(entities)
        col.flush()

    async def search_knowledge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """语义搜索知识库"""
        query_vector = await embed(query)
        col = Collection("knowledge_base")
        col.load()
        
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = col.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["content", "source", "doc_type"]
        )
        
        ret = []
        for hit in results[0]:
            ret.append({
                "content": hit.entity.get("content"),
                "source": hit.entity.get("source"),
                "doc_type": hit.entity.get("doc_type"),
                "score": hit.score
            })
        return ret

    async def update_user_profile(self, user_id: str, summary: str):
        """更新用户画像摘要"""
        query_vector = await embed(summary)
        col = Collection("user_profile")
        col.load()
        
        # 先删除旧的（如果存在）
        col.delete(f'user_id == "{user_id}"')
        
        entities = [
            [user_id],
            [summary],
            [query_vector],
            [int(time.time())]
        ]
        col.insert(entities)
        col.flush()

    async def get_user_context(self, user_id: str) -> str:
        """获取用户历史背景摘要"""
        col = Collection("user_profile")
        col.load()
        res = col.query(
            expr=f'user_id == "{user_id}"',
            output_fields=["summary"],
            limit=1
        )
        return res[0]["summary"] if res else ""

    async def upsert_graph(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]):
        """写入图数据"""
        # 1. 写入实体到 knowledge_base
        kb_docs = []
        for entity in entities:
            content = f"实体: {entity['name']}, 类型: {entity['type']}"
            embedding = await embed(content)
            kb_docs.append({
                "content": content,
                "source": "graph_extraction",
                "doc_type": "entity",
                "embedding": embedding
            })
        await self.upsert_knowledge(kb_docs)

        # 2. 写入关系到 graph_relationships
        col = Collection("graph_relationships")
        
        data_source = [r["source"] for r in relationships]
        data_target = [r["target"] for r in relationships]
        data_relation = [r["relation"] for r in relationships]
        
        embeddings = []
        for r in relationships:
            desc = f"{r['source']} --({r['relation']})--> {r['target']}"
            embeddings.append(await embed(desc))
            
        entities_to_insert = [
            data_source,
            data_target,
            data_relation,
            embeddings
        ]
        col.insert(entities_to_insert)
        col.flush()

    async def search_related_entities(self, entity_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索与指定实体相关的关系"""
        col = Collection("graph_relationships")
        col.load()
        # 精确匹配（假设名称一致性）或模糊匹配？
        # 这里先尝试精确匹配 source 或 target
        res = col.query(
            expr=f'source == "{entity_name}" or target == "{entity_name}"',
            output_fields=["source", "target", "relation"],
            limit=limit
        )
        return res

    async def search_causal_path(self, entity_name: str, depth: int = 2) -> List[Dict[str, Any]]:
        """
        递归搜索因果路径。
        用于诊断问答：查找导致 A 的原因，或 A 导致的结果。
        """
        causal_relations = ["导致", "触发", "前提", "由于"]
        visited = set()
        results = []
        
        col = Collection("graph_relationships")
        col.load()

        queue = [(entity_name, depth)]
        visited.add(entity_name)

        while queue:
            curr_name, curr_depth = queue.pop(0)
            if curr_depth <= 0:
                continue

            # 查找直接关联的因果关系
            res = col.query(
                expr=f'source == "{curr_name}" or target == "{curr_name}"',
                output_fields=["source", "target", "relation"],
                limit=20
            )

            for r in res:
                if r["relation"] in causal_relations:
                    # 避免重复添加完全相同的关系
                    rel_key = f"{r['source']}-{r['relation']}-{r['target']}"
                    if any(f"{x['source']}-{x['relation']}-{x['target']}" == rel_key for x in results):
                        continue
                        
                    results.append(r)
                    
                    # 确定下一个要探索的实体
                    next_node = r["target"] if r["source"] == curr_name else r["source"]
                    if next_node not in visited:
                        visited.add(next_node)
                        queue.append((next_node, curr_depth - 1))
        
        return results
