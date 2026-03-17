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
