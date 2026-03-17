import json
import time
from typing import List, Dict, Any, Optional
from app.llm import chat

class ShortTermMemory:
    """短期记忆模块：基于 Redis 存储对话上下文，支持锚点、蒸馏和快照"""
    
    def __init__(self, session_id: str, redis_client):
        self.session_id = session_id
        self.redis = redis_client
        self.key = f"stm:{session_id}:messages"
        self.ttl = 86400  # 24小时

    async def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """追加一条带元数据的消息"""
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        # 使用自增分数或时间戳作为 field 确保顺序，或者直接用时间戳
        field = str(time.time_ns())
        await self.redis.hset(self.key, field, json.dumps(message))
        await self.redis.expire(self.key, self.ttl)

    async def get_history(self, max_turns: int = 20) -> List[Dict[str, Any]]:
        """获取最近 N 轮对话，格式为消息列表"""
        all_msgs = await self.redis.hgetall(self.key)
        if not all_msgs:
            return []
        
        # 排序并取最后 max_turns 个
        sorted_keys = sorted(all_msgs.keys())
        recent_keys = sorted_keys[-max_turns:] if len(sorted_keys) > max_turns else sorted_keys
        
        messages = []
        for k in recent_keys:
            messages.append(json.loads(all_msgs[k]))
        return messages

    async def get_anchors(self) -> List[Dict[str, Any]]:
        """返回所有标记为 anchor 的关键消息"""
        all_msgs = await self.redis.hgetall(self.key)
        anchors = []
        for v in all_msgs.values():
            msg = json.loads(v)
            if msg.get("metadata", {}).get("is_anchor"):
                anchors.append(msg)
        return sorted(anchors, key=lambda x: x["timestamp"])

    async def distill(self, keep_turns: int = 10):
        """记忆蒸馏：将超出 keep_turns 的消息压缩为摘要，保留锚点消息"""
        all_msgs_dict = await self.redis.hgetall(self.key)
        if not all_msgs_dict:
            return

        sorted_keys = sorted(all_msgs_dict.keys())
        if len(sorted_keys) <= keep_turns:
            return

        # 分离需要蒸馏的消息和需要保留的消息
        to_distill_keys = sorted_keys[:-keep_turns]
        keep_keys = sorted_keys[-keep_turns:]

        to_distill_msgs = [json.loads(all_msgs_dict[k]) for k in to_distill_keys]
        
        # 筛选需要保留的锚点消息
        preserved_anchors_dict = {}
        msgs_for_summary = []
        for k in to_distill_keys:
            msg = json.loads(all_msgs_dict[k])
            if msg.get("metadata", {}).get("is_anchor"):
                preserved_anchors_dict[k] = all_msgs_dict[k]
            else:
                msgs_for_summary.append(msg)

        if msgs_for_summary:
            # 调用 LLM 制作摘要
            summary_prompt = "请简要总结以下对话内容，保留核心电信业务需求、用户身份及办理状态：\n\n"
            for m in msgs_for_summary:
                summary_prompt += f"{m['role']}: {m['content']}\n"
            
            summary_text = await chat([{"role": "user", "content": summary_prompt}])
            
            summary_msg = {
                "role": "system",
                "content": f"对话摘要：{summary_text}",
                "is_summary": True,
                "metadata": {},
                "timestamp": time.time()
            }
            
            # 清理旧消息，写入摘要。保留锚点
            await self.redis.delete(self.key)
            
            # 重新写入摘要（作为第一条）
            await self.redis.hset(self.key, str(time.time_ns()), json.dumps(summary_msg))
            
            # 重新写入保留的锚点
            for k, v in preserved_anchors_dict.items():
                await self.redis.hset(self.key, k, v)
                
            # 重新写入最后的 N 条消息
            for k in keep_keys:
                await self.redis.hset(self.key, k, all_msgs_dict[k])
            
            await self.redis.expire(self.key, self.ttl)

    async def snapshot(self) -> Dict[str, str]:
        """快照当前状态"""
        return await self.redis.hgetall(self.key)

    async def clear(self):
        """清除当前会话的所有消息"""
        await self.redis.delete(self.key)

    async def restore(self, snapshot: Dict[str, str]):
        """从快照恢复"""
        await self.redis.delete(self.key)
        if snapshot:
            await self.redis.hset(self.key, mapping=snapshot)
            await self.redis.expire(self.key, self.ttl)
