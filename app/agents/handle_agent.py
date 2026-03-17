class HandleAgent:
    """办理专家：负责业务变更、报障等事务性操作"""
    def __init__(self):
        pass

    async def handle_request(self, user_id: str, action: str):
        # TODO: 调用业务系统接口办理业务
        return f"已成功受理 {action} 业务"
