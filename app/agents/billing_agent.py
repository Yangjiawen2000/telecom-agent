class BillingAgent:
    """账务专家：查询话费、流量及账单详单"""
    def __init__(self):
        pass

    async def query_bill(self, user_id: str, month: str):
        # TODO: 调用账务接口
        return f"您 {month} 月的账单总计 88.00 元"
