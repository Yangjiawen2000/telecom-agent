from app.tools.registry import ToolRegistry
from app.tools.clients import get_user_info, get_plans, create_order, get_bill

def register_all_tools(registry: ToolRegistry):
    registry.register(
        "get_user_info", 
        get_user_info, 
        "获取用户信息", 
        {"phone": "str"}
    )
    registry.register(
        "get_plans", 
        get_plans, 
        "获取所有可用套餐", 
        {}
    )
    registry.register(
        "create_order", 
        create_order, 
        "办理业务/下订单", 
        {"phone": "str", "plan_id": "str", "type": "str"}
    )
    registry.register(
        "get_bill", 
        get_bill, 
        "查询账单信息", 
        {"phone": "str"}
    )
