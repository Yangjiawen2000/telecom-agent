from app.tools.registry import ToolRegistry
from app.tools.clients import (
    get_user_info, get_plans, create_order, get_bill, recharge_phone, 
    get_usage_info, get_billing_history, get_broadband, suspend_phone, 
    unsuspend_phone, check_portability, change_plan
)

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
        "查询最近一笔账单详情", 
        {"phone": "str"}
    )
    registry.register(
        "recharge_phone", 
        recharge_phone, 
        "为手机号充值话费", 
        {"phone": "str", "amount": "float"}
    )
    registry.register(
        "get_usage_info", 
        get_usage_info, 
        "查询套餐余量和使用情况（流量、通话等）", 
        {"phone": "str"}
    )
    registry.register(
        "get_billing_history", 
        get_billing_history, 
        "查询历史账单列表（过去几个月）", 
        {"phone": "str"}
    )
    registry.register(
        "get_broadband", 
        get_broadband, 
        "查询宽带状态、地址与光纤覆盖详情", 
        {"phone": "str"}
    )
    registry.register(
        "suspend_phone", 
        suspend_phone, 
        "办理手机挂失或停机保号", 
        {"phone": "str", "reason": "str"}
    )
    registry.register(
        "unsuspend_phone", 
        unsuspend_phone, 
        "办理解挂或手机复机", 
        {"phone": "str"}
    )
    registry.register(
        "check_portability", 
        check_portability, 
        "查询当前手机号是否符合携号转网（转出）要求", 
        {"phone": "str"}
    )
    registry.register(
        "change_plan", 
        change_plan, 
        "办理套餐变更", 
        {"phone": "str", "target_plan_id": "str"}
    )
