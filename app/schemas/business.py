from pydantic import BaseModel, Field
from typing import List, Optional, Any
from enum import Enum

class UserStatus(str, Enum):
    NORMAL = "正常"
    ARREARS = "欠费停机"
    SUSPENDED = "报停"

class UserInfo(BaseModel):
    phone: str
    name: str
    status: UserStatus = UserStatus.NORMAL
    plan: str
    balance: float = 0.0
    arrears: float = 0.0

class OrderType(str, Enum):
    NEW_CARD = "new_card"
    CHANGE_PLAN = "change_plan"
    BROADBAND = "broadband"

class Order(BaseModel):
    order_id: str
    phone: str
    plan_id: str
    order_type: OrderType
    status: str = "已提交"

class Bill(BaseModel):
    phone: str
    month: str
    amount: float
    status: str # "已缴纳" | "未缴纳"

class UsageInfo(BaseModel):
    phone: str
    data_used: str # e.g. "35.33GB"
    data_total: str
    voice_used: str # e.g. "470分钟"
    voice_total: str
    data_percentage: str

class RechargeResponse(BaseModel):
    phone: str
    amount: float
    status: str
    new_balance: float
    transaction_id: str

class BroadbandInfo(BaseModel):
    phone: str
    has_broadband: bool
    address: Optional[str] = None
    status: Optional[str] = None
    speed: Optional[str] = None
    fiber_coverage: bool = True

class PortabilityStatus(BaseModel):
    phone: str
    eligible: bool
    reason: Optional[str] = None

class ActionResponse(BaseModel):
    phone: str
    action: str
    status: str
    message: str
