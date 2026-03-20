import httpx
import logging
from typing import Dict, Any, List, Optional
from app.config import settings
from app.schemas.business import UserInfo, Bill, UsageInfo, RechargeResponse

logger = logging.getLogger(__name__)

async def get_user_info(phone: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://{settings.MOCK_API_HOST}:{settings.MOCK_API_PORT}/user/{phone}", timeout=5.0)
        resp.raise_for_status()
        return resp.json()

async def get_plans() -> List[Dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://{settings.MOCK_API_HOST}:{settings.MOCK_API_PORT}/plans", timeout=5.0)
        resp.raise_for_status()
        return resp.json()

async def create_order(phone: str, plan_id: str, type: str = "new", **kwargs) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        payload = {"phone": phone, "plan_id": plan_id, "order_type": type}
        resp = await client.post(f"http://{settings.MOCK_API_HOST}:{settings.MOCK_API_PORT}/order/create", json=payload, timeout=5.0)
        resp.raise_for_status()
        return resp.json()

async def get_bill(phone: str) -> Dict[str, Any]:
    """获取最近一笔账单"""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://{settings.MOCK_API_HOST}:{settings.MOCK_API_PORT}/user/{phone}/bills", timeout=5.0)
        resp.raise_for_status()
        bills = resp.json()
        return bills[0] if bills else {}

async def recharge_phone(phone: str, amount: float) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        payload = {"phone": phone, "amount": amount}
        resp = await client.post(f"http://{settings.MOCK_API_HOST}:{settings.MOCK_API_PORT}/user/recharge", json=payload, timeout=5.0)
        resp.raise_for_status()
        return resp.json()

async def get_usage_info(phone: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://{settings.MOCK_API_HOST}:{settings.MOCK_API_PORT}/user/{phone}/usage", timeout=5.0)
        resp.raise_for_status()
        return resp.json()

async def get_billing_history(phone: str) -> List[Dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://{settings.MOCK_API_HOST}:{settings.MOCK_API_PORT}/user/{phone}/bills", timeout=5.0)
        resp.raise_for_status()
        return resp.json()

async def get_broadband(phone: str) -> Dict[str, Any]:
    """查询宽带状态及覆盖详情"""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://{settings.MOCK_API_HOST}:{settings.MOCK_API_PORT}/broadband/{phone}", timeout=5.0)
        resp.raise_for_status()
        return resp.json()

async def suspend_phone(phone: str, reason: str = "挂失") -> Dict[str, Any]:
    """办理手机停机保号/挂失"""
    async with httpx.AsyncClient() as client:
        payload = {"phone": phone, "reason": reason}
        resp = await client.post(f"http://{settings.MOCK_API_HOST}:{settings.MOCK_API_PORT}/user/suspend", json=payload, timeout=5.0)
        resp.raise_for_status()
        return resp.json()

async def unsuspend_phone(phone: str) -> Dict[str, Any]:
    """办理手机复机"""
    async with httpx.AsyncClient() as client:
        payload = {"phone": phone}
        resp = await client.post(f"http://{settings.MOCK_API_HOST}:{settings.MOCK_API_PORT}/user/unsuspend", json=payload, timeout=5.0)
        resp.raise_for_status()
        return resp.json()

async def check_portability(phone: str) -> Dict[str, Any]:
    """查询手机号是否符合携号转网资格"""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://{settings.MOCK_API_HOST}:{settings.MOCK_API_PORT}/portability/check", params={"phone": phone}, timeout=5.0)
        resp.raise_for_status()
        return resp.json()

async def change_plan(phone: str, target_plan_id: str) -> Dict[str, Any]:
    """办理套餐变更"""
    async with httpx.AsyncClient() as client:
        payload = {"phone": phone, "target_plan_id": target_plan_id}
        resp = await client.post(f"http://{settings.MOCK_API_HOST}:{settings.MOCK_API_PORT}/plan/change", json=payload, timeout=5.0)
        resp.raise_for_status()
        return resp.json()
