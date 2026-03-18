import httpx
import logging
from typing import Dict, Any, List, Optional
from app.config import settings

logger = logging.getLogger(__name__)

async def get_user_info(phone: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://localhost:8001/user/{phone}", timeout=5.0)
        resp.raise_for_status()
        return resp.json()

async def get_plans() -> List[Dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8001/plans", timeout=5.0)
        resp.raise_for_status()
        return resp.json()

async def create_order(phone: str, plan_id: str, type: str = "new") -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        payload = {"phone": phone, "plan_id": plan_id, "type": type}
        resp = await client.post("http://localhost:8001/order/create", json=payload, timeout=5.0)
        resp.raise_for_status()
        return resp.json()

async def get_bill(phone: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://localhost:8001/bill/{phone}", timeout=5.0)
        resp.raise_for_status()
        return resp.json()
