import sqlite3
import os
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from faker import Faker
import uvicorn
import random
from typing import List, Dict, Any, Optional

from app.schemas.business import (
    UserStatus, UserInfo, OrderType, Order, Bill, UsageInfo, RechargeResponse,
    BroadbandInfo, PortabilityStatus, ActionResponse
)
app = FastAPI(title="Telecom Production Mock System (SQLite)")
fake = Faker(['zh_CN'])
DB_PATH = "mock_business.db"

# --- 数据库初始化 ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            phone TEXT PRIMARY KEY,
            name TEXT,
            status TEXT,
            plan TEXT,
            balance REAL,
            arrears REAL
        )
    ''')
    # 账单表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT,
            month TEXT,
            amount REAL,
            status TEXT
        )
    ''')
    # 插入初始数据 (如果不存在)
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        initial_users = [
            ("18612345678", "张三", "正常", "畅越129套餐", 50.0, 0.0),
            ("123782147921412", "小杨", "正常", "5G尊享199套餐", 159.0, 0.0)
        ]
        cursor.executemany("INSERT INTO users VALUES (?,?,?,?,?,?)", initial_users)
        
        initial_bills = [
            ("123782147921412", "2024-02", 129.0, "已缴纳"),
            ("123782147921412", "2024-01", 135.5, "已缴纳")
        ]
        cursor.executemany("INSERT INTO bills (phone, month, amount, status) VALUES (?,?,?,?)", initial_bills)
        
    conn.commit()
    conn.close()

init_db()

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# --- 模型定义 ---
class OrderCreate(BaseModel):
    phone: str
    plan_id: str
    order_type: str

class RechargeRequest(BaseModel):
    phone: str
    amount: float

class SuspendRequest(BaseModel):
    phone: str
    reason: Optional[str] = "挂失"

class UnsuspendRequest(BaseModel):
    phone: str

class ChangePlanRequest(BaseModel):
    phone: str
    target_plan_id: str

# --- 接口实现 ---

@app.get("/user/{phone}", response_model=UserInfo)
async def get_user(phone: str):
    with get_db() as db:
        user = db.execute("SELECT * FROM users WHERE phone = ?", (phone,)).fetchone()
        if not user:
            # 自动注册新用户方便测试
            name = fake.name()
            db.execute("INSERT INTO users VALUES (?,?,?,?,?,?)", (phone, name, "正常", "默认套餐", 0.0, 0.0))
            db.commit()
            user = db.execute("SELECT * FROM users WHERE phone = ?", (phone,)).fetchone()
        return dict(user)

@app.get("/plans")
async def get_plans():
    return [
        {"id": "v129", "name": "畅越129套餐", "price": 129, "data": "30GB", "voice": "500分钟"},
        {"id": "v199", "name": "5G尊享199套餐", "price": 199, "data": "60GB", "voice": "1000分钟"},
        {"id": "v39", "name": "大流量卡39元", "price": 39, "data": "100GB", "voice": "0分钟"}
    ]

@app.post("/order/create")
async def create_order(order: OrderCreate):
    return {
        "order_id": f"ORD{random.randint(10000, 99999)}",
        "status": "已提交",
        "phone": order.phone,
        "plan_id": order.plan_id
    }

@app.post("/user/recharge", response_model=RechargeResponse)
async def recharge_user(req: RechargeRequest):
    with get_db() as db:
        user = db.execute("SELECT * FROM users WHERE phone = ?", (req.phone,)).fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        new_balance = user["balance"] + req.amount
        db.execute("UPDATE users SET balance = ? WHERE phone = ?", (new_balance, req.phone))
        db.commit()
        
        return {
            "phone": req.phone,
            "amount": req.amount,
            "status": "success",
            "new_balance": new_balance,
            "transaction_id": f"REC{random.randint(100000, 999999)}"
        }

@app.get("/user/{phone}/usage", response_model=UsageInfo)
async def get_usage(phone: str):
    # 用量暂时保持随机生成，但基于手机号种子保持一致性
    random.seed(int(phone) if phone.isdigit() else 0)
    used = random.uniform(5, 50)
    return {
        "phone": phone,
        "data_used": f"{used:.2f}GB",
        "data_total": "100GB",
        "voice_used": f"{random.randint(10, 500)}分钟",
        "voice_total": "1000分钟",
        "data_percentage": f"{int(used)}%"
    }

@app.get("/user/{phone}/bills", response_model=List[Bill])
async def get_bill_history(phone: str):
    with get_db() as db:
        bills = db.execute("SELECT * FROM bills WHERE phone = ? ORDER BY month DESC", (phone,)).fetchall()
        return [dict(b) for b in bills]

@app.get("/broadband/{phone}", response_model=BroadbandInfo)
async def get_broadband(phone: str):
    # 用手机号随机生成宽带状态
    random.seed(int(phone) if phone.isdigit() else 0)
    has_broadband = random.choice([True, False])
    return {
        "phone": phone,
        "has_broadband": has_broadband,
        "address": "XX市YY区ZZ街道" if has_broadband else None,
        "status": "正常" if has_broadband else None,
        "speed": "1000M" if has_broadband else None,
        "fiber_coverage": True
    }

@app.post("/user/suspend", response_model=ActionResponse)
async def suspend_user(req: SuspendRequest):
    with get_db() as db:
        user = db.execute("SELECT * FROM users WHERE phone = ?", (req.phone,)).fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        db.execute("UPDATE users SET status = ? WHERE phone = ?", ("报停", req.phone))
        db.commit()
    return {"phone": req.phone, "action": "suspend", "status": "success", "message": f"手机号已成功报停，原因：{req.reason}"}

@app.post("/user/unsuspend", response_model=ActionResponse)
async def unsuspend_user(req: UnsuspendRequest):
    with get_db() as db:
        user = db.execute("SELECT * FROM users WHERE phone = ?", (req.phone,)).fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        db.execute("UPDATE users SET status = ? WHERE phone = ?", ("正常", req.phone))
        db.commit()
    return {"phone": req.phone, "action": "unsuspend", "status": "success", "message": "手机号已成功复机"}

@app.get("/portability/check", response_model=PortabilityStatus)
async def check_portability(phone: str):
    with get_db() as db:
        user = db.execute("SELECT * FROM users WHERE phone = ?", (phone,)).fetchone()
        if not user:
            return {"phone": phone, "eligible": False, "reason": "户口不存在，无法办理"}
        if user["arrears"] > 0 or user["balance"] < 0:
            return {"phone": phone, "eligible": False, "reason": "账户存在欠费，无法办理携号转网"}
        if user["status"] != "正常":
            return {"phone": phone, "eligible": False, "reason": f"账户状态异常({user['status']})，无法办理"}
    return {"phone": phone, "eligible": True, "reason": "符合携号转网条件，可发送短信获取授权码"}

@app.post("/plan/change", response_model=ActionResponse)
async def change_plan(req: ChangePlanRequest):
    with get_db() as db:
        user = db.execute("SELECT * FROM users WHERE phone = ?", (req.phone,)).fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # 为了演示，直接更新用户表中的 plan 字段
        db.execute("UPDATE users SET plan = ? WHERE phone = ?", (f"新套餐({req.target_plan_id})", req.phone))
        db.commit()
    return {"phone": req.phone, "action": "change_plan", "status": "success", "message": f"套餐变更申请已受理，将于次月生效。"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
