import random
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from faker import Faker
import uvicorn

app = FastAPI(title="Telecom Mock Business System")
fake = Faker(['zh_CN'])

# 模拟配置
SIMULATE_ERROR = True  # 设置为 True 会随机返回 500 错误

class OrderCreate(BaseModel):
    phone: str
    plan_id: str
    order_type: str  # e.g., "new_card", "change_plan"

def check_simulate_error():
    if SIMULATE_ERROR and random.random() < 0.2:
        raise HTTPException(status_code=500, detail="Internal Server Error (Simulated)")

@app.get("/user/{phone}")
async def get_user(phone: str):
    check_simulate_error()
    return {
        "phone": phone,
        "name": fake.name(),
        "status": random.choice(["正常", "欠费停机", "报停"]),
        "plan": random.choice(["畅越129套餐", "5G尊享199套餐", "大流量卡39元"]),
        "balance": float(random.randint(-100, 500)),
        "arrears": float(random.randint(0, 50)) if random.random() < 0.3 else 0.0
    }

@app.get("/plans")
async def get_plans():
    check_simulate_error()
    return [
        {"id": "v129", "name": "畅越129套餐", "price": 129, "data": "30GB", "voice": "500分钟"},
        {"id": "v199", "name": "5G尊享199套餐", "price": 199, "data": "60GB", "voice": "1000分钟"},
        {"id": "v39", "name": "大流量卡39元", "price": 39, "data": "100GB", "voice": "0分钟"}
    ]

@app.post("/order/create")
async def create_order(order: OrderCreate):
    check_simulate_error()
    return {
        "order_id": f"ORD{random.randint(10000, 99999)}",
        "status": "已提交",
        "phone": order.phone,
        "plan_id": order.plan_id
    }

@app.get("/order/{order_id}")
async def get_order(order_id: str):
    check_simulate_error()
    return {
        "order_id": order_id,
        "status": random.choice(["处理中", "已完成", "失败"]),
        "update_time": fake.date_time_this_month().isoformat()
    }

@app.post("/user/{phone}/activate")
async def activate_user(phone: str):
    check_simulate_error()
    return {
        "phone": phone,
        "result": "开机指令已发送",
        "status": "success"
    }

@app.get("/bill/{phone}")
async def get_bill(phone: str):
    check_simulate_error()
    return {
        "phone": phone,
        "month": "2024-03",
        "amount": float(random.randint(50, 300)),
        "status": random.choice(["已缴纳", "未缴纳"])
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
