from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    reply: str

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # TODO: 调用 Orchestrator 处理流程
    return ChatResponse(reply=f"接收到消息: {request.message}")
