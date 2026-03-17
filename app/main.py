from fastapi import FastAPI
from app.api import chat
from app.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    description="通信运营商多智能体客服系统",
    version="0.1.0"
)

app.include_router(chat.router, prefix="/api/v1", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "Welcome to Telecom Agent API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
