from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "telecom-agent"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Database Settings
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "telecom_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # Milvus Settings
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # LLM Settings
    LLM_PROVIDER: str = "kimi"  # kimi or qwen
    KIMI_API_KEY: str = "dummy"
    KIMI_BASE_URL: str = "https://api.moonshot.cn/v1"
    QWEN_API_KEY: str = "dummy"
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # Models
    KIMI_MODEL: str = "kimi-k2.5"
    QWEN_MODEL: str = "qwen-max"
    EMBEDDING_MODEL: str = "text-embedding-v4"

    # Security
    SECRET_KEY: str = "secret"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
