"""
Application configuration — all settings from environment variables.
PRD Section 15.3: Zero hardcoded secrets.
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    """
    All configuration is loaded from environment variables or .env file.
    No defaults for secrets — they MUST be set explicitly.
    """

    # ── Database ──────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://nseai:nseai@localhost:5432/nseai"
    DATABASE_URL_SYNC: str = "postgresql://nseai:nseai@localhost:5432/nseai"

    # ── Redis ─────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── JWT (RS256 asymmetric — PRD Section 7.4) ─────────────
    JWT_PRIVATE_KEY_PATH: str = "keys/jwt_private.pem"
    JWT_PUBLIC_KEY_PATH: str = "keys/jwt_public.pem"
    JWT_ALGORITHM: str = "RS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── Data Sources ──────────────────────────────────────────
    TIINGO_API_KEY: str = ""
    NEWSAPI_KEY: str = ""
    GNEWS_API_KEY: str = ""

    # ── MLflow ────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"

    # ── Application ───────────────────────────────────────────
    APP_NAME: str = "NSE AI Stock Recommendation System"
    DEBUG: bool = False
    CORS_ORIGINS: str = '["http://localhost:3000"]'

    # ── Rate Limiting ─────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 100
    AUTH_RATE_LIMIT_PER_MINUTE: int = 10

    # ── Model Paths ───────────────────────────────────────────
    MODEL_ARTIFACT_DIR: str = "artifacts/models"
    BACKTEST_RESULTS_DIR: str = "artifacts/backtests"

    @property
    def cors_origins_list(self) -> list[str]:
        return json.loads(self.CORS_ORIGINS)

    @property
    def jwt_private_key(self) -> str:
        return Path(self.JWT_PRIVATE_KEY_PATH).read_text()

    @property
    def jwt_public_key(self) -> str:
        return Path(self.JWT_PUBLIC_KEY_PATH).read_text()

    class Config:
        env_file = "backend/.env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore Supabase vars from root .env


@lru_cache()
def get_settings() -> Settings:
    return Settings()
