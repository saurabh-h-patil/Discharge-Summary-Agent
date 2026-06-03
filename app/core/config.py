"""
Configuration module — loads settings from environment variables / .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")
    openai_vision_model: str = Field(default="gpt-4o", env="OPENAI_VISION_MODEL")

    # Agent
    max_agent_iterations: int = Field(default=30, env="MAX_AGENT_ITERATIONS")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    vision_max_tokens: int = Field(default=4096, env="VISION_MAX_TOKENS")

    # Paths
    output_dir: str = Field(default="output", env="OUTPUT_DIR")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
