"""
Application configuration management.
"""
from typing import List, Literal
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        # Ignore .env file errors - use defaults if file is missing or invalid
        env_ignore_empty=True
    )
    
    # Application info
    APP_NAME: str = Field(default="MedGemma Note Backend", description="Application name")
    VERSION: str = Field(default="0.1.0", description="Application version")
    ENV: Literal["dev", "prod"] = Field(default="dev", description="Environment (dev/prod)")
    
    # API
    API_V1_STR: str = Field(default="/api/v1", description="API v1 prefix")
    HOST: str = Field(default="0.0.0.0", description="Host to bind")
    PORT: int = Field(default=8000, description="Port to bind")
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="List of allowed CORS origins"
    )
    
    # Model configuration
    MODEL_ID: str = Field(
        default="google/medgemma-4b-it",  # Medical model - new fix validates against embedding size
        description="Model identifier (HuggingFace model ID or path)"
    )
    DEVICE: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        description="Device to run model on (auto/cpu/cuda/mps)"
    )
    DEMO_MODE: bool = Field(
        default=False,
        description="Enable demo mode with stubbed responses (no model loading)"
    )
    MAX_NEW_TOKENS: int = Field(
        default=800,
        description="Maximum new tokens to generate"
    )
    TEMPERATURE: float = Field(
        default=0.2,
        description="Sampling temperature"
    )
    TOP_P: float = Field(
        default=0.9,
        description="Nucleus sampling top-p parameter"
    )
    
    # Prompt templates
    PROMPT_DIR: Path = Field(
        default=Path("app/templates"),
        description="Directory path for prompt templates"
    )
    
    # Data storage
    STORE_REQUESTS: bool = Field(
        default=False,
        description="Whether to store request/response data"
    )
    SQLITE_PATH: Path = Field(
        default=Path("./data/app.db"),
        description="Path to SQLite database file"
    )
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    
    # CUDA debugging
    CUDA_LAUNCH_BLOCKING: bool = Field(
        default=False,
        description="Enable CUDA launch blocking for debugging (set to '1' to enable)"
    )
    TORCH_USE_CUDA_DSA: bool = Field(
        default=False,
        description="Enable CUDA device-side assertions (set to '1' to enable)"
    )
    
    @property
    def DEBUG(self) -> bool:
        """Debug mode is enabled in dev environment."""
        return self.ENV == "dev"
    
    @property
    def PROJECT_NAME(self) -> str:
        """Backward compatibility alias for APP_NAME."""
        return self.APP_NAME


settings = Settings()
