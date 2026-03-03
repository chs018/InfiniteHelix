"""
Configuration management for Infinite Helix application.
Loads settings from environment variables with validation.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List
import os
from pathlib import Path
from dotenv import load_dotenv

# Get the backend directory path
BACKEND_DIR = Path(__file__).parent.parent
ENV_FILE = BACKEND_DIR / ".env"

# Load environment variables from .env file
load_dotenv(ENV_FILE)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = Field(default="Infinite Helix", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="production", env="ENVIRONMENT")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS
    allowed_origins: str = Field(
        default="http://localhost:3000",
        env="ALLOWED_ORIGINS"
    )
    allowed_methods: str = Field(
        default="GET,POST,PUT,DELETE,OPTIONS",
        env="ALLOWED_METHODS"
    )
    allowed_headers: str = Field(default="*", env="ALLOWED_HEADERS")
    
    # File Upload
    max_upload_size: int = Field(default=10485760, env="MAX_UPLOAD_SIZE")  # 10MB
    allowed_extensions: str = Field(
        default="pdf,png,jpg,jpeg,txt",
        env="ALLOWED_EXTENSIONS"
    )
    upload_dir: str = Field(default="uploads", env="UPLOAD_DIR")
    
    # OCR
    tesseract_path: str = Field(default="tesseract", env="TESSERACT_PATH")
    ocr_language: str = Field(default="eng", env="OCR_LANGUAGE")
    
    # NLP
    spacy_model: str = Field(default="en_core_web_md", env="SPACY_MODEL")
    use_gpu: bool = Field(default=False, env="USE_GPU")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")
    
    @validator("allowed_origins")
    def parse_origins(cls, v):
        """Parse comma-separated origins into a list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("allowed_methods")
    def parse_methods(cls, v):
        """Parse comma-separated methods into a list."""
        if isinstance(v, str):
            return [method.strip() for method in v.split(",")]
        return v
    
    @validator("allowed_extensions")
    def parse_extensions(cls, v):
        """Parse comma-separated extensions into a list."""
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",")]
        return v
    
    @validator("upload_dir")
    def create_upload_dir(cls, v):
        """Create upload directory if it doesn't exist."""
        os.makedirs(v, exist_ok=True)
        return v
    
    class Config:
        env_file = str(ENV_FILE)
        env_file_encoding = 'utf-8'
        case_sensitive = False
        extra = 'ignore'  # Ignore extra fields from .env


# Global settings instance
settings = Settings()
