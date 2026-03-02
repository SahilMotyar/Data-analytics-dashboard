from pathlib import Path
from pydantic import BaseModel
import os
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class Settings(BaseModel):
    app_name: str = "DataLens API"
    base_dir: Path = Path(__file__).resolve().parents[2]
    storage_dir: Path = base_dir / "storage"
    uploads_dir: Path = storage_dir / "uploads"
    outputs_dir: Path = storage_dir / "outputs"
    max_upload_mb: int = 50
    celery_broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    celery_result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    celery_task_always_eager: bool = os.getenv("CELERY_TASK_ALWAYS_EAGER", "true").lower() == "true"
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = "claude-sonnet-4-20250514"
    share_secret: str = os.getenv("SHARE_SECRET", "datalens-dev-secret")
    upload_daily_limit: int = 50


settings = Settings()
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.outputs_dir.mkdir(parents=True, exist_ok=True)
