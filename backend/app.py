from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import os
from core.config import settings

# Initialize logging
logger.add(settings.get_config_value("paths.log_file", "./assets/logs/api.log"), rotation="10 MB")

app = FastAPI(title=settings.api_title)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_config_value("security.allow_origins", ["*"]),  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=settings.get_config_value("security.allow_methods", ["*"]),
    allow_headers=settings.get_config_value("security.allow_headers", ["*"]),
)

# Load config on startup
@app.on_event("startup")
async def startup_event():
    # Ensure config is loaded
    config_path = settings.get_config_value("paths.config", "config.yaml")
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using defaults")

    # Create necessary directories from centralized config
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.output_dir, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "AI Trainer API is running"}

# Import and include routers
from routers import stream, video
from endpoints import config as config_endpoint

app.include_router(stream.router, prefix="/ws", tags=["stream"])
app.include_router(video.router, prefix="/api/video", tags=["video"])
app.include_router(config_endpoint.router)

# Mount static files
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=settings.output_dir), name="static")
