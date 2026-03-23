"""
Health check endpoints.
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException
from loguru import logger
import os

from models.responses import HealthResponse
from core.config import settings

router = APIRouter(prefix="/api/v1/health", tags=["Health"])


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.

    Returns the current health status of the API and its dependencies.
    Combines liveness, readiness, and detailed health information.
    """
    try:
        # Check dependencies
        dependencies = {}

        # Check if models exist
        model_path = settings.get_config_value("paths.model", "")
        segmentation_model_path = settings.get_config_value(
            "paths.segmentation_model", ""
        )

        dependencies["pose_model"] = (
            "available" if os.path.exists(model_path) else "missing"
        )
        dependencies["segmentation_model"] = (
            "available" if os.path.exists(segmentation_model_path) else "missing"
        )

        # Check if voice messages directory exists
        voice_path = settings.get_config_value("paths.voice_messages", "")
        dependencies["voice_messages"] = (
            "available" if os.path.exists(voice_path) else "missing"
        )

        # Check if output directory is writable
        try:
            test_file = os.path.join(settings.output_dir, "test_write.tmp")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            dependencies["output_directory"] = "writable"
        except Exception:
            dependencies["output_directory"] = "not_writable"

        # Check if upload directory is writable
        try:
            test_file = os.path.join(settings.upload_dir, "test_write.tmp")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            dependencies["upload_directory"] = "writable"
        except Exception:
            dependencies["upload_directory"] = "not_writable"

        # Check Python packages
        try:
            import cv2

            dependencies["opencv"] = "available"
        except ImportError:
            dependencies["opencv"] = "missing"

        try:
            import ultralytics

            dependencies["ultralytics"] = "available"
        except ImportError:
            dependencies["ultralytics"] = "missing"

        try:
            import pygame

            dependencies["pygame"] = "available"
        except ImportError:
            dependencies["pygame"] = "missing"

        try:
            import ffmpeg

            dependencies["ffmpeg"] = "available"
        except ImportError:
            dependencies["ffmpeg"] = "missing"

        # Determine overall status
        critical_deps = ["pose_model", "opencv", "ultralytics"]
        status = "healthy"

        for dep in critical_deps:
            if dependencies.get(dep) != "available":
                status = "unhealthy"
                break

        return HealthResponse(
            status=status,
            timestamp=datetime.now(),
            version=settings.api_version,
            dependencies=dependencies,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")
