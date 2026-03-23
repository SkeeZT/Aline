"""
Configuration endpoints for frontend to query backend settings.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any, List
from loguru import logger

from core.config import settings

router = APIRouter(prefix="/api/config", tags=["Configuration"])


class DualCameraConfig(BaseModel):
    """Dual camera configuration response."""
    enabled: bool = False
    side_camera_id: int = 0
    front_camera_id: int = 1
    use_gstreamer: bool = False
    sync_tolerance_ms: int = 50


class ExerciseConfig(BaseModel):
    """Exercise configuration response."""
    available_exercises: List[str]
    default_exercise: str = "squat"
    experience_levels: List[str] = ["beginner", "intermediate", "advanced"]
    default_experience_level: str = "intermediate"


class AppConfig(BaseModel):
    """Full application configuration response."""
    dual_camera: DualCameraConfig
    exercises: ExerciseConfig
    api_version: str


@router.get("/", response_model=AppConfig)
async def get_config():
    """
    Get application configuration.
    
    Returns configuration settings that the frontend needs to know about,
    including dual camera settings and available exercises.
    """
    try:
        # Get dual camera config
        dual_cfg = settings.get_config_value("dual_camera", {})
        dual_camera_config = DualCameraConfig(
            enabled=dual_cfg.get("enabled", False),
            side_camera_id=dual_cfg.get("side_camera", {}).get("device_id", 0),
            front_camera_id=dual_cfg.get("front_camera", {}).get("device_id", 1),
            use_gstreamer=dual_cfg.get("use_gstreamer", False),
            sync_tolerance_ms=dual_cfg.get("sync", {}).get("tolerance_ms", 50),
        )
        
        # Get exercise config
        available_exercises = [
            "squat", "pullup", "pushup", "dips", "lunges",
            "plank", "deadlift", "overhead_press", "bent_over_row",
            "glute_bridge", "wall_sit"
        ]
        
        exercise_config = ExerciseConfig(
            available_exercises=available_exercises,
            default_exercise="squat",
            experience_levels=["beginner", "intermediate", "advanced"],
            default_experience_level="intermediate",
        )
        
        return AppConfig(
            dual_camera=dual_camera_config,
            exercises=exercise_config,
            api_version=settings.api_version,
        )
        
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        # Return safe defaults
        return AppConfig(
            dual_camera=DualCameraConfig(enabled=False),
            exercises=ExerciseConfig(
                available_exercises=["squat"],
                default_exercise="squat",
            ),
            api_version="1.0.0",
        )


@router.get("/dual-camera", response_model=DualCameraConfig)
async def get_dual_camera_config():
    """
    Get dual camera configuration specifically.
    
    Use this endpoint to check if dual camera mode is enabled
    before attempting to use it.
    """
    dual_cfg = settings.get_config_value("dual_camera", {})
    
    return DualCameraConfig(
        enabled=dual_cfg.get("enabled", False),
        side_camera_id=dual_cfg.get("side_camera", {}).get("device_id", 0),
        front_camera_id=dual_cfg.get("front_camera", {}).get("device_id", 1),
        use_gstreamer=dual_cfg.get("use_gstreamer", False),
        sync_tolerance_ms=dual_cfg.get("sync", {}).get("tolerance_ms", 50),
    )
