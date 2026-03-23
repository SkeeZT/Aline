"""
Pydantic models for API requests.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum


class ExperienceLevel(str, Enum):
    """User experience levels for exercise analysis."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class VideoAnalysisRequest(BaseModel):
    """Request model for video analysis."""

    exercise_type: Literal["squat"] = Field(
        default="squat", description="Type of exercise to analyze"
    )

    experience_level: ExperienceLevel = Field(
        default=ExperienceLevel.INTERMEDIATE,
        description="User experience level for analysis thresholds",
    )

    force_front_view: bool = Field(
        default=False, description="Force front-view analysis (skip side positioning)"
    )

    enable_voice_feedback: bool = Field(
        default=True, description="Enable voice feedback during analysis"
    )

    enable_segmentation: bool = Field(
        default=True, description="Enable person segmentation overlay"
    )

    show_visualization: bool = Field(
        default=False, description="Show real-time visualization (for webcam analysis)"
    )

    voice_volume: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Voice feedback volume (0.0 to 1.0)"
    )


class WebcamAnalysisRequest(BaseModel):
    """Request model for webcam analysis."""

    webcam_id: int = Field(default=0, ge=0, description="Webcam device ID")

    exercise_type: Literal["squat"] = Field(
        default="squat", description="Type of exercise to analyze"
    )

    experience_level: ExperienceLevel = Field(
        default=ExperienceLevel.INTERMEDIATE,
        description="User experience level for analysis thresholds",
    )

    force_front_view: bool = Field(
        default=False, description="Force front-view analysis (skip side positioning)"
    )

    enable_voice_feedback: bool = Field(
        default=True, description="Enable voice feedback during analysis"
    )

    enable_segmentation: bool = Field(
        default=True, description="Enable person segmentation overlay"
    )

    show_visualization: bool = Field(
        default=True, description="Show real-time visualization"
    )

    voice_volume: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Voice feedback volume (0.0 to 1.0)"
    )


class AnalysisConfigRequest(BaseModel):
    """Request model for updating analysis configuration."""

    experience_level: Optional[ExperienceLevel] = Field(
        default=None, description="User experience level for analysis thresholds"
    )

    force_front_view: Optional[bool] = Field(
        default=None, description="Force front-view analysis"
    )

    enable_voice_feedback: Optional[bool] = Field(
        default=None, description="Enable voice feedback"
    )

    enable_segmentation: Optional[bool] = Field(
        default=None, description="Enable person segmentation"
    )

    voice_volume: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Voice feedback volume"
    )

    @validator("voice_volume")
    def validate_voice_volume(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Voice volume must be between 0.0 and 1.0")
        return v
