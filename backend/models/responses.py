"""
Pydantic models for API responses.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class AnalysisStatus(str, Enum):
    """Analysis status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisResponse(BaseModel):
    """Response model for analysis creation."""

    analysis_id: str = Field(description="Unique identifier for the analysis")

    status: AnalysisStatus = Field(description="Current status of the analysis")

    exercise_type: str = Field(description="Type of exercise being analyzed")

    experience_level: str = Field(description="User experience level used for analysis")

    created_at: datetime = Field(description="Timestamp when analysis was created")

    estimated_duration: Optional[int] = Field(
        default=None, description="Estimated processing duration in seconds"
    )


class AnalysisStatusResponse(BaseModel):
    """Response model for analysis status."""

    analysis_id: str = Field(description="Unique identifier for the analysis")

    status: AnalysisStatus = Field(description="Current status of the analysis")

    progress_percentage: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Processing progress percentage"
    )

    current_frame: Optional[int] = Field(
        default=None, description="Current frame being processed"
    )

    total_frames: Optional[int] = Field(
        default=None, description="Total frames to process"
    )

    processing_time: Optional[float] = Field(
        default=None, description="Processing time in seconds"
    )

    error_message: Optional[str] = Field(
        default=None, description="Error message if analysis failed"
    )

    successful_reps: Optional[int] = Field(
        default=None, description="Number of successful reps"
    )

    unsuccessful_reps: Optional[int] = Field(
        default=None, description="Number of unsuccessful reps"
    )

    total_reps: Optional[int] = Field(
        default=None, description="Total number of reps"
    )

    accuracy: Optional[float] = Field(
        default=None, description="Accuracy score (0-1)"
    )

    current_state: Optional[str] = Field(
        default=None, description="Current state of the exercise"
    )

    instruction: Optional[str] = Field(
        default=None, description="Current instruction or feedback"
    )

    current_instruction: Optional[str] = Field(
        default=None, description="Current instruction message"
    )

    frame: Optional[str] = Field(
        default=None, description="Base64 encoded frame"
    )

    stream_has_video: Optional[bool] = Field(
        default=None, description="Whether stream has video"
    )


class VelocityMetrics(BaseModel):
    """Velocity-based training metrics."""

    rep_number: int = Field(description="Rep number")

    total_duration: float = Field(description="Total rep duration in seconds")

    concentric_velocity: float = Field(description="Average concentric velocity")

    peak_velocity: float = Field(description="Peak velocity during rep")

    velocity_consistency: float = Field(
        description="Velocity consistency (coefficient of variation)"
    )

    rep_quality_score: float = Field(description="Rep quality score (0-1)")


class SummaryStatistics(BaseModel):
    """Summary statistics for all reps."""

    total_reps_analyzed: int = Field(description="Total number of reps analyzed")

    concentric_velocity_stats: Dict[str, float] = Field(
        description="Concentric velocity statistics"
    )

    peak_velocity_stats: Dict[str, float] = Field(
        description="Peak velocity statistics"
    )

    duration_stats: Dict[str, float] = Field(description="Duration statistics")

    quality_stats: Dict[str, float] = Field(description="Quality score statistics")

    consistency_metrics: Dict[str, float] = Field(description="Consistency metrics")


class ExerciseResults(BaseModel):
    """Exercise analysis results."""

    successful_reps: int = Field(description="Number of successful reps")

    unsuccessful_reps: int = Field(description="Number of unsuccessful reps")

    total_reps: int = Field(description="Total number of reps")

    facing_side: Optional[str] = Field(
        default=None, description="Detected facing side (left/right)"
    )

    anthropometrics: Optional[Dict[str, Any]] = Field(
        default=None, description="Anthropometric measurements"
    )

    back_curvature: Optional[Dict[str, Any]] = Field(
        default=None, description="Back curvature analysis"
    )


class AnalysisResultsResponse(BaseModel):
    """Complete analysis results response."""

    analysis_id: str = Field(description="Unique identifier for the analysis")

    status: AnalysisStatus = Field(description="Analysis status")

    exercise_results: ExerciseResults = Field(description="Exercise analysis results")

    velocity_metrics: List[VelocityMetrics] = Field(
        description="Velocity metrics for each rep"
    )

    summary_statistics: SummaryStatistics = Field(
        description="Summary statistics for all reps"
    )

    processing_time: float = Field(description="Total processing time in seconds")

    output_files: Dict[str, str] = Field(
        description="Paths to output files (video, audio, data)"
    )

    created_at: datetime = Field(description="Analysis creation timestamp")

    completed_at: Optional[datetime] = Field(
        default=None, description="Analysis completion timestamp"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")

    timestamp: datetime = Field(description="Health check timestamp")

    version: str = Field(description="API version")

    dependencies: Dict[str, str] = Field(description="Dependency status")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(description="Error type")

    message: str = Field(description="Error message")

    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )

    timestamp: datetime = Field(description="Error timestamp")
