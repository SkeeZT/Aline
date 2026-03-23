"""
Analysis results endpoints.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from core.config import settings
from models.responses import (
    AnalysisResultsResponse,
    VelocityMetrics,
    SummaryStatistics,
    ExerciseResults,
)
from services.analysis_service import analysis_service
from core.exceptions import AnalysisNotFoundError

router = APIRouter(prefix="/api/v1/analysis", tags=["Analysis Results"])


@router.get("/{analysis_id}", response_model=AnalysisResultsResponse)
async def get_analysis_results(analysis_id: str):
    """
    Get complete analysis results.

    Returns comprehensive analysis results including exercise metrics,
    velocity data, and summary statistics.
    """
    try:
        results = analysis_service.get_analysis_results(analysis_id)
        return results

    except AnalysisNotFoundError as e:
        logger.warning(f"Analysis results not found: {analysis_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting analysis results: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{analysis_id}/metrics", response_model=List[VelocityMetrics])
async def get_velocity_metrics(analysis_id: str):
    """
    Get velocity-based training metrics.

    Returns detailed velocity metrics for each rep analyzed.
    """
    try:
        metrics = analysis_service.get_velocity_metrics(analysis_id)
        return metrics

    except AnalysisNotFoundError as e:
        logger.warning(f"Velocity metrics not found: {analysis_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting velocity metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{analysis_id}/summary", response_model=SummaryStatistics)
async def get_summary_statistics(analysis_id: str):
    """
    Get summary statistics.

    Returns aggregated statistics across all reps analyzed.
    """
    try:
        stats = analysis_service.get_summary_statistics(analysis_id)
        return stats

    except AnalysisNotFoundError as e:
        logger.warning(f"Summary statistics not found: {analysis_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting summary statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{analysis_id}/exercise", response_model=ExerciseResults)
async def get_exercise_results(analysis_id: str):
    """
    Get exercise analysis results.

    Returns basic exercise analysis results including rep counts and form analysis.
    """
    try:
        results = analysis_service.get_exercise_results(analysis_id)
        return results

    except AnalysisNotFoundError as e:
        logger.warning(f"Exercise results not found: {analysis_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting exercise results: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{analysis_id}/files/{file_type}")
async def download_analysis_file(
    analysis_id: str,
    file_type: str,
):
    """
    Download specific analysis file.

    Downloads analysis files by type:
    - video: Processed video with overlays
    - audio: Voice feedback audio track
    - data: Velocity analysis data (JSON)
    - metadata: Analysis metadata (JSON)
    """
    try:
        if file_type not in ["video", "audio", "data", "metadata"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Must be one of: video, audio, data, metadata",
            )

        # Validate analysis_id to prevent path traversal
        if not _is_valid_analysis_id(analysis_id):
            logger.warning(f"Invalid analysis ID format: {analysis_id}")
            raise HTTPException(status_code=400, detail="Invalid analysis ID format")

        file_path = analysis_service.get_output_file_path(analysis_id, file_type)

        if not file_path:
            raise HTTPException(status_code=404, detail=f"{file_type} file not found")

        # Validate that the file path is safe (within allowed directories)
        if not _is_safe_path(file_path):
            logger.error(f"Unsafe file path detected: {file_path}")
            raise HTTPException(status_code=400, detail="Invalid file path")

        # Determine media type based on file type
        media_types = {
            "video": "video/mp4",
            "audio": "audio/wav",
            "data": "application/json",
            "metadata": "application/json",
        }

        # Determine file extension
        extensions = {
            "video": ".mp4",
            "audio": ".wav",
            "data": ".json",
            "metadata": ".json",
        }

        from fastapi.responses import FileResponse

        return FileResponse(
            path=file_path,
            media_type=media_types[file_type],
            filename=f"{analysis_id}_{file_type}{extensions[file_type]}",
        )

    except AnalysisNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading {file_type} file: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def _is_valid_analysis_id(analysis_id: str) -> bool:
    """Validate analysis ID format to prevent path traversal."""
    # Only allow alphanumeric characters, hyphens, and underscores
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', analysis_id))


def _is_safe_path(file_path: str) -> bool:
    """Check if the file path is within the allowed directories."""
    try:
        resolved_path = Path(file_path).resolve()
        allowed_dirs = [
            Path(settings.output_dir).resolve(),
            Path(settings.upload_dir).resolve(),
        ]

        for allowed_dir in allowed_dirs:
            try:
                resolved_path.relative_to(allowed_dir)
                return True
            except ValueError:
                continue

        return False
    except Exception:
        return False


@router.get("/", response_model=List[Dict[str, Any]])
async def list_analyses():
    """
    List all available analyses.

    Returns a list of all available analyses with basic information.
    """
    try:
        analyses = analysis_service.list_available_analyses()
        return analyses

    except Exception as e:
        logger.error(f"Error listing analyses: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/cleanup")
async def cleanup_old_analyses(
    days_old: int = Query(
        default=7, ge=1, le=30, description="Days after which to clean up files"
    ),
):
    """
    Clean up old analysis files.

    Removes analysis files older than the specified number of days.
    """
    try:
        cleaned_count = analysis_service.cleanup_old_analyses(days_old)

        return {
            "message": f"Cleaned up {cleaned_count} old analysis files",
            "files_removed": cleaned_count,
            "days_old": days_old,
        }

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
