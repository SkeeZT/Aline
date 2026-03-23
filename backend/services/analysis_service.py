"""
Analysis service for handling results and metrics.
"""

import os
import re
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from loguru import logger

from engine.velocity_calculator import VelocityCalculator
from core.config import settings
from core.exceptions import AnalysisNotFoundError, InsufficientDataError
from models.responses import (
    AnalysisResultsResponse,
    ExerciseResults,
    VelocityMetrics,
    SummaryStatistics,
    AnalysisStatus,
)


class AnalysisService:
    """Service for analysis results and metrics."""

    def __init__(self):
        self.results_cache: Dict[str, Dict[str, Any]] = {}

    def get_analysis_results(self, analysis_id: str) -> AnalysisResultsResponse:
        """
        Get complete analysis results.

        Args:
            analysis_id: Analysis ID

        Returns:
            Complete analysis results
        """
        # Check cache first
        if analysis_id in self.results_cache:
            return self._build_results_response(
                analysis_id, self.results_cache[analysis_id]
            )

        # Try to load from output directory
        results_data = self._load_results_from_files(analysis_id)
        if results_data:
            self.results_cache[analysis_id] = results_data
            return self._build_results_response(analysis_id, results_data)

        raise AnalysisNotFoundError(f"Analysis results not found for ID: {analysis_id}")

    def _load_results_from_files(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Load analysis results from output files."""
        try:
            # Validate analysis_id to prevent path traversal
            if not self._is_valid_analysis_id(analysis_id):
                logger.warning(f"Invalid analysis ID format: {analysis_id}")
                return None

            # Look for files with the analysis ID in the name
            output_dir = Path(settings.output_dir).resolve()

            # Find metadata file
            metadata_files = list(output_dir.glob(f"**/metadata_*{analysis_id}*.json"))
            if not metadata_files:
                # Try to find any metadata file that might contain this analysis
                metadata_files = list(output_dir.glob("**/metadata_*.json"))

            # Filter files to ensure they're within the allowed directory
            metadata_files = [f for f in metadata_files if self._is_safe_path(f, output_dir)]

            if not metadata_files:
                logger.warning(f"No metadata files found for analysis {analysis_id}")
                return None

            # Load the most recent metadata file
            metadata_file = max(metadata_files, key=os.path.getctime)

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Find velocity analysis file
            velocity_files = list(
                output_dir.glob(f"**/squat_vbt_analysis_*{analysis_id}*.json")
            )
            if not velocity_files:
                # Try to find any velocity analysis file
                velocity_files = list(output_dir.glob("**/squat_vbt_analysis_*.json"))

            # Filter velocity files to ensure they're within the allowed directory
            velocity_files = [f for f in velocity_files if self._is_safe_path(f, output_dir)]

            velocity_data = None
            if velocity_files:
                velocity_file = max(velocity_files, key=os.path.getctime)
                with open(velocity_file, "r") as f:
                    velocity_data = json.load(f)

            # Find output video files
            video_files = list(output_dir.glob(f"**/*{analysis_id}*.mp4"))
            # Filter video files to ensure they're within the allowed directory
            video_files = [f for f in video_files if self._is_safe_path(f, output_dir)]

            return {
                "metadata": metadata,
                "velocity_data": velocity_data,
                "video_files": [str(f) for f in video_files],
                "metadata_file": str(metadata_file),
                "velocity_file": str(velocity_file) if velocity_data else None,
            }

        except Exception as e:
            logger.error(
                f"Error loading results from files for analysis {analysis_id}: {e}"
            )
            return None

    def _is_valid_analysis_id(self, analysis_id: str) -> bool:
        """Validate analysis ID format to prevent path traversal."""
        # Only allow alphanumeric characters, hyphens, and underscores
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', analysis_id))

    def _is_safe_path(self, file_path: Path, base_dir: Path) -> bool:
        """Check if the file path is within the allowed base directory."""
        try:
            file_path.resolve().relative_to(base_dir.resolve())
            return True
        except ValueError:
            # Path is outside the base directory
            logger.warning(f"Unsafe path detected: {file_path}")
            return False

    def _build_results_response(
        self, analysis_id: str, results_data: Dict[str, Any]
    ) -> AnalysisResultsResponse:
        """Build AnalysisResultsResponse from loaded data."""
        metadata = results_data.get("metadata", {})
        velocity_data = results_data.get("velocity_data", {})

        # Build exercise results
        exercise_results = ExerciseResults(
            successful_reps=metadata.get("successful_reps", 0),
            unsuccessful_reps=metadata.get("unsuccessful_reps", 0),
            total_reps=metadata.get("total_reps", 0),
            facing_side=metadata.get("facing_side"),
            anthropometrics=metadata.get("anthropometrics"),
            back_curvature=metadata.get("back_curvature"),
        )

        # Build velocity metrics
        velocity_metrics = []
        if velocity_data and "rep_velocities" in velocity_data:
            for rep_data in velocity_data["rep_velocities"]:
                velocity_metrics.append(
                    VelocityMetrics(
                        rep_number=rep_data.get("rep_number", 0),
                        total_duration=rep_data.get("total_duration", 0.0),
                        concentric_velocity=rep_data.get("concentric_velocity", 0.0),
                        peak_velocity=rep_data.get("peak_velocity", 0.0),
                        velocity_consistency=rep_data.get("velocity_consistency", 0.0),
                        rep_quality_score=rep_data.get("rep_quality_score", 0.0),
                    )
                )

        # Build summary statistics
        summary_stats = SummaryStatistics(
            total_reps_analyzed=0,
            concentric_velocity_stats={},
            peak_velocity_stats={},
            duration_stats={},
            quality_stats={},
            consistency_metrics={},
        )

        if velocity_data and "summary_statistics" in velocity_data:
            stats = velocity_data["summary_statistics"]
            summary_stats = SummaryStatistics(
                total_reps_analyzed=stats.get("total_reps_analyzed", 0),
                concentric_velocity_stats=stats.get("concentric_velocity_stats", {}),
                peak_velocity_stats=stats.get("peak_velocity_stats", {}),
                duration_stats=stats.get("duration_stats", {}),
                quality_stats=stats.get("quality_stats", {}),
                consistency_metrics=stats.get("consistency_metrics", {}),
            )

        # Build output files dictionary
        output_files = {}
        if results_data.get("video_files"):
            output_files["processed_video"] = results_data["video_files"][0]
            if len(results_data["video_files"]) > 1:
                output_files["final_video"] = results_data["video_files"][1]

        if results_data.get("velocity_file"):
            output_files["velocity_data"] = results_data["velocity_file"]

        if results_data.get("metadata_file"):
            output_files["metadata"] = results_data["metadata_file"]

        # Calculate processing time
        created_at = datetime.fromisoformat(
            metadata.get("created_at", datetime.now().isoformat())
        )
        completed_at = datetime.now()  # This should be from metadata if available
        processing_time = (completed_at - created_at).total_seconds()

        return AnalysisResultsResponse(
            analysis_id=analysis_id,
            status=AnalysisStatus.COMPLETED,
            exercise_results=exercise_results,
            velocity_metrics=velocity_metrics,
            summary_statistics=summary_stats,
            processing_time=processing_time,
            output_files=output_files,
            created_at=created_at,
            completed_at=completed_at,
        )

    def get_velocity_metrics(self, analysis_id: str) -> List[VelocityMetrics]:
        """Get velocity metrics for analysis."""
        results = self.get_analysis_results(analysis_id)
        return results.velocity_metrics

    def get_summary_statistics(self, analysis_id: str) -> SummaryStatistics:
        """Get summary statistics for analysis."""
        results = self.get_analysis_results(analysis_id)
        return results.summary_statistics

    def get_exercise_results(self, analysis_id: str) -> ExerciseResults:
        """Get exercise results for analysis."""
        results = self.get_analysis_results(analysis_id)
        return results.exercise_results

    def get_output_file_path(self, analysis_id: str, file_type: str) -> Optional[str]:
        """
        Get path to specific output file.

        Args:
            analysis_id: Analysis ID
            file_type: Type of file ('video', 'audio', 'data', 'metadata')

        Returns:
            File path if found, None otherwise
        """
        try:
            results_data = self._load_results_from_files(analysis_id)
            if not results_data:
                return None

            if file_type == "video" and results_data.get("video_files"):
                return results_data["video_files"][0]
            elif file_type == "metadata" and results_data.get("metadata_file"):
                return results_data["metadata_file"]
            elif file_type == "data" and results_data.get("velocity_file"):
                return results_data["velocity_file"]

            return None

        except Exception as e:
            logger.error(
                f"Error getting output file path for analysis {analysis_id}: {e}"
            )
            return None

    def list_available_analyses(self) -> List[Dict[str, Any]]:
        """List all available analyses."""
        try:
            output_dir = Path(settings.output_dir)
            metadata_files = list(output_dir.glob("**/metadata_*.json"))

            analyses = []
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)

                    # Extract analysis ID from filename or metadata
                    analysis_id = metadata.get("analysis_id", metadata_file.stem)

                    analyses.append(
                        {
                            "analysis_id": analysis_id,
                            "created_at": metadata.get("created_at"),
                            "exercise_type": metadata.get("exercise_type", "squat"),
                            "total_reps": metadata.get("total_reps", 0),
                            "successful_reps": metadata.get("successful_reps", 0),
                            "metadata_file": str(metadata_file),
                        }
                    )

                except Exception as e:
                    logger.warning(f"Error reading metadata file {metadata_file}: {e}")
                    continue

            # Sort by creation time (newest first)
            analyses.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return analyses

        except Exception as e:
            logger.error(f"Error listing available analyses: {e}")
            return []

    def cleanup_old_analyses(self, days_old: int = 7) -> int:
        """
        Clean up old analysis files.

        Args:
            days_old: Number of days after which to clean up files

        Returns:
            Number of files cleaned up
        """
        try:
            output_dir = Path(settings.output_dir)
            cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)

            cleaned_count = 0

            # Find old files
            for file_path in output_dir.rglob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.info(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up file {file_path}: {e}")

            logger.info(f"Cleaned up {cleaned_count} old analysis files")
            return cleaned_count

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0


# Global service instance
analysis_service = AnalysisService()
