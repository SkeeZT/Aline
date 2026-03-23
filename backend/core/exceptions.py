"""
Custom exceptions for the FastAPI application.
"""

from typing import Optional, Dict, Any


class AITrainerException(Exception):
    """Base exception for AI Trainer API."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class AnalysisNotFoundError(AITrainerException):
    """Raised when analysis is not found."""

    pass


class AnalysisAlreadyExistsError(AITrainerException):
    """Raised when analysis with same ID already exists."""

    pass


class AnalysisInProgressError(AITrainerException):
    """Raised when trying to modify analysis that's in progress."""

    pass


class InvalidFileError(AITrainerException):
    """Raised when uploaded file is invalid."""

    pass


class FileTooLargeError(AITrainerException):
    """Raised when uploaded file is too large."""

    pass


class UnsupportedFileTypeError(AITrainerException):
    """Raised when file type is not supported."""

    pass


class ModelLoadError(AITrainerException):
    """Raised when AI model fails to load."""

    pass


class VideoProcessingError(AITrainerException):
    """Raised when video processing fails."""

    pass


class WebcamError(AITrainerException):
    """Raised when webcam access fails."""

    pass


class ConfigurationError(AITrainerException):
    """Raised when configuration is invalid."""

    pass


class AudioProcessingError(AITrainerException):
    """Raised when audio processing fails."""

    pass


class InsufficientDataError(AITrainerException):
    """Raised when there's insufficient data for analysis."""

    pass


class AnalysisTimeoutError(AITrainerException):
    """Raised when analysis times out."""

    pass
