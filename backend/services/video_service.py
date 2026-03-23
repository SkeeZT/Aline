"""
Video processing service that wraps the existing VideoProcessor.
"""

import asyncio
import copy
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

import cv2
from fastapi import UploadFile
from loguru import logger

from core.config import Config
from core.config import settings
from core.exceptions import (
    FileTooLargeError,
    InvalidFileError,
    ModelLoadError,
    UnsupportedFileTypeError,
    VideoProcessingError,
)
from engine.video_processor import VideoProcessor
from streaming.latest_frame_bus import latest_frame_bus


@dataclass
class WebcamPreviewSession:
    """State container for active webcam preview streams."""

    cap: cv2.VideoCapture
    lock: asyncio.Lock
    clients: int = 0
    stop_requested: bool = False
    closed_event: asyncio.Event = field(default_factory=asyncio.Event)


class VideoAnalysisService:
    """Service for video analysis operations."""

    def __init__(self):
        self.active_analyses: Dict[str, Dict[str, Any]] = {}
        self.config = None
        self._webcam_locks: Dict[int, asyncio.Lock] = {}
        self._preview_sessions: Dict[int, "WebcamPreviewSession"] = {}
        self._load_config()

    def _load_config(self):
        """Load configuration for video processing."""
        try:
            self.config = Config.load_config(settings.config_path)
            logger.info("Video processing configuration loaded")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ModelLoadError(f"Failed to load configuration: {e}")

    def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file."""
        try:
            # Check file size
            if hasattr(file, "size") and file.size > settings.max_file_size:
                raise FileTooLargeError(
                    f"File size ({file.size} bytes) exceeds maximum allowed size ({settings.max_file_size} bytes)"
                )

            # Check file extension
            if not file.filename:
                raise InvalidFileError("No filename provided")

            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in settings.allowed_extensions:
                raise UnsupportedFileTypeError(
                    f"File type '{file_extension}' is not supported. Allowed types: {settings.allowed_extensions}"
                )

            # Additional validation: check for path traversal in filename
            if '..' in file.filename or file.filename.startswith('/'):
                raise InvalidFileError("Invalid filename provided")

        except FileTooLargeError:
            raise
        except UnsupportedFileTypeError:
            raise
        except InvalidFileError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during file validation: {e}")
            raise InvalidFileError(f"File validation failed: {str(e)}")

    async def _save_uploaded_file(self, file: UploadFile) -> str:
        """Save uploaded file to temporary location."""
        try:
            # Create unique filename
            file_extension = Path(file.filename).suffix
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join(settings.upload_dir, unique_filename)

            # Validate the file path to prevent path traversal
            if not self._is_safe_path(file_path):
                raise VideoProcessingError("Invalid file path for upload")

            # Save file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            logger.info(f"File saved to: {file_path}")
            return file_path

        except OSError as e:
            logger.error(f"OS error saving uploaded file: {e}")
            raise VideoProcessingError(f"Failed to save uploaded file due to system error: {e}")
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            raise VideoProcessingError(f"Failed to save uploaded file: {e}")

    def _is_safe_path(self, file_path: str) -> bool:
        """Check if the file path is within the allowed upload directory."""
        try:
            from pathlib import Path
            resolved_path = Path(file_path).resolve()
            allowed_dir = Path(settings.upload_dir).resolve()
            resolved_path.relative_to(allowed_dir)
            return True
        except ValueError:
            # Path is outside the allowed directory
            return False
        except Exception:
            return False

    def _create_analysis_config(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create analysis configuration from request data."""
        config = copy.deepcopy(self.config)

        # Update configuration based on request
        if "experience_level" in request_data:
            config.setdefault("experience", {})["level"] = request_data["experience_level"]

        if "force_front_view" in request_data:
            config.setdefault("view", {})["force_front_view"] = request_data["force_front_view"]

        if "enable_voice_feedback" in request_data:
            config.setdefault("voice", {})["enabled"] = request_data["enable_voice_feedback"]

        if "voice_volume" in request_data:
            config.setdefault("voice", {})["volume"] = request_data["voice_volume"]

        if "enable_segmentation" in request_data:
            config.setdefault("segmentation", {})["enabled"] = request_data["enable_segmentation"]

        if "show_visualization" in request_data:
            config.setdefault("video", {})["show_visualize"] = request_data["show_visualization"]

        if "suppress_preview" in request_data:
            config.setdefault("video", {})["suppress_preview"] = request_data[
                "suppress_preview"
            ]

        return config

    async def analyze_video_file(
        self, file: UploadFile, request_data: Dict[str, Any]
    ) -> str:
        """
        Analyze uploaded video file.

        Args:
            file: Uploaded video file
            request_data: Analysis request parameters

        Returns:
            Analysis ID
        """
        analysis_id = str(uuid.uuid4())

        try:
            # Validate file
            self._validate_file(file)

            # Save uploaded file
            video_path = await self._save_uploaded_file(file)

            # Create analysis configuration
            analysis_config = self._create_analysis_config(request_data)

            # Update config with video path
            analysis_config["paths"]["input_video"] = video_path
            analysis_config["paths"]["output_dir"] = settings.output_dir

            # Create analysis record
            self.active_analyses[analysis_id] = {
                "status": "pending",
                "exercise_type": request_data.get("exercise_type", "squat"),
                "experience_level": request_data.get(
                    "experience_level", "intermediate"
                ),
                "video_path": video_path,
                "config": analysis_config,
                "created_at": datetime.now(),
                "progress": 0.0,
                "current_frame": 0,
                "total_frames": None,
                "error_message": None,
                "subscribers": [],
                "latest_payload": None,
            }

            self._merge_payload(
                analysis_id,
                {
                    "status": "pending",
                    "progress": 0.0,
                    "successful_reps": 0,
                    "unsuccessful_reps": 0,
                    "total_reps": 0,
                    "accuracy": 0.0,
                    "instruction": "Stand in front of the marker",
                    "current_state": "idle",
                },
            )

            logger.info(f"Video analysis created with ID: {analysis_id}")
            return analysis_id

        except Exception as e:
            logger.error(f"Error creating video analysis: {e}")
            # Clean up analysis record if it was created
            if analysis_id in self.active_analyses:
                del self.active_analyses[analysis_id]
            raise

    def _get_webcam_lock(self, webcam_id: int) -> asyncio.Lock:
        lock = self._webcam_locks.get(webcam_id)
        if lock is None:
            lock = asyncio.Lock()
            self._webcam_locks[webcam_id] = lock
        return lock

    async def _get_or_create_preview_session(
        self, webcam_id: int
    ) -> WebcamPreviewSession:
        loop = asyncio.get_running_loop()
        session = self._preview_sessions.get(webcam_id)

        if session is None or session.stop_requested and session.clients == 0:
            if session is not None:
                self._preview_sessions.pop(webcam_id, None)

            def _open_capture():
                # Use shared webcam manager to avoid conflicts
                from engine.webcam_manager import SharedWebcamCapture
                return SharedWebcamCapture(webcam_id)

            cap = await loop.run_in_executor(None, _open_capture)
            if not cap or not cap.isOpened():
                if cap:
                    cap.release()
                raise VideoProcessingError(
                    f"Unable to access webcam {webcam_id} for preview"
                )

            session = WebcamPreviewSession(cap=cap, lock=asyncio.Lock())
            self._preview_sessions[webcam_id] = session

        return session

    async def stream_webcam_preview(
        self, webcam_id: int
    ) -> AsyncGenerator[bytes, None]:
        """Yield JPEG frames for a live webcam preview without starting analysis."""

        session = await self._get_or_create_preview_session(webcam_id)
        loop = asyncio.get_running_loop()
        session.clients += 1

        try:
            while not session.stop_requested:
                async with session.lock:
                    ret, frame = await loop.run_in_executor(None, session.cap.read)

                if not ret:
                    await asyncio.sleep(0.05)
                    continue

                success, buffer = await loop.run_in_executor(
                    None,
                    lambda: cv2.imencode(
                        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    ),
                )

                if not success:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )

                if session.stop_requested:
                    break

                await asyncio.sleep(0.03)

        except asyncio.CancelledError:
            raise
        finally:
            session.clients = max(0, session.clients - 1)
            if session.clients == 0:
                await loop.run_in_executor(None, session.cap.release)
                session.closed_event.set()
                existing = self._preview_sessions.get(webcam_id)
                if existing is session:
                    self._preview_sessions.pop(webcam_id, None)

    async def stop_webcam_preview(self, webcam_id: int, timeout: float = 2.0) -> None:
        """Request any active preview session for a webcam to stop and release the device."""

        session = self._preview_sessions.get(webcam_id)
        if not session:
            return

        if session.stop_requested:
            try:
                await asyncio.wait_for(session.closed_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, session.cap.release)
                session.closed_event.set()
            finally:
                self._preview_sessions.pop(webcam_id, None)
            return

        session.stop_requested = True

        try:
            await asyncio.wait_for(session.closed_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, session.cap.release)
            session.closed_event.set()
        finally:
            self._preview_sessions.pop(webcam_id, None)

    def _find_active_webcam_analysis(self, webcam_id: int) -> Optional[str]:
        for analysis_id, record in self.active_analyses.items():
            if not record.get("is_webcam"):
                continue
            if record.get("webcam_id") != webcam_id:
                continue
            if record.get("status") in {"completed", "failed", "cancelled"}:
                continue
            return analysis_id
        return None

    async def start_webcam_analysis(
        self, request_data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any], bool]:
        """
        Start webcam analysis.

        Args:
            request_data: Analysis request parameters

        Returns:
            Tuple of (analysis_id, status_info, created_new)
        """

        webcam_id = request_data.get("webcam_id", 0)

        lock = self._get_webcam_lock(webcam_id)

        async with lock:
            # Ensure any existing preview stream releases the device before analysis starts
            await self.stop_webcam_preview(webcam_id)

            existing_id = self._find_active_webcam_analysis(webcam_id)
            if existing_id:
                logger.info(
                    "Reusing active webcam analysis %s for webcam %s",
                    existing_id,
                    webcam_id,
                )
                return existing_id, self.get_analysis_status(existing_id), False

            analysis_id = str(uuid.uuid4())

            try:
                # Create analysis configuration
                analysis_config = self._create_analysis_config(request_data)

                # Update config for webcam
                analysis_config.setdefault("video", {})
                analysis_config["video"]["use_webcam"] = True
                analysis_config["video"]["webcam_id"] = request_data.get("webcam_id", 0)
                analysis_config["video"]["show_visualize"] = False
                analysis_config["video"]["suppress_preview"] = request_data.get(
                    "suppress_preview", True
                )
                analysis_config["paths"]["output_dir"] = settings.output_dir

                # Create analysis record
                self.active_analyses[analysis_id] = {
                    "status": "pending",
                    "exercise_type": request_data.get("exercise_type", "squat"),
                    "experience_level": request_data.get(
                        "experience_level", "intermediate"
                    ),
                    "webcam_id": request_data.get("webcam_id", 0),
                    "config": analysis_config,
                    "created_at": datetime.now(),
                    "progress": 0.0,
                    "current_frame": 0,
                    "total_frames": None,
                    "error_message": None,
                    "is_webcam": True,
                    "subscribers": [],
                    "latest_payload": None,
                }

                self._merge_payload(
                    analysis_id,
                    {
                        "status": "pending",
                        "progress": 0.0,
                        "successful_reps": 0,
                        "unsuccessful_reps": 0,
                        "total_reps": 0,
                        "accuracy": 0.0,
                        "instruction": "Stand in front of the marker",
                        "current_state": "idle",
                        "stream_has_video": analysis_config["video"].get(
                            "use_webcam", False
                        ),
                    },
                )

                logger.info(f"Webcam analysis created with ID: {analysis_id}")
                return analysis_id, self.get_analysis_status(analysis_id), True

            except Exception as e:
                logger.error(f"Error creating webcam analysis: {e}")
                # Clean up analysis record if it was created
                if analysis_id in self.active_analyses:
                    del self.active_analyses[analysis_id]
                raise

    async def process_analysis(
        self, analysis_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process video analysis with progress updates.

        Args:
            analysis_id: Analysis ID

        Yields:
            Progress updates
        """
        if analysis_id not in self.active_analyses:
            raise VideoProcessingError(f"Analysis {analysis_id} not found")

        analysis = self.active_analyses[analysis_id]

        try:
            # Update status to processing
            analysis["status"] = "processing"
            analysis["progress"] = 0.0
            initial_payload = await self._notify_status(
                analysis_id,
                status="processing",
                progress=0.0,
                message="Starting analysis...",
            )
            yield initial_payload

            # Create video processor
            loop = asyncio.get_running_loop()

            processor = VideoProcessor(
                analysis_id=analysis_id,
                config=analysis["config"],
                exercise_type=analysis["exercise_type"],
                use_webcam=analysis.get("is_webcam", False),
                webcam_id=analysis.get("webcam_id", 0),
                video_path=analysis.get("video_path"),
                force_front_view=analysis["config"]
                .get("view", {})
                .get("force_front_view", False),
                experience_level=analysis["experience_level"],
                progress_callback=self._make_frame_callback(analysis_id, loop),
                stop_requested=lambda: self.active_analyses.get(analysis_id, {}).get("status") == "cancelled",
            )

            # Update progress to 25% when starting
            analysis["progress"] = 25.0
            await self._notify_status(
                analysis_id,
                status="processing",
                progress=25.0,
                message="Processing video...",
            )
            yield analysis["latest_payload"]

            # Run analysis in thread pool to avoid blocking
            await loop.run_in_executor(None, processor.process)

            # Update progress to 75% when processing is done
            analysis["progress"] = 75.0
            await self._notify_status(
                analysis_id,
                status="processing",
                progress=75.0,
                message="Finalizing results...",
            )
            yield analysis["latest_payload"]

            # Update status to completed
            analysis["status"] = "completed"
            analysis["progress"] = 100.0
            analysis["completed_at"] = datetime.now()
            await self._notify_status(
                analysis_id,
                status="completed",
                progress=100.0,
                message="Analysis completed successfully",
            )

            yield analysis["latest_payload"]

        except Exception as e:
            logger.error(f"Error processing analysis {analysis_id}: {e}")
            analysis["status"] = "failed"
            analysis["error_message"] = str(e)
            await self._notify_status(
                analysis_id,
                status="failed",
                progress=analysis.get("progress", 0.0),
                message=str(e),
            )

            yield analysis["latest_payload"]

    def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """Get analysis status."""
        if analysis_id not in self.active_analyses:
            raise VideoProcessingError(f"Analysis {analysis_id} not found")

        record = self.active_analyses[analysis_id]
        latest_payload = record.get("latest_payload") or {}

        created_at = record.get("created_at")
        completed_at = record.get("completed_at")

        processing_time = None
        if hasattr(created_at, "timestamp"):
            end_time = completed_at or datetime.now()
            if hasattr(end_time, "timestamp"):
                processing_time = (end_time - created_at).total_seconds()

        return {
            "analysis_id": analysis_id,
            "status": record.get("status", "pending"),
            "progress_percentage": record.get("progress", 0.0),
            "current_frame": record.get("current_frame"),
            "total_frames": record.get("total_frames"),
            "error_message": record.get("error_message"),
            "processing_time": processing_time,
            "exercise_type": record.get("exercise_type", "squat"),
            "experience_level": record.get("experience_level", "intermediate"),
            "created_at": record.get("created_at"),
            "successful_reps": latest_payload.get("successful_reps"),
            "unsuccessful_reps": latest_payload.get("unsuccessful_reps"),
            "total_reps": latest_payload.get("total_reps"),
            "accuracy": latest_payload.get("accuracy"),
            "current_state": latest_payload.get("current_state"),
            "instruction": latest_payload.get("instruction"),
            "current_instruction": latest_payload.get("instruction"),
            "frame": record.get("latest_frame"),
            "stream_has_video": record.get("stream_has_video", False),
        }

    def get_webcam_id(self, analysis_id: str) -> Optional[int]:
        record = self.active_analyses.get(analysis_id)
        if not record or not record.get("is_webcam"):
            return None
        return record.get("webcam_id")

    def has_active_webcam_analysis(self, webcam_id: int, exclude: Optional[str] = None) -> bool:
        for analysis_id, record in self.active_analyses.items():
            if exclude and analysis_id == exclude:
                continue
            if not record.get("is_webcam"):
                continue
            if record.get("webcam_id") != webcam_id:
                continue
            if record.get("status") not in {"completed", "failed", "cancelled"}:
                return True
        return False

    def update_analysis_progress(
        self,
        analysis_id: str,
        progress: float,
        current_frame: int = None,
        total_frames: int = None,
    ):
        """Update analysis progress."""
        if analysis_id in self.active_analyses:
            analysis = self.active_analyses[analysis_id]
            analysis["progress"] = max(0.0, min(100.0, progress))
            if current_frame is not None:
                analysis["current_frame"] = current_frame
            if total_frames is not None:
                analysis["total_frames"] = total_frames
            logger.debug(f"Updated progress for {analysis_id}: {progress}%")
            # Merge progress into latest payload for status polling
            self._merge_payload(
                analysis_id,
                {
                    "progress": analysis["progress"],
                    "current_frame": analysis.get("current_frame"),
                    "total_frames": analysis.get("total_frames"),
                },
            )

    async def register_subscriber(self, analysis_id: str) -> asyncio.Queue:
        """Register a websocket subscriber for live updates."""
        if analysis_id not in self.active_analyses:
            raise VideoProcessingError(f"Analysis {analysis_id} not found")

        queue: asyncio.Queue = asyncio.Queue(maxsize=20)
        analysis = self.active_analyses[analysis_id]
        subscribers = analysis.setdefault("subscribers", [])
        subscribers.append(queue)

        payload = self._build_live_payload(analysis_id)
        await queue.put(payload)

        logger.debug(f"Registered subscriber for analysis {analysis_id}")
        return queue

    def unregister_subscriber(self, analysis_id: str, queue: asyncio.Queue) -> None:
        """Remove subscriber queue."""
        analysis = self.active_analyses.get(analysis_id)
        if not analysis:
            return
        subscribers = analysis.get("subscribers", [])
        if queue in subscribers:
            subscribers.remove(queue)
            logger.debug(f"Unregistered subscriber for analysis {analysis_id}")

    async def _notify_status(
        self,
        analysis_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = self._merge_payload(
            analysis_id,
            {
                "event": "status",
                "status": status,
                "progress": progress,
                "message": message,
            },
        )
        await self._notify_subscribers(analysis_id, payload)
        return payload

    def _make_frame_callback(
        self, analysis_id: str, loop: asyncio.AbstractEventLoop
    ):
        """Create a thread-safe callback that broadcasts per-frame metrics."""

        def callback(metrics: Dict[str, Any]):
            payload = self._merge_payload(
                analysis_id,
                {
                    "event": "metrics",
                    **metrics,
                },
            )
            asyncio.run_coroutine_threadsafe(
                self._notify_subscribers(analysis_id, payload), loop
            )

        return callback

    async def _notify_subscribers(self, analysis_id: str, payload: Dict[str, Any]):
        """Push payload to all registered subscribers."""
        analysis = self.active_analyses.get(analysis_id)
        if not analysis:
            return

        for queue in list(analysis.get("subscribers", [])):
            try:
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await queue.put(payload)
            except RuntimeError:
                self._drop_subscriber_queue(analysis_id, queue)
            except Exception as exc:
                if isinstance(exc, asyncio.CancelledError):
                    raise
                self._drop_subscriber_queue(analysis_id, queue)

    def _merge_payload(self, analysis_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge new data with existing payload and return the updated payload."""
        analysis = self.active_analyses.get(analysis_id)
        if not analysis:
            return data

        frame_data = data.get("frame") if data else None
        if frame_data is not None:
            data = {k: v for k, v in data.items() if k != "frame"}

        existing = analysis.get("latest_payload") or {}
        merged = {
            "analysis_id": analysis_id,
            "status": data.get("status", analysis.get("status", "pending")),
            "progress": data.get("progress", analysis.get("progress", 0.0)),
            "timestamp": time.time(),
        }
        merged.update(existing)
        # Only override values that are provided (not None)
        merged.update({k: v for k, v in data.items() if v is not None})

        if "stream_has_video" not in merged:
            merged["stream_has_video"] = analysis.get("stream_has_video")

        # Mirror important fields back to analysis record
        analysis["progress"] = merged.get("progress", analysis.get("progress", 0.0))
        if "current_frame" in merged:
            analysis["current_frame"] = merged.get("current_frame")
        if "total_frames" in merged and merged["total_frames"] is not None:
            analysis["total_frames"] = merged["total_frames"]
        for key in (
            "successful_reps",
            "unsuccessful_reps",
            "total_reps",
            "accuracy",
            "current_state",
            "instruction",
            "stream_has_video",
        ):
            if key in merged and merged[key] is not None:
                analysis[key] = merged[key]

        analysis["latest_payload"] = merged

        if frame_data is not None:
            analysis["latest_frame"] = frame_data
            analysis["latest_frame_timestamp"] = time.time()
            payload_with_frame = dict(merged)
            payload_with_frame["frame"] = frame_data
            return payload_with_frame

        return merged

    def _build_live_payload(
        self, analysis_id: str, include_frame: bool = True
    ) -> Dict[str, Any]:
        """Assemble the latest live payload for subscribers."""

        analysis = self.active_analyses.get(analysis_id)
        if not analysis:
            raise VideoProcessingError(f"Analysis {analysis_id} not found")

        base_payload: Dict[str, Any] = {
            "analysis_id": analysis_id,
            "status": analysis.get("status", "pending"),
            "progress": analysis.get("progress", 0.0),
            "stream_has_video": analysis.get("stream_has_video", False),
        }

        latest = analysis.get("latest_payload") or {}
        payload = {**base_payload, **latest}

        if include_frame and "frame" not in payload:
            frame_data = analysis.get("latest_frame")
            if frame_data is not None:
                payload["frame"] = frame_data

        return payload

    def get_latest_frame(self, analysis_id: str) -> Optional[str]:
        analysis = self.active_analyses.get(analysis_id)
        if not analysis:
            return None
        return analysis.get("latest_frame")

    def get_latest_frame_age(self, analysis_id: str) -> Optional[float]:
        analysis = self.active_analyses.get(analysis_id)
        if not analysis:
            return None
        timestamp = analysis.get("latest_frame_timestamp")
        if timestamp is None:
            return None
        return max(0.0, time.time() - float(timestamp))

    def _drop_subscriber_queue(self, analysis_id: str, queue: asyncio.Queue) -> None:
        """Remove a stale subscriber queue and log the cleanup."""

        analysis = self.active_analyses.get(analysis_id)
        if not analysis:
            return
        subscribers = analysis.get("subscribers", [])
        if queue in subscribers:
            subscribers.remove(queue)
            logger.debug(f"Removed stale subscriber for analysis {analysis_id}")

    def stop_analysis(self, analysis_id: str) -> bool:
        """Stop running analysis."""
        if analysis_id not in self.active_analyses:
            return False

        analysis = self.active_analyses[analysis_id]
        if analysis["status"] in ["completed", "failed", "cancelled"]:
            return False

        analysis["status"] = "cancelled"
        latest_frame_bus.close(analysis_id)
        logger.info(f"Analysis {analysis_id} cancelled")
        return True

    def cleanup_analysis(self, analysis_id: str) -> bool:
        """Clean up analysis files and records."""
        if analysis_id not in self.active_analyses:
            return False

        analysis = self.active_analyses[analysis_id]

        try:
            # Clean up uploaded video file
            if "video_path" in analysis and os.path.exists(analysis["video_path"]):
                os.remove(analysis["video_path"])
                logger.info(f"Cleaned up video file: {analysis['video_path']}")

            # Close frame bus for this analysis
            latest_frame_bus.close(analysis_id)

            # Remove analysis record
            del self.active_analyses[analysis_id]
            logger.info(f"Cleaned up analysis: {analysis_id}")
            return True

        except Exception as e:
            logger.error(f"Error cleaning up analysis {analysis_id}: {e}")
            return False


# Global service instance
video_service = VideoAnalysisService()
