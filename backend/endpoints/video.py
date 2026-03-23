"""
Video analysis endpoints.
"""

from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger

from models.requests import WebcamAnalysisRequest
from models.responses import AnalysisResponse, AnalysisStatusResponse
from services.video_service import video_service
from services.analysis_service import analysis_service
# from services.preview_relay import preview_frame_relay  # TODO: Preview relay not needed for WebRTC
from core.exceptions import (
    VideoProcessingError,
    InvalidFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    AnalysisNotFoundError,
)

router = APIRouter(prefix="/api/v1/video", tags=["Video Analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to analyze"),
    exercise_type: str = Form(default="squat", description="Exercise type"),
    experience_level: str = Form(
        default="intermediate", description="Experience level"
    ),
    force_front_view: bool = Form(default=False, description="Force front view"),
    enable_voice_feedback: bool = Form(
        default=True, description="Enable voice feedback"
    ),
    enable_segmentation: bool = Form(default=True, description="Enable segmentation"),
    show_visualization: bool = Form(default=False, description="Show visualization"),
    voice_volume: float = Form(default=0.7, ge=0.0, le=1.0, description="Voice volume"),
):
    """
    Analyze uploaded video file for exercise form.

    This endpoint accepts a video file and analyzes it for exercise form,
    providing detailed metrics and feedback.
    """
    try:
        # Validate inputs
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Validate exercise type
        valid_exercises = ["squat", "pullup", "pushup", "dips", "lunges", "plank", "deadlift", "overhead_press", "bent_over_row", "glute_bridge", "wall_sit"]
        if exercise_type not in valid_exercises:
            raise HTTPException(status_code=400, detail=f"Exercise type '{exercise_type}' not supported. Valid types: {', '.join(valid_exercises)}")

        # Validate experience level
        if experience_level not in ["beginner", "intermediate", "advanced"]:
            raise HTTPException(status_code=400, detail=f"Invalid experience level: {experience_level}")

        # Validate voice volume
        if not 0.0 <= voice_volume <= 1.0:
            raise HTTPException(status_code=400, detail="Voice volume must be between 0.0 and 1.0")

        # Create request data
        request_data = {
            "exercise_type": exercise_type,
            "experience_level": experience_level,
            "force_front_view": force_front_view,
            "enable_voice_feedback": enable_voice_feedback,
            "enable_segmentation": enable_segmentation,
            "show_visualization": show_visualization,
            "voice_volume": voice_volume,
        }

        # Create analysis
        analysis_id = await video_service.analyze_video_file(file, request_data)

        # Start background processing
        background_tasks.add_task(process_video_analysis, analysis_id)

        # Get analysis info
        analysis_info = video_service.get_analysis_status(analysis_id)

        return AnalysisResponse(
            analysis_id=analysis_id,
            status=analysis_info["status"],
            exercise_type=analysis_info["exercise_type"],
            experience_level=analysis_info["experience_level"],
            created_at=analysis_info["created_at"],
        )

    except HTTPException:
        # Re-raise HTTP exceptions to maintain proper error codes
        raise
    except (InvalidFileError, FileTooLargeError, UnsupportedFileTypeError) as e:
        logger.warning(f"File validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except VideoProcessingError as e:
        logger.error(f"Video processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in video analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/webcam/preview/{webcam_id}")
async def webcam_preview(webcam_id: int = 0):
    """Stream live webcam frames as an MJPEG feed without running analysis."""

    try:
        generator = video_service.stream_webcam_preview(webcam_id)
        return StreamingResponse(
            generator,
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    except VideoProcessingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Unexpected error starting webcam preview: {exc}")
        raise HTTPException(status_code=500, detail="Failed to start webcam preview")


@router.post("/webcam/start", response_model=AnalysisResponse)
async def start_webcam_analysis(
    request: Request, background_tasks: BackgroundTasks, payload: WebcamAnalysisRequest
):
    """
    Start real-time webcam analysis.

    This endpoint starts a real-time analysis session using the webcam.
    The analysis runs continuously until stopped.
    """
    try:
        # Create or reuse analysis
        request_data = payload.dict()
        (
            analysis_id,
            analysis_info,
            created_new,
        ) = await video_service.start_webcam_analysis(request_data)

        # Start background processing
        if created_new:
            background_tasks.add_task(process_webcam_analysis, analysis_id)

        forwarded_header = request.headers.get("forwarded")
        forwarded_proto = request.headers.get("x-forwarded-proto")
        forwarded_host = request.headers.get("x-forwarded-host")

        def _parse_forwarded(header: Optional[str], key: str) -> Optional[str]:
            if not header:
                return None
            for item in header.split(","):
                for segment in item.split(";"):
                    if "=" not in segment:
                        continue
                    name, value = segment.split("=", 1)
                    if name.strip().lower() == key:
                        return value.strip().strip('"')
            return None

        def _normalize_proto(proto: Optional[str]) -> Optional[str]:
            if not proto:
                return None
            candidate = proto.strip().lower()
            if candidate in {"https", "wss"}:
                return "https"
            if candidate in {"http", "ws"}:
                return "http"
            return None

        proto_candidate: Optional[str] = None
        if forwarded_proto:
            proto_candidate = _normalize_proto(forwarded_proto.split(",")[0])
        if not proto_candidate:
            proto_candidate = _normalize_proto(_parse_forwarded(forwarded_header, "proto"))
        if not proto_candidate:
            proto_candidate = _normalize_proto(request.url.scheme)

        ws_scheme = "wss" if proto_candidate == "https" else "ws"

        host_candidate: Optional[str] = None
        if forwarded_host:
            host_candidate = forwarded_host.split(",")[0].strip()
        if not host_candidate:
            host_candidate = _parse_forwarded(forwarded_header, "host")
        if not host_candidate:
            host_candidate = request.headers.get("host") or request.url.netloc

        stream_url = request.url_for("stream_analysis", analysis_id=analysis_id)
        stream_url = stream_url.replace(scheme=ws_scheme)
        if host_candidate:
            stream_url = stream_url.replace(netloc=host_candidate)
        ws_url = str(stream_url)

        poll_interval_ms = 1000
        reconnect_backoff_ms = 750

        return AnalysisResponse(
            analysis_id=analysis_id,
            status=analysis_info["status"],
            exercise_type=analysis_info["exercise_type"],
            experience_level=analysis_info["experience_level"],
            created_at=analysis_info["created_at"],
            websocket_url=ws_url,
            stream_has_video=analysis_info.get("stream_has_video", False),
            poll_interval_ms=poll_interval_ms,
            reconnect_backoff_ms=reconnect_backoff_ms,
        )

    except VideoProcessingError as e:
        logger.error(f"Webcam analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": str(e),
                "hint": "Ensure the webcam is available and not used by another application.",
            },
        )
    except Exception as e:
        logger.error(f"Unexpected error in webcam analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Unexpected error while starting webcam analysis",
                "hint": "Check API logs for more information",
            },
        )


@router.get("/analysis/{analysis_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(analysis_id: str):
    """
    Get analysis status and progress.

    Returns the current status and progress of a video analysis.
    """
    try:
        # Validate analysis_id format to prevent path traversal
        if not _is_valid_analysis_id(analysis_id):
            logger.warning(f"Invalid analysis ID format: {analysis_id}")
            raise HTTPException(status_code=400, detail="Invalid analysis ID format")

        analysis_info = video_service.get_analysis_status(analysis_id)

        # Calculate processing time if analysis has started
        processing_time = None
        if analysis_info.get("created_at"):
            from datetime import datetime

            created_at = analysis_info["created_at"]
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except ValueError:
                    logger.warning(f"Invalid datetime format in analysis info: {created_at}")
                    created_at = datetime.now()
            processing_time = (datetime.now() - created_at).total_seconds()

        # Map status to proper enum value
        status = analysis_info.get("status", "pending")
        if status not in ["pending", "processing", "completed", "failed", "cancelled"]:
            status = "pending"

        return AnalysisStatusResponse(
            analysis_id=analysis_id,
            status=status,
            progress_percentage=analysis_info.get("progress_percentage", 0.0),
            current_frame=analysis_info.get("current_frame", 0),
            total_frames=analysis_info.get("total_frames"),
            processing_time=processing_time,
            error_message=analysis_info.get("error_message"),
            successful_reps=analysis_info.get("successful_reps"),
            unsuccessful_reps=analysis_info.get("unsuccessful_reps"),
            total_reps=analysis_info.get("total_reps"),
            accuracy=analysis_info.get("accuracy"),
            current_state=analysis_info.get("current_state"),
            instruction=analysis_info.get("instruction"),
            current_instruction=analysis_info.get("current_instruction"),
            frame=analysis_info.get("frame"),
            stream_has_video=analysis_info.get("stream_has_video"),
        )

    except HTTPException:
        # Re-raise HTTP exceptions to maintain proper error codes
        raise
    except VideoProcessingError as e:
        logger.warning(f"Analysis not found: {analysis_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting analysis status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def _is_valid_analysis_id(analysis_id: str) -> bool:
    """Validate analysis ID format to prevent path traversal."""
    import re
    # Only allow alphanumeric characters, hyphens, and underscores
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', analysis_id))


@router.get("/analysis/{analysis_id}/frame")
async def get_analysis_frame(analysis_id: str):
    try:
        frame = video_service.get_latest_frame(analysis_id)
        if not frame:
            raise HTTPException(status_code=404, detail="No frame available")

        age = video_service.get_latest_frame_age(analysis_id) or 0.0
        return JSONResponse(
            content={
                "analysis_id": analysis_id,
                "frame": frame,
                "age_seconds": age,
            }
        )

    except HTTPException:
        raise
    except VideoProcessingError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"Error retrieving frame for analysis {analysis_id}: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analysis/{analysis_id}/preview")
async def get_analysis_preview(analysis_id: str):
    """
    Get preview frame for analysis.
    
    Note: This endpoint is deprecated in favor of WebRTC streaming.
    Returns 204 No Content to indicate no preview frames are available via this method.
    """
    # Preview frames are now streamed via WebRTC
    return Response(status_code=204)


@router.websocket("/analysis/{analysis_id}/stream")
async def stream_analysis(websocket: WebSocket, analysis_id: str):
    """WebSocket endpoint that streams real-time analysis metrics.

    Payload contract (JSON objects):

    - ``analysis_id``: ID for the active session.
    - ``event``: Either ``status`` or ``metrics`` depending on origin.
    - ``status``: Current analysis status (pending|processing|completed|failed|cancelled).
    - ``progress``: Numeric percentage 0-100.
    - ``successful_reps``/``unsuccessful_reps``/``total_reps``: Running counts.
    - ``accuracy``: Normalised accuracy (0-1).
    - ``instruction``: Latest coaching cue.
    - ``current_state``: Exercise state machine tag.
    - ``stream_has_video``: Boolean indicating whether ``frame`` is expected.
    - ``frame`` (optional): Base64 encoded JPEG preview well suited for browser ``<img>`` tags.
    - ``timestamp``: Epoch seconds when payload was produced.

    Clients should listen for both ``status`` and ``metrics`` events, using the latest
    occurrence of each field for display. Missing fields indicate "no change".
    """

    await websocket.accept()
    try:
        queue = await video_service.register_subscriber(analysis_id)
    except VideoProcessingError as exc:
        await websocket.close(code=4404, reason=str(exc))
        return
    except Exception as exc:
        await websocket.close(code=1011, reason=str(exc))
        return

    try:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for analysis {analysis_id}")
    except Exception as exc:
        logger.warning(f"WebSocket error for analysis {analysis_id}: {exc}")
    finally:
        video_service.unregister_subscriber(analysis_id, queue)


@router.post("/analysis/{analysis_id}/stop")
async def stop_analysis(analysis_id: str):
    """
    Stop running analysis.

    Stops a running analysis (primarily for webcam analysis).
    """
    try:
        webcam_id = video_service.get_webcam_id(analysis_id)
        success = video_service.stop_analysis(analysis_id)

        if not success:
            raise HTTPException(
                status_code=404, detail="Analysis not found or not running"
            )

        # Preview relay cleanup not needed with WebRTC streaming
        # Cleanup is handled automatically when WebRTC connection closes
        try:
            # Notify subscribers of cancellation
            await video_service._notify_status(
                analysis_id, status="cancelled", message="Analysis cancelled by client"
            )
        except Exception:
            pass

        return {"message": f"Analysis {analysis_id} stopped successfully"}

    except Exception as e:
        logger.error(f"Error stopping analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """
    Delete analysis and clean up files.

    Removes analysis data and cleans up associated files.
    """
    try:
        webcam_id = video_service.get_webcam_id(analysis_id)
        success = video_service.cleanup_analysis(analysis_id)

        if not success:
            raise HTTPException(status_code=404, detail="Analysis not found")

        # Preview relay cleanup not needed with WebRTC streaming
        # Cleanup is handled automatically when WebRTC connection closes

        return {"message": f"Analysis {analysis_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analysis/{analysis_id}/download")
async def download_analysis_video(analysis_id: str):
    """
    Download processed analysis video.

    Downloads the processed video with analysis overlays.
    """
    try:
        import os
        from fastapi.responses import FileResponse

        video_path = analysis_service.get_output_file_path(analysis_id, "video")

        if not video_path or not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video file not found")

        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename=f"analysis_{analysis_id}.mp4",
        )

    except AnalysisNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def process_video_analysis(analysis_id: str):
    """Background task to process video analysis."""
    try:
        logger.info(f"Starting background processing for analysis {analysis_id}")

        async for progress_update in video_service.process_analysis(analysis_id):
            logger.info(f"Analysis {analysis_id} progress: {progress_update}")

    except Exception as e:
        logger.error(f"Error in background video processing: {e}")


async def process_webcam_analysis(analysis_id: str):
    """Background task to process webcam analysis."""
    try:
        logger.info(f"Starting background webcam processing for analysis {analysis_id}")

        async for progress_update in video_service.process_analysis(analysis_id):
            logger.info(f"Webcam analysis {analysis_id} progress: {progress_update}")

    except Exception as e:
        logger.error(f"Error in background webcam processing: {e}")
