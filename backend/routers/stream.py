import cv2
import json
import time
import base64
import asyncio
import platform
import numpy as np

from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Callable, Optional, Dict, Any

from core.config import Config
from engine.stream_processor import StreamProcessor
from engine.dual_camera_manager import DualCameraManager, CameraConfig, CameraPosition

router = APIRouter()

# Dependency function to get app state - set by main.py when router is included
_get_app_state: Optional[Callable[[], Dict[str, Any]]] = None


def set_app_state_getter(getter: Callable[[], Dict[str, Any]]):
    """Set the function to get app state. Called by main.py when router is included."""
    global _get_app_state
    _get_app_state = getter

# Helper functions
def open_camera(camera_id: int):
    """Open camera with best available backend."""
    start_time = time.time()
    
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        logger.warning("DirectShow failed/unavailable, trying default backend")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise IOError(f"Could not open camera {camera_id}")

    # Optimistic property setting
    for prop, val in [(cv2.CAP_PROP_FRAME_WIDTH, 640), (cv2.CAP_PROP_FRAME_HEIGHT, 480), (cv2.CAP_PROP_FPS, 30)]:
        cap.set(prop, val)

    logger.info(f"Camera opened in {time.time() - start_time:.2f}s")
    return cap

def create_processor(cfg, model, exercise_type="squat", experience_level="intermediate"):
    return StreamProcessor(cfg, exercise_type=exercise_type, experience_level=experience_level, model=model)


def create_dual_camera_manager(cfg) -> DualCameraManager:
    """Create dual camera manager from config."""
    dual_cfg = cfg.get("dual_camera", {})
    
    # Parse nested config
    front_cam_cfg = dual_cfg.get("front_camera", {})
    side_cam_cfg = dual_cfg.get("side_camera", {})

    front_config = CameraConfig(
        device_id=front_cam_cfg.get("device_id", 0),
        position=CameraPosition.FRONT,
        width=front_cam_cfg.get("width", 640),
        height=front_cam_cfg.get("height", 480),
        fps=front_cam_cfg.get("fps", 30),
        use_gstreamer=dual_cfg.get("use_gstreamer", False) or dual_cfg.get("sync_method") == "gstreamer",
        distance_from_subject_cm=front_cam_cfg.get("distance_from_subject_cm", 200),
        rotation=front_cam_cfg.get("rotation", 0)
    )
    
    side_config = CameraConfig(
        device_id=side_cam_cfg.get("device_id", 1),
        position=CameraPosition.SIDE,
        width=side_cam_cfg.get("width", 640),
        height=side_cam_cfg.get("height", 480),
        fps=side_cam_cfg.get("fps", 30),
        use_gstreamer=dual_cfg.get("use_gstreamer", False) or dual_cfg.get("sync_method") == "gstreamer",
        distance_from_subject_cm=side_cam_cfg.get("distance_from_subject_cm", 150),
        rotation=side_cam_cfg.get("rotation", 0)
    )
    
    return DualCameraManager(
        side_config=side_config,
        front_config=front_config,
        sync_threshold_ms=dual_cfg.get("sync_threshold_ms", 33.0)
    )


def encode_frame(frame) -> str:
    """Encode a single frame to base64 JPEG."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    _, encoded = cv2.imencode(".jpg", frame, encode_param)
    return f"data:image/jpeg;base64,{base64.b64encode(encoded).decode()}"


async def _send_frames(ws: WebSocket, raw, processed, metadata) -> bool:
    """Helper to encode and send frames. Returns True if successful, False if failed/disconnected."""
    try:
        await ws.send_json({
            "raw_frame": encode_frame(raw),
            "processed_frame": encode_frame(processed),
            "metadata": metadata
        })
        return True
    except Exception:
        return False # Client likely disconnected


async def _send_dual_frames(
    ws: WebSocket, 
    front_raw, front_processed,
    side_raw, side_processed,
    combined_frame,
    metadata,
    camera_status: Dict[str, bool]
) -> bool:
    """Helper to encode and send dual camera frames."""
    try:
        payload = {"metadata": metadata, "camera_status": camera_status}
        
        if front_raw is not None:
            payload["front_raw"] = encode_frame(front_raw)
        if front_processed is not None:
            payload["front_processed"] = encode_frame(front_processed)
        if side_raw is not None:
            payload["side_raw"] = encode_frame(side_raw)
        if side_processed is not None:
            payload["side_processed"] = encode_frame(side_processed)
        if combined_frame is not None:
            payload["combined_frame"] = encode_frame(combined_frame)
            
        await ws.send_json(payload)
        return True
    except Exception:
        return False

@router.websocket("/analyze")
async def websocket_endpoint(
    websocket: WebSocket, 
    target_reps: int = None,
    exercise_type: str = "squat",
    experience_level: str = "intermediate",
    dual_camera: str = "false"
):
    await websocket.accept()
    
    # Parse dual camera flag from request
    requested_dual_camera = dual_camera.lower() == "true"

    # Validate exercise type
    valid_exercises = [
        "squat", "pullup", "pushup", "dips", "lunges", "plank", "deadlift",
        "overhead_press", "bent_over_row", "glute_bridge", "wall_sit"
    ]
    if exercise_type not in valid_exercises:
        logger.warning(f"Invalid exercise type '{exercise_type}', defaulting to squat")
        exercise_type = "squat"
    
    # Validate experience level
    valid_levels = ["beginner", "intermediate", "advanced"]
    if experience_level not in valid_levels:
        logger.warning(f"Invalid experience level '{experience_level}', defaulting to intermediate")
        experience_level = "intermediate"

    # 1. Negotiate settings
    target_rep_count = target_reps
    try:
        # Quick check for initial config message
        initial_msg = await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
        msg_data = json.loads(initial_msg)
        if "target_reps" in msg_data:
            target_rep_count = int(msg_data["target_reps"])
            logger.info(f"Target rep count: {target_rep_count}")
        if "exercise_type" in msg_data and msg_data["exercise_type"] in valid_exercises:
            exercise_type = msg_data["exercise_type"]
        if "experience_level" in msg_data and msg_data["experience_level"] in valid_levels:
            experience_level = msg_data["experience_level"]
        if "dual_camera" in msg_data:
            requested_dual_camera = msg_data["dual_camera"] == True or msg_data["dual_camera"] == "true"
    except (asyncio.TimeoutError, ValueError, KeyError):
        pass # Continue if no config sent

    # 2. Setup Context
    cap = None
    dual_cam_manager = None
    processor = None
    # Will set executor size after checking config
    executor = None
    loop = asyncio.get_event_loop()

    try:
        # Load Config
        config = {}
        preloaded_model = None
        
        # Dependency injection for app state
        if _get_app_state:
            try:
                state = _get_app_state()
                if state.get("models_loaded"):
                    config = state["config"]
                    preloaded_model = state["yolo_model"]
            except Exception: 
                pass # Fallback to manual load

        if not config:
            config = Config.load_config("./config.yaml")

        # Check if dual camera is enabled in config
        dual_camera_config = config.get("dual_camera", {})
        dual_camera_enabled_in_config = dual_camera_config.get("enabled", False)
        
        # Only use dual camera if BOTH config enables it AND client requests it
        use_dual_camera = dual_camera_enabled_in_config and requested_dual_camera
        
        if requested_dual_camera and not dual_camera_enabled_in_config:
            logger.warning("Client requested dual camera but it's disabled in config. Using single camera.")
        
        logger.info(f"Dual Camera Decision: Config Enabled={dual_camera_enabled_in_config}, Client Requested={requested_dual_camera} (Value: {dual_camera})")
        
        # Now set executor size based on final decision
        executor = ThreadPoolExecutor(max_workers=4 if use_dual_camera else 2)
        
        logger.info(f"WebSocket connection for {exercise_type} (dual_camera={use_dual_camera})")

        # 3. Initialization Task
        if use_dual_camera:
            logger.info(f"Initializing dual cameras and processor for {exercise_type}...")
            
            # Create dual camera manager
            dual_cam_task = loop.run_in_executor(executor, create_dual_camera_manager, config)
            proc_task = loop.run_in_executor(
                executor, create_processor, config, preloaded_model, exercise_type, experience_level
            )
            
            dual_cam_manager, processor = await asyncio.gather(dual_cam_task, proc_task)
            
            # Enable dual camera on the exercise
            if hasattr(processor.exercise_manager.exercise, 'enable_dual_camera'):
                processor.exercise_manager.exercise.enable_dual_camera(True)
            
            # Open and start dual camera capture
            await loop.run_in_executor(executor, dual_cam_manager.open)
            await loop.run_in_executor(executor, dual_cam_manager.start)
            
            # Check camera status
            front_ok = dual_cam_manager.front_camera and dual_cam_manager.front_camera.is_running
            side_ok = dual_cam_manager.side_camera and dual_cam_manager.side_camera.is_running
            camera_status = {"front": front_ok, "side": side_ok}
            
            # Send Ready with camera status
            await websocket.send_json({
                "ready": True, 
                "message": "Dual cameras initialized",
                "camera_status": camera_status,
                "dual_camera_enabled": True
            })
        else:
            camera_id = config.get("video", {}).get("webcam_id", 0)
            logger.info(f"Initializing camera {camera_id} and processor for {exercise_type}...")
            
            # Run init in parallel
            cam_task = loop.run_in_executor(executor, open_camera, camera_id)
            proc_task = loop.run_in_executor(
                executor, create_processor, config, preloaded_model, exercise_type, experience_level
            )
            
            cap, processor = await asyncio.gather(cam_task, proc_task)
            
            # Ensure dual camera is disabled on the exercise
            if hasattr(processor.exercise_manager.exercise, 'enable_dual_camera'):
                processor.exercise_manager.exercise.enable_dual_camera(False)
            
            # Send Ready
            await websocket.send_json({
                "ready": True, 
                "message": "Initialized",
                "dual_camera_enabled": False
            })

        # 4. Stream Loop
        frame_interval = 1.0 / 30
        last_time = loop.time()
        rotation_angle = config.get("video", {}).get("rotation", 90)

        def rotate_frame(frame):
            """Apply rotation to frame based on config."""
            if rotation_angle == 90:
                return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_angle == 180:
                return cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation_angle == 270:
                return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return frame

        while True:
            # Rate limiting
            now = loop.time()
            if now - last_time < frame_interval:
                await asyncio.sleep(frame_interval - (now - last_time))
            last_time = loop.time()

            if use_dual_camera:
                # Dual camera mode
                synced_frame = await loop.run_in_executor(executor, dual_cam_manager.get_synced_frame)
                
                if synced_frame is None:
                    await asyncio.sleep(0.01)
                    continue
                
                front_raw = synced_frame.front_frame
                side_raw = synced_frame.side_frame
                
                # Check camera status
                front_ok = front_raw is not None
                side_ok = side_raw is not None
                camera_status = {"front": front_ok, "side": side_ok}
                
                if not front_ok and not side_ok:
                    logger.warning("Both camera frames missing")
                    await asyncio.sleep(0.1)
                    continue
                
                # Apply rotation to frames
                if front_raw is not None:
                    front_raw = rotate_frame(front_raw)
                if side_raw is not None:
                    side_raw = rotate_frame(side_raw)
                
                # Process frames - use side view as primary for exercises requiring side view
                # Front view as primary for exercises like pullup, dips
                frontal_exercises = ["pullup", "dips"]
                primary_frame = front_raw if exercise_type in frontal_exercises else (side_raw if side_raw is not None else front_raw)
                
                try:
                    # Process primary frame
                    processed, metadata = await loop.run_in_executor(
                        executor, processor.process_frame, primary_frame
                    )
                    
                    # Process secondary frame for visualization (reuse primary if same)
                    front_processed = None
                    side_processed = None
                    
                    # Determine which frame is primary to avoid double processing
                    is_front_primary = exercise_type in frontal_exercises
                    
                    if is_front_primary:
                        front_processed = processed
                        if side_raw is not None:
                            side_processed, _ = await loop.run_in_executor(
                                executor, processor.process_frame, side_raw
                            )
                    else:
                        side_processed = processed
                        if front_raw is not None:
                            # Use dedicated front view processing
                            front_processed, _ = await loop.run_in_executor(
                                executor, processor.process_frontal_view, front_raw
                            )
                    
                    # Create combined view (side-by-side)
                    combined_frame = None
                    if front_processed is not None and side_processed is not None:
                        # Validate dimensions before resize
                        if front_processed.shape[0] > 0 and side_processed.shape[0] > 0:
                            h = min(front_processed.shape[0], side_processed.shape[0])
                            front_resized = cv2.resize(front_processed, (int(front_processed.shape[1] * h / front_processed.shape[0]), h))
                            side_resized = cv2.resize(side_processed, (int(side_processed.shape[1] * h / side_processed.shape[0]), h))
                            combined_frame = np.hstack([front_resized, side_resized])
                            
                            # Add labels
                            cv2.putText(combined_frame, "FRONT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(combined_frame, "SIDE", (front_resized.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                    
                    # Check completion
                    if target_rep_count and metadata.get("total_reps", 0) >= target_rep_count:
                        metadata["completed"] = True
                        metadata["reason"] = "target_reached"
                        
                        logger.info("Target reps reached, finalizing session...")
                        timestamp = await loop.run_in_executor(executor, processor.finalize_session)
                        
                        metadata["redirect"] = True
                        metadata["exercise_type"] = exercise_type
                        if timestamp:
                            metadata["filename"] = f"webcam_stream_{exercise_type}_{timestamp}_final.mp4"
                        else:
                            metadata["filename"] = f"webcam_stream_{exercise_type}.mp4"

                        if not await _send_dual_frames(
                            websocket, front_raw, front_processed,
                            side_raw, side_processed, combined_frame,
                            metadata, camera_status
                        ):
                            break
                        break
                    
                    if not await _send_dual_frames(
                        websocket, front_raw, front_processed,
                        side_raw, side_processed, combined_frame,
                        metadata, camera_status
                    ):
                        break
                        
                except Exception as e:
                    logger.error(f"Dual camera processing error: {e}")
                    if not await _send_dual_frames(
                        websocket, front_raw, None, side_raw, None, None,
                        {"error": str(e)}, camera_status
                    ):
                        break
            else:
                # Single camera mode
                frame = await loop.run_in_executor(executor, cap.read)
                if not frame[0]:
                    logger.warning("Frame read failed")
                    await asyncio.sleep(0.1)
                    continue
                
                frame_img = rotate_frame(frame[1])

                try:
                    processed, metadata = await loop.run_in_executor(executor, processor.process_frame, frame_img)
                    
                    if target_rep_count and metadata.get("total_reps", 0) >= target_rep_count:
                        metadata["completed"] = True
                        metadata["reason"] = "target_reached"
                        
                        logger.info("Target reps reached, finalizing session...")
                        timestamp = await loop.run_in_executor(executor, processor.finalize_session)
                        
                        metadata["redirect"] = True
                        metadata["exercise_type"] = exercise_type
                        if timestamp:
                            metadata["filename"] = f"webcam_stream_{exercise_type}_{timestamp}_final.mp4"
                        else:
                            metadata["filename"] = f"webcam_stream_{exercise_type}.mp4"

                        if not await _send_frames(websocket, frame_img, processed, metadata):
                            break
                        break

                    if not await _send_frames(websocket, frame_img, processed, metadata):
                        break

                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    if not await _send_frames(websocket, frame_img, frame_img, {"error": str(e)}):
                        break

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Stream error: {e}")
    finally:
        # Cleanup
        if processor: 
            processor.finalize_session()
        if cap: 
            cap.release()
        if dual_cam_manager:
            dual_cam_manager.close()
        executor.shutdown(wait=False)
        try: 
            await websocket.close()
        except: 
            pass
