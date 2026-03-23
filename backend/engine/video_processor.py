"""
Enhanced video processing utilities with automatic rotation handling using ffmpeg.
"""

import base64
import os
import time
from typing import Any, Callable, Dict, Optional

import cv2
import ffmpeg
import numpy as np
from loguru import logger
from tqdm import tqdm
from ultralytics import YOLO

# Import required modules for the merged VideoProcessor class
from engine.audio_processing import SynchronizedOutputManager
from engine.exercises.exercise_manager import ExerciseManager
from engine.segmenter import PersonSegmenter
from engine.webcam_manager import SharedWebcamCapture
from streaming.latest_frame_bus import latest_frame_bus
from engine.core.pose_estimation import YOLOPoseEstimator, MediaPipePoseEstimator


class VideoProcessor:
    """Comprehensive video processor with automatic rotation handling and exercise analysis."""

    def __init__(
        self,
        analysis_id: str,
        config,
        exercise_type: str,
        use_webcam: bool = False,
        webcam_id: int = 0,
        video_path: str = None,
        force_front_view: bool = False,
        experience_level: str = "intermediate",
        video_id: str = None,  # Add video_id for monitoring
        progress_callback: Optional[Callable] = None,
        stop_requested: Optional[Callable] = None,
    ):
        """
        Initialize the video processor.

        Args:
            config: Configuration dictionary
            exercise_type: Type of exercise
            use_webcam: Whether to use webcam
            webcam_id: Webcam device ID
            video_path: Path to video file
            video_id: Unique ID for the video analysis session
        """
        self.analysis_id = analysis_id
        self.config = config
        self.exercise_type = exercise_type
        self.use_webcam = use_webcam
        self.webcam_id = webcam_id
        self.video_path = video_path
        self.video_id = video_id
        self.force_front_view = force_front_view
        self.experience_level = experience_level
        self.progress_callback = progress_callback
        self._stop_requested = stop_requested or (lambda: False)
        self.frame_emit_interval = 15  # emit ~2 fps assuming 30fps capture
        self.frame_max_width = 640
        self.suppress_preview = (
            config.get("video", {}).get("suppress_preview", False) if config else False
        )

        # Import monitor
        from core.monitor import monitor
        self.monitor = monitor

        # Initialize Pose Estimator based on config
        self.pose_estimator = self._create_pose_estimator()
        self.model = None # Deprecated direct access

        # Initialize segmentation model if enabled
        self.segmenter = None
        if config.get("segmentation", {}).get("enabled", True):
            segmentation_model_path = config["paths"]["segmentation_model"]
            self.segmenter = PersonSegmenter(segmentation_model_path)

        # Setup video capture
        self._setup_video_capture()

        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) if not use_webcam else 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not use_webcam:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Detect rotation for video files
            self.rotation_code = self.detect_rotation(self.video_path)
            # Adjust dimensions based on rotation
            if self.rotation_code in [1, 3]:  # 90 or 270 degrees
                self.width, self.height = self.height, self.width
        else:
            self.total_frames = float("inf")
            self.rotation_code = 0  # No rotation for webcam

        # Setup synchronized output manager
        self.output_manager = SynchronizedOutputManager(
            config, config["paths"]["output_dir"]
        )

        # Setup output
        self._setup_output()

        # Initialize exercise manager
        self.exercise_manager = ExerciseManager(
            config, exercise_type, fps=self.fps, segmenter=self.segmenter
        )
        if hasattr(self.exercise_manager.exercise, "set_experience_level"):
            self.exercise_manager.exercise.set_experience_level(self.experience_level)
            self.exercise_manager.exercise.set_front_view_mode(True)

    def _create_pose_estimator(self):
        """Factory method to create the appropriate pose estimator based on config."""
        pose_cfg = self.config.get("pose_estimation", {})
        provider = pose_cfg.get("provider", "yolo").lower()

        logger.info(f"Initializing pose estimator with provider: {provider}")

        if provider == "mediapipe":
            mp_config = pose_cfg.get("mediapipe", {})
            min_detection_confidence = mp_config.get("min_detection_confidence", 0.5)
            min_tracking_confidence = mp_config.get("min_tracking_confidence", 0.5)
            return MediaPipePoseEstimator(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        
        # Fallback or explicit YOLO
        yolo_config = pose_cfg.get("yolo", {})
        model_path = yolo_config.get("model_path", self.config["paths"]["model"])
        return YOLOPoseEstimator(model_path=model_path)

    def _setup_video_capture(self):
        """Setup video capture."""
        if self.use_webcam:
            # Use shared webcam capture to avoid device conflicts on Windows
            logger.info(f"Using shared webcam capture for webcam {self.webcam_id}")
            self.cap = SharedWebcamCapture(self.webcam_id)
        else:
            if not self.video_path:
                self.video_path = self.config["paths"]["input_video"]

            if not os.path.exists(self.video_path):
                raise FileNotFoundError(f"Video file not found: {self.video_path}")

            self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            source = (
                f"webcam {self.webcam_id} (shared)"
                if self.use_webcam
                else f"video {self.video_path}"
            )
            raise IOError(f"Could not open {source}")

    def _setup_output(self):
        """Setup output video writer."""
        # Generate output filename
        if self.use_webcam:
            prefix = f"webcam_{self.webcam_id}_squat"
        else:
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            prefix = f"{base_name}_squat"

        # Get synchronized video output path
        self.output_path = self.output_manager.get_video_output_path(prefix)

        # Setup video writer with correct dimensions (accounting for rotation)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )

        # Setup raw input video writer for webcam
        if self.use_webcam:
            raw_prefix = f"webcam_{self.webcam_id}_raw"
            raw_output_path = self.output_manager.get_video_output_path(
                raw_prefix, suffix="raw"
            )
            self.raw_video_writer = cv2.VideoWriter(
                raw_output_path, fourcc, self.fps, (self.width, self.height)
            )
            self.raw_output_path = raw_output_path
            logger.info(f"Raw webcam input will be saved to: {raw_output_path}")
        else:
            self.raw_video_writer = None
            self.raw_output_path = None

    def _setup_audio_output(self):
        """Setup audio output for voice messages."""
        if self.use_webcam:
            return

        # Prepare output directory
        output_dir = self.config["paths"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        # Generate audio output filename
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_output_filename = f"{base_name}_voice_{timestamp}.mp3"
        self.audio_output_path = os.path.join(output_dir, audio_output_filename)

        # Create an empty audio file to collect voice messages
        # We'll use this to store voice feedback during processing
        self.voice_messages = []
        self.voice_message_times = []

    def auto_rotate_frame(self, frame: np.ndarray, rotation_code: int) -> np.ndarray:
        """
        Automatically rotate a frame based on rotation code.

        Args:
            frame: Input frame
            rotation_code: Rotation code (0, 1, 2, 3 for 0, 90, 180, 270 degrees clockwise)

        Returns:
            Rotated frame
        """
        if rotation_code == 0:
            return frame
        elif rotation_code == 1:  # 90 degrees clockwise
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_code == 2:  # 180 degrees
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_code == 3:  # 270 degrees clockwise (90 counter-clockwise)
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return frame

    def detect_rotation(self, video_path: str) -> int:
        """
        Detect video rotation from metadata using ffmpeg.

        Args:
            video_path: Path to video file

        Returns:
            Rotation code (0, 1, 2, 3 for 0, 90, 180, 270 degrees)
        """
        try:
            # Check if file exists
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return 0

            # Use ffmpeg to probe video metadata
            probe = ffmpeg.probe(video_path)
            video_stream = next(
                (
                    stream
                    for stream in probe["streams"]
                    if stream["codec_type"] == "video"
                ),
                None,
            )

            if video_stream:
                # Check for rotation in side_data_list (newer method)
                if "side_data_list" in video_stream:
                    for side_data in video_stream["side_data_list"]:
                        if "rotation" in side_data:
                            rotation = int(float(side_data["rotation"]))
                            logger.info(f"Detected rotation from side_data: {rotation}")

                            # Convert rotation to our code system (0, 1, 2, 3 for 0, 90, 180, 270 degrees)
                            # Note: ffmpeg rotation is counterclockwise, so we need to convert
                            if rotation == 0:
                                return 0
                            elif rotation == -90 or rotation == 90:
                                return 1
                            elif rotation == 180 or rotation == -180:
                                return 2
                            elif rotation == -270 or rotation == 270:
                                return 3

                # Check for rotation tag in tags (older method)
                if "tags" in video_stream:
                    logger.debug(f"Video tags: {video_stream['tags']}")
                    # Check for rotation tag
                    rotation = video_stream["tags"].get("rotate", 0)
                    if isinstance(rotation, str):
                        rotation = int(rotation)

                    logger.info(f"Detected rotation from tags: {rotation}")

                    # Convert rotation to our code system (0, 1, 2, 3 for 0, 90, 180, 270 degrees)
                    if rotation == 0:
                        return 0
                    elif rotation == 90:
                        return 1
                    elif rotation == 180:
                        return 2
                    elif rotation == 270:
                        return 3

            logger.info("No rotation detected, returning 0")
            return 0  # Default no rotation
        except Exception as e:
            # If ffmpeg fails, return 0 (no rotation)
            logger.error(f"Error detecting rotation: {e}")
            return 0

    def process(self):
        """Process the video."""
        if self._stop_requested():
            logger.info(f"Cancellation requested before processing for {self.analysis_id}")
            return
        show_viz = self.config["video"].get("show_visualize", False)
        if self.suppress_preview:
            show_viz = False
        if self.use_webcam and show_viz:
            logger.info("Suppressing OpenCV preview for webcam analysis (stream handled upstream).")
            show_viz = False

        # Create a wrapper function that applies rotation during countdown
        def get_corrected_frame():
            ret, frame = self.cap.read()
            if not ret:
                return ret, frame

            # Apply automatic rotation correction for video files during countdown
            if not self.use_webcam and self.rotation_code != 0:
                frame = self.auto_rotate_frame(frame, self.rotation_code)

            return ret, frame

        # Define callback to publish annotated frames for WebRTC
        def publish_annotated_frame_callback(stage_name: str):
            """Create a callback to publish annotated frames for a given stage."""
            def callback(annotated_frame):
                """Publish annotated frame to frame bus for WebRTC."""
                if self._stop_requested():
                    return
                if self.use_webcam:
                    try:
                        meta = {
                            "stage": stage_name,
                            "timestamp": time.time(),
                        }
                        latest_frame_bus.publish(self.analysis_id, annotated_frame, meta)
                        logger.info(f"Published {stage_name} frame to bus for {self.analysis_id}")
                    except Exception as bus_exc:
                        logger.warning(f"Frame bus publish error during {stage_name}: {bus_exc}")
            return callback

        # Perform countdown with frame publication AND side detection
        self.exercise_manager.perform_countdown(
            self.cap,
            show_viz=show_viz,
            frame_processor=get_corrected_frame,
            on_annotated_frame=publish_annotated_frame_callback("countdown") if self.use_webcam else None,
            pose_estimator=self.pose_estimator,
        )

        # Skip separate positioning phase as it is now integrated into countdown
        logger.info("Countdown and side detection completed.")



        logger.success(
            "Side visibility detection completed! Starting exercise analysis..."
        )

        if not self.use_webcam:
            progress_bar = tqdm(
                total=self.total_frames,
                desc=f"Processing {self.exercise_manager.exercise_name} exercise",
                unit="frames",
            )
        else:
            logger.info(
                f"Processing webcam feed ({self.exercise_manager.exercise_name} exercise). Press 'q' to quit."
            )
            logger.info("Raw webcam input will be recorded alongside processed output.")

        frame_number = 0

        try:
            while True:
                if self._stop_requested():
                    logger.info(f"Cancellation observed in processing loop for {self.analysis_id}")
                    break
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_number += 1

                # Apply automatic rotation correction for video files
                if not self.use_webcam and self.rotation_code != 0:
                    frame = self.auto_rotate_frame(frame, self.rotation_code)

                # Save raw webcam frame if using webcam
                if self.use_webcam and self.raw_video_writer:
                    self.raw_video_writer.write(frame)

                # Run pose estimation
                results = self.pose_estimator.process_frame(frame)
                
                # Extract keypoints for internal logic
                # For validation zone, we might need standardized keypoints
                # MediaPipe doesn't have "boxes" in the same way YOLO does for person counting
                # But our standardized keypoints is (N, 3), but wait, the standardized keypoints 
                # is (17, 3), meaning SINGLE person.
                
                # If we need multi-person detection, our PoseEstimator interface might need 
                # to support returning multiple people or we just accept that MediaPipe is single person usually.
                # MediaPipe Pose detects SINGLE person usually unless using heavy models or tricks.
                # YOLO detects multiple.
                
                # Let's use get_keypoints which returns (17, 3) for the main person.
                keypoints = self.pose_estimator.get_keypoints(results, frame.shape[:2])

                # Validation: detection zone (middle third) and multi-person stop
                # Note: MediaPipe typically processes only one person or the most prominent one.
                # If we use YOLO, results is still the YOLO result object (list of Results).
                # To maintain feature parity for YOLO (person counting), we can check instance type.
                
                # Setup detection zone
                zone_x1 = int(self.width / 3)
                zone_x2 = int(2 * self.width / 3)

                persons_in_zone = 0
                if isinstance(self.pose_estimator, YOLOPoseEstimator):
                     # Use YOLO specific logic for multi-person safety check
                     try:
                        kp = results[0].keypoints
                        boxes = results[0].boxes if hasattr(results[0], "boxes") else None
                        
                        if kp is not None and len(kp.xy) > 0:
                            for person_idx in range(kp.xy.shape[0]):
                                pts = kp.xy[person_idx].cpu().numpy()
                                valid = pts[:, 0] > 0
                                if np.any(valid):
                                    cx = float(np.nanmean(pts[valid, 0]))
                                    if zone_x1 <= cx <= zone_x2:
                                        persons_in_zone += 1
                        elif boxes is not None and hasattr(boxes, "xyxy"):
                            bb = boxes.xyxy.cpu().numpy()
                            for b in bb:
                                cx = float(0.5 * (b[0] + b[2]))
                                if zone_x1 <= cx <= zone_x2:
                                    persons_in_zone += 1
                     except Exception as e:
                        logger.debug(f"YOLO Zone/person counting error: {e}")
                else:
                    # MediaPipe logic: It just returns one person usually.
                    # We can check if the single person is in the zone.
                    if keypoints is not None:
                         # Calculate center of mass or hip center
                         # Hips are 11 (left) and 12 (right)
                         # keypoints is (17, 3) -> [x, y, conf]
                         # indices for hips: 11, 12
                         left_hip = keypoints[11]
                         right_hip = keypoints[12]
                         center_x = 0
                         count = 0
                         if left_hip[2] > 0.5:
                             center_x += left_hip[0]
                             count += 1
                         if right_hip[2] > 0.5:
                             center_x += right_hip[0]
                             count += 1
                         
                         if count > 0:
                             center_x /= count
                             if zone_x1 <= center_x <= zone_x2:
                                 persons_in_zone = 1 # Just one person tracked

                # Visualize zone
                cv2.rectangle(
                    frame, (zone_x1, 0), (zone_x2, self.height), (0, 200, 255), 2
                )
                if persons_in_zone >= 2:
                    logger.error(
                        "Multiple persons detected in detection zone; stopping workout"
                    )
                    # Write the current frame with overlay then stop
                    if show_viz:
                        cv2.putText(
                            frame,
                            "Multiple persons in zone - stopping",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )
                        cv2.imshow("AI Trainer - Squat Analysis", frame)
                        cv2.waitKey(1)
                    break

                # Process frame with exercise analyzer
                # Process frame with exercise analyzer
                processed_frame = self.exercise_manager.process_frame(
                    frame=frame,
                    frame_number=frame_number,
                    keypoints=keypoints,
                    results=results # Pass results mainly for backward compatibility or direct access if needed
                )

                # Apply segmentation if enabled
                if self.segmenter is not None:
                    mask = self.segmenter.segment_person(frame)
                    if mask is not None:
                        # Get segmentation settings from config
                        seg_config = self.config.get("segmentation", {})
                        overlay_alpha = seg_config.get("overlay_alpha", 0.3)
                        contour_color = tuple(
                            seg_config.get("contour_color", [0, 255, 0])
                        )
                        contour_thickness = seg_config.get("contour_thickness", 2)

                        # Apply mask overlay
                        processed_frame = self.segmenter.apply_mask_overlay(
                            processed_frame, mask, contour_color, overlay_alpha
                        )

                        # Draw mask contour
                        processed_frame = self.segmenter.draw_mask_contour(
                            processed_frame, mask, contour_color, contour_thickness
                        )

                # Show visualization if enabled
                if show_viz:
                    cv2.imshow("AI Trainer - Squat Analysis", processed_frame)
                    if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
                        break

                # Write processed frame to output
                self.video_writer.write(processed_frame)

                # Broadcast frame if monitoring is active
                if self.video_id and not self.use_webcam:
                    try:
                        import base64
                        _, buffer = cv2.imencode('.jpg', processed_frame)
                        encoded_frame = base64.b64encode(buffer).decode('utf-8')
                        frame_data = f"data:image/jpeg;base64,{encoded_frame}"
                        
                        # We need to run this async, but we are in a sync method
                        # Use asyncio.run or create a task if there's a running loop
                        # Since this is running in a threadpool (likely), we can use a helper
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                        if loop.is_running():
                            # If we are in an async context (which we shouldn't be for this heavy CPU task)
                            # But actually run_analysis is async def but runs this in threadpool?
                            # No, run_analysis calls processor.process() which is sync.
                            # We need to fire and forget.
                            asyncio.run_coroutine_threadsafe(
                                self.monitor.broadcast_frame(self.video_id, frame_data),
                                loop
                            )
                        else:
                            # If no loop is running in this thread
                            loop.run_until_complete(self.monitor.broadcast_frame(self.video_id, frame_data))
                            
                    except Exception as e:
                        logger.warning(f"Failed to broadcast frame: {e}")

                if not self.use_webcam:
                    progress_bar.update(1)

                self._emit_metrics(frame_number, processed_frame)

        finally:
            # Cleanup
            if not self.use_webcam:
                progress_bar.close()

            self.cap.release()
            self.video_writer.release()

            # Close raw video writer if it exists
            if self.raw_video_writer:
                self.raw_video_writer.release()
                logger.info(f"Raw webcam input saved to: {self.raw_output_path}")

            cv2.destroyAllWindows()

            # Check if processing was stopped early
            was_stopped_early = self._stop_requested()
            if was_stopped_early:
                logger.info("Processing was stopped early by user request")

            # Finalize exercise analysis with synchronized timestamp
            synchronized_timestamp = self.output_manager.get_base_filename()
            self.exercise_manager.finalize_analysis(
                self.config["paths"]["output_dir"], synchronized_timestamp
            )

            # Create synchronized outputs
            self._create_synchronized_outputs()

            # Print results
            results = self.exercise_manager.get_results()
            logger.info("\nProcessing complete!")
            logger.info(f"Total frames processed: {frame_number}")
            logger.info(f"Successful reps: {results['successful_reps']}")
            logger.info(f"Unsuccessful reps: {results['unsuccessful_reps']}")
            logger.info(f"Total reps: {results['total_reps']}")
            logger.info(f"Detected facing side: {results['facing_side']}")
            logger.info(f"Output video saved to: {self.output_path}")

            if was_stopped_early:
                logger.info("Note: Analysis was stopped early, results may be incomplete")

            # Show raw video path if using webcam
            if self.use_webcam and self.raw_output_path:
                logger.info(f"Raw webcam input saved to: {self.raw_output_path}")

            # Print synchronized output information
            base_filename = self.output_manager.get_base_filename()
            logger.info(f"Synchronized outputs created with timestamp: {base_filename}")

    def _create_synchronized_outputs(self):
        """Create synchronized video, audio, and data outputs."""
        if self.use_webcam:
            return

        try:
            # Get voice messages from exercise manager
            voice_messages = self.exercise_manager.get_voice_messages()

            # Record voice messages in the synchronized output manager
            for timestamp, message_type in voice_messages:
                self.output_manager.record_voice_message(message_type, timestamp)

            # Calculate video duration
            duration = self.total_frames / self.fps

            # Save synchronized metadata
            metadata = {
                "input_video": self.video_path,
                "output_video": self.output_path,
                "duration": duration,
                "fps": self.fps,
                "total_frames": self.total_frames,
                "voice_messages_count": len(voice_messages),
                "voice_messages": voice_messages,
                "anthropometrics": self.exercise_manager.get_anthropometrics(),
                "back_curvature": getattr(
                    self.exercise_manager, "back_curvature", None
                ),
            }

            metadata_path = self.output_manager.save_synchronized_metadata(metadata)

            # Always create audio track if voice messages exist
            if voice_messages:
                audio_path = self.output_manager.create_audio_track(duration)
                if audio_path:
                    # Generate final merged output
                    final_output_path = self.output_path.replace(".mp4", "_final.mp4")
                    if self.output_manager.merge_video_audio(
                        self.output_path, audio_path, final_output_path
                    ):
                        logger.info(f"Final merged video saved to: {final_output_path}")
                    else:
                        logger.warning(
                            f"Could not merge audio with video. Audio track saved to: {audio_path}"
                        )
                else:
                    logger.warning("Could not create audio track")
            else:
                logger.info("No voice messages to process for audio creation")

            # Print synchronized output information
            base_filename = self.output_manager.get_base_filename()
            logger.info(f"Synchronized outputs created with timestamp: {base_filename}")
            logger.info(f"Metadata saved to: {metadata_path}")

        except Exception as e:
            logger.warning(f"Could not create synchronized outputs: {e}")

    def _emit_metrics(self, frame_number: int, frame) -> None:
        """Emit per-frame metrics through the provided callback."""
        if not self.progress_callback:
            return
        if self._stop_requested():
            return

        try:
            summary = self.exercise_manager.get_results()
            successful = summary.get("successful_reps", 0)
            unsuccessful = summary.get("unsuccessful_reps", 0)
            total = summary.get("total_reps", 0)
            accuracy = round(successful / total, 4) if total else 1.0
            state = summary.get("current_state")

            progress = min(100.0, (total / 10.0) * 100.0)

            payload: Dict[str, Any] = {
                "analysis_id": self.analysis_id,
                "event": "metrics",
                "successful_reps": successful,
                "unsuccessful_reps": unsuccessful,
                "total_reps": total,
                "accuracy": accuracy,
                "current_state": state,
                "instruction": self._instruction_for_state(state),
                "current_frame": frame_number,
                "progress": progress,
                "timestamp": time.time(),
            }

            if "front_view_mode" in summary:
                payload["front_view_mode"] = summary.get("front_view_mode")

            # Publish to LatestFrameBus for WebRTC
            if self.use_webcam and frame is not None:
                try:
                    meta = {
                        "frame_number": frame_number,
                        "timestamp": time.time(),
                    }
                    latest_frame_bus.publish(self.analysis_id, frame, meta)
                except Exception as bus_exc:
                    logger.debug(f"Frame bus publish error: {bus_exc}")

            include_frame = (
                self.use_webcam
                and frame is not None
                and self.progress_callback is not None
                and (frame_number % self.frame_emit_interval) == 0
            )

            if include_frame:
                try:
                    preview = frame
                    height, width = preview.shape[:2]
                    if width > self.frame_max_width:
                        scale = self.frame_max_width / width
                        preview = cv2.resize(
                            preview,
                            (int(width * scale), int(height * scale)),
                            interpolation=cv2.INTER_AREA,
                        )

                    success, buffer = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if success:
                        payload["frame"] = base64.b64encode(buffer.tobytes()).decode(
                            "ascii"
                        )
                except Exception as img_exc:
                    logger.debug(f"Frame encoding failed: {img_exc}")

            self.progress_callback(payload)
        except Exception as exc:
            logger.debug(f"Metrics callback error: {exc}")

    def _instruction_for_state(self, state: Optional[str]) -> str:
        mapping = {
            "going_down": "Lower with control",
            "coming_up": "Drive up through your heels",
            "up": "Great job, prepare for the next rep",
        }
        return mapping.get(state or "", "Maintain controlled movement")
