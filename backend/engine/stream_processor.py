import cv2
import os
import time
import numpy as np
from enum import Enum
from loguru import logger
from datetime import datetime
from typing import Dict, Any, Tuple

from engine.core.utils import _draw_positioning_info
from engine.exercises.exercise_manager import ExerciseManager
from engine.audio_processing import SynchronizedOutputManager
from engine.core.pose_estimation import YOLOPoseEstimator, MediaPipePoseEstimator


class StreamState(Enum):
    COUNTDOWN = "countdown"
    POSITIONING = "positioning"
    EXERCISING = "exercising"
    COMPLETED = "completed"


class StreamProcessor:

    def __init__(
        self,
        config: Dict[str, Any],
        exercise_type: str = "squat",
        experience_level: str = "intermediate",
        model: Any = None,  # Optional pre-loaded YOLO model
    ):
        self.config = config
        self.exercise_type = exercise_type
        self.experience_level = experience_level

        # Initialize state - skip countdown, start with positioning
        self.state = StreamState.POSITIONING
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 30.0  # Assumed FPS for streaming
        
        # Initialize Pose Estimator based on config
        self._init_pose_estimator()

        # Initialize exercise manager
        self.exercise_manager = ExerciseManager(config, exercise_type, fps=self.fps)
        if hasattr(self.exercise_manager.exercise, "set_experience_level"):
            self.exercise_manager.exercise.set_experience_level(self.experience_level)

    def _init_pose_estimator(self):
        """Initialize appropriate pose estimator."""
        pose_cfg = self.config.get("pose_estimation", {})
        provider = pose_cfg.get("provider", "yolo").lower()
        
        logger.info(f"StreamProcessor initializing pose estimator: {provider}")
        
        if provider == "mediapipe":
            mp_config = pose_cfg.get("mediapipe", {})
            self.pose_estimator = MediaPipePoseEstimator(
                min_detection_confidence=mp_config.get("min_detection_confidence", 0.5),
                min_tracking_confidence=mp_config.get("min_tracking_confidence", 0.5)
            )
        else:
            # YOLO Fallback
            yolo_config = pose_cfg.get("yolo", {})
            model_path = yolo_config.get("model_path", self.config["paths"]["model"])
            self.pose_estimator = YOLOPoseEstimator(model_path=model_path)

        # State-specific variables
        self.countdown_duration = self.config.get("countdown", {}).get("time", 5)

        # Positioning variables
        self.pos_start_time = None
        self.left_confidences = []
        self.right_confidences = []
        self.pos_frame_count = 0

        # Output Management for Recording
        self.output_manager = SynchronizedOutputManager(
            self.config, self.config["paths"]["output_dir"]
        )
        self.video_writer = None
        self.raw_video_writer = None
        self.session_active = True
        
        # Initialize video writers
        self._setup_video_writers()

        logger.info(f"StreamProcessor initialized for {self.exercise_type}")

    def _setup_video_writers(self):
        """Setup video writer paths. Writers are initialized on first frame."""
        # Initialize to None
        self.video_writer = None
        self.raw_video_writer = None
        self.front_video_writer = None
        self.raw_front_video_writer = None
        
        # Determine paths
        try:
             # Processed output path
            prefix = f"webcam_stream_{self.exercise_type}"
            self.output_path = self.output_manager.get_video_output_path(prefix)
            
            # Raw output path
            raw_prefix = f"webcam_stream_{self.exercise_type}_raw"
            self.raw_output_path = self.output_manager.get_video_output_path(
                raw_prefix, suffix="raw"
            )
            
            logger.info(f"Video paths setup: {self.output_path}")
            
        except Exception as e:
            logger.error(f"Failed to setup video paths: {e}")

    def _init_front_writer(self, width: int, height: int):
        """Initialize video writer for front camera."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            
            # Front camera output path
            front_prefix = f"webcam_stream_{self.exercise_type}_front"
            self.front_output_path = self.output_manager.get_video_output_path(
                front_prefix
            )
            
            self.front_video_writer = cv2.VideoWriter(
                self.front_output_path, fourcc, self.fps, (width, height)
            )
            
            # Raw Front camera output path
            raw_front_prefix = f"webcam_stream_{self.exercise_type}_front_raw"
            self.raw_front_output_path = self.output_manager.get_video_output_path(
                raw_front_prefix, suffix="raw"
            )
            
            self.raw_front_video_writer = cv2.VideoWriter(
                self.raw_front_output_path, fourcc, self.fps, (width, height)
            )
            
            logger.info(f"Initialized FRONT video writers: {self.front_output_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize front video writer: {e}")

    def _init_writers(self, width: int, height: int):
        """Initialize video writers with specific dimensions."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (width, height)
            )
            
            self.raw_video_writer = cv2.VideoWriter(
                self.raw_output_path, fourcc, self.fps, (width, height)
            )
            
            self.width = width
            self.height = height
            logger.info(f"Initialized video writers with resolution {width}x{height}")
            
        except Exception as e:
            logger.error(f"Failed to initialize video writers: {e}")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.frame_count += 1
        
        # Initialize writers on first frame
        if self.video_writer is None:
            h, w = frame.shape[:2]
            self._init_writers(w, h)
        
        # Save raw frame
        if self.raw_video_writer:
            self.raw_video_writer.write(frame)
            
        metadata = {"state": self.state.value}
        processed_frame = frame

        if self.state == StreamState.POSITIONING:
            processed_frame = self._process_positioning(frame)
        elif self.state == StreamState.EXERCISING:
            processed_frame = self._process_exercising(frame)
        
        # Save processed frame
        if self.video_writer:
            self.video_writer.write(processed_frame)
            
        return processed_frame, metadata

    def _process_positioning(self, frame: np.ndarray) -> np.ndarray:
        # Positioning logic adapted from side_visibility_detector
        # We need to accumulate confidence over 1 second (approx 30 frames)

        if self.pos_start_time is None:
            self.pos_start_time = time.time()

        # Run inference
        results = self.pose_estimator.process_frame(frame)

        # Extract confidences
        left_conf = 0.0
        right_conf = 0.0

        try:
            keypoints = self.pose_estimator.get_keypoints(results, frame.shape[:2])
            if keypoints is not None:
                confidences = keypoints[:, 2]

                # Indices from utils.py
                left_idxs = [1, 3, 5, 7, 9]
                right_idxs = [2, 4, 6, 8, 10]

                left_vals = [confidences[i] for i in left_idxs if i < len(confidences)]
                right_vals = [
                    confidences[i] for i in right_idxs if i < len(confidences)
                ]

                if left_vals:
                    left_conf = float(np.mean(left_vals))
                if right_vals:
                    right_conf = float(np.mean(right_vals))
        except Exception as e:
            logger.debug(f"Pose estimation error: {e}")

        self.left_confidences.append(left_conf)
        self.right_confidences.append(right_conf)
        self.pos_frame_count += 1

        # Check every ~1 second (30 frames)
        if self.pos_frame_count >= 30:
            avg_left = np.mean(self.left_confidences)
            avg_right = np.mean(self.right_confidences)
            diff = abs(avg_left - avg_right)

            threshold = self.config.get("side_difference", {}).get(
                "confidence_threshold", 0.25
            )

            if diff >= threshold:
                self.state = StreamState.EXERCISING
                logger.info(
                    f"Positioning complete. Diff: {diff:.3f}. Starting EXERCISING"
                )
                # Reset for next phase
                self.left_confidences = []
                self.right_confidences = []
                self.pos_frame_count = 0
                return frame

            # Reset buffers but keep trying
            self.left_confidences = []
            self.right_confidences = []
            self.pos_frame_count = 0

            # Check timeout
            timeout = self.config.get("side_difference", {}).get("timeout_seconds", 10)
            if time.time() - self.pos_start_time > timeout:
                logger.warning("Positioning timeout, forcing front view")
                if hasattr(self.exercise_manager.exercise, "set_front_view_mode"):
                    self.exercise_manager.exercise.set_front_view_mode(True)
                self.state = StreamState.EXERCISING
                return frame

        # Draw visualization
        threshold = self.config.get("side_difference", {}).get(
            "confidence_threshold", 0.25
        )
        timeout = self.config.get("side_difference", {}).get("timeout_seconds", 10)
        remaining = timeout - (time.time() - self.pos_start_time)

        return _draw_positioning_info(
            frame, left_conf, right_conf, threshold, self.pos_frame_count, 30, remaining
        )

    def _process_exercising(self, frame: np.ndarray) -> np.ndarray:
        """Process frame during exercising phase."""
        try:
            # Run inference
            results = self.pose_estimator.process_frame(frame)
            keypoints = self.pose_estimator.get_keypoints(results, frame.shape[:2])

            # Process with exercise manager
            processed_frame = self.exercise_manager.process_frame(
                frame=frame,
                frame_number=self.frame_count,
                keypoints=keypoints,
                results=results
            )

            return processed_frame
        except Exception as e:
            logger.error(f"Error in _process_exercising: {e}", exc_info=True)
            # Return original frame if processing fails
            return frame

    def process_frontal_view(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process secondary front view frame for dual camera mode.
        Calculates and draws front-view specific metrics (knee valgus, etc).
        """
        if frame is None:
            return frame, {}
            
        # Keep a copy for raw recording because drawing modifies in-place
        raw_frame = frame.copy()
            
        # Pose estimation
        results = self.pose_estimator.process_frame(frame)
        keypoints = self.pose_estimator.get_keypoints(results, frame.shape[:2])
        
        metadata = {}
        
        if keypoints is not None:
            # Check if exercise supports front view processing
            exercise = self.exercise_manager.exercise
            if hasattr(exercise, "process_front_frame"):
                # Get metrics
                metrics = exercise.process_front_frame(
                    keypoints, frame.shape[:2], self.frame_count
                )
                metadata = metrics
                
                # Draw visualization
                if hasattr(exercise, "visualizer") and \
                   hasattr(exercise.visualizer, "draw_front_view_metrics"):
                    try:
                        frame = exercise.visualizer.draw_front_view_metrics(
                            frame, 
                            keypoints[:, :2], 
                            metrics, 
                            exercise.config.get("keypoints", {})
                        )
                        
                        # Draw Skeleton
                        if hasattr(exercise.visualizer, "draw_skeleton"):
                            frame = exercise.visualizer.draw_skeleton(
                                frame,
                                keypoints[:, :2],
                                exercise.config.get("keypoints", {})
                            )
                    except Exception as e:
                        logger.error(f"Error drawing front view metrics: {e}")
        
        # Write to front video writer
        if self.front_video_writer is None:
            h, w = frame.shape[:2]
            self._init_front_writer(w, h)
            
            
        if self.front_video_writer:
            self.front_video_writer.write(frame)
            
        if self.raw_front_video_writer:
            self.raw_front_video_writer.write(raw_frame)
        
        return frame, metadata

    def finalize_session(self):
        """Finalize the session: save metadata, merge audio/video, cleanup."""
        if not self.session_active:
            return
            
        self.session_active = False
        logger.info("Finalizing stream session...")
        
        try:
            # Release video writers
            if self.video_writer:
                self.video_writer.release()
            if self.raw_video_writer:
                self.raw_video_writer.release()
            if self.front_video_writer:
                self.front_video_writer.release()
            if self.raw_front_video_writer:
                self.raw_front_video_writer.release()
                
            # Get voice messages
            voice_messages = self.exercise_manager.get_voice_messages()
            
            # Record voice messages to input manager for later
            for timestamp, message_type in voice_messages:
                self.output_manager.record_voice_message(message_type, timestamp)
                
            # Save metadata
            duration = self.frame_count / self.fps
            timestamp = self.output_manager.timestamp
            
            # Ensure detailed analysis is finalized and saved with matching timestamp
            self.exercise_manager.finalize_analysis(
                self.config["paths"]["output_dir"], 
                timestamp=timestamp
            )

            metadata = {
                "exercise_type": self.exercise_type,
                "output_video": self.output_path,
                "duration": duration,
                "fps": self.fps,
                "total_frames": self.frame_count,
                "voice_messages": voice_messages,
                "results": self.exercise_manager.get_results(),
                "created_at": datetime.now().isoformat()
            }
            
            self.output_manager.save_synchronized_metadata(metadata)
            
            # Create audio track and merge
            if voice_messages:
                audio_path = self.output_manager.create_audio_track(duration)
                if audio_path:
                    final_path = self.output_path.replace(".mp4", "_final.mp4")
                    self.output_manager.merge_video_audio(self.output_path, audio_path, final_path)
            
            logger.success(f"Session finalized and saved successfully. Timestamp: {timestamp}")
            return timestamp
            
        except Exception as e:
            logger.error(f"Error finalizing session: {e}", exc_info=True)
            return None
