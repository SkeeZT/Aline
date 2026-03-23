"""
Exercise management utilities for AI Trainer application.

This module handles exercise-specific logic, including:
- Exercise instantiation
- Exercise state management
- Coordination between video processing and exercise analysis
"""
import numpy as np
from loguru import logger
from typing import Dict, Any, Optional

from engine.core.utils import start_countdown, side_visibility_detector


class ExerciseManager:
    """Manager for exercise-specific logic and coordination."""

    def __init__(
        self,
        config: Dict[str, Any],
        exercise_type: str,
        fps: float = 30.0,
        segmenter: Any = None,
    ):
        """
        Initialize the exercise manager.

        Args:
            config: Configuration dictionary
            exercise_type: Type of exercise to analyze
            fps: Video frames per second
            segmenter: Optional person segmenter
        """
        self.config = config
        self.exercise_type = exercise_type
        self.fps = fps
        self.segmenter = segmenter

        # Initialize exercise analyzer
        self.exercise = self.create_exercise(
            exercise_type, config, fps=fps, segmenter=segmenter
        )

        logger.info(f"Exercise manager initialized for {exercise_type} analysis")

    def create_exercise(
        self,
        exercise_type: str,
        config: Dict[str, Any],
        fps: float = 30.0,
        segmenter: Any = None,
    ):
        """
        Create an exercise analyzer of the specified type.

        Args:
            exercise_type: Type of exercise to create
            config: Exercise-specific configuration
            fps: Video frames per second
            segmenter: Optional person segmenter

        Returns:
            Exercise analyzer instance

        Raises:
            ValueError: If exercise type is not supported
        """
        if exercise_type == "squat":
            from engine.exercises.squat import SquatExercise

            return SquatExercise(config, fps=fps, segmenter=segmenter)
        elif exercise_type == "pullup":
            from engine.exercises.pullup import PullupExercise

            return PullupExercise(config, fps=fps, segmenter=segmenter)
        elif exercise_type == "pushup":
            from engine.exercises.pushup import PushupExercise

            return PushupExercise(config, fps=fps, segmenter=segmenter)
        elif exercise_type == "dips":
            from engine.exercises.dips import DipsExercise

            return DipsExercise(config, fps=fps, segmenter=segmenter)
        elif exercise_type == "lunges":
            from engine.exercises.lunges import LungesExercise

            return LungesExercise(config, fps=fps, segmenter=segmenter)
        elif exercise_type == "plank":
            from engine.exercises.plank import PlankExercise

            return PlankExercise(config, fps=fps, segmenter=segmenter)
        elif exercise_type == "deadlift":
            from engine.exercises.deadlift import DeadliftExercise

            return DeadliftExercise(config, fps=fps, segmenter=segmenter)
        elif exercise_type == "overhead_press":
            from engine.exercises.overhead_press import OverheadPressExercise

            return OverheadPressExercise(config, fps=fps, segmenter=segmenter)
        elif exercise_type == "bent_over_row":
            from engine.exercises.bent_over_row import BentOverRowExercise

            return BentOverRowExercise(config, fps=fps, segmenter=segmenter)
        elif exercise_type == "glute_bridge":
            from engine.exercises.glute_bridge import GluteBridgeExercise

            return GluteBridgeExercise(config, fps=fps, segmenter=segmenter)
        elif exercise_type == "wall_sit":
            from engine.exercises.wall_sit import WallSitExercise

            return WallSitExercise(config, fps=fps, segmenter=segmenter)
        elif exercise_type == "bench_press":
            from engine.exercises.bench_press import BenchPressExercise

            return BenchPressExercise(config, fps=fps, segmenter=segmenter)
        else:
            raise ValueError(
                f"Exercise type '{exercise_type}' not supported. Available exercises: 'squat', 'pullup', 'pushup', 'dips', 'lunges', 'plank', 'deadlift', 'overhead_press', 'bent_over_row', 'glute_bridge', 'wall_sit', 'bench_press'."
            )

    def perform_countdown(self, cap, show_viz: bool = True, frame_processor=None, on_annotated_frame=None, pose_estimator=None):
        """
        Perform countdown before starting exercise analysis.

        Args:
            cap: Video capture object
            show_viz: Whether to show visualization
            frame_processor: Optional frame processor function
            on_annotated_frame: Optional callback for annotated countdown frames
            pose_estimator: Optional pose estimator for side detection during countdown
        """
        countdown_duration = self.config.get("countdown", {}).get("time", 5)
        
        # Pass pose_estimator to start_countdown to enable side voting
        determined_side = start_countdown(
            cap,
            fps=self.fps,
            show_viz=show_viz,
            frame_processor=frame_processor,
            countdown_duration=countdown_duration,
            on_annotated_frame=on_annotated_frame,
            pose_estimator=pose_estimator
        )
        
        # If side was determined during countdown, set it explicitly on the exercise
        if determined_side and hasattr(self.exercise, "set_facing_side"):
            logger.info(f"Countdown detection complete. Setting facing side to: {determined_side}")
            self.exercise.set_facing_side(determined_side)

    def wait_for_optimal_positioning(
        self, 
        cap, 
        show_viz: bool = True, 
        frame_processor=None, 
        pose_estimator=None, 
        on_annotated_frame=None
    ):
        """
        Wait for optimal side positioning before exercise analysis.

        Args:
            cap: Video capture object
            show_viz: Whether to show visualization
            frame_processor: Optional frame processor function
            model: YOLO model for pose detection
            on_annotated_frame: Optional callback for annotated positioning frames

        Returns:
            bool: True if optimal positioning achieved, False otherwise
        """
        confidence_threshold = self.config.get("side_difference", {}).get(
            "confidence_threshold", 0.25
        )

        # Enable optional timeout to fall back to front-view mode if the user can't turn sideways
        timeout_seconds = self.config.get("side_difference", {}).get(
            "timeout_seconds", None
        )

        success = side_visibility_detector(
            cap,
            fps=self.fps,
            confidence_threshold=confidence_threshold,
            show_viz=show_viz,
            frame_processor=frame_processor,
            pose_estimator=pose_estimator,
            timeout_seconds=timeout_seconds,
            on_annotated_frame=on_annotated_frame,
        )

        if not success and timeout_seconds is not None:
            # Proceed in front-view mode
            logger.warning(
                "Proceeding in front-view mode due to side positioning timeout"
            )
            # Inform the underlying exercise analyzer
            if hasattr(self.exercise, "set_front_view_mode"):
                self.exercise.set_front_view_mode(True)
            return True

        return success

    def process_frame(
        self, 
        frame: np.ndarray, 
        frame_number: int,
        keypoints: Optional[np.ndarray] = None,
        results: Any = None
    ):
        """
        Process a single frame with the exercise analyzer.

        Args:
            frame: Video frame to process
            frame_number: Current frame number
            keypoints: Standardized keypoints [x, y, conf]
            results: Original inference results (optional)

        Returns:
            Processed frame with visualizations
        """
        return self.exercise.process_frame(
            frame=frame, 
            frame_number=frame_number, 
            keypoints=keypoints, 
            results=results
        )

    def finalize_analysis(self, output_dir: str, timestamp: str = None):
        """
        Finalize exercise analysis and save results.

        Args:
            output_dir: Directory to save results
            timestamp: Optional timestamp for output files
        """
        self.exercise.finalize_analysis(output_dir, timestamp)

    def get_results(self) -> Dict[str, Any]:
        """
        Get current exercise analysis results.

        Returns:
            Dictionary with current results
        """
        return self.exercise.get_results()

    def get_voice_messages(self) -> list:
        """
        Get recorded voice messages with timestamps.

        Returns:
            List of (timestamp, message_type) tuples
        """
        return self.exercise.get_voice_messages()

    def get_anthropometrics(self) -> Optional[Dict[str, Any]]:
        """
        Get computed anthropometrics if available.

        Returns:
            Dictionary with anthropometric data or None
        """
        return self.exercise.get_anthropometrics()

    @property
    def exercise_name(self) -> str:
        """
        Get the exercise name.

        Returns:
            Name of the exercise
        """
        return self.exercise.get_exercise_name()
