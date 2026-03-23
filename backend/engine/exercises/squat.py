"""
Squat exercise analyzer with complete logic implementation.
Includes side detection, rep counting, form analysis, VBT calculations, and voice feedback.
Supports dual-camera mode for comprehensive form analysis (knee valgus, hip alignment).
"""

import os
import json
import math
import cv2
import numpy as np
from datetime import datetime
from loguru import logger
from typing import Dict, Any, Optional, Tuple

from engine.voice_message_player import VoiceMessagePlayer
from engine.core.side_detection import determine_facing_side
from engine.core.visualization import SquatVisualizer
from engine.velocity_calculator import VelocityCalculator
from engine.core.utils import compute_anthropometrics, compute_back_curvature
from engine.exercises.dual_camera_mixin import DualCameraExerciseMixin, DualCameraMetrics


class SquatExercise(DualCameraExerciseMixin):
    """Complete squat exercise analyzer with integrated logic and voice feedback."""

    def __init__(
        self, config: Dict[str, Any], fps: float = 30.0, segmenter: Any = None
    ):
        """
        Initialize the complete squat analyzer.

        Args:
            config: Configuration dictionary
            fps: Video frames per second
        """
        self.config = config
        self.fps = fps
        self.cooldown_frames = int(fps * 0.5)  # Increased to prevent false rep counts
        self.state_change_cooldown = int(fps * 0.3)  # Increased for stability

        self.last_rep_frame = -self.cooldown_frames
        self.last_state_change_frame = 0
        self.min_knee_angle_this_rep = float('inf')
        self.min_thigh_angle_this_rep = float('inf')
        self.min_hip_angle_this_rep = float('inf')
        self.max_hip_angle_this_rep = 0.0

        # ROM tracking variables
        self.knee_max_baseline = None
        self.knee_min_baseline = None
        self.rom_established = False
        self.knee_range = None

        # Rep counting
        self.successful_reps = 0
        self.unsuccessful_reps = 0
        self.total_reps = 0
        self.state = "up"  # "up", "going_down", "coming_up"
        self.min_depth_reached = False
        self.max_rom_this_rep = 0.0
        self.bottom_position_frame = 0
        
        # State tracking for failure justifications
        self.last_failure_justification = []
        self.failure_justification_timer = 0 # in frames
        self.failure_display_duration = int(fps * 3.0) # Show for 3 seconds
        self.rep_start_frame = 0
        self._prev_knee_angle = 0
        self._knee_velocity = 0
        self.knee_past_toes = False
        self.knee_past_toes_violation = False

        # Form detection flags
        self.forward_lean_detected = False
        self.hyperextension_detected = False

        # Summary logging
        self.reps_summary = []

        # Side detection
        self.facing_side: Optional[str] = None
        self.side_determined = False

        # VBT calculator
        self.vbt_calculator = VelocityCalculator(fps=fps)

        # Visualizer
        self.visualizer = SquatVisualizer(config)

        # Voice message player
        self.voice_player = VoiceMessagePlayer(config)

        # Voice message recording for audio merging
        self.voice_messages = []  # List of (timestamp, message_type) tuples

        # Workout state tracking
        self.workout_started = False
        self.workout_ended = False

        # Voice feedback tracking
        self.last_voice_feedback_frame = -self.cooldown_frames
        self.voice_feedback_cooldown_frames = int(
            fps * 3
        )  # 3 seconds between voice messages
        self.last_feedback_times = {} # Track last time each specific feedback was played

        logger.info("Squat exercise analyzer initialized")

        self.anthropometrics = None
        self.segmenter = segmenter

        # Front-view mode and metrics
        self.front_view_mode = False
        self.front_view_metrics = None

        # Movement boundary detection
        self.movement_boundaries = None
        self.initial_position = None
        self.thigh_length = None
        self.shin_length = None

        # Experience level thresholds (from config with defaults)
        self.experience_level = (self.config.get("experience", {}) or {}).get(
            "level", "intermediate"
        )
        self.thresholds = self._build_thresholds(self.experience_level)

        # Dual camera support
        DualCameraExerciseMixin.__init__(self)
        self.dual_camera_enabled = False
        self.front_view_keypoints = None
        self.last_front_analysis = None
        
        # Dual camera form issues (detected from front view)
        self.knee_valgus_detected = False
        self.hip_drop_detected = False

    def determine_side_from_keypoints(self, keypoints_data: np.ndarray, allow_override: bool = False) -> None:
        """
        Determine which side is facing the camera using keypoint analysis.

        Args:
            keypoints_data: YOLO keypoints data for one person
            allow_override: Whether to override an already determined side
        """
        # If side is already determined and we don't allow override, do nothing
        if self.side_determined and not allow_override:
            return

        if keypoints_data is not None:
            new_side = determine_facing_side(keypoints_data)
            self.facing_side = new_side
            self.side_determined = True
            logger.info(f"Determined facing side: {self.facing_side}")

    def set_facing_side(self, side: str) -> None:
        """
        Explicitly set the facing side (e.g., from countdown detection).
        This locks the side and prevents auto-detection from changing it.
        """
        if side and side.lower() in ["left", "right"]:
            self.facing_side = side.lower()
            self.side_determined = True
            logger.info(f"Facing side explicitly set to: {self.facing_side}")

    def set_front_view_mode(self, enabled: bool) -> None:
        """
        Enable or disable front-view mode.

        In front-view mode, we compute additional metrics such as hip alignment
        and knees distance, and we do not require left/right facing side.
        """
        self.front_view_mode = bool(enabled)
        if self.front_view_mode:
            logger.warning(
                "Front-view mode enabled: computing hip alignment and knees distance"
            )

    def set_experience_level(self, level: str) -> None:
        """Set user experience level and update thresholds."""
        level = (level or "intermediate").lower()
        if level not in ("beginner", "intermediate", "advanced"):
            level = "intermediate"
        self.experience_level = level
        self.thresholds = self._build_thresholds(level)
        logger.info(f"Experience level set to {level}")

    def reset_rom_baselines(self) -> None:
        """Reset ROM baselines to allow recalibration."""
        self.knee_max_baseline = None
        self.knee_min_baseline = None
        self.rom_established = False
        self.knee_range = None
        logger.info("ROM baselines reset for recalibration")

    def _build_thresholds(self, level: str) -> Dict[str, float]:
        """
        Build thresholds primarily from exercise-specific configuration.

        Returns dict: knee_min, knee_max, hip_min, hip_max, rep thresholds...
        """
        # First check for squat-specific config, then fall back to experience config
        squat_config = self.config.get("squat", {}) or {}
        squat_thresholds = squat_config.get("thresholds", {})
        
        # Fallback to legacy experience config if squat-specific not found
        if not squat_thresholds:
            experience_config = self.config.get("experience", {}) or {}
            squat_thresholds = experience_config.get("thresholds", {})
            min_rep_duration = float(experience_config.get("min_rep_duration", 1.0))
        else:
            min_rep_duration = float(squat_config.get("min_rep_duration", 0.2))
        
        level = (level or "intermediate").lower()
        level_cfg = squat_thresholds.get(level, {})

        # 1) Metrics with safe defaults
        knee_min = float(level_cfg.get("knee_min", 80.0))
        knee_max = float(level_cfg.get("knee_max", 175.0))
        hip_min = float(level_cfg.get("hip_min", 145.0))
        hip_max = float(level_cfg.get("hip_max", 195.0))

        # 2) Rep thresholds (now per-level)
        rep_attempt_threshold = float(level_cfg.get("attempt_threshold", 0.05))
        rep_success_threshold = float(level_cfg.get("success_threshold", 0.25))
        very_shallow_threshold = float(level_cfg.get("very_shallow_threshold", 0.10))
        moderate_shallow_threshold = float(level_cfg.get("moderate_shallow_threshold", 0.20))
        knees_over_toes_tolerance = float(level_cfg.get("knees_over_toes_tolerance", 50.0))
        
        # Thigh angle thresholds for depth
        # 0 degrees is horizontal. Positive means knee below hip (Deep).
        thigh_depth_threshold = -10.0 

        return {
            "knee_min": knee_min,
            "knee_max": knee_max,
            "hip_min": hip_min,
            "hip_max": hip_max,
            "thigh_depth_threshold": thigh_depth_threshold,
            "rep_attempt_threshold": rep_attempt_threshold,
            "rep_success_threshold": rep_success_threshold,
            "very_shallow_threshold": very_shallow_threshold,
            "moderate_shallow_threshold": moderate_shallow_threshold,
            "knees_over_toes_tolerance": knees_over_toes_tolerance,
            "min_rep_duration": min_rep_duration,
            "relative_angle_difference": float(level_cfg.get("relative_angle_difference", 30.0)),
            "back_curvature_threshold": float(level_cfg.get("back_curvature_threshold", 1.5)),
        }

    def calculate_body_dimensions(self, keypoints: np.ndarray) -> Dict[str, float]:
        """
        Calculate body dimensions based on keypoints for movement boundary detection.

        Args:
            keypoints: Standardized keypoints array [x, y, confidence]

        Returns:
            Dictionary with body measurements
        """
        try:
            kpt = self.config["keypoints"]

            # Get required keypoint indices
            left_hip_idx = kpt["left_hip"]
            right_hip_idx = kpt["right_hip"]
            left_knee_idx = kpt["left_knee"]
            right_knee_idx = kpt["right_knee"]
            left_ankle_idx = kpt["left_ankle"]
            right_ankle_idx = kpt["right_ankle"]

            # Check if keypoints have sufficient confidence
            required_indices = [left_hip_idx, right_hip_idx, left_knee_idx,
                              right_knee_idx, left_ankle_idx, right_ankle_idx]

            for idx in required_indices:
                if keypoints[idx, 2] < 0.5:  # confidence threshold
                    return {"error": "Insufficient confidence in keypoint detection"}

            # Calculate leg length (average of left and right leg lengths)
            # Leg = thigh + shin = hip to knee + knee to ankle
            left_hip = keypoints[left_hip_idx, :2]
            right_hip = keypoints[right_hip_idx, :2]
            left_knee = keypoints[left_knee_idx, :2]
            right_knee = keypoints[right_knee_idx, :2]
            left_ankle = keypoints[left_ankle_idx, :2]
            right_ankle = keypoints[right_ankle_idx, :2]

            left_thigh = float(np.linalg.norm(left_hip - left_knee))
            left_shin = float(np.linalg.norm(left_knee - left_ankle))

            right_thigh = float(np.linalg.norm(right_hip - right_knee))
            right_shin = float(np.linalg.norm(right_knee - right_ankle))

            avg_thigh_length = (left_thigh + right_thigh) / 2.0
            avg_shin_length = (left_shin + right_shin) / 2.0

            return {
                "thigh_length": avg_thigh_length,
                "shin_length": avg_shin_length,
            }
        except Exception as e:
            logger.warning(f"Error calculating body dimensions: {e}")
            return {"error": f"Error calculating body dimensions: {str(e)}"}

    def set_initial_position(self, keypoints: np.ndarray) -> bool:
        """
        Set the initial position for movement boundary detection.

        Args:
            keypoints: Standardized keypoints array [x, y, confidence]

        Returns:
            True if initial position was successfully set, False otherwise
        """
        try:
            # Calculate body dimensions
            dimensions = self.calculate_body_dimensions(keypoints)
            if "error" in dimensions:
                return False

            # Get hip center as the reference point
            kpt = self.config["keypoints"]
            left_hip_idx = kpt["left_hip"]
            right_hip_idx = kpt["right_hip"]

            left_hip = keypoints[left_hip_idx, :2]
            right_hip = keypoints[right_hip_idx, :2]

            # Calculate hip center
            hip_center_x = (left_hip[0] + right_hip[0]) / 2.0
            hip_center_y = (left_hip[1] + right_hip[1]) / 2.0
            
            # Use current hip center as initial position
            self.initial_position = np.array([hip_center_x, hip_center_y])
            self.thigh_length = dimensions["thigh_length"]
            self.shin_length = dimensions["shin_length"]
            
            # New specific lengths
            thigh_length = dimensions["thigh_length"]
            shin_length = dimensions["shin_length"]

            # Define user facing direction
            # Default to "right" if not determined (facing right -> forward is +X, backward is -X)
            # If facing left -> forward is -X, backward is +X
            facing_right = (self.facing_side != "left") 

            # Define limits from initial position
            # Backward limit = thigh_length
            # Forward limit = shin_length
            
            if facing_right:
                # User >>>> Camera view
                # Forward is +X, Backward is -X
                x_min = self.initial_position[0] - thigh_length # Backward limit
                x_max = self.initial_position[0] + shin_length  # Forward limit
            else:
                # <<<< User Camera view
                # Forward is -X, Backward is +X
                x_min = self.initial_position[0] - shin_length  # Forward limit (left)
                x_max = self.initial_position[0] + thigh_length # Backward limit (right)

            # Define Y limits
            # Top (y_min): 0 (top of image)
            y_min = 0
            # Bottom (y_max): Very large number (effectively infinity to cover bottom)
            y_max = 10000 

            self.movement_boundaries = {
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max)
            }

            logger.info(f"Initial position set: {self.initial_position}, Side: {self.facing_side or 'right'}")
            logger.info(f"Movement boundaries: {self.movement_boundaries}")
            logger.info(f"Body dimensions - Thigh: {thigh_length:.1f}, Shin: {shin_length:.1f}")

            return True
        except Exception as e:
            logger.warning(f"Error setting initial position: {e}")
            return False

    def check_movement_boundary(self, keypoints: np.ndarray) -> Dict[str, any]:
        """
        Check if the current position exceeds movement boundaries.

        Args:
            keypoints: Standardized keypoints array [x, y, confidence]

        Returns:
            Dictionary with boundary status information
        """
        if self.initial_position is None or self.movement_boundaries is None:
            # If initial position not set, try to set it
            success = self.set_initial_position(keypoints)
            if not success:
                return {
                    "within_bounds": False,
                    "boundary_status": {},
                    "current_position": None,
                    "boundaries": None
                }

        try:
            # Get current hip center position
            kpt = self.config["keypoints"]
            left_hip_idx = kpt["left_hip"]
            right_hip_idx = kpt["right_hip"]

            # Check confidence
            if keypoints[left_hip_idx, 2] < 0.5 or keypoints[right_hip_idx, 2] < 0.5:
                return {
                    "within_bounds": False,
                    "boundary_status": {},
                    "current_position": None,
                    "boundaries": self.movement_boundaries
                }

            left_hip = keypoints[left_hip_idx, :2]
            right_hip = keypoints[right_hip_idx, :2]

            # Calculate current hip center
            current_hip_center_x = (left_hip[0] + right_hip[0]) / 2.0
            current_hip_center_y = (left_hip[1] + right_hip[1]) / 2.0
            current_position = np.array([current_hip_center_x, current_hip_center_y])

            # Check if within boundaries (only X allowed)
            x_val = current_position[0]
            
            x_min = self.movement_boundaries["x_min"]
            x_max = self.movement_boundaries["x_max"]

            x_within = x_min <= x_val <= x_max
            # Y check is ignored as requested
            
            within_bounds = x_within

            boundary_status = {
                "within_bounds": x_within,
                "x_forward_exceeded": False,
                "x_backward_exceeded": False,
                "x_within": x_within,
                "out_of_bounds_left": x_val < x_min,
                "out_of_bounds_right": x_val > x_max,
            }

            return {
                "within_bounds": within_bounds,
                "boundary_status": boundary_status,
                "current_position": current_position.tolist(),
                "initial_position": self.initial_position.tolist() if self.initial_position is not None else None,
                "boundaries": self.movement_boundaries,
                "body_dimensions": {
                    "thigh_length": self.thigh_length,
                    "shin_length": self.shin_length
                }
            }
        except Exception as e:
            logger.warning(f"Error checking movement boundary: {e}")
            return {
                "within_bounds": False,
                "boundary_status": {},
                "current_position": None,
                "boundaries": self.movement_boundaries,
                "body_dimensions": {
                    "thigh_length": self.thigh_length,
                    "shin_length": self.shin_length
                },
                "error": str(e)
            }



    def get_side_specific_keypoints(self) -> Dict[str, int]:
        """
        Get keypoint indices based on the determined facing side.

        Returns:
            Dictionary with keypoint indices for the detected side
        """
        if self.facing_side == "left":
            # If the user is facing LEFT, their RIGHT side is visible to the camera
            return {
                "shoulder": self.config["keypoints"]["right_shoulder"],
                "hip": self.config["keypoints"]["right_hip"],
                "knee": self.config["keypoints"]["right_knee"],
                "ankle": self.config["keypoints"]["right_ankle"],
            }
        else:  # Default to right side
            # If the user is facing RIGHT, their LEFT side is visible to the camera
            return {
                "shoulder": self.config["keypoints"]["left_shoulder"],
                "hip": self.config["keypoints"]["left_hip"],
                "knee": self.config["keypoints"]["left_knee"],
                "ankle": self.config["keypoints"]["left_ankle"],
            }

    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle between three points.

        Args:
            p1, p2, p3: Points as [x, y] coordinates

        Returns:
            Angle in degrees
        """
        # Convert to numpy arrays
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2

        # Calculate angle
        dot_prod = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norm_product == 0:
            return 0.0
            
        cos_angle = dot_prod / norm_product
        cos_angle = np.clip(cos_angle, -1, 1)  # Avoid numerical errors
        angle = math.degrees(math.acos(cos_angle))

        return angle

    def calculate_thigh_angle(self, hip: np.ndarray, knee: np.ndarray, hip_world=None, knee_world=None) -> float:
        # Use 3D World Landmarks for accurate depth (gravity-aligned) if available
        if hip_world is not None and knee_world is not None:
            # MediaPipe World Landmarks: Y is height (up/down? Check model spec)
            # Usually +Y is down. We want angle with horizontal plane (XZ).
            # Vector = knee - hip.
            # We want angle of this vector vs Horizontal Plane.
            # Or simpler: Angle between Thigh Vector and Vertical (Y-axis).
            # Then Thigh Angle (vs Horizontal) = 90 - Angle(Thigh, Vertical).
            
            dy = knee_world.y - hip_world.y
            dist = math.sqrt((knee_world.x - hip_world.x)**2 + (knee_world.z - hip_world.z)**2 + dy**2)
            if dist == 0: return 90.0
            
            # Angle with vertical (Y-axis)
            # cos(theta) = dy / dist
            angle_vertical = math.degrees(math.acos(dy / dist))
            
            # If dy > 0 (knee below hip), angle < 90.
            # We want: 0 = parallel. Positive = deep.
            # Angle relative to horizontal:
            from_horizontal = 90 - angle_vertical
            return from_horizontal

        # Fallback 2D
        dy = knee[1] - hip[1]
        dx = abs(knee[0] - hip[0])
        if dx == 0: return 90.0
        return math.degrees(math.atan2(dy, dx))

    def _establish_rom_baselines(self, knee_angle: float, hip_angle: float, frame_number: int) -> bool:
        """
        Establish ROM baselines for proper rep counting.

        Args:
            knee_angle: Current knee angle
            hip_angle: Current hip angle
            frame_number: Current frame number

        Returns:
            bool: True if ROM baselines are established
        """
        # If we haven't established baselines yet, use the first valid standing position
        if self.knee_max_baseline is None and knee_angle > 160:  # Standing position
            self.knee_max_baseline = knee_angle
            logger.info(f"Established knee max baseline: {knee_angle:.2f}")

        # Establish or update knee min baseline if we see a deeper position
        knee_min_threshold = self.thresholds.get("knee_min", 90.0)
        if self.knee_max_baseline is not None and knee_angle < knee_min_threshold:
            # If deeper than current baseline (or first time hitting threshold)
            if self.knee_min_baseline is None or knee_angle < self.knee_min_baseline:
                self.knee_min_baseline = knee_angle
                self.knee_range = self.knee_max_baseline - self.knee_min_baseline
                self.rom_established = True
                logger.info(f"Updated knee min baseline: {knee_angle:.2f}, ROM range: {self.knee_range:.2f}")

        # If we still don't have min baseline after some frames, estimate it
        if (self.knee_max_baseline is not None and
            self.knee_min_baseline is None and
            frame_number > self.fps * 5):  # After 5 seconds of processing
            # Use the minimum knee angle seen so far as the baseline
            if knee_angle < self.knee_max_baseline - 20:  # Reasonable deep position
                self.knee_min_baseline = knee_angle
                self.knee_range = self.knee_max_baseline - self.knee_min_baseline
                self.rom_established = True
                logger.info(f"Estimated knee min baseline: {knee_angle:.2f}, ROM range: {self.knee_range:.2f}")

        return self.rom_established

    def _calculate_rom_percentage(self, current_knee_angle: float) -> float:
        """
        Calculate the percentage of ROM achieved based on current knee angle.

        Args:
            current_knee_angle: Current knee angle

        Returns:
            float: Percentage of ROM achieved (0.0 to 1.0)
        """
        # Validate input
        if not isinstance(current_knee_angle, (int, float)) or np.isnan(current_knee_angle):
            logger.warning(f"Invalid knee angle received: {current_knee_angle}")
            return 0.0

        # Determine working ranges for percentage calculation
        working_max = self.knee_max_baseline if self.knee_max_baseline is not None else 175.0
        working_min = self.knee_min_baseline
        working_range = self.knee_range

        # If baseline not fully established, use threshold as a temporary baseline
        # This allows the ROM to "approach" 1.0 instead of jumping from 0 to 1
        if not self.rom_established or working_min is None:
            working_min = self.thresholds.get("knee_min", 90.0)
            working_range = working_max - working_min

        if working_range <= 0:
            return 0.0

        rom_percentage = (working_max - current_knee_angle) / working_range
        return max(0.0, min(1.0, rom_percentage))  # Clamp between 0 and 1

    def extract_keypoints(self, results) -> Optional[np.ndarray]:
        """
        Extract keypoints from YOLO results.

        Args:
            results: YOLO inference results

        Returns:
            Keypoints array or None if not detected with sufficient confidence
        """
        if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            return None

        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        confidences = results[0].keypoints.conf[0].cpu().numpy()

        # Get side-specific keypoints (not used for front-view metrics calculation)
        kpt_indices = self.get_side_specific_keypoints()

        # Check if required keypoints are detected with good confidence
        required_indices = [
            kpt_indices["shoulder"],
            kpt_indices["hip"],
            kpt_indices["knee"],
            kpt_indices["ankle"],
        ]

        for idx in required_indices:
            if confidences[idx] < 0.5:  # Confidence threshold
                return None

        return keypoints

    def _compute_front_view_metrics_from_keypoints(self, keypoints: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Compute hip alignment and knees distance for front-view mode using standardized keypoints.
        """
        try:
            if keypoints is None:
                return None

            kpt = self.config["keypoints"]
            li = kpt.get("left_hip")
            ri = kpt.get("right_hip")
            lk = kpt.get("left_knee")
            rk = kpt.get("right_knee")

            # Check confidence
            needed = [li, ri, lk, rk]
            # keypoints is (17, 3) -> [x, y, conf]
            for idx in needed:
                if keypoints[idx, 2] < 0.35:
                    return None

            left_hip = keypoints[li, :2]
            right_hip = keypoints[ri, :2]
            left_knee = keypoints[lk, :2]
            right_knee = keypoints[rk, :2]

            # Hip alignment angle relative to horizontal
            dy = float(left_hip[1] - right_hip[1])
            dx = float(left_hip[0] - right_hip[0])
            if dx == 0:
                hip_alignment_angle = 90.0
            else:
                hip_alignment_angle = abs(math.degrees(math.atan2(dy, dx)))
                # fold to [0,90]
                if hip_alignment_angle > 90:
                    hip_alignment_angle = 180.0 - hip_alignment_angle

            hip_vertical_misalignment = abs(dy)

            # Knees distance
            knees_distance_px = float(np.linalg.norm(left_knee - right_knee))

            # Normalize by hip width when available
            hip_width = float(np.linalg.norm(left_hip - right_hip))
            knees_distance_norm = (
                knees_distance_px / hip_width if hip_width > 1.0 else None
            )

            return {
                "hip_alignment_angle": hip_alignment_angle,
                "hip_vertical_misalignment": hip_vertical_misalignment,
                "knees_distance_px": knees_distance_px,
                "knees_distance_norm": knees_distance_norm,
            }
        except Exception as e:
            logger.debug(f"Front-view metrics error: {e}")
            return None

    def _compute_front_view_metrics(self, results) -> Optional[Dict[str, Any]]:
        """
        Compute hip alignment and knees distance for front-view mode.

        Returns a dict with:
            - hip_alignment_angle: angle (deg) between the line connecting hips and horizontal
            - hip_vertical_misalignment: |y_left_hip - y_right_hip| in pixels
            - knees_distance_px: Euclidean distance between knees in pixels
            - knees_distance_norm: knees distance normalized by hip width (if available)
        """
        try:
            if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
                return None

            keypoints_xy = results[0].keypoints.xy[0].cpu().numpy()
            confidences = results[0].keypoints.conf[0].cpu().numpy()
            kpt = self.config["keypoints"]

            li = kpt.get("left_hip")
            ri = kpt.get("right_hip")
            lk = kpt.get("left_knee")
            rk = kpt.get("right_knee")

            # Validate presence and minimum confidence
            needed = [li, ri, lk, rk]
            if any(i is None for i in needed):
                return None
            if any(confidences[i] < 0.35 for i in needed):
                return None

            left_hip = keypoints_xy[li]
            right_hip = keypoints_xy[ri]
            left_knee = keypoints_xy[lk]
            right_knee = keypoints_xy[rk]

            # Hip alignment angle relative to horizontal
            dy = float(left_hip[1] - right_hip[1])
            dx = float(left_hip[0] - right_hip[0])
            if dx == 0:
                hip_alignment_angle = 90.0
            else:
                hip_alignment_angle = abs(math.degrees(math.atan2(dy, dx)))
                # fold to [0,90]
                if hip_alignment_angle > 90:
                    hip_alignment_angle = 180.0 - hip_alignment_angle

            hip_vertical_misalignment = abs(dy)

            # Knees distance
            knees_distance_px = float(np.linalg.norm(left_knee[:2] - right_knee[:2]))

            # Normalize by hip width when available
            hip_width = float(np.linalg.norm(left_hip[:2] - right_hip[:2]))
            knees_distance_norm = (
                knees_distance_px / hip_width if hip_width > 1.0 else None
            )

            return {
                "hip_alignment_angle": hip_alignment_angle,
                "hip_vertical_misalignment": hip_vertical_misalignment,
                "knees_distance_px": knees_distance_px,
                "knees_distance_norm": knees_distance_norm,
            }
        except Exception as e:
            logger.debug(f"Front-view metrics error: {e}")
            return None

    def _is_knee_past_toes(self, knee: np.ndarray, ankle: np.ndarray, foot_index: np.ndarray = None, heel: np.ndarray = None) -> bool:
        """
        Determine if the knee has passed the toe position.
        Uses MediaPipe landmarks if available, otherwise estimates from ankle.
        Prioritizes foot_index (toe) when available as it's more accurate than heel.
        """
        try:
            forward_dir = -1 if self.facing_side == "left" else 1
            knee_x = float(knee[0])

            # Add tolerance buffer (pixels) to make it harder to trigger
            tolerance = float(self.thresholds.get("knees_over_toes_tolerance", 50.0)) 

            # Prioritize foot_index (toe) as it represents the toe position more accurately
            if foot_index is not None:
                toe_x = float(foot_index[0])
                # Check if knee X passed toe X in forward direction by more than tolerance
                if forward_dir > 0:
                     return knee_x > toe_x + tolerance
                else:
                     return knee_x < toe_x - tolerance
            elif heel is not None:
                # If foot index not available, use heel but adjust for toe position
                # The toe is typically about 10-15% of foot length in front of the heel
                heel_x = float(heel[0])
                # Estimate toe position from heel (toe is forward of heel)
                estimated_toe_x = heel_x + (15 if forward_dir > 0 else -15)  # 15 pixels adjustment
                if forward_dir > 0:
                    return knee_x > estimated_toe_x + tolerance
                else:
                    return knee_x < estimated_toe_x - tolerance

            # Legacy fallback: Estimate based on shank length
            shank_len = float(np.linalg.norm(knee - ankle))
            toes_x_est = float(ankle[0]) + forward_dir * (0.62 * shank_len)

            if forward_dir > 0:
                return knee_x > toes_x_est + tolerance
            else:
                 return knee_x < toes_x_est - tolerance

        except Exception:
            return False

    def update_rep_count(
        self,
        knee_angle: float,
        hip_angle: float,
        thigh_angle: float,
        back_curvature: float,
        frame_number: int
    ) -> None:
        """
        Update rep count based on ROM (Range of Motion) calculation.
        """
        # Establish ROM baselines if not already done
        self._establish_rom_baselines(knee_angle, hip_angle, frame_number)

        # Calculate current ROM percentage
        self.current_rom_percentage = self._calculate_rom_percentage(knee_angle)

        # Track maximum ROM achieved during this rep
        if self.state in ["going_down", "coming_up"]:
            self.max_rom_this_rep = max(self.max_rom_this_rep, self.current_rom_percentage)

        # Calculate angle velocity (change per frame)
        if not hasattr(self, "_prev_knee_angle"):
            self._prev_knee_angle = knee_angle
            self._knee_velocity = 0
        else:
            self._knee_velocity = knee_angle - self._prev_knee_angle
            self._prev_knee_angle = knee_angle

        # Check for timeout / reset
        if self._check_timeout(frame_number):
            return

        # State Machine Dispatch
        if self.state == "up":
            self._handle_state_up(knee_angle, frame_number)
        elif self.state == "going_down":
            self._handle_state_descent(knee_angle, hip_angle, thigh_angle, frame_number)
        elif self.state == "coming_up":
            self._handle_state_ascent(knee_angle, hip_angle, thigh_angle, back_curvature, frame_number)

    def _check_timeout(self, frame_number: int) -> bool:
        """Reset state if stuck for too long."""
        if self.state != "up" and (frame_number - self.last_state_change_frame > self.fps * 4):
            self.state = "up"
            self.forward_lean_detected = False
            self.hyperextension_detected = False
            self.min_depth_reached = False
            logger.debug("State reset due to timeout")
            return True
        return False

    def _handle_state_up(self, knee_angle: float, frame_number: int) -> None:
        """Monitor for the start of a squat."""
        state_change_allowed = (
            frame_number - self.last_state_change_frame > self.state_change_cooldown
        )

        significant_knee_bend = False
        if self.knee_max_baseline is not None:
             significant_knee_bend = knee_angle < (self.knee_max_baseline - 5.0)
        else:
             significant_knee_bend = knee_angle < 170.0

        if (
            self._knee_velocity < -1.0  # Velocity check
            and significant_knee_bend       # Angle change check
            and state_change_allowed
        ):
            self.state = "going_down"
            # Reset rep tracking variables
            self.min_depth_reached = False
            self.bottom_position_frame = 0
            self.min_knee_angle_this_rep = 180
            self.min_thigh_angle_this_rep = float('inf')
            self.min_hip_angle_this_rep = 180
            self.max_hip_angle_this_rep = 0
            self.last_state_change_frame = frame_number
            self.max_rom_this_rep = 0.0
            self.rep_start_frame = frame_number
            self.forward_lean_detected = False
            self.hyperextension_detected = False
            logger.debug(f"State: GOING_DOWN | Frame: {frame_number} | ROM: {self.current_rom_percentage:.2f}")

    def _handle_state_descent(self, knee_angle: float, hip_angle: float, thigh_angle: float, frame_number: int) -> None:
        """Monitor descent phase."""
        self.min_knee_angle_this_rep = min(self.min_knee_angle_this_rep, knee_angle)
        self.min_thigh_angle_this_rep = min(self.min_thigh_angle_this_rep, thigh_angle)
        self.min_hip_angle_this_rep = min(self.min_hip_angle_this_rep, hip_angle)
        self.max_hip_angle_this_rep = max(self.max_hip_angle_this_rep, hip_angle)

        # Forward Lean Check
        relative_diff_threshold = self.thresholds.get("relative_angle_difference", 30.0)
        if (knee_angle - hip_angle > relative_diff_threshold):
                self.forward_lean_detected = True
                logger.debug(f"Forward Lean (Descent): Knee={knee_angle:.1f}, Hip={hip_angle:.1f}, Diff={knee_angle - hip_angle:.1f}")

        # Depth Check for Attempt
        rep_attempt_threshold = self.thresholds.get("rep_attempt_threshold", 0.10)
        is_rep_attempt = self.current_rom_percentage >= rep_attempt_threshold
        
        if is_rep_attempt and not self.min_depth_reached:
            self.min_depth_reached = True
            self.bottom_position_frame = frame_number

        # Switch to Ascent
        state_change_allowed = (
            frame_number - self.last_state_change_frame > self.state_change_cooldown
        )
        if (
            self._knee_velocity > 1.0  # Clearly moving up
            and state_change_allowed
        ):
            self.state = "coming_up"
            self.last_state_change_frame = frame_number
            logger.debug(f"State: COMING_UP | ROM: {self.current_rom_percentage:.2f}")

    def _handle_state_ascent(self, knee_angle: float, hip_angle: float, thigh_angle: float, back_curvature: float, frame_number: int) -> None:
        """Monitor ascent phase and check for completion."""
        self.min_thigh_angle_this_rep = min(self.min_thigh_angle_this_rep, thigh_angle)
        self.min_hip_angle_this_rep = min(self.min_hip_angle_this_rep, hip_angle)
        self.max_hip_angle_this_rep = max(self.max_hip_angle_this_rep, hip_angle)

        # Form Checks
        hip_max_threshold = self.thresholds.get("hip_max", 190.0)
        relative_diff_threshold = self.thresholds.get("relative_angle_difference", 30.0)
        curvature_threshold = self.thresholds.get("back_curvature_threshold", 1.5)

        # Hyperextension
        if curvature_threshold > 0 and back_curvature > curvature_threshold:
                self.hyperextension_detected = True
                logger.debug(f"Hyperextension (Curvature): {back_curvature:.2f} > {curvature_threshold}")
        elif hip_angle > hip_max_threshold:
                self.hyperextension_detected = True
                logger.debug(f"Hyperextension (Hip Angle): {hip_angle:.1f} > {hip_max_threshold}")
        
        # Good Morning Squat (Ascent Forward Lean)
        elif (knee_angle - hip_angle > relative_diff_threshold):
                self.forward_lean_detected = True
                logger.debug(f"Good Morning Squat (Ascent): Knee={knee_angle:.1f}, Hip={hip_angle:.1f}, Diff={knee_angle - hip_angle:.1f}")

        # Completion Check
        is_near_standing = knee_angle > self.thresholds["knee_max"] - 10
        is_moving_slowly_up = self._knee_velocity > -1.0

        if is_near_standing and is_moving_slowly_up:
            if frame_number - self.last_rep_frame > self.cooldown_frames:
                self._complete_rep(frame_number)

    def _complete_rep(self, frame_number: int) -> None:
        """Finalize the rep logic."""
        if not self.min_depth_reached:
                # Reset without counting
                self._reset_rep_state()
                logger.debug("Movement ignored - insufficient depth for rep attempt")
                return

        # Duration Check
        rep_duration = (frame_number - self.rep_start_frame) / self.fps
        min_rep_duration = self.thresholds.get("min_rep_duration", 1.0)
            
        if rep_duration < min_rep_duration:
            self._reset_rep_state()
            logger.debug(f"Rep ignored - Duration {rep_duration:.2f}s < {min_rep_duration}s")
            return

        # Complete Rep
        self.total_reps += 1
        self.last_rep_frame = frame_number
        self.last_state_change_frame = frame_number

        # Evaluate Success
        success_threshold_val = self.thresholds.get("rep_success_threshold", 0.40)
        depth_success = self.max_rom_this_rep >= success_threshold_val 

        form_issues = []
        failure_justifications = {}

        # 1. Depth
        if not depth_success:
            form_issues.append("insufficient_depth")
            very_shallow_threshold = self.thresholds.get("very_shallow_threshold", 0.15)
            moderate_shallow_threshold = self.thresholds.get("moderate_shallow_threshold", 0.30)
            failure_justifications["insufficient_depth"] = f"ROM {self.max_rom_this_rep:.2f} < {success_threshold_val}"

            if self.max_rom_this_rep < very_shallow_threshold:
                form_issues.append("very_shallow_rep")
                failure_justifications["very_shallow_rep"] = f"ROM {self.max_rom_this_rep:.2f} < {very_shallow_threshold}"
            elif self.max_rom_this_rep < moderate_shallow_threshold:
                form_issues.append("moderately_shallow_rep")
                failure_justifications["moderately_shallow_rep"] = f"ROM {self.max_rom_this_rep:.2f} < {moderate_shallow_threshold}"

        # 2. Knees Over Toes
        if self.knee_past_toes_violation:
            form_issues.append("knees_over_toes")
            failure_justifications["knees_over_toes"] = "Knee passed toes"

        # 3. Forward Lean
        if self.forward_lean_detected:
            form_issues.append("excessive_forward_lean")
            failure_justifications["excessive_forward_lean"] = f"Hip-Knee Divergence > {self.thresholds.get('relative_angle_difference', 30.0)}°"

        # 4. Hyperextension
        if self.hyperextension_detected:
            form_issues.append("hyperextension")
            if hasattr(self, "back_curvature") and self.back_curvature.get("curvature", 0) > self.thresholds.get("back_curvature_threshold", 1.5):
                    failure_justifications["hyperextension"] = f"Back Curvature {self.back_curvature['curvature']:.2f} > {self.thresholds.get('back_curvature_threshold', 1.5)}"
            else:
                    hip_max_threshold = self.thresholds.get("hip_max", 190.0)
                    failure_justifications["hyperextension"] = f"Max Hip Angle {self.max_hip_angle_this_rep:.1f}° > {hip_max_threshold}°"

        # 5. Dual Camera Issues (Front View)
        if self.dual_camera_enabled:
            if self.knee_valgus_detected:
                form_issues.append("knee_valgus")
                failure_justifications["knee_valgus"] = "Knees caving inward detected (front view)"
            
            if self.hip_drop_detected:
                form_issues.append("hip_drop")
                failure_justifications["hip_drop"] = "Hip dropping on one side (front view)"

        success = len(form_issues) == 0 and self.max_rom_this_rep >= success_threshold_val

        if success:
            self.successful_reps += 1
            logger.success(f"✅ Rep {self.total_reps} Good! (Successful Rep #{self.successful_reps}) | ROM: {self.max_rom_this_rep:.2f}")
        else:
            self.unsuccessful_reps += 1
            self._handle_voice_feedback(form_issues, frame_number)
            
            justification_str = " | ".join([f"{k}: {v}" for k, v in failure_justifications.items()])
            logger.warning(f"❌ Rep Failed: {form_issues} | {justification_str}")
            self.last_failure_justification = [f"{v}" for k, v in failure_justifications.items()]
            self.failure_justification_timer = self.failure_display_duration

        # Log Summary
        self.reps_summary.append({
            "rep_number": self.total_reps,
            "success": success,
            "rom_percentage": self.max_rom_this_rep,
            "min_knee_angle": self.min_knee_angle_this_rep,
            "min_thigh_angle": self.min_thigh_angle_this_rep,
            "min_hip_angle": self.min_hip_angle_this_rep,
            "max_hip_angle": self.max_hip_angle_this_rep,
            "knee_min_baseline": self.knee_min_baseline,
            "knee_max_baseline": self.knee_max_baseline,
            "issues": form_issues,
            "justifications": failure_justifications
        })

        self._reset_rep_state()

    def _reset_rep_state(self):
        """Reset temporary state variables for the next rep."""
        self.state = "up"
        self.forward_lean_detected = False
        self.hyperextension_detected = False
        self.knee_past_toes_violation = False
        self.min_depth_reached = False
        self.min_knee_angle_this_rep = 180
        self.min_thigh_angle_this_rep = float('inf')
        self.min_hip_angle_this_rep = 180
        self.max_hip_angle_this_rep = 0
        # Reset dual camera flags
        self.reset_dual_camera_flags()
    
    def _handle_voice_feedback(self, form_issues, frame_number):
        issue = form_issues[0] if form_issues else "unknown"
        if frame_number - self.last_voice_feedback_frame > self.voice_feedback_cooldown_frames:
            correction_map = {
                "insufficient_depth": "limited_depth",
                "very_shallow_rep": "limited_depth",
                "moderately_shallow_rep": "limited_depth",
                "knees_over_toes": "knees_over_toes",
                "excessive_forward_lean": "forward_lean",
                "hyperextension": "generic"
            }
            voice_key = correction_map.get(issue, "generic")
            
            # Check specific feedback suppression
            current_time = frame_number / self.fps
            last_time = self.last_feedback_times.get(voice_key, -100)
            
            if current_time - last_time > 8.0:
                self.voice_player.play_form_correction(voice_key)
                self.voice_messages.append((current_time, f"form_{voice_key}"))
                self.last_voice_feedback_frame = frame_number
                self.last_feedback_times[voice_key] = current_time
            else:
                logger.debug(f"Suppressed repetitive feedback: {voice_key}")

    def process_frame(self, frame: np.ndarray, frame_number: int, keypoints: Optional[np.ndarray] = None, results: Any = None) -> np.ndarray:
        # If keypoints not provided but results are (legacy path), extract them
        if keypoints is None and results is not None:
             # Only attempt extraction if results look like YOLO (list)
             if isinstance(results, list):
                 keypoints = self.extract_keypoints(results)

        # Play start workout message on first frame with valid keypoints
        if not self.workout_started:
            if keypoints is not None:
                self.voice_player.play_start_workout()
                # Record timestamp for voice message
                self.voice_messages.append((frame_number / self.fps, "start_workout"))
                self.workout_started = True
                logger.info("Workout started - playing start message")

        if keypoints is None:
            # No valid keypoints detected, return original frame
            # Reset velocity tracking when no keypoints detected
            if hasattr(self, "_prev_knee_angle"):
                delattr(self, "_prev_knee_angle")
            return frame

        # Initialize previous angle on first detection
        if not hasattr(self, "_prev_knee_angle"):
            # Get side-specific keypoint indices
            kpt_indices = self.get_side_specific_keypoints()

            # Extract relevant points
            # keypoints is (17, 3), we need (2,) [x, y]
            hip_point = keypoints[kpt_indices["hip"]][:2]
            knee_point = keypoints[kpt_indices["knee"]][:2]
            ankle_point = keypoints[kpt_indices["ankle"]][:2]

            # Calculate initial angle
            initial_angle = self.calculate_angle(hip_point, knee_point, ankle_point)
            self._prev_knee_angle = initial_angle
            self._knee_velocity = 0

        # Determine side if not already done and not in front-view mode
        if not self.front_view_mode and not self.side_determined:
            # We need to adapt determine_facing_side to work with (17, 3) format
            # Currently it likely expects YOLO format or (N, 3)
            # Standardized keypoints are (17, 3) which is compatible with simple array usage
            self.determine_side_from_keypoints(keypoints)

        # Metrics
        kpt_indices = self.get_side_specific_keypoints()
        # Normal 2D points (pixels)
        shoulder_point = keypoints[kpt_indices["shoulder"]][:2]
        hip_point = keypoints[kpt_indices["hip"]][:2]
        knee_point = keypoints[kpt_indices["knee"]][:2]
        ankle_point = keypoints[kpt_indices["ankle"]][:2]

        # Extract 3D World Landmarks if available (for precise depth)
        hip_world = None
        knee_world = None
        foot_index_point = None
        heel_point = None

        if hasattr(results, "pose_world_landmarks") and results.pose_world_landmarks is not None and len(results.pose_world_landmarks) > 0:
             pl = results.pose_world_landmarks[0]
             # Indices: 23/24=Hip, 25/26=Knee
             # 29/30=Heel, 31/32=Foot Index
             
             # MediaPipe uses: Left=Odd (23), Right=Even (24) for body parts?
             # Wait, usually MP is: 
             # 23=left_hip, 24=right_hip
             # 25=left_knee, 26=right_knee
             
             # If facing LEFT -> Visible side is RIGHT.
             # So we want indices for RIGHT (24, 26, ...)
             
             mp_kpts = self.config.get("mediapipe_keypoints", {})
             
             if self.facing_side == "left":
                 hip_idx = mp_kpts.get("left_hip", 23)
                 knee_idx = mp_kpts.get("left_knee", 25)
                 heel_idx = mp_kpts.get("left_heel", 29)
                 foot_idx = mp_kpts.get("left_foot_index", 31)
             else:
                 hip_idx = mp_kpts.get("right_hip", 24)
                 knee_idx = mp_kpts.get("right_knee", 26)
                 heel_idx = mp_kpts.get("right_heel", 30)
                 foot_idx = mp_kpts.get("right_foot_index", 32)

             hip_world = pl[hip_idx]
             knee_world = pl[knee_idx]
             
             # Extract 2D/3D Foot points for knee-over-toe
             # We need pixel coords for knee-over-toe (camera plane check)
             if hasattr(results, "pose_landmarks") and results.pose_landmarks is not None and len(results.pose_landmarks) > 0:
                 npl = results.pose_landmarks[0] # Normalized
                 # Convert normalized to pixel
                 h, w = frame.shape[:2]
                 
                 heel_node = npl[heel_idx]
                 foot_node = npl[foot_idx]
                 
                 heel_point = np.array([heel_node.x * w, heel_node.y * h])
                 foot_index_point = np.array([foot_node.x * w, foot_node.y * h])

        knee_angle = self.calculate_angle(hip_point, knee_point, ankle_point)
        hip_angle = self.calculate_angle(shoulder_point, hip_point, knee_point)
        
        # Thigh Angle (Depth) - prefers 3D if available
        thigh_angle = self.calculate_thigh_angle(hip_point, knee_point, hip_world, knee_world)

        # Knee over toes - prefers foot index/heel if available
        self.knee_past_toes = self._is_knee_past_toes(knee_point, ankle_point, foot_index_point, heel_point)

        if self.state == "up" and knee_angle > 165:
            self.knee_past_toes_violation = False
        elif self.state in ("going_down", "coming_up") and self.knee_past_toes:
            self.knee_past_toes_violation = True

        # Prepare back curvature to pass to update_rep_count
        back_curvature_val = 0.0
        try:
            if hasattr(self, "segmenter") and self.segmenter is not None:
                mask = self.segmenter.segment_person(frame)
                if mask is not None:
                    curvature_info = compute_back_curvature(
                        mask,
                        keypoints[:, :2], # Pass xy
                        self.config["keypoints"],
                        self.facing_side or "right",
                    )
                    # Attach to self for metadata access later
                    setattr(self, "back_curvature", curvature_info)
                    back_curvature_val = curvature_info.get("curvature", 0.0)
                    if np.isnan(back_curvature_val):
                         back_curvature_val = 0.0
        except Exception as e:
            logger.debug(f"Back curvature computation error: {e}")

        # Process logic
        self.update_rep_count(knee_angle, hip_angle, thigh_angle, back_curvature_val, frame_number)

        # Rest of logic (VBT, boundaries, visualizations)
        # Check movement boundaries
        movement_boundary_info = self.check_movement_boundary(keypoints)

        # Add VBT data
        try:
            hip_idx = kpt_indices["hip"]
            knee_idx = kpt_indices["knee"]
            ankle_idx = kpt_indices["ankle"]

            vbt_keypoints = np.array([keypoints[hip_idx], keypoints[knee_idx], keypoints[ankle_idx]])
            vbt_stage = "down" if self.state == "going_down" else "up"

            self.vbt_calculator.add_frame_data(
                frame_number=frame_number,
                keypoints=vbt_keypoints,
                rep_count=self.total_reps + 1,
                stage=vbt_stage,
                angle=knee_angle,
            )
        except Exception as e:
            logger.debug(f"VBT calculation error: {e}")

        # Compute anthropometrics once
        try:
            if self.anthropometrics is None:
                self.anthropometrics = compute_anthropometrics(
                    keypoints[:, :2], keypoints[:, 2], self.config["keypoints"]
                )
                logger.info("Anthropometrics computed")
        except Exception as e:
            logger.debug(f"Anthropometrics computation error: {e}")

        # Prepare MediaPipe points for visualization
        mediapipe_points = {}
        if heel_point is not None:
            mediapipe_points['heel'] = heel_point
        if foot_index_point is not None:
            mediapipe_points['foot_index'] = foot_index_point

        # Manage failure justification timer
        if self.failure_justification_timer > 0:
            self.failure_justification_timer -= 1
        
        # Draw visualization
        frame = self.visualizer.visualize(
            frame,
            keypoints[:, :2],
            knee_angle,
            hip_angle,
            self.successful_reps,
            self.unsuccessful_reps,
            self.total_reps,
            self.state,
            self.facing_side or "right",
            mediapipe_points if mediapipe_points else None,
            movement_boundary_info["boundary_status"] if movement_boundary_info else None,
            thresholds=self.thresholds,
            failure_justifications=self.last_failure_justification if self.failure_justification_timer > 0 else None
        )

        # Draw movement boundary box
        frame = self.visualizer.draw_movement_boundary(frame, movement_boundary_info)

        # Front-view overlays
        if self.front_view_mode:
            self.front_view_metrics = self._compute_front_view_metrics_from_keypoints(keypoints)
            if self.front_view_metrics is not None:
                try:
                    frame = self.visualizer.draw_front_view_metrics(
                        frame, keypoints[:, :2], self.front_view_metrics, self.config["keypoints"]
                    )
                except Exception as e:
                    logger.debug(f"Front-view overlay error: {e}")

        return frame

    # DUAL CAMERA METHODS
    
    def process_front_frame(
        self, 
        front_keypoints: np.ndarray, 
        frame_shape: Tuple[int, int],
        frame_number: int
    ) -> Dict[str, Any]:
        """
        Process front-view frame for additional form analysis.
        
        Detects:
        - Knee valgus/varus (knees caving in/out)
        - Hip alignment (hip drop)
        - Stance width
        
        Args:
            front_keypoints: Keypoints from front camera (17, 3)
            frame_shape: (height, width) of frame
            frame_number: Current frame number
            
        Returns:
            Dictionary with front-view metrics
        """
        if front_keypoints is None:
            return {}
        
        kpt_config = self.config.get("keypoints", {})
        
        # Compute front-view specific metrics
        knee_valgus = self.compute_knee_valgus(front_keypoints, kpt_config, frame_shape)
        hip_alignment = self.compute_hip_alignment(front_keypoints, kpt_config, frame_shape)
        stance = self.compute_stance_width(front_keypoints, kpt_config, frame_shape)
        
        # Update form flags during active rep
        if self.state in ["going_down", "coming_up"]:
            if knee_valgus.get("valgus_detected"):
                self.knee_valgus_detected = True
                logger.debug(f"Knee valgus detected in front view: L={knee_valgus['left_valgus_angle']:.1f}°, R={knee_valgus['right_valgus_angle']:.1f}°")
            
            if hip_alignment.get("hip_drop_detected"):
                self.hip_drop_detected = True
                logger.debug(f"Hip drop detected: {hip_alignment['hip_drop_side']} side, {hip_alignment['hip_tilt_angle']:.1f}°")
        
        # Store for visualization
        self.last_front_analysis = {
            "knee_valgus": knee_valgus,
            "hip_alignment": hip_alignment,
            "stance": stance,
            "frame_number": frame_number,
        }
        
        return self.last_front_analysis
    
    def process_dual_frames(
        self,
        side_frame: np.ndarray,
        front_frame: np.ndarray,
        side_keypoints: Optional[np.ndarray],
        front_keypoints: Optional[np.ndarray],
        frame_number: int,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process both side and front camera frames for comprehensive analysis.
        
        Args:
            side_frame: Frame from side camera
            front_frame: Frame from front camera
            side_keypoints: Keypoints from side view
            front_keypoints: Keypoints from front view
            frame_number: Current frame number
            
        Returns:
            Tuple of (processed_side_frame, combined_metadata)
        """
        # Process side frame (primary analysis)
        processed_frame = self.process_frame(side_frame, frame_number, side_keypoints)
        
        # Process front frame for additional metrics
        front_metrics = {}
        if front_keypoints is not None:
            front_metrics = self.process_front_frame(
                front_keypoints, 
                front_frame.shape[:2], 
                frame_number
            )
        
        # Combine metadata
        combined_metadata = {
            "state": self.state,
            "total_reps": self.total_reps,
            "successful_reps": self.successful_reps,
            "unsuccessful_reps": self.unsuccessful_reps,
            "front_view_metrics": front_metrics,
            "dual_camera_active": True,
        }
        
        return processed_frame, combined_metadata
    
    def get_dual_camera_issues(self) -> list:
        """Get form issues detected from dual camera analysis."""
        issues = []
        
        if self.knee_valgus_detected:
            issues.append("knee_valgus")
        
        if self.hip_drop_detected:
            issues.append("hip_drop")
        
        return issues
    
    def reset_dual_camera_flags(self):
        """Reset dual camera form flags at start of new rep."""
        self.knee_valgus_detected = False
        self.hip_drop_detected = False

    def finalize_analysis(self, output_dir: str, timestamp: str = None) -> None:
        """
        Finalize analysis and save VBT results.

        Args:
            output_dir: Directory to save results
            timestamp: Optional timestamp to use for VBT analysis filename
        """
        # Play end workout message and wait for it to finish
        if not self.workout_ended:
            self.voice_player.play_end_workout_and_wait()
            # Record timestamp for voice message
            if self.voice_messages:  # Get last timestamp if available
                last_timestamp = self.voice_messages[-1][0] + 1.0  # Add 1 second buffer
            else:
                last_timestamp = 0
            self.voice_messages.append((last_timestamp, "end_workout"))
            self.workout_ended = True
            logger.info("Workout ended - end message completed")

        # Log Summary
        logger.info("=== REP ANALYSIS SUMMARY ===")
        for rep in self.reps_summary:
            logger.info(f"Rep {rep['rep_number']}: Success={rep['success']}, Issues={rep['issues']}, MinThighAngle={rep['min_thigh_angle']:.1f}")

        try:
            # Finalize VBT analysis
            self.vbt_calculator.finalize_analysis()
            vbt_file_path = self.vbt_calculator.save_to_json(
                output_dir=output_dir, exercise_name="squat", timestamp=timestamp
            )
            logger.info(f"VBT analysis saved to: {vbt_file_path}")
        except Exception as e:
            logger.warning(f"VBT analysis error: {e}")

        # Log final results
        logger.success(
            f"Final results - Successful: {self.successful_reps}, "
            f"Unsuccessful: {self.unsuccessful_reps}, Total: {self.total_reps}"
        )

        # Cleanup voice player
        self.voice_player.cleanup()

        # Save detailed analysis data
        self._save_exercise_data(output_dir, timestamp)

    def _save_exercise_data(self, output_dir: str, timestamp: str = None) -> None:
        """
        Save detailed exercise analysis data to JSON.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        filename = f"squat_analysis_details_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        data = {
            "exercise_type": "squat",
            "timestamp": timestamp,
            "summary": {
                "total_reps": self.total_reps,
                "successful_reps": self.successful_reps,
                "unsuccessful_reps": self.unsuccessful_reps,
                "facing_side": self.facing_side,
            },
            "reps_detail": self.reps_summary,
            "voice_messages": self.voice_messages,
            "anthropometrics": self.anthropometrics,
            "movement_boundaries": {
                "boundaries": self.movement_boundaries,
                "initial_position": self.initial_position.tolist() if self.initial_position is not None else None,
            },
            "config": self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Exercise details saved to: {filepath}")



    def get_exercise_name(self) -> str:
        """Get the exercise name."""
        return "squat"

    def get_results(self) -> Dict[str, Any]:
        """
        Get current analysis results.

        Returns:
            Dictionary with current results
        """
        results = {
            "successful_reps": self.successful_reps,
            "unsuccessful_reps": self.unsuccessful_reps,
            "total_reps": self.total_reps,
            "facing_side": self.facing_side,
            "current_state": self.state,
            "rom_established": self.rom_established,
            "knee_max_baseline": self.knee_max_baseline,
            "knee_min_baseline": self.knee_min_baseline,
            "knee_range": self.knee_range,
        }
        # Attach front-view data if available
        results.update(
            {
                "front_view_mode": self.front_view_mode,
                "front_view_metrics": self.front_view_metrics,
            }
        )
        # Attach movement boundary data if available
        results.update(
            {
                "movement_boundaries": self.movement_boundaries,
                "initial_position": self.initial_position.tolist() if self.initial_position is not None else None,
            }
        )
        return results

    def get_anthropometrics(self) -> Optional[Dict[str, Any]]:
        """Return computed anthropometrics if available."""
        return self.anthropometrics

    def get_voice_messages(self) -> list:
        """
        Get recorded voice messages with timestamps.

        Returns:
            List of (timestamp, message_type) tuples
        """
        return self.voice_messages.copy()
