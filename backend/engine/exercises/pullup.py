"""
Pullup exercise analyzer with complete logic implementation.
Includes side detection, rep counting, form analysis, VBT calculations, and voice feedback.
Supports dual-camera for kipping detection (side) and arm symmetry (front).
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
from engine.velocity_calculator import VelocityCalculator
from engine.core.utils import compute_anthropometrics
from engine.exercises.dual_camera_mixin import DualCameraExerciseMixin


class PullupVisualizer:
    """Visualizer for pullup exercise analysis with clean UI design."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pullup visualizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Modern color palette
        self.colors = {
            "primary": (70, 130, 180),  # Steel Blue
            "success": (34, 139, 34),  # Forest Green
            "warning": (255, 165, 0),  # Orange
            "danger": (220, 20, 60),  # Crimson
            "info": (30, 144, 255),  # Dodger Blue
            "dark": (25, 25, 35),  # Dark slate
            "light": (245, 245, 245),  # Almost white
            "keypoints": (255, 105, 180),  # Hot pink
            "elbow_line": (50, 205, 50),  # Lime Green
            "shoulder_line": (255, 215, 0),  # Gold
            "angle_arc": (186, 85, 211),  # Medium Purple
            "progress_full": (34, 139, 34),  # Forest Green
            "progress_partial": (255, 165, 0),  # Orange
            "progress_low": (220, 20, 60),  # Crimson
        }

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 1

    def _get_keypoint_indices(self, facing_side: str) -> Dict[str, int]:
        """Helper to get keypoint indices based on facing side."""
        # For pullups, we primarily look at the visible arm
        if facing_side == "left":
            prefix = "right"  # Visible side when facing left
        else:
            prefix = "left"  # Visible side when facing right
            
        kpt = self.config["keypoints"]
        return {
            "shoulder": kpt[f"{prefix}_shoulder"],
            "elbow": kpt[f"{prefix}_elbow"],
            "wrist": kpt[f"{prefix}_wrist"],
            "hip": kpt[f"{prefix}_hip"],
            "ear": kpt[f"{prefix}_ear"],
        }

    def draw_rounded_rectangle(
        self,
        frame: np.ndarray,
        top_left: tuple,
        bottom_right: tuple,
        color: tuple,
        radius: int = 10,
        thickness: int = -1,
    ) -> None:
        """Draw a rounded rectangle."""
        x1, y1 = top_left
        x2, y2 = bottom_right

        if thickness == -1:
            cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1)
        else:
            cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
            cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

    def draw_text(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple,
        color: tuple,
        scale: float = None,
        thickness: int = None,
    ) -> None:
        """Draw text with shadow for better visibility."""
        if scale is None:
            scale = self.font_scale
        if thickness is None:
            thickness = self.font_thickness

        x, y = position
        # Shadow
        cv2.putText(frame, text, (x + 1, y + 1), self.font, scale, (30, 30, 30), thickness)
        # Main text
        cv2.putText(frame, text, (x, y), self.font, scale, color, thickness, cv2.LINE_AA)

    def draw_keypoints(
        self, frame: np.ndarray, keypoints: np.ndarray, facing_side: str = "right"
    ) -> np.ndarray:
        """Draw keypoints with modern styling."""
        indices = self._get_keypoint_indices(facing_side)
        
        for key, idx in indices.items():
            if idx >= len(keypoints):
                continue
            x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
            
            if key == "wrist":
                # Special styling for wrist (grip point)
                cv2.circle(frame, (x, y), 12, self.colors["primary"], 2)
                cv2.circle(frame, (x, y), 6, self.colors["primary"], -1)
            else:
                cv2.circle(frame, (x, y), 10, self.colors["keypoints"], 2)
                cv2.circle(frame, (x, y), 6, self.colors["light"], -1)
                cv2.circle(frame, (x, y), 2, self.colors["keypoints"], -1)

        return frame

    def draw_lines(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        elbow_angle: float,
        facing_side: str = "right",
    ) -> np.ndarray:
        """Draw arm lines and angles."""
        indices = self._get_keypoint_indices(facing_side)
        
        shoulder = (int(keypoints[indices["shoulder"]][0]), int(keypoints[indices["shoulder"]][1]))
        elbow = (int(keypoints[indices["elbow"]][0]), int(keypoints[indices["elbow"]][1]))
        wrist = (int(keypoints[indices["wrist"]][0]), int(keypoints[indices["wrist"]][1]))

        # Draw arm lines
        cv2.line(frame, shoulder, elbow, self.colors["shoulder_line"], 4, cv2.LINE_AA)
        cv2.line(frame, elbow, wrist, self.colors["elbow_line"], 4, cv2.LINE_AA)

        # Draw angle arc at elbow
        self._draw_angle_arc(frame, elbow, shoulder, wrist, 40, self.colors["angle_arc"])

        # Elbow angle text
        elbow_text_pos = (elbow[0] + 50, elbow[1] - 20)
        self.draw_text(frame, f"{elbow_angle:.0f}°", elbow_text_pos, self.colors["elbow_line"], scale=0.7)

        return frame

    def _draw_angle_arc(
        self,
        frame: np.ndarray,
        center: tuple,
        p1: tuple,
        p3: tuple,
        radius: int = 50,
        color: tuple = (156, 39, 176),
    ) -> None:
        """Draw an arc to visualize the angle."""
        try:
            v1 = np.array(p1) - np.array(center)
            v3 = np.array(p3) - np.array(center)

            angle1 = math.degrees(math.atan2(v1[1], v1[0]))
            angle3 = math.degrees(math.atan2(v3[1], v3[0]))

            a1 = angle1 % 360
            a3 = angle3 % 360

            diff = (a3 - a1 + 360) % 360

            if diff <= 180:
                start, end = a1, a3
                if end < start:
                    end += 360
            else:
                start, end = a3, a1
                if end < start:
                    end += 360

            cv2.ellipse(frame, center, (radius, radius), 0, start, end, color, 2, cv2.LINE_AA)
        except Exception:
            pass

    def draw_stats(
        self,
        frame: np.ndarray,
        successful_reps: int,
        unsuccessful_reps: int,
        total_reps: int,
        state: str,
        failure_justifications: list = None,
    ) -> np.ndarray:
        """Draw statistics panel."""
        frame_height, frame_width = frame.shape[:2]

        panel_width = 380
        panel_height = 140
        panel_x = frame_width - panel_width - 20
        panel_y = 20

        # Draw panel background
        self._draw_overlay_panel(
            frame, 
            (panel_x, panel_y, panel_width, panel_height), 
            self.colors["primary"], 
            bg_color=self.colors["dark"],
            alpha=0.6,
            radius=10
        )

        # Header
        header_x = panel_x + 12
        header_y = panel_y + 25
        self.draw_text(frame, "Pull-up Stats", (header_x, header_y), self.colors["light"], scale=0.6)

        y_pos = header_y + 25
        line_spacing = 22

        self.draw_text(frame, f"Good: {successful_reps}", (header_x, y_pos), self.colors["success"], scale=0.55)
        y_pos += line_spacing

        self.draw_text(frame, f"Bad: {unsuccessful_reps}", (header_x, y_pos), self.colors["danger"], scale=0.55)
        y_pos += line_spacing

        self.draw_text(frame, f"Total: {total_reps}", (header_x, y_pos), self.colors["info"], scale=0.55)
        y_pos += line_spacing

        # State indicator
        state_text = "UP" if state == "up" else "DOWN"
        state_color = self.colors["success"] if state == "up" else self.colors["warning"]
        self.draw_text(frame, f"Phase: {state_text}", (header_x, y_pos), state_color, scale=0.55)

        # Failure justifications
        if failure_justifications:
            just_text = ", ".join([j[0].upper() + j[1:] for j in failure_justifications if j])
            if just_text:
                (text_w, _), _ = cv2.getTextSize(f"! {just_text}", self.font, 0.50, self.font_thickness)
                just_width = max(panel_width, text_w + 40)
                just_x = panel_x + panel_width - just_width
                just_y = panel_y + panel_height + 10
                
                self._draw_overlay_panel(
                    frame,
                    (just_x, just_y, just_width, 44),
                    self.colors["danger"],
                    bg_color=(20, 20, 20),
                    alpha=0.5,
                    radius=8
                )
                self.draw_text(frame, f"! {just_text}", (just_x + 20, just_y + 25), self.colors["danger"], scale=0.50)

        return frame

    def draw_progress_bar(
        self, frame: np.ndarray, elbow_angle: float, elbow_min: float, elbow_max: float
    ) -> np.ndarray:
        """Draw progress bar showing pullup progress."""
        frame_height, frame_width = frame.shape[:2]
        bar_width = 350
        bar_height = 12
        bar_x = (frame_width - bar_width) // 2
        bar_y = frame_height - 40

        # Background bar
        self.draw_rounded_rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (80, 80, 80),
            radius=6,
        )

        # Calculate progress (inverted: lower angle = higher progress for pullup)
        if elbow_angle <= elbow_min:
            progress = 1.0
            color = self.colors["progress_full"]
        elif elbow_angle >= elbow_max:
            progress = 0.0
            color = self.colors["progress_low"]
        else:
            progress = (elbow_max - elbow_angle) / (elbow_max - elbow_min)
            color = self.colors["progress_partial"]

        fill_width = int(bar_width * progress)
        if fill_width > 0:
            self.draw_rounded_rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + fill_width, bar_y + bar_height),
                color,
                radius=6,
            )

        self.draw_text(frame, "Pull", (bar_x - 50, bar_y + 10), self.colors["light"], scale=0.5)
        percentage = int(progress * 100)
        self.draw_text(frame, f"{percentage}%", (bar_x + bar_width + 15, bar_y + 10), color, scale=0.5)

        return frame

    def _draw_overlay_panel(
        self,
        frame: np.ndarray,
        rect: tuple,
        border_color: tuple,
        bg_color: tuple = (20, 20, 20),
        alpha: float = 0.5,
        radius: int = 10
    ) -> None:
        """Helper to draw a semi-transparent rounded panel."""
        x, y, w, h = rect
        overlay = frame.copy()
        self.draw_rounded_rectangle(overlay, (x, y), (x + w, y + h), bg_color, radius=radius)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        self.draw_rounded_rectangle(frame, (x, y), (x + w, y + h), border_color, radius=radius, thickness=1)

    def visualize(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        elbow_angle: float,
        successful_reps: int,
        unsuccessful_reps: int,
        total_reps: int,
        state: str,
        facing_side: str = "right",
        thresholds: dict = None,
        failure_justifications: list = None,
    ) -> np.ndarray:
        """Main visualization function."""
        frame = self.draw_keypoints(frame, keypoints, facing_side)
        frame = self.draw_lines(frame, keypoints, elbow_angle, facing_side)
        frame = self.draw_stats(
            frame, successful_reps, unsuccessful_reps, total_reps, state, failure_justifications
        )

        elbow_min = 60.0
        elbow_max = 170.0
        if thresholds:
            elbow_min = thresholds.get("elbow_min", elbow_min)
            elbow_max = thresholds.get("elbow_max", elbow_max)

        frame = self.draw_progress_bar(frame, elbow_angle, elbow_min, elbow_max)

        return frame


class PullupExercise(DualCameraExerciseMixin):
    """Complete pullup exercise analyzer with integrated logic and voice feedback."""

    def __init__(
        self, config: Dict[str, Any], fps: float = 30.0, segmenter: Any = None
    ):
        """
        Initialize the pullup analyzer.

        Args:
            config: Configuration dictionary
            fps: Video frames per second
            segmenter: Optional segmenter for body analysis
        """
        self.config = config
        self.fps = fps
        self.cooldown_frames = int(fps * 0.5)
        self.state_change_cooldown = int(fps * 0.3)

        self.last_rep_frame = -self.cooldown_frames
        self.last_state_change_frame = 0
        self.min_elbow_angle_this_rep = float('inf')

        # ROM tracking
        self.elbow_max_baseline = None  # Extended (hanging)
        self.elbow_min_baseline = None  # Bent (top of pull)
        self.rom_established = False
        self.elbow_range = None

        # Rep counting
        self.successful_reps = 0
        self.unsuccessful_reps = 0
        self.total_reps = 0
        self.state = "down"  # "down", "going_up", "coming_down"
        self.min_height_reached = False
        self.max_rom_this_rep = 0.0
        self.top_position_frame = 0
        
        # State tracking
        self.last_failure_justification = []
        self.failure_justification_timer = 0
        
        # Dual camera support
        DualCameraExerciseMixin.__init__(self)
        self.dual_camera_enabled = False
        self.front_view_keypoints = None
        self.last_front_analysis = None
        
        # Dual camera form issues
        self.asymmetric_pull_detected = False
        self.grip_width_issue_detected = False
        self.body_twist_detected = False
        self.failure_display_duration = int(fps * 3.0)
        self.rep_start_frame = 0
        self._prev_elbow_angle = 0
        self._elbow_velocity = 0

        # Form detection flags
        self.kipping_detected = False
        self.asymmetric_pull_detected = False
        self.shrugging_detected = False
        self.scapular_pull_missing_detected = False
        self.fast_eccentric_detected = False
        self.forearm_offset_detected = False
        self.torso_incline_detected = False

        # Summary logging
        self.reps_summary = []
        
        # New angle tracking variables
        self.current_forearm_angle = 0.0
        self.current_torso_incline = 0.0
        self.current_neck_shoulder_angle = 180.0
        self.current_hip_x = 0.0
        self.current_shoulder_y = 0.0

        # Side detection
        self.facing_side: Optional[str] = None
        self.side_determined = False

        # VBT calculator
        self.vbt_calculator = VelocityCalculator(fps=fps)

        # Visualizer
        self.visualizer = PullupVisualizer(config)

        # Voice message player
        self.voice_player = VoiceMessagePlayer(config)
        self.voice_messages = []

        # Workout state tracking
        self.workout_started = False
        self.workout_ended = False

        # Voice feedback tracking
        self.last_voice_feedback_frame = -self.cooldown_frames
        self.voice_feedback_cooldown_frames = int(fps * 3)
        self.last_feedback_times = {}

        # Kipping detection - track hip position variance
        self.hip_y_history = []
        self.max_hip_y_history = 30  # Track for 1 second at 30fps

        logger.info("Pullup exercise analyzer initialized")

        self.anthropometrics = None
        self.segmenter = segmenter

        # Experience level thresholds
        self.experience_level = (self.config.get("experience", {}) or {}).get(
            "level", "intermediate"
        )
        self.thresholds = self._build_thresholds(self.experience_level)

    def _build_thresholds(self, level: str) -> Dict[str, float]:
        """Build thresholds for pullup based on experience level from config."""
        level = (level or "intermediate").lower()
        
        # Get pullup-specific config
        pullup_config = self.config.get("pullup", {}) or {}
        pullup_thresholds = pullup_config.get("thresholds", {})
        min_rep_duration = float(pullup_config.get("min_rep_duration", 1.0))
        
        # Get level-specific config
        level_cfg = pullup_thresholds.get(level, pullup_thresholds.get("intermediate", {}))
        
        return {
            "elbow_min": float(level_cfg.get("elbow_min", 60.0)),
            "elbow_max": float(level_cfg.get("elbow_max", 170.0)),
            "attempt_threshold": float(level_cfg.get("attempt_threshold", 0.35)),
            "success_threshold": float(level_cfg.get("success_threshold", 0.70)),
            "kipping_threshold": float(level_cfg.get("kipping_threshold", 30.0)),
            "min_rep_duration": min_rep_duration,
            "forearm_vertical_tolerance": float(level_cfg.get("forearm_vertical_tolerance", 12.0)),
            "torso_incline_min": float(level_cfg.get("torso_incline_min", 15.0)),
            "torso_incline_max": float(level_cfg.get("torso_incline_max", 45.0)),
            "hip_x_oscillation_max": float(level_cfg.get("hip_x_oscillation_max", 6.0)),
            "neck_shoulder_min": float(level_cfg.get("neck_shoulder_min", 85.0)),
            "eccentric_velocity_max": float(level_cfg.get("eccentric_velocity_max", 4.0)),
        }

    def determine_side_from_keypoints(self, keypoints_data: np.ndarray, allow_override: bool = False) -> None:
        """Determine which side is facing the camera."""
        if self.side_determined and not allow_override:
            return

        if keypoints_data is not None:
            new_side = determine_facing_side(keypoints_data)
            self.facing_side = new_side
            self.side_determined = True
            logger.info(f"Determined facing side: {self.facing_side}")

    def set_facing_side(self, side: str) -> None:
        """Explicitly set the facing side."""
        if side and side.lower() in ["left", "right"]:
            self.facing_side = side.lower()
            self.side_determined = True
            logger.info(f"Facing side explicitly set to: {self.facing_side}")

    def set_experience_level(self, level: str) -> None:
        """Set user experience level and update thresholds."""
        level = (level or "intermediate").lower()
        if level not in ("beginner", "intermediate", "advanced"):
            level = "intermediate"
        self.experience_level = level
        self.thresholds = self._build_thresholds(level)
        logger.info(f"Experience level set to {level}")

    def get_side_specific_keypoints(self) -> Dict[str, int]:
        """Get keypoint indices based on the determined facing side."""
        if self.facing_side == "left":
            return {
                "shoulder": self.config["keypoints"]["right_shoulder"],
                "elbow": self.config["keypoints"]["right_elbow"],
                "wrist": self.config["keypoints"]["right_wrist"],
                "hip": self.config["keypoints"]["right_hip"],
                "ear": self.config["keypoints"]["right_ear"],
            }
        else:
            return {
                "shoulder": self.config["keypoints"]["left_shoulder"],
                "elbow": self.config["keypoints"]["left_elbow"],
                "wrist": self.config["keypoints"]["left_wrist"],
                "hip": self.config["keypoints"]["left_hip"],
                "ear": self.config["keypoints"]["left_ear"],
            }

    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points."""
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        v1 = p1 - p2
        v2 = p3 - p2
        dot_prod = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norm_product == 0:
            return 0.0
            
        cos_angle = np.clip(dot_prod / norm_product, -1, 1)
        return math.degrees(math.acos(cos_angle))

    def calculate_vertical_angle(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate angle of a line formed by p1-p2 relative to vertical line."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if dy == 0:
            return 90.0
            
        angle = np.degrees(np.arctan(abs(dx)/abs(dy)))
        return angle

    def calculate_horizontal_angle(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate angle of a line formed by p1-p2 relative to horizontal line."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if dx == 0:
            return 90.0
            
        angle = np.degrees(np.arctan(abs(dy)/abs(dx)))
        return angle

    def extract_keypoints(self, results) -> Optional[np.ndarray]:
        """Extract keypoints from YOLO results."""
        if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            return None

        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        confidences = results[0].keypoints.conf[0].cpu().numpy()

        kpt_indices = self.get_side_specific_keypoints()
        required_indices = [
            kpt_indices["shoulder"],
            kpt_indices["elbow"],
            kpt_indices["wrist"],
        ]

        for idx in required_indices:
            if confidences[idx] < 0.5:
                return None

        return keypoints

    def _establish_rom_baselines(self, elbow_angle: float, frame_number: int) -> bool:
        """Establish ROM baselines for proper rep counting."""
        # Extended position (hanging)
        if self.elbow_max_baseline is None and elbow_angle > 150:
            self.elbow_max_baseline = elbow_angle
            logger.info(f"Established elbow max baseline: {elbow_angle:.2f}")

        # Bent position (top of pull)
        elbow_min_threshold = self.thresholds.get("elbow_min", 70.0)
        if self.elbow_max_baseline is not None and elbow_angle < elbow_min_threshold:
            if self.elbow_min_baseline is None or elbow_angle < self.elbow_min_baseline:
                self.elbow_min_baseline = elbow_angle
                self.elbow_range = self.elbow_max_baseline - self.elbow_min_baseline
                self.rom_established = True
                logger.info(f"Updated elbow min baseline: {elbow_angle:.2f}, ROM range: {self.elbow_range:.2f}")

        return self.rom_established

    def _calculate_rom_percentage(self, current_elbow_angle: float) -> float:
        """Calculate the percentage of ROM achieved."""
        if not isinstance(current_elbow_angle, (int, float)) or np.isnan(current_elbow_angle):
            return 0.0

        working_max = self.elbow_max_baseline if self.elbow_max_baseline is not None else 170.0
        working_min = self.elbow_min_baseline
        working_range = self.elbow_range

        if not self.rom_established or working_min is None:
            working_min = self.thresholds.get("elbow_min", 70.0)
            working_range = working_max - working_min

        if working_range <= 0:
            return 0.0

        rom_percentage = (working_max - current_elbow_angle) / working_range
        return max(0.0, min(1.0, rom_percentage))

    def _detect_kipping(self, hip_y: float) -> bool:
        """Detect kipping motion by monitoring hip position variance."""
        self.hip_y_history.append(hip_y)
        if len(self.hip_y_history) > self.max_hip_y_history:
            self.hip_y_history.pop(0)

        if len(self.hip_y_history) >= 10:
            variance = np.std(self.hip_y_history)
            kipping_threshold = self.thresholds.get("kipping_threshold", 30.0)
            return variance > kipping_threshold
        return False

    def update_rep_count(self, elbow_angle: float, hip_y: float, frame_number: int) -> None:
        """Update rep count based on ROM calculation."""
        self._establish_rom_baselines(elbow_angle, frame_number)
        self.current_rom_percentage = self._calculate_rom_percentage(elbow_angle)

        if self.state in ["going_up", "coming_down"]:
            self.max_rom_this_rep = max(self.max_rom_this_rep, self.current_rom_percentage)

        # Calculate velocity
        if not hasattr(self, "_prev_elbow_angle"):
            self._prev_elbow_angle = elbow_angle
            self._elbow_velocity = 0
        else:
            self._elbow_velocity = elbow_angle - self._prev_elbow_angle
            self._prev_elbow_angle = elbow_angle

        # Check for timeout
        if self._check_timeout(frame_number):
            return

        # State machine
        if self.state == "down":
            self._handle_state_down(elbow_angle, frame_number)
        elif self.state == "going_up":
            self._handle_state_ascent(elbow_angle, hip_y, frame_number)
        elif self.state == "coming_down":
            self._handle_state_descent(elbow_angle, frame_number)

    def _check_timeout(self, frame_number: int) -> bool:
        """Reset state if stuck for too long."""
        if self.state != "down" and (frame_number - self.last_state_change_frame > self.fps * 5):
            self.state = "down"
            self.kipping_detected = False
            self.min_height_reached = False
            self.hip_y_history.clear()
            logger.debug("State reset due to timeout")
            return True
        return False

    def _handle_state_down(self, elbow_angle: float, frame_number: int) -> None:
        """Monitor for the start of a pullup."""
        state_change_allowed = (
            frame_number - self.last_state_change_frame > self.state_change_cooldown
        )

        significant_bend = False
        if self.elbow_max_baseline is not None:
            significant_bend = elbow_angle < (self.elbow_max_baseline - 10.0)
        else:
            significant_bend = elbow_angle < 160.0

        if (
            self._elbow_velocity < -2.0  # Angle decreasing = pulling up
            and significant_bend
            and state_change_allowed
        ):
            self.state = "going_up"
            self.min_height_reached = False
            self.top_position_frame = 0
            self.min_elbow_angle_this_rep = 180
            self.last_state_change_frame = frame_number
            self.max_rom_this_rep = 0.0
            self.rep_start_frame = frame_number
            self.kipping_detected = False
            self.hip_y_history.clear()
            logger.debug(f"State: GOING_UP | Frame: {frame_number}")

    def _handle_state_ascent(self, elbow_angle: float, hip_y: float, frame_number: int) -> None:
        """Monitor ascent phase."""
        self.min_elbow_angle_this_rep = min(self.min_elbow_angle_this_rep, elbow_angle)

        # Check for kipping and other motion issues
        if self._detect_kipping(hip_y):
            self.kipping_detected = True
            logger.debug("Kipping detected during ascent")

        # Evaluate pendulum swing
        if not hasattr(self, "_rep_start_hip_x"):
            self._rep_start_hip_x = self.current_hip_x
        elif abs(self.current_hip_x - self._rep_start_hip_x) > self.thresholds.get("hip_x_oscillation_max", 6.0) * 10:
            # Roughly scaled * 10 from config
            self.kipping_detected = True

        # Check for scapular pull (if shoulders haven't dropped but elbows > 5 deg flexion)
        if not hasattr(self, "_rep_start_shoulder_y"):
            self._rep_start_shoulder_y = self.current_shoulder_y
        
        # In setup, shoulder_y should decrease (move down in world, increase in pixel space)
        pixel_shoulder_drop = self.current_shoulder_y - self._rep_start_shoulder_y
        elbow_flexion = self.elbow_max_baseline - elbow_angle if self.elbow_max_baseline else 170.0 - elbow_angle
        if elbow_flexion > 5.0 and pixel_shoulder_drop < 5:
            # Elbows bent before shoulders depressed
            self.scapular_pull_missing_detected = True

        # Forearm vertical check
        if abs(self.current_forearm_angle) > self.thresholds.get("forearm_vertical_tolerance", 12.0):
            self.forearm_offset_detected = True

        # Torso incline bounds
        torso_min = self.thresholds.get("torso_incline_min", 15.0)
        torso_max = self.thresholds.get("torso_incline_max", 45.0)
        # Assuming angle is returned absolute
        if abs(self.current_torso_incline) < torso_min or abs(self.current_torso_incline) > torso_max:
            self.torso_incline_detected = True

        # Check if reached top
        rep_attempt_threshold = self.thresholds.get("attempt_threshold", 0.35)
        is_rep_attempt = self.current_rom_percentage >= rep_attempt_threshold
        
        if is_rep_attempt and not self.min_height_reached:
            self.min_height_reached = True
            self.top_position_frame = frame_number
            # Check for shrugging at the top
            if self.current_neck_shoulder_angle < self.thresholds.get("neck_shoulder_min", 85.0):
                self.shrugging_detected = True

        # Switch to descent
        state_change_allowed = (
            frame_number - self.last_state_change_frame > self.state_change_cooldown
        )
        if self._elbow_velocity > 2.0 and state_change_allowed:  # Angle increasing = going down
            self.state = "coming_down"
            self.last_state_change_frame = frame_number
            logger.debug(f"State: COMING_DOWN | ROM: {self.current_rom_percentage:.2f}")

    def _handle_state_descent(self, elbow_angle: float, frame_number: int) -> None:
        """Monitor descent phase and check for completion."""
        # Check eccentric velocity
        eccentric_vel = self._elbow_velocity
        eccentric_max = self.thresholds.get("eccentric_velocity_max", 4.0)
        if eccentric_vel > eccentric_max:
            self.fast_eccentric_detected = True

        # Check for completion - back to near full extension
        is_near_hanging = elbow_angle > self.thresholds["elbow_max"] - 15
        is_moving_slowly = self._elbow_velocity < 2.0

        if is_near_hanging and is_moving_slowly:
            if frame_number - self.last_rep_frame > self.cooldown_frames:
                self._complete_rep(frame_number)

    def _complete_rep(self, frame_number: int) -> None:
        """Finalize the rep logic."""
        if not self.min_height_reached:
            self._reset_rep_state()
            logger.debug("Movement ignored - insufficient height for rep attempt")
            return

        # Duration check
        rep_duration = (frame_number - self.rep_start_frame) / self.fps
        min_rep_duration = self.thresholds.get("min_rep_duration", 1.0)
            
        if rep_duration < min_rep_duration:
            self._reset_rep_state()
            logger.debug(f"Rep ignored - Duration {rep_duration:.2f}s < {min_rep_duration}s")
            return

        self.total_reps += 1
        self.last_rep_frame = frame_number
        self.last_state_change_frame = frame_number

        # Evaluate success
        success_threshold_val = self.thresholds.get("success_threshold", 0.70)
        rom_success = self.max_rom_this_rep >= success_threshold_val

        form_issues = []
        failure_justifications = {}

        # 1. ROM check
        if not rom_success:
            form_issues.append("incomplete_range")
            failure_justifications["incomplete_range"] = f"ROM {self.max_rom_this_rep:.2f} < {success_threshold_val}"

        # 2. Kipping check
        if self.kipping_detected:
            form_issues.append("kipping")
            failure_justifications["kipping"] = "Excessive body swing detected"
            
        # 3. Form checks (Standards)
        if self.scapular_pull_missing_detected:
            form_issues.append("weak_scapular_pull")
            failure_justifications["weak_scapular_pull"] = "Elbows bent before shoulders dropped"
            
        if self.forearm_offset_detected:
            form_issues.append("forearms_not_vertical")
            failure_justifications["forearms_not_vertical"] = "Forearms deviated from vertical line"
            
        if self.torso_incline_detected:
            form_issues.append("poor_torso_angle")
            failure_justifications["poor_torso_angle"] = "Torso angle outside required range"
            
        if self.shrugging_detected:
            form_issues.append("shrugging_top")
            failure_justifications["shrugging_top"] = "Neck-shoulder angle shows shrugging"
            
        if self.fast_eccentric_detected:
            form_issues.append("dropping_fast")
            failure_justifications["dropping_fast"] = "Descent was too fast"

        success = len(form_issues) == 0 and rom_success

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

        # Log summary
        self.reps_summary.append({
            "rep_number": self.total_reps,
            "success": success,
            "rom_percentage": self.max_rom_this_rep,
            "min_elbow_angle": self.min_elbow_angle_this_rep,
            "kipping": self.kipping_detected,
            "issues": form_issues,
            "justifications": failure_justifications
        })

        self._reset_rep_state()

    def _reset_rep_state(self):
        """Reset temporary state variables for the next rep."""
        self.state = "down"
        self.kipping_detected = False
        self.scapular_pull_missing_detected = False
        self.fast_eccentric_detected = False
        self.forearm_offset_detected = False
        self.torso_incline_detected = False
        self.shrugging_detected = False
        self.min_height_reached = False
        self.min_elbow_angle_this_rep = float('inf')
        self.hip_y_history.clear()

    def _handle_voice_feedback(self, form_issues, frame_number):
        """Handle voice feedback for form issues."""
        issue = form_issues[0] if form_issues else "unknown"
        if frame_number - self.last_voice_feedback_frame > self.voice_feedback_cooldown_frames:
            correction_map = {
                "incomplete_range": "limited_depth",
                "kipping": "generic",
            }
            voice_key = correction_map.get(issue, "generic")
            
            current_time = frame_number / self.fps
            last_time = self.last_feedback_times.get(voice_key, -100)
            
            if current_time - last_time > 8.0:
                self.voice_player.play_form_correction(voice_key)
                self.voice_messages.append((current_time, f"form_{voice_key}"))
                self.last_voice_feedback_frame = frame_number
                self.last_feedback_times[voice_key] = current_time

    def process_frame(self, frame: np.ndarray, frame_number: int, keypoints: Optional[np.ndarray] = None, results: Any = None) -> np.ndarray:
        """Process a single frame for pullup analysis."""
        if keypoints is None and results is not None:
            if isinstance(results, list):
                keypoints = self.extract_keypoints(results)

        # Play start workout message
        if not self.workout_started:
            if keypoints is not None:
                self.voice_player.play_start_workout()
                self.voice_messages.append((frame_number / self.fps, "start_workout"))
                self.workout_started = True
                logger.info("Workout started - playing start message")

        if keypoints is None:
            if hasattr(self, "_prev_elbow_angle"):
                delattr(self, "_prev_elbow_angle")
            return frame

        # Initialize previous angle
        if not hasattr(self, "_prev_elbow_angle"):
            kpt_indices = self.get_side_specific_keypoints()
            shoulder_point = keypoints[kpt_indices["shoulder"]][:2]
            elbow_point = keypoints[kpt_indices["elbow"]][:2]
            wrist_point = keypoints[kpt_indices["wrist"]][:2]
            initial_angle = self.calculate_angle(shoulder_point, elbow_point, wrist_point)
            self._prev_elbow_angle = initial_angle
            self._elbow_velocity = 0

        # Determine side if not done
        if not self.side_determined:
            self.determine_side_from_keypoints(keypoints)

        # Calculate metrics
        kpt_indices = self.get_side_specific_keypoints()
        shoulder_point = keypoints[kpt_indices["shoulder"]][:2]
        elbow_point = keypoints[kpt_indices["elbow"]][:2]
        wrist_point = keypoints[kpt_indices["wrist"]][:2]
        hip_point = keypoints[kpt_indices["hip"]][:2]
        ear_point = keypoints[kpt_indices["ear"]][:2]

        elbow_angle = self.calculate_angle(shoulder_point, elbow_point, wrist_point)
        self.current_neck_shoulder_angle = self.calculate_angle(ear_point, shoulder_point, elbow_point)
        self.current_forearm_angle = self.calculate_vertical_angle(wrist_point, elbow_point)
        self.current_torso_incline = self.calculate_vertical_angle(shoulder_point, hip_point)
        self.current_hip_x = hip_point[0]
        self.current_shoulder_y = shoulder_point[1]
        
        hip_y = hip_point[1]  # For kipping detection

        # Update rep count
        self.update_rep_count(elbow_angle, hip_y, frame_number)

        # VBT data
        try:
            shoulder_idx = kpt_indices["shoulder"]
            elbow_idx = kpt_indices["elbow"]
            wrist_idx = kpt_indices["wrist"]

            vbt_keypoints = np.array([keypoints[shoulder_idx], keypoints[elbow_idx], keypoints[wrist_idx]])
            vbt_stage = "up" if self.state == "going_up" else "down"

            self.vbt_calculator.add_frame_data(
                frame_number=frame_number,
                keypoints=vbt_keypoints,
                rep_count=self.total_reps + 1,
                stage=vbt_stage,
                angle=elbow_angle,
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

        # Manage failure justification timer
        if self.failure_justification_timer > 0:
            self.failure_justification_timer -= 1

        # Draw visualization
        frame = self.visualizer.visualize(
            frame,
            keypoints[:, :2],
            elbow_angle,
            self.successful_reps,
            self.unsuccessful_reps,
            self.total_reps,
            self.state,
            self.facing_side or "right",
            thresholds=self.thresholds,
            failure_justifications=self.last_failure_justification if self.failure_justification_timer > 0 else None
        )

        return frame

    # DUAL CAMERA METHODS
    
    def process_front_frame(
        self, 
        front_keypoints: np.ndarray, 
        frame_shape: Tuple[int, int],
        frame_number: int
    ) -> Dict[str, Any]:
        """
        Process front-view frame for pull-up specific analysis.
        
        Detects:
        - Asymmetric pull (one arm pulling more than the other)
        - Grip width issues
        - Body twist during pull
        """
        if front_keypoints is None:
            return {}
        
        kpt_config = self.config.get("keypoints", {})
        
        # Compute front-view specific metrics
        arm_symmetry = self.compute_arm_symmetry(front_keypoints, kpt_config, frame_shape)
        shoulder_alignment = self.compute_shoulder_alignment(front_keypoints, kpt_config, frame_shape)
        hip_alignment = self.compute_hip_alignment(front_keypoints, kpt_config, frame_shape)
        
        # Check for body twist (hip vs shoulder alignment)
        body_twist_angle = abs(
            shoulder_alignment.get("shoulder_tilt_angle", 0) - 
            hip_alignment.get("hip_tilt_angle", 0)
        )
        
        # Update form flags during active rep
        if self.state in ["going_up", "coming_down"]:
            if arm_symmetry.get("asymmetric_detected"):
                self.asymmetric_pull_detected = True
                logger.debug(f"Asymmetric pull detected: score={arm_symmetry['symmetry_score']:.1f}")
            
            if body_twist_angle > 10.0:
                self.body_twist_detected = True
                logger.debug(f"Body twist detected: {body_twist_angle:.1f}°")
        
        self.last_front_analysis = {
            "arm_symmetry": arm_symmetry,
            "shoulder_alignment": shoulder_alignment,
            "hip_alignment": hip_alignment,
            "body_twist_angle": body_twist_angle,
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
        """Process both side and front camera frames for comprehensive analysis."""
        # Process side frame (primary analysis - kipping is detected from side)
        processed_frame = self.process_frame(side_frame, frame_number, side_keypoints)
        
        # Process front frame for symmetry metrics
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
            "kipping_detected": self.kipping_detected,
            "front_view_metrics": front_metrics,
            "dual_camera_active": True,
        }
        
        return processed_frame, combined_metadata
    
    def get_dual_camera_issues(self) -> list:
        """Get form issues detected from dual camera analysis."""
        issues = []
        
        if self.asymmetric_pull_detected:
            issues.append("asymmetric_pull")
        
        if self.body_twist_detected:
            issues.append("body_twist")
        
        return issues
    
    def reset_dual_camera_flags(self):
        """Reset dual camera form flags at start of new rep."""
        self.asymmetric_pull_detected = False
        self.grip_width_issue_detected = False
        self.body_twist_detected = False

    def finalize_analysis(self, output_dir: str, timestamp: str = None) -> None:
        """Finalize analysis and save results."""
        if not self.workout_ended:
            self.voice_player.play_end_workout_and_wait()
            if self.voice_messages:
                last_timestamp = self.voice_messages[-1][0] + 1.0
            else:
                last_timestamp = 0
            self.voice_messages.append((last_timestamp, "end_workout"))
            self.workout_ended = True
            logger.info("Workout ended - end message completed")

        logger.info("=== REP ANALYSIS SUMMARY ===")
        for rep in self.reps_summary:
            logger.info(f"Rep {rep['rep_number']}: Success={rep['success']}, Issues={rep['issues']}, MinElbowAngle={rep['min_elbow_angle']:.1f}")

        try:
            self.vbt_calculator.finalize_analysis()
            vbt_file_path = self.vbt_calculator.save_to_json(
                output_dir=output_dir, exercise_name="pullup", timestamp=timestamp
            )
            logger.info(f"VBT analysis saved to: {vbt_file_path}")
        except Exception as e:
            logger.warning(f"VBT analysis error: {e}")

        logger.success(
            f"Final results - Successful: {self.successful_reps}, "
            f"Unsuccessful: {self.unsuccessful_reps}, Total: {self.total_reps}"
        )

        self.voice_player.cleanup()
        self._save_exercise_data(output_dir, timestamp)

    def _save_exercise_data(self, output_dir: str, timestamp: str = None) -> None:
        """Save detailed exercise analysis data to JSON."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        filename = f"pullup_analysis_details_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        data = {
            "exercise_type": "pullup",
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
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Exercise details saved to: {filepath}")

    def get_exercise_name(self) -> str:
        """Get the exercise name."""
        return "pullup"

    def get_results(self) -> Dict[str, Any]:
        """Get current analysis results."""
        return {
            "successful_reps": self.successful_reps,
            "unsuccessful_reps": self.unsuccessful_reps,
            "total_reps": self.total_reps,
            "facing_side": self.facing_side,
            "current_state": self.state,
            "rom_established": self.rom_established,
            "elbow_max_baseline": self.elbow_max_baseline,
            "elbow_min_baseline": self.elbow_min_baseline,
            "elbow_range": self.elbow_range,
        }

    def get_anthropometrics(self) -> Optional[Dict[str, Any]]:
        """Return computed anthropometrics if available."""
        return self.anthropometrics

    def get_voice_messages(self) -> list:
        """Get recorded voice messages with timestamps."""
        return self.voice_messages.copy()
