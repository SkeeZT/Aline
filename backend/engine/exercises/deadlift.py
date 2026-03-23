"""
Deadlift Exercise Analyzer

This module implements deadlift analysis with:
- Hip hinge movement pattern tracking
- Back curvature monitoring using segmentation
- Knee angle tracking
- Bar path analysis (when visible)
- Form issue detection (rounded back, knees caving, etc.)
- Rep counting based on hip and knee angles
- Dual-camera support for knee valgus and stance width detection
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

from engine.core.utils import compute_back_curvature
from engine.exercises.dual_camera_mixin import DualCameraExerciseMixin


class DeadliftVisualizer:
    """Visualization helper for deadlift exercise."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        vis_cfg = config.get("visualization", {})
        colors = vis_cfg.get("colors", {})

        # Colors for visualization
        self.hip_line_color = tuple(colors.get("hip_line", [255, 165, 0]))  # Orange
        self.knee_line_color = tuple(colors.get("knee_line", [0, 255, 255]))  # Cyan
        self.back_line_color = tuple(colors.get("back_line", [255, 0, 255]))  # Magenta
        self.good_form_color = tuple(colors.get("good_form", [0, 255, 0]))  # Green
        self.bad_form_color = tuple(colors.get("bad_form", [0, 0, 255]))  # Red
        self.warning_color = tuple(colors.get("warning", [0, 165, 255]))  # Orange
        self.text_color = tuple(colors.get("text", [255, 255, 255]))  # White
        self.progress_bg = tuple(colors.get("progress_bg", [50, 50, 50]))

    def draw_overlay(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        kpt_indices: Dict[str, int],
        deadlift_data: Dict[str, Any],
    ) -> np.ndarray:
        """Draw deadlift-specific visualization overlay."""
        h, w = frame.shape[:2]

        # Draw angle lines
        frame = self._draw_angle_lines(frame, keypoints, kpt_indices, deadlift_data)

        # Draw stats panel
        self._draw_stats_panel(frame, deadlift_data)

        # Draw ROM progress bar
        self._draw_rom_progress(frame, deadlift_data)

        # Draw form issues
        if deadlift_data.get("current_issues"):
            self._draw_issues(frame, deadlift_data["current_issues"])

        return frame

    def _draw_angle_lines(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        kpt_indices: Dict[str, int],
        deadlift_data: Dict[str, Any],
    ) -> np.ndarray:
        """Draw lines showing hip hinge and knee angles."""
        h, w = frame.shape[:2]

        try:
            # Get keypoints
            shoulder = keypoints[kpt_indices["shoulder"]][:2] * np.array([w, h])
            hip = keypoints[kpt_indices["hip"]][:2] * np.array([w, h])
            knee = keypoints[kpt_indices["knee"]][:2] * np.array([w, h])
            ankle = keypoints[kpt_indices["ankle"]][:2] * np.array([w, h])

            # Draw hip angle (shoulder-hip-knee)
            pts_hip = np.array([shoulder, hip, knee], dtype=np.int32)
            cv2.polylines(frame, [pts_hip], False, self.hip_line_color, 3)

            # Draw knee angle (hip-knee-ankle)
            pts_knee = np.array([hip, knee, ankle], dtype=np.int32)
            cv2.polylines(frame, [pts_knee], False, self.knee_line_color, 3)

            # Draw keypoint circles
            for pt, color in [
                (shoulder, self.back_line_color),
                (hip, self.hip_line_color),
                (knee, self.knee_line_color),
                (ankle, self.knee_line_color),
            ]:
                cv2.circle(frame, tuple(pt.astype(int)), 8, color, -1)
                cv2.circle(frame, tuple(pt.astype(int)), 10, (255, 255, 255), 2)

            # Draw angle labels
            hip_angle = deadlift_data.get("hip_angle", 0)
            knee_angle = deadlift_data.get("knee_angle", 0)

            cv2.putText(
                frame,
                f"{hip_angle:.0f}°",
                tuple((hip + np.array([15, -10])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.hip_line_color,
                2,
            )

            cv2.putText(
                frame,
                f"{knee_angle:.0f}°",
                tuple((knee + np.array([15, -10])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.knee_line_color,
                2,
            )

        except Exception as e:
            logger.debug(f"Error drawing angle lines: {e}")

        return frame

    def _draw_stats_panel(self, frame: np.ndarray, deadlift_data: Dict[str, Any]) -> None:
        """Draw statistics panel on frame."""
        h, w = frame.shape[:2]

        # Panel dimensions
        panel_h = 200
        panel_w = 280
        panel_x = w - panel_w - 20
        panel_y = 20

        # Background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Border
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            self.hip_line_color,
            2,
        )

        # Title
        cv2.putText(
            frame,
            "DEADLIFT",
            (panel_x + 10, panel_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.hip_line_color,
            2,
        )

        # State
        state = deadlift_data.get("state", "standing").upper()
        cv2.putText(
            frame,
            f"State: {state}",
            (panel_x + 10, panel_y + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.text_color,
            1,
        )

        # Reps
        reps = deadlift_data.get("reps", 0)
        cv2.putText(
            frame,
            f"Reps: {reps}",
            (panel_x + 10, panel_y + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.good_form_color,
            2,
        )

        # Hip angle
        hip_angle = deadlift_data.get("hip_angle", 0)
        cv2.putText(
            frame,
            f"Hip Angle: {hip_angle:.1f}°",
            (panel_x + 10, panel_y + 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.hip_line_color,
            1,
        )

        # Knee angle
        knee_angle = deadlift_data.get("knee_angle", 0)
        cv2.putText(
            frame,
            f"Knee Angle: {knee_angle:.1f}°",
            (panel_x + 10, panel_y + 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.knee_line_color,
            1,
        )

        # Back curvature
        back_curvature = deadlift_data.get("back_curvature", 0)
        curvature_color = (
            self.good_form_color if back_curvature < 0.15 else
            self.warning_color if back_curvature < 0.25 else
            self.bad_form_color
        )
        cv2.putText(
            frame,
            f"Back Curve: {back_curvature:.2f}",
            (panel_x + 10, panel_y + 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            curvature_color,
            1,
        )

        # ROM
        rom = deadlift_data.get("rom_percentage", 0)
        cv2.putText(
            frame,
            f"ROM: {rom:.0f}%",
            (panel_x + 10, panel_y + 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.text_color,
            1,
        )

    def _draw_rom_progress(self, frame: np.ndarray, deadlift_data: Dict[str, Any]) -> None:
        """Draw ROM progress bar."""
        h, w = frame.shape[:2]

        # Progress bar dimensions
        bar_w = 30
        bar_h = 200
        bar_x = 20
        bar_y = (h - bar_h) // 2

        # Background
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            self.progress_bg,
            -1,
        )

        # ROM fill
        rom = deadlift_data.get("rom_percentage", 0) / 100.0
        fill_h = int(bar_h * rom)
        
        # Color based on ROM
        if rom < 0.5:
            fill_color = self.warning_color
        elif rom < 0.7:
            fill_color = (0, 200, 255)  # Yellow-ish
        else:
            fill_color = self.good_form_color

        cv2.rectangle(
            frame,
            (bar_x, bar_y + bar_h - fill_h),
            (bar_x + bar_w, bar_y + bar_h),
            fill_color,
            -1,
        )

        # Border
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            (255, 255, 255),
            2,
        )

        # Label
        cv2.putText(
            frame,
            "ROM",
            (bar_x - 5, bar_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.text_color,
            1,
        )

    def _draw_issues(self, frame: np.ndarray, issues: List[str]) -> None:
        """Draw current form issues."""
        h, w = frame.shape[:2]

        y_offset = h - 30 - (len(issues) * 25)
        for issue in issues:
            cv2.putText(
                frame,
                f"⚠ {issue}",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.bad_form_color,
                2,
            )
            y_offset += 25


class DeadliftExercise(DualCameraExerciseMixin):
    """
    Deadlift exercise analyzer with back curvature monitoring.
    
    Tracks hip hinge movement pattern, uses segmentation for
    spine curvature analysis, and provides real-time form feedback.
    Supports dual-camera for knee valgus and stance width detection.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        fps: float = 30.0,
        segmenter: Any = None,
    ):
        self.config = config
        self.fps = fps
        self.segmenter = segmenter

        # Experience level
        self.experience_level = config.get("experience", {}).get("level", "intermediate")
        self.thresholds = self._build_thresholds(self.experience_level)

        # Keypoint indices
        self.facing_side = None
        self.kpt_indices = self._get_keypoint_indices()
        
        # Dual camera support
        DualCameraExerciseMixin.__init__(self)
        self.dual_camera_enabled = False
        self.front_view_keypoints = None
        self.last_front_analysis = None
        
        # Dual camera form issues (detected from front view)
        self.knee_valgus_detected = False
        self.stance_issue_detected = False

        # State machine
        self.state = "standing"  # standing, hinging_down, bottom, coming_up
        self.prev_state = "standing"

        # Rep counting
        self.rep_count = 0
        self.successful_reps = 0
        self.failed_reps = 0

        # Angle tracking
        self.current_hip_angle = 180.0
        self.current_knee_angle = 180.0
        self.current_torso_angle = 90.0
        self.current_shin_angle = 0.0
        self.min_hip_angle = 180.0
        self.max_hip_angle = 0.0

        # Barbell tracking (proxy using wrists)
        self.current_bar_x = 0.0
        self.current_bar_y = 0.0
        self.rep_start_bar_x = 0.0
        
        # Velocity and Sync Tracking
        self.hip_y_history = []
        self.shoulder_y_history = []
        self.sync_issues_detected = 0

        # ROM tracking
        self.rom_baseline_set = False
        self.standing_hip_angle = 170.0
        self.bottom_hip_angle = 90.0

        # Back curvature tracking
        self.current_back_curvature = 0.0
        self.max_back_curvature = 0.0
        self.back_curvature_history: List[float] = []

        # Issue detection
        self.current_issues: List[str] = []
        self.issue_counts: Dict[str, int] = {
            "rounded_back": 0,
            "knees_caving": 0,
            "early_hip_extension": 0,
            "lockout_incomplete": 0,
        }

        # Timing
        self.rep_start_time: Optional[float] = None
        self.rep_durations: List[float] = []

        # Voice feedback
        self.voice_messages: List[Tuple[float, str]] = []
        self.last_voice_time = 0.0
        self.voice_cooldown = 3.0
        self.session_start_time = time.time()

        # Frame tracking
        self.frame_count = 0

        # Rep history
        self.rep_history: List[Dict[str, Any]] = []

        # Visualizer
        self.visualizer = DeadliftVisualizer(config)

        logger.info(f"DeadliftExercise initialized with level: {self.experience_level}")

    def _get_keypoint_indices(self) -> Dict[str, int]:
        """Get keypoint indices based on facing side."""
        kpt_cfg = self.config.get("keypoints", {})
        facing_side = self.facing_side or "right"

        if facing_side == "left":
            return {
                "nose": kpt_cfg.get("nose", 0),
                "shoulder": kpt_cfg.get("left_shoulder", 5),
                "hip": kpt_cfg.get("left_hip", 11),
                "knee": kpt_cfg.get("left_knee", 13),
                "ankle": kpt_cfg.get("left_ankle", 15),
            }
        else:
            return {
                "nose": kpt_cfg.get("nose", 0),
                "shoulder": kpt_cfg.get("right_shoulder", 6),
                "hip": kpt_cfg.get("right_hip", 12),
                "knee": kpt_cfg.get("right_knee", 14),
                "ankle": kpt_cfg.get("right_ankle", 16),
                "wrist": kpt_cfg.get("right_wrist", 10),
            }

    def _build_thresholds(self, level: str) -> Dict[str, float]:
        """Build thresholds for deadlift based on experience level."""
        level = (level or "intermediate").lower()

        # Get deadlift-specific config
        dl_config = self.config.get("deadlift", {}) or {}
        dl_thresholds = dl_config.get("thresholds", {})
        min_rep_duration = float(dl_config.get("min_rep_duration", 1.5))

        # Get level-specific config
        level_cfg = dl_thresholds.get(level, dl_thresholds.get("intermediate", {}))

        # Hip angle thresholds
        hip_standing = float(level_cfg.get("hip_standing", 170.0))
        hip_bottom = float(level_cfg.get("hip_bottom", 90.0))
        
        # Knee angle thresholds
        knee_standing = float(level_cfg.get("knee_standing", 170.0))
        knee_bottom = float(level_cfg.get("knee_bottom", 120.0))
        
        # Back curvature threshold
        back_curvature_threshold = float(level_cfg.get("back_curvature_threshold", 0.2))
        back_curvature_warning = float(level_cfg.get("back_curvature_warning", 0.15))
        
        attempt_threshold = float(level_cfg.get("attempt_threshold", 0.30))
        success_threshold = float(level_cfg.get("success_threshold", 0.70))
        
        torso_angle_min = float(level_cfg.get("torso_angle_min", 25.0))
        torso_angle_max = float(level_cfg.get("torso_angle_max", 40.0))
        shin_angle_min = float(level_cfg.get("shin_angle_min", 5.0))
        shin_angle_max = float(level_cfg.get("shin_angle_max", 15.0))
        bar_drift_max = float(level_cfg.get("bar_drift_max", 3.0))
        sync_rise_tolerance = float(level_cfg.get("sync_rise_tolerance", 5.0))

        return {
            "hip_standing": hip_standing,
            "hip_bottom": hip_bottom,
            "knee_standing": knee_standing,
            "knee_bottom": knee_bottom,
            "back_curvature_threshold": back_curvature_threshold,
            "back_curvature_warning": back_curvature_warning,
            "attempt_threshold": attempt_threshold,
            "success_threshold": success_threshold,
            "min_rep_duration": min_rep_duration,
            "torso_angle_min": torso_angle_min,
            "torso_angle_max": torso_angle_max,
            "shin_angle_min": shin_angle_min,
            "shin_angle_max": shin_angle_max,
            "bar_drift_max": bar_drift_max,
            "sync_rise_tolerance": sync_rise_tolerance,
        }

    def set_experience_level(self, level: str) -> None:
        """Update experience level."""
        self.experience_level = level
        self.thresholds = self._build_thresholds(level)
        logger.info(f"Deadlift experience level set to: {level}")

    def set_facing_side(self, side: str) -> None:
        """Set facing side."""
        self.facing_side = side
        self.kpt_indices = self._get_keypoint_indices()
        logger.info(f"Deadlift facing side set to: {side}")

    def calculate_angle(
        self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
    ) -> float:
        """Calculate angle at p2 formed by p1-p2-p3."""
        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        return angle
        
    def calculate_horizontal_angle(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate angle of a line formed by p1-p2 relative to horizontal line."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
        return angle
        
    def calculate_vertical_angle(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate angle of a line formed by p1-p2 relative to vertical line."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if dy == 0:
            return 90.0
            
        angle = np.degrees(np.arctan(abs(dx)/abs(dy)))
        # Return positive for forward lean, negative for backward lean
        # In opencv, y increases downwards. If dx>0 (p2 is right of p1), and right is forward, it's forward lean.
        # This depends on facing side.
        if self.facing_side == "right":
            return angle if dx > 0 else -angle
        else:
            return angle if dx < 0 else -angle

    def _compute_back_curvature(
        self, frame: np.ndarray, keypoints: np.ndarray
    ) -> float:
        """Compute back curvature using segmentation."""
        if self.segmenter is None:
            return 0.0

        try:
            mask = self.segmenter.segment_person(frame)
            if mask is None:
                return 0.0

            curvature_info = compute_back_curvature(
                mask,
                keypoints[:, :2],
                self.config.get("keypoints", {}),
                self.facing_side or "right",
            )

            curvature = curvature_info.get("curvature", 0.0)
            if np.isnan(curvature):
                return 0.0

            return curvature

        except Exception as e:
            logger.debug(f"Back curvature computation error: {e}")
            return 0.0

    def _detect_issues(
        self, hip_angle: float, knee_angle: float, back_curvature: float
    ) -> List[str]:
        """Detect form issues."""
        issues = []

        # Initialize new issue counts if needed
        for key in ["knees_forward", "stiff_leg", "squatting_deadlift", "bar_drift"]:
            if key not in self.issue_counts:
                self.issue_counts[key] = 0

        # Rounded back
        if back_curvature > self.thresholds["back_curvature_threshold"]:
            issues.append("Rounded back - keep spine neutral!")
            self.issue_counts["rounded_back"] += 1
        elif back_curvature > self.thresholds["back_curvature_warning"]:
            issues.append("Watch your back position")

        # Setup checks (when not moving up yet)
        if self.state in ["hinging_down", "bottom"]:
            # Shin angle check
            if self.current_shin_angle > self.thresholds["shin_angle_max"]:
                issues.append("Knees too far forward")
                self.issue_counts["knees_forward"] += 1
            
            # Torso setup angle check
            if self.current_torso_angle < self.thresholds["torso_angle_min"]:
                issues.append("Torso too horizontal (Stiff-leg)")
                self.issue_counts["stiff_leg"] += 1
            elif self.current_torso_angle > self.thresholds["torso_angle_max"]:
                issues.append("Torso too upright (Squatting)")
                self.issue_counts["squatting_deadlift"] += 1

        # Movement checks (when lifting)
        if self.state == "coming_up":
            # Bar drift check
            if abs(self.current_bar_x - self.rep_start_bar_x) > self.thresholds["bar_drift_max"]:
                # Normalizing drift in pixels isn't ideal without calibration, 
                # but standard asks for <3 cm drift. Assuming ~30px drift threshold.
                # Here bar_drift_max from config is used as a rough proxy scaler * 10
                if abs(self.current_bar_x - self.rep_start_bar_x) > self.thresholds["bar_drift_max"] * 10:
                    issues.append("Bar drifting away from body")
                    self.issue_counts["bar_drift"] += 1

            # Sync Tracking (Hips rising faster than shoulders)
            if len(self.hip_y_history) >= 2:
                # Need vertical rise magnitude (y decreases as they move UP in pixel coords)
                hip_rise = self.hip_y_history[-2] - self.hip_y_history[-1]
                shoulder_rise = self.shoulder_y_history[-2] - self.shoulder_y_history[-1]
                
                # If both are rising
                if hip_rise > 0 and shoulder_rise > 0:
                    diff = hip_rise - shoulder_rise
                    # Using sync_rise_tolerance as a pixel diff ratio per frame check
                    if diff > self.thresholds["sync_rise_tolerance"] * 2:
                        issues.append("Hips rising too fast")
                        self.sync_issues_detected += 1
                        if self.sync_issues_detected > 3:  # Only flag if persistent over frames
                            self.issue_counts["early_hip_extension"] += 1

            # Lockout check
            if hip_angle > 160:
                if hip_angle < 170:
                    issues.append("Complete the lockout")

        return issues

    def _update_state(self, hip_angle: float, knee_angle: float) -> None:
        """Update state machine based on current angles."""
        self.prev_state = self.state

        if self.state == "standing":
            # Check if starting to hinge
            if hip_angle < 160:
                self.state = "hinging_down"
                self.rep_start_time = time.time()
                self.min_hip_angle = hip_angle
                
                # Capture starting info for rep
                self.rep_start_bar_x = self.current_bar_x
                self.hip_y_history = []
                self.shoulder_y_history = []
                self.sync_issues_detected = 0
                
                logger.debug("Deadlift: hinging_down")

        elif self.state == "hinging_down":
            # Track minimum hip angle
            self.min_hip_angle = min(self.min_hip_angle, hip_angle)

            # Check if reached bottom
            if hip_angle <= self.thresholds["hip_bottom"] + 20:
                self.state = "bottom"
                logger.debug("Deadlift: bottom")
            # Check if coming back up without reaching bottom
            elif hip_angle > self.min_hip_angle + 10:
                self.state = "coming_up"
                logger.debug("Deadlift: coming_up (early)")

        elif self.state == "bottom":
            # Check if starting to come up
            if hip_angle > self.min_hip_angle + 10:
                self.state = "coming_up"
                # Record bar X position at the absolute bottom
                self.rep_start_bar_x = self.current_bar_x
                logger.debug("Deadlift: coming_up")

        elif self.state == "coming_up":
            # Check if returned to standing
            if hip_angle >= self.thresholds["hip_standing"] - 10:
                self._complete_rep()
                self.state = "standing"
                logger.debug("Deadlift: standing (rep complete)")
            # Check if going back down
            elif hip_angle < self.min_hip_angle:
                self.state = "hinging_down"
                self.min_hip_angle = hip_angle
                logger.debug("Deadlift: back to hinging_down")

    def _complete_rep(self) -> None:
        """Complete a rep and evaluate quality."""
        current_time = time.time()

        # Calculate rep duration
        if self.rep_start_time:
            duration = current_time - self.rep_start_time
            self.rep_durations.append(duration)
        else:
            duration = 0.0

        # Check minimum duration
        if duration < self.thresholds["min_rep_duration"]:
            logger.debug(f"Rep too fast: {duration:.2f}s")
            return

        # Calculate ROM
        angle_range = self.thresholds["hip_standing"] - self.thresholds["hip_bottom"]
        achieved_range = self.thresholds["hip_standing"] - self.min_hip_angle
        rom_percentage = min(100.0, (achieved_range / angle_range) * 100)

        # Determine if successful based on ROM
        is_successful = rom_percentage >= self.thresholds["success_threshold"] * 100

        self.rep_count += 1
        if is_successful:
            self.successful_reps += 1
        else:
            self.failed_reps += 1

        # Check for back curvature issues during rep
        avg_curvature = (
            np.mean(self.back_curvature_history[-30:])
            if self.back_curvature_history
            else 0.0
        )

        # Record rep
        rep_data = {
            "rep_number": self.rep_count,
            "successful": is_successful,
            "rom_percentage": rom_percentage,
            "min_hip_angle": self.min_hip_angle,
            "duration": duration,
            "avg_back_curvature": avg_curvature,
            "max_back_curvature": self.max_back_curvature,
        }
        self.rep_history.append(rep_data)

        # Reset for next rep
        self.min_hip_angle = 180.0
        self.max_back_curvature = 0.0
        self.rep_start_time = None

        logger.info(
            f"Deadlift rep {self.rep_count}: ROM={rom_percentage:.0f}%, "
            f"Back={avg_curvature:.2f}, Success={is_successful}"
        )

    def _trigger_voice_feedback(self, issue: str) -> None:
        """Trigger voice feedback for form issues."""
        current_time = time.time()

        if current_time - self.last_voice_time < self.voice_cooldown:
            return

        voice_map = {
            "Rounded back - keep spine neutral!": "rounded_back",
            "Watch your back position": "back_warning",
            "Complete the lockout": "lockout",
        }

        for pattern, voice_key in voice_map.items():
            if pattern in issue:
                self.voice_messages.append(
                    (current_time - self.session_start_time, voice_key)
                )
                self.last_voice_time = current_time
                break

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        keypoints: Optional[np.ndarray] = None,
        results: Any = None,
    ) -> np.ndarray:
        """Process a frame for deadlift analysis."""
        self.frame_count = frame_number

        if keypoints is None or len(keypoints) == 0:
            return frame

        h, w = frame.shape[:2]

        try:
            # Get keypoint positions
            shoulder = keypoints[self.kpt_indices["shoulder"]][:2] * np.array([w, h])
            hip = keypoints[self.kpt_indices["hip"]][:2] * np.array([w, h])
            knee = keypoints[self.kpt_indices["knee"]][:2] * np.array([w, h])
            ankle = keypoints[self.kpt_indices["ankle"]][:2] * np.array([w, h])

            # Calculate angles
            hip_angle = self.calculate_angle(shoulder, hip, knee)
            knee_angle = self.calculate_angle(hip, knee, ankle)
            torso_angle = self.calculate_horizontal_angle(shoulder, hip)
            shin_angle = self.calculate_vertical_angle(knee, ankle)

            self.current_hip_angle = hip_angle
            self.current_knee_angle = knee_angle
            self.current_torso_angle = torso_angle
            self.current_shin_angle = shin_angle
            
            # Use wrist as proxy for barbell tracking
            if "wrist" in self.kpt_indices:
                wrist = keypoints[self.kpt_indices["wrist"]][:2] * np.array([w, h])
                self.current_bar_x = wrist[0]
                self.current_bar_y = wrist[1]

            # Compute back curvature
            back_curvature = self._compute_back_curvature(frame, keypoints)
            self.current_back_curvature = back_curvature
            self.back_curvature_history.append(back_curvature)
            self.max_back_curvature = max(self.max_back_curvature, back_curvature)

            # Keep history manageable
            if len(self.back_curvature_history) > 300:
                self.back_curvature_history = self.back_curvature_history[-150:]
                
            # Track vertical velocities for sync
            self.hip_y_history.append(hip[1])
            self.shoulder_y_history.append(shoulder[1])
            if len(self.hip_y_history) > 5:
                self.hip_y_history.pop(0)
                self.shoulder_y_history.pop(0)

            # Update state machine
            self._update_state(hip_angle, knee_angle)

            # Detect issues
            self.current_issues = self._detect_issues(hip_angle, knee_angle, back_curvature)

            # Trigger voice feedback for issues
            for issue in self.current_issues:
                self._trigger_voice_feedback(issue)

            # Calculate ROM percentage
            angle_range = self.thresholds["hip_standing"] - self.thresholds["hip_bottom"]
            current_range = self.thresholds["hip_standing"] - hip_angle
            rom_percentage = max(0, min(100, (current_range / angle_range) * 100))

            # Prepare visualization data
            deadlift_data = {
                "state": self.state,
                "reps": self.rep_count,
                "hip_angle": hip_angle,
                "knee_angle": knee_angle,
                "back_curvature": back_curvature,
                "rom_percentage": rom_percentage,
                "current_issues": self.current_issues,
                "successful_reps": self.successful_reps,
                "failed_reps": self.failed_reps,
            }

            # Draw overlay
            frame = self.visualizer.draw_overlay(
                frame, keypoints, self.kpt_indices, deadlift_data
            )

        except Exception as e:
            logger.error(f"Error processing deadlift frame: {e}", exc_info=True)

        return frame

    # DUAL CAMERA METHODS
    
    def process_front_frame(
        self, 
        front_keypoints: np.ndarray, 
        frame_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        """
        Process front-view frame for additional form analysis.
        
        Detects:
        - Knee valgus/varus (knees caving in/out)
        - Stance width
        - Weight distribution asymmetry
        """
        if front_keypoints is None:
            return {}
        
        kpt_config = self.config.get("keypoints", {})
        
        # Compute front-view specific metrics
        knee_valgus = self.compute_knee_valgus(front_keypoints, kpt_config, frame_shape)
        stance = self.compute_stance_width(front_keypoints, kpt_config, frame_shape)
        hip_alignment = self.compute_hip_alignment(front_keypoints, kpt_config, frame_shape)
        
        # Update form flags during active rep
        if self.state in ["hinging_down", "bottom", "lifting"]:
            if knee_valgus.get("valgus_detected"):
                self.knee_valgus_detected = True
                if "knee_valgus" not in self.current_issues:
                    self.current_issues.append("knee_valgus")
                    self.issue_counts["knee_valgus"] = self.issue_counts.get("knee_valgus", 0) + 1
        
        self.last_front_analysis = {
            "knee_valgus": knee_valgus,
            "stance": stance,
            "hip_alignment": hip_alignment,
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
        # Process side frame (primary analysis)
        processed_frame = self.process_frame(side_frame, side_keypoints, frame_number)
        
        # Process front frame for additional metrics
        front_metrics = {}
        if front_keypoints is not None:
            front_metrics = self.process_front_frame(
                front_keypoints, 
                front_frame.shape[:2],
            )
        
        # Combine metadata
        metadata = self.get_metadata()
        metadata["front_view_metrics"] = front_metrics
        metadata["dual_camera_active"] = True
        
        return processed_frame, metadata

    def get_results(self) -> Dict[str, Any]:
        """Get deadlift session results."""
        avg_curvature = (
            np.mean(self.back_curvature_history) if self.back_curvature_history else 0.0
        )
        avg_duration = np.mean(self.rep_durations) if self.rep_durations else 0.0

        return {
            "exercise_type": "deadlift",
            "total_reps": self.rep_count,
            "successful_reps": self.successful_reps,
            "failed_reps": self.failed_reps,
            "accuracy": (
                (self.successful_reps / max(1, self.rep_count)) * 100
            ),
            "avg_back_curvature": round(avg_curvature, 3),
            "issue_counts": self.issue_counts,
            "avg_rep_duration": round(avg_duration, 2),
            "rep_history": self.rep_history,
            "dual_camera_enabled": self.dual_camera_enabled,
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get current frame metadata."""
        return {
            "exercise_type": "deadlift",
            "state": self.state,
            "total_reps": self.rep_count,
            "successful_reps": self.successful_reps,
            "hip_angle": round(self.current_hip_angle, 1),
            "knee_angle": round(self.current_knee_angle, 1),
            "back_curvature": round(self.current_back_curvature, 3),
            "current_issues": self.current_issues,
        }

    def get_voice_messages(self) -> List[Tuple[float, str]]:
        """Get recorded voice messages."""
        return self.voice_messages

    def finalize_analysis(self, output_dir: str, timestamp: str = None) -> None:
        """Finalize analysis."""
        logger.info(f"Deadlift analysis finalized. Total reps: {self.rep_count}")

    def get_exercise_name(self) -> str:
        """Return exercise name."""
        return "deadlift"

    def get_anthropometrics(self) -> Optional[Dict[str, Any]]:
        """Return None - no anthropometrics computed for deadlift."""
        return None

    def reset(self) -> None:
        """Reset exercise state."""
        self.state = "standing"
        self.prev_state = "standing"
        self.rep_count = 0
        self.successful_reps = 0
        self.failed_reps = 0
        self.current_hip_angle = 180.0
        self.current_knee_angle = 180.0
        self.min_hip_angle = 180.0
        self.current_back_curvature = 0.0
        self.max_back_curvature = 0.0
        self.back_curvature_history = []
        self.current_issues = []
        self.issue_counts = {k: 0 for k in self.issue_counts}
        self.rep_history = []
        self.rep_durations = []
        self.voice_messages = []
        self.frame_count = 0
        logger.info("Deadlift exercise reset")
