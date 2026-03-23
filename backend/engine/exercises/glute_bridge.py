"""
Glute Bridge Exercise Analyzer

This module implements glute bridge analysis with:
- Hip extension angle tracking
- Knee angle monitoring
- Rep counting based on hip elevation
- Form feedback for proper alignment
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger


class GluteBridgeVisualizer:
    """Visualization helper for glute bridge exercise."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        vis_cfg = config.get("visualization", {})
        colors = vis_cfg.get("colors", {})

        # Colors
        self.hip_line_color = tuple(colors.get("hip_line", [255, 165, 0]))  # Orange
        self.knee_line_color = tuple(colors.get("knee_line", [0, 255, 255]))  # Cyan
        self.body_line_color = tuple(colors.get("body_line", [255, 215, 0]))  # Gold
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
        bridge_data: Dict[str, Any],
    ) -> np.ndarray:
        """Draw glute bridge visualization overlay."""
        h, w = frame.shape[:2]

        # Draw body lines
        frame = self._draw_body_lines(frame, keypoints, kpt_indices, bridge_data)

        # Draw stats panel
        self._draw_stats_panel(frame, bridge_data)

        # Draw ROM progress bar
        self._draw_rom_progress(frame, bridge_data)

        # Draw form issues
        if bridge_data.get("current_issues"):
            self._draw_issues(frame, bridge_data["current_issues"])

        return frame

    def _draw_body_lines(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        kpt_indices: Dict[str, int],
        bridge_data: Dict[str, Any],
    ) -> np.ndarray:
        """Draw body alignment lines."""
        h, w = frame.shape[:2]

        try:
            shoulder = keypoints[kpt_indices["shoulder"]][:2] * np.array([w, h])
            hip = keypoints[kpt_indices["hip"]][:2] * np.array([w, h])
            knee = keypoints[kpt_indices["knee"]][:2] * np.array([w, h])
            ankle = keypoints[kpt_indices["ankle"]][:2] * np.array([w, h])

            # Draw shoulder-hip-knee line (body alignment)
            pts_body = np.array([shoulder, hip, knee], dtype=np.int32)
            cv2.polylines(frame, [pts_body], False, self.body_line_color, 3)

            # Draw hip-knee-ankle line (leg angle)
            pts_leg = np.array([hip, knee, ankle], dtype=np.int32)
            cv2.polylines(frame, [pts_leg], False, self.knee_line_color, 3)

            # Draw keypoint circles
            for pt, color in [
                (shoulder, self.body_line_color),
                (hip, self.hip_line_color),
                (knee, self.knee_line_color),
                (ankle, self.knee_line_color),
            ]:
                cv2.circle(frame, tuple(pt.astype(int)), 8, color, -1)
                cv2.circle(frame, tuple(pt.astype(int)), 10, (255, 255, 255), 2)

            # Draw angle labels
            hip_angle = bridge_data.get("hip_angle", 0)
            knee_angle = bridge_data.get("knee_angle", 0)

            cv2.putText(
                frame, f"Hip: {hip_angle:.0f}°",
                tuple((hip + np.array([15, -10])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.hip_line_color, 2
            )

            cv2.putText(
                frame, f"Knee: {knee_angle:.0f}°",
                tuple((knee + np.array([15, -10])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.knee_line_color, 2
            )

        except Exception as e:
            logger.debug(f"Error drawing body lines: {e}")

        return frame

    def _draw_stats_panel(self, frame: np.ndarray, bridge_data: Dict[str, Any]) -> None:
        """Draw statistics panel."""
        h, w = frame.shape[:2]

        panel_h = 180
        panel_w = 280
        panel_x = w - panel_w - 20
        panel_y = 20

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                      self.hip_line_color, 2)

        # Title
        cv2.putText(frame, "GLUTE BRIDGE", (panel_x + 10, panel_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.hip_line_color, 2)

        # State
        state = bridge_data.get("state", "down").upper()
        cv2.putText(frame, f"State: {state}", (panel_x + 10, panel_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

        # Reps
        reps = bridge_data.get("reps", 0)
        cv2.putText(frame, f"Reps: {reps}", (panel_x + 10, panel_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.good_form_color, 2)

        # Hip angle
        hip_angle = bridge_data.get("hip_angle", 0)
        cv2.putText(frame, f"Hip Angle: {hip_angle:.1f}°", (panel_x + 10, panel_y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.hip_line_color, 1)

        # Knee angle
        knee_angle = bridge_data.get("knee_angle", 0)
        cv2.putText(frame, f"Knee Angle: {knee_angle:.1f}°", (panel_x + 10, panel_y + 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.knee_line_color, 1)

        # ROM
        rom = bridge_data.get("rom_percentage", 0)
        cv2.putText(frame, f"ROM: {rom:.0f}%", (panel_x + 10, panel_y + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

    def _draw_rom_progress(self, frame: np.ndarray, bridge_data: Dict[str, Any]) -> None:
        """Draw ROM progress bar."""
        h, w = frame.shape[:2]

        bar_w = 30
        bar_h = 200
        bar_x = 20
        bar_y = (h - bar_h) // 2

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      self.progress_bg, -1)

        # ROM fill
        rom = bridge_data.get("rom_percentage", 0) / 100.0
        fill_h = int(bar_h * rom)

        if rom < 0.5:
            fill_color = self.warning_color
        elif rom < 0.7:
            fill_color = (0, 200, 255)
        else:
            fill_color = self.good_form_color

        cv2.rectangle(frame, (bar_x, bar_y + bar_h - fill_h),
                      (bar_x + bar_w, bar_y + bar_h), fill_color, -1)

        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (255, 255, 255), 2)

        cv2.putText(frame, "ROM", (bar_x - 5, bar_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

    def _draw_issues(self, frame: np.ndarray, issues: List[str]) -> None:
        """Draw current form issues."""
        h, w = frame.shape[:2]

        y_offset = h - 30 - (len(issues) * 25)
        for issue in issues:
            cv2.putText(frame, f"⚠ {issue}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.bad_form_color, 2)
            y_offset += 25


class GluteBridgeExercise:
    """
    Glute Bridge exercise analyzer.
    
    Tracks hip extension movement from floor position,
    monitors body alignment, and provides form feedback.
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

        # State machine
        self.state = "down"  # down, lifting, top, lowering
        self.prev_state = "down"

        # Rep counting
        self.rep_count = 0
        self.successful_reps = 0
        self.failed_reps = 0

        # Angle tracking
        self.current_hip_angle = 90.0
        self.current_knee_angle = 90.0
        self.max_hip_angle = 0.0

        # ROM tracking
        self.hip_angle_down = 90.0
        self.hip_angle_up = 180.0

        # Issue detection
        self.current_issues: List[str] = []
        self.issue_counts: Dict[str, int] = {
            "hyperextension": 0,
            "incomplete_lift": 0,
            "knees_caving": 0,
        }

        # Timing
        self.rep_start_time: Optional[float] = None
        self.rep_durations: List[float] = []
        self.hold_time_at_top = 0.0
        self.top_start_time: Optional[float] = None

        # Voice feedback
        self.voice_messages: List[Tuple[float, str]] = []
        self.last_voice_time = 0.0
        self.voice_cooldown = 3.0
        self.session_start_time = time.time()

        # Frame tracking
        self.frame_count = 0

        # Rep history
        self.rep_history: List[Dict[str, Any]] = []

        # Velocity tracking
        self._prev_hip_angle = 90.0
        self._hip_velocity = 0.0

        # Visualizer
        self.visualizer = GluteBridgeVisualizer(config)

        logger.info(f"GluteBridgeExercise initialized with level: {self.experience_level}")

    def _get_keypoint_indices(self) -> Dict[str, int]:
        """Get keypoint indices based on facing side."""
        kpt_cfg = self.config.get("keypoints", {})
        facing_side = self.facing_side or "right"

        if facing_side == "left":
            return {
                "shoulder": kpt_cfg.get("left_shoulder", 5),
                "hip": kpt_cfg.get("left_hip", 11),
                "knee": kpt_cfg.get("left_knee", 13),
                "ankle": kpt_cfg.get("left_ankle", 15),
            }
        else:
            return {
                "shoulder": kpt_cfg.get("right_shoulder", 6),
                "hip": kpt_cfg.get("right_hip", 12),
                "knee": kpt_cfg.get("right_knee", 14),
                "ankle": kpt_cfg.get("right_ankle", 16),
            }

    def _build_thresholds(self, level: str) -> Dict[str, float]:
        """Build thresholds based on experience level from config."""
        level = (level or "intermediate").lower()

        # Get exercise-specific config
        ex_config = self.config.get("glute_bridge", {}) or {}
        ex_thresholds = ex_config.get("thresholds", {})
        min_rep_duration = float(ex_config.get("min_rep_duration", 1.0))

        # Get level-specific config
        level_cfg = ex_thresholds.get(level, ex_thresholds.get("intermediate", {}))

        return {
            "hip_angle_down": float(level_cfg.get("hip_angle_down", 90.0)),
            "hip_angle_up": float(level_cfg.get("hip_angle_up", 170.0)),
            "knee_angle_min": float(level_cfg.get("knee_angle_min", 80.0)),
            "knee_angle_max": float(level_cfg.get("knee_angle_max", 100.0)),
            "hyperextension_threshold": float(level_cfg.get("hyperextension_threshold", 185.0)),
            "attempt_threshold": float(level_cfg.get("attempt_threshold", 0.35)),
            "success_threshold": float(level_cfg.get("success_threshold", 0.70)),
            "min_rep_duration": min_rep_duration,
        }

    def set_experience_level(self, level: str) -> None:
        """Update experience level."""
        self.experience_level = level
        self.thresholds = self._build_thresholds(level)
        logger.info(f"Glute bridge experience level set to: {level}")

    def set_facing_side(self, side: str) -> None:
        """Set facing side."""
        self.facing_side = side
        self.kpt_indices = self._get_keypoint_indices()
        logger.info(f"Glute bridge facing side set to: {side}")

    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by p1-p2-p3."""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _calculate_rom_percentage(self, hip_angle: float) -> float:
        """Calculate ROM percentage based on hip angle."""
        hip_down = self.thresholds["hip_angle_down"]
        hip_up = self.thresholds["hip_angle_up"]

        if hip_up <= hip_down:
            return 0.0

        rom = (hip_angle - hip_down) / (hip_up - hip_down)
        return max(0.0, min(1.0, rom)) * 100

    def _detect_issues(self, hip_angle: float, knee_angle: float) -> List[str]:
        """Detect form issues."""
        issues = []

        # Hyperextension at top
        if hip_angle > self.thresholds["hyperextension_threshold"]:
            issues.append("Hyperextension - don't over-arch!")
            self.issue_counts["hyperextension"] += 1

        # Knee angle check
        if knee_angle < self.thresholds["knee_angle_min"]:
            issues.append("Knees too bent")
        elif knee_angle > self.thresholds["knee_angle_max"]:
            issues.append("Knees too extended")

        return issues

    def _update_state(self, hip_angle: float, knee_angle: float) -> None:
        """Update state machine."""
        self.prev_state = self.state

        # Calculate velocity
        self._hip_velocity = hip_angle - self._prev_hip_angle
        self._prev_hip_angle = hip_angle

        current_time = time.time()

        if self.state == "down":
            # Check if starting to lift
            if self._hip_velocity > 2.0 and hip_angle > self.thresholds["hip_angle_down"] + 10:
                self.state = "lifting"
                self.rep_start_time = current_time
                self.max_hip_angle = hip_angle
                logger.debug("Glute bridge: lifting")

        elif self.state == "lifting":
            self.max_hip_angle = max(self.max_hip_angle, hip_angle)

            # Check if reached top
            if hip_angle >= self.thresholds["hip_angle_up"] - 10:
                self.state = "top"
                self.top_start_time = current_time
                logger.debug("Glute bridge: top")
            # Check if lowering early
            elif self._hip_velocity < -2.0:
                self.state = "lowering"
                logger.debug("Glute bridge: lowering (early)")

        elif self.state == "top":
            # Track hold time
            if self.top_start_time:
                self.hold_time_at_top = current_time - self.top_start_time

            # Check if starting to lower
            if self._hip_velocity < -2.0:
                self.state = "lowering"
                logger.debug("Glute bridge: lowering")

        elif self.state == "lowering":
            # Check if returned to bottom
            if hip_angle <= self.thresholds["hip_angle_down"] + 10:
                self._complete_rep(hip_angle)
                self.state = "down"
                logger.debug("Glute bridge: down (rep complete)")

    def _complete_rep(self, hip_angle: float) -> None:
        """Complete a rep and evaluate quality."""
        current_time = time.time()

        # Calculate duration
        duration = 0.0
        if self.rep_start_time:
            duration = current_time - self.rep_start_time
            self.rep_durations.append(duration)

        # Check minimum duration
        if duration < self.thresholds["min_rep_duration"]:
            logger.debug(f"Rep too fast: {duration:.2f}s")
            self._reset_rep()
            return

        # Calculate ROM
        rom_percentage = self._calculate_rom_percentage(self.max_hip_angle)

        # Determine success
        is_successful = rom_percentage >= self.thresholds["success_threshold"] * 100

        self.rep_count += 1
        if is_successful:
            self.successful_reps += 1
        else:
            self.failed_reps += 1

        # Record rep
        rep_data = {
            "rep_number": self.rep_count,
            "successful": is_successful,
            "rom_percentage": rom_percentage,
            "max_hip_angle": self.max_hip_angle,
            "hold_time_at_top": self.hold_time_at_top,
            "duration": duration,
        }
        self.rep_history.append(rep_data)

        logger.info(f"Glute bridge rep {self.rep_count}: ROM={rom_percentage:.0f}%, Hold={self.hold_time_at_top:.1f}s, Success={is_successful}")

        self._reset_rep()

    def _reset_rep(self) -> None:
        """Reset rep tracking variables."""
        self.max_hip_angle = 0.0
        self.hold_time_at_top = 0.0
        self.rep_start_time = None
        self.top_start_time = None

    def _trigger_voice_feedback(self, issue: str) -> None:
        """Trigger voice feedback."""
        current_time = time.time()

        if current_time - self.last_voice_time < self.voice_cooldown:
            return

        voice_map = {
            "Hyperextension - don't over-arch!": "hyperextension",
            "Knees too bent": "knee_position",
            "Knees too extended": "knee_position",
        }

        for pattern, voice_key in voice_map.items():
            if pattern in issue:
                self.voice_messages.append((current_time - self.session_start_time, voice_key))
                self.last_voice_time = current_time
                break

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        keypoints: Optional[np.ndarray] = None,
        results: Any = None,
    ) -> np.ndarray:
        """Process a frame for glute bridge analysis."""
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

            self.current_hip_angle = hip_angle
            self.current_knee_angle = knee_angle

            # Update state machine
            self._update_state(hip_angle, knee_angle)

            # Detect issues
            self.current_issues = self._detect_issues(hip_angle, knee_angle)

            # Trigger voice feedback
            for issue in self.current_issues:
                self._trigger_voice_feedback(issue)

            # Calculate ROM
            rom_percentage = self._calculate_rom_percentage(hip_angle)

            # Prepare visualization data
            bridge_data = {
                "state": self.state,
                "reps": self.rep_count,
                "successful_reps": self.successful_reps,
                "hip_angle": hip_angle,
                "knee_angle": knee_angle,
                "rom_percentage": rom_percentage,
                "current_issues": self.current_issues,
            }

            # Draw overlay
            frame = self.visualizer.draw_overlay(frame, keypoints, self.kpt_indices, bridge_data)

        except Exception as e:
            logger.error(f"Error processing glute bridge frame: {e}", exc_info=True)

        return frame

    def get_results(self) -> Dict[str, Any]:
        """Get session results."""
        avg_duration = np.mean(self.rep_durations) if self.rep_durations else 0.0
        avg_hold = np.mean([r.get("hold_time_at_top", 0) for r in self.rep_history]) if self.rep_history else 0.0

        return {
            "exercise_type": "glute_bridge",
            "total_reps": self.rep_count,
            "successful_reps": self.successful_reps,
            "failed_reps": self.failed_reps,
            "accuracy": (self.successful_reps / max(1, self.rep_count)) * 100,
            "avg_hold_time": round(avg_hold, 2),
            "issue_counts": self.issue_counts,
            "avg_rep_duration": round(avg_duration, 2),
            "rep_history": self.rep_history,
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get current frame metadata."""
        return {
            "exercise_type": "glute_bridge",
            "state": self.state,
            "total_reps": self.rep_count,
            "successful_reps": self.successful_reps,
            "hip_angle": round(self.current_hip_angle, 1),
            "knee_angle": round(self.current_knee_angle, 1),
            "current_issues": self.current_issues,
        }

    def get_voice_messages(self) -> List[Tuple[float, str]]:
        """Get recorded voice messages."""
        return self.voice_messages

    def finalize_analysis(self, output_dir: str, timestamp: str = None) -> None:
        """Finalize analysis."""
        logger.info(f"Glute bridge analysis finalized. Total reps: {self.rep_count}")

    def get_exercise_name(self) -> str:
        """Return exercise name."""
        return "glute_bridge"

    def get_anthropometrics(self) -> Optional[Dict[str, Any]]:
        """Return None - no anthropometrics computed for glute bridge."""
        return None

    def reset(self) -> None:
        """Reset exercise state."""
        self.state = "down"
        self.prev_state = "down"
        self.rep_count = 0
        self.successful_reps = 0
        self.failed_reps = 0
        self.current_hip_angle = 90.0
        self.current_knee_angle = 90.0
        self.max_hip_angle = 0.0
        self.current_issues = []
        self.issue_counts = {k: 0 for k in self.issue_counts}
        self.rep_history = []
        self.rep_durations = []
        self.voice_messages = []
        self.frame_count = 0
        self._reset_rep()
        logger.info("Glute bridge exercise reset")
