"""
Overhead Press Exercise Analyzer

This module implements overhead press analysis with:
- Shoulder and elbow angle tracking
- Lockout detection at top position
- Back arch monitoring
- Core engagement tracking
- Rep counting and form feedback
- Dual-camera support for arm symmetry and lockout detection
"""

import time
import math
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

from engine.exercises.dual_camera_mixin import DualCameraExerciseMixin


class OverheadPressVisualizer:
    """Visualization helper for overhead press exercise."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        vis_cfg = config.get("visualization", {})
        colors = vis_cfg.get("colors", {})

        # Colors
        self.shoulder_line_color = tuple(colors.get("shoulder_line", [255, 215, 0]))  # Gold
        self.elbow_line_color = tuple(colors.get("elbow_line", [0, 255, 255]))  # Cyan
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
        press_data: Dict[str, Any],
    ) -> np.ndarray:
        """Draw overhead press visualization overlay."""
        h, w = frame.shape[:2]

        # Draw arm lines and angles
        frame = self._draw_arm_lines(frame, keypoints, kpt_indices, press_data)

        # Draw stats panel
        self._draw_stats_panel(frame, press_data)

        # Draw ROM progress bar
        self._draw_rom_progress(frame, press_data)

        # Draw form issues
        if press_data.get("current_issues"):
            self._draw_issues(frame, press_data["current_issues"])

        return frame

    def _draw_arm_lines(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        kpt_indices: Dict[str, int],
        press_data: Dict[str, Any],
    ) -> np.ndarray:
        """Draw arm lines and shoulder angle."""
        h, w = frame.shape[:2]

        try:
            shoulder = keypoints[kpt_indices["shoulder"]][:2] * np.array([w, h])
            elbow = keypoints[kpt_indices["elbow"]][:2] * np.array([w, h])
            wrist = keypoints[kpt_indices["wrist"]][:2] * np.array([w, h])
            hip = keypoints[kpt_indices["hip"]][:2] * np.array([w, h])

            # Draw shoulder to hip line
            cv2.line(frame, tuple(shoulder.astype(int)), tuple(hip.astype(int)),
                     self.shoulder_line_color, 3)

            # Draw arm lines (shoulder-elbow-wrist)
            pts_arm = np.array([shoulder, elbow, wrist], dtype=np.int32)
            cv2.polylines(frame, [pts_arm], False, self.elbow_line_color, 3)

            # Draw keypoint circles
            for pt, color in [
                (shoulder, self.shoulder_line_color),
                (elbow, self.elbow_line_color),
                (wrist, self.elbow_line_color),
            ]:
                cv2.circle(frame, tuple(pt.astype(int)), 8, color, -1)
                cv2.circle(frame, tuple(pt.astype(int)), 10, (255, 255, 255), 2)

            # Draw angle labels
            elbow_angle = press_data.get("elbow_angle", 0)
            shoulder_angle = press_data.get("shoulder_angle", 0)

            cv2.putText(
                frame,
                f"Elbow: {elbow_angle:.0f}°",
                tuple((elbow + np.array([15, -10])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.elbow_line_color,
                2,
            )

            cv2.putText(
                frame,
                f"Shoulder: {shoulder_angle:.0f}°",
                tuple((shoulder + np.array([15, 15])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.shoulder_line_color,
                2,
            )

        except Exception as e:
            logger.debug(f"Error drawing arm lines: {e}")

        return frame

    def _draw_stats_panel(self, frame: np.ndarray, press_data: Dict[str, Any]) -> None:
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
                      self.shoulder_line_color, 2)

        # Title
        cv2.putText(frame, "OVERHEAD PRESS", (panel_x + 10, panel_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.shoulder_line_color, 2)

        # State
        state = press_data.get("state", "down").upper()
        cv2.putText(frame, f"State: {state}", (panel_x + 10, panel_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

        # Reps
        reps = press_data.get("reps", 0)
        cv2.putText(frame, f"Reps: {reps}", (panel_x + 10, panel_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.good_form_color, 2)

        # Success rate
        success = press_data.get("successful_reps", 0)
        total = press_data.get("total_reps", 0)
        if total > 0:
            rate = (success / total) * 100
            cv2.putText(frame, f"Success: {rate:.0f}%", (panel_x + 150, panel_y + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

        # Elbow angle
        elbow_angle = press_data.get("elbow_angle", 0)
        cv2.putText(frame, f"Elbow: {elbow_angle:.1f}°", (panel_x + 10, panel_y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.elbow_line_color, 1)

        # Shoulder angle
        shoulder_angle = press_data.get("shoulder_angle", 0)
        cv2.putText(frame, f"Shoulder: {shoulder_angle:.1f}°", (panel_x + 10, panel_y + 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.shoulder_line_color, 1)

        # ROM
        rom = press_data.get("rom_percentage", 0)
        cv2.putText(frame, f"ROM: {rom:.0f}%", (panel_x + 10, panel_y + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

    def _draw_rom_progress(self, frame: np.ndarray, press_data: Dict[str, Any]) -> None:
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
        rom = press_data.get("rom_percentage", 0) / 100.0
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

        # Label
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


class OverheadPressExercise(DualCameraExerciseMixin):
    """
    Overhead Press exercise analyzer.
    
    Tracks vertical arm movement pattern, lockout position,
    and provides real-time form feedback.
    Supports dual-camera for arm symmetry and full lockout detection.
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
        self.state = "down"  # down, pressing_up, top, lowering
        self.prev_state = "down"

        # Rep counting
        self.rep_count = 0
        self.successful_reps = 0
        self.failed_reps = 0

        # Angle tracking
        self.current_elbow_angle = 90.0
        self.current_shoulder_angle = 0.0
        self.min_elbow_angle = 180.0
        self.max_shoulder_angle = 0.0
        
        # Dual camera support
        DualCameraExerciseMixin.__init__(self)
        self.dual_camera_enabled = False
        self.front_view_keypoints = None
        self.last_front_analysis = None
        
        # Dual camera form issues (detected from front view)
        self.asymmetric_lockout_detected = False
        self.arm_asymmetry_detected = False
        self.lateral_lean_detected = False

        # ROM tracking
        self.elbow_baseline_down = 90.0
        self.elbow_baseline_up = 170.0
        self.rom_established = False

        # Issue detection
        self.current_issues: List[str] = []
        self.issue_counts: Dict[str, int] = {
            "incomplete_lockout": 0,
            "back_arch": 0,
            "elbow_flare": 0,
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

        # Velocity tracking
        self._prev_elbow_angle = 90.0
        self._elbow_velocity = 0.0

        # Visualizer
        self.visualizer = OverheadPressVisualizer(config)

        logger.info(f"OverheadPressExercise initialized with level: {self.experience_level}")

    def _get_keypoint_indices(self) -> Dict[str, int]:
        """Get keypoint indices based on facing side."""
        kpt_cfg = self.config.get("keypoints", {})
        facing_side = self.facing_side or "right"

        if facing_side == "left":
            return {
                "nose": kpt_cfg.get("nose", 0),
                "shoulder": kpt_cfg.get("left_shoulder", 5),
                "elbow": kpt_cfg.get("left_elbow", 7),
                "wrist": kpt_cfg.get("left_wrist", 9),
                "hip": kpt_cfg.get("left_hip", 11),
                "knee": kpt_cfg.get("left_knee", 13),
            }
        else:
            return {
                "nose": kpt_cfg.get("nose", 0),
                "shoulder": kpt_cfg.get("right_shoulder", 6),
                "elbow": kpt_cfg.get("right_elbow", 8),
                "wrist": kpt_cfg.get("right_wrist", 10),
                "hip": kpt_cfg.get("right_hip", 12),
                "knee": kpt_cfg.get("right_knee", 14),
            }

    def _build_thresholds(self, level: str) -> Dict[str, float]:
        """Build thresholds based on experience level from config."""
        level = (level or "intermediate").lower()

        # Get exercise-specific config
        ex_config = self.config.get("overhead_press", {}) or {}
        ex_thresholds = ex_config.get("thresholds", {})
        min_rep_duration = float(ex_config.get("min_rep_duration", 1.0))

        # Get level-specific config
        level_cfg = ex_thresholds.get(level, ex_thresholds.get("intermediate", {}))

        return {
            "elbow_down": float(level_cfg.get("elbow_down", 90.0)),
            "elbow_up": float(level_cfg.get("elbow_up", 170.0)),
            "shoulder_min": float(level_cfg.get("shoulder_min", 150.0)),
            "back_arch_threshold": float(level_cfg.get("back_arch_threshold", 15.0)),
            "attempt_threshold": float(level_cfg.get("attempt_threshold", 0.35)),
            "success_threshold": float(level_cfg.get("success_threshold", 0.70)),
            "min_rep_duration": min_rep_duration,
        }

    def set_experience_level(self, level: str) -> None:
        """Update experience level."""
        self.experience_level = level
        self.thresholds = self._build_thresholds(level)
        logger.info(f"Overhead press experience level set to: {level}")

    def set_facing_side(self, side: str) -> None:
        """Set facing side."""
        self.facing_side = side
        self.kpt_indices = self._get_keypoint_indices()
        logger.info(f"Overhead press facing side set to: {side}")

    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by p1-p2-p3."""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _calculate_shoulder_angle(
        self, shoulder: np.ndarray, elbow: np.ndarray, hip: np.ndarray
    ) -> float:
        """Calculate shoulder angle (arm relative to torso)."""
        # Angle between arm (shoulder-elbow) and torso (shoulder-hip)
        return self.calculate_angle(elbow, shoulder, hip)

    def _calculate_rom_percentage(self, elbow_angle: float) -> float:
        """Calculate ROM percentage based on elbow angle."""
        elbow_down = self.thresholds["elbow_down"]
        elbow_up = self.thresholds["elbow_up"]
        
        if elbow_up <= elbow_down:
            return 0.0
            
        rom = (elbow_angle - elbow_down) / (elbow_up - elbow_down)
        return max(0.0, min(1.0, rom)) * 100

    def _detect_issues(
        self, elbow_angle: float, shoulder_angle: float, hip: np.ndarray, shoulder: np.ndarray
    ) -> List[str]:
        """Detect form issues."""
        issues = []

        # Incomplete lockout
        if self.state == "top" and elbow_angle < self.thresholds["elbow_up"] - 10:
            issues.append("Incomplete lockout - extend fully!")
            self.issue_counts["incomplete_lockout"] += 1

        # Back arch detection (simplified - would need spine tracking for accuracy)
        # Check if shoulder is significantly behind hip
        if shoulder[0] < hip[0] - self.thresholds["back_arch_threshold"]:
            issues.append("Excessive back arch")
            self.issue_counts["back_arch"] += 1

        return issues

    def _update_state(self, elbow_angle: float, shoulder_angle: float) -> None:
        """Update state machine based on current angles."""
        self.prev_state = self.state

        # Calculate velocity
        self._elbow_velocity = elbow_angle - self._prev_elbow_angle
        self._prev_elbow_angle = elbow_angle

        if self.state == "down":
            # Check if starting to press
            if self._elbow_velocity > 2.0 and elbow_angle > self.thresholds["elbow_down"] + 10:
                self.state = "pressing_up"
                self.rep_start_time = time.time()
                self.min_elbow_angle = elbow_angle
                self.max_shoulder_angle = shoulder_angle
                logger.debug("Overhead press: pressing_up")

        elif self.state == "pressing_up":
            self.max_shoulder_angle = max(self.max_shoulder_angle, shoulder_angle)

            # Check if reached top
            if elbow_angle >= self.thresholds["elbow_up"] - 10:
                self.state = "top"
                logger.debug("Overhead press: top")
            # Check if lowering early
            elif self._elbow_velocity < -2.0:
                self.state = "lowering"
                logger.debug("Overhead press: lowering (early)")

        elif self.state == "top":
            # Check if starting to lower
            if self._elbow_velocity < -2.0:
                self.state = "lowering"
                logger.debug("Overhead press: lowering")

        elif self.state == "lowering":
            # Check if returned to bottom
            if elbow_angle <= self.thresholds["elbow_down"] + 10:
                self._complete_rep(elbow_angle, shoulder_angle)
                self.state = "down"
                logger.debug("Overhead press: down (rep complete)")

    def _complete_rep(self, elbow_angle: float, shoulder_angle: float) -> None:
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
            return

        # Calculate ROM
        rom_percentage = self._calculate_rom_percentage(self.max_shoulder_angle)

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
            "max_shoulder_angle": self.max_shoulder_angle,
            "duration": duration,
        }
        self.rep_history.append(rep_data)

        # Reset
        self.min_elbow_angle = 180.0
        self.max_shoulder_angle = 0.0
        self.rep_start_time = None

        logger.info(f"Overhead press rep {self.rep_count}: ROM={rom_percentage:.0f}%, Success={is_successful}")

    def _trigger_voice_feedback(self, issue: str) -> None:
        """Trigger voice feedback."""
        current_time = time.time()

        if current_time - self.last_voice_time < self.voice_cooldown:
            return

        voice_map = {
            "Incomplete lockout - extend fully!": "lockout",
            "Excessive back arch": "back_arch",
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
        """Process a frame for overhead press analysis."""
        self.frame_count = frame_number

        if keypoints is None or len(keypoints) == 0:
            return frame

        h, w = frame.shape[:2]

        try:
            # Get keypoint positions
            shoulder = keypoints[self.kpt_indices["shoulder"]][:2] * np.array([w, h])
            elbow = keypoints[self.kpt_indices["elbow"]][:2] * np.array([w, h])
            wrist = keypoints[self.kpt_indices["wrist"]][:2] * np.array([w, h])
            hip = keypoints[self.kpt_indices["hip"]][:2] * np.array([w, h])

            # Calculate angles
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            shoulder_angle = self._calculate_shoulder_angle(shoulder, elbow, hip)

            self.current_elbow_angle = elbow_angle
            self.current_shoulder_angle = shoulder_angle

            # Update state machine
            self._update_state(elbow_angle, shoulder_angle)

            # Detect issues
            self.current_issues = self._detect_issues(elbow_angle, shoulder_angle, hip, shoulder)

            # Trigger voice feedback
            for issue in self.current_issues:
                self._trigger_voice_feedback(issue)

            # Calculate ROM
            rom_percentage = self._calculate_rom_percentage(elbow_angle)

            # Prepare visualization data
            press_data = {
                "state": self.state,
                "reps": self.rep_count,
                "successful_reps": self.successful_reps,
                "total_reps": self.rep_count,
                "elbow_angle": elbow_angle,
                "shoulder_angle": shoulder_angle,
                "rom_percentage": rom_percentage,
                "current_issues": self.current_issues,
            }

            # Draw overlay
            frame = self.visualizer.draw_overlay(frame, keypoints, self.kpt_indices, press_data)

        except Exception as e:
            logger.error(f"Error processing overhead press frame: {e}", exc_info=True)

        return frame

    # DUAL CAMERA METHODS
    
    def process_front_frame(
        self, 
        front_keypoints: np.ndarray, 
        frame_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        """
        Process front-view frame for overhead press specific analysis.
        
        Detects:
        - Arm symmetry (both arms pressing evenly)
        - Lockout symmetry (both arms fully locked out)
        - Lateral lean (body leaning to one side)
        """
        if front_keypoints is None:
            return {}
        
        kpt_config = self.config.get("keypoints", {})
        h, w = frame_shape[:2]
        
        # Compute front-view specific metrics
        arm_symmetry = self.compute_arm_symmetry(front_keypoints, kpt_config, frame_shape)
        shoulder_alignment = self.compute_shoulder_alignment(front_keypoints, kpt_config, frame_shape)
        
        # Calculate individual arm angles from front view
        try:
            # Get keypoint indices
            l_shoulder_idx = kpt_config["left_shoulder"]
            r_shoulder_idx = kpt_config["right_shoulder"]
            l_elbow_idx = kpt_config["left_elbow"]
            r_elbow_idx = kpt_config["right_elbow"]
            l_wrist_idx = kpt_config["left_wrist"]
            r_wrist_idx = kpt_config["right_wrist"]
            
            if front_keypoints.max() <= 1.0:
                scale = np.array([w, h])
                l_shoulder = front_keypoints[l_shoulder_idx][:2] * scale
                r_shoulder = front_keypoints[r_shoulder_idx][:2] * scale
                l_elbow = front_keypoints[l_elbow_idx][:2] * scale
                r_elbow = front_keypoints[r_elbow_idx][:2] * scale
                l_wrist = front_keypoints[l_wrist_idx][:2] * scale
                r_wrist = front_keypoints[r_wrist_idx][:2] * scale
            else:
                l_shoulder = front_keypoints[l_shoulder_idx][:2]
                r_shoulder = front_keypoints[r_shoulder_idx][:2]
                l_elbow = front_keypoints[l_elbow_idx][:2]
                r_elbow = front_keypoints[r_elbow_idx][:2]
                l_wrist = front_keypoints[l_wrist_idx][:2]
                r_wrist = front_keypoints[r_wrist_idx][:2]
            
            # Calculate elbow angles
            left_elbow_angle = self.calculate_angle_3point(l_shoulder, l_elbow, l_wrist)
            right_elbow_angle = self.calculate_angle_3point(r_shoulder, r_elbow, r_wrist)
            
            # Check for asymmetric lockout (one arm not fully extended)
            lockout_threshold = 160.0  # degrees
            left_locked = left_elbow_angle > lockout_threshold
            right_locked = right_elbow_angle > lockout_threshold
            
            asymmetric_lockout = (left_locked != right_locked)
            
        except Exception as e:
            logger.debug(f"Error computing front view arm angles: {e}")
            left_elbow_angle = 0.0
            right_elbow_angle = 0.0
            asymmetric_lockout = False
            left_locked = False
            right_locked = False
        
        # Update form flags during active rep
        if self.state in ["pressing_up", "top"]:
            if arm_symmetry.get("asymmetric_detected"):
                self.arm_asymmetry_detected = True
                logger.debug(f"Arm asymmetry detected: score={arm_symmetry['symmetry_score']:.1f}")
            
            if asymmetric_lockout and self.state == "top":
                self.asymmetric_lockout_detected = True
                logger.debug(f"Asymmetric lockout: L={left_elbow_angle:.1f}° R={right_elbow_angle:.1f}°")
            
            if shoulder_alignment.get("shoulder_tilt_angle", 0) > 10.0:
                self.lateral_lean_detected = True
                logger.debug(f"Lateral lean detected: {shoulder_alignment['shoulder_drop_side']}")
        
        self.last_front_analysis = {
            "arm_symmetry": arm_symmetry,
            "shoulder_alignment": shoulder_alignment,
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,
            "left_locked": left_locked,
            "right_locked": right_locked,
            "asymmetric_lockout": asymmetric_lockout,
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
    
    def get_dual_camera_issues(self) -> list:
        """Get form issues detected from dual camera analysis."""
        issues = []
        
        if self.arm_asymmetry_detected:
            issues.append("arm_asymmetry")
        
        if self.asymmetric_lockout_detected:
            issues.append("asymmetric_lockout")
        
        if self.lateral_lean_detected:
            issues.append("lateral_lean")
        
        return issues
    
    def reset_dual_camera_flags(self):
        """Reset dual camera form flags at start of new rep."""
        self.asymmetric_lockout_detected = False
        self.arm_asymmetry_detected = False
        self.lateral_lean_detected = False

    def get_results(self) -> Dict[str, Any]:
        """Get session results."""
        avg_duration = np.mean(self.rep_durations) if self.rep_durations else 0.0

        return {
            "exercise_type": "overhead_press",
            "total_reps": self.rep_count,
            "successful_reps": self.successful_reps,
            "failed_reps": self.failed_reps,
            "accuracy": (self.successful_reps / max(1, self.rep_count)) * 100,
            "issue_counts": self.issue_counts,
            "avg_rep_duration": round(avg_duration, 2),
            "rep_history": self.rep_history,
            "dual_camera_enabled": self.dual_camera_enabled,
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get current frame metadata."""
        return {
            "exercise_type": "overhead_press",
            "state": self.state,
            "total_reps": self.rep_count,
            "successful_reps": self.successful_reps,
            "elbow_angle": round(self.current_elbow_angle, 1),
            "shoulder_angle": round(self.current_shoulder_angle, 1),
            "current_issues": self.current_issues,
        }

    def get_voice_messages(self) -> List[Tuple[float, str]]:
        """Get recorded voice messages."""
        return self.voice_messages

    def finalize_analysis(self, output_dir: str, timestamp: str = None) -> None:
        """Finalize analysis."""
        logger.info(f"Overhead press analysis finalized. Total reps: {self.rep_count}")

    def get_exercise_name(self) -> str:
        """Return exercise name."""
        return "overhead_press"

    def get_anthropometrics(self) -> Optional[Dict[str, Any]]:
        """Return None - no anthropometrics computed for overhead press."""
        return None

    def reset(self) -> None:
        """Reset exercise state."""
        self.state = "down"
        self.prev_state = "down"
        self.rep_count = 0
        self.successful_reps = 0
        self.failed_reps = 0
        self.current_elbow_angle = 90.0
        self.current_shoulder_angle = 0.0
        self.min_elbow_angle = 180.0
        self.max_shoulder_angle = 0.0
        self.current_issues = []
        self.issue_counts = {k: 0 for k in self.issue_counts}
        self.rep_history = []
        self.rep_durations = []
        self.voice_messages = []
        self.frame_count = 0
        logger.info("Overhead press exercise reset")
