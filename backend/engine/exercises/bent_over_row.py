"""
Bent-Over Row Exercise Analyzer

This module implements bent-over row analysis with:
- Elbow angle tracking for ROM
- Back curvature monitoring using segmentation
- Hip hinge position maintenance
- Rep counting and form feedback
- Dual-camera support for arm symmetry detection (front view)
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

from engine.core.utils import compute_back_curvature
from engine.exercises.dual_camera_mixin import DualCameraExerciseMixin


class BentOverRowVisualizer:
    """Visualization helper for bent-over row exercise."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        vis_cfg = config.get("visualization", {})
        colors = vis_cfg.get("colors", {})

        # Colors
        self.elbow_line_color = tuple(colors.get("elbow_line", [0, 255, 255]))  # Cyan
        self.back_line_color = tuple(colors.get("back_line", [255, 0, 255]))  # Magenta
        self.hip_line_color = tuple(colors.get("hip_line", [255, 165, 0]))  # Orange
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
        row_data: Dict[str, Any],
    ) -> np.ndarray:
        """Draw bent-over row visualization overlay."""
        h, w = frame.shape[:2]

        # Draw body lines
        frame = self._draw_body_lines(frame, keypoints, kpt_indices, row_data)

        # Draw stats panel
        self._draw_stats_panel(frame, row_data)

        # Draw ROM progress bar
        self._draw_rom_progress(frame, row_data)

        # Draw form issues
        if row_data.get("current_issues"):
            self._draw_issues(frame, row_data["current_issues"])

        return frame

    def _draw_body_lines(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        kpt_indices: Dict[str, int],
        row_data: Dict[str, Any],
    ) -> np.ndarray:
        """Draw body lines and angles."""
        h, w = frame.shape[:2]

        try:
            shoulder = keypoints[kpt_indices["shoulder"]][:2] * np.array([w, h])
            elbow = keypoints[kpt_indices["elbow"]][:2] * np.array([w, h])
            wrist = keypoints[kpt_indices["wrist"]][:2] * np.array([w, h])
            hip = keypoints[kpt_indices["hip"]][:2] * np.array([w, h])
            knee = keypoints[kpt_indices["knee"]][:2] * np.array([w, h])

            # Draw hip hinge line (shoulder-hip-knee)
            pts_hip = np.array([shoulder, hip, knee], dtype=np.int32)
            cv2.polylines(frame, [pts_hip], False, self.hip_line_color, 3)

            # Draw arm lines (shoulder-elbow-wrist)
            pts_arm = np.array([shoulder, elbow, wrist], dtype=np.int32)
            cv2.polylines(frame, [pts_arm], False, self.elbow_line_color, 3)

            # Draw keypoint circles
            for pt, color in [
                (shoulder, self.back_line_color),
                (elbow, self.elbow_line_color),
                (wrist, self.elbow_line_color),
                (hip, self.hip_line_color),
            ]:
                cv2.circle(frame, tuple(pt.astype(int)), 8, color, -1)
                cv2.circle(frame, tuple(pt.astype(int)), 10, (255, 255, 255), 2)

            # Draw angle labels
            elbow_angle = row_data.get("elbow_angle", 0)
            hip_angle = row_data.get("hip_angle", 0)

            cv2.putText(
                frame, f"Elbow: {elbow_angle:.0f}°",
                tuple((elbow + np.array([15, -10])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.elbow_line_color, 2
            )

            cv2.putText(
                frame, f"Hip: {hip_angle:.0f}°",
                tuple((hip + np.array([15, -10])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.hip_line_color, 2
            )

        except Exception as e:
            logger.debug(f"Error drawing body lines: {e}")

        return frame

    def _draw_stats_panel(self, frame: np.ndarray, row_data: Dict[str, Any]) -> None:
        """Draw statistics panel."""
        h, w = frame.shape[:2]

        panel_h = 200
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
                      self.elbow_line_color, 2)

        # Title
        cv2.putText(frame, "BENT-OVER ROW", (panel_x + 10, panel_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.elbow_line_color, 2)

        # State
        state = row_data.get("state", "extended").upper()
        cv2.putText(frame, f"State: {state}", (panel_x + 10, panel_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

        # Reps
        reps = row_data.get("reps", 0)
        cv2.putText(frame, f"Reps: {reps}", (panel_x + 10, panel_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.good_form_color, 2)

        # Elbow angle
        elbow_angle = row_data.get("elbow_angle", 0)
        cv2.putText(frame, f"Elbow: {elbow_angle:.1f}°", (panel_x + 10, panel_y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.elbow_line_color, 1)

        # Hip angle
        hip_angle = row_data.get("hip_angle", 0)
        cv2.putText(frame, f"Hip Hinge: {hip_angle:.1f}°", (panel_x + 10, panel_y + 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.hip_line_color, 1)

        # Back curvature
        back_curvature = row_data.get("back_curvature", 0)
        curvature_color = (
            self.good_form_color if back_curvature < 0.15 else
            self.warning_color if back_curvature < 0.25 else
            self.bad_form_color
        )
        cv2.putText(frame, f"Back Curve: {back_curvature:.2f}", (panel_x + 10, panel_y + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, curvature_color, 1)

        # ROM
        rom = row_data.get("rom_percentage", 0)
        cv2.putText(frame, f"ROM: {rom:.0f}%", (panel_x + 10, panel_y + 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

    def _draw_rom_progress(self, frame: np.ndarray, row_data: Dict[str, Any]) -> None:
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
        rom = row_data.get("rom_percentage", 0) / 100.0
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


class BentOverRowExercise(DualCameraExerciseMixin):
    """
    Bent-Over Row exercise analyzer.
    
    Tracks rowing movement with back curvature monitoring,
    hip hinge maintenance, and provides real-time form feedback.
    Supports dual-camera for arm symmetry detection.
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
        self.state = "extended"  # extended, rowing, contracted, lowering
        self.prev_state = "extended"

        # Rep counting
        self.rep_count = 0
        self.successful_reps = 0
        self.failed_reps = 0

        # Angle tracking
        self.current_elbow_angle = 170.0
        self.current_hip_angle = 90.0
        self.min_elbow_angle = 180.0
        
        # Dual camera support
        DualCameraExerciseMixin.__init__(self)
        self.dual_camera_enabled = False
        self.front_view_keypoints = None
        self.last_front_analysis = None
        
        # Dual camera form issues
        self.asymmetric_row_detected = False
        self.shoulder_rotation_detected = False

        # ROM tracking
        self.elbow_extended = 170.0
        self.elbow_contracted = 60.0

        # Back curvature
        self.current_back_curvature = 0.0
        self.max_back_curvature = 0.0
        self.back_curvature_history: List[float] = []

        # Issue detection
        self.current_issues: List[str] = []
        self.issue_counts: Dict[str, int] = {
            "rounded_back": 0,
            "hip_rise": 0,
            "incomplete_row": 0,
        }

        # Hip hinge baseline
        self.hip_angle_baseline: Optional[float] = None
        self.hip_angle_tolerance = 15.0

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
        self._prev_elbow_angle = 170.0
        self._elbow_velocity = 0.0

        # Visualizer
        self.visualizer = BentOverRowVisualizer(config)

        logger.info(f"BentOverRowExercise initialized with level: {self.experience_level}")

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
                "ankle": kpt_cfg.get("left_ankle", 15),
            }
        else:
            return {
                "nose": kpt_cfg.get("nose", 0),
                "shoulder": kpt_cfg.get("right_shoulder", 6),
                "elbow": kpt_cfg.get("right_elbow", 8),
                "wrist": kpt_cfg.get("right_wrist", 10),
                "hip": kpt_cfg.get("right_hip", 12),
                "knee": kpt_cfg.get("right_knee", 14),
                "ankle": kpt_cfg.get("right_ankle", 16),
            }

    def _build_thresholds(self, level: str) -> Dict[str, float]:
        """Build thresholds based on experience level from config."""
        level = (level or "intermediate").lower()

        # Get exercise-specific config
        ex_config = self.config.get("bent_over_row", {}) or {}
        ex_thresholds = ex_config.get("thresholds", {})
        min_rep_duration = float(ex_config.get("min_rep_duration", 1.0))

        # Get level-specific config
        level_cfg = ex_thresholds.get(level, ex_thresholds.get("intermediate", {}))

        return {
            "elbow_extended": float(level_cfg.get("elbow_extended", 160.0)),
            "elbow_contracted": float(level_cfg.get("elbow_contracted", 60.0)),
            "hip_angle_min": float(level_cfg.get("hip_angle_min", 70.0)),
            "hip_angle_max": float(level_cfg.get("hip_angle_max", 120.0)),
            "back_curvature_warning": float(level_cfg.get("back_curvature_warning", 0.15)),
            "back_curvature_threshold": float(level_cfg.get("back_curvature_threshold", 0.20)),
            "hip_rise_threshold": float(level_cfg.get("hip_rise_threshold", 15.0)),
            "attempt_threshold": float(level_cfg.get("attempt_threshold", 0.35)),
            "success_threshold": float(level_cfg.get("success_threshold", 0.70)),
            "min_rep_duration": min_rep_duration,
        }

    def set_experience_level(self, level: str) -> None:
        """Update experience level."""
        self.experience_level = level
        self.thresholds = self._build_thresholds(level)
        logger.info(f"Bent-over row experience level set to: {level}")

    def set_facing_side(self, side: str) -> None:
        """Set facing side."""
        self.facing_side = side
        self.kpt_indices = self._get_keypoint_indices()
        logger.info(f"Bent-over row facing side set to: {side}")

    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by p1-p2-p3."""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _compute_back_curvature(self, frame: np.ndarray, keypoints: np.ndarray) -> float:
        """Compute back curvature using segmentation."""
        if self.segmenter is None:
            return 0.0

        try:
            mask = self.segmenter.segment_person(frame)
            if mask is None:
                return 0.0

            curvature_info = compute_back_curvature(
                mask, keypoints[:, :2],
                self.config.get("keypoints", {}),
                self.facing_side or "right"
            )
            curvature = curvature_info.get("curvature", 0.0)
            return 0.0 if np.isnan(curvature) else curvature

        except Exception as e:
            logger.debug(f"Back curvature computation error: {e}")
            return 0.0

    def _calculate_rom_percentage(self, elbow_angle: float) -> float:
        """Calculate ROM percentage based on elbow angle."""
        elbow_extended = self.thresholds["elbow_extended"]
        elbow_contracted = self.thresholds["elbow_contracted"]

        if elbow_extended <= elbow_contracted:
            return 0.0

        rom = (elbow_extended - elbow_angle) / (elbow_extended - elbow_contracted)
        return max(0.0, min(1.0, rom)) * 100

    def _detect_issues(
        self, elbow_angle: float, hip_angle: float, back_curvature: float
    ) -> List[str]:
        """Detect form issues."""
        issues = []

        # Rounded back
        if back_curvature > self.thresholds["back_curvature_threshold"]:
            issues.append("Rounded back - keep spine neutral!")
            self.issue_counts["rounded_back"] += 1
        elif back_curvature > self.thresholds["back_curvature_warning"]:
            issues.append("Watch your back position")

        # Hip rise (standing up during row)
        if self.hip_angle_baseline is not None:
            if hip_angle > self.hip_angle_baseline + self.hip_angle_tolerance:
                issues.append("Keep hip hinge - don't stand up!")
                self.issue_counts["hip_rise"] += 1

        return issues

    def _update_state(self, elbow_angle: float, hip_angle: float) -> None:
        """Update state machine."""
        self.prev_state = self.state

        # Calculate velocity
        self._elbow_velocity = elbow_angle - self._prev_elbow_angle
        self._prev_elbow_angle = elbow_angle

        # Set hip baseline on first frame in position
        if self.hip_angle_baseline is None and 60 < hip_angle < 130:
            self.hip_angle_baseline = hip_angle
            logger.debug(f"Hip angle baseline set: {hip_angle:.1f}°")

        if self.state == "extended":
            # Check if starting to row
            if self._elbow_velocity < -2.0 and elbow_angle < self.thresholds["elbow_extended"] - 10:
                self.state = "rowing"
                self.rep_start_time = time.time()
                self.min_elbow_angle = elbow_angle
                logger.debug("Bent-over row: rowing")

        elif self.state == "rowing":
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

            # Check if reached contracted position
            if elbow_angle <= self.thresholds["elbow_contracted"] + 20:
                self.state = "contracted"
                logger.debug("Bent-over row: contracted")
            # Check if extending early
            elif self._elbow_velocity > 2.0:
                self.state = "lowering"
                logger.debug("Bent-over row: lowering (early)")

        elif self.state == "contracted":
            # Check if starting to lower
            if self._elbow_velocity > 2.0:
                self.state = "lowering"
                logger.debug("Bent-over row: lowering")

        elif self.state == "lowering":
            # Check if returned to extended
            if elbow_angle >= self.thresholds["elbow_extended"] - 10:
                self._complete_rep(elbow_angle)
                self.state = "extended"
                logger.debug("Bent-over row: extended (rep complete)")

    def _complete_rep(self, elbow_angle: float) -> None:
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
        rom_percentage = self._calculate_rom_percentage(self.min_elbow_angle)

        # Calculate average back curvature during rep
        avg_curvature = np.mean(self.back_curvature_history[-30:]) if self.back_curvature_history else 0.0

        # Determine success
        is_successful = (
            rom_percentage >= self.thresholds["success_threshold"] * 100 and
            avg_curvature < self.thresholds["back_curvature_threshold"]
        )

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
            "min_elbow_angle": self.min_elbow_angle,
            "avg_back_curvature": avg_curvature,
            "duration": duration,
        }
        self.rep_history.append(rep_data)

        # Reset
        self.min_elbow_angle = 180.0
        self.max_back_curvature = 0.0
        self.rep_start_time = None

        logger.info(f"Bent-over row rep {self.rep_count}: ROM={rom_percentage:.0f}%, Back={avg_curvature:.2f}, Success={is_successful}")

    def _trigger_voice_feedback(self, issue: str) -> None:
        """Trigger voice feedback."""
        current_time = time.time()

        if current_time - self.last_voice_time < self.voice_cooldown:
            return

        voice_map = {
            "Rounded back - keep spine neutral!": "rounded_back",
            "Keep hip hinge - don't stand up!": "hip_rise",
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
        """Process a frame for bent-over row analysis."""
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
            knee = keypoints[self.kpt_indices["knee"]][:2] * np.array([w, h])

            # Calculate angles
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            hip_angle = self.calculate_angle(shoulder, hip, knee)

            self.current_elbow_angle = elbow_angle
            self.current_hip_angle = hip_angle

            # Compute back curvature
            back_curvature = self._compute_back_curvature(frame, keypoints)
            self.current_back_curvature = back_curvature
            self.back_curvature_history.append(back_curvature)
            self.max_back_curvature = max(self.max_back_curvature, back_curvature)

            # Keep history manageable
            if len(self.back_curvature_history) > 300:
                self.back_curvature_history = self.back_curvature_history[-150:]

            # Update state machine
            self._update_state(elbow_angle, hip_angle)

            # Detect issues
            self.current_issues = self._detect_issues(elbow_angle, hip_angle, back_curvature)

            # Trigger voice feedback
            for issue in self.current_issues:
                self._trigger_voice_feedback(issue)

            # Calculate ROM
            rom_percentage = self._calculate_rom_percentage(elbow_angle)

            # Prepare visualization data
            row_data = {
                "state": self.state,
                "reps": self.rep_count,
                "successful_reps": self.successful_reps,
                "elbow_angle": elbow_angle,
                "hip_angle": hip_angle,
                "back_curvature": back_curvature,
                "rom_percentage": rom_percentage,
                "current_issues": self.current_issues,
            }

            # Draw overlay
            frame = self.visualizer.draw_overlay(frame, keypoints, self.kpt_indices, row_data)

        except Exception as e:
            logger.error(f"Error processing bent-over row frame: {e}", exc_info=True)

        return frame

    # DUAL CAMERA METHODS
    def process_front_frame(
        self, 
        front_keypoints: np.ndarray, 
        frame_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        """
        Process front-view frame for bent-over row specific analysis.
        
        Detects:
        - Asymmetric rowing (one arm pulling more)
        - Shoulder rotation during row
        """
        if front_keypoints is None:
            return {}
        
        kpt_config = self.config.get("keypoints", {})
        
        # Compute front-view specific metrics
        arm_symmetry = self.compute_arm_symmetry(front_keypoints, kpt_config, frame_shape)
        shoulder_alignment = self.compute_shoulder_alignment(front_keypoints, kpt_config, frame_shape)
        hip_alignment = self.compute_hip_alignment(front_keypoints, kpt_config, frame_shape)
        
        # Check for shoulder rotation (shoulders should stay level during row)
        shoulder_rotation = shoulder_alignment.get("shoulder_tilt_angle", 0)
        
        # Update form flags during active rep
        if self.state in ["rowing", "contracted"]:
            if arm_symmetry.get("asymmetric_detected"):
                self.asymmetric_row_detected = True
                if "asymmetric_row" not in self.current_issues:
                    self.current_issues.append("asymmetric_row")
                logger.debug(f"Asymmetric row detected: score={arm_symmetry['symmetry_score']:.1f}")
            
            if shoulder_rotation > 15.0:  # More than 15 degrees rotation
                self.shoulder_rotation_detected = True
                if "shoulder_rotation" not in self.current_issues:
                    self.current_issues.append("shoulder_rotation")
                logger.debug(f"Shoulder rotation detected: {shoulder_rotation:.1f}°")
        
        self.last_front_analysis = {
            "arm_symmetry": arm_symmetry,
            "shoulder_alignment": shoulder_alignment,
            "hip_alignment": hip_alignment,
            "shoulder_rotation": shoulder_rotation,
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
        
        if self.asymmetric_row_detected:
            issues.append("asymmetric_row")
        
        if self.shoulder_rotation_detected:
            issues.append("shoulder_rotation")
        
        return issues
    
    def reset_dual_camera_flags(self):
        """Reset dual camera form flags at start of new rep."""
        self.asymmetric_row_detected = False
        self.shoulder_rotation_detected = False

    def get_results(self) -> Dict[str, Any]:
        """Get session results."""
        avg_curvature = np.mean(self.back_curvature_history) if self.back_curvature_history else 0.0
        avg_duration = np.mean(self.rep_durations) if self.rep_durations else 0.0

        return {
            "exercise_type": "bent_over_row",
            "total_reps": self.rep_count,
            "successful_reps": self.successful_reps,
            "failed_reps": self.failed_reps,
            "accuracy": (self.successful_reps / max(1, self.rep_count)) * 100,
            "avg_back_curvature": round(avg_curvature, 3),
            "issue_counts": self.issue_counts,
            "avg_rep_duration": round(avg_duration, 2),
            "rep_history": self.rep_history,
            "dual_camera_enabled": self.dual_camera_enabled,
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get current frame metadata."""
        return {
            "exercise_type": "bent_over_row",
            "state": self.state,
            "total_reps": self.rep_count,
            "successful_reps": self.successful_reps,
            "elbow_angle": round(self.current_elbow_angle, 1),
            "hip_angle": round(self.current_hip_angle, 1),
            "back_curvature": round(self.current_back_curvature, 3),
            "current_issues": self.current_issues,
        }

    def get_voice_messages(self) -> List[Tuple[float, str]]:
        """Get recorded voice messages."""
        return self.voice_messages

    def finalize_analysis(self, output_dir: str, timestamp: str = None) -> None:
        """Finalize analysis."""
        logger.info(f"Bent-over row analysis finalized. Total reps: {self.rep_count}")

    def get_exercise_name(self) -> str:
        """Return exercise name."""
        return "bent_over_row"

    def get_anthropometrics(self) -> Optional[Dict[str, Any]]:
        """Return None - no anthropometrics computed for bent-over row."""
        return None

    def reset(self) -> None:
        """Reset exercise state."""
        self.state = "extended"
        self.prev_state = "extended"
        self.rep_count = 0
        self.successful_reps = 0
        self.failed_reps = 0
        self.current_elbow_angle = 170.0
        self.current_hip_angle = 90.0
        self.min_elbow_angle = 180.0
        self.current_back_curvature = 0.0
        self.max_back_curvature = 0.0
        self.back_curvature_history = []
        self.current_issues = []
        self.issue_counts = {k: 0 for k in self.issue_counts}
        self.hip_angle_baseline = None
        self.rep_history = []
        self.rep_durations = []
        self.voice_messages = []
        self.frame_count = 0
        logger.info("Bent-over row exercise reset")
