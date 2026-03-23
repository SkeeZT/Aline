"""
Plank Exercise Analyzer - Isometric (Time-Based) Exercise

This module implements plank analysis with:
- Body alignment monitoring (shoulder-hip-ankle line)
- Hip sag/pike detection
- Head position tracking
- Hold duration tracking instead of rep counting
- Real-time form feedback
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger


class PlankVisualizer:
    """Visualization helper for plank exercise."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        vis_cfg = config.get("visualization", {})
        colors = vis_cfg.get("colors", {})

        # Colors for visualization
        self.body_line_color = tuple(colors.get("body_line", [255, 215, 0]))  # Gold
        self.good_form_color = tuple(colors.get("good_form", [0, 255, 0]))  # Green
        self.bad_form_color = tuple(colors.get("bad_form", [0, 0, 255]))  # Red
        self.warning_color = tuple(colors.get("warning", [0, 165, 255]))  # Orange
        self.text_color = tuple(colors.get("text", [255, 255, 255]))  # White

    def draw_overlay(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        kpt_indices: Dict[str, int],
        plank_data: Dict[str, Any],
    ) -> np.ndarray:
        """Draw plank-specific visualization overlay."""
        h, w = frame.shape[:2]

        # Extract keypoints
        shoulder_idx = kpt_indices.get("shoulder")
        hip_idx = kpt_indices.get("hip")
        ankle_idx = kpt_indices.get("ankle")
        knee_idx = kpt_indices.get("knee")

        if shoulder_idx is None or hip_idx is None or ankle_idx is None:
            return frame

        shoulder = keypoints[shoulder_idx][:2] * np.array([w, h])
        hip = keypoints[hip_idx][:2] * np.array([w, h])
        ankle = keypoints[ankle_idx][:2] * np.array([w, h])

        # Determine form quality color
        form_quality = plank_data.get("form_quality", "unknown")
        if form_quality == "good":
            line_color = self.good_form_color
        elif form_quality == "warning":
            line_color = self.warning_color
        else:
            line_color = self.bad_form_color

        # Draw body alignment line (shoulder to hip to ankle)
        pts = np.array([shoulder, hip, ankle], dtype=np.int32)
        cv2.polylines(frame, [pts], False, line_color, 3)

        # Draw keypoint circles
        for pt in [shoulder, hip, ankle]:
            cv2.circle(frame, tuple(pt.astype(int)), 8, line_color, -1)
            cv2.circle(frame, tuple(pt.astype(int)), 10, (255, 255, 255), 2)

        # Draw ideal straight line for reference
        cv2.line(
            frame,
            tuple(shoulder.astype(int)),
            tuple(ankle.astype(int)),
            (100, 100, 100),
            1,
            cv2.LINE_AA,
        )

        # Draw stats panel
        self._draw_stats_panel(frame, plank_data)

        return frame

    def _draw_stats_panel(self, frame: np.ndarray, plank_data: Dict[str, Any]) -> None:
        """Draw statistics panel on frame."""
        h, w = frame.shape[:2]

        # Panel background
        panel_h = 180
        panel_w = 280
        panel_x = w - panel_w - 20
        panel_y = 20

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
            self.body_line_color,
            2,
        )

        # Title
        cv2.putText(
            frame,
            "PLANK",
            (panel_x + 10, panel_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.body_line_color,
            2,
        )

        # Hold time
        hold_time = plank_data.get("hold_time", 0.0)
        minutes = int(hold_time // 60)
        seconds = int(hold_time % 60)
        cv2.putText(
            frame,
            f"Hold: {minutes:02d}:{seconds:02d}",
            (panel_x + 10, panel_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.text_color,
            2,
        )

        # Body angle
        body_angle = plank_data.get("body_angle", 180.0)
        cv2.putText(
            frame,
            f"Body Angle: {body_angle:.1f}°",
            (panel_x + 10, panel_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.text_color,
            1,
        )

        # Form quality
        form_quality = plank_data.get("form_quality", "unknown")
        form_color = (
            self.good_form_color
            if form_quality == "good"
            else self.warning_color if form_quality == "warning" else self.bad_form_color
        )
        cv2.putText(
            frame,
            f"Form: {form_quality.upper()}",
            (panel_x + 10, panel_y + 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            form_color,
            2,
        )

        # Current issue (if any)
        current_issue = plank_data.get("current_issue", "")
        if current_issue:
            cv2.putText(
                frame,
                current_issue,
                (panel_x + 10, panel_y + 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.bad_form_color,
                1,
            )


class PlankExercise:
    """
    Plank exercise analyzer - isometric (time-based) exercise.
    
    Unlike rep-based exercises, plank tracks:
    - Hold duration
    - Continuous form quality
    - Time spent in good vs bad form
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
        self.kpt_indices = self._get_keypoint_indices()

        # State tracking
        self.is_in_plank = False
        self.plank_start_time: Optional[float] = None
        self.total_hold_time = 0.0
        self.good_form_time = 0.0
        self.bad_form_time = 0.0

        # Current frame data
        self.current_body_angle = 180.0
        self.current_form_quality = "unknown"
        self.current_issue = ""

        # Issue tracking
        self.issues: List[Dict[str, Any]] = []
        self.issue_counts: Dict[str, int] = {
            "hip_sag": 0,
            "hip_pike": 0,
            "head_drop": 0,
            "head_up": 0,
        }

        # Session tracking
        self.session_start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = time.time()

        # Voice feedback
        self.voice_messages: List[Tuple[float, str]] = []
        self.last_voice_time = 0.0
        self.voice_cooldown = 3.0  # Minimum seconds between voice messages

        # Visualizer
        self.visualizer = PlankVisualizer(config)

        logger.info(f"PlankExercise initialized with level: {self.experience_level}")

    def _get_keypoint_indices(self) -> Dict[str, int]:
        """Get keypoint indices based on config or use YOLO defaults."""
        kpt_cfg = self.config.get("keypoints", {})
        
        # Determine facing side for left/right selection
        facing_side = getattr(self, "facing_side", None) or "right"
        
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
        """Build thresholds for plank based on experience level from config."""
        level = (level or "intermediate").lower()

        # Get plank-specific config
        plank_config = self.config.get("plank", {}) or {}
        plank_thresholds = plank_config.get("thresholds", {})

        # Get level-specific config
        level_cfg = plank_thresholds.get(level, plank_thresholds.get("intermediate", {}))

        # Body angle thresholds (ideal is 180 degrees - straight line)
        body_angle_min = float(level_cfg.get("body_angle_min", 170.0))  # Below = sag
        body_angle_max = float(level_cfg.get("body_angle_max", 190.0))  # Above = pike
        
        # Warning thresholds (less strict)
        warning_angle_min = float(level_cfg.get("warning_angle_min", 165.0))
        warning_angle_max = float(level_cfg.get("warning_angle_max", 195.0))
        
        # Head position thresholds
        head_drop_threshold = float(level_cfg.get("head_drop_threshold", 20.0))
        head_up_threshold = float(level_cfg.get("head_up_threshold", 20.0))
        
        # Minimum time to consider "in plank"
        min_hold_time = float(level_cfg.get("min_hold_time", 2.0))

        return {
            "body_angle_min": body_angle_min,
            "body_angle_max": body_angle_max,
            "warning_angle_min": warning_angle_min,
            "warning_angle_max": warning_angle_max,
            "head_drop_threshold": head_drop_threshold,
            "head_up_threshold": head_up_threshold,
            "min_hold_time": min_hold_time,
        }

    def set_experience_level(self, level: str) -> None:
        """Update experience level and rebuild thresholds."""
        self.experience_level = level
        self.thresholds = self._build_thresholds(level)
        logger.info(f"Plank experience level set to: {level}")

    def set_facing_side(self, side: str) -> None:
        """Set facing side and update keypoint indices."""
        self.facing_side = side
        self.kpt_indices = self._get_keypoint_indices()
        logger.info(f"Plank facing side set to: {side}")

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

    def _detect_plank_position(self, keypoints: np.ndarray, frame_shape: Tuple[int, int]) -> bool:
        """Detect if person is in plank position based on body orientation."""
        h, w = frame_shape[:2]
        
        try:
            shoulder = keypoints[self.kpt_indices["shoulder"]][:2] * np.array([w, h])
            hip = keypoints[self.kpt_indices["hip"]][:2] * np.array([w, h])
            ankle = keypoints[self.kpt_indices["ankle"]][:2] * np.array([w, h])
            
            # Check if body is roughly horizontal (plank position)
            # Shoulder and ankle should be at similar y-levels
            shoulder_ankle_y_diff = abs(shoulder[1] - ankle[1])
            body_length = np.linalg.norm(shoulder - ankle)
            
            # If y-difference is less than 40% of body length, likely horizontal
            is_horizontal = shoulder_ankle_y_diff < 0.4 * body_length
            
            # Also check that hip is between shoulder and ankle
            hip_in_line = (
                min(shoulder[0], ankle[0]) - 50 <= hip[0] <= max(shoulder[0], ankle[0]) + 50
            )
            
            return is_horizontal and hip_in_line
            
        except Exception as e:
            logger.debug(f"Error detecting plank position: {e}")
            return False

    def _analyze_form(
        self, keypoints: np.ndarray, frame_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Analyze plank form and return form data."""
        h, w = frame_shape[:2]

        try:
            # Get keypoints
            shoulder = keypoints[self.kpt_indices["shoulder"]][:2] * np.array([w, h])
            hip = keypoints[self.kpt_indices["hip"]][:2] * np.array([w, h])
            ankle = keypoints[self.kpt_indices["ankle"]][:2] * np.array([w, h])
            nose = keypoints[self.kpt_indices["nose"]][:2] * np.array([w, h])

            # Calculate body angle (shoulder-hip-ankle)
            body_angle = self.calculate_angle(shoulder, hip, ankle)
            self.current_body_angle = body_angle

            # Determine form quality
            form_quality = "good"
            current_issue = ""

            # Check for hip sag (angle < threshold)
            if body_angle < self.thresholds["body_angle_min"]:
                if body_angle < self.thresholds["warning_angle_min"]:
                    form_quality = "bad"
                    current_issue = "Hip sag - lift hips!"
                    self.issue_counts["hip_sag"] += 1
                else:
                    form_quality = "warning"
                    current_issue = "Slight hip sag"

            # Check for hip pike (angle > threshold)
            elif body_angle > self.thresholds["body_angle_max"]:
                if body_angle > self.thresholds["warning_angle_max"]:
                    form_quality = "bad"
                    current_issue = "Hip pike - lower hips!"
                    self.issue_counts["hip_pike"] += 1
                else:
                    form_quality = "warning"
                    current_issue = "Slight hip pike"

            # Check head position (relative to shoulder line)
            shoulder_to_ankle = ankle - shoulder
            shoulder_to_nose = nose - shoulder
            
            # Project nose onto shoulder-ankle line
            t = np.dot(shoulder_to_nose, shoulder_to_ankle) / (np.dot(shoulder_to_ankle, shoulder_to_ankle) + 1e-6)
            projected = shoulder + t * shoulder_to_ankle
            head_deviation = np.linalg.norm(nose - projected)
            
            # Determine if head is up or down based on y-coordinate
            if nose[1] > projected[1] + self.thresholds["head_drop_threshold"]:
                if form_quality == "good":
                    form_quality = "warning"
                current_issue = current_issue or "Head dropping"
                self.issue_counts["head_drop"] += 1
            elif nose[1] < projected[1] - self.thresholds["head_up_threshold"]:
                if form_quality == "good":
                    form_quality = "warning"
                current_issue = current_issue or "Head too high"
                self.issue_counts["head_up"] += 1

            self.current_form_quality = form_quality
            self.current_issue = current_issue

            return {
                "body_angle": body_angle,
                "form_quality": form_quality,
                "current_issue": current_issue,
                "head_deviation": head_deviation,
            }

        except Exception as e:
            logger.debug(f"Error analyzing plank form: {e}")
            return {
                "body_angle": 180.0,
                "form_quality": "unknown",
                "current_issue": "",
                "head_deviation": 0.0,
            }

    def _update_timing(self, form_quality: str) -> None:
        """Update hold time tracking."""
        current_time = time.time()
        
        if self.is_in_plank:
            elapsed = current_time - self.last_frame_time
            self.total_hold_time += elapsed
            
            if form_quality == "good":
                self.good_form_time += elapsed
            elif form_quality in ("warning", "bad"):
                self.bad_form_time += elapsed
        
        self.last_frame_time = current_time

    def _trigger_voice_feedback(self, issue: str) -> None:
        """Trigger voice feedback for form issues."""
        current_time = time.time()
        
        if current_time - self.last_voice_time < self.voice_cooldown:
            return
        
        voice_map = {
            "Hip sag - lift hips!": "hip_sag",
            "Hip pike - lower hips!": "hip_pike",
            "Head dropping": "head_drop",
            "Head too high": "head_up",
        }
        
        if issue in voice_map:
            self.voice_messages.append((current_time - self.session_start_time, voice_map[issue]))
            self.last_voice_time = current_time

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        keypoints: Optional[np.ndarray] = None,
        results: Any = None,
    ) -> np.ndarray:
        """Process a frame for plank analysis."""
        self.frame_count = frame_number

        if keypoints is None or len(keypoints) == 0:
            return frame

        frame_shape = frame.shape[:2]

        # Detect if in plank position
        in_plank = self._detect_plank_position(keypoints, frame_shape)

        if in_plank:
            if not self.is_in_plank:
                # Just entered plank
                self.is_in_plank = True
                self.plank_start_time = time.time()
                logger.info("Plank position detected - starting hold timer")

            # Analyze form
            form_data = self._analyze_form(keypoints, frame_shape)
            
            # Update timing
            self._update_timing(form_data["form_quality"])
            
            # Trigger voice feedback if needed
            if form_data["current_issue"]:
                self._trigger_voice_feedback(form_data["current_issue"])

            # Prepare visualization data
            plank_data = {
                "hold_time": self.total_hold_time,
                "body_angle": form_data["body_angle"],
                "form_quality": form_data["form_quality"],
                "current_issue": form_data["current_issue"],
                "good_form_time": self.good_form_time,
                "bad_form_time": self.bad_form_time,
            }

            # Draw overlay
            frame = self.visualizer.draw_overlay(
                frame, keypoints, self.kpt_indices, plank_data
            )

        else:
            if self.is_in_plank:
                # Just left plank position
                self.is_in_plank = False
                logger.info(f"Plank ended - total hold time: {self.total_hold_time:.1f}s")

            self.last_frame_time = time.time()

        return frame

    def get_results(self) -> Dict[str, Any]:
        """Get plank session results."""
        form_score = 0.0
        if self.total_hold_time > 0:
            form_score = (self.good_form_time / self.total_hold_time) * 100

        return {
            "exercise_type": "plank",
            "total_hold_time": round(self.total_hold_time, 1),
            "good_form_time": round(self.good_form_time, 1),
            "bad_form_time": round(self.bad_form_time, 1),
            "form_score": round(form_score, 1),
            "issue_counts": self.issue_counts,
            "is_isometric": True,
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get current frame metadata."""
        return {
            "exercise_type": "plank",
            "hold_time": round(self.total_hold_time, 1),
            "body_angle": round(self.current_body_angle, 1),
            "form_quality": self.current_form_quality,
            "current_issue": self.current_issue,
            "is_in_plank": self.is_in_plank,
            "good_form_percentage": round(
                (self.good_form_time / max(0.1, self.total_hold_time)) * 100, 1
            ),
        }

    def get_voice_messages(self) -> List[Tuple[float, str]]:
        """Get recorded voice messages."""
        return self.voice_messages

    def finalize_analysis(self, output_dir: str, timestamp: str = None) -> None:
        """Finalize analysis - minimal implementation for isometric exercises."""
        logger.info(f"Plank analysis finalized. Total hold time: {self.total_hold_time:.1f}s")

    def get_exercise_name(self) -> str:
        """Return exercise name."""
        return "plank"

    def get_anthropometrics(self) -> Optional[Dict[str, Any]]:
        """Return None - no anthropometrics computed for plank."""
        return None

    def reset(self) -> None:
        """Reset exercise state."""
        self.is_in_plank = False
        self.plank_start_time = None
        self.total_hold_time = 0.0
        self.good_form_time = 0.0
        self.bad_form_time = 0.0
        self.current_body_angle = 180.0
        self.current_form_quality = "unknown"
        self.current_issue = ""
        self.issues = []
        self.issue_counts = {k: 0 for k in self.issue_counts}
        self.voice_messages = []
        self.frame_count = 0
        logger.info("Plank exercise reset")
