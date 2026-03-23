"""
Wall Sit Exercise Analyzer - Isometric (Time-Based) Exercise

This module implements wall sit analysis with:
- Knee and hip angle monitoring
- Back alignment tracking
- Hold duration tracking
- Real-time form feedback
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger


class WallSitVisualizer:
    """Visualization helper for wall sit exercise."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        vis_cfg = config.get("visualization", {})
        colors = vis_cfg.get("colors", {})

        self.knee_line_color = tuple(colors.get("knee_line", [0, 255, 255]))
        self.hip_line_color = tuple(colors.get("hip_line", [255, 165, 0]))
        self.good_form_color = tuple(colors.get("good_form", [0, 255, 0]))
        self.bad_form_color = tuple(colors.get("bad_form", [0, 0, 255]))
        self.warning_color = tuple(colors.get("warning", [0, 165, 255]))
        self.text_color = tuple(colors.get("text", [255, 255, 255]))

    def draw_overlay(
        self, frame: np.ndarray, keypoints: np.ndarray,
        kpt_indices: Dict[str, int], wall_sit_data: Dict[str, Any],
    ) -> np.ndarray:
        """Draw wall sit visualization overlay."""
        h, w = frame.shape[:2]

        try:
            shoulder = keypoints[kpt_indices["shoulder"]][:2] * np.array([w, h])
            hip = keypoints[kpt_indices["hip"]][:2] * np.array([w, h])
            knee = keypoints[kpt_indices["knee"]][:2] * np.array([w, h])
            ankle = keypoints[kpt_indices["ankle"]][:2] * np.array([w, h])

            # Determine color based on form quality
            form_quality = wall_sit_data.get("form_quality", "unknown")
            if form_quality == "good":
                line_color = self.good_form_color
            elif form_quality == "warning":
                line_color = self.warning_color
            else:
                line_color = self.bad_form_color

            # Draw body lines
            pts = np.array([shoulder, hip, knee, ankle], dtype=np.int32)
            cv2.polylines(frame, [pts], False, line_color, 3)

            for pt in [shoulder, hip, knee, ankle]:
                cv2.circle(frame, tuple(pt.astype(int)), 8, line_color, -1)
                cv2.circle(frame, tuple(pt.astype(int)), 10, (255, 255, 255), 2)

            # Draw angle labels
            knee_angle = wall_sit_data.get("knee_angle", 0)
            hip_angle = wall_sit_data.get("hip_angle", 0)

            cv2.putText(frame, f"{knee_angle:.0f}°",
                        tuple((knee + np.array([15, -10])).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.knee_line_color, 2)
            cv2.putText(frame, f"{hip_angle:.0f}°",
                        tuple((hip + np.array([15, -10])).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.hip_line_color, 2)

        except Exception as e:
            logger.debug(f"Error drawing body lines: {e}")

        # Draw stats panel
        self._draw_stats_panel(frame, wall_sit_data)
        return frame

    def _draw_stats_panel(self, frame: np.ndarray, data: Dict[str, Any]) -> None:
        """Draw statistics panel."""
        h, w = frame.shape[:2]
        panel_x, panel_y = w - 300, 20
        panel_w, panel_h = 280, 180

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), self.knee_line_color, 2)

        cv2.putText(frame, "WALL SIT", (panel_x + 10, panel_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.knee_line_color, 2)

        hold_time = data.get("hold_time", 0.0)
        minutes, seconds = int(hold_time // 60), int(hold_time % 60)
        cv2.putText(frame, f"Hold: {minutes:02d}:{seconds:02d}", (panel_x + 10, panel_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)

        cv2.putText(frame, f"Knee: {data.get('knee_angle', 0):.1f}°", (panel_x + 10, panel_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.knee_line_color, 1)
        cv2.putText(frame, f"Hip: {data.get('hip_angle', 0):.1f}°", (panel_x + 10, panel_y + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.hip_line_color, 1)

        form_quality = data.get("form_quality", "unknown")
        form_color = self.good_form_color if form_quality == "good" else self.warning_color if form_quality == "warning" else self.bad_form_color
        cv2.putText(frame, f"Form: {form_quality.upper()}", (panel_x + 10, panel_y + 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, form_color, 2)

        if data.get("current_issue"):
            cv2.putText(frame, data["current_issue"], (panel_x + 10, panel_y + 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.bad_form_color, 1)


class WallSitExercise:
    """Wall Sit exercise analyzer - isometric (time-based) exercise."""

    def __init__(self, config: Dict[str, Any], fps: float = 30.0, segmenter: Any = None):
        self.config = config
        self.fps = fps
        self.segmenter = segmenter

        self.experience_level = config.get("experience", {}).get("level", "intermediate")
        self.thresholds = self._build_thresholds(self.experience_level)
        self.facing_side = None
        self.kpt_indices = self._get_keypoint_indices()

        # State tracking
        self.is_in_position = False
        self.position_start_time: Optional[float] = None
        self.total_hold_time = 0.0
        self.good_form_time = 0.0
        self.bad_form_time = 0.0

        self.current_knee_angle = 90.0
        self.current_hip_angle = 90.0
        self.current_form_quality = "unknown"
        self.current_issue = ""

        self.issue_counts: Dict[str, int] = {"knees_too_extended": 0, "knees_too_bent": 0, "hips_too_high": 0, "hips_too_low": 0}

        self.session_start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = time.time()

        self.voice_messages: List[Tuple[float, str]] = []
        self.last_voice_time = 0.0
        self.voice_cooldown = 3.0

        self.visualizer = WallSitVisualizer(config)
        logger.info(f"WallSitExercise initialized with level: {self.experience_level}")

    def _get_keypoint_indices(self) -> Dict[str, int]:
        kpt_cfg = self.config.get("keypoints", {})
        facing_side = self.facing_side or "right"
        prefix = "left" if facing_side == "left" else "right"
        return {
            "shoulder": kpt_cfg.get(f"{prefix}_shoulder", 5 if prefix == "left" else 6),
            "hip": kpt_cfg.get(f"{prefix}_hip", 11 if prefix == "left" else 12),
            "knee": kpt_cfg.get(f"{prefix}_knee", 13 if prefix == "left" else 14),
            "ankle": kpt_cfg.get(f"{prefix}_ankle", 15 if prefix == "left" else 16),
        }

    def _build_thresholds(self, level: str) -> Dict[str, float]:
        level = (level or "intermediate").lower()
        ex_config = self.config.get("wall_sit", {}) or {}
        ex_thresholds = ex_config.get("thresholds", {})
        level_cfg = ex_thresholds.get(level, ex_thresholds.get("intermediate", {}))

        return {
            "knee_angle_min": float(level_cfg.get("knee_angle_min", 85.0)),
            "knee_angle_max": float(level_cfg.get("knee_angle_max", 95.0)),
            "hip_angle_min": float(level_cfg.get("hip_angle_min", 85.0)),
            "hip_angle_max": float(level_cfg.get("hip_angle_max", 95.0)),
            "warning_tolerance": float(level_cfg.get("warning_tolerance", 10.0)),
            "min_hold_time": float(level_cfg.get("min_hold_time", 2.0)),
        }

    def set_experience_level(self, level: str) -> None:
        self.experience_level = level
        self.thresholds = self._build_thresholds(level)
        logger.info(f"Wall sit experience level set to: {level}")

    def set_facing_side(self, side: str) -> None:
        self.facing_side = side
        self.kpt_indices = self._get_keypoint_indices()
        logger.info(f"Wall sit facing side set to: {side}")

    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        v1, v2 = p1 - p2, p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def _detect_wall_sit_position(self, keypoints: np.ndarray, frame_shape: Tuple[int, int]) -> bool:
        h, w = frame_shape[:2]
        try:
            hip = keypoints[self.kpt_indices["hip"]][:2] * np.array([w, h])
            knee = keypoints[self.kpt_indices["knee"]][:2] * np.array([w, h])
            ankle = keypoints[self.kpt_indices["ankle"]][:2] * np.array([w, h])

            knee_angle = self.calculate_angle(hip, knee, ankle)
            # Wall sit position: knee around 90 degrees
            return 70 < knee_angle < 120
        except:
            return False

    def _analyze_form(self, keypoints: np.ndarray, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        h, w = frame_shape[:2]
        try:
            shoulder = keypoints[self.kpt_indices["shoulder"]][:2] * np.array([w, h])
            hip = keypoints[self.kpt_indices["hip"]][:2] * np.array([w, h])
            knee = keypoints[self.kpt_indices["knee"]][:2] * np.array([w, h])
            ankle = keypoints[self.kpt_indices["ankle"]][:2] * np.array([w, h])

            knee_angle = self.calculate_angle(hip, knee, ankle)
            hip_angle = self.calculate_angle(shoulder, hip, knee)

            self.current_knee_angle = knee_angle
            self.current_hip_angle = hip_angle

            form_quality, current_issue = "good", ""
            warn_tol = self.thresholds["warning_tolerance"]

            if knee_angle < self.thresholds["knee_angle_min"]:
                if knee_angle < self.thresholds["knee_angle_min"] - warn_tol:
                    form_quality, current_issue = "bad", "Knees too bent"
                    self.issue_counts["knees_too_bent"] += 1
                else:
                    form_quality, current_issue = "warning", "Adjust knee angle"
            elif knee_angle > self.thresholds["knee_angle_max"]:
                if knee_angle > self.thresholds["knee_angle_max"] + warn_tol:
                    form_quality, current_issue = "bad", "Knees too extended"
                    self.issue_counts["knees_too_extended"] += 1
                else:
                    form_quality, current_issue = "warning", "Adjust knee angle"

            if hip_angle < self.thresholds["hip_angle_min"] - warn_tol:
                if form_quality == "good":
                    form_quality = "warning"
                current_issue = current_issue or "Hips too low"
                self.issue_counts["hips_too_low"] += 1
            elif hip_angle > self.thresholds["hip_angle_max"] + warn_tol:
                if form_quality == "good":
                    form_quality = "warning"
                current_issue = current_issue or "Hips too high"
                self.issue_counts["hips_too_high"] += 1

            self.current_form_quality = form_quality
            self.current_issue = current_issue

            return {"knee_angle": knee_angle, "hip_angle": hip_angle, "form_quality": form_quality, "current_issue": current_issue}
        except Exception as e:
            logger.debug(f"Error analyzing wall sit form: {e}")
            return {"knee_angle": 90.0, "hip_angle": 90.0, "form_quality": "unknown", "current_issue": ""}

    def _update_timing(self, form_quality: str) -> None:
        current_time = time.time()
        if self.is_in_position:
            elapsed = current_time - self.last_frame_time
            self.total_hold_time += elapsed
            if form_quality == "good":
                self.good_form_time += elapsed
            elif form_quality in ("warning", "bad"):
                self.bad_form_time += elapsed
        self.last_frame_time = current_time

    def _trigger_voice_feedback(self, issue: str) -> None:
        current_time = time.time()
        if current_time - self.last_voice_time < self.voice_cooldown:
            return
        voice_map = {"Knees too bent": "knee_position", "Knees too extended": "knee_position", "Hips too low": "hip_position", "Hips too high": "hip_position"}
        if issue in voice_map:
            self.voice_messages.append((current_time - self.session_start_time, voice_map[issue]))
            self.last_voice_time = current_time

    def process_frame(self, frame: np.ndarray, frame_number: int, keypoints: Optional[np.ndarray] = None, results: Any = None) -> np.ndarray:
        self.frame_count = frame_number
        if keypoints is None or len(keypoints) == 0:
            return frame

        frame_shape = frame.shape[:2]
        in_position = self._detect_wall_sit_position(keypoints, frame_shape)

        if in_position:
            if not self.is_in_position:
                self.is_in_position = True
                self.position_start_time = time.time()
                logger.info("Wall sit position detected")

            form_data = self._analyze_form(keypoints, frame_shape)
            self._update_timing(form_data["form_quality"])

            if form_data["current_issue"]:
                self._trigger_voice_feedback(form_data["current_issue"])

            wall_sit_data = {
                "hold_time": self.total_hold_time,
                "knee_angle": form_data["knee_angle"],
                "hip_angle": form_data["hip_angle"],
                "form_quality": form_data["form_quality"],
                "current_issue": form_data["current_issue"],
            }
            frame = self.visualizer.draw_overlay(frame, keypoints, self.kpt_indices, wall_sit_data)
        else:
            if self.is_in_position:
                self.is_in_position = False
                logger.info(f"Wall sit ended - total hold: {self.total_hold_time:.1f}s")
            self.last_frame_time = time.time()

        return frame

    def get_results(self) -> Dict[str, Any]:
        form_score = (self.good_form_time / self.total_hold_time) * 100 if self.total_hold_time > 0 else 0.0
        return {
            "exercise_type": "wall_sit",
            "total_hold_time": round(self.total_hold_time, 1),
            "good_form_time": round(self.good_form_time, 1),
            "bad_form_time": round(self.bad_form_time, 1),
            "form_score": round(form_score, 1),
            "issue_counts": self.issue_counts,
            "is_isometric": True,
        }

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "exercise_type": "wall_sit",
            "hold_time": round(self.total_hold_time, 1),
            "knee_angle": round(self.current_knee_angle, 1),
            "hip_angle": round(self.current_hip_angle, 1),
            "form_quality": self.current_form_quality,
            "current_issue": self.current_issue,
            "is_in_position": self.is_in_position,
        }

    def get_voice_messages(self) -> List[Tuple[float, str]]:
        return self.voice_messages

    def finalize_analysis(self, output_dir: str, timestamp: str = None) -> None:
        """Finalize analysis - minimal implementation for isometric exercises."""
        logger.info(f"Wall sit analysis finalized. Total hold time: {self.total_hold_time:.1f}s")

    def get_exercise_name(self) -> str:
        """Return exercise name."""
        return "wall_sit"

    def get_anthropometrics(self) -> Optional[Dict[str, Any]]:
        """Return None - no anthropometrics computed for wall sit."""
        return None

    def reset(self) -> None:
        self.is_in_position = False
        self.position_start_time = None
        self.total_hold_time = 0.0
        self.good_form_time = 0.0
        self.bad_form_time = 0.0
        self.current_knee_angle = 90.0
        self.current_hip_angle = 90.0
        self.current_form_quality = "unknown"
        self.current_issue = ""
        self.issue_counts = {k: 0 for k in self.issue_counts}
        self.voice_messages = []
        self.frame_count = 0
        logger.info("Wall sit exercise reset")
