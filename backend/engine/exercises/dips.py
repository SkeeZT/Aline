"""
Dips exercise analyzer with complete logic implementation.
Includes side detection, rep counting, form analysis, VBT calculations, and voice feedback.
Supports dual-camera for arm symmetry and shoulder alignment detection.
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


class DipsVisualizer:
    """Visualizer for dips exercise analysis with clean UI design."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.colors = {
            "primary": (70, 130, 180),
            "success": (34, 139, 34),
            "warning": (255, 165, 0),
            "danger": (220, 20, 60),
            "info": (30, 144, 255),
            "dark": (25, 25, 35),
            "light": (245, 245, 245),
            "keypoints": (255, 105, 180),
            "elbow_line": (50, 205, 50),
            "shoulder_line": (255, 215, 0),
            "angle_arc": (186, 85, 211),
            "progress_full": (34, 139, 34),
            "progress_partial": (255, 165, 0),
            "progress_low": (220, 20, 60),
        }

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 1

    def _get_keypoint_indices(self, facing_side: str) -> Dict[str, int]:
        if facing_side == "left":
            prefix = "right"
        else:
            prefix = "left"
            
        kpt = self.config["keypoints"]
        return {
            "shoulder": kpt[f"{prefix}_shoulder"],
            "elbow": kpt[f"{prefix}_elbow"],
            "wrist": kpt[f"{prefix}_wrist"],
            "hip": kpt[f"{prefix}_hip"],
        }

    def draw_rounded_rectangle(self, frame, top_left, bottom_right, color, radius=10, thickness=-1):
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

    def draw_text(self, frame, text, position, color, scale=None, thickness=None):
        if scale is None:
            scale = self.font_scale
        if thickness is None:
            thickness = self.font_thickness
        x, y = position
        cv2.putText(frame, text, (x + 1, y + 1), self.font, scale, (30, 30, 30), thickness)
        cv2.putText(frame, text, (x, y), self.font, scale, color, thickness, cv2.LINE_AA)

    def draw_keypoints(self, frame, keypoints, facing_side="right"):
        indices = self._get_keypoint_indices(facing_side)
        
        for key, idx in indices.items():
            if idx >= len(keypoints):
                continue
            x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
            
            if key == "wrist":
                cv2.circle(frame, (x, y), 12, self.colors["primary"], 2)
                cv2.circle(frame, (x, y), 6, self.colors["primary"], -1)
            else:
                cv2.circle(frame, (x, y), 10, self.colors["keypoints"], 2)
                cv2.circle(frame, (x, y), 6, self.colors["light"], -1)
                cv2.circle(frame, (x, y), 2, self.colors["keypoints"], -1)

        return frame

    def draw_lines(self, frame, keypoints, elbow_angle, facing_side="right"):
        indices = self._get_keypoint_indices(facing_side)
        
        shoulder = (int(keypoints[indices["shoulder"]][0]), int(keypoints[indices["shoulder"]][1]))
        elbow = (int(keypoints[indices["elbow"]][0]), int(keypoints[indices["elbow"]][1]))
        wrist = (int(keypoints[indices["wrist"]][0]), int(keypoints[indices["wrist"]][1]))

        cv2.line(frame, shoulder, elbow, self.colors["shoulder_line"], 4, cv2.LINE_AA)
        cv2.line(frame, elbow, wrist, self.colors["elbow_line"], 4, cv2.LINE_AA)

        elbow_text_pos = (elbow[0] + 50, elbow[1] - 20)
        self.draw_text(frame, f"{elbow_angle:.0f}°", elbow_text_pos, self.colors["elbow_line"], scale=0.7)

        return frame

    def draw_stats(self, frame, successful_reps, unsuccessful_reps, total_reps, state, failure_justifications=None):
        frame_height, frame_width = frame.shape[:2]

        panel_width = 380
        panel_height = 140
        panel_x = frame_width - panel_width - 20
        panel_y = 20

        self._draw_overlay_panel(frame, (panel_x, panel_y, panel_width, panel_height), 
                                  self.colors["primary"], bg_color=self.colors["dark"], alpha=0.6, radius=10)

        header_x = panel_x + 12
        header_y = panel_y + 25
        self.draw_text(frame, "Dips Stats", (header_x, header_y), self.colors["light"], scale=0.6)

        y_pos = header_y + 25
        line_spacing = 22

        self.draw_text(frame, f"Good: {successful_reps}", (header_x, y_pos), self.colors["success"], scale=0.55)
        y_pos += line_spacing
        self.draw_text(frame, f"Bad: {unsuccessful_reps}", (header_x, y_pos), self.colors["danger"], scale=0.55)
        y_pos += line_spacing
        self.draw_text(frame, f"Total: {total_reps}", (header_x, y_pos), self.colors["info"], scale=0.55)
        y_pos += line_spacing

        state_text = "UP" if state == "up" else "DOWN"
        state_color = self.colors["success"] if state == "up" else self.colors["warning"]
        self.draw_text(frame, f"Phase: {state_text}", (header_x, y_pos), state_color, scale=0.55)

        if failure_justifications:
            just_text = ", ".join([j[0].upper() + j[1:] for j in failure_justifications if j])
            if just_text:
                (text_w, _), _ = cv2.getTextSize(f"! {just_text}", self.font, 0.50, self.font_thickness)
                just_width = max(panel_width, text_w + 40)
                just_x = panel_x + panel_width - just_width
                just_y = panel_y + panel_height + 10
                self._draw_overlay_panel(frame, (just_x, just_y, just_width, 44),
                                          self.colors["danger"], bg_color=(20, 20, 20), alpha=0.5, radius=8)
                self.draw_text(frame, f"! {just_text}", (just_x + 20, just_y + 25), self.colors["danger"], scale=0.50)

        return frame

    def draw_progress_bar(self, frame, elbow_angle, elbow_min, elbow_max):
        frame_height, frame_width = frame.shape[:2]
        bar_width = 350
        bar_height = 12
        bar_x = (frame_width - bar_width) // 2
        bar_y = frame_height - 40

        self.draw_rounded_rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (80, 80, 80), radius=6)

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
            self.draw_rounded_rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, radius=6)

        self.draw_text(frame, "Depth", (bar_x - 50, bar_y + 10), self.colors["light"], scale=0.5)
        percentage = int(progress * 100)
        self.draw_text(frame, f"{percentage}%", (bar_x + bar_width + 15, bar_y + 10), color, scale=0.5)

        return frame

    def _draw_overlay_panel(self, frame, rect, border_color, bg_color=(20, 20, 20), alpha=0.5, radius=10):
        x, y, w, h = rect
        overlay = frame.copy()
        self.draw_rounded_rectangle(overlay, (x, y), (x + w, y + h), bg_color, radius=radius)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        self.draw_rounded_rectangle(frame, (x, y), (x + w, y + h), border_color, radius=radius, thickness=1)

    def visualize(self, frame, keypoints, elbow_angle, successful_reps, unsuccessful_reps, total_reps,
                  state, facing_side="right", thresholds=None, failure_justifications=None):
        frame = self.draw_keypoints(frame, keypoints, facing_side)
        frame = self.draw_lines(frame, keypoints, elbow_angle, facing_side)
        frame = self.draw_stats(frame, successful_reps, unsuccessful_reps, total_reps, state, failure_justifications)

        elbow_min = 90.0
        elbow_max = 170.0
        if thresholds:
            elbow_min = thresholds.get("elbow_min", elbow_min)
            elbow_max = thresholds.get("elbow_max", elbow_max)

        frame = self.draw_progress_bar(frame, elbow_angle, elbow_min, elbow_max)
        return frame


class DipsExercise(DualCameraExerciseMixin):
    """Complete dips exercise analyzer with integrated logic and voice feedback."""

    def __init__(self, config: Dict[str, Any], fps: float = 30.0, segmenter: Any = None):
        self.config = config
        self.fps = fps
        self.cooldown_frames = int(fps * 0.5)
        self.state_change_cooldown = int(fps * 0.3)

        self.last_rep_frame = -self.cooldown_frames
        self.last_state_change_frame = 0
        self.min_elbow_angle_this_rep = float('inf')

        # ROM tracking
        self.elbow_max_baseline = None
        self.elbow_min_baseline = None
        self.rom_established = False
        self.elbow_range = None

        # Rep counting
        self.successful_reps = 0
        self.unsuccessful_reps = 0
        self.total_reps = 0
        self.state = "up"
        self.min_depth_reached = False
        self.max_rom_this_rep = 0.0
        self.bottom_position_frame = 0

        # State tracking
        self.last_failure_justification = []
        self.failure_justification_timer = 0
        self.failure_display_duration = int(fps * 3.0)
        self.rep_start_frame = 0
        self._prev_elbow_angle = 0
        self._elbow_velocity = 0

        # Form detection
        self.shoulder_shrug_detected = False
        self.forward_lean_detected = False
        
        # Dual camera support
        DualCameraExerciseMixin.__init__(self)
        self.dual_camera_enabled = False
        self.front_view_keypoints = None
        self.last_front_analysis = None
        
        # Dual camera form issues
        self.asymmetric_dip_detected = False
        self.body_sway_detected = False

        self.reps_summary = []
        self.facing_side: Optional[str] = None
        self.side_determined = False

        self.vbt_calculator = VelocityCalculator(fps=fps)
        self.visualizer = DipsVisualizer(config)
        self.voice_player = VoiceMessagePlayer(config)
        self.voice_messages = []

        self.workout_started = False
        self.workout_ended = False
        self.last_voice_feedback_frame = -self.cooldown_frames
        self.voice_feedback_cooldown_frames = int(fps * 3)
        self.last_feedback_times = {}

        # Shoulder shrug tracking
        self.initial_shoulder_y = None
        self.shoulder_y_history = []

        logger.info("Dips exercise analyzer initialized")

        self.anthropometrics = None
        self.segmenter = segmenter
        self.experience_level = (self.config.get("experience", {}) or {}).get("level", "intermediate")
        self.thresholds = self._build_thresholds(self.experience_level)

    def _build_thresholds(self, level: str) -> Dict[str, float]:
        level = (level or "intermediate").lower()
        
        dips_config = self.config.get("dips", {}) or {}
        dips_thresholds = dips_config.get("thresholds", {})
        min_rep_duration = float(dips_config.get("min_rep_duration", 1.0))
        
        level_cfg = dips_thresholds.get(level, dips_thresholds.get("intermediate", {}))
        
        return {
            "elbow_min": float(level_cfg.get("elbow_min", 90.0)),
            "elbow_max": float(level_cfg.get("elbow_max", 170.0)),
            "attempt_threshold": float(level_cfg.get("attempt_threshold", 0.35)),
            "success_threshold": float(level_cfg.get("success_threshold", 0.70)),
            "shoulder_shrug_threshold": float(level_cfg.get("shoulder_shrug_threshold", 30.0)),
            "min_rep_duration": min_rep_duration,
        }

    def determine_side_from_keypoints(self, keypoints_data: np.ndarray, allow_override: bool = False) -> None:
        if self.side_determined and not allow_override:
            return
        if keypoints_data is not None:
            new_side = determine_facing_side(keypoints_data)
            self.facing_side = new_side
            self.side_determined = True
            logger.info(f"Determined facing side: {self.facing_side}")

    def set_facing_side(self, side: str) -> None:
        if side and side.lower() in ["left", "right"]:
            self.facing_side = side.lower()
            self.side_determined = True
            logger.info(f"Facing side explicitly set to: {self.facing_side}")

    def set_experience_level(self, level: str) -> None:
        level = (level or "intermediate").lower()
        if level not in ("beginner", "intermediate", "advanced"):
            level = "intermediate"
        self.experience_level = level
        self.thresholds = self._build_thresholds(level)
        logger.info(f"Experience level set to {level}")

    def get_side_specific_keypoints(self) -> Dict[str, int]:
        if self.facing_side == "left":
            return {
                "shoulder": self.config["keypoints"]["right_shoulder"],
                "elbow": self.config["keypoints"]["right_elbow"],
                "wrist": self.config["keypoints"]["right_wrist"],
                "hip": self.config["keypoints"]["right_hip"],
            }
        else:
            return {
                "shoulder": self.config["keypoints"]["left_shoulder"],
                "elbow": self.config["keypoints"]["left_elbow"],
                "wrist": self.config["keypoints"]["left_wrist"],
                "hip": self.config["keypoints"]["left_hip"],
            }

    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        v1 = p1 - p2
        v2 = p3 - p2
        dot_prod = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return 0.0
        cos_angle = np.clip(dot_prod / norm_product, -1, 1)
        return math.degrees(math.acos(cos_angle))

    def extract_keypoints(self, results) -> Optional[np.ndarray]:
        if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            return None
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        confidences = results[0].keypoints.conf[0].cpu().numpy()
        kpt_indices = self.get_side_specific_keypoints()
        required_indices = [kpt_indices["shoulder"], kpt_indices["elbow"], kpt_indices["wrist"]]
        for idx in required_indices:
            if confidences[idx] < 0.5:
                return None
        return keypoints

    def _establish_rom_baselines(self, elbow_angle: float, frame_number: int) -> bool:
        if self.elbow_max_baseline is None and elbow_angle > 150:
            self.elbow_max_baseline = elbow_angle
            logger.info(f"Established elbow max baseline: {elbow_angle:.2f}")

        elbow_min_threshold = self.thresholds.get("elbow_min", 90.0)
        if self.elbow_max_baseline is not None and elbow_angle < elbow_min_threshold + 10:
            if self.elbow_min_baseline is None or elbow_angle < self.elbow_min_baseline:
                self.elbow_min_baseline = elbow_angle
                self.elbow_range = self.elbow_max_baseline - self.elbow_min_baseline
                self.rom_established = True
                logger.info(f"Updated elbow min baseline: {elbow_angle:.2f}, ROM range: {self.elbow_range:.2f}")

        return self.rom_established

    def _calculate_rom_percentage(self, current_elbow_angle: float) -> float:
        if not isinstance(current_elbow_angle, (int, float)) or np.isnan(current_elbow_angle):
            return 0.0

        working_max = self.elbow_max_baseline if self.elbow_max_baseline is not None else 170.0
        working_min = self.elbow_min_baseline
        working_range = self.elbow_range

        if not self.rom_established or working_min is None:
            working_min = self.thresholds.get("elbow_min", 90.0)
            working_range = working_max - working_min

        if working_range <= 0:
            return 0.0

        rom_percentage = (working_max - current_elbow_angle) / working_range
        return max(0.0, min(1.0, rom_percentage))

    def _detect_shoulder_shrug(self, shoulder_y: float) -> bool:
        if self.initial_shoulder_y is None:
            self.initial_shoulder_y = shoulder_y
            return False
            
        # Shoulder rising = Y decreasing (image coordinates)
        rise = self.initial_shoulder_y - shoulder_y
        threshold = self.thresholds.get("shoulder_shrug_threshold", 30.0)
        return rise > threshold

    def update_rep_count(self, elbow_angle: float, shoulder_y: float, frame_number: int) -> None:
        self._establish_rom_baselines(elbow_angle, frame_number)
        self.current_rom_percentage = self._calculate_rom_percentage(elbow_angle)

        if self.state in ["going_down", "coming_up"]:
            self.max_rom_this_rep = max(self.max_rom_this_rep, self.current_rom_percentage)

        if not hasattr(self, "_prev_elbow_angle"):
            self._prev_elbow_angle = elbow_angle
            self._elbow_velocity = 0
        else:
            self._elbow_velocity = elbow_angle - self._prev_elbow_angle
            self._prev_elbow_angle = elbow_angle

        if self._check_timeout(frame_number):
            return

        if self.state == "up":
            self._handle_state_up(elbow_angle, frame_number)
        elif self.state == "going_down":
            self._handle_state_descent(elbow_angle, shoulder_y, frame_number)
        elif self.state == "coming_up":
            self._handle_state_ascent(elbow_angle, shoulder_y, frame_number)

    def _check_timeout(self, frame_number: int) -> bool:
        if self.state != "up" and (frame_number - self.last_state_change_frame > self.fps * 5):
            self.state = "up"
            self.shoulder_shrug_detected = False
            self.min_depth_reached = False
            self.initial_shoulder_y = None
            logger.debug("State reset due to timeout")
            return True
        return False

    def _handle_state_up(self, elbow_angle: float, frame_number: int) -> None:
        state_change_allowed = (frame_number - self.last_state_change_frame > self.state_change_cooldown)

        significant_bend = False
        if self.elbow_max_baseline is not None:
            significant_bend = elbow_angle < (self.elbow_max_baseline - 10.0)
        else:
            significant_bend = elbow_angle < 160.0

        if self._elbow_velocity < -2.0 and significant_bend and state_change_allowed:
            self.state = "going_down"
            self.min_depth_reached = False
            self.bottom_position_frame = 0
            self.min_elbow_angle_this_rep = 180
            self.last_state_change_frame = frame_number
            self.max_rom_this_rep = 0.0
            self.rep_start_frame = frame_number
            self.shoulder_shrug_detected = False
            self.initial_shoulder_y = None
            logger.debug(f"State: GOING_DOWN | Frame: {frame_number}")

    def _handle_state_descent(self, elbow_angle: float, shoulder_y: float, frame_number: int) -> None:
        self.min_elbow_angle_this_rep = min(self.min_elbow_angle_this_rep, elbow_angle)

        if self._detect_shoulder_shrug(shoulder_y):
            self.shoulder_shrug_detected = True
            logger.debug("Shoulder shrug detected during descent")

        rep_attempt_threshold = self.thresholds.get("attempt_threshold", 0.35)
        is_rep_attempt = self.current_rom_percentage >= rep_attempt_threshold
        
        if is_rep_attempt and not self.min_depth_reached:
            self.min_depth_reached = True
            self.bottom_position_frame = frame_number

        state_change_allowed = (frame_number - self.last_state_change_frame > self.state_change_cooldown)
        if self._elbow_velocity > 2.0 and state_change_allowed:
            self.state = "coming_up"
            self.last_state_change_frame = frame_number
            logger.debug(f"State: COMING_UP | ROM: {self.current_rom_percentage:.2f}")

    def _handle_state_ascent(self, elbow_angle: float, shoulder_y: float, frame_number: int) -> None:
        if self._detect_shoulder_shrug(shoulder_y):
            self.shoulder_shrug_detected = True

        is_near_extended = elbow_angle > self.thresholds["elbow_max"] - 15
        is_moving_slowly = self._elbow_velocity < 2.0

        if is_near_extended and is_moving_slowly:
            if frame_number - self.last_rep_frame > self.cooldown_frames:
                self._complete_rep(frame_number)

    def _complete_rep(self, frame_number: int) -> None:
        if not self.min_depth_reached:
            self._reset_rep_state()
            logger.debug("Movement ignored - insufficient depth for rep attempt")
            return

        rep_duration = (frame_number - self.rep_start_frame) / self.fps
        min_rep_duration = self.thresholds.get("min_rep_duration", 1.0)
            
        if rep_duration < min_rep_duration:
            self._reset_rep_state()
            logger.debug(f"Rep ignored - Duration {rep_duration:.2f}s < {min_rep_duration}s")
            return

        self.total_reps += 1
        self.last_rep_frame = frame_number
        self.last_state_change_frame = frame_number

        success_threshold_val = self.thresholds.get("success_threshold", 0.70)
        rom_success = self.max_rom_this_rep >= success_threshold_val

        form_issues = []
        failure_justifications = {}

        if not rom_success:
            form_issues.append("insufficient_depth")
            failure_justifications["insufficient_depth"] = f"ROM {self.max_rom_this_rep:.2f} < {success_threshold_val}"

        if self.shoulder_shrug_detected:
            form_issues.append("shoulder_shrug")
            failure_justifications["shoulder_shrug"] = "Shoulders rising during movement"

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

        self.reps_summary.append({
            "rep_number": self.total_reps,
            "success": success,
            "rom_percentage": self.max_rom_this_rep,
            "min_elbow_angle": self.min_elbow_angle_this_rep,
            "shoulder_shrug": self.shoulder_shrug_detected,
            "issues": form_issues,
            "justifications": failure_justifications
        })

        self._reset_rep_state()

    def _reset_rep_state(self):
        self.state = "up"
        self.shoulder_shrug_detected = False
        self.min_depth_reached = False
        self.min_elbow_angle_this_rep = float('inf')
        self.initial_shoulder_y = None

    def _handle_voice_feedback(self, form_issues, frame_number):
        issue = form_issues[0] if form_issues else "unknown"
        if frame_number - self.last_voice_feedback_frame > self.voice_feedback_cooldown_frames:
            correction_map = {
                "insufficient_depth": "limited_depth",
                "shoulder_shrug": "generic",
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
        if keypoints is None and results is not None:
            if isinstance(results, list):
                keypoints = self.extract_keypoints(results)

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

        if not hasattr(self, "_prev_elbow_angle"):
            kpt_indices = self.get_side_specific_keypoints()
            shoulder_point = keypoints[kpt_indices["shoulder"]][:2]
            elbow_point = keypoints[kpt_indices["elbow"]][:2]
            wrist_point = keypoints[kpt_indices["wrist"]][:2]
            initial_angle = self.calculate_angle(shoulder_point, elbow_point, wrist_point)
            self._prev_elbow_angle = initial_angle
            self._elbow_velocity = 0

        if not self.side_determined:
            self.determine_side_from_keypoints(keypoints)

        kpt_indices = self.get_side_specific_keypoints()
        shoulder_point = keypoints[kpt_indices["shoulder"]][:2]
        elbow_point = keypoints[kpt_indices["elbow"]][:2]
        wrist_point = keypoints[kpt_indices["wrist"]][:2]

        elbow_angle = self.calculate_angle(shoulder_point, elbow_point, wrist_point)
        shoulder_y = shoulder_point[1]

        self.update_rep_count(elbow_angle, shoulder_y, frame_number)

        try:
            shoulder_idx = kpt_indices["shoulder"]
            elbow_idx = kpt_indices["elbow"]
            wrist_idx = kpt_indices["wrist"]
            vbt_keypoints = np.array([keypoints[shoulder_idx], keypoints[elbow_idx], keypoints[wrist_idx]])
            vbt_stage = "down" if self.state == "going_down" else "up"
            self.vbt_calculator.add_frame_data(
                frame_number=frame_number,
                keypoints=vbt_keypoints,
                rep_count=self.total_reps + 1,
                stage=vbt_stage,
                angle=elbow_angle,
            )
        except Exception as e:
            logger.debug(f"VBT calculation error: {e}")

        try:
            if self.anthropometrics is None:
                self.anthropometrics = compute_anthropometrics(
                    keypoints[:, :2], keypoints[:, 2], self.config["keypoints"]
                )
                logger.info("Anthropometrics computed")
        except Exception as e:
            logger.debug(f"Anthropometrics computation error: {e}")

        if self.failure_justification_timer > 0:
            self.failure_justification_timer -= 1

        frame = self.visualizer.visualize(
            frame, keypoints[:, :2], elbow_angle,
            self.successful_reps, self.unsuccessful_reps, self.total_reps,
            self.state, self.facing_side or "right",
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
        Process front-view frame for dips specific analysis.
        
        Detects:
        - Asymmetric dipping (one arm dipping more than the other)
        - Body sway/lean during movement
        """
        if front_keypoints is None:
            return {}
        
        kpt_config = self.config.get("keypoints", {})
        
        # Compute front-view specific metrics
        arm_symmetry = self.compute_arm_symmetry(front_keypoints, kpt_config, frame_shape)
        shoulder_alignment = self.compute_shoulder_alignment(front_keypoints, kpt_config, frame_shape)
        hip_alignment = self.compute_hip_alignment(front_keypoints, kpt_config, frame_shape)
        
        # Check for body sway (hip vs shoulder alignment)
        body_sway_angle = abs(
            shoulder_alignment.get("shoulder_tilt_angle", 0) - 
            hip_alignment.get("hip_tilt_angle", 0)
        )
        
        # Update form flags during active rep
        if self.state in ["going_down", "coming_up"]:
            if arm_symmetry.get("asymmetric_detected"):
                self.asymmetric_dip_detected = True
                logger.debug(f"Asymmetric dip detected: score={arm_symmetry['symmetry_score']:.1f}")
            
            if body_sway_angle > 10.0:
                self.body_sway_detected = True
                logger.debug(f"Body sway detected: {body_sway_angle:.1f}°")
        
        self.last_front_analysis = {
            "arm_symmetry": arm_symmetry,
            "shoulder_alignment": shoulder_alignment,
            "hip_alignment": hip_alignment,
            "body_sway_angle": body_sway_angle,
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
            "shoulder_shrug_detected": self.shoulder_shrug_detected,
            "front_view_metrics": front_metrics,
            "dual_camera_active": True,
        }
        
        return processed_frame, combined_metadata
    
    def get_dual_camera_issues(self) -> list:
        """Get form issues detected from dual camera analysis."""
        issues = []
        
        if self.asymmetric_dip_detected:
            issues.append("asymmetric_dip")
        
        if self.body_sway_detected:
            issues.append("body_sway")
        
        return issues
    
    def reset_dual_camera_flags(self):
        """Reset dual camera form flags at start of new rep."""
        self.asymmetric_dip_detected = False
        self.body_sway_detected = False

    def finalize_analysis(self, output_dir: str, timestamp: str = None) -> None:
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
            logger.info(f"Rep {rep['rep_number']}: Success={rep['success']}, Issues={rep['issues']}")

        try:
            self.vbt_calculator.finalize_analysis()
            vbt_file_path = self.vbt_calculator.save_to_json(output_dir=output_dir, exercise_name="dips", timestamp=timestamp)
            logger.info(f"VBT analysis saved to: {vbt_file_path}")
        except Exception as e:
            logger.warning(f"VBT analysis error: {e}")

        logger.success(f"Final results - Successful: {self.successful_reps}, Unsuccessful: {self.unsuccessful_reps}, Total: {self.total_reps}")
        self.voice_player.cleanup()
        self._save_exercise_data(output_dir, timestamp)

    def _save_exercise_data(self, output_dir: str, timestamp: str = None) -> None:
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dips_analysis_details_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        data = {
            "exercise_type": "dips",
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
        return "dips"

    def get_results(self) -> Dict[str, Any]:
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
        return self.anthropometrics

    def get_voice_messages(self) -> list:
        return self.voice_messages.copy()
