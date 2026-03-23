"""
Bench Press exercise analyzer with integrated form analysis and voice feedback.
Analyzes eccentric descent, concentric pressing, and J-curve trajectory.
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


class BenchPressVisualizer:
    """Visualizer for bench press exercise analysis."""

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
            "knee": kpt[f"{prefix}_knee"],
            "ankle": kpt[f"{prefix}_ankle"],
            "ear": kpt[f"{prefix}_ear"],
        }

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
        # Basic visualization logic needed...
        return frame


class BenchPressExercise(DualCameraExerciseMixin):
    """Bench Press exercise analyzer."""

    def __init__(self, config: Dict[str, Any], fps: float = 30.0, segmenter: Any = None):
        self.config = config
        self.fps = fps
        self.cooldown_frames = int(fps * 0.5)
        self.state_change_cooldown = int(fps * 0.3)

        self.last_rep_frame = -self.cooldown_frames
        self.last_state_change_frame = 0

        # Rep counting variables
        self.successful_reps = 0
        self.unsuccessful_reps = 0
        self.total_reps = 0
        
        # State: begins at top waiting to unrack -> going_down -> going_up -> up
        self.state = "up"
        
        self.rep_start_frame = 0
        self.min_height_reached = False
        self.max_rom_this_rep = 0.0
        self.min_elbow_angle_this_rep = float('inf')
        
        self.last_failure_justification = []
        self.failure_justification_timer = 0
        self.failure_display_duration = int(fps * 3.0)

        DualCameraExerciseMixin.__init__(self)
        self.dual_camera_enabled = False
        self.front_view_keypoints = None
        self.last_front_analysis = None
        
        # Form flags
        self.head_lifted_detected = False
        self.forearm_not_vertical_detected = False
        self.fast_eccentric_detected = False
        
        # Summary logging
        self.reps_summary = []

        self.facing_side: Optional[str] = None
        self.side_determined = False
        
        self.vbt_calculator = VelocityCalculator(fps=fps)
        self.visualizer = BenchPressVisualizer(config)
        self.voice_player = VoiceMessagePlayer(config)
        self.voice_messages = []

        self.workout_started = False
        self.workout_ended = False
        self.last_voice_feedback_frame = -self.cooldown_frames
        self.voice_feedback_cooldown_frames = int(fps * 3)
        self.last_feedback_times = {}

        self.anthropometrics = None
        self.segmenter = segmenter

        self.experience_level = (self.config.get("experience", {}) or {}).get("level", "intermediate")
        self.thresholds = self._build_thresholds(self.experience_level)

    def _build_thresholds(self, level: str) -> Dict[str, float]:
        level = (level or "intermediate").lower()
        bp_config = self.config.get("bench_press", {}) or {}
        bp_thresholds = bp_config.get("thresholds", {})
        min_rep_duration = float(bp_config.get("min_rep_duration", 1.0))
        
        level_cfg = bp_thresholds.get(level, bp_thresholds.get("intermediate", {}))
        
        return {
            "shoulder_elbow_wrist_min": float(level_cfg.get("shoulder_elbow_wrist_min", 90.0)),
            "shoulder_elbow_wrist_max": float(level_cfg.get("shoulder_elbow_wrist_max", 170.0)),
            "forearm_vertical_tolerance": float(level_cfg.get("forearm_vertical_tolerance", 10.0)),
            "head_neck_horizontal_max": float(level_cfg.get("head_neck_horizontal_max", 10.0)),
            "eccentric_velocity_max": float(level_cfg.get("eccentric_velocity_max", 1.0)),
            "attempt_threshold": float(level_cfg.get("attempt_threshold", 0.35)),
            "success_threshold": float(level_cfg.get("success_threshold", 0.70)),
            "min_rep_duration": min_rep_duration,
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

    def calculate_horizontal_angle(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate angle of a line formed by p1-p2 relative to horizontal line."""
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        if dx == 0:
            return 90.0
        return np.degrees(np.arctan(abs(dy)/abs(dx)))

    def calculate_vertical_angle(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate angle of a line formed by p1-p2 relative to vertical line."""
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        if dy == 0:
            return 90.0
        return np.degrees(np.arctan(abs(dx)/abs(dy)))

    def _calculate_rom_percentage(self, current_elbow_angle: float) -> float:
        working_max = self.thresholds.get("shoulder_elbow_wrist_max", 170.0)
        working_min = self.thresholds.get("shoulder_elbow_wrist_min", 90.0)
        working_range = working_max - working_min

        if working_range <= 0:
            return 0.0

        rom_percentage = (working_max - current_elbow_angle) / working_range
        return max(0.0, min(1.0, rom_percentage))

    def process_frame(self, frame: np.ndarray, frame_number: int, keypoints: Optional[np.ndarray] = None, results: Any = None) -> np.ndarray:
        if keypoints is None and results is not None:
            if isinstance(results, list):
                if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                    keypoints = results[0].keypoints.xy[0].cpu().numpy()

        if not self.workout_started and keypoints is not None:
            self.voice_player.play_start_workout()
            self.voice_messages.append((frame_number / self.fps, "start_workout"))
            self.workout_started = True

        if keypoints is None:
            if hasattr(self, "_prev_elbow_angle"):
                delattr(self, "_prev_elbow_angle")
            return frame

        kpt_indices = self.visualizer._get_keypoint_indices(self.facing_side or "right")
        shoulder_point = keypoints[kpt_indices["shoulder"]][:2]
        elbow_point = keypoints[kpt_indices["elbow"]][:2]
        wrist_point = keypoints[kpt_indices["wrist"]][:2]
        ear_point = keypoints[kpt_indices["ear"]][:2]
        
        elbow_angle = self.calculate_angle(shoulder_point, elbow_point, wrist_point)
        self.current_head_neck_angle = self.calculate_horizontal_angle(ear_point, shoulder_point)
        self.current_forearm_angle = self.calculate_vertical_angle(wrist_point, elbow_point)

        if not hasattr(self, "_prev_elbow_angle"):
            self._prev_elbow_angle = elbow_angle
            self._elbow_velocity = 0.0
        else:
            self._elbow_velocity = elbow_angle - self._prev_elbow_angle
            self._prev_elbow_angle = elbow_angle

        self.update_rep_count(elbow_angle, frame_number)

        try:
            vbt_keypoints = np.array([shoulder_point, elbow_point, wrist_point])
            vbt_stage = "down" if self.state == "going_down" else "up"
            self.vbt_calculator.add_frame_data(
                frame_number=frame_number,
                keypoints=vbt_keypoints,
                rep_count=self.total_reps + 1,
                stage=vbt_stage,
                angle=elbow_angle,
            )
        except Exception:
            pass

        frame = self.visualizer.visualize(
            frame, keypoints, elbow_angle, self.successful_reps, self.unsuccessful_reps,
            self.total_reps, self.state, self.facing_side or "right", self.thresholds,
            self.last_failure_justification if self.failure_justification_timer > 0 else None
        )
        if self.failure_justification_timer > 0:
            self.failure_justification_timer -= 1
            
        return frame

    def update_rep_count(self, elbow_angle: float, frame_number: int) -> None:
        self.current_rom_percentage = self._calculate_rom_percentage(elbow_angle)

        if self.state in ["going_down", "going_up"]:
            self.max_rom_this_rep = max(self.max_rom_this_rep, self.current_rom_percentage)

        if self.state != "up" and (frame_number - self.last_state_change_frame > self.fps * 4):
            self.state = "up"
            self.rep_start_frame = 0
            self.min_height_reached = False
            self.head_lifted_detected = False
            self.forearm_not_vertical_detected = False
            self.fast_eccentric_detected = False
            return

        if self.state == "up":
            self._handle_state_up(elbow_angle, frame_number)
        elif self.state == "going_down":
            self._handle_state_descent(elbow_angle, frame_number)
        elif self.state == "going_up":
            self._handle_state_ascent(elbow_angle, frame_number)

    def _handle_state_up(self, elbow_angle: float, frame_number: int) -> None:
        state_change_allowed = (frame_number - self.last_state_change_frame > self.state_change_cooldown)
        elbow_max = self.thresholds.get("shoulder_elbow_wrist_max", 170.0)
        significant_bend = elbow_angle < (elbow_max - 5.0)

        if self._elbow_velocity < -1.0 and significant_bend and state_change_allowed:
            self.state = "going_down"
            self.min_height_reached = False
            self.min_elbow_angle_this_rep = 180.0
            self.last_state_change_frame = frame_number
            self.max_rom_this_rep = 0.0
            self.rep_start_frame = frame_number
            
            # Reset form flags
            self.head_lifted_detected = False
            self.forearm_not_vertical_detected = False
            self.fast_eccentric_detected = False

    def _handle_state_descent(self, elbow_angle: float, frame_number: int) -> None:
        self.min_elbow_angle_this_rep = min(self.min_elbow_angle_this_rep, elbow_angle)

        # Form Checks
        if abs(self.current_head_neck_angle) > self.thresholds.get("head_neck_horizontal_max", 10.0):
            self.head_lifted_detected = True
            
        if abs(self.current_forearm_angle) > self.thresholds.get("forearm_vertical_tolerance", 10.0):
            self.forearm_not_vertical_detected = True
            
        ecc_vel = self._elbow_velocity
        ecc_max = self.thresholds.get("eccentric_velocity_max", 1.0)
        if ecc_vel < -ecc_max: # Velocity is negative on the way down, check absolute rate
            self.fast_eccentric_detected = True

        rep_attempt_threshold = self.thresholds.get("attempt_threshold", 0.35)
        is_rep_attempt = self.current_rom_percentage >= rep_attempt_threshold
        
        if is_rep_attempt and not self.min_height_reached:
            self.min_height_reached = True

        state_change_allowed = (frame_number - self.last_state_change_frame > self.state_change_cooldown)
        if self._elbow_velocity > 1.0 and state_change_allowed:
            self.state = "going_up"
            self.last_state_change_frame = frame_number

    def _handle_state_ascent(self, elbow_angle: float, frame_number: int) -> None:
        if abs(self.current_head_neck_angle) > self.thresholds.get("head_neck_horizontal_max", 10.0):
            self.head_lifted_detected = True
            
        if abs(self.current_forearm_angle) > self.thresholds.get("forearm_vertical_tolerance", 10.0):
            self.forearm_not_vertical_detected = True

        # Check for completion
        elbow_max = self.thresholds.get("shoulder_elbow_wrist_max", 170.0)
        is_near_top = elbow_angle > elbow_max - 10.0
        is_moving_slowly = self._elbow_velocity > -1.0

        if is_near_top and is_moving_slowly:
            if frame_number - self.last_rep_frame > self.cooldown_frames:
                self._complete_rep(frame_number)

    def _complete_rep(self, frame_number: int) -> None:
        if not self.min_height_reached:
            self.state = "up"
            return
            
        rep_duration = (frame_number - self.rep_start_frame) / self.fps
        min_rep_duration = self.thresholds.get("min_rep_duration", 1.0)
        if rep_duration < min_rep_duration:
            self.state = "up"
            return

        self.total_reps += 1
        self.last_rep_frame = frame_number
        self.last_state_change_frame = frame_number

        success_threshold_val = self.thresholds.get("success_threshold", 0.70)
        depth_success = self.max_rom_this_rep >= success_threshold_val 

        form_issues = []
        failure_justifications = {}

        if not depth_success:
            form_issues.append("insufficient_depth")
            failure_justifications["insufficient_depth"] = f"ROM {self.max_rom_this_rep:.2f} < {success_threshold_val}"

        if self.head_lifted_detected:
            form_issues.append("head_lifted")
            failure_justifications["head_lifted"] = "Head deviated significantly from bench (horizontal)"
            
        if self.forearm_not_vertical_detected:
            form_issues.append("forearms_not_vertical")
            failure_justifications["forearms_not_vertical"] = "Forearms exceeded vertical tolerance during rep"
            
        if self.fast_eccentric_detected:
            form_issues.append("dropping_fast")
            failure_justifications["dropping_fast"] = "Descent was too fast, lack of control"

        success = len(form_issues) == 0 and depth_success

        if success:
            self.successful_reps += 1
            logger.success(f"✅ Bench Press Rep {self.total_reps} Good!")
        else:
            self.unsuccessful_reps += 1
            justification_str = " | ".join([f"{k}: {v}" for k, v in failure_justifications.items()])
            logger.warning(f"❌ Rep Failed: {form_issues} | {justification_str}")
            self.last_failure_justification = [f"{v}" for k, v in failure_justifications.items()]
            self.failure_justification_timer = self.failure_display_duration

        self.reps_summary.append({
            "rep_number": self.total_reps,
            "success": success,
            "rom_percentage": self.max_rom_this_rep,
            "issues": form_issues,
            "justifications": failure_justifications
        })

        self.state = "up"

    def get_exercise_name(self) -> str:
        return "bench_press"

    def get_results(self) -> Dict[str, Any]:
        return {}
