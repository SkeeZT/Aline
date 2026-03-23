"""
Dual Camera Exercise Mixin

Provides common functionality for exercises that support dual-camera analysis
(front view + side view) for more comprehensive form detection.
"""

import math
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger


class DualCameraMetrics:
    """Container for metrics computed from both camera views."""
    
    def __init__(self):
        # Side view metrics
        self.side_detected = False
        self.side_angles: Dict[str, float] = {}
        self.side_issues: List[str] = []
        
        # Front view metrics
        self.front_detected = False
        self.front_angles: Dict[str, float] = {}
        self.front_issues: List[str] = []
        
        # Combined metrics
        self.combined_issues: List[str] = []
        self.confidence_score: float = 0.0
    
    def merge_issues(self) -> List[str]:
        """Merge issues from both views, removing duplicates."""
        all_issues = list(set(self.side_issues + self.front_issues))
        self.combined_issues = all_issues
        return all_issues


class DualCameraExerciseMixin:
    """
    Mixin class providing dual-camera analysis capabilities for exercises.
    
    This mixin adds methods for:
    - Front-view specific detections (knee valgus, hip alignment, symmetry)
    - Combining metrics from both camera views
    - Calculating confidence based on both views
    """
    
    def __init__(self):
        # Dual camera state
        self.dual_camera_enabled = False
        self.last_front_metrics: Optional[Dict[str, Any]] = None
        self.last_side_metrics: Optional[Dict[str, Any]] = None
        self.dual_metrics = DualCameraMetrics()
    
    def enable_dual_camera(self, enabled: bool = True) -> None:
        """Enable or disable dual-camera mode."""
        self.dual_camera_enabled = enabled
        logger.info(f"Dual camera mode {'enabled' if enabled else 'disabled'}")
    
    # FRONT VIEW COMMON METRICS
    
    def compute_knee_valgus(
        self,
        keypoints: np.ndarray,
        kpt_config: Dict[str, int],
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Compute knee valgus/varus from front view.
        
        Knee valgus: Knees collapse inward (knock-knees)
        Knee varus: Knees bow outward
        
        Returns dict with:
            - left_valgus_angle: Angle deviation for left knee
            - right_valgus_angle: Angle deviation for right knee
            - valgus_detected: True if significant valgus detected
            - varus_detected: True if significant varus detected
        """
        h, w = frame_shape[:2]
        
        try:
            # Get keypoint indices
            l_hip_idx = kpt_config["left_hip"]
            r_hip_idx = kpt_config["right_hip"]
            l_knee_idx = kpt_config["left_knee"]
            r_knee_idx = kpt_config["right_knee"]
            l_ankle_idx = kpt_config["left_ankle"]
            r_ankle_idx = kpt_config["right_ankle"]
            
            # Extract points (handle both normalized and pixel coords)
            if keypoints.max() <= 1.0:  # Normalized
                scale = np.array([w, h])
                l_hip = keypoints[l_hip_idx][:2] * scale
                r_hip = keypoints[r_hip_idx][:2] * scale
                l_knee = keypoints[l_knee_idx][:2] * scale
                r_knee = keypoints[r_knee_idx][:2] * scale
                l_ankle = keypoints[l_ankle_idx][:2] * scale
                r_ankle = keypoints[r_ankle_idx][:2] * scale
            else:  # Pixel coords
                l_hip = keypoints[l_hip_idx][:2]
                r_hip = keypoints[r_hip_idx][:2]
                l_knee = keypoints[l_knee_idx][:2]
                r_knee = keypoints[r_knee_idx][:2]
                l_ankle = keypoints[l_ankle_idx][:2]
                r_ankle = keypoints[r_ankle_idx][:2]
            
            # Calculate ideal knee position (straight line from hip to ankle)
            # Valgus/Varus is the deviation of knee from this line
            
            def calculate_knee_deviation(hip, knee, ankle):
                """Calculate how much knee deviates from hip-ankle line."""
                # Vector from hip to ankle
                hip_ankle = ankle - hip
                # Vector from hip to knee
                hip_knee = knee - hip
                
                # Project knee onto hip-ankle line
                hip_ankle_len = np.linalg.norm(hip_ankle)
                if hip_ankle_len < 1:
                    return 0.0, "neutral"
                
                # Normalized direction
                direction = hip_ankle / hip_ankle_len
                
                # Projection length
                proj_len = np.dot(hip_knee, direction)
                
                # Projected point
                projected = hip + direction * proj_len
                
                # Deviation (positive = valgus/inward, negative = varus/outward)
                deviation = knee[0] - projected[0]
                
                # Normalize by leg length
                deviation_ratio = deviation / hip_ankle_len
                
                # Convert to angle (approximate)
                deviation_angle = math.degrees(math.atan2(abs(deviation), proj_len))
                
                if deviation_ratio > 0.05:  # Knee is inward
                    return deviation_angle, "valgus"
                elif deviation_ratio < -0.05:  # Knee is outward
                    return deviation_angle, "varus"
                else:
                    return deviation_angle, "neutral"
            
            left_angle, left_type = calculate_knee_deviation(l_hip, l_knee, l_ankle)
            right_angle, right_type = calculate_knee_deviation(r_hip, r_knee, r_ankle)
            
            # Thresholds for detection
            config = getattr(self, "config", {})
            fv_thresholds = config.get("front_view_thresholds", {})
            valgus_threshold = float(fv_thresholds.get("knee_valgus_error", 15.0))  # degrees
            
            return {
                "left_valgus_angle": left_angle,
                "right_valgus_angle": right_angle,
                "left_type": left_type,
                "right_type": right_type,
                "valgus_detected": (left_type == "valgus" and left_angle > valgus_threshold) or 
                                   (right_type == "valgus" and right_angle > valgus_threshold),
                "varus_detected": (left_type == "varus" and left_angle > valgus_threshold) or 
                                  (right_type == "varus" and right_angle > valgus_threshold),
            }
            
        except Exception as e:
            logger.debug(f"Error computing knee valgus: {e}")
            return {
                "left_valgus_angle": 0.0,
                "right_valgus_angle": 0.0,
                "left_type": "unknown",
                "right_type": "unknown",
                "valgus_detected": False,
                "varus_detected": False,
            }
    
    def compute_hip_alignment(
        self,
        keypoints: np.ndarray,
        kpt_config: Dict[str, int],
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Compute hip alignment from front view.
        
        Checks if hips are level (not dropping on one side).
        
        Returns dict with:
            - hip_tilt_angle: Angle of hip line from horizontal
            - hip_drop_side: "left", "right", or "level"
            - hip_drop_detected: True if significant drop detected
        """
        h, w = frame_shape[:2]
        
        try:
            l_hip_idx = kpt_config["left_hip"]
            r_hip_idx = kpt_config["right_hip"]
            
            # Extract points
            if keypoints.max() <= 1.0:
                scale = np.array([w, h])
                l_hip = keypoints[l_hip_idx][:2] * scale
                r_hip = keypoints[r_hip_idx][:2] * scale
            else:
                l_hip = keypoints[l_hip_idx][:2]
                r_hip = keypoints[r_hip_idx][:2]
            
            # Calculate tilt angle
            dx = r_hip[0] - l_hip[0]
            dy = r_hip[1] - l_hip[1]  # Positive = right hip lower
            
            if abs(dx) < 1:
                tilt_angle = 90.0 if dy > 0 else -90.0
            else:
                tilt_angle = math.degrees(math.atan2(dy, dx))
            
            # Determine drop side
            hip_drop_threshold = 5.0  # degrees
            
            if tilt_angle > hip_drop_threshold:
                drop_side = "right"
                drop_detected = True
            elif tilt_angle < -hip_drop_threshold:
                drop_side = "left"
                drop_detected = True
            else:
                drop_side = "level"
                drop_detected = False
            
            return {
                "hip_tilt_angle": abs(tilt_angle),
                "hip_drop_side": drop_side,
                "hip_drop_detected": drop_detected,
                "hip_vertical_diff": abs(dy),
            }
            
        except Exception as e:
            logger.debug(f"Error computing hip alignment: {e}")
            return {
                "hip_tilt_angle": 0.0,
                "hip_drop_side": "unknown",
                "hip_drop_detected": False,
                "hip_vertical_diff": 0.0,
            }
    
    def compute_arm_symmetry(
        self,
        keypoints: np.ndarray,
        kpt_config: Dict[str, int],
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Compute arm symmetry from front view.
        
        Useful for Pull-ups, Overhead Press, Dips.
        
        Returns dict with:
            - elbow_height_diff: Height difference between elbows
            - wrist_height_diff: Height difference between wrists
            - symmetry_score: 0-100 score (100 = perfect symmetry)
            - asymmetric_detected: True if significant asymmetry
        """
        h, w = frame_shape[:2]
        
        try:
            l_elbow_idx = kpt_config["left_elbow"]
            r_elbow_idx = kpt_config["right_elbow"]
            l_wrist_idx = kpt_config["left_wrist"]
            r_wrist_idx = kpt_config["right_wrist"]
            l_shoulder_idx = kpt_config["left_shoulder"]
            r_shoulder_idx = kpt_config["right_shoulder"]
            
            if keypoints.max() <= 1.0:
                scale = np.array([w, h])
                l_elbow = keypoints[l_elbow_idx][:2] * scale
                r_elbow = keypoints[r_elbow_idx][:2] * scale
                l_wrist = keypoints[l_wrist_idx][:2] * scale
                r_wrist = keypoints[r_wrist_idx][:2] * scale
                l_shoulder = keypoints[l_shoulder_idx][:2] * scale
                r_shoulder = keypoints[r_shoulder_idx][:2] * scale
            else:
                l_elbow = keypoints[l_elbow_idx][:2]
                r_elbow = keypoints[r_elbow_idx][:2]
                l_wrist = keypoints[l_wrist_idx][:2]
                r_wrist = keypoints[r_wrist_idx][:2]
                l_shoulder = keypoints[l_shoulder_idx][:2]
                r_shoulder = keypoints[r_shoulder_idx][:2]
            
            # Calculate height differences
            elbow_height_diff = abs(l_elbow[1] - r_elbow[1])
            wrist_height_diff = abs(l_wrist[1] - r_wrist[1])
            
            # Calculate arm lengths for normalization
            l_arm_len = np.linalg.norm(l_shoulder - l_wrist)
            r_arm_len = np.linalg.norm(r_shoulder - r_wrist)
            avg_arm_len = (l_arm_len + r_arm_len) / 2
            
            if avg_arm_len < 1:
                return {
                    "elbow_height_diff": 0.0,
                    "wrist_height_diff": 0.0,
                    "symmetry_score": 100.0,
                    "asymmetric_detected": False,
                }
            
            # Normalize differences
            elbow_diff_norm = elbow_height_diff / avg_arm_len
            wrist_diff_norm = wrist_height_diff / avg_arm_len
            
            # Calculate symmetry score (100 = perfect)
            avg_diff = (elbow_diff_norm + wrist_diff_norm) / 2
            symmetry_score = max(0, 100 - avg_diff * 500)  # 20% diff = 0 score
            
            # Threshold for asymmetry detection
            asymmetry_threshold = 0.1  # 10% of arm length
            asymmetric = elbow_diff_norm > asymmetry_threshold or wrist_diff_norm > asymmetry_threshold
            
            return {
                "elbow_height_diff": elbow_height_diff,
                "wrist_height_diff": wrist_height_diff,
                "elbow_diff_normalized": elbow_diff_norm,
                "wrist_diff_normalized": wrist_diff_norm,
                "symmetry_score": symmetry_score,
                "asymmetric_detected": asymmetric,
            }
            
        except Exception as e:
            logger.debug(f"Error computing arm symmetry: {e}")
            return {
                "elbow_height_diff": 0.0,
                "wrist_height_diff": 0.0,
                "symmetry_score": 100.0,
                "asymmetric_detected": False,
            }
    
    def compute_stance_width(
        self,
        keypoints: np.ndarray,
        kpt_config: Dict[str, int],
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Compute stance width from front view.
        
        Useful for Squat, Deadlift.
        
        Returns dict with:
            - stance_width_px: Width in pixels
            - stance_width_ratio: Width relative to hip width
            - stance_type: "narrow", "normal", "wide"
        """
        h, w = frame_shape[:2]
        
        try:
            l_ankle_idx = kpt_config["left_ankle"]
            r_ankle_idx = kpt_config["right_ankle"]
            l_hip_idx = kpt_config["left_hip"]
            r_hip_idx = kpt_config["right_hip"]
            
            if keypoints.max() <= 1.0:
                scale = np.array([w, h])
                l_ankle = keypoints[l_ankle_idx][:2] * scale
                r_ankle = keypoints[r_ankle_idx][:2] * scale
                l_hip = keypoints[l_hip_idx][:2] * scale
                r_hip = keypoints[r_hip_idx][:2] * scale
            else:
                l_ankle = keypoints[l_ankle_idx][:2]
                r_ankle = keypoints[r_ankle_idx][:2]
                l_hip = keypoints[l_hip_idx][:2]
                r_hip = keypoints[r_hip_idx][:2]
            
            # Calculate widths
            stance_width = abs(l_ankle[0] - r_ankle[0])
            hip_width = abs(l_hip[0] - r_hip[0])
            
            if hip_width < 1:
                return {
                    "stance_width_px": stance_width,
                    "stance_width_ratio": 1.0,
                    "stance_type": "normal",
                }
            
            stance_ratio = stance_width / hip_width
            
            # Classify stance
            if stance_ratio < 0.8:
                stance_type = "narrow"
            elif stance_ratio > 1.5:
                stance_type = "wide"
            else:
                stance_type = "normal"
            
            return {
                "stance_width_px": stance_width,
                "hip_width_px": hip_width,
                "stance_width_ratio": stance_ratio,
                "stance_type": stance_type,
            }
            
        except Exception as e:
            logger.debug(f"Error computing stance width: {e}")
            return {
                "stance_width_px": 0.0,
                "stance_width_ratio": 1.0,
                "stance_type": "unknown",
            }
    
    def compute_shoulder_alignment(
        self,
        keypoints: np.ndarray,
        kpt_config: Dict[str, int],
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Compute shoulder alignment from front view.
        
        Checks if shoulders are level and properly positioned.
        
        Returns dict with:
            - shoulder_tilt_angle: Angle from horizontal
            - shoulder_drop_side: "left", "right", or "level"
        """
        h, w = frame_shape[:2]
        
        try:
            l_shoulder_idx = kpt_config["left_shoulder"]
            r_shoulder_idx = kpt_config["right_shoulder"]
            
            if keypoints.max() <= 1.0:
                scale = np.array([w, h])
                l_shoulder = keypoints[l_shoulder_idx][:2] * scale
                r_shoulder = keypoints[r_shoulder_idx][:2] * scale
            else:
                l_shoulder = keypoints[l_shoulder_idx][:2]
                r_shoulder = keypoints[r_shoulder_idx][:2]
            
            dx = r_shoulder[0] - l_shoulder[0]
            dy = r_shoulder[1] - l_shoulder[1]
            
            if abs(dx) < 1:
                tilt_angle = 90.0 if dy > 0 else -90.0
            else:
                tilt_angle = math.degrees(math.atan2(dy, dx))
            
            tilt_threshold = 5.0
            
            if tilt_angle > tilt_threshold:
                drop_side = "right"
            elif tilt_angle < -tilt_threshold:
                drop_side = "left"
            else:
                drop_side = "level"
            
            return {
                "shoulder_tilt_angle": abs(tilt_angle),
                "shoulder_drop_side": drop_side,
                "shoulder_width": abs(dx),
            }
            
        except Exception as e:
            logger.debug(f"Error computing shoulder alignment: {e}")
            return {
                "shoulder_tilt_angle": 0.0,
                "shoulder_drop_side": "unknown",
                "shoulder_width": 0.0,
            }
    
    def calculate_angle_3point(
        self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
    ) -> float:
        """Calculate angle between three points (angle at p2)."""
        v1 = p1 - p2
        v2 = p3 - p2
        
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norm < 1e-6:
            return 0.0
        
        cos_angle = np.clip(dot / norm, -1.0, 1.0)
        return math.degrees(math.acos(cos_angle))
    
    # DUAL VIEW PROCESSING
    
    def process_dual_frame(
        self,
        side_keypoints: Optional[np.ndarray],
        front_keypoints: Optional[np.ndarray],
        frame_number: int,
        side_frame_shape: Tuple[int, int],
        front_frame_shape: Tuple[int, int],
    ) -> DualCameraMetrics:
        """
        Process frames from both cameras and combine metrics.
        
        Override this method in subclasses for exercise-specific logic.
        """
        metrics = DualCameraMetrics()
        
        if side_keypoints is not None:
            metrics.side_detected = True
            # Subclasses should populate side_angles and side_issues
        
        if front_keypoints is not None:
            metrics.front_detected = True
            # Subclasses should populate front_angles and front_issues
        
        # Calculate confidence based on detection
        if metrics.side_detected and metrics.front_detected:
            metrics.confidence_score = 1.0
        elif metrics.side_detected or metrics.front_detected:
            metrics.confidence_score = 0.6
        else:
            metrics.confidence_score = 0.0
        
        metrics.merge_issues()
        return metrics
