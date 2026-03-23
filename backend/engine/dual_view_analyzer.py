"""
Dual View Exercise Analyzer

Combines pose estimation from side and front cameras for
comprehensive exercise form analysis.

Side view detects:
- Depth/ROM (knee angle, hip angle)
- Forward lean
- Back curvature
- Knees over toes

Front view detects:
- Knee valgus (knees caving in)
- Shoulder symmetry
- Weight shift (left/right imbalance)
- Elbow flare
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from loguru import logger

from engine.dual_camera_manager import SyncedFrame, CameraPosition


@dataclass
class DualPoseData:
    """Combined pose data from both camera views."""
    
    # Side view keypoints and angles
    side_keypoints: Optional[np.ndarray] = None
    side_confidence: float = 0.0
    
    # Front view keypoints and angles
    front_keypoints: Optional[np.ndarray] = None
    front_confidence: float = 0.0
    
    # Derived metrics
    knee_angle: float = 180.0
    hip_angle: float = 180.0
    elbow_angle: float = 180.0
    
    # Front-view specific
    knee_valgus_left: float = 0.0  # Inward deviation in pixels
    knee_valgus_right: float = 0.0
    shoulder_tilt: float = 0.0  # Degrees from horizontal
    hip_tilt: float = 0.0
    weight_balance: float = 0.5  # 0 = all left, 1 = all right, 0.5 = balanced
    
    # Side-view specific
    knees_over_toes_distance: float = 0.0  # Pixels past toes
    forward_lean_angle: float = 0.0
    back_curvature: float = 0.0
    
    # Combined confidence
    @property
    def confidence(self) -> float:
        if self.side_confidence > 0 and self.front_confidence > 0:
            return (self.side_confidence + self.front_confidence) / 2
        return max(self.side_confidence, self.front_confidence)
    
    @property
    def has_both_views(self) -> bool:
        return self.side_keypoints is not None and self.front_keypoints is not None


@dataclass
class DualViewIssue:
    """Form issue detected from dual camera analysis."""
    name: str
    severity: str  # "warning", "error"
    source: str  # "side", "front", "combined"
    message: str
    correction: str


class DualViewAnalyzer:
    """
    Analyzes exercise form using both side and front camera views.
    
    Combines pose estimation results to detect issues that are only
    visible from specific angles.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        exercise_type: str = "squat",
        experience_level: str = "intermediate",
    ):
        self.config = config
        self.exercise_type = exercise_type
        self.experience_level = experience_level
        
        # Get keypoint indices
        self.kpt_config = config.get("keypoints", {})
        
        # Issue detection thresholds
        self.thresholds = self._build_thresholds()
        
        # State
        self.current_pose: Optional[DualPoseData] = None
        self.current_issues: List[DualViewIssue] = []
        
        logger.info(f"DualViewAnalyzer initialized for {exercise_type}")
    
    def _build_thresholds(self) -> Dict[str, float]:
        """Build thresholds for dual-view analysis."""
        level = self.experience_level.lower()
        
        # Base thresholds
        thresholds = {
            # Knee valgus (pixels inward from ankle)
            "knee_valgus_warning": 20.0 if level == "beginner" else 15.0 if level == "intermediate" else 10.0,
            "knee_valgus_error": 40.0 if level == "beginner" else 30.0 if level == "intermediate" else 20.0,
            
            # Shoulder tilt (degrees from horizontal)
            "shoulder_tilt_warning": 8.0 if level == "beginner" else 5.0 if level == "intermediate" else 3.0,
            "shoulder_tilt_error": 15.0 if level == "beginner" else 10.0 if level == "intermediate" else 7.0,
            
            # Hip tilt (degrees from horizontal)
            "hip_tilt_warning": 10.0 if level == "beginner" else 7.0 if level == "intermediate" else 5.0,
            "hip_tilt_error": 18.0 if level == "beginner" else 12.0 if level == "intermediate" else 8.0,
            
            # Weight balance (deviation from 0.5)
            "weight_balance_warning": 0.15,
            "weight_balance_error": 0.25,
            
            # Knees over toes (pixels)
            "knees_over_toes_warning": 30.0 if level == "beginner" else 20.0 if level == "intermediate" else 15.0,
            "knees_over_toes_error": 50.0 if level == "beginner" else 40.0 if level == "intermediate" else 30.0,
        }
        
        return thresholds
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by p1-p2-p3."""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def _calculate_tilt(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate tilt angle from horizontal (degrees)."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.degrees(np.arctan2(dy, dx))
    
    def _get_keypoint(
        self, 
        keypoints: np.ndarray, 
        name: str, 
        side: str = "right"
    ) -> Optional[np.ndarray]:
        """Get keypoint by name."""
        key = f"{side}_{name}" if side else name
        idx = self.kpt_config.get(key)
        if idx is not None and idx < len(keypoints):
            return keypoints[idx][:2]
        return None
    
    def analyze_side_view(
        self, 
        keypoints: np.ndarray,
        facing_side: str = "right"
    ) -> Dict[str, float]:
        """Analyze side view for depth, lean, and position."""
        results = {
            "knee_angle": 180.0,
            "hip_angle": 180.0,
            "knees_over_toes_distance": 0.0,
            "forward_lean_angle": 0.0,
        }
        
        # Determine which side keypoints to use
        side = "left" if facing_side == "left" else "right"
        
        try:
            # Get keypoints
            shoulder = self._get_keypoint(keypoints, "shoulder", side)
            hip = self._get_keypoint(keypoints, "hip", side)
            knee = self._get_keypoint(keypoints, "knee", side)
            ankle = self._get_keypoint(keypoints, "ankle", side)
            
            if all(p is not None for p in [shoulder, hip, knee, ankle]):
                # Knee angle (hip-knee-ankle)
                results["knee_angle"] = self._calculate_angle(hip, knee, ankle)
                
                # Hip angle (shoulder-hip-knee)
                results["hip_angle"] = self._calculate_angle(shoulder, hip, knee)
                
                # Knees over toes (horizontal distance)
                # Positive = knee past ankle
                toe_x = ankle[0]  # Approximation - use ankle as toe reference
                results["knees_over_toes_distance"] = knee[0] - toe_x
                
                # Forward lean (torso angle from vertical)
                torso_angle = self._calculate_tilt(hip, shoulder)
                results["forward_lean_angle"] = 90 - abs(torso_angle)
        
        except Exception as e:
            logger.debug(f"Error analyzing side view: {e}")
        
        return results
    
    def analyze_front_view(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Analyze front view for symmetry and alignment."""
        results = {
            "knee_valgus_left": 0.0,
            "knee_valgus_right": 0.0,
            "shoulder_tilt": 0.0,
            "hip_tilt": 0.0,
            "weight_balance": 0.5,
        }
        
        try:
            # Get bilateral keypoints
            left_shoulder = self._get_keypoint(keypoints, "shoulder", "left")
            right_shoulder = self._get_keypoint(keypoints, "shoulder", "right")
            left_hip = self._get_keypoint(keypoints, "hip", "left")
            right_hip = self._get_keypoint(keypoints, "hip", "right")
            left_knee = self._get_keypoint(keypoints, "knee", "left")
            right_knee = self._get_keypoint(keypoints, "knee", "right")
            left_ankle = self._get_keypoint(keypoints, "ankle", "left")
            right_ankle = self._get_keypoint(keypoints, "ankle", "right")
            
            # Shoulder tilt
            if left_shoulder is not None and right_shoulder is not None:
                results["shoulder_tilt"] = self._calculate_tilt(left_shoulder, right_shoulder)
            
            # Hip tilt
            if left_hip is not None and right_hip is not None:
                results["hip_tilt"] = self._calculate_tilt(left_hip, right_hip)
            
            # Knee valgus detection
            # Valgus = knee moves inward relative to ankle-hip line
            if all(p is not None for p in [left_hip, left_knee, left_ankle]):
                # Expected knee X position on line from hip to ankle
                t = (left_knee[1] - left_hip[1]) / (left_ankle[1] - left_hip[1] + 1e-6)
                expected_x = left_hip[0] + t * (left_ankle[0] - left_hip[0])
                # Positive = knee inside (valgus)
                results["knee_valgus_left"] = expected_x - left_knee[0]
            
            if all(p is not None for p in [right_hip, right_knee, right_ankle]):
                t = (right_knee[1] - right_hip[1]) / (right_ankle[1] - right_hip[1] + 1e-6)
                expected_x = right_hip[0] + t * (right_ankle[0] - right_hip[0])
                # Positive = knee inside (valgus)
                results["knee_valgus_right"] = right_knee[0] - expected_x
            
            # Weight balance estimation (based on hip position relative to ankles)
            if all(p is not None for p in [left_ankle, right_ankle, left_hip, right_hip]):
                hip_center_x = (left_hip[0] + right_hip[0]) / 2
                ankle_center_x = (left_ankle[0] + right_ankle[0]) / 2
                ankle_width = abs(right_ankle[0] - left_ankle[0])
                
                if ankle_width > 0:
                    # Normalize: 0 = all left, 1 = all right, 0.5 = center
                    shift = (hip_center_x - ankle_center_x) / (ankle_width / 2)
                    results["weight_balance"] = 0.5 + (shift * 0.25)  # Scale to 0.25-0.75 range
                    results["weight_balance"] = np.clip(results["weight_balance"], 0.0, 1.0)
        
        except Exception as e:
            logger.debug(f"Error analyzing front view: {e}")
        
        return results
    
    def detect_issues(self, pose_data: DualPoseData) -> List[DualViewIssue]:
        """Detect form issues from combined pose data."""
        issues = []
        
        # Front view issues
        if pose_data.front_keypoints is not None:
            # Knee valgus
            max_valgus = max(pose_data.knee_valgus_left, pose_data.knee_valgus_right)
            if max_valgus > self.thresholds["knee_valgus_error"]:
                issues.append(DualViewIssue(
                    name="knee_valgus",
                    severity="error",
                    source="front",
                    message=f"Knees caving in ({max_valgus:.0f}px)",
                    correction="Push your knees out over your toes"
                ))
            elif max_valgus > self.thresholds["knee_valgus_warning"]:
                issues.append(DualViewIssue(
                    name="knee_valgus",
                    severity="warning",
                    source="front",
                    message="Slight knee valgus detected",
                    correction="Focus on keeping knees tracking over toes"
                ))
            
            # Shoulder asymmetry
            if abs(pose_data.shoulder_tilt) > self.thresholds["shoulder_tilt_error"]:
                issues.append(DualViewIssue(
                    name="shoulder_asymmetry",
                    severity="error",
                    source="front",
                    message=f"Shoulder tilt ({pose_data.shoulder_tilt:.1f}°)",
                    correction="Keep shoulders level and square"
                ))
            elif abs(pose_data.shoulder_tilt) > self.thresholds["shoulder_tilt_warning"]:
                issues.append(DualViewIssue(
                    name="shoulder_asymmetry",
                    severity="warning",
                    source="front",
                    message="Slight shoulder asymmetry",
                    correction="Check shoulder position"
                ))
            
            # Hip shift / weight balance
            balance_deviation = abs(pose_data.weight_balance - 0.5)
            if balance_deviation > self.thresholds["weight_balance_error"]:
                side = "left" if pose_data.weight_balance < 0.5 else "right"
                issues.append(DualViewIssue(
                    name="weight_shift",
                    severity="error",
                    source="front",
                    message=f"Weight shifted to {side}",
                    correction="Distribute weight evenly between both legs"
                ))
            elif balance_deviation > self.thresholds["weight_balance_warning"]:
                issues.append(DualViewIssue(
                    name="weight_shift",
                    severity="warning",
                    source="front",
                    message="Slight weight imbalance detected",
                    correction="Center your weight"
                ))
        
        # Side view issues
        if pose_data.side_keypoints is not None:
            # Knees over toes
            if pose_data.knees_over_toes_distance > self.thresholds["knees_over_toes_error"]:
                issues.append(DualViewIssue(
                    name="knees_over_toes",
                    severity="error",
                    source="side",
                    message=f"Knees past toes ({pose_data.knees_over_toes_distance:.0f}px)",
                    correction="Sit back into the movement, weight on heels"
                ))
            elif pose_data.knees_over_toes_distance > self.thresholds["knees_over_toes_warning"]:
                issues.append(DualViewIssue(
                    name="knees_over_toes",
                    severity="warning",
                    source="side",
                    message="Knees approaching toe line",
                    correction="Shift weight back slightly"
                ))
        
        return issues
    
    def process_synced_frame(
        self,
        synced_frame: SyncedFrame,
        side_keypoints: Optional[np.ndarray],
        front_keypoints: Optional[np.ndarray],
        facing_side: str = "right",
    ) -> DualPoseData:
        """
        Process a synchronized frame pair and extract pose data.
        
        Args:
            synced_frame: Synchronized frame container
            side_keypoints: Keypoints from side camera (from pose estimator)
            front_keypoints: Keypoints from front camera (from pose estimator)
            facing_side: Which side the subject is facing
        
        Returns:
            Combined pose data from both views
        """
        pose_data = DualPoseData()
        pose_data.side_keypoints = side_keypoints
        pose_data.front_keypoints = front_keypoints
        
        # Analyze side view
        if side_keypoints is not None:
            pose_data.side_confidence = np.mean(side_keypoints[:, 2]) if side_keypoints.shape[1] > 2 else 1.0
            side_results = self.analyze_side_view(side_keypoints, facing_side)
            pose_data.knee_angle = side_results["knee_angle"]
            pose_data.hip_angle = side_results["hip_angle"]
            pose_data.knees_over_toes_distance = side_results["knees_over_toes_distance"]
            pose_data.forward_lean_angle = side_results["forward_lean_angle"]
        
        # Analyze front view
        if front_keypoints is not None:
            pose_data.front_confidence = np.mean(front_keypoints[:, 2]) if front_keypoints.shape[1] > 2 else 1.0
            front_results = self.analyze_front_view(front_keypoints)
            pose_data.knee_valgus_left = front_results["knee_valgus_left"]
            pose_data.knee_valgus_right = front_results["knee_valgus_right"]
            pose_data.shoulder_tilt = front_results["shoulder_tilt"]
            pose_data.hip_tilt = front_results["hip_tilt"]
            pose_data.weight_balance = front_results["weight_balance"]
        
        # Detect issues
        self.current_issues = self.detect_issues(pose_data)
        self.current_pose = pose_data
        
        return pose_data
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for visualization overlay."""
        if self.current_pose is None:
            return {}
        
        return {
            "knee_angle": self.current_pose.knee_angle,
            "hip_angle": self.current_pose.hip_angle,
            "knee_valgus": max(self.current_pose.knee_valgus_left, self.current_pose.knee_valgus_right),
            "shoulder_tilt": self.current_pose.shoulder_tilt,
            "weight_balance": self.current_pose.weight_balance,
            "issues": [
                {"name": i.name, "severity": i.severity, "message": i.message}
                for i in self.current_issues
            ],
            "has_both_views": self.current_pose.has_both_views,
            "side_confidence": self.current_pose.side_confidence,
            "front_confidence": self.current_pose.front_confidence,
        }
