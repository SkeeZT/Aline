"""
Velocity-Based Training (VBT) calculations for squat analysis.
This module calculates velocity metrics for each rep based on keypoint movements.
"""

import os
import json
import numpy as np
from loguru import logger
from datetime import datetime
from scipy.signal import savgol_filter
from typing import Dict, List, Optional, Any


class VelocityCalculator:
    """Class for calculating velocity-based training metrics for squats."""

    def __init__(self, fps: float = 30.0, smoothing_window: int = 5):
        """
        Initialize the velocity calculator.

        Args:
            fps: Frames per second of the video
            smoothing_window: Window size for smoothing velocity calculations
        """
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.frame_data = []  # Store all frame data for analysis
        self.rep_velocities = []  # Store velocity data for each rep

        # Tracking variables
        self.current_rep_data = None
        self.previous_position = None
        self.previous_frame = None

    def add_frame_data(
        self,
        frame_number: int,
        keypoints: np.ndarray,
        rep_count: int,
        stage: str,
        angle: float,
    ) -> None:
        """
        Add frame data for velocity calculation.

        Args:
            frame_number: Current frame number
            keypoints: Array of keypoints [hip, knee, ankle] with shape (3, 3) [x, y, confidence]
            rep_count: Current rep count
            stage: Current stage ('up' or 'down')
            angle: Current knee angle
        """
        # Validate keypoints
        if keypoints is None or keypoints.shape[0] != 3 or keypoints.shape[1] != 3:
            return

        # Check if all keypoints are detected (confidence > 0)
        if not all(kpt[2] > 0 for kpt in keypoints):
            return

        # Calculate center of mass (COM) position for the leg segment
        com_position = self._calculate_center_of_mass(keypoints)

        # Calculate velocity if we have previous data
        velocity = 0.0
        acceleration = 0.0

        if self.previous_position is not None and self.previous_frame is not None:
            dt = (frame_number - self.previous_frame) / self.fps
            if dt > 0:
                # Calculate velocity (pixels per second, then convert to relative units)
                displacement = np.linalg.norm(com_position - self.previous_position)
                velocity = displacement / dt

                # Calculate acceleration if we have velocity history
                if len(self.frame_data) > 1:
                    prev_velocity = self.frame_data[-1]["velocity"]
                    acceleration = (velocity - prev_velocity) / dt

        # Store frame data
        frame_data = {
            "frame": frame_number,
            "position": com_position,
            "velocity": velocity,
            "acceleration": acceleration,
            "rep_count": rep_count,
            "stage": stage,
            "angle": angle,
            "keypoints": keypoints.copy(),
            "timestamp": frame_number / self.fps,
        }

        self.frame_data.append(frame_data)

        # Track rep transitions and calculate rep velocities
        self._track_rep_transitions(frame_data)

        # Update previous values
        self.previous_position = com_position
        self.previous_frame = frame_number

    def _calculate_center_of_mass(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Calculate center of mass for the leg segment.

        Args:
            keypoints: Array of keypoints [hip, knee, ankle] with shape (3, 3)

        Returns:
            2D position of center of mass [x, y]
        """
        # Weight distribution for leg segments (approximate)
        # Hip: 40%, Knee: 35%, Ankle: 25% (based on biomechanics literature)
        weights = np.array([0.4, 0.35, 0.25])

        # Extract x, y coordinates
        positions = keypoints[:, :2]  # [hip, knee, ankle] x,y coordinates

        # Calculate weighted center of mass
        com = np.average(positions, axis=0, weights=weights)

        return com

    def _track_rep_transitions(self, frame_data: Dict) -> None:
        """
        Track rep transitions and calculate velocities for complete reps.

        Args:
            frame_data: Current frame data
        """
        current_rep = frame_data["rep_count"]
        current_stage = frame_data["stage"]

        # Initialize tracking for first rep
        if self.current_rep_data is None:
            self.current_rep_data = {
                "rep_number": current_rep,
                "start_frame": frame_data["frame"],
                "frames": [frame_data],
                "phases": {"down": [], "up": []},
                "stage_transitions": [],
            }
            return

        # Check for rep completion (rep count increased)
        if current_rep > self.current_rep_data["rep_number"]:
            # Complete the previous rep
            self._finalize_rep_velocity()

            # Start new rep
            self.current_rep_data = {
                "rep_number": current_rep,
                "start_frame": frame_data["frame"],
                "frames": [frame_data],
                "phases": {"down": [], "up": []},
                "stage_transitions": [],
            }
        else:
            # Continue current rep
            self.current_rep_data["frames"].append(frame_data)

            # Track stage transitions
            if len(self.current_rep_data["frames"]) > 1:
                prev_stage = self.current_rep_data["frames"][-2]["stage"]
                if prev_stage != current_stage:
                    self.current_rep_data["stage_transitions"].append(
                        {
                            "from": prev_stage,
                            "to": current_stage,
                            "frame": frame_data["frame"],
                        }
                    )

            # Categorize frame by phase
            self.current_rep_data["phases"][current_stage].append(frame_data)

    def _finalize_rep_velocity(self) -> None:
        """Finalize velocity calculations for the completed rep."""
        if not self.current_rep_data or len(self.current_rep_data["frames"]) < 3:
            return

        rep_frames = self.current_rep_data["frames"]
        rep_number = self.current_rep_data["rep_number"]

        # Calculate rep metrics
        rep_velocity_data = self._calculate_rep_metrics(rep_frames, rep_number)

        if rep_velocity_data:
            self.rep_velocities.append(rep_velocity_data)
            logger.info(
                f"Rep {rep_number} VBT Analysis: "
                f"Concentric Velocity: {rep_velocity_data['concentric_velocity']:.2f}, "
                f"Peak Velocity: {rep_velocity_data['peak_velocity']:.2f}, "
                f"Duration: {rep_velocity_data['total_duration']:.2f}s"
            )
            logger.info("-" * 50)

    def _calculate_rep_metrics(
        self, rep_frames: List[Dict], rep_number: int
    ) -> Optional[Dict]:
        """
        Calculate comprehensive velocity metrics for a rep.

        Args:
            rep_frames: List of frame data for the rep
            rep_number: Rep number

        Returns:
            Dictionary with velocity metrics
        """
        if len(rep_frames) < 3:
            return None

        # Extract data arrays
        positions = np.array([frame["position"] for frame in rep_frames])
        velocities = np.array([frame["velocity"] for frame in rep_frames])
        accelerations = np.array([frame["acceleration"] for frame in rep_frames])
        timestamps = np.array([frame["timestamp"] for frame in rep_frames])
        stages = [frame["stage"] for frame in rep_frames]

        # Smooth velocity data
        if len(velocities) >= self.smoothing_window:
            smoothed_velocities = savgol_filter(
                velocities,
                min(self.smoothing_window, len(velocities)),
                2 if len(velocities) >= 3 else 1,
            )
        else:
            smoothed_velocities = velocities

        # Find phase transitions
        down_phases = [i for i, stage in enumerate(stages) if stage == "down"]
        up_phases = [i for i, stage in enumerate(stages) if stage == "up"]

        # Calculate phase-specific metrics
        eccentric_velocity = (
            np.mean(smoothed_velocities[down_phases]) if down_phases else 0
        )
        concentric_velocity = (
            np.mean(smoothed_velocities[up_phases]) if up_phases else 0
        )

        # Calculate peak velocities
        peak_velocity = (
            np.max(smoothed_velocities) if len(smoothed_velocities) > 0 else 0
        )
        peak_eccentric = np.max(smoothed_velocities[down_phases]) if down_phases else 0
        peak_concentric = np.max(smoothed_velocities[up_phases]) if up_phases else 0

        # Calculate power metrics (simplified - would need load data for true power)
        # Using velocity as proxy for relative power
        average_power = np.mean(smoothed_velocities)
        peak_power = peak_velocity

        # Calculate durations
        total_duration = timestamps[-1] - timestamps[0]
        eccentric_duration = len(down_phases) / self.fps if down_phases else 0
        concentric_duration = len(up_phases) / self.fps if up_phases else 0

        # Calculate range of motion (displacement)
        total_displacement = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))

        # Calculate velocity consistency (coefficient of variation)
        velocity_cv = (
            np.std(smoothed_velocities) / np.mean(smoothed_velocities)
            if np.mean(smoothed_velocities) > 0
            else 0
        )

        # Determine rep quality based on velocity patterns
        rep_quality = self._assess_rep_quality(smoothed_velocities, stages)

        return {
            "rep_number": rep_number,
            "total_duration": round(total_duration, 3),
            "concentric_velocity": round(concentric_velocity, 3),
            "peak_velocity": round(peak_velocity, 3),
            "velocity_consistency": round(velocity_cv, 3),
            "rep_quality_score": rep_quality,
        }

    def _assess_rep_quality(self, velocities: np.ndarray, stages: List[str]) -> float:
        """
        Assess rep quality based on velocity patterns.

        Args:
            velocities: Array of velocity values
            stages: List of stage labels

        Returns:
            Quality score between 0 and 1
        """
        if len(velocities) < 3:
            return 0.0

        quality_factors = []

        # Factor 1: Velocity smoothness (lower coefficient of variation is better)
        if np.mean(velocities) > 0:
            cv = np.std(velocities) / np.mean(velocities)
            smoothness_score = max(0, 1 - cv)  # Lower CV = higher score
            quality_factors.append(smoothness_score)

        # Factor 2: Presence of both eccentric and concentric phases
        has_down = "down" in stages
        has_up = "up" in stages
        phase_completeness = 1.0 if (has_down and has_up) else 0.5
        quality_factors.append(phase_completeness)

        # Factor 3: Minimum movement threshold
        if np.max(velocities) > np.mean(velocities) * 0.5:  # Peak should be meaningful
            movement_score = 1.0
        else:
            movement_score = 0.6
        quality_factors.append(movement_score)

        # Factor 4: Rep duration (not too fast, not too slow)
        rep_duration = len(velocities) / self.fps
        if 1.0 <= rep_duration <= 4.0:  # Reasonable squat duration
            duration_score = 1.0
        elif 0.5 <= rep_duration < 1.0 or 4.0 < rep_duration <= 6.0:
            duration_score = 0.7
        else:
            duration_score = 0.4
        quality_factors.append(duration_score)

        # Calculate overall quality score
        return round(np.mean(quality_factors), 3)

    def finalize_analysis(self) -> None:
        """Finalize the analysis and handle any remaining rep data."""
        # Handle any incomplete rep at the end
        if (
            self.current_rep_data
            and len(self.current_rep_data["frames"]) > 3
            and self.current_rep_data["rep_number"]
            not in [rep["rep_number"] for rep in self.rep_velocities]
        ):
            rep_frames = self.current_rep_data["frames"]
            rep_number = self.current_rep_data["rep_number"]
            rep_velocity_data = self._calculate_rep_metrics(rep_frames, rep_number)

            if rep_velocity_data:
                self.rep_velocities.append(rep_velocity_data)
                logger.info(
                    f"Final Rep {rep_number} VBT Analysis: "
                    f"Concentric Velocity: {rep_velocity_data['concentric_velocity']:.2f}, "
                    f"Peak Velocity: {rep_velocity_data['peak_velocity']:.2f}"
                )
                logger.info("-" * 50)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate summary statistics for all reps.

        Returns:
            Dictionary with summary statistics
        """
        if not self.rep_velocities:
            return {}

        # Extract metrics for summary
        concentric_velocities = [
            rep["concentric_velocity"] for rep in self.rep_velocities
        ]
        peak_velocities = [rep["peak_velocity"] for rep in self.rep_velocities]
        durations = [rep["total_duration"] for rep in self.rep_velocities]
        quality_scores = [rep["rep_quality_score"] for rep in self.rep_velocities]

        return {
            "total_reps_analyzed": len(self.rep_velocities),
            "concentric_velocity_stats": {
                "mean": round(np.mean(concentric_velocities), 3),
                "std": round(np.std(concentric_velocities), 3),
                "min": round(np.min(concentric_velocities), 3),
                "max": round(np.max(concentric_velocities), 3),
            },
            "peak_velocity_stats": {
                "mean": round(np.mean(peak_velocities), 3),
                "std": round(np.std(peak_velocities), 3),
                "min": round(np.min(peak_velocities), 3),
                "max": round(np.max(peak_velocities), 3),
            },
            "duration_stats": {
                "mean": round(np.mean(durations), 3),
                "std": round(np.std(durations), 3),
                "min": round(np.min(durations), 3),
                "max": round(np.max(durations), 3),
            },
            "quality_stats": {
                "mean": round(np.mean(quality_scores), 3),
                "std": round(np.std(quality_scores), 3),
                "min": round(np.min(quality_scores), 3),
                "max": round(np.max(quality_scores), 3),
            },
            "consistency_metrics": {
                "concentric_velocity_cv": round(
                    np.std(concentric_velocities) / np.mean(concentric_velocities), 3
                )
                if np.mean(concentric_velocities) > 0
                else 0
            },
        }

    def save_to_json(
        self, output_dir: str, exercise_name: str = "squat", timestamp: str = None
    ) -> str:
        """
        Save velocity analysis to JSON file.

        Args:
            output_dir: Base output directory
            exercise_name: Name of the exercise
            timestamp: Optional timestamp to use for filename. If None, generates current timestamp.

        Returns:
            Path to the saved JSON file
        """
        # Create velocity calculations directory
        velocity_dir = os.path.join(output_dir, "velocity_calculations")
        os.makedirs(velocity_dir, exist_ok=True)

        # Use provided timestamp or generate current one
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{exercise_name}_vbt_analysis_{timestamp}.json"
        filepath = os.path.join(velocity_dir, filename)

        # Prepare data for JSON export
        vbt_data = {
            "analysis_info": {
                "exercise": exercise_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "fps": self.fps,
                "total_frames_analyzed": len(self.frame_data),
                "smoothing_window": self.smoothing_window,
            },
            "rep_velocities": self.rep_velocities,
            "summary_statistics": self.get_summary_statistics(),
            "methodology": {
                "velocity_calculation": "Center of mass displacement per unit time",
                "center_of_mass_weights": {"hip": 0.4, "knee": 0.35, "ankle": 0.25},
                "smoothing_method": "Savitzky-Golay filter",
                "quality_assessment_factors": [
                    "velocity_smoothness",
                    "phase_completeness",
                    "movement_threshold",
                    "rep_duration",
                ],
                "simplified_metrics": "Only essential VBT metrics included for practical application",
            },
        }

        # Save to JSON file
        with open(filepath, "w") as f:
            json.dump(vbt_data, f, indent=2)

        logger.info(f"VBT analysis saved to: {filepath}")
        return filepath
