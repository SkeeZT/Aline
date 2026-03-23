"""
Facing side detection utilities for pose estimation.
Determines which side of the body is facing the camera using keypoint analysis.
"""

import numpy as np
from loguru import logger


def determine_facing_side(keypoints_data) -> str:
    """
    Determine which side of the body is facing the camera using multiple heuristics.

    Args:
        keypoints_data: Array of keypoints from YOLO pose estimation

    Returns:
        String indicating the side facing the camera: 'left' or 'right'
    """
    # Make sure we have valid keypoint data
    if keypoints_data is None or len(keypoints_data) < 17:
        logger.warning("Invalid keypoint data, defaulting to right side")
        return "right"

    # Initialize votes for each side
    votes = {"left": 0, "right": 0}

    # Extract needed keypoints
    nose = keypoints_data[0]
    left_eye = keypoints_data[1]
    right_eye = keypoints_data[2]
    left_ear = keypoints_data[3]
    right_ear = keypoints_data[4]
    left_shoulder = keypoints_data[5]
    right_shoulder = keypoints_data[6]

    # Check if key keypoints are valid
    if left_shoulder[2] <= 0 or right_shoulder[2] <= 0:
        logger.warning("Shoulder keypoints not detected, defaulting to right side")
        return "right"

    # Heuristic 1: Shoulder Width Analysis
    logger.debug("Applying Heuristic 1: Shoulder Width Analysis")

    # Calculate Y-coordinate differences to estimate depth
    left_y = left_shoulder[1]
    right_y = right_shoulder[1]
    left_x = left_shoulder[0]
    right_x = right_shoulder[0]

    # If the Y-coordinates are significantly different, the person is likely at an angle
    y_diff = abs(left_y - right_y)

    h1_result = "right"  # Default

    if y_diff > 10:  # Threshold to consider significant difference
        if left_y < right_y:
            h1_result = "left"
        else:
            h1_result = "right"
    # If Y difference is not significant, check which shoulder appears more to the front
    else:
        if abs(left_x) < abs(right_x):
            h1_result = "left"
        else:
            h1_result = "right"

    votes[h1_result] += 1
    logger.debug(f"  Heuristic 1 result: {h1_result}")

    # Heuristic 2: Nose Position Relative to Shoulders
    logger.debug("Applying Heuristic 2: Nose Position Relative to Shoulders")

    h2_result = "right"  # Default

    if nose[2] > 0:  # If nose is detected
        nose_x = nose[0]

        # Calculate midpoint between shoulders
        shoulder_midpoint_x = (left_shoulder[0] + right_shoulder[0]) / 2

        # If nose is closer to left shoulder than the midpoint, person is likely facing left
        if nose_x < shoulder_midpoint_x:
            h2_result = "left"
        else:
            h2_result = "right"

    votes[h2_result] += 1
    logger.debug(f"  Heuristic 2 result: {h2_result}")

    # Heuristic 3: Eye/Ear Visibility or Confidence
    logger.debug("Applying Heuristic 3: Eye/Ear Visibility/Confidence")

    h3_result = "right"  # Default

    # Calculate confidence scores for left and right facial features
    left_confidence = 0
    right_confidence = 0

    # Add confidence scores if features are detected
    if left_eye[2] > 0:
        left_confidence += left_eye[2]
    if left_ear[2] > 0:
        left_confidence += left_ear[2]
    if right_eye[2] > 0:
        right_confidence += right_eye[2]
    if right_ear[2] > 0:
        right_confidence += right_ear[2]

    # Determine which side has higher confidence
    if left_confidence > right_confidence:
        h3_result = "left"
    else:
        h3_result = "right"

    votes[h3_result] += 1
    logger.debug(f"  Heuristic 3 result: {h3_result}")

    # Final voting decision
    final_side = "left" if votes["left"] > votes["right"] else "right"

    logger.info(f"Voting results: Left: {votes['left']}, Right: {votes['right']}")
    logger.info(f"Final determined facing side: {final_side}")

    return final_side


def analyze_shoulder_symmetry(keypoints_data) -> dict:
    """
    Analyze shoulder symmetry to provide additional insights for facing side detection.

    Args:
        keypoints_data: Array of keypoints from YOLO pose estimation

    Returns:
        Dictionary with shoulder analysis metrics
    """
    if keypoints_data is None or len(keypoints_data) < 7:
        return {}

    left_shoulder = keypoints_data[5]
    right_shoulder = keypoints_data[6]

    if left_shoulder[2] <= 0 or right_shoulder[2] <= 0:
        return {}

    # Calculate various metrics
    x_diff = abs(left_shoulder[0] - right_shoulder[0])
    y_diff = abs(left_shoulder[1] - right_shoulder[1])
    confidence_diff = abs(left_shoulder[2] - right_shoulder[2])

    # Calculate shoulder angle relative to horizontal
    if x_diff > 0:
        shoulder_angle = np.degrees(np.arctan(y_diff / x_diff))
    else:
        shoulder_angle = 90.0

    return {
        "x_difference": x_diff,
        "y_difference": y_diff,
        "confidence_difference": confidence_diff,
        "shoulder_angle": shoulder_angle,
        "left_confidence": left_shoulder[2],
        "right_confidence": right_shoulder[2],
    }


def get_keypoint_visibility_stats(keypoints_data) -> dict:
    """
    Get visibility statistics for all keypoints to help with debugging side detection.

    Args:
        keypoints_data: Array of keypoints from YOLO pose estimation

    Returns:
        Dictionary with visibility statistics
    """
    if keypoints_data is None:
        return {}

    # Define keypoint names for YOLO pose model
    keypoint_names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    stats = {
        "total_keypoints": len(keypoints_data),
        "visible_keypoints": 0,
        "keypoint_details": {},
    }

    for i, kpt in enumerate(keypoints_data):
        if i < len(keypoint_names):
            name = keypoint_names[i]
            is_visible = kpt[2] > 0 if len(kpt) > 2 else False
            confidence = kpt[2] if len(kpt) > 2 else 0

            stats["keypoint_details"][name] = {
                "visible": is_visible,
                "confidence": confidence,
                "position": [kpt[0], kpt[1]] if len(kpt) >= 2 else [0, 0],
            }

            if is_visible:
                stats["visible_keypoints"] += 1

    stats["visibility_percentage"] = (
        stats["visible_keypoints"] / stats["total_keypoints"]
    ) * 100

    return stats
