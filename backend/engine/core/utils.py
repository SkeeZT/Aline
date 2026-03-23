"""
General utilities for AI Trainer application.
"""

import os
import cv2
import time
import yaml
import numpy as np
from loguru import logger
from typing import Dict, Any, Optional, Callable


def countdown_visualizer(frame: np.ndarray, count: int, extra_text: Optional[str] = None) -> np.ndarray:
    """
    Draw countdown visualization on frame.

    Args:
        frame: Input video frame
        count: Current countdown number (5, 4, 3, 2, 1)
        extra_text: Optional text to display below the countdown (e.g. side detection status)

    Returns:
        Frame with countdown visualization
    """
    # Get frame dimensions
    height, width = frame.shape[:2]

    # Draw countdown number with modern styling
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 6
    thickness = 8
    color = (66, 133, 244)  # Google Blue

    # Get text size for centering
    text = str(count)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    # Draw main text with subtle shadow
    cv2.putText(
        frame,
        text,
        (text_x + 3, text_y + 3),
        font,
        font_scale,
        (30, 30, 30),  # Dark gray shadow
        thickness,
    )
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

    # Draw "Starting in..." text above with modern styling
    info_text = "Starting in..."
    info_font_scale = 1.0
    info_thickness = 2
    info_color = (240, 240, 240)  # Light gray

    info_size = cv2.getTextSize(info_text, font, info_font_scale, info_thickness)[0]
    info_x = (width - info_size[0]) // 2
    info_y = text_y - 100

    cv2.putText(
        frame,
        info_text,
        (info_x, info_y),
        font,
        info_font_scale,
        info_color,
        info_thickness,
    )
    
    # Draw extra text (e.g. detecting side) below
    if extra_text:
        extra_font_scale = 0.8
        extra_thickness = 2
        extra_size = cv2.getTextSize(extra_text, font, extra_font_scale, extra_thickness)[0]
        extra_x = (width - extra_size[0]) // 2
        extra_y = text_y + 80
        
        # Color based on status (Amber for detecting, Green for side found)
        if "Detecting..." in extra_text:
            extra_color = (0, 165, 255) # Orange
        else:
            extra_color = (0, 255, 0) # Green for verified side results

        cv2.putText(
            frame,
            extra_text,
            (extra_x, extra_y),
            font,
            extra_font_scale,
            extra_color,
            extra_thickness,
        )

    return frame


def start_countdown(
    cap,
    fps: float = 30.0,
    show_viz: bool = True,
    frame_processor=None,
    countdown_duration=5,
    on_annotated_frame=None,
    pose_estimator=None,
) -> Optional[str]:
    """
    Perform countdown before starting the exercise analysis, optionally detecting facing side.

    Args:
        cap: Video capture object (cv2.VideoCapture)
        fps: Frames per second for timing
        show_viz: Whether to show visualization window
        frame_processor: Optional function to process frames (e.g., apply rotation)
        countdown_duration: Duration of countdown in seconds
        on_annotated_frame: Optional callback for annotated countdown frames
        pose_estimator: Optional pose estimator to detect side during countdown

    Returns:
        Determined side ('left' or 'right') if pose_estimator provided, else None
    """
    logger.info(f"Starting {countdown_duration}-second countdown...")

    frames_per_second = int(fps)
    total_frames = countdown_duration * frames_per_second

    # Voting counters for side detection
    left_votes = 0
    right_votes = 0
    
    # Keypoint indices for confidence check
    left_keypoints = [1, 3, 5, 7, 9]   # L_Eye, L_Ear, L_Shoulder, L_Elbow, L_Wrist
    right_keypoints = [2, 4, 6, 8, 10] # R_Eye, R_Ear, R_Shoulder, R_Elbow, R_Wrist

    for i in range(total_frames):
        if frame_processor:
            ret, frame = frame_processor()
        else:
            ret, frame = cap.read()

        if not ret:
            logger.warning("Failed to read frame during countdown")
            break

        # Side Detection Logic (if estimator provided)
        current_side_guess = None
        if pose_estimator is not None:
            try:
                results = pose_estimator.process_frame(frame)
                keypoints = pose_estimator.get_keypoints(results, frame.shape[:2]) # (17, 3)

                if keypoints is not None:
                    confidences = keypoints[:, 2]
                    
                    # Calculate simple average confidence for side
                    def get_side_conf(indices):
                        vals = [confidences[idx] for idx in indices if idx < len(confidences)]
                        return np.mean(vals) if vals else 0.0

                    l_conf = get_side_conf(left_keypoints)
                    r_conf = get_side_conf(right_keypoints)
                    
                    # Vote if there is a distinction
                    if abs(l_conf - r_conf) > 0.1: # 10% diff threshold
                        if l_conf > r_conf:
                            left_votes += 1
                            current_side_guess = "left"
                        else:
                            right_votes += 1
                            current_side_guess = "right"
            except Exception as e:
                pass # Silently fail detection on this frame

        # Calculate current countdown number
        remaining_seconds = countdown_duration - (i // frames_per_second)
        if remaining_seconds <= 0:
            remaining_seconds = 1

        # Draw countdown visualization
        detected_side_disp = None
        if pose_estimator:
            # Show current winning side in real-time
            if left_votes > right_votes:
                detected_side_disp = "Detecting: LEFT"
            elif right_votes > left_votes:
                detected_side_disp = "Detecting: RIGHT"
            else:
                detected_side_disp = "Detecting..."

        frame_with_countdown = countdown_visualizer(frame, remaining_seconds, extra_text=detected_side_disp)

        # Notify listener of annotated frame if callback provided
        if on_annotated_frame:
            try:
                on_annotated_frame(frame_with_countdown)
            except Exception as exc:
                logger.warning(f"Countdown frame callback error: {exc}")

        if show_viz:
            cv2.imshow("AI Trainer - Squat Analysis", frame_with_countdown)
            if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
                break

        # Small delay to maintain proper timing (naive loop)
        # Note: In a real streaming loop this might delay too much, but for countdown it's fine
        # time.sleep(1.0 / fps) 

    # Final decision
    determined_side = None
    if pose_estimator:
        if left_votes > right_votes:
            determined_side = "left"
        elif right_votes > left_votes:
            determined_side = "right"
        # If equal or zero, default to 'right' or None? 
        # Making a safe choice: default to right if undetermined but significant frames processed
        elif (left_votes + right_votes) > 10:
             determined_side = "right"
        
        logger.info(f"Countdown complete! Votes - Left: {left_votes}, Right: {right_votes}. Final Decision: {determined_side}")
    else:
        logger.info("Countdown complete! Starting exercise analysis...")

    return determined_side


def side_visibility_detector(
    cap,
    fps: float = 30.0,
    confidence_threshold: float = 0.25,
    show_viz: bool = True,
    frame_processor: Optional[Callable] = None,
    pose_estimator=None,
    model=None,
    timeout_seconds: int = None,
    on_annotated_frame=None,
) -> bool:
    """
    Wait until one side of the body is clearly more visible than the other.

    This function analyzes keypoint confidence scores to determine when the user
    is positioned optimally with one side facing the camera more clearly.

    Args:
        cap: Video capture object (cv2.VideoCapture)
        fps: Frames per second for timing
        confidence_threshold: Minimum difference between left/right confidence (default: 0.25)
        show_viz: Whether to show visualization window
        frame_processor: Optional function to process frames (e.g., apply rotation)
        pose_estimator: PoseEstimator instance
        model: Legacy YOLO model (deprecated)
        timeout_seconds: Maximum time to wait for side positioning (None = wait indefinitely)

    Returns:
        True if optimal positioning achieved, False if failed/interrupted/timeout
    """
    if timeout_seconds is not None:
        logger.info(
            f"Waiting for optimal side positioning (confidence difference > {confidence_threshold}) with {timeout_seconds}s timeout..."
        )
    else:
        logger.info(
            f"Waiting for optimal side positioning (confidence difference > {confidence_threshold})..."
        )

    # Keypoint indices for left and right groups
    left_keypoints = [
        1,
        3,
        5,
        7,
        9,
    ]  # Left_Eye, Left_Ear, Left_Shoulder, Left_Elbow, Left_Wrist
    right_keypoints = [
        2,
        4,
        6,
        8,
        10,
    ]  # Right_Eye, Right_Ear, Right_Shoulder, Right_Elbow, Right_Wrist

    frames_per_second = int(fps)
    start_time = time.time()

    while True:
        # Check timeout if specified
        if timeout_seconds is not None:
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout_seconds:
                logger.warning(
                    f"Side positioning timeout after {timeout_seconds} seconds"
                )
                return False

        # Collect confidence scores for one second
        left_confidences = []
        right_confidences = []

        logger.info("Analyzing positioning for 1 second...")

        for frame_idx in range(frames_per_second):
            if frame_processor:
                ret, frame = frame_processor()
            else:
                ret, frame = cap.read()

            if not ret:
                logger.warning("Failed to read frame during side visibility detection")
                return False

            # Run pose estimation
            confidences = None
            if pose_estimator is not None:
                try:
                    results = pose_estimator.process_frame(frame)
                    keypoints = pose_estimator.get_keypoints(results) # (17, 3) 
                    # Note: get_keypoints might need shape if we used MediaPipe directly without passing it before, 
                    # but our MediaPipePoseEstimator.get_keypoints takes optional frame_shape, 
                    # wait, I need to check if I can pass frame shape to get_keypoints if needed?
                    # My MediaPipePoseEstimator implementation signature: get_keypoints(results, frame_shape)
                    # YOLOPoseEstimator: get_keypoints(results)
                    # I should check the signature compatibility.
                    
                    keypoints = pose_estimator.get_keypoints(results, frame.shape[:2])

                    if keypoints is not None:
                        confidences = keypoints[:, 2]
                except Exception as e:
                    logger.debug(f"Error in pose estimation: {e}")

            elif model is not None:
                 # Legacy
                try:
                    results = model(frame, verbose=False)
                    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                         confidences = results[0].keypoints.conf[0].cpu().numpy()
                except Exception as e:
                    logger.debug(f"Error in legacy pose estimation: {e}")

            # Calculate metrics if we have confidences
            if confidences is not None:
                # Calculate average confidence for left keypoints
                left_conf_frame = []
                for idx in left_keypoints:
                    if idx < len(confidences):
                        left_conf_frame.append(confidences[idx])

                # Calculate average confidence for right keypoints
                right_conf_frame = []
                for idx in right_keypoints:
                    if idx < len(confidences):
                        right_conf_frame.append(confidences[idx])

                # Store frame averages
                left_confidences.append(np.mean(left_conf_frame) if left_conf_frame else 0.0)
                right_confidences.append(np.mean(right_conf_frame) if right_conf_frame else 0.0)
            else:
                 left_confidences.append(0.0)
                 right_confidences.append(0.0)

            # Draw waiting visualization with timeout info
            remaining_time = None
            if timeout_seconds is not None:
                remaining_time = timeout_seconds - (time.time() - start_time)

            frame_with_info = _draw_positioning_info(
                frame,
                left_confidences[-1] if left_confidences else 0.0,
                right_confidences[-1] if right_confidences else 0.0,
                confidence_threshold,
                frame_idx + 1,
                frames_per_second,
                remaining_time,
            )
            
            # Notify listener of annotated frame if callback provided
            if on_annotated_frame:
                try:
                    on_annotated_frame(frame_with_info)
                except Exception as exc:
                    logger.warning(f"Positioning frame callback error: {exc}")
            
            # Show in OpenCV window if enabled
            if show_viz:
                cv2.imshow("AI Trainer - Squat Analysis", frame_with_info)
                if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
                    return False

        # Calculate average confidence for the entire second
        if left_confidences and right_confidences:
            avg_left_confidence = np.mean(left_confidences)
            avg_right_confidence = np.mean(right_confidences)

            confidence_difference = abs(avg_left_confidence - avg_right_confidence)

            logger.info(f"Average left confidence: {avg_left_confidence:.3f}")
            logger.info(f"Average right confidence: {avg_right_confidence:.3f}")
            logger.info(f"Confidence difference: {confidence_difference:.3f}")

            if confidence_difference >= confidence_threshold:
                preferred_side = (
                    "left" if avg_left_confidence > avg_right_confidence else "right"
                )
                logger.success(
                    f"Optimal positioning achieved! Preferred side: {preferred_side}"
                )
                logger.success(f"Starting exercise analysis...")
                return True
            else:
                logger.info(
                    f"Confidence difference ({confidence_difference:.3f}) below threshold ({confidence_threshold})"
                )
                if timeout_seconds is not None:
                    remaining_time = timeout_seconds - (time.time() - start_time)
                    logger.info(
                        f"Please adjust your position to face more to one side... ({remaining_time:.1f}s remaining)"
                    )
                else:
                    logger.info(
                        "Please adjust your position to face more to one side..."
                    )
        else:
            logger.warning("No valid keypoint data collected")


def _draw_positioning_info(
    frame: np.ndarray,
    left_conf: float,
    right_conf: float,
    threshold: float,
    current_frame: int,
    total_frames: int,
    remaining_time: Optional[float] = None,
) -> np.ndarray:
    """
    Draw positioning information on the frame during side visibility detection.

    Args:
        frame: Input video frame
        left_conf: Current left side confidence
        right_conf: Current right side confidence
        threshold: Confidence difference threshold
        current_frame: Current frame number in the second
        total_frames: Total frames in the second
        remaining_time: Remaining time until timeout (None if no timeout)

    Returns:
        Frame with positioning information drawn
    """
    height, width = frame.shape[:2]

    # Modern positioning panel - positioned at top center
    panel_width = 350
    panel_height = 150 if remaining_time is not None else 120
    panel_x = (width - panel_width) // 2
    panel_y = 30

    # Draw rounded panel background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (30, 30, 30),  # Dark gray
        -1,
    )

    # Blend overlay with original frame
    alpha = 0.8
    frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    # Draw panel border with modern styling
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (66, 133, 244),  # Google Blue
        2,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    title_text = "Positioning"
    cv2.putText(
        frame, title_text, (panel_x + 20, panel_y + 30), font, 0.7, (240, 240, 240), 2
    )

    # Progress bar for current second
    progress_width = panel_width - 40
    progress_height = 6
    progress_x = panel_x + 20
    progress_y = panel_y + 45

    # Background progress bar
    cv2.rectangle(
        frame,
        (progress_x, progress_y),
        (progress_x + progress_width, progress_y + progress_height),
        (80, 80, 80),  # Darker gray
        -1,
    )

    # Progress fill
    progress_fill = int((current_frame / total_frames) * progress_width)
    cv2.rectangle(
        frame,
        (progress_x, progress_y),
        (progress_x + progress_fill, progress_y + progress_height),
        (66, 133, 244),  # Google Blue
        -1,
    )

    # Confidence information with modern styling
    y_offset = 80
    cv2.putText(
        frame,
        f"L: {left_conf:.2f}",
        (panel_x + 20, panel_y + y_offset),
        font,
        0.6,
        (100, 200, 255),  # Light blue
        1,
    )

    cv2.putText(
        frame,
        f"R: {right_conf:.2f}",
        (panel_x + 120, panel_y + y_offset),
        font,
        0.6,
        (255, 200, 100),  # Light orange
        1,
    )

    y_offset += 25
    difference = abs(left_conf - right_conf)
    color = (52, 168, 83) if difference >= threshold else (251, 188, 5)  # Green if good, Yellow if not
    cv2.putText(
        frame,
        f"Diff: {difference:.2f}",
        (panel_x + 20, panel_y + y_offset),
        font,
        0.6,
        color,
        1,
    )

    cv2.putText(
        frame,
        f"Req: {threshold:.2f}",
        (panel_x + 120, panel_y + y_offset),
        font,
        0.6,
        (240, 240, 240),  # Light gray
        1,
    )

    # Timeout information if available
    if remaining_time is not None:
        y_offset += 25
        timeout_color = (251, 188, 5) if remaining_time > 2.0 else (234, 67, 53)  # Yellow or Red
        cv2.putText(
            frame,
            f"Time: {remaining_time:.1f}s",
            (panel_x + 20, panel_y + y_offset),
            font,
            0.6,
            timeout_color,
            1,
        )

    # Instructions at bottom
    instruction_y = panel_y + panel_height + 30
    if difference < threshold:
        if remaining_time is not None and remaining_time <= 2.0:
            instruction = "Continuing in front view mode soon..."
            color = (0, 255, 255)
        else:
            instruction = "Please face more to one side"
            color = (0, 100, 255)
    else:
        instruction = "Good positioning! Starting soon..."
        color = (0, 255, 0)

    # Center the instruction text
    text_size = cv2.getTextSize(instruction, font, 0.7, 2)[0]
    text_x = (width - text_size[0]) // 2
    cv2.putText(frame, instruction, (text_x, instruction_y), font, 0.7, color, 2)

    return frame


# Anthropometrics functions
def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute Euclidean distance between two 2D points."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return float(np.linalg.norm(p1[:2] - p2[:2]))


def compute_basic_lengths(
    keypoints_xy: np.ndarray,
    confidences: Optional[np.ndarray],
    kpt: dict,
) -> dict:
    """
    Compute key limb segment lengths in pixels.

    Args:
        keypoints_xy: Array of shape (N, 2) with (x, y) for each keypoint index per YOLO.
        confidences: Optional array of shape (N,) with confidence [0,1] per keypoint.
        kpt: Mapping of anatomical names to YOLO indices, e.g., from config["keypoints"].

    Returns:
        Dict of segment lengths in pixels.
    """

    def ok(idx: int) -> bool:
        if confidences is None:
            return True
        return idx is not None and idx >= 0 and confidences[idx] > 0.3

    def safe_len(a: str, b: str) -> float:
        ia = kpt.get(a)
        ib = kpt.get(b)
        if ia is None or ib is None:
            return float("nan")
        if not ok(ia) or not ok(ib):
            return float("nan")
        return _distance(keypoints_xy[ia], keypoints_xy[ib])

    lengths = {
        # Arms
        "upper_arm_left": safe_len("left_shoulder", "left_elbow"),
        "forearm_left": safe_len("left_elbow", "left_wrist"),
        "upper_arm_right": safe_len("right_shoulder", "right_elbow"),
        "forearm_right": safe_len("right_elbow", "right_wrist"),
        # Legs
        "thigh_left": safe_len("left_hip", "left_knee"),
        "shank_left": safe_len("left_knee", "left_ankle"),
        "thigh_right": safe_len("right_hip", "right_knee"),
        "shank_right": safe_len("right_knee", "right_ankle"),
        # Torso/shoulders/hips
        "shoulder_width": safe_len("left_shoulder", "right_shoulder"),
        "hip_width": safe_len("left_hip", "right_hip"),
        # Approx body height proxy (nose-to-ankle average side)
        "nose_to_left_ankle": safe_len("nose", "left_ankle"),
        "nose_to_right_ankle": safe_len("nose", "right_ankle"),
    }

    # Aggregate helpers
    def nanmean(vals):
        arr = np.array(vals, dtype=float)
        if np.all(np.isnan(arr)):
            return float("nan")
        return float(np.nanmean(arr))

    # Symmetric aggregates
    lengths.update(
        {
            "upper_arm_avg": nanmean(
                [lengths["upper_arm_left"], lengths["upper_arm_right"]]
            ),
            "forearm_avg": nanmean([lengths["forearm_left"], lengths["forearm_right"]]),
            "thigh_avg": nanmean([lengths["thigh_left"], lengths["thigh_right"]]),
            "shank_avg": nanmean([lengths["shank_left"], lengths["shank_right"]]),
            "leg_total_left": nanmean([lengths["thigh_left"], lengths["shank_left"]]),
            "leg_total_right": nanmean(
                [lengths["thigh_right"], lengths["shank_right"]]
            ),
            "leg_total_avg": nanmean(
                [
                    nanmean([lengths["thigh_left"], lengths["shank_left"]]),
                    nanmean([lengths["thigh_right"], lengths["shank_right"]]),
                ]
            ),
            "body_height_proxy": nanmean(
                [lengths["nose_to_left_ankle"], lengths["nose_to_right_ankle"]]
            ),
        }
    )

    return lengths


def compute_ratios(lengths: dict) -> dict:
    """
    Compute useful ratios from segment lengths.

    Returns:
        Dict of ratios (unitless).
    """

    def ratio(a: str, b: str) -> float:
        va = lengths.get(a)
        vb = lengths.get(b)
        if va is None or vb is None or np.isnan(va) or np.isnan(vb) or vb == 0:
            return float("nan")
        return float(va / vb)

    # Ratios relative to body height proxy
    ratios = {
        "upper_arm_to_height": ratio("upper_arm_avg", "body_height_proxy"),
        "forearm_to_height": ratio("forearm_avg", "body_height_proxy"),
        "thigh_to_height": ratio("thigh_avg", "body_height_proxy"),
        "shank_to_height": ratio("shank_avg", "body_height_proxy"),
        "leg_total_to_height": ratio("leg_total_avg", "body_height_proxy"),
        # Intra-limb ratios
        "upper_to_lower_arm": ratio("upper_arm_avg", "forearm_avg"),
        "thigh_to_shank": ratio("thigh_avg", "shank_avg"),
        # Widths
        "shoulder_to_height": ratio("shoulder_width", "body_height_proxy"),
        "hip_to_height": ratio("hip_width", "body_height_proxy"),
        "shoulder_to_hip": ratio("shoulder_width", "hip_width"),
    }

    return ratios


def compute_anthropometrics(
    keypoints_xy: np.ndarray,
    confidences: Optional[np.ndarray],
    keypoint_indices: dict,
) -> dict:
    """
    Convenience wrapper to compute both lengths and ratios.
    """
    lengths = compute_basic_lengths(keypoints_xy, confidences, keypoint_indices)
    ratios = compute_ratios(lengths)

    # Sanitize NaNs to None for JSON compatibility
    def _nan_to_none(d: dict) -> dict:
        clean = {}
        for k, v in d.items():
            if isinstance(v, float) and np.isnan(v):
                clean[k] = None
            else:
                clean[k] = v
        return clean

    return {"lengths": _nan_to_none(lengths), "ratios": _nan_to_none(ratios)}


# Back curvature functions
def _safe_int(v: float) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return 0


def _collect_back_boundary_points(
    mask: np.ndarray,
    y_top: int,
    y_bottom: int,
    facing_side: str,
) -> np.ndarray:
    """
    Collect boundary points along the posterior side of the person mask for rows in [y_top, y_bottom].

    Returns array of shape (M, 2) with (x, y) points.
    """
    h, w = mask.shape[:2]
    y0 = max(0, min(y_top, y_bottom))
    y1 = min(h - 1, max(y_top, y_bottom))

    points = []
    choose_left = True if facing_side == "right" else False

    for y in range(y0, y1 + 1):
        row = mask[y]
        # Find indices where mask is present (threshold at 0.5 if float)
        if row.dtype != np.uint8:
            row_bin = row > 0.5
        else:
            row_bin = row > 127
        xs = np.flatnonzero(row_bin)
        if xs.size == 0:
            continue
        x = int(xs[0] if choose_left else xs[-1])
        points.append((x, y))

    if not points:
        return np.empty((0, 2), dtype=float)
    return np.asarray(points, dtype=float)


def _fit_curvature_metric(points: np.ndarray) -> tuple:
    """
    Fit x = f(y) polynomial (degree 3) and compute mean absolute second derivative over normalized y.

    Returns (curvature_metric, coeffs) where coeffs are polynomial coefficients or None if failed.
    """
    if points.shape[0] < 6:
        return float("nan"), None

    xs = points[:, 0]
    ys = points[:, 1]

    # Normalize y to [0, 1] for scale invariance
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    span = max(1.0, y_max - y_min)
    y_norm = (ys - y_min) / span

    try:
        coeffs = np.polyfit(y_norm, xs, deg=3)  # x = a*y^3 + b*y^2 + c*y + d
        a, b, c, d = coeffs
        # Second derivative: d2x/dy2 = 6*a*y + 2*b
        y_samples = np.linspace(0.0, 1.0, num=50)
        d2 = 6.0 * a * y_samples + 2.0 * b
        curvature = float(np.mean(np.abs(d2)))
        # Normalize by the overall lateral extent to reduce pixel scaling
        x_span = max(1.0, float(np.max(xs) - np.min(xs)))
        curvature_norm = curvature / x_span
        return curvature_norm, coeffs
    except Exception:
        return float("nan"), None


def compute_back_curvature(
    mask: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoint_indices: dict,
    facing_side: str = "right",
) -> dict:
    """
    Compute a back curvature metric and provide sampled back boundary points.

    Args:
        mask: Binary person mask (H, W) float or uint8
        keypoints_xy: YOLO keypoints (N, 2)
        keypoint_indices: Mapping with at least 'nose' (or 'left/right_eye'), 'left/right_hip'
        facing_side: 'left' or 'right' to pick posterior boundary

    Returns:
        Dict with fields: 'curvature', 'num_points', 'y_span', 'x_span'
    """
    # Choose head proxy: prefer nose, otherwise average of eyes if available
    head_idx = keypoint_indices.get("nose")
    if head_idx is None:
        le = keypoint_indices.get("left_eye")
        re = keypoint_indices.get("right_eye")
        if le is not None and re is not None:
            head_point = 0.5 * (keypoints_xy[le][:2] + keypoints_xy[re][:2])
        else:
            return {
                "curvature": float("nan"),
                "num_points": 0,
                "y_span": None,
                "x_span": None,
            }
    else:
        head_point = keypoints_xy[head_idx][:2]

    # Choose hip on the facing side for torso span
    hip_idx = (
        keypoint_indices.get("left_hip")
        if facing_side == "left"
        else keypoint_indices.get("right_hip")
    )
    if hip_idx is None:
        return {
            "curvature": float("nan"),
            "num_points": 0,
            "y_span": None,
            "x_span": None,
        }

    y_top = _safe_int(min(head_point[1], keypoints_xy[hip_idx][1]))
    y_bot = _safe_int(max(head_point[1], keypoints_xy[hip_idx][1]))
    if y_bot - y_top < 10:
        return {
            "curvature": float("nan"),
            "num_points": 0,
            "y_span": None,
            "x_span": None,
        }

    points = _collect_back_boundary_points(mask, y_top, y_bot, facing_side)
    if points.shape[0] == 0:
        return {
            "curvature": float("nan"),
            "num_points": 0,
            "y_span": None,
            "x_span": None,
        }

    curvature, _ = _fit_curvature_metric(points)
    ys = points[:, 1]
    xs = points[:, 0]
    return {
        "curvature": curvature,
        "num_points": int(points.shape[0]),
        "y_span": float(np.max(ys) - np.min(ys)),
        "x_span": float(np.max(xs) - np.min(xs)),
    }


# Config class moved to api.core.config for unified configuration management
