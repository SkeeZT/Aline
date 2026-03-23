"""
Modern visualization utilities for squat analysis.
Features a clean, modern design with better layout and visual elements.
"""

import cv2
import math
import numpy as np
from typing import Dict, Any
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import os


class SquatVisualizer:
    """Modern visualizer for squat exercise analysis with clean UI design."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the modern squat visualizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Modern color palette with improved contrast and aesthetics
        self.colors = {
            "primary": (70, 130, 180),  # Steel Blue
            "success": (34, 139, 34),  # Forest Green
            "warning": (255, 165, 0),  # Orange
            "danger": (220, 20, 60),  # Crimson
            "info": (30, 144, 255),  # Dodger Blue
            "dark": (25, 25, 35),  # Dark slate
            "light": (245, 245, 245),  # Almost white
            "keypoints": (255, 105, 180),  # Hot pink
            "knee_line": (50, 205, 50),  # Lime Green
            "hip_line": (255, 215, 0),  # Gold
            "angle_arc": (186, 85, 211),  # Medium Purple
            "progress_full": (34, 139, 34),  # Forest Green
            "progress_partial": (255, 165, 0),  # Orange
            "progress_low": (220, 20, 60),  # Crimson
            "background": (15, 15, 25),  # Very dark background
        }

        # Font settings with improved readability
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 1
        self.small_font_scale = 0.4

        # For smooth angle visualization
        self.angle_history = deque(maxlen=10)

        # Try to load a better font for degree symbol rendering
        self.pil_font = None
        try:
            # Try to use a system font that renders degree symbols correctly
            # You might need to adjust this path based on your system
            font_paths = [
                "C:/Windows/Fonts/arial.ttf",  # Windows
                "/System/Library/Fonts/Arial.ttf",  # macOS
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                "/usr/share/fonts/TTF/arial.ttf",  # Arch Linux
            ]

            for font_path in font_paths:
                if os.path.exists(font_path):
                    self.pil_font = ImageFont.truetype(font_path, 24)
                    break

            # If no system font found, use default PIL font
            if self.pil_font is None:
                self.pil_font = ImageFont.load_default()
        except:
            # Fallback to default font
            self.pil_font = ImageFont.load_default()

    def _draw_metric_badge(
        self, frame: np.ndarray, text: str, top_left: tuple, color: tuple
    ) -> None:
        x, y = top_left
        padding_x, padding_y = 8, 6
        (tw, th), _ = cv2.getTextSize(text, self.font, 0.5, 1)
        w = tw + padding_x * 2
        h = th + padding_y * 2
        self.draw_rounded_rectangle(
            frame, (x, y), (x + w, y + h), self.colors["dark"], radius=8
        )
        self.draw_rounded_rectangle(
            frame, (x, y), (x + w, y + h), color, radius=8, thickness=1
        )
        self.draw_text(
            frame,
            text,
            (x + padding_x, y + padding_y + th),
            self.colors["light"],
            scale=0.5,
        )

    def draw_front_view_metrics(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        metrics: dict,
        kpt: dict,
    ) -> np.ndarray:
        """
        Overlay front-view metrics (hip alignment and knees distance).
        """
        try:
            # Draw hips line
            li = kpt.get("left_hip")
            ri = kpt.get("right_hip")
            lk = kpt.get("left_knee")
            rk = kpt.get("right_knee")
            if None in (li, ri, lk, rk):
                return frame

            lh = (int(keypoints[li][0]), int(keypoints[li][1]))
            rh = (int(keypoints[ri][0]), int(keypoints[ri][1]))
            lkpt = (int(keypoints[lk][0]), int(keypoints[lk][1]))
            rkpt = (int(keypoints[rk][0]), int(keypoints[rk][1]))

            # Hips line and horizontal reference
            cv2.line(frame, lh, rh, self.colors["primary"], 2, cv2.LINE_AA)
            midx = (lh[0] + rh[0]) // 2
            midy = (lh[1] + rh[1]) // 2

            # Knee distance line
            cv2.line(frame, lkpt, rkpt, self.colors["warning"], 2, cv2.LINE_AA)

        except:
            pass
        return frame

    def draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        kpt_config: Dict[str, int] = None,
    ) -> np.ndarray:
        """
        Draw full body skeleton for front view.
        """
        if kpt_config is None:
            kpt_config = self.config.get("keypoints", {})

        try:
            # Helper to safely get point
            def get_point(name):
                idx = kpt_config.get(name)
                if idx is not None and idx < len(keypoints):
                    return (int(keypoints[idx][0]), int(keypoints[idx][1]))
                return None

            # 1. Draw Connections (Lines)
            # Define pairs of joints to connect
            connections = [
                # Arms
                ("left_shoulder", "left_elbow"),
                ("left_elbow", "left_wrist"),
                ("right_shoulder", "right_elbow"),
                ("right_elbow", "right_wrist"),
                # Shoulders
                ("left_shoulder", "right_shoulder"),
                # Torso (Shoulders to Hips)
                ("left_shoulder", "left_hip"),
                ("right_shoulder", "right_hip"),
                # Hips
                ("left_hip", "right_hip"),
                # Legs
                ("left_hip", "left_knee"),
                ("left_knee", "left_ankle"),
                ("right_hip", "right_knee"),
                ("right_knee", "right_ankle"),
            ]

            line_color = self.colors["info"] # Dodger Blue for skeleton
            
            for start_name, end_name in connections:
                p1 = get_point(start_name)
                p2 = get_point(end_name)
                if p1 and p2:
                    cv2.line(frame, p1, p2, line_color, 2, cv2.LINE_AA)

            # 2. Draw Keypoints (Joints)
            joints = [
                "left_shoulder", "right_shoulder",
                "left_elbow", "right_elbow",
                "left_wrist", "right_wrist",
                "left_hip", "right_hip",
                "left_knee", "right_knee",
                "left_ankle", "right_ankle",
                "nose", "left_eye", "right_eye"
            ]

            kp_color = self.colors["keypoints"] # Hot pink
            
            for name in joints:
                p = get_point(name)
                if p:
                    # Outer glow
                    cv2.circle(frame, p, 6, kp_color, 1, cv2.LINE_AA)
                    # Center
                    cv2.circle(frame, p, 3, self.colors["light"], -1, cv2.LINE_AA)

        except Exception as e:
            # logger.error(f"Error drawing skeleton: {e}")
            pass

        return frame

    def draw_rounded_rectangle(
        self,
        frame: np.ndarray,
        top_left: tuple,
        bottom_right: tuple,
        color: tuple,
        radius: int = 10,
        thickness: int = -1,
    ) -> None:
        """
        Draw a rounded rectangle.

        Args:
            frame: Video frame to draw on
            top_left: Top left corner (x, y)
            bottom_right: Bottom right corner (x, y)
            color: Rectangle color
            radius: Corner radius
            thickness: Line thickness (-1 for filled)
        """
        x1, y1 = top_left
        x2, y2 = bottom_right

        # Draw filled rectangles
        if thickness == -1:
            # Main rectangle
            cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            # Top rectangle
            cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)

            # Draw circles at corners
            cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1)
        else:
            # Outline with rounded corners
            cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)

            # Draw arcs at corners
            cv2.ellipse(
                frame,
                (x1 + radius, y1 + radius),
                (radius, radius),
                180,
                0,
                90,
                color,
                thickness,
                cv2.LINE_AA
            )
            cv2.ellipse(
                frame,
                (x2 - radius, y1 + radius),
                (radius, radius),
                270,
                0,
                90,
                color,
                thickness,
                cv2.LINE_AA
            )
            cv2.ellipse(
                frame,
                (x1 + radius, y2 - radius),
                (radius, radius),
                90,
                0,
                90,
                color,
                thickness,
                cv2.LINE_AA
            )
            cv2.ellipse(
                frame,
                (x2 - radius, y2 - radius),
                (radius, radius),
                0,
                0,
                90,
                color,
                thickness,
                cv2.LINE_AA
            )

    def draw_keypoints(
        self, frame: np.ndarray, keypoints: np.ndarray, facing_side: str = "right",
        mediapipe_points: dict = None
    ) -> np.ndarray:
        """
        Draw keypoints with modern styling.

        Args:
            frame: Video frame to draw on
            keypoints: YOLO keypoints array
            facing_side: Which side is facing the camera
            mediapipe_points: Optional dict containing additional MediaPipe points like heel, foot_index

        Returns:
            Frame with keypoints drawn
        """
        # Get keypoint indices based on facing side
        indices = self._get_keypoint_indices(facing_side)
        shoulder_idx = indices["shoulder"]
        elbow_idx = indices["elbow"]
        wrist_idx = indices["wrist"]
        hip_idx = indices["hip"]
        knee_idx = indices["knee"]
        ankle_idx = indices["ankle"]

        # Draw standard YOLO keypoints with modern styling
        # Added Elbow and Wrist
        for idx in [shoulder_idx, elbow_idx, wrist_idx, hip_idx, knee_idx, ankle_idx]:
            x, y = int(keypoints[idx][0]), int(keypoints[idx][1])

            # Special Pivot Point style for ankle
            if idx == ankle_idx:
                # Pivot marker with enhanced styling
                cv2.circle(frame, (x, y), 12, self.colors["primary"], 2)  # Outer ring
                cv2.circle(frame, (x, y), 6, self.colors["primary"], -1)  # Solid center
            else:
                # Standard keypoint with enhanced styling
                # Outer glow
                cv2.circle(frame, (x, y), 10, self.colors["keypoints"], 2)
                # Inner circle
                cv2.circle(frame, (x, y), 6, self.colors["light"], -1)
                # Center dot
                cv2.circle(frame, (x, y), 2, self.colors["keypoints"], -1)

        # Draw additional MediaPipe points if provided
        if mediapipe_points:
            # Draw heel point
            if 'heel' in mediapipe_points and mediapipe_points['heel'] is not None:
                heel_x, heel_y = int(mediapipe_points['heel'][0]), int(mediapipe_points['heel'][1])
                # Draw heel as a smaller circle with different color
                cv2.circle(frame, (heel_x, heel_y), 8, self.colors["warning"], 2)
                cv2.circle(frame, (heel_x, heel_y), 3, self.colors["warning"], -1)

            # Draw foot index point
            if 'foot_index' in mediapipe_points and mediapipe_points['foot_index'] is not None:
                foot_x, foot_y = int(mediapipe_points['foot_index'][0]), int(mediapipe_points['foot_index'][1])
                # Draw foot index as a smaller circle with different color
                cv2.circle(frame, (foot_x, foot_y), 8, self.colors["info"], 2)
                cv2.circle(frame, (foot_x, foot_y), 3, self.colors["info"], -1)

        return frame

    def draw_lines(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        knee_angle: float,
        hip_angle: float,
        facing_side: str = "right",
        mediapipe_points: dict = None,
    ) -> np.ndarray:
        """
        Draw modern angle lines with enhanced styling.
        Includes arm lines for better posture visualization.

        Args:
            frame: Video frame to draw on
            keypoints: YOLO keypoints array
            knee_angle: Current knee angle
            hip_angle: Current hip angle
            facing_side: Which side is facing the camera

        Returns:
            Frame with lines and angles drawn
        """
        # Get keypoint indices based on facing side
        indices = self._get_keypoint_indices(facing_side)
        shoulder_idx = indices["shoulder"]
        elbow_idx = indices["elbow"]
        wrist_idx = indices["wrist"]
        hip_idx = indices["hip"]
        knee_idx = indices["knee"]
        ankle_idx = indices["ankle"]

        # Get points
        shoulder_point = (
            int(keypoints[shoulder_idx][0]),
            int(keypoints[shoulder_idx][1]),
        )
        elbow_point = (
            int(keypoints[elbow_idx][0]),
            int(keypoints[elbow_idx][1]),
        )
        wrist_point = (
            int(keypoints[wrist_idx][0]),
            int(keypoints[wrist_idx][1]),
        )
        hip_point = (int(keypoints[hip_idx][0]), int(keypoints[hip_idx][1]))
        knee_point = (int(keypoints[knee_idx][0]), int(keypoints[knee_idx][1]))
        ankle_point = (int(keypoints[ankle_idx][0]), int(keypoints[ankle_idx][1]))

        # Draw arm lines (Shoulder -> Elbow -> Wrist) with distinct color
        # Using 'info' color (Dodger Blue) which is distinct from hips/knee lines
        arm_color = self.colors["info"] 
        cv2.line(frame, shoulder_point, elbow_point, arm_color, 3, cv2.LINE_AA)
        cv2.line(frame, elbow_point, wrist_point, arm_color, 3, cv2.LINE_AA)

        # Draw knee angle lines with enhanced styling
        cv2.line(frame, hip_point, knee_point, self.colors["knee_line"], 4, cv2.LINE_AA)
        cv2.line(frame, knee_point, ankle_point, self.colors["knee_line"], 4, cv2.LINE_AA)

        # Draw hip angle lines with enhanced styling
        cv2.line(frame, shoulder_point, hip_point, self.colors["hip_line"], 4, cv2.LINE_AA)
        cv2.line(frame, hip_point, knee_point, self.colors["hip_line"], 4, cv2.LINE_AA)

        # Draw foot lines if MediaPipe points are available (Ankle -> Heel -> Foot Index)
        if mediapipe_points:
            heel_pt = mediapipe_points.get('heel')
            foot_pt = mediapipe_points.get('foot_index')
            
            # Ankle -> Heel
            if heel_pt is not None:
                heel_coord = (int(heel_pt[0]), int(heel_pt[1]))
                cv2.line(frame, ankle_point, heel_coord, self.colors["knee_line"], 3, cv2.LINE_AA)
                
                # Heel -> Foot Index
                if foot_pt is not None:
                    foot_coord = (int(foot_pt[0]), int(foot_pt[1]))
                    cv2.line(frame, heel_coord, foot_coord, self.colors["knee_line"], 3, cv2.LINE_AA)

        # Draw angle arcs with enhanced styling
        self.draw_angle_arc(
            frame, knee_point, hip_point, ankle_point, 40, self.colors["angle_arc"]
        )
        self.draw_angle_arc(
            frame, hip_point, shoulder_point, knee_point, 35, self.colors["angle_arc"]
        )

        # Draw angle text with enhanced styling
        # Knee angle text
        knee_text_pos = (knee_point[0] + 50, knee_point[1] - 20)
        self.draw_text(
            frame, f"{knee_angle:.0f}°", knee_text_pos, self.colors["knee_line"], scale=0.7
        )

        # Hip angle text
        hip_text_pos = (hip_point[0] + 50, hip_point[1] + 30)
        self.draw_text(
            frame, f"{hip_angle:.0f}°", hip_text_pos, self.colors["hip_line"], scale=0.7
        )

        return frame

    def draw_angle_arc(
        self,
        frame: np.ndarray,
        center: tuple,
        p1: tuple,
        p3: tuple,
        radius: int = 50,
        color: tuple = (156, 39, 176),
    ) -> None:
        """
        Draw an arc to visualize the angle.

        Args:
            frame: Video frame to draw on
            center: Center point of the angle
            p1: First point
            p3: Third point
            radius: Arc radius
            color: Arc color
        """
        try:
            # Calculate vectors
            v1 = np.array(p1) - np.array(center)
            v3 = np.array(p3) - np.array(center)

            # Calculate angles in degrees [0, 360)? No, atan2 is (-180, 180]
            angle1 = math.degrees(math.atan2(v1[1], v1[0]))
            angle3 = math.degrees(math.atan2(v3[1], v3[0]))

            # We want to draw from angle1 to angle3, or vice versa, such that we cover the MINOR arc (<180)
            # Normalize angles to [0, 360) for easier logic
            a1 = angle1 % 360
            a3 = angle3 % 360

            # Calculate clockwise difference from a1 to a3
            diff = (a3 - a1 + 360) % 360

            if diff <= 180:
                # a1 -> a3 is the minor arc (Clockwise in OpenCV ellipse coordinate space? check below)
                # OpenCV ellipse draws CLOCKWISE from startAngle to endAngle.
                # So if a1->a3 is the short way, we just draw start=a1, end=a3?
                # Actually, diff is calculates counter-clockwise or clockwise depending on axes.
                # Since Y is down, standard atan2 (y,x) increases CLOCKWISE visually.
                # So (a3 - a1) is the clockwise angle from a1 to a3.
                start = a1
                end = a3
                # But wait, if end < start in value (e.g. 350 -> 10), OpenCV expects start=350, end=370?
                # Or simply pass them as is? OpenCV handles wrapped angles usually if we calculate arc length.
                # However, to be safe and consistent with ellipse params:
                if end < start:
                   end += 360
            else:
                # a1 -> a3 is the MAJOR arc (>180). We want the other way: a3 -> a1.
                start = a3
                end = a1
                if end < start:
                   end += 360

            # Draw the arc with enhanced styling
            cv2.ellipse(
                frame,
                center,
                (radius, radius),
                0,
                start,
                end,
                color,
                2,
                cv2.LINE_AA
            )

        except Exception as e:
            pass  # Skip if calculation fails

    def draw_text(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple,
        color: tuple,
        scale: float = None,
        thickness: int = None,
    ) -> None:
        """
        Draw modern text with better degree symbol rendering.

        Args:
            frame: Video frame to draw on
            text: Text to draw
            position: Text position (x, y)
            color: Text color
            scale: Font scale
            thickness: Font thickness
        """
        if scale is None:
            scale = self.font_scale
        if thickness is None:
            thickness = self.font_thickness

        x, y = position

        # For better degree symbol rendering, use PIL
        if "°" in text and self.pil_font is not None:
            try:
                # Convert BGR to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(pil_image)

                # PIL uses top-left coordinates, OpenCV uses baseline-left.
                # Adjust y-coordinate to match OpenCV's baseline positioning.
                # Get the bounding box of the text to find its height.
                bbox = draw.textbbox((0, 0), text, font=self.pil_font)
                text_height = bbox[3] - bbox[1]  # bottom - top
                # Shift y up by the text height to align baselines
                pil_y = y - text_height

                # Draw text with PIL
                # Convert color from BGR to RGB
                rgb_color = (color[2], color[1], color[0])  # BGR to RGB
                draw.text((x, pil_y), text, font=self.pil_font, fill=rgb_color)

                # Convert back to BGR
                frame[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                return
            except:
                # If PIL fails, fall back to OpenCV
                pass

        # Fallback to OpenCV text rendering with enhanced styling
        # Draw subtle shadow for better readability
        cv2.putText(
            frame, text, (x + 1, y + 1), self.font, scale, (30, 30, 30), thickness
        )

        # Draw main text with anti-aliasing
        cv2.putText(frame, text, (x, y), self.font, scale, color, thickness, cv2.LINE_AA)

    def draw_stats(
        self,
        frame: np.ndarray,
        successful_reps: int,
        unsuccessful_reps: int,
        total_reps: int,
        state: str,
        movement_status: Dict = None,
        failure_justifications: list = None,
    ) -> np.ndarray:
        """
        Draw modern statistics panel with clean design.

        Args:
            frame: Video frame to draw on
            successful_reps: Number of successful reps
            unsuccessful_reps: Number of unsuccessful reps
            total_reps: Total reps
            knee_angle: Current knee angle
            hip_angle: Current hip angle
            state: Current exercise state

        Returns:
            Frame with statistics drawn
        """
        frame_height, frame_width = frame.shape[:2]

        # Modern stats panel - Positioned Top Right with transparency
        panel_width = 380
        panel_height = 160

        # Position with padding from top-right corner
        panel_x = frame_width - panel_width - 20
        panel_y = 20

        # Draw stats panel background and border
        self._draw_overlay_panel(
            frame, 
            (panel_x, panel_y, panel_width, panel_height), 
            self.colors["primary"], 
            bg_color=self.colors["dark"],
            alpha=0.6,
            radius=10
        )

        # Draw header
        header_text = "Exercise Stats"
        header_x = panel_x + 12
        header_y = panel_y + 25
        self.draw_text(
            frame, header_text, (header_x, header_y), self.colors["light"], scale=0.6
        )

        # Stats positions
        y_pos = header_y + 25
        line_spacing = 22

        # Rep counts with better visual hierarchy
        self.draw_text(
            frame,
            f"Good: {successful_reps}",
            (header_x, y_pos),
            self.colors["success"],
            scale=0.55
        )
        y_pos += line_spacing

        self.draw_text(
            frame,
            f"Bad: {unsuccessful_reps}",
            (header_x, y_pos),
            self.colors["danger"],
            scale=0.55
        )
        y_pos += line_spacing

        self.draw_text(
            frame,
            f"Total: {total_reps}",
            (header_x, y_pos),
            self.colors["info"],
            scale=0.55
        )
        y_pos += line_spacing

        # State indicator with better visual feedback
        state_text = "UP" if state == "up" else "DOWN"
        state_color = (
            self.colors["success"] if state == "up" else self.colors["warning"]
        )
        self.draw_text(
            frame,
            f"Phase: {state_text}",
            (header_x, y_pos),
            state_color,
            scale=0.55
        )
        y_pos += line_spacing

        # Movement Status
        if movement_status:
            is_ok = movement_status.get("within_bounds", True)
            if is_ok:
                move_text = "Movement: OK"
                move_color = self.colors["success"]
            else:
                move_text = "Mov: EXCEEDED"
                move_color = self.colors["danger"]

            self.draw_text(
                frame,
                move_text,
                (header_x, y_pos),
                move_color,
                scale=0.55
            )
            y_pos += line_spacing

        # Draw Failure Justifications if enabled and available
        show_details = self.config.get("visualization", {}).get("show_failure_justifications", False)
        if show_details and failure_justifications:
            # Combine all justifications into one line
            just_text = ", ".join([j[0].upper() + j[1:] for j in failure_justifications if j])
            display_text = f"! {just_text}" if just_text else ""
            
            if display_text:
                # Calculate required width
                (text_w, text_h), _ = cv2.getTextSize(display_text, self.font, 0.50, self.font_thickness)
                required_width = text_w + 40 # 20px padding each side
                
                # Use the larger of panel_width or required_width
                just_width = max(panel_width, required_width)
                
                # Align right side with the main panel
                panel_right_edge = panel_x + panel_width
                just_x = panel_right_edge - just_width

                # Draw justifications under the main panel
                just_y = panel_y + panel_height + 10
                just_line_height = 28
                just_height = just_line_height + 16 # Single line height + padding
                
                # Draw justifications background and border
                self._draw_overlay_panel(
                    frame,
                    (just_x, just_y, just_width, just_height),
                    self.colors["danger"],
                    bg_color=(20, 20, 20),
                    alpha=0.5,
                    radius=8
                )
                
                # Justification text
                text_y = just_y + 25
                self.draw_text(frame, display_text, (just_x + 20, text_y), self.colors["danger"], scale=0.50)

        return frame

    def draw_progress_bar(self, frame: np.ndarray, knee_angle: float, knee_min: float, knee_max: float) -> np.ndarray:
        """
        Draw a modern progress bar showing squat depth.

        Args:
            frame: Video frame to draw on
            knee_angle: Current knee angle
            knee_min: Minimum knee angle (deepest)
            knee_max: Maximum knee angle (standing)

        Returns:
            Frame with progress bar drawn
        """

        # Progress bar dimensions - positioned at bottom center
        frame_height, frame_width = frame.shape[:2]
        bar_width = 350
        bar_height = 12
        bar_x = (frame_width - bar_width) // 2
        bar_y = frame_height - 40

        # Draw background bar with rounded corners
        self.draw_rounded_rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (80, 80, 80),  # Dark gray
            radius=6,
        )

        # Calculate progress (inverted because lower angle = deeper squat)
        if knee_angle <= knee_min:
            progress = 1.0
            color = self.colors["progress_full"]
        elif knee_angle >= knee_max:
            progress = 0.0
            color = self.colors["progress_low"]
        else:
            progress = (knee_max - knee_angle) / (knee_max - knee_min)
            color = self.colors["progress_partial"]

        # Draw progress fill with rounded corners
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            self.draw_rounded_rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + fill_width, bar_y + bar_height),
                color,
                radius=6,
            )

        # Draw labels
        self.draw_text(
            frame, "Depth", (bar_x - 50, bar_y + 10), self.colors["light"], scale=0.5
        )

        # Percentage text
        percentage = int(progress * 100)
        percentage_text = f"{percentage}%"
        text_x = bar_x + bar_width + 15
        self.draw_text(frame, percentage_text, (text_x, bar_y + 10), color, scale=0.5)

        return frame

    def visualize(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        knee_angle: float,
        hip_angle: float,
        successful_reps: int,
        unsuccessful_reps: int,
        total_reps: int,
        state: str,
        facing_side: str = "right",
        mediapipe_points: dict = None,
        movement_status: dict = None,
        thresholds: dict = None,
        failure_justifications: list = None,
    ) -> np.ndarray:
        """
        Main visualization function with modern design.

        Args:
            frame: Video frame to draw on
            keypoints: YOLO keypoints array
            knee_angle: Current knee angle
            hip_angle: Current hip angle
            successful_reps: Number of successful reps
            unsuccessful_reps: Number of unsuccessful reps
            total_reps: Total reps
            state: Current exercise state
            facing_side: Which side is facing the camera
            mediapipe_points: Optional dict containing additional MediaPipe points like heel, foot_index

        Returns:
            Frame with all visualizations applied
        """
        # Draw modern keypoints (with MediaPipe points if available)
        frame = self.draw_keypoints(frame, keypoints, facing_side, mediapipe_points)

        # Draw modern lines and angles
        frame = self.draw_lines(frame, keypoints, knee_angle, hip_angle, facing_side, mediapipe_points)

        # Draw modern statistics
        frame = self.draw_stats(
            frame,
            successful_reps,
            unsuccessful_reps,
            total_reps,
            state,
            movement_status,
            failure_justifications,
        )

        # Draw modern progress bar
        # Use defaults if thresholds not provided (though they should be)
        knee_min = 80.0
        knee_max = 175.0
        
        if thresholds:
            knee_min = thresholds.get("knee_min", knee_min)
            knee_max = thresholds.get("knee_max", knee_max)
            
        frame = self.draw_progress_bar(frame, knee_angle, knee_min, knee_max)

        return frame

    def draw_movement_boundary(self, frame: np.ndarray, boundary_info: Dict) -> np.ndarray:
        """
        Draw the movement boundary box on the frame.
        """
        try:
            if boundary_info.get("boundaries") is None:
                return frame

            # Get frame dimensions
            height, width = frame.shape[:2]

            # Get boundary coordinates
            boundaries = boundary_info["boundaries"]
            x_min = int(boundaries["x_min"])
            x_max = int(boundaries["x_max"])
            # Use full frame height for visualization
            y_min = 0 
            y_max = height
            
            # Clamp X to frame
            x_min = max(0, min(width, x_min))
            x_max = max(0, min(width, x_max))

            # Determine color based on whether within bounds
            if boundary_info.get("within_bounds", True):
                # Green for within bounds
                color = self.colors["success"]
                text = "Movement: OK"
            else:
                # Red for exceeded bounds
                color = self.colors["danger"]
                text = "Movement: EXCEEDED"
                
                # Check for specific directions
                status = boundary_info.get("boundary_status", {})
                failures = []
                if status.get("out_of_bounds_left"): failures.append("LEFT")
                if status.get("out_of_bounds_right"): failures.append("RIGHT")
                
                if failures:
                     text += f" ({', '.join(failures)})"

            # Draw the boundary rectangle
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

            # Draw center point (initial position)
            if boundary_info.get("initial_position"):
                initial_pos = boundary_info["initial_position"]
                center_x, center_y = int(initial_pos[0]), int(initial_pos[1])
                cv2.circle(frame, (center_x, center_y), 5, self.colors["hip_line"], -1)  # Gold center

            # Draw current position
            if boundary_info.get("current_position"):
                current_pos = boundary_info["current_position"]
                curr_x, curr_y = int(current_pos[0]), int(current_pos[1])
                # Draw yellow dot for current hip center
                cv2.circle(frame, (curr_x, curr_y), 5, self.colors["warning"], -1)  
                # Draw line from initial to current
                if boundary_info.get("initial_position"):
                    initial_pos = boundary_info["initial_position"]
                    init_x, init_y = int(initial_pos[0]), int(initial_pos[1])
                    cv2.line(frame, (init_x, init_y), (curr_x, curr_y), self.colors["light"], 1)

            # Text overlay removed - moved to stats panel
            # self.draw_text(frame, text, (10, 30), color, scale=0.7)

            # Add body dimension info (Hidden) 
            # body_dims = boundary_info.get("body_dimensions", {})
            # if body_dims:
            #      thigh = body_dims.get("thigh_length", 0)
            #      shin = body_dims.get("shin_length", 0)
            #      info_text = f"Thigh: {thigh:.1f}px, Shin: {shin:.1f}px"
            #      self.draw_text(frame, info_text, (10, 60), self.colors["light"], scale=0.5)

            return frame
        except Exception as e:
            # logger.warning(f"Error drawing movement boundary: {e}") # Visualization class usually doesn't have logger, or imports it?
            # It imports os, math, cv2, np... checked file: imports usually at top. 
            # checked visualization.py: NO logger imported.
            pass
            return frame

    def _get_keypoint_indices(self, facing_side: str) -> Dict[str, int]:
        """Helper to get keypoint indices based on facing side."""
        side_prefix = "left" if facing_side == "left" else "right"
        kpt = self.config["keypoints"]
        return {
            "shoulder": kpt[f"{side_prefix}_shoulder"],
            "elbow": kpt[f"{side_prefix}_elbow"],
            "wrist": kpt[f"{side_prefix}_wrist"],
            "hip": kpt[f"{side_prefix}_hip"],
            "knee": kpt[f"{side_prefix}_knee"],
            "ankle": kpt[f"{side_prefix}_ankle"],
        }

    def _draw_overlay_panel(
        self,
        frame: np.ndarray,
        rect: tuple,
        border_color: tuple,
        bg_color: tuple = (20, 20, 20),
        alpha: float = 0.5,
        radius: int = 10
    ) -> None:
        """Helper to draw a semi-transparent rounded panel."""
        x, y, w, h = rect
        overlay = frame.copy()
        self.draw_rounded_rectangle(
            overlay, (x, y), (x + w, y + h), bg_color, radius=radius
        )
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        self.draw_rounded_rectangle(
            frame, (x, y), (x + w, y + h), border_color, radius=radius, thickness=1
        )

