"""
Person segmentation utilities for AI Trainer application.
Uses YOLO11s-seg model to segment the person doing the workout.
"""

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO
from typing import Optional, Tuple


class PersonSegmenter:
    """Person segmentation using YOLO11s-seg model."""

    def __init__(self, model_path: str):
        """
        Initialize the person segmenter.

        Args:
            model_path: Path to YOLO11s-seg model
        """
        self.model_path = model_path
        self.model = None
        
        try:
            self.model = YOLO(model_path)
            logger.info(f"Person segmentation model loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")

    def segment_person(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Segment person in the frame and return the mask.

        Args:
            frame: Input video frame

        Returns:
            Binary mask of the person (None if no person detected or model failed)
        """
        if self.model is None:
            return None

        try:
            # Run segmentation
            results = self.model(frame, verbose=False)
            
            # Check if any results with masks
            if not results or results[0].masks is None:
                return None

            # Get masks and filter for person class (class 0 in COCO dataset)
            masks = results[0].masks.data.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
            
            # Find person masks (class 0)
            person_masks = []
            for i, cls in enumerate(classes):
                if cls == 0:  # Person class
                    person_masks.append(masks[i])
            
            if not person_masks:
                return None
            
            # Combine all person masks if multiple detected
            combined_mask = np.zeros_like(person_masks[0])
            for mask in person_masks:
                combined_mask = np.maximum(combined_mask, mask)
            
            # Resize mask to frame dimensions if needed
            if combined_mask.shape != frame.shape[:2]:
                combined_mask = cv2.resize(combined_mask, (frame.shape[1], frame.shape[0]))
            
            return combined_mask

        except Exception as e:
            logger.debug(f"Error in person segmentation: {e}")
            return None

    def apply_mask_overlay(
        self, 
        frame: np.ndarray, 
        mask: np.ndarray, 
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Apply segmentation mask overlay to the frame.

        Args:
            frame: Original frame
            mask: Binary segmentation mask
            color: Overlay color (B, G, R)
            alpha: Transparency factor (0.0 to 1.0)

        Returns:
            Frame with mask overlay applied
        """
        try:
            # Convert mask to 3-channel if needed
            if len(mask.shape) == 2:
                mask_3ch = cv2.merge([mask, mask, mask])
            else:
                mask_3ch = mask

            # Create colored overlay
            overlay = np.zeros_like(frame)
            overlay[mask > 0.5] = color

            # Apply overlay with transparency
            result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

            return result

        except Exception as e:
            logger.debug(f"Error applying mask overlay: {e}")
            return frame

    def draw_mask_contour(
        self, 
        frame: np.ndarray, 
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw mask contour on the frame.

        Args:
            frame: Original frame
            mask: Binary segmentation mask
            color: Contour color (B, G, R)
            thickness: Contour line thickness

        Returns:
            Frame with mask contour drawn
        """
        try:
            # Convert mask to uint8 if needed
            if mask.dtype != np.uint8:
                mask_uint8 = (mask * 255).astype(np.uint8)
            else:
                mask_uint8 = mask

            # Find contours
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours
            cv2.drawContours(frame, contours, -1, color, thickness)

            return frame

        except Exception as e:
            logger.debug(f"Error drawing mask contour: {e}")
            return frame
