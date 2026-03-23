"""
Shared webcam capture manager to avoid device conflicts on Windows.
"""

import cv2
from loguru import logger
from typing import Optional


class SharedWebcamCapture:
    """
    A wrapper around cv2.VideoCapture that helps manage shared webcam access.
    
    This is particularly useful on Windows where multiple processes trying to
    access the same webcam device can cause conflicts. This class provides
    a simple interface similar to cv2.VideoCapture but with better error handling.
    """
    
    def __init__(self, webcam_id: int = 0):
        """
        Initialize webcam capture.
        
        Args:
            webcam_id: The webcam device ID (default: 0)
        """
        self.webcam_id = webcam_id
        self._cap: Optional[cv2.VideoCapture] = None
        self._open()
    
    def _open(self):
        """Open the webcam device."""
        try:
            self._cap = cv2.VideoCapture(self.webcam_id)
            if not self._cap.isOpened():
                logger.error(f"Failed to open webcam {self.webcam_id}")
                raise IOError(f"Could not open webcam {self.webcam_id}")
            logger.info(f"Successfully opened webcam {self.webcam_id}")
        except Exception as e:
            logger.error(f"Error opening webcam {self.webcam_id}: {e}")
            if self._cap:
                self._cap.release()
            raise
    
    def read(self):
        """
        Read a frame from the webcam.
        
        Returns:
            Tuple of (ret, frame) where ret is True if frame was read successfully
        """
        if not self._cap:
            return False, None
        return self._cap.read()
    
    def isOpened(self) -> bool:
        """Check if the webcam is opened."""
        if not self._cap:
            return False
        return self._cap.isOpened()
    
    def release(self):
        """Release the webcam device."""
        if self._cap:
            self._cap.release()
            self._cap = None
            logger.info(f"Released webcam {self.webcam_id}")
    
    def get(self, prop_id):
        """Get a property from the capture."""
        if not self._cap:
            return 0
        return self._cap.get(prop_id)
    
    def set(self, prop_id, value):
        """Set a property on the capture."""
        if not self._cap:
            return False
        return self._cap.set(prop_id, value)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()







