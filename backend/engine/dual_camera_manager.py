"""
Dual Camera Manager for AIMaxiFAI

Provides synchronized capture from side and front cameras for
comprehensive exercise form analysis.

Supports:
- Software frame synchronization
- GStreamer pipelines for low-latency capture
- Frame timestamp alignment
- Camera calibration for distortion correction
"""

import os
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Callable
from enum import Enum
import cv2
import numpy as np
from loguru import logger


class CameraPosition(Enum):
    """Camera position identifiers."""
    SIDE = "side"
    FRONT = "front"


@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    device_id: int  # /dev/video0 = 0, /dev/video1 = 1, etc.
    position: CameraPosition
    width: int = 1280
    height: int = 720
    fps: int = 30
    use_gstreamer: bool = False
    
    # Distortion correction (for wide-angle lenses)
    calibration_file: Optional[str] = None
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None
    
    # Physical placement
    distance_from_subject_cm: float = 60.0
    height_from_ground_cm: float = 90.0  # Camera height
    angle_degrees: float = 0.0  # Tilt angle
    rotation: int = 0  # Rotation in degrees (0, 90, 180, 270)


@dataclass
class SyncedFrame:
    """Container for synchronized frames from both cameras."""
    side_frame: Optional[np.ndarray] = None
    front_frame: Optional[np.ndarray] = None
    side_timestamp: float = 0.0
    front_timestamp: float = 0.0
    frame_number: int = 0
    sync_error_ms: float = 0.0  # Time difference between frames
    
    @property
    def is_complete(self) -> bool:
        """Check if both frames are available."""
        return self.side_frame is not None and self.front_frame is not None
    
    @property
    def is_synced(self) -> bool:
        """Check if frames are within acceptable sync threshold (33ms = 1 frame at 30fps)."""
        return self.sync_error_ms < 33.0


class CameraCapture:
    """Single camera capture with optional GStreamer pipeline."""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=5)
        self._capture_thread: Optional[threading.Thread] = None
        self._last_frame_time = 0.0
        
        # Undistortion maps (computed once from calibration)
        self._map1: Optional[np.ndarray] = None
        self._map2: Optional[np.ndarray] = None
        
    def _build_gstreamer_pipeline(self) -> str:
        """Build GStreamer pipeline for low-latency capture."""
        # Linux V4L2 pipeline
        pipeline = (
            f"v4l2src device=/dev/video{self.config.device_id} ! "
            f"video/x-raw,width={self.config.width},height={self.config.height},"
            f"framerate={self.config.fps}/1 ! "
            f"videoconvert ! "
            f"appsink drop=1 max-buffers=2"
        )
        return pipeline
    
    def _build_windows_gstreamer_pipeline(self) -> str:
        """Build GStreamer pipeline for Windows."""
        pipeline = (
            f"mfvideosrc device-index={self.config.device_id} ! "
            f"video/x-raw,width={self.config.width},height={self.config.height},"
            f"framerate={self.config.fps}/1 ! "
            f"videoconvert ! "
            f"appsink drop=1 max-buffers=2"
        )
        return pipeline
    
    def _load_calibration(self) -> bool:
        """Load camera calibration for distortion correction."""
        if self.config.calibration_file and os.path.exists(self.config.calibration_file):
            try:
                data = np.load(self.config.calibration_file)
                self.config.camera_matrix = data['camera_matrix']
                self.config.dist_coeffs = data['dist_coeffs']
                logger.info(f"Loaded calibration for {self.config.position.value} camera")
                return True
            except Exception as e:
                logger.warning(f"Failed to load calibration: {e}")
        return False
    
    def _init_undistortion_maps(self) -> None:
        """Initialize undistortion maps for fast frame correction."""
        if self.config.camera_matrix is not None and self.config.dist_coeffs is not None:
            h, w = self.config.height, self.config.width
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                self.config.camera_matrix, 
                self.config.dist_coeffs, 
                (w, h), 1, (w, h)
            )
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                self.config.camera_matrix,
                self.config.dist_coeffs,
                None,
                new_camera_matrix,
                (w, h),
                cv2.CV_16SC2
            )
            logger.info(f"Initialized undistortion maps for {self.config.position.value}")
    
    def open(self) -> bool:
        """Open camera capture."""
        try:
            if self.config.use_gstreamer:
                # Determine OS and build appropriate pipeline
                if os.name == 'nt':  # Windows
                    pipeline = self._build_windows_gstreamer_pipeline()
                else:  # Linux
                    pipeline = self._build_gstreamer_pipeline()
                
                logger.info(f"Opening {self.config.position.value} camera with GStreamer: {pipeline}")
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            else:
                logger.info(f"Opening {self.config.position.value} camera: device {self.config.device_id}")
                
                # Use DirectShow on Windows for faster opening
                if os.name == 'nt':
                     self.cap = cv2.VideoCapture(self.config.device_id, cv2.CAP_DSHOW)
                else:
                     self.cap = cv2.VideoCapture(self.config.device_id)
                
                # Set properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffering for low latency
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open {self.config.position.value} camera")
                return False
            
            # Load calibration and init undistortion
            self._load_calibration()
            self._init_undistortion_maps()
            
            logger.success(f"Opened {self.config.position.value} camera successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False
    
    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Rotate frame if configured."""
        if self.config.rotation == 0:
            return frame
        elif self.config.rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.config.rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.config.rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def close(self) -> None:
        """Close camera capture."""
        self.is_running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info(f"Closed {self.config.position.value} camera")
    
    def _capture_loop(self) -> None:
        """Background thread for continuous capture."""
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            timestamp = time.time()
            
            if ret:
                if self._map1 is not None and self._map2 is not None:
                    frame = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
                
                # Apply rotation
                frame = self._rotate_frame(frame)
                
                # Put in queue (drop old frames if full)
                try:
                    self._frame_queue.put_nowait((frame, timestamp))
                except queue.Full:
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait((frame, timestamp))
                    except queue.Empty:
                        pass
            else:
                time.sleep(0.001)
    
    def start_capture(self) -> None:
        """Start background capture thread."""
        if not self.is_running:
            self.is_running = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            logger.info(f"Started capture thread for {self.config.position.value} camera")
    
    def get_frame(self, timeout: float = 0.1) -> Tuple[Optional[np.ndarray], float]:
        """Get latest frame with timestamp."""
        try:
            frame, timestamp = self._frame_queue.get(timeout=timeout)
            self._last_frame_time = timestamp
            return frame, timestamp
        except queue.Empty:
            return None, 0.0
    
    def read_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Synchronous frame read (alternative to threaded capture)."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            timestamp = time.time()
            
            if ret:
                if self._map1 is not None and self._map2 is not None:
                    frame = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
                
                # Apply rotation
                frame = self._rotate_frame(frame)
                return frame, timestamp
        return None, 0.0


class DualCameraManager:
    """
    Manages synchronized capture from side and front cameras.
    
    Provides frame synchronization and combined analysis support.
    """
    
    def __init__(
        self,
        side_config: CameraConfig,
        front_config: CameraConfig,
        sync_threshold_ms: float = 50.0,
    ):
        """
        Initialize dual camera manager.
        
        Args:
            side_config: Configuration for side view camera
            front_config: Configuration for front view camera
            sync_threshold_ms: Maximum acceptable time difference between frames
        """
        self.side_camera = CameraCapture(side_config)
        self.front_camera = CameraCapture(front_config)
        self.sync_threshold_ms = sync_threshold_ms
        
        self._is_running = False
        self._sync_thread: Optional[threading.Thread] = None
        self._synced_frame_queue: queue.Queue = queue.Queue(maxsize=5)
        self._frame_counter = 0
        
        # Callbacks for frame processing
        self._on_synced_frame: Optional[Callable[[SyncedFrame], None]] = None
        
        logger.info("DualCameraManager initialized")
    
    def open(self) -> bool:
        """Open both cameras."""
        side_ok = self.side_camera.open()
        front_ok = self.front_camera.open()
        
        if not side_ok:
            logger.error("Failed to open side camera")
        if not front_ok:
            logger.error("Failed to open front camera")
        
        return side_ok and front_ok
    
    def close(self) -> None:
        """Close both cameras."""
        self._is_running = False
        if self._sync_thread:
            self._sync_thread.join(timeout=2.0)
        self.side_camera.close()
        self.front_camera.close()
        logger.info("DualCameraManager closed")
    
    def start(self) -> None:
        """Start synchronized capture."""
        if not self._is_running:
            self.side_camera.start_capture()
            self.front_camera.start_capture()
            
            self._is_running = True
            self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            self._sync_thread.start()
            
            logger.info("Started synchronized dual-camera capture")
    
    def _sync_loop(self) -> None:
        """Background thread for frame synchronization."""
        while self._is_running:
            side_frame, side_ts = self.side_camera.get_frame(timeout=0.05)
            front_frame, front_ts = self.front_camera.get_frame(timeout=0.05)
            
            if side_frame is not None or front_frame is not None:
                self._frame_counter += 1
                
                # Calculate sync error
                if side_ts > 0 and front_ts > 0:
                    sync_error_ms = abs(side_ts - front_ts) * 1000
                else:
                    sync_error_ms = float('inf')
                
                synced = SyncedFrame(
                    side_frame=side_frame,
                    front_frame=front_frame,
                    side_timestamp=side_ts,
                    front_timestamp=front_ts,
                    frame_number=self._frame_counter,
                    sync_error_ms=sync_error_ms,
                )
                
                # Put in queue
                try:
                    self._synced_frame_queue.put_nowait(synced)
                except queue.Full:
                    try:
                        self._synced_frame_queue.get_nowait()
                        self._synced_frame_queue.put_nowait(synced)
                    except queue.Empty:
                        pass
                
                # Callback if registered
                if self._on_synced_frame:
                    self._on_synced_frame(synced)
            
            time.sleep(0.001)  # Prevent busy-waiting
    
    def get_synced_frame(self, timeout: float = 0.1) -> Optional[SyncedFrame]:
        """Get next synchronized frame pair."""
        try:
            return self._synced_frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def set_frame_callback(self, callback: Callable[[SyncedFrame], None]) -> None:
        """Set callback for each synchronized frame."""
        self._on_synced_frame = callback
    
    def read_synced_frame(self) -> SyncedFrame:
        """Synchronous read of both cameras (alternative to threaded)."""
        side_frame, side_ts = self.side_camera.read_frame()
        front_frame, front_ts = self.front_camera.read_frame()
        
        self._frame_counter += 1
        
        sync_error_ms = abs(side_ts - front_ts) * 1000 if side_ts > 0 and front_ts > 0 else float('inf')
        
        return SyncedFrame(
            side_frame=side_frame,
            front_frame=front_frame,
            side_timestamp=side_ts,
            front_timestamp=front_ts,
            frame_number=self._frame_counter,
            sync_error_ms=sync_error_ms,
        )
    
    @staticmethod
    def create_side_by_side_view(
        synced_frame: SyncedFrame,
        labels: bool = True,
    ) -> np.ndarray:
        """Create side-by-side visualization of both camera views."""
        side = synced_frame.side_frame
        front = synced_frame.front_frame
        
        # Handle missing frames
        if side is None and front is None:
            return np.zeros((480, 1280, 3), dtype=np.uint8)
        
        if side is None:
            side = np.zeros_like(front)
        if front is None:
            front = np.zeros_like(side)
        
        # Resize to same height
        h = min(side.shape[0], front.shape[0])
        side_resized = cv2.resize(side, (int(side.shape[1] * h / side.shape[0]), h))
        front_resized = cv2.resize(front, (int(front.shape[1] * h / front.shape[0]), h))
        
        # Concatenate
        combined = np.hstack([side_resized, front_resized])
        
        # Add labels
        if labels:
            cv2.putText(combined, "SIDE VIEW", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "FRONT VIEW", (side_resized.shape[1] + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Sync status
            sync_color = (0, 255, 0) if synced_frame.is_synced else (0, 0, 255)
            cv2.putText(combined, f"Sync: {synced_frame.sync_error_ms:.1f}ms", 
                       (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sync_color, 2)
        
        return combined


def create_dual_camera_setup(
    side_device: int = 0,
    front_device: int = 1,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    use_gstreamer: bool = False,
    side_calibration: Optional[str] = None,
    front_calibration: Optional[str] = None,
) -> DualCameraManager:
    """
    Convenience function to create a dual camera setup.
    
    Args:
        side_device: Device ID for side camera
        front_device: Device ID for front camera
        width: Frame width
        height: Frame height
        fps: Frames per second
        use_gstreamer: Whether to use GStreamer pipelines
        side_calibration: Path to side camera calibration file
        front_calibration: Path to front camera calibration file
    
    Returns:
        Configured DualCameraManager
    """
    side_config = CameraConfig(
        device_id=side_device,
        position=CameraPosition.SIDE,
        width=width,
        height=height,
        fps=fps,
        use_gstreamer=use_gstreamer,
        calibration_file=side_calibration,
    )
    
    front_config = CameraConfig(
        device_id=front_device,
        position=CameraPosition.FRONT,
        width=width,
        height=height,
        fps=fps,
        use_gstreamer=use_gstreamer,
        calibration_file=front_calibration,
    )
    
    return DualCameraManager(side_config, front_config)
