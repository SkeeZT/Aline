import os
import abc
import cv2
import numpy as np
from loguru import logger
from typing import Any, Optional, Tuple
import requests
import shutil

import mediapipe as mp
from ultralytics import YOLO

class PoseEstimator(abc.ABC):
    """
    Abstract base class for pose estimation.
    """
    
    @abc.abstractmethod
    def process_frame(self, frame: np.ndarray) -> Any:
        """
        Process a single frame and return results in a standardized format.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Standardized result object (TBD, currently returning raw-like object or standardized dict)
        """
        pass
    
    @abc.abstractmethod
    def get_keypoints(self, results: Any, frame_shape: Tuple[int, int] = None) -> Optional[np.ndarray]:
        """
        Extract keypoints in standardized (N, 3) format: [x, y, confidence].
        Indices should match COCO/YOLO format (17 keypoints).
        
        0: Nose
        1: L Eye
        2: R Eye
        3: L Ear
        4: R Ear
        5: L Shoulder
        6: R Shoulder
        7: L Elbow
        8: R Elbow
        9: L Wrist
        10: R Wrist
        11: L Hip
        12: R Hip
        13: L Knee
        14: R Knee
        15: L Ankle
        16: R Ankle
        """
        pass


class YOLOPoseEstimator(PoseEstimator):
    """
    YOLO-based pose estimator.
    """
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold
        logger.info(f"Initialized YOLO Pose Estimator: {model_path}")

    def process_frame(self, frame: np.ndarray) -> Any:
        return self.model(frame, verbose=False)

    def get_keypoints(self, results: Any, frame_shape: Tuple[int, int] = None) -> Optional[np.ndarray]:
        # YOLO results[0].keypoints.xy is (N, 17, 2), conf is (N, 17)
        if results is None or len(results) == 0:
            return None
        
        r = results[0]
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            return None
            
        # Get coordinates and confidence
        # We only take the first person for now (index 0)
        xy = r.keypoints.xy[0].cpu().numpy()  # (17, 2)
        conf = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones((17,)) # (17,)
        
        # Combine to (17, 3) format: [x, y, conf]
        return np.column_stack((xy, conf))


class MediaPipePoseEstimator(PoseEstimator):
    """
    MediaPipe-based pose estimator using the Tasks API.
    """
    def __init__(self, 
                 model_path: str = "./assets/models/pose_landmarker_heavy.task",
                 min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5):
        
        # Verify model exists
        if not os.path.exists(model_path):
             # Try absolute path or default
             logger.warning(f"Model path {model_path} not found. Trying local default.")
             model_path = os.path.abspath(model_path)
             if not os.path.exists(model_path):
                 logger.info(f"Model not found at {model_path}. Attempting to download...")
                 try:
                     # Create directory if it doesn't exist
                     model_dir = os.path.dirname(model_path)
                     os.makedirs(model_dir, exist_ok=True)

                     url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
                     logger.info(f"Downloading from {url}...")
                     
                     response = requests.get(url, stream=True)
                     if response.status_code == 200:
                         with open(model_path, 'wb') as f:
                             response.raw.decode_content = True
                             shutil.copyfileobj(response.raw, f)
                         logger.success(f"Model successfully downloaded to {model_path}")
                     else:
                         raise Exception(f"Failed to download model. Status code: {response.status_code}")
                 
                 except Exception as e:
                     logger.error(f"Failed to download MediaPipe model: {e}")
                     raise FileNotFoundError(f"MediaPipe model not found at {model_path} and download failed.")

        # Import Tasks API components
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Determine running mode
        # For simplicity and stateless compatibility we use IMAGE mode by default.
        # If VIDEO mode is needed, we must provide timestamps. 
        # Since process_frame interface doesn't enforce timestamps, IMAGE mode is safer.
        running_mode = VisionRunningMode.IMAGE

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence, # Use same for presence
            min_tracking_confidence=min_tracking_confidence,
        )
        
        self.landmarker = PoseLandmarker.create_from_options(options)
        logger.info(f"Initialized MediaPipe Pose Estimator (Tasks API) with model: {model_path}")
        
        # Mapping from MediaPipe (33) to YOLO/COCO (17)
        self.mp_to_coco_map = {
            0: 0,   # Nose
            1: 2,   # L Eye
            2: 5,   # R Eye
            3: 7,   # L Ear
            4: 8,   # R Ear
            5: 11,  # L Shoulder
            6: 12,  # R Shoulder
            7: 13,  # L Elbow
            8: 14,  # R Elbow
            9: 15,  # L Wrist
            10: 16, # R Wrist
            11: 23, # L Hip
            12: 24, # R Hip
            13: 25, # L Knee
            14: 26, # R Knee
            15: 27, # L Ankle
            16: 28  # R Ankle
        }

    def process_frame(self, frame: np.ndarray) -> Any:
        # Create MP Image from numpy array (BGR -> RGB auto conversion? No, need to check doc)
        # mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_array) assumes RGB if SRGB is used.
        # OpenCV is BGR.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Run detection
        # Returns PoseLandmarkerResult
        detection_result = self.landmarker.detect(mp_image)
        return detection_result

    def get_keypoints(self, results: Any, frame_shape: Tuple[int, int] = None) -> Optional[np.ndarray]:
        """
        Extract and convert MediaPipe landmarks to COCO format (17, 3).
        Args:
            results: PoseLandmarkerResult object
            frame_shape: (height, width) of the image, required for un-normalization
        """
        # Check if any poses detected
        if results is None:
            return None
            
        if not hasattr(results, 'pose_landmarks') or results.pose_landmarks is None or len(results.pose_landmarks) == 0:
            return None
            
        if frame_shape is None:
            logger.warning("Frame shape not provided for MediaPipe keypoint extraction")
            return None
            
        height, width = frame_shape
        
        # Get first pose (single person mode)
        landmarks = results.pose_landmarks[0]
        
        # Initialize output array (17 keypoints, 3 values: x, y, conf)
        coco_keypoints = np.zeros((17, 3), dtype=np.float32)
        
        for coco_idx, mp_idx in self.mp_to_coco_map.items():
            if mp_idx < len(landmarks):
                lm = landmarks[mp_idx]
                # Convert normalized [0,1] to pixel coordinates
                x = lm.x * width
                y = lm.y * height
                # Use visibility or presence? visibility is usually available.
                conf = lm.visibility if hasattr(lm, "visibility") else lm.presence
                
                coco_keypoints[coco_idx] = [x, y, conf]
            
        return coco_keypoints
