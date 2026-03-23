"""
Camera Calibration Utility

Provides tools for calibrating wide-angle and fisheye cameras
to correct lens distortion for accurate pose estimation.

Usage:
    1. Print a checkerboard pattern (9x6 inner corners recommended)
    2. Run: python -m engine.camera_calibration --device 0 --output calibration_side.npz
    3. Show checkerboard at various angles (capture 20+ images)
    4. Press 'c' to capture, 'q' to finish and calibrate
"""

import os
import argparse
import cv2
import numpy as np
from typing import List, Tuple, Optional
from loguru import logger


class CameraCalibrator:
    """Calibrate camera for distortion correction."""
    
    def __init__(
        self,
        checkerboard_size: Tuple[int, int] = (9, 6),
        square_size_mm: float = 25.0,
        is_fisheye: bool = False,
    ):
        """
        Initialize calibrator.
        
        Args:
            checkerboard_size: Number of inner corners (width, height)
            square_size_mm: Size of each square in millimeters
            is_fisheye: Whether to use fisheye calibration model
        """
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        self.is_fisheye = is_fisheye
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0), ...
        self.objp = np.zeros(
            (checkerboard_size[0] * checkerboard_size[1], 3), 
            np.float32
        )
        self.objp[:, :2] = np.mgrid[
            0:checkerboard_size[0], 
            0:checkerboard_size[1]
        ].T.reshape(-1, 2)
        self.objp *= square_size_mm
        
        # Storage for calibration data
        self.obj_points: List[np.ndarray] = []
        self.img_points: List[np.ndarray] = []
        self.image_size: Optional[Tuple[int, int]] = None
        
        # Results
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.rvecs: Optional[List[np.ndarray]] = None
        self.tvecs: Optional[List[np.ndarray]] = None
        self.reprojection_error: float = 0.0
    
    def find_corners(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Find checkerboard corners in image.
        
        Returns:
            Tuple of (success, corners)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, flags)
        
        if ret:
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        return ret, corners
    
    def add_image(self, image: np.ndarray) -> bool:
        """
        Add calibration image.
        
        Returns:
            True if checkerboard was found and added
        """
        if self.image_size is None:
            self.image_size = (image.shape[1], image.shape[0])
        
        ret, corners = self.find_corners(image)
        
        if ret:
            self.obj_points.append(self.objp)
            self.img_points.append(corners)
            logger.info(f"Added calibration image ({len(self.img_points)} total)")
            return True
        
        return False
    
    def calibrate(self) -> bool:
        """
        Run calibration using collected images.
        
        Returns:
            True if calibration succeeded
        """
        if len(self.obj_points) < 10:
            logger.warning(f"Only {len(self.obj_points)} images - need at least 10 for good calibration")
        
        if len(self.obj_points) < 3:
            logger.error("Need at least 3 calibration images")
            return False
        
        logger.info(f"Calibrating with {len(self.obj_points)} images...")
        
        if self.is_fisheye:
            return self._calibrate_fisheye()
        else:
            return self._calibrate_standard()
    
    def _calibrate_standard(self) -> bool:
        """Standard camera calibration."""
        try:
            ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
                self.obj_points,
                self.img_points,
                self.image_size,
                None,
                None,
                flags=cv2.CALIB_RATIONAL_MODEL
            )
            
            self.reprojection_error = ret
            logger.success(f"Calibration complete. Reprojection error: {ret:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return False
    
    def _calibrate_fisheye(self) -> bool:
        """Fisheye camera calibration."""
        try:
            # Fisheye calibration flags
            flags = (
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                cv2.fisheye.CALIB_CHECK_COND +
                cv2.fisheye.CALIB_FIX_SKEW
            )
            
            # Reshape for fisheye calibration
            obj_pts = [pts.reshape(1, -1, 3).astype(np.float64) for pts in self.obj_points]
            img_pts = [pts.reshape(1, -1, 2).astype(np.float64) for pts in self.img_points]
            
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            
            ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.fisheye.calibrate(
                obj_pts,
                img_pts,
                self.image_size,
                K, D,
                flags=flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
            
            self.reprojection_error = ret
            logger.success(f"Fisheye calibration complete. Reprojection error: {ret:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Fisheye calibration failed: {e}")
            return False
    
    def save(self, filepath: str) -> None:
        """Save calibration to file."""
        if self.camera_matrix is None or self.dist_coeffs is None:
            logger.error("No calibration data to save")
            return
        
        np.savez(
            filepath,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            image_size=np.array(self.image_size),
            is_fisheye=np.array([self.is_fisheye]),
            reprojection_error=np.array([self.reprojection_error]),
        )
        logger.success(f"Calibration saved to {filepath}")
    
    def load(self, filepath: str) -> bool:
        """Load calibration from file."""
        try:
            data = np.load(filepath)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.image_size = tuple(data['image_size'])
            self.is_fisheye = bool(data['is_fisheye'][0])
            self.reprojection_error = float(data['reprojection_error'][0])
            logger.info(f"Loaded calibration from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
    
    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Apply distortion correction to image."""
        if self.camera_matrix is None or self.dist_coeffs is None:
            return image
        
        if self.is_fisheye:
            h, w = image.shape[:2]
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.camera_matrix, self.dist_coeffs, (w, h), np.eye(3)
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, np.eye(3), new_K, (w, h), cv2.CV_16SC2
            )
            return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        else:
            return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
    
    def draw_corners(self, image: np.ndarray) -> np.ndarray:
        """Draw detected corners on image for visualization."""
        ret, corners = self.find_corners(image)
        output = image.copy()
        
        if ret:
            cv2.drawChessboardCorners(output, self.checkerboard_size, corners, ret)
        
        return output


def run_interactive_calibration(
    device_id: int = 0,
    output_file: str = "calibration.npz",
    checkerboard_size: Tuple[int, int] = (9, 6),
    is_fisheye: bool = False,
    width: int = 1280,
    height: int = 720,
):
    """
    Run interactive camera calibration.
    
    Controls:
        'c' - Capture image (if checkerboard detected)
        'q' - Quit and run calibration
        'r' - Reset (clear all captured images)
    """
    logger.info(f"Starting calibration for device {device_id}")
    logger.info(f"Checkerboard size: {checkerboard_size}")
    logger.info(f"Controls: 'c' to capture, 'q' to finish, 'r' to reset")
    
    cap = cv2.VideoCapture(device_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        logger.error(f"Failed to open camera {device_id}")
        return
    
    calibrator = CameraCalibrator(
        checkerboard_size=checkerboard_size,
        is_fisheye=is_fisheye
    )
    
    window_name = "Camera Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Try to find corners
        display = frame.copy()
        found, corners = calibrator.find_corners(frame)
        
        if found:
            cv2.drawChessboardCorners(display, checkerboard_size, corners, found)
            cv2.putText(display, "CHECKERBOARD FOUND - Press 'c' to capture",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No checkerboard detected",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Status
        cv2.putText(display, f"Captured: {len(calibrator.img_points)} images",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "Press 'q' to finish calibration",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') and found:
            if calibrator.add_image(frame):
                logger.info(f"Image captured ({len(calibrator.img_points)} total)")
        
        elif key == ord('r'):
            calibrator.obj_points.clear()
            calibrator.img_points.clear()
            logger.info("Reset - all images cleared")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Run calibration if we have images
    if len(calibrator.img_points) >= 3:
        if calibrator.calibrate():
            calibrator.save(output_file)
            logger.success(f"Calibration saved to {output_file}")
            
            # Print results
            print("\n" + "="*50)
            print("CALIBRATION RESULTS")
            print("="*50)
            print(f"Images used: {len(calibrator.img_points)}")
            print(f"Reprojection error: {calibrator.reprojection_error:.4f} pixels")
            print(f"\nCamera Matrix:\n{calibrator.camera_matrix}")
            print(f"\nDistortion Coefficients:\n{calibrator.dist_coeffs.flatten()}")
            print("="*50)
    else:
        logger.warning("Not enough images for calibration")


def main():
    parser = argparse.ArgumentParser(description="Camera Calibration Tool")
    parser.add_argument("--device", type=int, default=0, help="Camera device ID")
    parser.add_argument("--output", type=str, default="calibration.npz", help="Output file")
    parser.add_argument("--rows", type=int, default=6, help="Checkerboard rows (inner corners)")
    parser.add_argument("--cols", type=int, default=9, help="Checkerboard columns (inner corners)")
    parser.add_argument("--fisheye", action="store_true", help="Use fisheye calibration model")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    
    args = parser.parse_args()
    
    run_interactive_calibration(
        device_id=args.device,
        output_file=args.output,
        checkerboard_size=(args.cols, args.rows),
        is_fisheye=args.fisheye,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
