import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Link each image prefix (out1, out2, ...) to its camera calibration folder.
CAMERA_MAP = {
    'out1': 'cam_1',
    'out2': 'cam_2',
    'out3': 'cam_3',
    'out4': 'cam_4',
    'out5': 'cam_5',
    'out7': 'cam_7',
}


@dataclass
class CameraModel:
    name: str
    K: np.ndarray       # intrinsic matrix
    dist: np.ndarray    # distortion coefficients
    R: np.ndarray       # rotation matrix
    rvec: np.ndarray    # rotation vector (OpenCV format)
    t: np.ndarray       # translation vector
    P: np.ndarray       # projection matrix K[R|t]


def load_calibration(calib_path: str) -> CameraModel:
    # Read calibration parameters from a JSON file
    with open(calib_path, 'r') as f:
        calib = json.load(f)

    K = np.array(calib['mtx'], dtype=np.float64)        # Camera intrinsic parameters
    dist = np.array(calib['dist'], dtype=np.float64)    # Lens distortion coefficients

    # Camera pose (rotation and translation)
    rvec = np.array(calib['rvecs'], dtype=np.float64).reshape(3, 1)
    tvec = np.array(calib['tvecs'], dtype=np.float64).reshape(3, 1)

    R, _ = cv2.Rodrigues(rvec)      # Convert rotation vector into a rotation matrix
    P = K @ np.hstack([R, tvec])    # Projection matrix used for triangulation

    name = os.path.basename(os.path.dirname(os.path.dirname(calib_path)))   # Camera name extracted from the folder structure
    return CameraModel(name=name, K=K, dist=dist, R=R, rvec=rvec, t=tvec, P=P)


def find_calibration_file(calib_root: str, cam_name: str) -> Optional[str]:
    # Try the refined calibration first, then standard files
    calib_dir = os.path.join(calib_root, cam_name, 'calib')
    
    # Priority order: refined -> standard -> fallback
    files_to_try = [
        os.path.join(calib_dir, 'camera_calib_refined.json'),
        os.path.join(calib_dir, 'camera_calib.json'),
        os.path.join(calib_dir, 'camera_calib_real.json'),
    ]
    
    for calib_file in files_to_try:
        if os.path.exists(calib_file):
            return calib_file
    return None


def find_court_points_file(calib_root: str, cam_name: str) -> Optional[str]:
    # Locate the "img_points.json" file for one camera: it stores the known
    # 3D positions of the court corners (real_corners) and where they fall
    # in that camera's image (img_corners). These are GROUND TRUTH
    # correspondences (independent of any triangulation), so they are the
    # correct reference to use for Bundle Adjustment (Task 3).
    calib_dir = os.path.join(calib_root, cam_name, 'calib')
    court_file = os.path.join(calib_dir, 'img_points.json')

    if os.path.exists(court_file):
        return court_file
    return None


def load_court_points(court_points_path: str) -> Tuple[np.ndarray, np.ndarray]:
    # Read the court correspondences for one camera.
    # real_corners: known 3D world position of each court marking (x, y, z)
    # img_corners: where that same marking was clicked/annotated in the image (u, v)
    with open(court_points_path, 'r') as f:
        data = json.load(f)

    real_corners = np.array(data['real_corners'], dtype=np.float64)
    img_corners = np.array(data['img_corners'], dtype=np.float64)
    return real_corners, img_corners


def load_camera_models(calib_root: str) -> Dict[str, CameraModel]:
    # Load all camera calibrations defined in CAMERA_MAP.
    # Returns a dictionary indexed by view name (out1, out2, ...)
    cameras: Dict[str, CameraModel] = {}
    for view, cam_name in CAMERA_MAP.items():
        calib_path = find_calibration_file(calib_root, cam_name)
        if calib_path is None:
            raise FileNotFoundError(f'calibration file not found for {view} -> {cam_name}')
        cameras[view] = load_calibration(calib_path)
    return cameras


def undistort_uv(camera: CameraModel, uv: np.ndarray) -> np.ndarray:
    # Remove lens distortion from a 2D image point
    # The input point comes from the original image
    # The returned point corresponds to its corrected position
    uv = uv.reshape(1, 1, 2).astype(np.float64)
    undistorted = cv2.undistortPoints(uv, camera.K, camera.dist, P=camera.K)
    return undistorted.reshape(2)
