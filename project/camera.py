import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

# map image prefix to the matching calibration folder.
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
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    rvec: np.ndarray
    t: np.ndarray
    P: np.ndarray


def load_calibration(calib_path: str) -> CameraModel:
    # read calibration parameters from a JSON file
    with open(calib_path, 'r') as f:
        calib = json.load(f)

    K = np.array(calib['mtx'], dtype=np.float64)
    dist = np.array(calib['dist'], dtype=np.float64)
    rvec = np.array(calib['rvecs'], dtype=np.float64).reshape(3, 1)
    tvec = np.array(calib['tvecs'], dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    P = K @ np.hstack([R, tvec])
    name = os.path.basename(os.path.dirname(os.path.dirname(calib_path)))
    return CameraModel(name=name, K=K, dist=dist, R=R, rvec=rvec, t=tvec, P=P)


def find_calibration_file(calib_root: str, cam_name: str) -> Optional[str]:
    # try the standard file first, otherwise use the fallback file
    preferred = os.path.join(calib_root, cam_name, 'calib', 'camera_calib.json')
    fallback = os.path.join(calib_root, cam_name, 'calib', 'camera_calib_real.json')
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(fallback):
        return fallback
    return None


def load_camera_models(calib_root: str) -> Dict[str, CameraModel]:
    # load all cameras defined in CAMERA_MAP
    cameras: Dict[str, CameraModel] = {}
    for view, cam_name in CAMERA_MAP.items():
        calib_path = find_calibration_file(calib_root, cam_name)
        if calib_path is None:
            raise FileNotFoundError(f'calibration file not found for {view} -> {cam_name}')
        cameras[view] = load_calibration(calib_path)
    return cameras


def undistort_uv(camera: CameraModel, uv: np.ndarray) -> np.ndarray:
    # remove distortion from a 2D point in the image
    uv = uv.reshape(1, 1, 2).astype(np.float64)
    undistorted = cv2.undistortPoints(uv, camera.K, camera.dist, P=camera.K)
    return undistorted.reshape(2)
