"""
Bundle Adjustment using the annotated COURT points as reference.

Task 3: Improve the camera calibration by performing Bundle Adjustment on the points.

WHY THIS VERSION IS DIFFERENT FROM THE ORIGINAL APPROACH:
The previous implementation used the triangulated 3D player joints as the
reference for Bundle Adjustment. This is circular: those 3D points were
themselves computed using the very calibration we are trying to fix. If the
calibration has a systematic error, the triangulated points inherit that same
error, so "refining" the cameras to match those points cannot remove the real
error - it can only make the cameras agree with each other on a wrong answer.
This is most likely why the original pipeline stayed stuck around ~180px of
reprojection error no matter what.

The annotated court points (img_points.json, one file per camera) give a real,
independent ground truth: a known 3D position on the court (real_corners) and
where that same point was manually clicked in the image (img_corners). This
matches exactly what the project statement asks for: "Use the annotated court
points and their known 3D positions to optimize the camera parameters."

We also found (by testing on the cam_1 data you provided) that simply
re-solving the pose (rvec/tvec) while keeping the old intrinsic matrix K fixed
is NOT enough: the old K itself is broken (fx=17801 vs fy=9573, distortion
coefficients in the hundreds/thousands). This is a classic symptom of
calibrating intrinsics from a single, fully planar set of points (all
real_corners have z=0), which is a degenerate/unstable configuration. So this
version recalibrates K, dist, rvec AND tvec together (this *is* Bundle
Adjustment: jointly minimizing reprojection error over all parameters), but
with a few sane constraints so it doesn't overfit again with so few points.
"""

import json
import os
import numpy as np
import cv2
from typing import Dict, Tuple, Optional

from camera import (
    CAMERA_MAP,
    find_calibration_file,
    find_court_points_file,
    load_calibration,
    load_court_points,
)
from dataset import get_image_sizes_by_view


def calibrate_camera_from_court_points(
    real_corners: np.ndarray,
    img_corners: np.ndarray,
    image_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Recompute K, dist, rvec, tvec for ONE camera using its court correspondences.
    #
    # Constraints used (chosen because we only have ~20 points from ONE view,
    # which is too little to safely estimate a full, unconstrained model):
    #   - CALIB_FIX_ASPECT_RATIO  -> forces fx == fy (a normal camera has square
    #                                pixels; the old K had fx almost 2x fy, which
    #                                is not physically realistic)
    #   - CALIB_ZERO_TANGENT_DIST -> tangential distortion (p1, p2) is negligible
    #                                for modern lenses and just adds noise here
    #   - CALIB_FIX_K3            -> the 3rd order radial term needs a lot of
    #                                points spread across the image to estimate
    #                                reliably; with only ~20 points it's safer
    #                                to disable it than let it explode like before
    width, height = image_size

    # Initial guess: focal length close to image width, principal point at the image center.
    # cv2.calibrateCamera will refine this guess to minimize reprojection error.
    focal_guess = float(max(width, height))
    K_init = np.array([
        [focal_guess, 0.0, width / 2.0],
        [0.0, focal_guess, height / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3

    object_points = [real_corners.astype(np.float32)]
    image_points = [img_corners.astype(np.float32)]

    _rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, K_init, None, flags=flags
    )

    return K, dist, rvecs[0], tvecs[0]


def compute_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray
) -> float:
    # Compute mean reprojection error
    image_points_proj, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    image_points_proj = image_points_proj.reshape(-1, 2)
    errors = np.linalg.norm(image_points_proj - image_points, axis=1)
    return float(np.mean(errors))


def bundle_adjustment_with_court_points(
    calib_root: str,
    annotations_path: str,
    min_points: int = 6
) -> Dict[str, Dict]:
    # Perform Bundle Adjustment using the annotated court points as ground truth.
    # For each camera:
    #   1. Load its original calibration (for the "before" comparison)
    #   2. Load its court correspondences (img_points.json)
    #   3. Recalibrate K, dist, rvec, tvec from those correspondences
    #   4. Compare reprojection error before vs after

    # We need each camera's image resolution to recalibrate; read it once from
    # the COCO annotation file instead of hardcoding it.
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    image_sizes = get_image_sizes_by_view(data)

    results = {}

    for view, cam_name in CAMERA_MAP.items():
        old_calib_path = find_calibration_file(calib_root, cam_name)
        court_points_path = find_court_points_file(calib_root, cam_name)

        if old_calib_path is None:
            print(f"Skipping {view}: no calibration found")
            continue
        if court_points_path is None:
            print(f"Skipping {view}: no court points file (img_points.json) found")
            continue

        old_cam = load_calibration(old_calib_path)
        real_corners, img_corners = load_court_points(court_points_path)

        if len(real_corners) < min_points:
            print(f"Skipping {view}: only {len(real_corners)} court points (need >= {min_points})")
            continue

        # "Before" error: how well the ORIGINAL calibration explains the court points
        old_error = compute_reprojection_error(
            real_corners, img_corners, old_cam.rvec, old_cam.t, old_cam.K, old_cam.dist
        )

        # Image size needed by cv2.calibrateCamera
        image_size = image_sizes.get(view)
        if image_size is None:
            # Fallback if the view isn't found in the annotation file: assume
            # the image is at least as large as the farthest annotated pixel.
            image_size = (
                int(img_corners[:, 0].max()) + 1,
                int(img_corners[:, 1].max()) + 1
            )

        try:
            new_K, new_dist, new_rvec, new_tvec = calibrate_camera_from_court_points(
                real_corners, img_corners, image_size
            )
        except cv2.error as e:
            # If recalibration fails for some reason, keep the old parameters
            # instead of dropping the camera entirely.
            print(f"  calibrateCamera failed for {view} ({e}), keeping old calibration")
            new_K, new_dist = old_cam.K, old_cam.dist
            new_rvec, new_tvec = old_cam.rvec, old_cam.t

        new_error = compute_reprojection_error(
            real_corners, img_corners, new_rvec, new_tvec, new_K, new_dist
        )

        results[view] = {
            'old_error_px': old_error,
            'new_error_px': new_error,
            'improvement_px': old_error - new_error,
            'rvec': np.asarray(new_rvec).flatten().tolist(),
            'tvec': np.asarray(new_tvec).flatten().tolist(),
            'K': np.asarray(new_K).tolist(),
            'dist': np.asarray(new_dist).flatten().tolist(),
            'n_points': len(real_corners),
            'n_inliers': len(real_corners)  # all court points are used directly (no RANSAC filtering here)
        }

    return results


def save_refined_calibrations(ba_results: Dict, output_dir: str):
    # Save refined calibration parameters
    os.makedirs(output_dir, exist_ok=True)
    
    for view, result in ba_results.items():
        cam_name = CAMERA_MAP.get(view)
        if not cam_name:
            continue
        
        cam_output_dir = os.path.join(output_dir, cam_name, 'calib')
        os.makedirs(cam_output_dir, exist_ok=True)
        
        calib_data = {
            'mtx': result['K'],
            'dist': result['dist'],
            'rvecs': result['rvec'],
            'tvecs': result['tvec']
        }
        
        calib_path = os.path.join(cam_output_dir, 'camera_calib_refined.json')
        with open(calib_path, 'w') as f:
            json.dump(calib_data, f, indent=2)


def print_summary(ba_results: Dict):
    # Print summary of improvements
    print("\n" + "="*60)
    print("BUNDLE ADJUSTMENT SUMMARY (Court-points-based)")
    print("="*60)
    
    total_old = 0
    total_new = 0
    count = 0
    
    for view, result in sorted(ba_results.items()):
        old = result['old_error_px']
        new = result['new_error_px']
        imp = result['improvement_px']
        
        total_old += old
        total_new += new
        count += 1
        
        print(f"\n{view}:")
        print(f"  Old error: {old:.2f} px")
        print(f"  New error: {new:.2f} px")
        print(f"  Improvement: {imp:+.2f} px ({imp/old*100:+.1f}%)")
        print(f"  Points used: {result['n_points']}, Inliers: {result['n_inliers']}")
    
    if count > 0:
        print(f"\n{'MEAN':>15}:")
        print(f"  {'Old:':>12}{total_old/count:.2f} px")
        print(f"  {'New:':>12}{total_new/count:.2f} px")
        print(f"  {'Improvement:':>12}{(total_old-total_new)/count:+.2f} px ({(1-total_new/total_old)*100:+.1f}%)")
    
    print("="*60)


if __name__ == '__main__':
    # Paths
    ANNOTATIONS = 'hpe_04.coco/train/_annotations.coco.json'
    CALIB_ROOT = 'material4project-20260603T095832Z-3-001/material4project/3D Pose Estimation Material/camera_data_with_Rvecs/camera_data'
    OUTPUT_DIR = 'output/ba_results'
    REFINED_CALIB_DIR = 'output/refined_calibrations'
    
    # Run court-points-based Bundle Adjustment.
    # NOTE: each camera folder (CALIB_ROOT/cam_X/calib/) must contain its own
    # img_points.json with that camera's real_corners/img_corners, the same
    # way cam_1's was provided.
    ba_results = bundle_adjustment_with_court_points(
        CALIB_ROOT,
        ANNOTATIONS
    )
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'bundle_adjustment_results.json'), 'w') as f:
        json.dump(ba_results, f, indent=2)
    
    # Save refined calibrations
    save_refined_calibrations(ba_results, REFINED_CALIB_DIR)
    
    # Display ONLY the summary
    print_summary(ba_results)
