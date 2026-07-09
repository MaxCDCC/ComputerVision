"""
Diagnostic script for the triangulation pipeline.

This script does NOT triangulate anything. Its goal is to find out WHERE
the problem is coming from, by checking two things independently:

1. Camera sanity check: prints camera position in world coordinates,
   focal length, principal point, distortion. Useful to spot obviously
   wrong values (wrong units, wrong resolution, etc).

2. Epipolar consistency check: for every pair of cameras, it takes real
   annotated keypoints that are known to be the SAME physical 3D point
   (same player, same joint, seen from two different cameras) and checks
   how far that point lands from the epipolar line predicted by the two
   cameras' R/t. This check does not depend on triangulation at all -
   it only depends on whether the two cameras' extrinsics are expressed
   in the same, correct, shared world coordinate system.

   If the calibration is correct, this distance should be small
   (typically a few pixels). If it's large (tens/hundreds of pixels) for
   a given camera pair, that pair's extrinsics are NOT consistent with
   each other, and no triangulation algorithm can fix that - the
   calibration data itself needs to be corrected.

Usage:
    Adjust ANNOTATIONS and CALIB_ROOT below to match your paths, then run:
    python3 diagnose_calibration.py
"""

import json
from collections import defaultdict

import numpy as np

from camera import CameraModel, load_camera_models, undistort_uv
from dataset import build_frames, group_annotations_by_frame, parse_coco_keypoints

# Adjust these paths to your project
ANNOTATIONS = 'hpe_04.coco/train/_annotations.coco.json'
CALIB_ROOT = 'material4project-20260603T095832Z-3-001/material4project/3D Pose Estimation Material/camera_data_with_Rvecs/camera_data'


def camera_center(cam: CameraModel) -> np.ndarray:
    # The optical center of a camera in world coordinates is C = -R^T * t.
    # This is just a sanity-check value: for a court setup, camera centers
    # should be a few meters apart, not absurd values like thousands of
    # units or all clustered at the same point.
    return (-cam.R.T @ cam.t).flatten()


def print_camera_sanity(cameras):
    print("=" * 70)
    print("CAMERA SANITY CHECK")
    print("=" * 70)
    for view, cam in sorted(cameras.items()):
        center = camera_center(cam)
        fx, fy = cam.K[0, 0], cam.K[1, 1]
        cx, cy = cam.K[0, 2], cam.K[1, 2]
        print(f"\n{view} ({cam.name}):")
        print(f"  Camera center (world coords): {center}")
        print(f"  Focal length (fx, fy): ({fx:.1f}, {fy:.1f})")
        print(f"  Principal point (cx, cy): ({cx:.1f}, {cy:.1f})")
        print(f"  Distortion coeffs: {cam.dist.flatten()}")


def fundamental_matrix_from_P(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    # Build the Fundamental matrix F directly from two projection matrices.
    # F encodes the epipolar geometry: for a true correspondence (p1, p2),
    # p2^T * F * p1 should be (close to) 0.
    # Formula (Hartley & Zisserman, "Multiple View Geometry", eq. 9.1):
    #   F = [e2]_x * P2 * pinv(P1)
    # where e2 is the epipole in image 2 (projection of camera 1's center
    # into camera 2), and [e2]_x is its skew-symmetric matrix.

    # Camera center of P1 = right null space of P1 (homogeneous 4-vector)
    _, _, Vt = np.linalg.svd(P1)
    C1 = Vt[-1]
    C1 = C1 / C1[3]

    e2 = P2 @ C1  # epipole of camera 1 seen in image 2
    e2_skew = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0],
    ])

    P1_pinv = np.linalg.pinv(P1)
    F = e2_skew @ P2 @ P1_pinv
    return F


def epipolar_distance(F: np.ndarray, pt1: np.ndarray, pt2: np.ndarray) -> float:
    # Average (symmetric) distance of each point to the other's epipolar line
    p1 = np.array([pt1[0], pt1[1], 1.0])
    p2 = np.array([pt2[0], pt2[1], 1.0])

    line2 = F @ p1
    d2 = abs(p2 @ line2) / np.hypot(line2[0], line2[1])

    line1 = F.T @ p2
    d1 = abs(p1 @ line1) / np.hypot(line1[0], line1[1])

    return (d1 + d2) / 2


def check_epipolar_consistency(cameras, annotation_path: str):
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    image_info, ann_by_image = build_frames(data)
    grouped = group_annotations_by_frame(image_info, ann_by_image)

    # Precompute the Fundamental matrix for every camera pair once
    views = sorted(cameras.keys())
    F_cache = {}
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            v1, v2 = views[i], views[j]
            F_cache[(v1, v2)] = fundamental_matrix_from_P(cameras[v1].P, cameras[v2].P)

    pair_errors = defaultdict(list)

    for frame_idx, players in grouped.items():
        for category_id, view_list in players.items():
            # Collect keypoints per view for this player in this frame
            view_kps = {}
            for view, ann in view_list:
                if view not in cameras:
                    continue
                view_kps[view] = parse_coco_keypoints(ann['keypoints'])

            present_views = sorted(view_kps.keys())
            for i in range(len(present_views)):
                for j in range(i + 1, len(present_views)):
                    v1, v2 = present_views[i], present_views[j]
                    F = F_cache.get((v1, v2))
                    if F is None:
                        continue

                    kps1, kps2 = view_kps[v1], view_kps[v2]
                    for k in range(min(len(kps1), len(kps2))):
                        x1, y1, vis1 = kps1[k]
                        x2, y2, vis2 = kps2[k]
                        if vis1 <= 0 or vis2 <= 0:
                            continue

                        # Undistort points before the epipolar check, since F
                        # (built from P = K[R|t]) assumes a pinhole model
                        u1 = undistort_uv(cameras[v1], np.array([x1, y1]))
                        u2 = undistort_uv(cameras[v2], np.array([x2, y2]))

                        d = epipolar_distance(F, u1, u2)
                        pair_errors[(v1, v2)].append(d)

    print("\n" + "=" * 70)
    print("EPIPOLAR CONSISTENCY CHECK (does not depend on triangulation)")
    print("=" * 70)
    print("Expected if calibration is correct: a few pixels.")
    print("Large values (tens / hundreds of px) mean that pair of cameras'")
    print("R/t are NOT expressed in the same coordinate system.\n")

    for (v1, v2), errs in sorted(pair_errors.items()):
        print(f"{v1} <-> {v2}: mean={np.mean(errs):7.1f}px  "
              f"median={np.median(errs):7.1f}px  n={len(errs)}")


if __name__ == '__main__':
    cameras = load_camera_models(CALIB_ROOT)
    print_camera_sanity(cameras)
    check_epipolar_consistency(cameras, ANNOTATIONS)