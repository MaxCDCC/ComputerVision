
import json
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from camera import CameraModel, load_camera_models
from dataset import build_frames, group_annotations_by_frame, parse_coco_keypoints


def triangulate_point(observations: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Optional[np.ndarray]:

    # At least two camera views are required to reconstruct a 3D point
    if len(observations) < 2:
        return None

    A = []

    # Build the linear system used by DLT triangulation
    for P, uv in observations:
        u, v = uv

        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])

    A = np.vstack(A)

    # Solve AX = 0 using Singular Value Decomposition (SVD)
    _, _, Vt = np.linalg.svd(A)

    # The last singular vector gives the 3D point in homogeneous coordinates
    X = Vt[-1]

    # Avoid division by zero
    if abs(X[3]) < 1e-8:
        return None

    # Convert homogeneous coordinates to 3D coordinates
    return X[:3] / X[3]


def compute_reprojection_error(point_3d: np.ndarray, camera: CameraModel, uv: np.ndarray) -> float:
    # OpenCV expects the point in shape (1,1,3)
    point_3d = point_3d.reshape(1, 1, 3)

    # Project the reconstructed 3D point back into the image
    projected, _ = cv2.projectPoints(
        point_3d,
        camera.rvec,
        camera.t,
        camera.K,
        camera.dist
    )

    projected = projected.reshape(2)

    # Compute the distance between prediction and annotation
    return float(np.linalg.norm(projected - uv))


def process_dataset(annotation_path: str, calib_root: str) -> dict:
    # Load the COCO annotation file
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    # Organize images and annotations
    image_info, ann_by_image = build_frames(data)

    # Group players by frame
    grouped = group_annotations_by_frame(image_info, ann_by_image)

    # Load all camera calibrations
    cameras = load_camera_models(calib_root)

    results = {
        'frames': {},
        'summary': {
            'triangulated_skeletons': 0,
            'points_triangulated': 0,
            'mean_reprojection_error_px': None,
            'frames_processed': len(grouped),
        },
    }

    total_errors: List[float] = []
    total_points = 0

    # Process each frame independently
    for frame_idx, players in sorted(grouped.items()):
        frame_result = {'players': {}}

        # Process each player in the frame
        for category_id, view_list in sorted(players.items()):
            player_result = {
                'views': {},
                'triangulated_points': []
            }

            # Collect all camera observations of this player
            for view, ann in view_list:
                kps = parse_coco_keypoints(ann['keypoints'])

                player_result['views'][view] = {
                    'image_id': ann['image_id'],
                    'keypoints': kps,
                }

            if not player_result['views']:
                continue

            valid_points = []

            # Number of joints in the skeleton 
            n_points = max(len(info['keypoints']) for info in player_result['views'].values())

            # Reconstruct each joint separately
            for kp_idx in range(n_points):
                observations = []
                reference_uv = []

                # Search for this joint in all available camera views
                for view, info in player_result['views'].items():

                    if kp_idx >= len(info['keypoints']):
                        continue
                    x, y, v = info['keypoints'][kp_idx]
                    # Ignore invisible or missing joints
                    if v <= 0:
                        continue

                    camera = cameras.get(view)
                    if camera is None:
                        continue

                    # Store projection matrix and image coordinates
                    observations.append((camera.P, np.array([x, y], dtype=np.float64)))
                    reference_uv.append((view, np.array([x, y], dtype=np.float64)))

                # Triangulation requires at least two views
                if len(observations) < 2:
                    continue

                point_3d = triangulate_point(observations)

                if point_3d is None:
                    continue

                point_error = []

                # Evaluate the reconstruction by reprojection
                for view, uv in reference_uv:
                    camera = cameras[view]
                    err = compute_reprojection_error(point_3d, camera, uv)
                    point_error.append(err)
                    total_errors.append(err)
                    total_points += 1

                mean_err = (
                    float(np.mean(point_error))
                    if point_error else None
                )

                valid_points.append({
                    'keypoint_index': kp_idx,
                    'point_3d': point_3d.tolist(),
                    'mean_reprojection_error_px': mean_err,

                    # Cameras used for triangulation
                    'views_used': [view for view, _ in reference_uv],
                })

            # Store the reconstructed skeleton
            if valid_points:
                player_result['triangulated_points'] = valid_points
                frame_result['players'][str(category_id)] = player_result
                results['summary']['triangulated_skeletons'] += 1
                results['summary']['points_triangulated'] += len(valid_points)

        # Save the frame only if at least one player was reconstructed
        if frame_result['players']:
            results['frames'][str(frame_idx)] = frame_result

    # Compute the global reprojection error
    if total_points > 0:
        results['summary']['mean_reprojection_error_px'] = float(np.mean(total_errors))

    return results