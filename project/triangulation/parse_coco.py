import numpy as np
import cv2
"""
Parse COCO keypoints annotations for multi-camera triangulation.
This script loads a COCO JSON file containing images from multiple cameras.
It allows filtering by camera (e.g. 'out1', 'out5', etc.) and extracts 2D keypoints
for a given joint, player, frame, and camera.
"""
import json
import os
from typing import List, Dict

# --- Parameters ---
script_dir = os.path.dirname(os.path.abspath(__file__))
coco_ann_path = os.path.abspath(os.path.join(script_dir, '../../ComputerVision/hpe_merged_all.coco/train/_annotations.coco.json'))  # merged multi-cam
camera1_prefix = 'out1'  # e.g. 'out1'
camera2_prefix = 'out5'  # e.g. 'out5'
joint_idx = 0  # e.g. 0 = nose, 1 = left eye, etc.
player_id = 1  # change as needed

# --- Load COCO annotations ---
def load_coco(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)

def get_camera_from_filename(filename: str) -> str:
    # Extract camera prefix from filename, e.g. 'out1' from 'out1_frame_0004_...png'
    return filename.split('_')[0]

def get_keypoints_for_frame(coco, frame_name, joint_idx, player_id):
    # Find image id for frame
    # --- Main ---
    img_id = None
    for img in coco['images']:
        if img['file_name'] == frame_name:
            img_id = img['id']
            break
    if img_id is None:
        return None
    # Find annotation for this image and player
    for ann in coco['annotations']:
        if ann['image_id'] == img_id and ann.get('category_id', 1) == player_id:
            kps = ann['keypoints']
            x = kps[3*joint_idx]
            y = kps[3*joint_idx+1]
            v = kps[3*joint_idx+2]
            return (x, y, v)
    return None

# --- Main ---
if __name__ == "__main__":
    coco = load_coco(coco_ann_path)
    # Filter images by camera
    # Filter images by camera
    images_cam1 = [img for img in coco['images'] if get_camera_from_filename(img['file_name']) == camera1_prefix]
    images_cam2 = [img for img in coco['images'] if get_camera_from_filename(img['file_name']) == camera2_prefix]
    print(f"Found {len(images_cam1)} images for camera {camera1_prefix}.")
    print(f"Found {len(images_cam2)} images for camera {camera2_prefix}.")

    # Find synchronized frames (same index/number)
    def get_frame_index(filename):
        # Example: out1_frame_0004_png.rf.76JZJqPvK0ZADQCSC9SC.png
        # Extract '0004' as frame index
        parts = filename.split('_')
        for part in parts:
            if part.startswith('frame'):
                return part.replace('frame', '')
        return None

    frames_cam1 = {get_frame_index(img['file_name']): img['file_name'] for img in images_cam1}
    frames_cam2 = {get_frame_index(img['file_name']): img['file_name'] for img in images_cam2}
    common_indices = sorted(set(frames_cam1.keys()) & set(frames_cam2.keys()))
    print(f"Found {len(common_indices)} synchronized frames between {camera1_prefix} and {camera2_prefix}.")

    # For each synchronized frame, triangulate the full 3D skeleton
    num_joints = 17  # COCO format (change if needed)
    def get_all_keypoints(coco, frame_name, player_id, num_joints):
        img_id = None
        for img in coco['images']:
            if img['file_name'] == frame_name:
                img_id = img['id']
                break
        if img_id is None:
            return None
        for ann in coco['annotations']:
            if ann['image_id'] == img_id and ann.get('category_id', 1) == player_id:
                kps = ann['keypoints']
                # Return as list of (x, y, v)
                return [(kps[3*j], kps[3*j+1], kps[3*j+2]) for j in range(num_joints)]
        return None

    # --- Camera calibration (to be replaced with real values) ---
    # Example: P1 and P2 are 3x4 projection matrices for each camera
    # Replace these with your real calibration data!
    P1 = np.array([
        [1000, 0, 960, 0],
        [0, 1000, 540, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32)
    P2 = np.array([
        [1000, 0, 960, -100],
        [0, 1000, 540, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32)

    for idx in common_indices[:5]:  # just a few for demo
        frame1 = frames_cam1[idx]
        frame2 = frames_cam2[idx]
        print(f"\nFrame index: {idx}")
        print(f"  {camera1_prefix}: {frame1}")
        print(f"  {camera2_prefix}: {frame2}")

        kps1 = get_all_keypoints(coco, frame1, player_id, num_joints)
        kps2 = get_all_keypoints(coco, frame2, player_id, num_joints)

        skeleton3D = []
        for j in range(num_joints):
            kp1 = kps1[j] if kps1 else (0,0,0)
            kp2 = kps2[j] if kps2 else (0,0,0)
            if kp1[2] > 0 and kp2[2] > 0:
                pt1 = np.array([[kp1[0]], [kp1[1]]], dtype=np.float32)
                pt2 = np.array([[kp2[0]], [kp2[1]]], dtype=np.float32)
                pts1 = pt1.reshape(2, 1)
                pts2 = pt2.reshape(2, 1)
                point4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
                point3D = (point4D[:3] / point4D[3]).ravel()
                skeleton3D.append(point3D)
            else:
                skeleton3D.append(None)

        print("  3D skeleton:")
        for j, pt3d in enumerate(skeleton3D):
            if pt3d is not None:
                print(f"    Joint {j}: {pt3d}")
            else:
                print(f"    Joint {j}: Not visible in both cameras")
        print(f"Frame index: {idx}")
        print(f"  {camera1_prefix}: {frame1}  Keypoint: {kp1}")
        print(f"  {camera2_prefix}: {frame2}  Keypoint: {kp2}")
