# HPE Project — 3D Triangulation and Reprojection

This folder contains the triangulation and reprojection evaluation script for the dataset annotated in `hpe_04.coco`.

## Objective

- compute the 3D position of a player from synchronized cameras
- reproject the 3D points into each view
- compare the reprojection with the 2D annotations

## Usage

1. Verify that the dataset and calibration files exist:
   - `hpe_04.coco/train/_annotations.coco.json`
   - `material4project-20260603T095832Z-3-001/material4project/3D Pose Estimation Material/camera_data_with_Rvecs/camera_data`

2. Run the script from the workspace root:

```bash
python project/triangulation.py \
  --annotations hpe_04.coco/train/_annotations.coco.json \
  --calib-root material4project-20260603T095832Z-3-001/material4project/3D\ Pose\ Estimation\ Material/camera_data_with_Rvecs/camera_data \
  --output project/triangulation_results.json
```

3. Results

- `project/triangulation_results.json` contains the triangulated 3D positions and the reprojection errors.

## Notes

- the script uses the views `out1`, `out2`, `out3`, `out4`, `out5`, `out7`
- the mapping between view and calibration is defined in `triangulation.py`
- the script triangulates a point only if that same point is visible in at least two cameras
