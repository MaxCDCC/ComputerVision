import json
import os

from triangulator import process_dataset


# Project paths
ANNOTATIONS = 'hpe_04.coco/train/_annotations.coco.json'
CALIB_ROOT = 'material4project-20260603T095832Z-3-001/material4project/3D Pose Estimation Material/camera_data_with_Rvecs/camera_data'
OUTPUT = 'project/output/triangulation_results.json'


# Run triangulation and reprojection
results = process_dataset(
    ANNOTATIONS,
    CALIB_ROOT
)

# Create output folder if it does not exist
output_dir = os.path.dirname(OUTPUT)

if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# Save results to JSON
with open(OUTPUT, 'w') as f:
    json.dump(results, f, indent=2)

# Display a short summary
print('Done.')
print(f"Frames processed: {results['summary']['frames_processed']}")
print(f"Triangulated skeletons: {results['summary']['triangulated_skeletons']}")
print(f"Points triangulated: {results['summary']['points_triangulated']}")

mean_err = results['summary']['mean_reprojection_error_px']

if mean_err is not None:
    print(f"Mean reprojection error: {mean_err:.3f} px")
else:
    print('Mean reprojection error: None')