import argparse
import json
import os

from triangulator import process_dataset

# default paths for the project.
# you can change these here if needed, without using CLI options.
DEFAULT_ANNOTATIONS = 'hpe_04.coco/train/_annotations.coco.json'
DEFAULT_CALIB_ROOT = 'material4project-20260603T095832Z-3-001/material4project/3D Pose Estimation Material/camera_data_with_Rvecs/camera_data'
DEFAULT_OUTPUT = 'project/triangulation_results.json'


def main() -> None:
    # read CLI arguments with simple defaults
    parser = argparse.ArgumentParser(description='Triangulation and reprojection for HPE dataset')
    parser.add_argument('--annotations', type=str, default=DEFAULT_ANNOTATIONS, help='path to _annotations.coco.json')
    parser.add_argument('--calib-root', type=str, default=DEFAULT_CALIB_ROOT, help='root folder for camera calibrations')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT, help='output JSON file')
    args = parser.parse_args()

    # process the dataset and compute results
    results = process_dataset(args.annotations, args.calib_root)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # print a simple summary
    print('Done.')
    print(f"frames processed: {results['summary']['frames_processed']}")
    print(f"triangulated skeletons: {results['summary']['triangulated_skeletons']}")
    print(f"points triangulated: {results['summary']['points_triangulated']}")
    mean_err = results['summary']['mean_reprojection_error_px']
    if mean_err is not None:
        print(f"mean reprojection error: {mean_err:.3f} px")
    else:
        print('mean reprojection error: None')


if __name__ == '__main__':
    main()
