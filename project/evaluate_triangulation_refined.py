"""
Evaluate triangulation accuracy with improved camera parameters.

Task 3a: Evaluate the accuracy of the triangulation with the improved camera parameters.

This script runs triangulation using the refined calibrations from bundle adjustment
and compares the results with the original calibration.
"""

import json
import os
from typing import Dict

from triangulator import process_dataset


def load_original_results():
    # Load original triangulation results
    original_path = 'output/triangulation_results.json'
    if os.path.exists(original_path):
        with open(original_path, 'r') as f:
            return json.load(f)
    return None


def run_triangulation_with_calib(calib_root: str, output_path: str, annotations_path: str):
    # Run triangulation with a given calibration set
    results = process_dataset(annotations_path, calib_root)
    
    # Save results
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def compare_results(original: Dict, refined: Dict) -> Dict:
    # Compare results before and after bundle adjustment
    comparison = {
        'original': {
            'frames_processed': original['summary']['frames_processed'],
            'triangulated_skeletons': original['summary']['triangulated_skeletons'],
            'points_triangulated': original['summary']['points_triangulated'],
            'mean_reprojection_error_px': original['summary']['mean_reprojection_error_px']
        },
        'refined': {
            'frames_processed': refined['summary']['frames_processed'],
            'triangulated_skeletons': refined['summary']['triangulated_skeletons'],
            'points_triangulated': refined['summary']['points_triangulated'],
            'mean_reprojection_error_px': refined['summary']['mean_reprojection_error_px']
        }
    }
    
    # Calculate improvements
    if comparison['original']['mean_reprojection_error_px'] is not None and \
       comparison['refined']['mean_reprojection_error_px'] is not None:
        old_err = comparison['original']['mean_reprojection_error_px']
        new_err = comparison['refined']['mean_reprojection_error_px']
        comparison['improvement'] = {
            'absolute_px': old_err - new_err,
            'relative_pct': (old_err - new_err) / old_err * 100
        }
    else:
        comparison['improvement'] = None
    
    return comparison


def print_comparison(comparison: Dict):
    # Print formatted comparison of results
    print("\n" + "="*70)
    print("COMPARISON: Triangulation before and after Bundle Adjustment")
    print("="*70)
    
    orig = comparison['original']
    ref = comparison['refined']
    
    print(f"\n{'Metric':<35} {'Original':>15} {'Refined':>15} {'Change':>15}")
    print("-"*70)
    
    # Frames processed
    print(f"{'Frames processed':<35} {orig['frames_processed']:>15} {ref['frames_processed']:>15}")
    
    # Skeletons
    print(f"{'Triangulated skeletons':<35} {orig['triangulated_skeletons']:>15} {ref['triangulated_skeletons']:>15}")
    
    # Points
    print(f"{'Points triangulated':<35} {orig['points_triangulated']:>15} {ref['points_triangulated']:>15}")
    
    # Mean reprojection error
    orig_err = orig['mean_reprojection_error_px']
    ref_err = ref['mean_reprojection_error_px']
    
    if orig_err is not None and ref_err is not None:
        print(f"{'Mean reprojection error (px)':<35} {orig_err:>15.3f} {ref_err:>15.3f}")
        
        if comparison.get('improvement'):
            imp = comparison['improvement']
            change_str = f"{imp['absolute_px']:+.3f} ({imp['relative_pct']:+.1f}%)"
            print(f"{'Improvement':<35} {'':<15} {'':<15} {change_str:>15}")
    else:
        print(f"{'Mean reprojection error (px)':<35} {'N/A':>15} {'N/A':>15}")
    
    print("="*70)


def setup_refined_calibration_directory():
    # Return the directory for refined calibrations
    return 'output/refined_calibrations'


if __name__ == '__main__':
    # Paths
    ANNOTATIONS = '../hpe_04.coco/train/_annotations.coco.json'
    ORIGINAL_CALIB = '../material4project-20260603T095832Z-3-001/material4project/3D Pose Estimation Material/camera_data_with_Rvecs/camera_data'
    
    # Use refined calibrations (player-based)
    REFINED_CALIB_DIR = 'output/refined_calibrations'
    
    # Output paths
    REFINED_OUTPUT = 'output/triangulation_results_refined.json'
    COMPARISON_OUTPUT = 'output/comparison_results.json'
    
    # Load original results
    original_results = load_original_results()
    if original_results is None:
        original_results = run_triangulation_with_calib(
            ORIGINAL_CALIB, 
            'output/triangulation_results_original.json',
            ANNOTATIONS
        )
    
    # Run triangulation with improved calibration
    refined_results = run_triangulation_with_calib(
        REFINED_CALIB_DIR,
        REFINED_OUTPUT,
        ANNOTATIONS
    )
    
    # Compare results
    comparison = compare_results(original_results, refined_results)
    
    # Save comparison
    os.makedirs('output', exist_ok=True)
    with open(COMPARISON_OUTPUT, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Display ONLY the comparison table
    print_comparison(comparison)
