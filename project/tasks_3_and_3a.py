#!/usr/bin/env python3
"""
Main script for running tasks 3 and 3a:

3. Improve the camera calibration by performing Bundle Adjustment on the points
3a. Evaluate the accuracy of the triangulation with the improved camera parameters

This script runs:
1. Bundle Adjustment using triangulated player points
2. Triangulation evaluation with improved parameters
3. Comparison of results before and after

Usage:
    python3 tasks_3_and_3a.py
"""

import os
import sys
import json
import subprocess


def run_bundle_adjustment():
    # Run Bundle Adjustment
    print("="*70)
    print("TASK 3: Bundle Adjustment")
    print("="*70)
    print("\nRunning bundle_adjustment.py...")
    result = subprocess.run(
        [sys.executable, 'bundle_adjustment.py'],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
        return False
    return True


def run_evaluation():
    # Run evaluation with improved calibration
    print("\n" + "="*70)
    print("TASK 3a: Evaluation with improved calibration")
    print("="*70)
    print("\nRunning evaluate_triangulation_refined.py...")
    result = subprocess.run(
        [sys.executable, 'evaluate_triangulation_refined.py'],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
        return False
    return True


def print_final_summary():
    # Print final summary of results
    print("\n" + "="*70)
    print("FINAL SUMMARY: Tasks 3 and 3a")
    print("="*70)
    
    # Load results
    comparison_path = 'output/comparison_results.json'
    ba_results_path = 'output/ba_results/bundle_adjustment_results.json'
    
    if os.path.exists(comparison_path):
        with open(comparison_path) as f:
            comparison = json.load(f)
        
        orig = comparison['original']
        ref = comparison['refined']
        
        print("\nTriangulation comparison:")
        print(f"  Original mean error:     {orig['mean_reprojection_error_px']:.3f} px")
        print(f"  Refined mean error:     {ref['mean_reprojection_error_px']:.3f} px")
        
        if 'improvement' in comparison:
            imp = comparison['improvement']
            print(f"  Improvement:             {imp['absolute_px']:+.3f} px ({imp['relative_pct']:+.1f}%)")
    
    if os.path.exists(ba_results_path):
        with open(ba_results_path) as f:
            ba_results = json.load(f)
        
        print("\nImprovement per camera:")
        for view in sorted(ba_results.keys()):
            result = ba_results[view]
            old = result['old_error_px']
            new = result['new_error_px']
            imp = result['improvement_px']
            pct = imp / old * 100 if old > 0 else 0
            print(f"  {view}: {old:.1f} px -> {new:.1f} px ({imp:+.1f} px, {pct:+.1f}%)")
    
    print("\nGenerated files:")
    print("  - Bundle Adjustment results: output/ba_results/")
    print("  - Refined calibrations:   output/refined_calibrations/")
    print("  - Refined triangulation:    output/triangulation_results_refined.json")
    print("  - Comparison:               output/comparison_results.json")
    
    print("\nTasks 3 and 3a completed successfully!")
    print("="*70)


if __name__ == '__main__':
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    # Check that initial triangulation was run
    if not os.path.exists('output/triangulation_results.json'):
        print("Initial triangulation not found.")
        print("Please run: python3 triangulation.py")
        sys.exit(1)
    
    # Run tasks
    success = True
    
    if not run_bundle_adjustment():
        success = False
    
    if not run_evaluation():
        success = False
    
    if success:
        print_final_summary()
    else:
        print("\nError occurred. See messages above.")
        sys.exit(1)
