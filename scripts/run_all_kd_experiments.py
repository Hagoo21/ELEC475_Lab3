"""
Run All Knowledge Distillation Experiments
===========================================

This script runs all three KD training experiments:
1. Response-based distillation only
2. Feature-based distillation only  
3. Combined (both methods) - optional

Then generates comparison results.

Usage:
    python scripts/run_all_kd_experiments.py [--epochs EPOCHS] [--skip-combined]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import subprocess


def run_experiment(method, epochs, batch_size=8):
    """
    Run a single KD training experiment.
    
    Args:
        method: 'response', 'feature', or 'both'
        epochs: Number of training epochs
        batch_size: Batch size
    """
    print("\n" + "=" * 80)
    print(f"Running {method.upper()} Knowledge Distillation Training")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        'scripts/train_knowledge_distillation.py',
        '--method', method,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size)
    ]
    
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    # Run training
    result = subprocess.run(cmd, cwd=os.getcwd())
    
    if result.returncode != 0:
        print(f"\nERROR: Training failed for method '{method}'")
        return False
    
    print(f"\n✓ {method.upper()} training completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run all KD experiments')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs for each experiment')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--skip-combined', action='store_true',
                       help='Skip the combined (both) method experiment')
    parser.add_argument('--only', type=str, choices=['response', 'feature', 'both'],
                       help='Run only a specific experiment')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Knowledge Distillation Experiments Suite")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Epochs per experiment: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Skip combined: {args.skip_combined}")
    
    if args.only:
        print(f"  Running only: {args.only}")
    
    print("\n" + "=" * 80)
    
    results = {}
    
    # Run experiments
    if args.only:
        # Run only specified experiment
        success = run_experiment(args.only, args.epochs, args.batch_size)
        results[args.only] = success
    else:
        # Run all experiments
        # 1. Response-based
        print("\n" + "=" * 80)
        print("EXPERIMENT 1/3: Response-Based Distillation")
        print("=" * 80)
        success = run_experiment('response', args.epochs, args.batch_size)
        results['response'] = success
        
        if not success:
            print("\nWARNING: Response-based training failed. Continuing anyway...")
        
        # 2. Feature-based
        print("\n" + "=" * 80)
        print("EXPERIMENT 2/3: Feature-Based Distillation")
        print("=" * 80)
        success = run_experiment('feature', args.epochs, args.batch_size)
        results['feature'] = success
        
        if not success:
            print("\nWARNING: Feature-based training failed. Continuing anyway...")
        
        # 3. Combined (optional)
        if not args.skip_combined:
            print("\n" + "=" * 80)
            print("EXPERIMENT 3/3: Combined (Response + Feature) Distillation")
            print("=" * 80)
            success = run_experiment('both', args.epochs, args.batch_size)
            results['both'] = success
            
            if not success:
                print("\nWARNING: Combined training failed.")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENTS SUMMARY")
    print("=" * 80)
    
    for method, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {method.capitalize():<20} {status}")
    
    print("\n" + "=" * 80)
    
    # Run comparison if at least 2 experiments succeeded
    successful_count = sum(1 for s in results.values() if s)
    
    if successful_count >= 2:
        print("\nRunning comparison analysis...")
        print("=" * 80)
        
        cmd = [sys.executable, 'scripts/compare_kd_methods.py']
        subprocess.run(cmd, cwd=os.getcwd())
        
        print("\n✓ All experiments completed!")
        print("\nGenerated files:")
        print("  - Checkpoints: checkpoints_optimized/student_kd_*_best.pth")
        print("  - Training history: checkpoints_optimized/kd_training_history_*.pth")
        print("  - Comparison results: checkpoints_optimized/kd_comparison_results.txt")
    else:
        print("\nWARNING: Not enough successful experiments to run comparison.")
        print(f"Need at least 2, but only {successful_count} succeeded.")
        print("\nYou can still manually run comparison later:")
        print("  python scripts/compare_kd_methods.py")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

