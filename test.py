#!/usr/bin/env python3
"""
Main test script for evidential deep learning models.

This script runs comprehensive testing and uncertainty analysis on trained
evidential models across multiple datasets including DS1, DS3 slabs, and CCNY datasets.

Usage:
    python test.py                          # Run with default settings
    python test.py --model custom_model     # Use custom model name
    python test.py --experiment exp2        # Use custom experiment name
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from testing import main_enhanced_analysis


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test evidential deep learning models with uncertainty quantification'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name (default: evidential_transformer)'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        help='Experiment name (default: evidential_transformer)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to model checkpoint (default: weights/{model_name}.pth)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for testing."""
    args = parse_args()
    
    print("=" * 80)
    print("Evidential Deep Learning Model Testing")
    print("=" * 80)
    print()
    
    if args.model:
        print(f"Model: {args.model}")
    if args.experiment:
        print(f"Experiment: {args.experiment}")
    if args.model_path:
        print(f"Model path: {args.model_path}")
    print()
    
    # Run the enhanced analysis
    success = main_enhanced_analysis(
        experiment_name=args.experiment,
        model_name=args.model,
        model_path=args.model_path,
    )
    
    if success:
        print("\n" + "=" * 80)
        print("✅ Testing completed successfully!")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("❌ Testing failed. Please check the error messages above.")
        print("=" * 80)
        sys.exit(1)


if __name__ == '__main__':
    main()
