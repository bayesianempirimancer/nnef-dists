#!/usr/bin/env python3
"""
Plotting utilities for training results.

This module provides functions for generating various plots from training results,
including learning curves, error analysis, and model performance visualizations.

Usage:
    from scripts.plotting.generate_plots import generate_plots
    generate_plots(output_dir, data_path)
"""

from pathlib import Path
from typing import Union

# Handle imports for both module usage and direct script execution
if __name__ == "__main__":
    # When run as script, add project root to path
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from scripts.load_model_and_data import load_model_and_data
from scripts.plotting.plot_learning_errors import create_learning_errors_plot


def generate_plots(output_dir: Union[str, Path], data_path: str) -> bool:
    """
    Generate training plots using the plot_learning_errors functionality.
    
    Args:
        output_dir: Directory where the model and results are saved
        data_path: Path to the training data file
        
    Returns:
        bool: True if plots were generated successfully, False otherwise
    """
    print("\nGenerating training plots...")
    
    try:
        # Load model and data directly
        config, results, data, model, params, metadata = load_model_and_data(str(output_dir), data_path)
        
        # Set save path and create plot
        save_path = Path(output_dir) / "learning_errors.png"
        create_learning_errors_plot(config, results, data, model, params, metadata, save_path)
        
        print("✅ Training plots generated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate training plots from model results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/plotting/generate_plots.py artifacts/my_model data/training_data.pkl
  python scripts/plotting/generate_plots.py --output-dir artifacts/my_model --data data/training_data.pkl
        """
    )
    parser.add_argument(
        "output_dir", 
        nargs="?", 
        help="Directory containing model artifacts (config.json, model_params.pkl, training_results.pkl)")
    parser.add_argument(
        "data_path", 
        nargs="?", 
        help="Path to training data pickle file")
    
    # Alternative argument names for clarity
    parser.add_argument(
        "--output-dir", 
        dest="output_dir_alt",
        help="Alternative way to specify output directory"
    )
    parser.add_argument(
        "--data", 
        dest="data_path_alt",
        help="Alternative way to specify data path"
    )
    
    args = parser.parse_args()
    
    # Determine final values (positional args take precedence)
    output_dir = args.output_dir or args.output_dir_alt
    data_path = args.data_path or args.data_path_alt
    
    if not output_dir:
        parser.error("Output directory is required. Use positional argument or --output-dir")
    if not data_path:
        parser.error("Data path is required. Use positional argument or --data")
    
    success = generate_plots(output_dir, data_path)
