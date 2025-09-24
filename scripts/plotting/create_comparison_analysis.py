#!/usr/bin/env python3
"""
Create comprehensive analysis from all trained models.

This script collects results from all individual model training runs
and creates comprehensive comparison plots and analysis.
"""

import os
import sys
import pickle
import json
import time
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from collections import defaultdict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def collect_all_results(base_dir="artifacts", mode="full"):
    """
    Collect results from all trained models.
    
    Args:
        base_dir: Base directory to search for results
        mode: "full" for ET_models/logZ_models or "test" for tests directory
    
    Returns:
        Dictionary with all model results
    """
    base_path = Path(base_dir)
    all_results = {}
    
    # Determine which directories to search based on mode
    if mode == "full":
        search_dirs = ["ET_models", "logZ_models"]
        print("🔍 Collecting results from comprehensive comparison (ET_models, logZ_models)...")
    elif mode == "test":
        search_dirs = ["tests"]
        print("🔍 Collecting results from test runs (tests directory)...")
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'full' or 'test'.")
    
    # Search for results in specified directories
    for model_type_dir in search_dirs:
        type_path = base_path / model_type_dir
        
        if not type_path.exists():
            continue
            
        for model_dir in type_path.iterdir():
            if not model_dir.is_dir():
                continue
                
            model_name = model_dir.name
            
            # Look for results.json file (created by run_comprehensive_model_comparison.py)
            results_file = model_dir / "results.json"
            training_summary_file = model_dir / "training_summary.pkl"
            
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        results_data = json.load(f)
                    
                    # Extract model results from the JSON structure
                    if model_name in results_data:
                        model_result = results_data[model_name]
                        
                        # Convert to expected format
                        # Determine model type (handle tests directory)
                        if model_type_dir == 'tests':
                            # Determine model type from model name
                            if model_name.endswith('_ET'):
                                model_type = 'ET'
                            elif model_name.endswith('_logZ'):
                                model_type = 'logZ'
                            else:
                                model_type = 'Unknown'
                        else:
                            model_type = model_type_dir.replace('_models', '')
                        
                        all_results[model_name] = {
                            'model_name': model_name,
                            'model_type': model_type,
                            'training_time': model_result.get('training_time', np.nan),
                            'parameter_count': model_result.get('parameter_count', np.nan),
                            'test_results': {
                                'mse': model_result.get('mse', np.nan),
                                'mae': model_result.get('mae', np.nan)
                            },
                            'inference_timing': {
                                'per_sample_ms': model_result.get('inference_time_per_sample', np.nan) * 1000 if model_result.get('inference_time_per_sample') else np.nan
                            },
                            'predictions': model_result.get('predictions', []),
                            'ground_truth': model_result.get('ground_truth', [])
                        }
                        
                        print(f"✅ Loaded {model_name} results from results.json")
                    else:
                        print(f"⚠️  Model {model_name} not found in results.json")
                    
                except Exception as e:
                    print(f"❌ Failed to load {model_name} from results.json: {e}")
                    
            elif training_summary_file.exists():
                try:
                    with open(training_summary_file, 'rb') as f:
                        summary_data = pickle.load(f)
                    
                    # Extract model results from training summary
                    if model_name in summary_data:
                        model_result = summary_data[model_name]
                        
                        # Create results from training summary
                        # Determine model type (handle tests directory)
                        if model_type_dir == 'tests':
                            # Determine model type from model name
                            if model_name.endswith('_ET'):
                                model_type = 'ET'
                            elif model_name.endswith('_logZ'):
                                model_type = 'logZ'
                            else:
                                model_type = 'Unknown'
                        else:
                            model_type = model_type_dir.replace('_models', '')
                        
                        all_results[model_name] = {
                            'model_name': model_name,
                            'model_type': model_type,
                            'training_time': model_result.get('training_time', np.nan),
                            'parameter_count': model_result.get('parameter_count', np.nan),
                            'test_results': {
                                'mse': model_result.get('mse', np.nan),
                                'mae': model_result.get('mae', np.nan)
                            },
                            'inference_timing': {
                                'per_sample_ms': model_result.get('inference_time_per_sample', np.nan) * 1000 if model_result.get('inference_time_per_sample') else np.nan
                            }
                        }
                        
                        print(f"✅ Loaded {model_name} from training_summary.pkl")
                    else:
                        print(f"⚠️  Model {model_name} not found in training_summary.pkl")
                    
                except Exception as e:
                    print(f"❌ Failed to load {model_name} from training_summary.pkl: {e}")
    
    return all_results


def measure_inference_times(all_results, data_file="data/easy_3d_gaussian.pkl"):
    """
    Measure inference times for models that don't have timing data.
    """
    # Load test data
    try:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        test_eta = data['test']['eta'][:100]  # Use subset for timing
    except:
        print("⚠️  Could not load test data for inference timing")
        return all_results
    
    # For models without timing data, add placeholder
    for model_name, result in all_results.items():
        if 'inference_timing' not in result or result['inference_timing']['per_sample_ms'] == np.nan:
            # Add placeholder timing (could be measured if model is available)
            result['inference_timing'] = {
                'per_sample_ms': np.nan,
                'mean_time_ms': np.nan
            }
    
    return all_results


def create_performance_comparison_table(all_results):
    """Create a detailed performance comparison table."""
    
    # Prepare data for table
    table_data = []
    
    for model_name, result in all_results.items():
        perf = result.get('test_results', result.get('test_performance', {}))
        timing = result.get('inference_timing', {})
        
        # Get MSE (handle different field names)
        mse = perf.get('mse', perf.get('mean_mse', np.nan))
        mae = perf.get('mae', perf.get('mean_mae', np.nan))
        
        row = {
            'Model': model_name,
            'Type': result.get('model_type', 'Unknown'),
            'Parameters': result.get('parameter_count', np.nan),
            'Training Time (s)': result.get('training_time', np.nan),
            'Test MSE': mse,
            'Test MAE': mae,
            'Inference (ms/sample)': timing.get('per_sample_ms', np.nan)
        }
        
        table_data.append(row)
    
    # Sort by Test MSE (best first, handling NaN values)
    def sort_key(row):
        mse = row['Test MSE']
        return (np.isnan(mse), mse)  # NaN values go to end
    
    table_data.sort(key=sort_key)
    
    return table_data


def create_comprehensive_plots(all_results, save_dir, mode="full"):
    """Create comprehensive comparison plots."""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter valid results
    valid_results = {name: result for name, result in all_results.items() 
                    if result.get('test_results') is not None or result.get('test_performance') is not None}
    
    if len(valid_results) < 1:
        print("⚠️  Need at least 1 model for comparison plots")
        return
    
    # Extract metrics
    model_names = list(valid_results.keys())
    model_types = [valid_results[name].get('model_type', 'Unknown') for name in model_names]
    
    # Performance metrics
    test_mses = []
    test_maes = []
    param_counts = []
    training_times = []
    inference_times = []
    
    for name in model_names:
        result = valid_results[name]
        perf = result.get('test_results', result.get('test_performance', {}))
        
        # MSE/MAE
        mse = perf.get('mse', perf.get('mean_mse', np.nan))
        mae = perf.get('mae', perf.get('mean_mae', np.nan))
        test_mses.append(mse)
        test_maes.append(mae)
        
        # Computational metrics
        param_counts.append(result.get('parameter_count', np.nan))
        training_times.append(result.get('training_time', np.nan))
        
        timing = result.get('inference_timing', {})
        inference_times.append(timing.get('per_sample_ms', np.nan))
    
    # Create color mapping for model types
    unique_types = list(set(model_types))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_types)))
    type_colors = {t: colors[i] for i, t in enumerate(unique_types)}
    bar_colors = [type_colors[t] for t in model_types]
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    # 1. Test MSE comparison
    bars1 = axes[0, 0].bar(range(len(model_names)), test_mses, 
                          alpha=0.8, color=bar_colors)
    axes[0, 0].set_xticks(range(len(model_names)))
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Test MSE')
    axes[0, 0].set_title('Test MSE Comparison (Lower is Better)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mse in zip(bars1, test_mses):
        if not np.isnan(mse):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{mse:.2e}', ha='center', va='bottom', fontsize=8)
    
    # 2. Parameter count comparison
    bars2 = axes[0, 1].bar(range(len(model_names)), param_counts, 
                          alpha=0.8, color=bar_colors)
    axes[0, 1].set_xticks(range(len(model_names)))
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Parameter Count')
    axes[0, 1].set_title('Model Size Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars2, param_counts):
        if not np.isnan(count):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{int(count):,}', ha='center', va='bottom', fontsize=8)
    
    # 3. Training time comparison
    bars3 = axes[0, 2].bar(range(len(model_names)), training_times, 
                          alpha=0.8, color=bar_colors)
    axes[0, 2].set_xticks(range(len(model_names)))
    axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 2].set_ylabel('Training Time (seconds)')
    axes[0, 2].set_title('Training Time Comparison')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, time_val in zip(bars3, training_times):
        if not np.isnan(time_val):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{time_val:.1f}s', ha='center', va='bottom', fontsize=8)
    
    # 4. Inference time comparison
    valid_inf_times = [t for t in inference_times if not np.isnan(t)]
    valid_inf_names = [name for name, t in zip(model_names, inference_times) if not np.isnan(t)]
    valid_inf_colors = [color for color, t in zip(bar_colors, inference_times) if not np.isnan(t)]
    
    if valid_inf_times:
        bars4 = axes[0, 3].bar(range(len(valid_inf_names)), valid_inf_times, 
                              alpha=0.8, color=valid_inf_colors)
        axes[0, 3].set_xticks(range(len(valid_inf_names)))
        axes[0, 3].set_xticklabels(valid_inf_names, rotation=45, ha='right')
        axes[0, 3].set_ylabel('Inference Time (ms/sample)')
        axes[0, 3].set_title('Inference Speed Comparison (Lower is Better)')
        axes[0, 3].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, time_val in zip(bars4, valid_inf_times):
            axes[0, 3].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{time_val:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        axes[0, 3].text(0.5, 0.5, 'No inference timing data available', 
                       ha='center', va='center', transform=axes[0, 3].transAxes)
        axes[0, 3].set_title('Inference Speed Comparison')
    
    # 5. Accuracy vs Model Size
    valid_indices = [i for i, (mse, params) in enumerate(zip(test_mses, param_counts)) 
                    if not (np.isnan(mse) or np.isnan(params))]
    
    if valid_indices:
        scatter_mses = [test_mses[i] for i in valid_indices]
        scatter_params = [param_counts[i] for i in valid_indices]
        scatter_names = [model_names[i] for i in valid_indices]
        scatter_colors = [bar_colors[i] for i in valid_indices]
        
        axes[1, 0].scatter(scatter_params, scatter_mses, s=100, alpha=0.7, c=scatter_colors)
        for i, name in enumerate(scatter_names):
            axes[1, 0].annotate(name, (scatter_params[i], scatter_mses[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[1, 0].set_xlabel('Parameter Count')
        axes[1, 0].set_ylabel('Test MSE')
        axes[1, 0].set_title('Accuracy vs Model Size')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 6. Training Speed vs Accuracy
    valid_indices = [i for i, (mse, time_val) in enumerate(zip(test_mses, training_times)) 
                    if not (np.isnan(mse) or np.isnan(time_val))]
    
    if valid_indices:
        scatter_mses = [test_mses[i] for i in valid_indices]
        scatter_times = [training_times[i] for i in valid_indices]
        scatter_names = [model_names[i] for i in valid_indices]
        scatter_colors = [bar_colors[i] for i in valid_indices]
        
        axes[1, 1].scatter(scatter_times, scatter_mses, s=100, alpha=0.7, c=scatter_colors)
        for i, name in enumerate(scatter_names):
            axes[1, 1].annotate(name, (scatter_times[i], scatter_mses[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[1, 1].set_xlabel('Training Time (seconds)')
        axes[1, 1].set_ylabel('Test MSE')
        axes[1, 1].set_title('Training Speed vs Accuracy')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 7. Computational Efficiency Score
    efficiency_scores = []
    efficiency_names = []
    efficiency_colors = []
    
    for i, (name, mse, params, inf_time) in enumerate(zip(model_names, test_mses, param_counts, inference_times)):
        if not (np.isnan(mse) or np.isnan(params)):
            # Efficiency = MSE * log(parameters) * (inference_time or 1)
            inf_penalty = inf_time if not np.isnan(inf_time) else 1.0
            score = mse * np.log10(params) * inf_penalty
            efficiency_scores.append(score)
            efficiency_names.append(name)
            efficiency_colors.append(bar_colors[i])
    
    if efficiency_scores:
        bars7 = axes[1, 2].bar(range(len(efficiency_names)), efficiency_scores, 
                              alpha=0.8, color=efficiency_colors)
        axes[1, 2].set_xticks(range(len(efficiency_names)))
        axes[1, 2].set_xticklabels(efficiency_names, rotation=45, ha='right')
        axes[1, 2].set_ylabel('Efficiency Score')
        axes[1, 2].set_title('Overall Efficiency (Lower is Better)')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True, alpha=0.3)
    
    # 8. Model Type Legend and Summary
    axes[1, 3].axis('off')
    
    # Create legend for model types
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=type_colors[t], alpha=0.8) 
                      for t in unique_types]
    axes[1, 3].legend(legend_elements, unique_types, loc='upper left', title='Model Types')
    
    # Add summary statistics
    summary_text = f"""
Summary Statistics:
• Models compared: {len(model_names)}
• Model types: {', '.join(unique_types)}
• Best MSE: {np.nanmin(test_mses):.2e}
• Parameter range: {np.nanmin(param_counts):,.0f} - {np.nanmax(param_counts):,.0f}
• Training time range: {np.nanmin(training_times):.1f}s - {np.nanmax(training_times):.1f}s
    """.strip()
    
    axes[1, 3].text(0.05, 0.6, summary_text, transform=axes[1, 3].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Create mode-specific title
    if mode == "test":
        title = 'Small Architecture Test Results - 3D Gaussian Exponential Family'
    else:
        title = 'Comprehensive Model Comparison - 3D Gaussian Exponential Family'
    
    plt.suptitle(title, fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_file = save_dir / "comprehensive_model_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive plot saved: {plot_file}")
    
    plt.close()
    
    return plot_file


def create_performance_table(all_results, save_dir):
    """Create and save performance comparison table."""
    
    table_data = create_performance_comparison_table(all_results)
    
    save_dir = Path(save_dir)
    
    # Save as CSV
    csv_file = save_dir / "model_comparison_table.csv"
    with open(csv_file, 'w') as f:
        # Write header
        headers = ['Model', 'Type', 'Parameters', 'Training Time (s)', 'Test MSE', 'Test MAE', 'Inference (ms/sample)']
        f.write(','.join(headers) + '\n')
        
        # Write data
        for row in table_data:
            values = []
            for header in headers:
                val = row[header]
                if isinstance(val, float) and not np.isnan(val):
                    if header in ['Test MSE', 'Test MAE']:
                        values.append(f'{val:.2e}')
                    elif header == 'Inference (ms/sample)':
                        values.append(f'{val:.3f}')
                    else:
                        values.append(f'{val:.1f}')
                else:
                    values.append(str(val))
            f.write(','.join(values) + '\n')
    
    # Save as formatted text
    txt_file = save_dir / "model_comparison_table.txt"
    with open(txt_file, 'w') as f:
        f.write("COMPREHENSIVE MODEL COMPARISON TABLE\n")
        f.write("="*80 + "\n\n")
        
        # Format table
        f.write(f"{'Model':<20} {'Type':<6} {'Params':<10} {'Train(s)':<10} {'MSE':<12} {'MAE':<12} {'Infer(ms)':<12}\n")
        f.write("-" * 80 + "\n")
        
        for row in table_data:
            mse = row['Test MSE']
            mae = row['Test MAE']
            params = row['Parameters']
            train_time = row['Training Time (s)']
            infer_time = row['Inference (ms/sample)']
            
            mse_str = f"{mse:.2e}" if not np.isnan(mse) else "N/A"
            mae_str = f"{mae:.2e}" if not np.isnan(mae) else "N/A"
            params_str = f"{int(params):,}" if not np.isnan(params) else "N/A"
            train_str = f"{train_time:.1f}" if not np.isnan(train_time) else "N/A"
            infer_str = f"{infer_time:.3f}" if not np.isnan(infer_time) else "N/A"
            
            f.write(f"{row['Model']:<20} {row['Type']:<6} {params_str:<10} {train_str:<10} {mse_str:<12} {mae_str:<12} {infer_str:<12}\n")
        
        f.write("\n\nNotes:\n")
        f.write("- MSE/MAE: Lower is better\n")
        f.write("- Parameters: Model complexity\n") 
        f.write("- Training Time: Time to train (seconds)\n")
        f.write("- Inference Time: Time per prediction (milliseconds)\n")
    
    print(f"📊 Performance table saved: {csv_file}")
    print(f"📊 Performance table (text): {txt_file}")
    
    return table_data


def main():
    """Create comprehensive comparison analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create comprehensive model comparison analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/create_comparison_analysis.py --mode full
  python scripts/create_comparison_analysis.py --mode test
  python scripts/create_comparison_analysis.py --mode full --output artifacts/comprehensive_analysis
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'test'],
        default='full',
        help='Analysis mode: "full" for ET_models/logZ_models, "test" for tests directory'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='artifacts/comprehensive_comparison',
        help='Output directory for analysis results'
    )
    
    args = parser.parse_args()
    
    mode = args.mode
    output_dir = args.output
    
    print("📊 Creating Comprehensive Model Comparison Analysis")
    print("="*55)
    print(f"🎯 Mode: {mode}")
    print(f"📁 Output: {output_dir}")
    
    # Collect all results
    all_results = collect_all_results(mode=mode)
    
    if not all_results:
        print("❌ No model results found!")
        if mode == "full":
            print("Please train models first using:")
            print("  python scripts/run_comprehensive_model_comparison.py --data data/easy_3d_gaussian.pkl --ef gaussian_3d")
        else:  # test mode
            print("Please run test training first using:")
            print("  python scripts/test_all_training_scripts.py")
        return 1
    
    print(f"✅ Found results for {len(all_results)} models:")
    for name in sorted(all_results.keys()):
        result = all_results[name]
        model_type = result.get('model_type', 'Unknown')
        print(f"   • {name} ({model_type})")
    
    # Measure any missing inference times
    all_results = measure_inference_times(all_results)
    
    # Create save directory
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive plots
    print("\n📊 Creating comprehensive comparison plots...")
    plot_file = create_comprehensive_plots(all_results, save_dir, mode)
    
    # Create performance table
    print("\n📋 Creating performance comparison table...")
    table_data = create_performance_table(all_results, save_dir)
    
    # Save all results as JSON
    json_file = save_dir / "all_results_summary.json"
    
    # Convert results to JSON-serializable format
    def make_json_serializable(obj):
        """Recursively convert JAX/numpy arrays to lists."""
        if isinstance(obj, (np.ndarray, jnp.ndarray)):
            return obj.tolist()
        elif hasattr(obj, '__array__'):  # JAX ArrayImpl
            return np.array(obj).tolist()
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    json_results = {}
    for name, result in all_results.items():
        # Skip non-serializable items like predict_fn
        filtered_result = {k: v for k, v in result.items() 
                          if k not in ['predict_fn', 'params']}
        json_results[name] = make_json_serializable(filtered_result)
    
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"💾 Complete results saved: {json_file}")
    
    # Print summary
    print(f"\n{'='*55}")
    print("🏆 COMPREHENSIVE ANALYSIS COMPLETED")
    print(f"{'='*55}")
    print(f"📊 {len(all_results)} models analyzed")
    print(f"📁 Results saved to: {save_dir}")
    print(f"📈 Main plot: {plot_file}")
    
    # Print top performers
    print(f"\n🥇 TOP PERFORMERS:")
    # table_data is already sorted by MSE
    for i, row in enumerate(table_data[:3], 1):
        mse = row['Test MSE']
        if not np.isnan(mse):
            print(f"   {i}. {row['Model']} (MSE: {mse:.2e})")
        else:
            print(f"   {i}. {row['Model']} (MSE: N/A)")
    
    print(f"\n🎯 View all results in: {save_dir}/")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
