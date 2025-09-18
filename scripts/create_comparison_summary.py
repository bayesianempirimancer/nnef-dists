#!/usr/bin/env python3
"""
Create a comprehensive comparison summary of all neural network approaches.

This script loads results from all three approaches and creates unified comparison plots.
"""

import sys
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results():
    """Load results from all approaches."""
    results = {}
    
    # Standard MLP results
    try:
        with open("artifacts/large_1d_training_history_7a6d32f607c09d157b94129eff71e641.pkl", 'rb') as f:
            mlp_data = pickle.load(f)
        results['Standard MLP'] = {
            'final_train_mse': mlp_data['history']['train_mse'][-1],
            'final_val_mse': mlp_data['history']['val_mse'][-1],
            'test_mse': mlp_data['history']['val_mse'][-1],  # Use validation as proxy for test
            'training_time': 'N/A',  # Not recorded in this format
            'approach': 'Supervised Learning'
        }
        print("✅ Loaded Standard MLP results")
    except (FileNotFoundError, KeyError) as e:
        print(f"❌ Standard MLP results not found: {e}")
        results['Standard MLP'] = None
    
    # NoProp-CT results
    try:
        with open("artifacts/noprop_ct_demo/demo_summary_gaussian_1d.txt", 'r') as f:
            noprop_summary = f.read()
        
        # Parse key metrics from summary
        lines = noprop_summary.split('\\n')
        train_time = None
        final_mse = None
        
        for line in lines:
            if 'Training time:' in line:
                train_time = float(line.split(':')[1].strip().replace('s', ''))
            elif 'MSE:' in line and 'Test Performance' in noprop_summary:
                final_mse = float(line.split(':')[1].strip())
                break
        
        results['NoProp-CT'] = {
            'final_train_mse': 'N/A',
            'final_val_mse': 'N/A', 
            'test_mse': final_mse,
            'training_time': train_time,
            'approach': 'Continuous-Time Neural ODE'
        }
        print("✅ Loaded NoProp-CT results")
    except FileNotFoundError:
        print("❌ NoProp-CT results not found")
        results['NoProp-CT'] = None
    
    # Simple INN results
    try:
        with open("artifacts/simple_inn_demo/summary.txt", 'r') as f:
            inn_summary = f.read()
        
        # Parse key metrics
        lines = inn_summary.split('\\n')
        train_time = None
        forward_mse = None
        invertibility_error = None
        
        for line in lines:
            if 'Training time:' in line:
                train_time = float(line.split(':')[1].strip().replace('s', ''))
            elif 'Forward MSE' in line:
                forward_mse = float(line.split(':')[1].strip())
            elif 'Invertibility error:' in line:
                invertibility_error = float(line.split(':')[1].strip())
        
        results['Simple INN'] = {
            'final_train_mse': 'N/A',
            'final_val_mse': 'N/A',
            'test_mse': forward_mse,
            'training_time': train_time,
            'invertibility_error': invertibility_error,
            'approach': 'Invertible Neural Network'
        }
        print("✅ Loaded Simple INN results")
    except FileNotFoundError:
        print("❌ Simple INN results not found")
        results['Simple INN'] = None
    
    return results


def create_comparison_plots(results):
    """Create comprehensive comparison plots."""
    
    # Filter out None results
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) < 2:
        print("Not enough results for comparison")
        return
    
    # Extract data for plotting
    methods = list(valid_results.keys())
    test_mses = [valid_results[method].get('test_mse', valid_results[method].get('final_val_mse', 0)) 
                 for method in methods]
    training_times = [valid_results[method].get('training_time', 0) for method in methods]
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Neural Network Approaches Comparison\\nExponential Family Moment Mapping', fontsize=16)
    
    # Test MSE comparison
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    bars1 = axes[0, 0].bar(methods, test_mses, color=colors[:len(methods)], alpha=0.8)
    axes[0, 0].set_ylabel('Test MSE')
    axes[0, 0].set_title('Test Performance Comparison')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, mse in zip(bars1, test_mses):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{mse:.3f}', ha='center', va='bottom')
    
    # Training time comparison
    valid_times = [(method, time) for method, time in zip(methods, training_times) if time != 'N/A' and time > 0]
    if valid_times:
        time_methods, times = zip(*valid_times)
        bars2 = axes[0, 1].bar(time_methods, times, color=colors[:len(time_methods)], alpha=0.8)
        axes[0, 1].set_ylabel('Training Time (seconds)')
        axes[0, 1].set_title('Training Efficiency Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, time in zip(bars2, times):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{time:.1f}s', ha='center', va='bottom')
    
    # Performance vs Speed scatter
    if valid_times and len(valid_times) >= 2:
        perf_speed_methods = []
        perf_speed_mses = []
        perf_speed_times = []
        
        for method in methods:
            if (method in [t[0] for t in valid_times] and 
                valid_results[method].get('test_mse') is not None):
                perf_speed_methods.append(method)
                perf_speed_mses.append(valid_results[method]['test_mse'])
                perf_speed_times.append(valid_results[method]['training_time'])
        
        if perf_speed_methods:
            scatter = axes[1, 0].scatter(perf_speed_times, perf_speed_mses, 
                                       c=colors[:len(perf_speed_methods)], s=100, alpha=0.8)
            
            for i, method in enumerate(perf_speed_methods):
                axes[1, 0].annotate(method, (perf_speed_times[i], perf_speed_mses[i]),
                                   xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            axes[1, 0].set_xlabel('Training Time (seconds)')
            axes[1, 0].set_ylabel('Test MSE')
            axes[1, 0].set_title('Performance vs Speed Trade-off')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
    
    # Summary table
    axes[1, 1].axis('off')
    
    # Create summary text
    summary_text = "Method Comparison Summary\\n" + "="*30 + "\\n\\n"
    
    for method, data in valid_results.items():
        summary_text += f"{method}:\\n"
        summary_text += f"  Approach: {data.get('approach', 'Unknown')}\\n"
        
        if data.get('test_mse') is not None:
            summary_text += f"  Test MSE: {data['test_mse']:.6f}\\n"
        elif data.get('final_val_mse') is not None:
            summary_text += f"  Val MSE: {data['final_val_mse']:.6f}\\n"
        
        if data.get('training_time') not in ['N/A', None]:
            summary_text += f"  Time: {data['training_time']:.1f}s\\n"
        
        if 'invertibility_error' in data:
            summary_text += f"  Invertibility: {data['invertibility_error']:.2e}\\n"
        
        summary_text += "\\n"
    
    # Best performance
    best_mse = min([d.get('test_mse', d.get('final_val_mse', float('inf'))) 
                   for d in valid_results.values()])
    best_method = [k for k, v in valid_results.items() 
                   if v.get('test_mse', v.get('final_val_mse', float('inf'))) == best_mse][0]
    
    summary_text += f"🏆 Best Performance: {best_method}\\n"
    summary_text += f"   MSE: {best_mse:.6f}\\n"
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save comparison
    output_dir = Path("artifacts/comparison_summary")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "methods_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "methods_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"\\n📊 Comparison plots saved to {output_dir}/methods_comparison.png")
    
    # Save detailed summary
    with open(output_dir / "detailed_summary.txt", 'w') as f:
        f.write("Comprehensive Neural Network Approaches Comparison\\n")
        f.write("="*50 + "\\n\\n")
        
        for method, data in valid_results.items():
            f.write(f"{method}:\\n")
            for key, value in data.items():
                f.write(f"  {key}: {value}\\n")
            f.write("\\n")
    
    return valid_results


def main():
    print("Creating comprehensive comparison summary...")
    print("="*50)
    
    # Load all results
    results = load_results()
    
    # Create comparison plots
    comparison_data = create_comparison_plots(results)
    
    # Print summary
    print("\\n📈 Performance Summary:")
    for method, data in comparison_data.items():
        test_mse = data.get('test_mse', data.get('final_val_mse', 'N/A'))
        time = data.get('training_time', 'N/A')
        print(f"  {method}: MSE = {test_mse}, Time = {time}")
    
    print("\\n✅ Comparison summary completed!")
    print("📁 Check artifacts/comparison_summary/ for detailed results")


if __name__ == "__main__":
    main()
