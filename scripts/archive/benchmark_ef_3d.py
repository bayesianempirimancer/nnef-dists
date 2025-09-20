#!/usr/bin/env python3
"""
Accurate performance benchmark for ef.py 3D Gaussian operations.

This script separates JAX compilation time from execution time to give
accurate performance measurements for the core operations.
"""

import sys
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import random
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ef import MultivariateNormal


def benchmark_with_warmup(func, *args, warmup_runs=3, benchmark_runs=10, name="Operation"):
    """Benchmark a function with proper JAX compilation warmup."""
    print(f"\n🔥 Benchmarking: {name}")
    
    # Warmup runs to trigger compilation
    print(f"  Warming up with {warmup_runs} runs (includes compilation)...")
    warmup_times = []
    for i in range(warmup_runs):
        start = time.time()
        result = func(*args)
        # Force computation to complete
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, dict):
            for v in result.values():
                if hasattr(v, 'block_until_ready'):
                    v.block_until_ready()
        end = time.time()
        warmup_times.append(end - start)
        print(f"    Warmup {i+1}: {warmup_times[-1]:.4f}s")
    
    # Actual benchmark runs (post-compilation)
    print(f"  Running {benchmark_runs} benchmark iterations...")
    benchmark_times = []
    for i in range(benchmark_runs):
        start = time.time()
        result = func(*args)
        # Force computation to complete
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, dict):
            for v in result.values():
                if hasattr(v, 'block_until_ready'):
                    v.block_until_ready()
        end = time.time()
        benchmark_times.append(end - start)
    
    # Statistics
    first_run_time = warmup_times[0]  # Includes compilation
    avg_warmup = np.mean(warmup_times[1:]) if len(warmup_times) > 1 else warmup_times[0]
    avg_benchmark = np.mean(benchmark_times)
    std_benchmark = np.std(benchmark_times)
    min_benchmark = np.min(benchmark_times)
    max_benchmark = np.max(benchmark_times)
    
    print(f"  📊 Results:")
    print(f"    First run (w/ compilation): {first_run_time:.4f}s")
    print(f"    Avg warmup (post-compile):  {avg_warmup:.4f}s")
    print(f"    Avg execution time:         {avg_benchmark:.4f}s ± {std_benchmark:.4f}s")
    print(f"    Min/Max execution:          {min_benchmark:.4f}s / {max_benchmark:.4f}s")
    print(f"    Compilation overhead:       {first_run_time - avg_benchmark:.4f}s")
    
    return {
        'first_run': first_run_time,
        'avg_warmup': avg_warmup,
        'avg_execution': avg_benchmark,
        'std_execution': std_benchmark,
        'min_execution': min_benchmark,
        'max_execution': max_benchmark,
        'compilation_overhead': first_run_time - avg_benchmark
    }


def run_ef_benchmarks():
    """Run comprehensive benchmarks for 3D Gaussian ef.py operations."""
    print("🚀 ACCURATE 3D GAUSSIAN ef.py PERFORMANCE BENCHMARK")
    print("=" * 65)
    print("Separating JAX compilation time from execution time")
    
    # Setup
    ef_3d = MultivariateNormal(x_shape=(3,))
    rng = random.PRNGKey(42)
    
    # Test data sizes
    sizes = [100, 1000, 10000]
    results = {}
    
    for n_samples in sizes:
        print(f"\n{'='*65}")
        print(f"📊 TESTING WITH {n_samples:,} SAMPLES")
        print(f"{'='*65}")
        
        # Generate test data
        x_samples = random.normal(rng, (n_samples, 3))
        eta_samples = random.normal(random.split(rng)[0], (n_samples, 12))
        
        # Create test stats dict
        stats_dict = {
            "x": random.normal(random.split(rng)[1], (n_samples, 3)),
            "xxT": random.normal(random.split(rng)[0], (n_samples, 3, 3))
        }
        
        size_results = {}
        
        # 1. Compute Stats
        def compute_stats_fn():
            return ef_3d.compute_stats(x_samples, flatten=False)
        
        size_results['compute_stats'] = benchmark_with_warmup(
            compute_stats_fn, 
            name=f"Compute Stats ({n_samples:,} samples)"
        )
        
        # 2. Compute Expected Stats
        def expected_stats_fn():
            return ef_3d.compute_expected_stats(x_samples, flatten=False)
        
        size_results['expected_stats'] = benchmark_with_warmup(
            expected_stats_fn,
            name=f"Expected Stats ({n_samples:,} samples)"
        )
        
        # 3. Flatten Stats
        def flatten_stats_fn():
            return ef_3d.flatten_stats_or_eta(stats_dict)
        
        size_results['flatten_stats'] = benchmark_with_warmup(
            flatten_stats_fn,
            name=f"Flatten Stats ({n_samples:,} samples)"
        )
        
        # 4. Unflatten Stats
        flattened = ef_3d.flatten_stats_or_eta(stats_dict)
        def unflatten_stats_fn():
            return ef_3d.unflatten_stats_or_eta(flattened)
        
        size_results['unflatten_stats'] = benchmark_with_warmup(
            unflatten_stats_fn,
            name=f"Unflatten Stats ({n_samples:,} samples)"
        )
        
        # 5. Log Unnormalized (batch)
        def log_unnormalized_fn():
            return ef_3d.log_unnormalized(x_samples, eta_samples)
        
        size_results['log_unnormalized'] = benchmark_with_warmup(
            log_unnormalized_fn,
            name=f"Log Unnormalized ({n_samples:,} samples)"
        )
        
        results[n_samples] = size_results
    
    # Summary analysis
    print(f"\n{'='*65}")
    print("📈 PERFORMANCE SUMMARY")
    print(f"{'='*65}")
    
    operations = ['compute_stats', 'expected_stats', 'flatten_stats', 'unflatten_stats', 'log_unnormalized']
    
    print(f"\n{'Operation':<20} {'Size':<8} {'Compilation':<12} {'Execution':<12} {'Throughput':<15}")
    print("-" * 75)
    
    for op in operations:
        for n_samples in sizes:
            if op in results[n_samples]:
                res = results[n_samples][op]
                compilation = res['compilation_overhead']
                execution = res['avg_execution']
                throughput = n_samples / execution if execution > 0 else 0
                
                print(f"{op:<20} {n_samples:<8,} {compilation:<12.4f} {execution:<12.4f} {throughput:<15,.0f}")
    
    # Scaling analysis
    print(f"\n🔍 SCALING ANALYSIS:")
    for op in operations:
        if all(op in results[size] for size in sizes):
            times = [results[size][op]['avg_execution'] for size in sizes]
            ratios = [times[i]/times[i-1] for i in range(1, len(times))]
            size_ratios = [sizes[i]/sizes[i-1] for i in range(1, len(sizes))]
            
            print(f"  {op}:")
            for i, (size_ratio, time_ratio) in enumerate(zip(size_ratios, ratios)):
                efficiency = size_ratio / time_ratio
                print(f"    {sizes[i]//1000}k→{sizes[i+1]//1000}k: {time_ratio:.2f}x time for {size_ratio:.0f}x data (efficiency: {efficiency:.2f})")
    
    # Memory efficiency estimate
    print(f"\n💾 MEMORY EFFICIENCY:")
    for n_samples in sizes:
        # Estimate memory usage
        input_size = n_samples * 3 * 4  # float32 bytes
        output_size = n_samples * 12 * 4  # float32 bytes for flattened stats
        total_mb = (input_size + output_size) / (1024 * 1024)
        
        exec_time = results[n_samples]['compute_stats']['avg_execution']
        throughput_mb_s = total_mb / exec_time if exec_time > 0 else 0
        
        print(f"  {n_samples:,} samples: ~{total_mb:.1f}MB, {throughput_mb_s:.0f} MB/s throughput")
    
    return results


if __name__ == "__main__":
    print("Starting accurate ef.py 3D Gaussian benchmarks...")
    print("This will separate JAX compilation time from execution time.")
    print()
    
    try:
        results = run_ef_benchmarks()
        print(f"\n✅ Benchmarking completed successfully!")
        print(f"\n💡 Key Takeaway: Execution times exclude JAX compilation overhead.")
        print(f"   First-run times include compilation and are much slower.")
        
    except Exception as e:
        print(f"\n❌ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
