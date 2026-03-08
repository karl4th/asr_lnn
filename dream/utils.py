"""
Performance utilities for DREAM.

Includes:
- Benchmarking tools
- Memory profiling
- Speed optimization helpers
"""

import torch
import time
from typing import Dict, List
from . import DREAM, DREAMConfig, DREAMCell


def benchmark_dream(
    model: torch.nn.Module,
    input_shape: tuple,
    device: str = 'cuda',
    num_warmup: int = 10,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark DREAM model performance.

    Parameters
    ----------
    model : torch.nn.Module
        DREAM model to benchmark
    input_shape : tuple
        Input tensor shape (batch, time, features)
    device : str
        Device to use ('cuda' or 'cpu')
    num_warmup : int
        Number of warmup runs
    num_runs : int
        Number of benchmark runs

    Returns
    -------
    dict
        Benchmark results (latency, throughput)
    """
    model = model.to(device)
    model.eval()

    # Create input
    x = torch.randn(*input_shape, device=device)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start_time

    # Compute metrics
    avg_latency = elapsed / num_runs * 1000  # ms
    throughput = num_runs / elapsed  # iterations/sec
    samples_per_sec = input_shape[0] * throughput

    return {
        'avg_latency_ms': avg_latency,
        'throughput_iter_sec': throughput,
        'samples_per_sec': samples_per_sec,
        'total_time_sec': elapsed,
    }


def compare_optimizations(
    config: DREAMConfig,
    input_shape: tuple = (4, 100, 80),
    device: str = 'cuda'
) -> Dict[str, Dict[str, float]]:
    """
    Compare performance of different optimizations.

    Parameters
    ----------
    config : DREAMConfig
        Model configuration
    input_shape : tuple
        Input shape for benchmarking
    device : str
        Device to use

    Returns
    -------
    dict
        Comparison results for each optimization
    """
    results = {}

    # Standard DREAM
    print("Benchmarking standard DREAM...")
    model_standard = DREAM(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        rank=config.rank,
    )
    results['standard'] = benchmark_dream(model_standard, input_shape, device)

    # Optimized DREAM
    print("Benchmarking optimized DREAM...")
    try:
        from .cell_optimized import DREAMCellOptimized
        model_optimized = DREAMCellOptimized(config)
        results['optimized'] = benchmark_dream(model_optimized, input_shape, device)
    except ImportError:
        print("  Optimized version not available")

    # Mixed Precision
    if torch.cuda.is_available():
        print("Benchmarking mixed precision (AMP)...")
        try:
            from .cell_optimized import DREAMCellAMP
            model_amp = DREAMCellAMP(config)
            results['amp'] = benchmark_dream(model_amp, input_shape, 'cuda')
        except ImportError:
            print("  AMP version not available")

    # Print comparison
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)

    baseline = results['standard']['throughput_iter_sec']

    for name, metrics in results.items():
        speedup = metrics['throughput_iter_sec'] / baseline
        print(f"{name:15} | "
              f"Latency: {metrics['avg_latency_ms']:6.2f}ms | "
              f"Throughput: {metrics['throughput_iter_sec']:8.1f} it/s | "
              f"Speedup: {speedup:.2f}x")

    return results


def profile_memory(
    model: torch.nn.Module,
    input_shape: tuple,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Profile memory usage of DREAM model.

    Parameters
    ----------
    model : torch.nn.Module
        DREAM model
    input_shape : tuple
        Input tensor shape
    device : str
        Device to use

    Returns
    -------
    dict
        Memory usage statistics
    """
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}

    model = model.to(device)
    model.eval()

    # Reset peak memory
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    x = torch.randn(*input_shape, device=device)

    # Forward pass
    with torch.no_grad():
        _ = model(x)

    # Get memory stats
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
    memory_peak = torch.cuda.max_memory_allocated(device) / 1024**2  # MB

    return {
        'allocated_mb': memory_allocated,
        'reserved_mb': memory_reserved,
        'peak_mb': memory_peak,
    }


def get_optimization_recommendations(
    sequence_length: int,
    batch_size: int,
    device: str = 'cuda'
) -> List[str]:
    """
    Get optimization recommendations based on use case.

    Parameters
    ----------
    sequence_length : int
        Length of sequences to process
    batch_size : int
        Batch size
    device : str
        Target device

    Returns
    -------
    list
        List of recommendations
    """
    recommendations = []

    # Device-specific recommendations
    if device == 'cuda':
        recommendations.append("✓ Use CUDA for 10-50x speedup")
        recommendations.append("✓ Enable TF32 for faster matmul (Ampere GPUs)")
        recommendations.append("✓ Use mixed precision (AMP) for 2-3x speedup")
    else:
        recommendations.append("✓ Use CPU with OpenMP enabled")
        recommendations.append("✓ Reduce batch size for better cache usage")

    # Sequence length recommendations
    if sequence_length > 500:
        recommendations.append("✓ Use truncated BPTT with segment_size=100")
        recommendations.append("✓ Consider gradient checkpointing for memory")
    elif sequence_length > 1000:
        recommendations.append("✓ Use segment_size=50 for long sequences")

    # Batch size recommendations
    if batch_size > 32:
        recommendations.append("✓ Reduce batch size if OOM")
    elif batch_size < 8:
        recommendations.append("✓ Increase batch size for better GPU utilization")

    # General recommendations
    recommendations.append("✓ Use freeze_fast_weights=True during training")
    recommendations.append("✓ Pre-allocate tensors when possible")
    recommendations.append("✓ Use torch.compile() for additional speedup (PyTorch 2.0+)")

    return recommendations


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    print("DREAM Performance Utilities")
    print("=" * 60)

    config = DREAMConfig(
        input_dim=80,
        hidden_dim=256,
        rank=16,
    )

    # Benchmark
    print("\nRunning benchmarks...")
    results = compare_optimizations(
        config,
        input_shape=(4, 100, 80),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Memory profile
    if torch.cuda.is_available():
        print("\nMemory Profile:")
        model = DREAM(**config.__dict__)
        mem_stats = profile_memory(model, (4, 100, 80), 'cuda')
        print(f"  Allocated: {mem_stats['allocated_mb']:.1f} MB")
        print(f"  Reserved:  {mem_stats['reserved_mb']:.1f} MB")
        print(f"  Peak:      {mem_stats['peak_mb']:.1f} MB")

    # Recommendations
    print("\nOptimization Recommendations:")
    recs = get_optimization_recommendations(
        sequence_length=500,
        batch_size=32,
        device='cuda'
    )
    for rec in recs:
        print(f"  {rec}")
