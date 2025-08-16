#!/usr/bin/env python3
"""
TinyServe Example: Demonstrating Query-Aware Cache Selection for Efficient LLM Serving

This example shows how to use TinyServe to serve tiny language models with:
- Query-aware KV page selection
- Structured sparsity
- Plugin-based optimization
- Performance monitoring

Based on the paper: "TinyServe: Query-Aware Cache Selection for Efficient LLM Serving"
"""

import torch
import time
import uuid
from typing import List, Dict, Any

# Import TinyServe components
from vescale.tinyserve import (
    TinyServe, 
    TinyServeConfig, 
    TinyServeRequest,
    create_optimized_config_for_model
)


def create_sample_prompts() -> List[str]:
    """Create sample prompts for testing."""
    return [
        "The quick brown fox jumps over the lazy dog. Please continue this story:",
        "Explain the concept of machine learning in simple terms:",
        "Write a short poem about artificial intelligence:",
        "What are the benefits of renewable energy? Please elaborate:",
        "Describe the process of photosynthesis step by step:"
    ]


def benchmark_tinyserve(tinyserve: TinyServe, prompts: List[str], 
                       max_tokens: int = 100) -> Dict[str, Any]:
    """
    Benchmark TinyServe performance on multiple prompts.
    
    Args:
        tinyserve: TinyServe instance
        prompts: List of prompts to test
        max_tokens: Maximum tokens to generate per prompt
        
    Returns:
        Benchmark results
    """
        print(f"\n[BENCHMARK] Starting TinyServe benchmark with {len(prompts)} prompts...")
    
    results = {
        'total_requests': len(prompts),
        'total_tokens': 0,
        'total_latency_ms': 0.0,
        'total_memory_gb': 0.0,
        'responses': []
    }
    
    for i, prompt in enumerate(prompts):
        print(f"\n[PROCESS] Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        # Create request
        request = TinyServeRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            request_id=str(uuid.uuid4())
        )
        
        # Measure performance
        start_time = time.time()
        response = tinyserve.serve(request)
        end_time = time.time()
        
        # Record results
        results['total_tokens'] += len(response.tokens)
        results['total_latency_ms'] += response.latency_ms
        results['total_memory_gb'] += response.memory_usage_gb
        
        results['responses'].append({
            'prompt': prompt,
            'generated_text': response.generated_text,
            'tokens_generated': len(response.tokens),
            'latency_ms': response.latency_ms,
            'memory_gb': response.memory_usage_gb,
            'kv_hit_rate': response.kv_cache_hit_rate
        })
        
        print(f"   [SUCCESS] Generated {len(response.tokens)} tokens in {response.latency_ms:.2f}ms")
        print(f"   [MEMORY] Memory: {response.memory_usage_gb:.3f}GB, KV Hit: {response.kv_cache_hit_rate:.1%}")
    
    # Calculate averages
    results['avg_latency_ms'] = results['total_latency_ms'] / len(prompts)
    results['avg_memory_gb'] = results['total_memory_gb'] / len(prompts)
    results['avg_tokens_per_request'] = results['total_tokens'] / len(prompts)
    results['throughput_tokens_per_sec'] = results['total_tokens'] / (results['total_latency_ms'] / 1000)
    
    return results


def print_benchmark_results(results: Dict[str, Any]):
    """Print benchmark results in a formatted way."""
    print("\n" + "="*60)
    print("[RESULTS] TINYSERVE BENCHMARK RESULTS")
    print("="*60)
    
    print(f"[METRICS] Performance Metrics:")
    print(f"   • Total Requests: {results['total_requests']}")
    print(f"   • Total Tokens Generated: {results['total_tokens']}")
    print(f"   • Average Latency: {results['avg_latency_ms']:.2f}ms")
    print(f"   • Average Memory Usage: {results['avg_memory_gb']:.3f}GB")
    print(f"   • Average Tokens per Request: {results['avg_tokens_per_request']:.1f}")
    print(f"   • Throughput: {results['throughput_tokens_per_sec']:.1f} tokens/sec")
    
    print(f"\n[DETAILS] Detailed Results:")
    for i, response in enumerate(results['responses']):
        print(f"   Request {i+1}:")
        print(f"     Prompt: {response['prompt'][:50]}...")
        print(f"     Generated: {response['generated_text'][:100]}...")
        print(f"     Tokens: {response['tokens_generated']}, Latency: {response['latency_ms']:.2f}ms")
        print(f"     Memory: {response['memory_gb']:.3f}GB, KV Hit: {response['kv_hit_rate']:.1%}")


def demonstrate_plugin_system(tinyserve: TinyServe):
    """Demonstrate TinyServe's plugin system."""
    print("\n[PLUGINS] Demonstrating Plugin System...")
    
    # Get plugin status
    plugin_status = tinyserve.plugin_manager.get_plugin_status()
    print(f"   Active Plugins: {list(plugin_status.keys())}")
    
    # Show plugin configurations
    for plugin_name, status in plugin_status.items():
        config = tinyserve.plugin_manager.get_plugin_config(plugin_name)
        print(f"   {plugin_name}: {'✅ Enabled' if status['enabled'] else '❌ Disabled'}")
        print(f"     Config: {config}")
    
    # Get system statistics
    stats = tinyserve.get_stats()
    print(f"\n[STATS] System Statistics:")
    print(f"   Total Requests: {stats['total_requests']}")
    print(f"   Total Tokens: {stats['total_tokens']}")
    print(f"   Average Latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"   Average Memory: {stats['avg_memory_gb']:.3f}GB")
    print(f"   KV Hit Rate: {stats['kv_hit_rate']:.1%}")


def demonstrate_kv_optimization(tinyserve: TinyServe):
    """Demonstrate KV cache optimization features."""
    print("\n[KV_CACHE] Demonstrating KV Cache Optimization...")
    
    # Get page statistics
    page_stats = tinyserve.kv_retriever.get_page_statistics(tinyserve.page_metadata)
    print(f"   Page Statistics:")
    print(f"     Number of Pages: {page_stats['num_pages']}")
    print(f"     Total Tokens: {page_stats['total_tokens']}")
    print(f"     Average Page Size: {page_stats['avg_page_size']:.1f}")
    print(f"     Memory Usage: {page_stats['memory_usage_gb']:.3f}GB")
    
    # Get attention statistics
    attention_stats = tinyserve.attention_executor.get_attention_stats()
    print(f"\n   [ATTENTION] Attention Statistics:")
    print(f"     Total Attention Calls: {attention_stats['total_attention_calls']}")
    print(f"     Average Attention Time: {attention_stats['avg_attention_time_ms']:.2f}ms")
    print(f"     Total Sparse Operations: {attention_stats['total_sparse_operations']}")


def main():
    """Main function demonstrating TinyServe capabilities."""
    print("🎯 TinyServe: Query-Aware Cache Selection for Efficient LLM Serving")
    print("="*70)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"[CUDA] CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("[WARNING] CUDA not available, using CPU (performance may be limited)")
    
    # Create optimized configuration for TinyLLaMA
    print("\n[CONFIG] Creating optimized configuration for TinyLLaMA...")
    config = create_optimized_config_for_model(
        model_name="tinylama",
        target_latency_ms=50.0,
        target_memory_gb=4.0
    )
    
    print(f"   Page Size: {config.page_size}")
    print(f"   Selection Ratio: {config.selection_ratio}")
    print(f"   Target Latency: {config.target_latency_ms}ms")
    print(f"   Target Memory: {config.target_memory_gb}GB")
    
    # Initialize TinyServe
    print("\n[INIT] Initializing TinyServe...")
    tinyserve = TinyServe(config)
    
    # Load model (this would require actual model files)
    print("\n[MODEL] Loading model (simulated)...")
    try:
        # In a real scenario, you would load an actual model
        # tinyserve.load_model("microsoft/DialoGPT-small", "gpt2")
        print("   [SUCCESS] Model loaded successfully (simulated)")
    except Exception as e:
        print(f"   [WARNING] Model loading failed: {e}")
        print("   Continuing with demonstration...")
    
    # Create sample prompts
    prompts = create_sample_prompts()
    
    # Run benchmark
    results = benchmark_tinyserve(tinyserve, prompts, max_tokens=50)
    
    # Print results
    print_benchmark_results(results)
    
    # Demonstrate plugin system
    demonstrate_plugin_system(tinyserve)
    
    # Demonstrate KV optimization
    demonstrate_kv_optimization(tinyserve)
    
    # Show configuration optimization
    print("\n[OPTIMIZATION] Demonstrating Configuration Optimization...")
    current_performance = {
        'latency_ms': results['avg_latency_ms'],
        'memory_gb': results['avg_memory_gb']
    }
    
    optimized_config = config.get_optimized_config(current_performance)
    print(f"   Original Page Size: {config.page_size}")
    print(f"   Optimized Page Size: {optimized_config.page_size}")
    print(f"   Original Selection Ratio: {config.selection_ratio}")
    print(f"   Optimized Selection Ratio: {optimized_config.selection_ratio}")
    
    # Cleanup
    print("\n[CLEANUP] Cleaning up...")
    tinyserve.clear_cache()
    
    print("\n[SUCCESS] TinyServe demonstration completed!")
    print("\n[FEATURES] Key Features Demonstrated:")
    print("   • Query-aware KV page selection")
    print("   • Structured sparsity with bounding-box metadata")
    print("   • Plugin-based optimization system")
    print("   • Performance monitoring and statistics")
    print("   • Dynamic configuration optimization")
    print("   • Multi-component architecture")


if __name__ == "__main__":
    main()
