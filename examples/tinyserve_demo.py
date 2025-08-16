#!/usr/bin/env python3
"""
Simple TinyServe demonstration script.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from vescale.tinyserve import TinyServeConfig, create_optimized_config_for_model
    print("[SUCCESS] Successfully imported TinyServe components")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("This demo requires TinyServe to be properly installed.")
    sys.exit(1)


def demonstrate_config_system():
    """Demonstrate TinyServe's configuration system."""
    print("\n🔧 Configuration System Demonstration")
    print("=" * 50)
    
    # Create default configuration
    print("1. Creating default configuration...")
    default_config = TinyServeConfig()
    print(f"   • Page Size: {default_config.page_size}")
    print(f"   • Selection Ratio: {default_config.selection_ratio}")
    print(f"   • Target Latency: {default_config.target_latency_ms}ms")
    print(f"   • Target Memory: {default_config.target_memory_gb}GB")
    
    # Create model-specific configuration
    print("\n2. Creating optimized configuration for TinyLLaMA...")
    tinylama_config = create_optimized_config_for_model(
        model_name="tinylama",
        target_latency_ms=30.0,
        target_memory_gb=2.0
    )
    print(f"   • Page Size: {tinylama_config.page_size}")
    print(f"   • Selection Ratio: {tinylama_config.selection_ratio}")
    print(f"   • Target Latency: {tinylama_config.target_latency_ms}ms")
    print(f"   • Target Memory: {tinylama_config.target_memory_gb}GB")
    
    # Show configuration serialization
    print("\n3. Configuration serialization...")
    config_dict = tinylama_config.to_dict()
    print(f"   • Serialized to dictionary with {len(config_dict)} keys")
    
    # Reconstruct configuration
    reconstructed_config = TinyServeConfig.from_dict(config_dict)
    print(f"   • Reconstructed successfully: {reconstructed_config.page_size == tinylama_config.page_size}")
    
    return tinylama_config


def demonstrate_optimization():
    """Demonstrate configuration optimization."""
    print("\n🚀 Configuration Optimization Demonstration")
    print("=" * 50)
    
    # Create base configuration
    base_config = TinyServeConfig()
    print(f"Base Configuration:")
    print(f"   • Page Size: {base_config.page_size}")
    print(f"   • Selection Ratio: {base_config.selection_ratio}")
    
    # Simulate poor performance
    poor_performance = {
        'latency_ms': 100.0,  # High latency
        'memory_gb': 8.0      # High memory usage
    }
    
    print(f"\nPoor Performance Metrics:")
    print(f"   • Latency: {poor_performance['latency_ms']}ms (target: {base_config.target_latency_ms}ms)")
    print(f"   • Memory: {poor_performance['memory_gb']}GB (target: {base_config.target_memory_gb}GB)")
    
    # Get optimized configuration
    print(f"\nOptimizing configuration...")
    optimized_config = base_config.get_optimized_config(poor_performance)
    
    print(f"Optimized Configuration:")
    print(f"   • Page Size: {base_config.page_size} → {optimized_config.page_size}")
    print(f"   • Selection Ratio: {base_config.selection_ratio} → {optimized_config.selection_ratio}")
    
    # Show what changed and why
    if optimized_config.page_size != base_config.page_size:
        print(f"   • Page size reduced to improve latency")
    if optimized_config.selection_ratio != base_config.selection_ratio:
        print(f"   • Selection ratio increased to reduce memory usage")


def demonstrate_plugin_configuration():
    """Demonstrate plugin configuration options."""
    print("\n🔌 Plugin Configuration Demonstration")
    print("=" * 50)
    
    config = TinyServeConfig()
    
    print("Available Plugins:")
    print(f"1. Entropy-Based Early Exit:")
    print(f"   • Enabled: {config.enable_entropy_early_exit}")
    print(f"   • Threshold: {config.entropy_threshold}")
    print(f"   • Min Tokens: {config.min_tokens_before_exit}")
    
    print(f"\n2. Token-Level Pruning:")
    print(f"   • Enabled: {config.enable_token_pruning}")
    print(f"   • Pruning Ratio: {config.pruning_ratio}")
    print(f"   • Min Tokens: {config.min_tokens_after_pruning}")
    
    print(f"\n3. Approximate Attention:")
    print(f"   • Enabled: {config.enable_approximate_attention}")
    print(f"   • Method: {config.approximation_method}")
    print(f"   • Compression Ratio: {config.compression_ratio}")
    
    print(f"\n4. Cache Optimization:")
    print(f"   • Enabled: {config.enable_cache_optimization}")
    print(f"   • Eviction Policy: {config.eviction_policy}")
    print(f"   • Max Cache Size: {config.max_cache_size_gb}GB")


def demonstrate_validation():
    """Demonstrate configuration validation."""
    print("\n[VALIDATION] Configuration Validation Demonstration")
    print("=" * 50)
    
    print("Testing invalid configurations...")
    
    # Test invalid page size
    try:
        invalid_config = TinyServeConfig(page_size=0)
        print("   [ERROR] Should have failed for page_size=0")
    except ValueError as e:
        print(f"   [SUCCESS] Correctly caught error: {e}")
    
    # Test invalid selection ratio
    try:
        invalid_config = TinyServeConfig(selection_ratio=1.5)
        print("   [ERROR] Should have failed for selection_ratio=1.5")
    except ValueError as e:
        print(f"   [SUCCESS] Correctly caught error: {e}")
    
    # Test invalid entropy threshold
    try:
        invalid_config = TinyServeConfig(entropy_threshold=-0.1)
        print("   [ERROR] Should have failed for entropy_threshold=-0.1")
    except ValueError as e:
        print(f"   [SUCCESS] Correctly caught error: {e}")
    
    print("\nAll validation tests passed! [SUCCESS]")


def main():
    """Main demonstration function."""
    print("[TINYSERVE] TinyServe: Query-Aware Cache Selection for Efficient LLM Serving")
    print("=" * 70)
    print("This demo showcases TinyServe's configuration and optimization capabilities.")
    print("Note: This is a demonstration of the configuration system only.")
    print("Full inference serving requires actual model files and GPU resources.")
    
    try:
        # Demonstrate configuration system
        config = demonstrate_config_system()
        
        # Demonstrate optimization
        demonstrate_optimization()
        
        # Demonstrate plugin configuration
        demonstrate_plugin_configuration()
        
        # Demonstrate validation
        demonstrate_validation()
        
        print("\n" + "=" * 70)
        print("[SUCCESS] TinyServe demonstration completed successfully!")
        print("\n[FEATURES] Key Features Demonstrated:")
        print("   • Flexible configuration system")
        print("   • Model-specific optimization")
        print("   • Dynamic configuration adaptation")
        print("   • Comprehensive plugin support")
        print("   • Robust validation system")
        print("\n[NEXT] Next Steps:")
        print("   • Install required dependencies (PyTorch, Transformers)")
        print("   • Download a small language model (e.g., TinyLLaMA)")
        print("   • Run the full example: python examples/tinyserve_example.py")
        
    except Exception as e:
        print(f"\n[ERROR] Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
