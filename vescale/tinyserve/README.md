# TinyServe: Query-Aware Cache Selection for Efficient LLM Serving

TinyServe is a lightweight and extensible runtime system for deploying tiny language models (e.g., TinyLLaMA, GPT2-345M) with support for structured KV sparsity, plugin-based token selection, and hardware-efficient attention kernels.

## Overview

Based on the research paper "TinyServe: Query-Aware Cache Selection for Efficient LLM Serving" (Liu & Yu, 2025), TinyServe enables efficient inference serving at small scale while maintaining the interpretability and control needed for systems research.

### Key Features

- **Query-Aware KV Selection**: Dynamically selects relevant key-value blocks based on current query vectors
- **Structured Sparsity**: Uses bounding-box metadata for efficient page-level relevance estimation
- **Plugin Architecture**: Modular system supporting entropy-based early exit, token pruning, and more
- **Fused CUDA Kernels**: Hardware-efficient attention computation with minimal memory movement
- **Multi-GPU Support**: Scalable deployment across multiple GPUs
- **Performance Monitoring**: Comprehensive metrics and optimization suggestions

## Architecture

TinyServe is organized around three core components:

### 1. Query-Aware KV Retriever
- Dynamically selects relevant key-value blocks at decode time
- Uses lightweight metadata (channel-wise min/max values) for relevance estimation
- Enables efficient selection of top-K pages with minimal overhead

### 2. Modular Scheduling Pipeline
- Handles incoming queries and routes them through configurable plugins
- Supports different sparsity strategies without modifying core models
- Manages session state and request prioritization

### 3. Sparse Attention Executor
- Efficiently computes attention over selected KV pages
- Fused CUDA kernels for page scoring, sparse memory access, and masked attention
- Support for FP16/INT8 KV formats

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd TinyServe

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from vescale.tinyserve import TinyServe, TinyServeConfig, TinyServeRequest

# Create configuration
config = TinyServeConfig(
    page_size=16,
    selection_ratio=0.3,
    target_latency_ms=50.0,
    target_memory_gb=4.0
)

# Initialize TinyServe
tinyserve = TinyServe(config)

# Load a model
tinyserve.load_model("microsoft/DialoGPT-small", "gpt2")

# Create a request
request = TinyServeRequest(
    prompt="Explain machine learning in simple terms:",
    max_tokens=100,
    temperature=0.7
)

# Serve the request
response = tinyserve.serve(request)
print(response.generated_text)
```

### Configuration

TinyServe supports extensive configuration options:

```python
config = TinyServeConfig(
    # Page-based KV cache
    page_size=16,                    # Tokens per page
    selection_ratio=0.3,             # Top-K ratio for selection
    
    # Attention optimization
    use_fused_kernel=True,           # Enable fused CUDA kernels
    attention_chunk_size=256,        # Chunk size for attention
    
    # Plugin configurations
    enable_entropy_early_exit=True,  # Early stopping based on entropy
    entropy_threshold=0.5,           # Entropy threshold for early exit
    enable_token_pruning=True,       # Enable token-level pruning
    pruning_ratio=0.1,               # Fraction of tokens to prune
    
    # Performance targets
    target_latency_ms=50.0,          # Target latency in milliseconds
    target_memory_gb=4.0             # Target memory usage in GB
)
```

## Plugin System

TinyServe includes several built-in plugins:

### Entropy-Based Early Exit
Stops generation when attention entropy is below a threshold, indicating high confidence:

```python
# Configure early exit
config.enable_entropy_early_exit = True
config.entropy_threshold = 0.5
config.min_tokens_before_exit = 10
```

### Token-Level Pruning
Removes low-importance tokens from KV cache to reduce memory usage:

```python
# Configure token pruning
config.enable_token_pruning = True
config.pruning_ratio = 0.1
config.min_tokens_after_pruning = 100
```

### Custom Plugins
Register custom plugins for specialized optimization:

```python
def custom_optimization_plugin(context):
    # Custom optimization logic
    return modified_context

tinyserve.plugin_manager.register_plugin('custom_opt', custom_optimization_plugin)
```

## Performance Monitoring

TinyServe provides comprehensive performance metrics:

```python
# Get system statistics
stats = tinyserve.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Average latency: {stats['avg_latency_ms']:.2f}ms")
print(f"KV hit rate: {stats['kv_hit_rate']:.1%}")

# Get attention statistics
attention_stats = tinyserve.attention_executor.get_attention_stats()
print(f"Attention calls: {attention_stats['total_attention_calls']}")

# Get plugin statistics
plugin_stats = tinyserve.plugin_manager.get_plugin_stats()
print(f"Early exits: {plugin_stats['early_exit_count']}")
```

## Advanced Features

### Multi-GPU Deployment

```python
config = TinyServeConfig(
    num_gpus=4,
    gpu_ids=[0, 1, 2, 3]
)
```

### Dynamic Configuration Optimization

```python
# Get optimized configuration based on performance
current_performance = {
    'latency_ms': 75.0,
    'memory_gb': 6.0
}
optimized_config = config.get_optimized_config(current_performance)
```

### Session Management

```python
# Get session information
sessions = tinyserve.scheduler.list_sessions()
for session_id in sessions:
    info = tinyserve.scheduler.get_session_info(session_id)
    print(f"Session {session_id}: {info['request_count']} requests")
```

## Benchmarking

TinyServe includes built-in benchmarking capabilities:

```python
# Benchmark attention performance
benchmark_results = tinyserve.attention_executor.benchmark_attention(
    input_size=2048,
    num_pages=128
)

print(f"Throughput: {benchmark_results['throughput_tokens_per_ms']:.1f} tokens/ms")
```

## Research Applications

TinyServe is designed for LLM inference research:

- **Sparsity Analysis**: Study different sparsity patterns and their impact
- **Cache Behavior**: Analyze KV cache reuse and eviction patterns
- **Attention Optimization**: Experiment with attention approximation methods
- **System Design**: Test new serving architectures without full-scale deployment

## Paper Reference

This implementation is based on:

```
Liu, D., & Yu, Y. (2025). TinyServe: Query-Aware Cache Selection for Efficient LLM Serving. 
In Proceedings of the 33rd ACM International Conference on Multimedia (MM '25).
```