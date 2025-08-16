"""
Configuration and utility functions for TinyServe.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import os
import json


@dataclass
class TinyServeConfig:
    """
    Configuration class for TinyServe.
    
    Contains all the parameters described in the paper for:
    - Query-aware KV selection
    - Page-based memory management
    - Plugin configurations
    - Performance targets
    """
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Page-based KV cache configuration
    page_size: int = 16  # Tokens per page (from paper: tested 4, 8, 16, 32, 64)
    selection_ratio: float = 0.3  # Top-K ratio for page selection (from paper: tested 0.1, 0.2, 0.3, 0.5)
    
    # Attention configuration
    num_attention_heads: int = 12
    head_dim: int = 64
    use_fused_kernel: bool = True
    attention_chunk_size: int = 256
    
    # Plugin configurations
    enable_entropy_early_exit: bool = True
    entropy_threshold: float = 0.5
    min_tokens_before_exit: int = 10
    
    enable_token_pruning: bool = True
    pruning_ratio: float = 0.1
    min_tokens_after_pruning: int = 100
    
    enable_approximate_attention: bool = False
    approximation_method: str = "linear"
    compression_ratio: float = 0.5
    
    enable_cache_optimization: bool = True
    eviction_policy: str = "lru"
    max_cache_size_gb: float = 8.0
    
    # Performance targets
    target_latency_ms: float = 50.0
    target_memory_gb: float = 4.0
    
    # Session management
    session_timeout: float = 300.0  # 5 minutes
    
    # Multi-GPU configuration
    num_gpus: int = 1
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    
    # Memory management
    max_sequence_length: int = 8192
    kv_cache_dtype: str = "float16"
    
    # Logging and monitoring
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_profiling: bool = False
    
    # Model-specific configurations
    model_type: str = "auto"  # tinylama, gpt2, opt, auto
    model_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        
        if not 0.0 < self.selection_ratio <= 1.0:
            raise ValueError("selection_ratio must be between 0 and 1")
        
        if self.entropy_threshold < 0.0:
            raise ValueError("entropy_threshold must be non-negative")
        
        if self.pruning_ratio < 0.0 or self.pruning_ratio > 1.0:
            raise ValueError("pruning_ratio must be between 0 and 1")
        
        if self.max_cache_size_gb <= 0.0:
            raise ValueError("max_cache_size_gb must be positive")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TinyServeConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json_file(cls, file_path: str) -> 'TinyServeConfig':
        """Load configuration from JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'device': self.device,
            'page_size': self.page_size,
            'selection_ratio': self.selection_ratio,
            'num_attention_heads': self.num_attention_heads,
            'head_dim': self.head_dim,
            'use_fused_kernel': self.use_fused_kernel,
            'attention_chunk_size': self.attention_chunk_size,
            'enable_entropy_early_exit': self.enable_entropy_early_exit,
            'entropy_threshold': self.entropy_threshold,
            'min_tokens_before_exit': self.min_tokens_before_exit,
            'enable_token_pruning': self.enable_token_pruning,
            'pruning_ratio': self.pruning_ratio,
            'min_tokens_after_pruning': self.min_tokens_after_pruning,
            'enable_approximate_attention': self.enable_approximate_attention,
            'approximation_method': self.approximation_method,
            'compression_ratio': self.compression_ratio,
            'enable_cache_optimization': self.enable_cache_optimization,
            'eviction_policy': self.eviction_policy,
            'max_cache_size_gb': self.max_cache_size_gb,
            'target_latency_ms': self.target_latency_ms,
            'target_memory_gb': self.target_memory_gb,
            'session_timeout': self.session_timeout,
            'num_gpus': self.num_gpus,
            'gpu_ids': self.gpu_ids,
            'max_sequence_length': self.max_sequence_length,
            'kv_cache_dtype': self.kv_cache_dtype,
            'enable_logging': self.enable_logging,
            'log_level': self.log_level,
            'enable_profiling': self.enable_profiling,
            'model_type': self.model_type,
            'model_path': self.model_path
        }
    
    def save_to_json(self, file_path: str):
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration key: {key}")
        
        # Re-validate after updates
        self.__post_init__()
    
    def get_optimized_config(self, performance_metrics: Dict[str, float]) -> 'TinyServeConfig':
        """
        Get optimized configuration based on performance metrics.
        
        Args:
            performance_metrics: Current performance metrics
            
        Returns:
            Optimized configuration
        """
        current_latency = performance_metrics.get('latency_ms', 0)
        current_memory = performance_metrics.get('memory_gb', 0)
        
        # Create a copy for optimization
        optimized = TinyServeConfig.from_dict(self.to_dict())
        
        # Optimize page size based on latency
        if current_latency > self.target_latency_ms:
            optimized.page_size = max(4, self.page_size // 2)
        
        # Optimize selection ratio based on memory
        if current_memory > self.target_memory_gb:
            optimized.selection_ratio = min(0.5, self.selection_ratio + 0.1)
        
        # Optimize plugin configurations
        if current_latency > self.target_latency_ms:
            optimized.enable_entropy_early_exit = True
            optimized.entropy_threshold = min(0.8, self.entropy_threshold + 0.1)
        
        if current_memory > self.target_memory_gb:
            optimized.enable_token_pruning = True
            optimized.pruning_ratio = min(0.3, self.pruning_ratio + 0.05)
        
        return optimized
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'max_sequence_length': self.max_sequence_length,
            'kv_cache_dtype': self.kv_cache_dtype
        }
    
    def get_plugin_config(self) -> Dict[str, Any]:
        """Get plugin configuration."""
        return {
            'entropy_early_exit': {
                'enabled': self.enable_entropy_early_exit,
                'threshold': self.entropy_threshold,
                'min_tokens': self.min_tokens_before_exit
            },
            'token_pruning': {
                'enabled': self.enable_token_pruning,
                'pruning_ratio': self.pruning_ratio,
                'min_tokens': self.min_tokens_after_pruning
            },
            'approximate_attention': {
                'enabled': self.enable_approximate_attention,
                'method': self.approximation_method,
                'compression_ratio': self.compression_ratio
            },
            'cache_optimization': {
                'enabled': self.enable_cache_optimization,
                'eviction_policy': self.eviction_policy,
                'max_cache_size_gb': self.max_cache_size_gb
            }
        }


def create_default_config() -> TinyServeConfig:
    """Create a default TinyServe configuration."""
    return TinyServeConfig()


def create_optimized_config_for_model(model_name: str, 
                                    target_latency_ms: float = 50.0,
                                    target_memory_gb: float = 4.0) -> TinyServeConfig:
    """
    Create an optimized configuration for a specific model.
    
    Args:
        model_name: Name of the model (e.g., "tinylama", "gpt2", "opt")
        target_latency_ms: Target latency in milliseconds
        target_memory_gb: Target memory usage in GB
        
    Returns:
        Optimized configuration
    """
    config = TinyServeConfig()
    
    # Model-specific optimizations
    if "tinylama" in model_name.lower():
        config.page_size = 16
        config.selection_ratio = 0.3
        config.num_attention_heads = 12
        config.head_dim = 64
    elif "gpt2" in model_name.lower():
        config.page_size = 32
        config.selection_ratio = 0.2
        config.num_attention_heads = 12
        config.head_dim = 64
    elif "opt" in model_name.lower():
        config.page_size = 16
        config.selection_ratio = 0.25
        config.num_attention_heads = 16
        config.head_dim = 64
    
    # Performance targets
    config.target_latency_ms = target_latency_ms
    config.target_memory_gb = target_memory_gb
    
    return config


def validate_config(config: TinyServeConfig) -> List[str]:
    """
    Validate configuration and return list of warnings/errors.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation messages
    """
    warnings = []
    
    # Check device availability
    if config.device == "cuda" and not torch.cuda.is_available():
        warnings.append("CUDA device requested but not available")
    
    # Check memory constraints
    if config.max_cache_size_gb > 32:
        warnings.append("Very large cache size may cause memory issues")
    
    # Check performance targets
    if config.target_latency_ms < 10:
        warnings.append("Very low latency target may be unrealistic")
    
    if config.target_memory_gb < 1:
        warnings.append("Very low memory target may cause issues")
    
    # Check plugin configurations
    if config.enable_entropy_early_exit and config.entropy_threshold < 0.1:
        warnings.append("Very low entropy threshold may cause premature stopping")
    
    if config.enable_token_pruning and config.pruning_ratio > 0.5:
        warnings.append("High pruning ratio may significantly impact quality")
    
    return warnings
