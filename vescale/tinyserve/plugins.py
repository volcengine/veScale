"""
Plugin Manager for TinyServe supporting various optimization plugins.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Callable
import math
import time


class PluginManager:
    """
    Manages various plugins for TinyServe including:
    - Entropy-based early exit
    - Token-level pruning
    - Approximate attention
    - Cache optimization
    """
    
    def __init__(self, config):
        """
        Initialize the plugin manager.
        
        Args:
            config: Configuration containing plugin parameters
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Plugin registry
        self.plugins = {}
        self.enabled_plugins = set()
        
        # Plugin configurations
        self.plugin_configs = {
            'entropy_early_exit': {
                'enabled': getattr(config, 'enable_entropy_early_exit', True),
                'threshold': getattr(config, 'entropy_threshold', 0.5),
                'min_tokens': getattr(config, 'min_tokens_before_exit', 10)
            },
            'token_pruning': {
                'enabled': getattr(config, 'enable_token_pruning', True),
                'pruning_ratio': getattr(config, 'pruning_ratio', 0.1),
                'min_tokens': getattr(config, 'min_tokens_after_pruning', 100)
            },
            'approximate_attention': {
                'enabled': getattr(config, 'enable_approximate_attention', False),
                'approximation_method': getattr(config, 'approximation_method', 'linear'),
                'compression_ratio': getattr(config, 'compression_ratio', 0.5)
            },
            'cache_optimization': {
                'enabled': getattr(config, 'enable_cache_optimization', True),
                'eviction_policy': getattr(config, 'eviction_policy', 'lru'),
                'max_cache_size_gb': getattr(config, 'max_cache_size_gb', 8.0)
            }
        }
        
        # Initialize default plugins
        self._init_default_plugins()
        
        # Performance tracking
        self.plugin_stats = {
            'total_plugin_calls': 0,
            'plugin_success_count': 0,
            'plugin_error_count': 0,
            'early_exit_count': 0,
            'pruning_count': 0
        }
    
    def _init_default_plugins(self):
        """Initialize default plugins."""
        # Entropy-based early exit plugin
        self.register_plugin('entropy_early_exit', self._entropy_early_exit_plugin)
        
        # Token pruning plugin
        self.register_plugin('token_pruning', self._token_pruning_plugin)
        
        # Approximate attention plugin
        self.register_plugin('approximate_attention', self._approximate_attention_plugin)
        
        # Cache optimization plugin
        self.register_plugin('cache_optimization', self._cache_optimization_plugin)
    
    def register_plugin(self, name: str, plugin_func: Callable):
        """
        Register a new plugin.
        
        Args:
            name: Plugin name
            plugin_func: Plugin function to execute
        """
        self.plugins[name] = plugin_func
        
        # Enable if configured to be enabled
        if self.plugin_configs.get(name, {}).get('enabled', False):
            self.enable_plugin(name)
    
    def enable_plugin(self, name: str):
        """Enable a specific plugin."""
        if name in self.plugins:
            self.enabled_plugins.add(name)
    
    def disable_plugin(self, name: str):
        """Disable a specific plugin."""
        if name in self.plugins:
            self.enabled_plugins.discard(name)
    
    def should_stop_early(self, generated_tokens: List[int], attention_output: torch.Tensor) -> bool:
        """
        Check if generation should stop early based on plugin logic.
        
        Args:
            generated_tokens: List of generated tokens so far
            attention_output: Current attention output
            
        Returns:
            True if generation should stop early
        """
        if 'entropy_early_exit' not in self.enabled_plugins:
            return False
        
        try:
            return self._entropy_early_exit_plugin({
                'generated_tokens': generated_tokens,
                'attention_output': attention_output
            })
        except Exception as e:
            print(f"Error in entropy early exit plugin: {e}")
            return False
    
    def _entropy_early_exit_plugin(self, context: Dict[str, Any]) -> bool:
        """
        Entropy-based early exit plugin.
        
        Stops generation when the entropy of the attention distribution is below a threshold,
        indicating the model is confident in its predictions.
        """
        config = self.plugin_configs['entropy_early_exit']
        if not config['enabled']:
            return False
        
        generated_tokens = context.get('generated_tokens', [])
        attention_output = context.get('attention_output', None)
        
        # Check minimum token requirement
        if len(generated_tokens) < config['min_tokens']:
            return False
        
        if attention_output is not None:
            # Calculate entropy of attention distribution
            attention_probs = F.softmax(attention_output, dim=-1)
            entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8))
            
            # Stop if entropy is below threshold
            if entropy < config['threshold']:
                self.plugin_stats['early_exit_count'] += 1
                return True
        
        return False
    
    def _token_pruning_plugin(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Token-level pruning plugin.
        
        Removes low-importance tokens from the KV cache to reduce memory usage
        while maintaining generation quality.
        """
        config = self.plugin_configs['token_pruning']
        if not config['enabled']:
            return context
        
        # Implementation would prune tokens based on importance scores
        # For now, return context unchanged
        self.plugin_stats['pruning_count'] += 1
        return context
    
    def _approximate_attention_plugin(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Approximate attention plugin.
        
        Uses approximation methods to reduce attention computation cost
        while maintaining reasonable accuracy.
        """
        config = self.plugin_configs['approximate_attention']
        if not config['enabled']:
            return context
        
        # Implementation would apply attention approximation
        # For now, return context unchanged
        return context
    
    def _cache_optimization_plugin(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cache optimization plugin.
        
        Optimizes KV cache usage through intelligent eviction policies
        and memory management.
        """
        config = self.plugin_configs['cache_optimization']
        if not config['enabled']:
            return context
        
        # Implementation would optimize cache usage
        # For now, return context unchanged
        return context
    
    def execute_plugin_pipeline(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all enabled plugins in sequence.
        
        Args:
            context: Context data for plugins
            
        Returns:
            Updated context after plugin execution
        """
        self.plugin_stats['total_plugin_calls'] += 1
        
        try:
            result = context.copy()
            
            for plugin_name in self.enabled_plugins:
                if plugin_name in self.plugins:
                    try:
                        result = self.plugins[plugin_name](result)
                        if result is None:
                            # Plugin requested to stop processing
                            break
                    except Exception as e:
                        print(f"Plugin {plugin_name} error: {e}")
                        self.plugin_stats['plugin_error_count'] += 1
                        continue
            
            self.plugin_stats['plugin_success_count'] += 1
            return result
            
        except Exception as e:
            print(f"Error in plugin pipeline: {e}")
            self.plugin_stats['plugin_error_count'] += 1
            return context
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get configuration for a specific plugin."""
        return self.plugin_configs.get(plugin_name, {}).copy()
    
    def update_plugin_config(self, plugin_name: str, config_updates: Dict[str, Any]):
        """
        Update configuration for a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            config_updates: Configuration updates to apply
        """
        if plugin_name in self.plugin_configs:
            self.plugin_configs[plugin_name].update(config_updates)
    
    def get_plugin_stats(self) -> Dict[str, Any]:
        """Get current plugin statistics."""
        return self.plugin_stats.copy()
    
    def get_enabled_plugins(self) -> List[str]:
        """Get list of currently enabled plugins."""
        return list(self.enabled_plugins)
    
    def get_all_plugins(self) -> List[str]:
        """Get list of all registered plugins."""
        return list(self.plugins.keys())
    
    def clear_stats(self):
        """Clear plugin statistics."""
        self.plugin_stats = {
            'total_plugin_calls': 0,
            'plugin_success_count': 0,
            'plugin_error_count': 0,
            'early_exit_count': 0,
            'pruning_count': 0
        }
    
    def benchmark_plugin(self, plugin_name: str, input_data: Any) -> Dict[str, float]:
        """
        Benchmark a specific plugin's performance.
        
        Args:
            plugin_name: Name of the plugin to benchmark
            input_data: Input data for the plugin
            
        Returns:
            Benchmark results
        """
        if plugin_name not in self.plugins:
            return {'error': 'Plugin not found'}
        
        plugin_func = self.plugins[plugin_name]
        
        # Warmup
        for _ in range(10):
            try:
                _ = plugin_func(input_data)
            except:
                pass
        
        # Benchmark
        start_time = time.time()
        iterations = 100
        
        for _ in range(iterations):
            try:
                _ = plugin_func(input_data)
            except:
                pass
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_ms = (total_time / iterations) * 1000
        
        return {
            'avg_execution_time_ms': avg_time_ms,
            'throughput_ops_per_sec': iterations / total_time
        }
    
    def optimize_plugin_configs(self, performance_metrics: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """
        Dynamically optimize plugin configurations based on performance metrics.
        
        Args:
            performance_metrics: Current performance metrics
            
        Returns:
            Optimized plugin configurations
        """
        optimizations = {}
        
        current_latency = performance_metrics.get('latency_ms', 0)
        current_memory = performance_metrics.get('memory_gb', 0)
        
        # Optimize entropy early exit
        if current_latency > self.config.target_latency_ms:
            optimizations['entropy_early_exit'] = {
                'threshold': min(0.8, self.plugin_configs['entropy_early_exit']['threshold'] + 0.1),
                'min_tokens': max(5, self.plugin_configs['entropy_early_exit']['min_tokens'] - 2)
            }
        
        # Optimize token pruning
        if current_memory > self.config.target_memory_gb:
            optimizations['token_pruning'] = {
                'pruning_ratio': min(0.3, self.plugin_configs['token_pruning']['pruning_ratio'] + 0.05)
            }
        
        return optimizations
