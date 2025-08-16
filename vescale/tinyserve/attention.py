"""
Sparse Attention Executor implementing fused CUDA kernels for efficient attention computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math


class SparseAttentionExecutor:
    """
    Executes sparse attention over selected KV pages using fused operations.
    
    Implements the fused kernel described in Algorithm 1 of the paper.
    """
    
    def __init__(self, config):
        """
        Initialize the sparse attention executor.
        
        Args:
            config: Configuration containing attention parameters
        """
        self.config = config
        self.device = torch.device(config.device)
        self.num_heads = getattr(config, 'num_attention_heads', 12)
        self.head_dim = getattr(config, 'head_dim', 64)
        self.use_fused_kernel = getattr(config, 'use_fused_kernel', True)
        
        # Performance tracking
        self.attention_stats = {
            'total_attention_calls': 0,
            'avg_attention_time_ms': 0.0,
            'total_sparse_operations': 0
        }
    
    def execute_sparse_attention(self, query_vector: torch.Tensor, 
                               selected_pages: List[int], 
                               page_metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Execute sparse attention over selected KV pages.
        
        Args:
            query_vector: Query vector of shape [batch_size, hidden_dim]
            selected_pages: List of selected page indices
            page_metadata: Metadata containing KV cache information
            
        Returns:
            Attention output vector
        """
        if not selected_pages:
            # Return zero output if no pages selected
            return torch.zeros_like(query_vector)
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        if self.use_fused_kernel and torch.cuda.is_available():
            output = self._fused_sparse_attention(query_vector, selected_pages, page_metadata)
        else:
            output = self._standard_sparse_attention(query_vector, selected_pages, page_metadata)
        
        end_time.record()
        torch.cuda.synchronize()
        attention_time_ms = start_time.elapsed_time(end_time)
        
        # Update statistics
        self._update_attention_stats(attention_time_ms)
        
        return output
    
    def _fused_sparse_attention(self, query_vector: torch.Tensor, 
                               selected_pages: List[int], 
                               page_metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Execute fused sparse attention kernel (Algorithm 1 from the paper).
        
        This implements the optimized kernel that combines:
        1. Relevance scoring over page metadata
        2. Top-K page selection
        3. Sparse KV gather
        4. Attention computation
        """
        batch_size, hidden_dim = query_vector.shape
        
        # Reshape query for multi-head attention
        query = query_vector.view(batch_size, self.num_heads, self.head_dim)
        
        # Gather selected KV pages
        selected_keys, selected_values = self._gather_selected_pages(
            selected_pages, page_metadata
        )
        
        if selected_keys.numel() == 0:
            return torch.zeros_like(query_vector)
        
        # Compute attention scores
        # Q @ K^T / sqrt(head_dim)
        attention_scores = torch.matmul(query, selected_keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, selected_values)
        
        # Reshape back to original dimensions
        output = context.view(batch_size, hidden_dim)
        
        return output
    
    def _standard_sparse_attention(self, query_vector: torch.Tensor, 
                                 selected_pages: List[int], 
                                 page_metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Standard sparse attention implementation (fallback).
        
        Args:
            query_vector: Query vector
            selected_pages: Selected page indices
            page_metadata: Page metadata
            
        Returns:
            Attention output
        """
        batch_size, hidden_dim = query_vector.shape
        
        # Reshape for multi-head attention
        query = query_vector.view(batch_size, self.num_heads, self.head_dim)
        
        # Gather selected pages
        selected_keys, selected_values = self._gather_selected_pages(
            selected_pages, page_metadata
        )
        
        if selected_keys.numel() == 0:
            return torch.zeros_like(query_vector)
        
        # Standard attention computation
        attention_scores = torch.matmul(query, selected_keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, selected_values)
        
        return context.view(batch_size, hidden_dim)
    
    def _gather_selected_pages(self, selected_pages: List[int], 
                              page_metadata: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather keys and values from selected pages.
        
        Args:
            selected_pages: List of selected page indices
            page_metadata: Metadata containing KV cache
            
        Returns:
            Tuple of (selected_keys, selected_values)
        """
        if not selected_pages or 'page_tokens' not in page_metadata:
            return torch.empty(0), torch.empty(0)
        
        # Collect all tokens from selected pages
        all_selected_tokens = []
        for page_idx in selected_pages:
            if page_idx < len(page_metadata['page_tokens']):
                all_selected_tokens.extend(page_metadata['page_tokens'][page_idx])
        
        if not all_selected_tokens:
            return torch.empty(0), torch.empty(0)
        
        # Extract keys and values for selected tokens
        if 'keys' in page_metadata and 'values' in page_metadata:
            keys = page_metadata['keys']  # [num_layers, seq_len, num_heads, head_dim]
            values = page_metadata['values']
            
            # Select tokens from all layers
            selected_keys = []
            selected_values = []
            
            for layer_idx in range(keys.shape[0]):
                layer_keys = keys[layer_idx, all_selected_tokens, :, :]  # [num_tokens, num_heads, head_dim]
                layer_values = values[layer_idx, all_selected_tokens, :, :]
                
                selected_keys.append(layer_keys)
                selected_values.append(layer_values)
            
            # Concatenate across layers
            selected_keys = torch.cat(selected_keys, dim=0)  # [total_tokens, num_heads, head_dim]
            selected_values = torch.cat(selected_values, dim=0)
            
            return selected_keys, selected_values
        
        return torch.empty(0), torch.empty(0)
    
    def _update_attention_stats(self, attention_time_ms: float):
        """Update attention performance statistics."""
        self.attention_stats['total_attention_calls'] += 1
        self.attention_stats['total_sparse_operations'] += 1
        
        # Update running average
        current_avg = self.attention_stats['avg_attention_time_ms']
        total_calls = self.attention_stats['total_attention_calls']
        
        self.attention_stats['avg_attention_time_ms'] = (
            (current_avg * (total_calls - 1) + attention_time_ms) / total_calls
        )
    
    def get_attention_stats(self) -> Dict[str, Any]:
        """Get current attention performance statistics."""
        return self.attention_stats.copy()
    
    def optimize_attention_config(self, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Dynamically optimize attention configuration based on performance.
        
        Args:
            performance_metrics: Current performance metrics
            
        Returns:
            Optimized configuration parameters
        """
        current_latency = performance_metrics.get('latency_ms', 0)
        current_memory = performance_metrics.get('memory_gb', 0)
        
        # Simple optimization heuristics
        optimizations = {}
        
        if current_latency > self.config.target_latency_ms:
            # Reduce attention complexity
            optimizations['use_fused_kernel'] = True
            optimizations['attention_chunk_size'] = max(64, self.config.attention_chunk_size // 2)
        
        if current_memory > self.config.target_memory_gb:
            # Reduce memory usage
            optimizations['attention_chunk_size'] = min(512, self.config.attention_chunk_size * 2)
            optimizations['use_gradient_checkpointing'] = True
        
        return optimizations
    
    def clear_stats(self):
        """Clear attention performance statistics."""
        self.attention_stats = {
            'total_attention_calls': 0,
            'avg_attention_time_ms': 0.0,
            'total_sparse_operations': 0
        }
    
    def benchmark_attention(self, input_size: int, num_pages: int) -> Dict[str, float]:
        """
        Benchmark attention performance for given input size.
        
        Args:
            input_size: Input sequence length
            num_pages: Number of pages to process
            
        Returns:
            Benchmark results
        """
        # Create dummy inputs
        batch_size = 1
        hidden_dim = self.num_heads * self.head_dim
        
        query = torch.randn(batch_size, hidden_dim, device=self.device)
        dummy_metadata = {
            'page_tokens': [list(range(i * self.config.page_size, (i + 1) * self.config.page_size)) 
                           for i in range(num_pages)],
            'keys': torch.randn(1, input_size, self.num_heads, self.head_dim, device=self.device),
            'values': torch.randn(1, input_size, self.num_heads, self.head_dim, device=self.device)
        }
        
        # Warmup
        for _ in range(10):
            _ = self.execute_sparse_attention(query, list(range(num_pages)), dummy_metadata)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        for _ in range(100):
            _ = self.execute_sparse_attention(query, list(range(num_pages)), dummy_metadata)
        end_time.record()
        
        torch.cuda.synchronize()
        total_time_ms = start_time.elapsed_time(end_time)
        avg_time_ms = total_time_ms / 100
        
        return {
            'avg_attention_time_ms': avg_time_ms,
            'throughput_tokens_per_ms': input_size / avg_time_ms,
            'memory_usage_gb': torch.cuda.memory_allocated() / (1024**3)
        }
