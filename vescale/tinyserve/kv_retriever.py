"""
Query-Aware KV Retriever for dynamic page selection based on query relevance.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import math


class QueryAwareKVRetriever:
    """
    Implements query-aware KV page selection using bounding-box metadata.
    
    Based on the paper's methodology section 3.4: Query-Aware Page Selection
    """
    
    def __init__(self, config):
        """
        Initialize the KV retriever.
        
        Args:
            config: Configuration containing page_size, selection_ratio, etc.
        """
        self.config = config
        self.page_size = config.page_size
        self.selection_ratio = config.selection_ratio
        self.device = torch.device(config.device)
        
        # Cache for storing computed relevance scores
        self.relevance_cache = {}
        
    def select_relevant_pages(self, query_vector: torch.Tensor, 
                            page_metadata: Dict[str, Any]) -> List[int]:
        """
        Select the most relevant KV pages based on query vector.
        
        Args:
            query_vector: Current query vector of shape [batch_size, hidden_dim]
            page_metadata: Metadata containing page bounds and information
            
        Returns:
            List of selected page indices
        """
        if not page_metadata['page_bounds']:
            return []
        
        # Calculate relevance scores for all pages
        relevance_scores = self._compute_relevance_scores(query_vector, page_metadata)
        
        # Select top-K pages based on selection ratio
        num_pages = len(page_metadata['page_bounds'])
        k = max(1, int(num_pages * self.selection_ratio))
        
        # Get top-K page indices
        selected_pages = self._select_top_k_pages(relevance_scores, k)
        
        return selected_pages
    
    def _compute_relevance_scores(self, query_vector: torch.Tensor, 
                                 page_metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Compute relevance scores using directional bounding-box estimator.
        
        This implements the relevance function from equation (2) in the paper:
        r(q_t, φ(K_j)) = Σᵢ (q_t,i · M_j,i if q_t,i ≥ 0 else q_t,i · m_j,i)
        """
        batch_size, hidden_dim = query_vector.shape
        num_pages = len(page_metadata['page_bounds'])
        
        # Initialize relevance scores
        relevance_scores = torch.zeros(batch_size, num_pages, device=self.device)
        
        for page_idx, page_bounds in enumerate(page_metadata['page_bounds']):
            min_bounds, max_bounds = page_bounds
            
            # Ensure bounds have correct shape
            if min_bounds.dim() == 1:
                min_bounds = min_bounds.unsqueeze(0).expand(batch_size, -1)
            if max_bounds.dim() == 1:
                max_bounds = max_bounds.unsqueeze(0).expand(batch_size, -1)
            
            # Compute relevance using directional bounding-box approach
            # For positive query components, use max bounds
            # For negative query components, use min bounds
            positive_mask = query_vector >= 0
            negative_mask = ~positive_mask
            
            relevance = torch.zeros_like(query_vector)
            relevance[positive_mask] = query_vector[positive_mask] * max_bounds[positive_mask]
            relevance[negative_mask] = query_vector[negative_mask] * min_bounds[negative_mask]
            
            # Sum across hidden dimensions
            relevance_scores[:, page_idx] = relevance.sum(dim=1)
        
        return relevance_scores
    
    def _select_top_k_pages(self, relevance_scores: torch.Tensor, k: int) -> List[int]:
        """
        Select top-K pages based on relevance scores.
        
        Args:
            relevance_scores: Relevance scores of shape [batch_size, num_pages]
            k: Number of pages to select
            
        Returns:
            List of selected page indices
        """
        # For simplicity, use the first batch element
        scores = relevance_scores[0] if relevance_scores.dim() > 1 else relevance_scores
        
        # Get top-k indices
        _, top_indices = torch.topk(scores, k=min(k, len(scores)), dim=0)
        
        return top_indices.tolist()
    
    def update_page_metadata(self, page_metadata: Dict[str, Any], 
                           new_kv_cache: Dict[str, torch.Tensor]):
        """
        Update page metadata when new KV cache is added.
        
        Args:
            page_metadata: Current page metadata
            new_kv_cache: New KV cache to add
        """
        # Extract key and value tensors
        keys = new_kv_cache['keys']  # Shape: [num_layers, seq_len, num_heads, head_dim]
        values = new_kv_cache['values']
        
        num_layers, seq_len, num_heads, head_dim = keys.shape
        
        # Calculate page boundaries
        num_pages = math.ceil(seq_len / self.page_size)
        
        for page_idx in range(num_pages):
            start_idx = page_idx * self.page_size
            end_idx = min(start_idx + self.page_size, seq_len)
            
            # Extract page keys and values
            page_keys = keys[:, start_idx:end_idx, :, :]
            page_values = values[:, start_idx:end_idx, :, :]
            
            # Compute bounding box metadata for this page
            # Min and max bounds across all dimensions
            min_bounds = page_keys.min(dim=(0, 1, 2)).values  # [head_dim]
            max_bounds = page_keys.max(dim=(0, 1, 2)).values  # [head_dim]
            
            # Store metadata
            page_metadata['page_bounds'].append((min_bounds, max_bounds))
            page_metadata['page_tokens'].append(list(range(start_idx, end_idx)))
            
            # Update the stored keys and values
            if len(page_metadata['keys']) == 0:
                page_metadata['keys'] = page_keys
                page_metadata['values'] = page_values
            else:
                page_metadata['keys'] = torch.cat([page_metadata['keys'], page_keys], dim=1)
                page_metadata['values'] = torch.cat([page_metadata['values'], page_values], dim=1)
    
    def get_page_statistics(self, page_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about the current page organization.
        
        Args:
            page_metadata: Current page metadata
            
        Returns:
            Dictionary containing page statistics
        """
        if not page_metadata['page_bounds']:
            return {
                'num_pages': 0,
                'total_tokens': 0,
                'avg_page_size': 0,
                'memory_usage_gb': 0.0
            }
        
        num_pages = len(page_metadata['page_bounds'])
        total_tokens = sum(len(tokens) for tokens in page_metadata['page_tokens'])
        avg_page_size = total_tokens / num_pages if num_pages > 0 else 0
        
        # Estimate memory usage (rough calculation)
        if 'keys' in page_metadata and page_metadata['keys'].numel() > 0:
            keys_memory = page_metadata['keys'].numel() * page_metadata['keys'].element_size()
            values_memory = page_metadata['values'].numel() * page_metadata['values'].element_size()
            total_memory = (keys_memory + values_memory) / (1024**3)  # Convert to GB
        else:
            total_memory = 0.0
        
        return {
            'num_pages': num_pages,
            'total_tokens': total_tokens,
            'avg_page_size': avg_page_size,
            'memory_usage_gb': total_memory
        }
    
    def clear_cache(self):
        """Clear the relevance score cache."""
        self.relevance_cache.clear()
    
    def optimize_page_size(self, current_performance: Dict[str, float]) -> int:
        """
        Dynamically optimize page size based on performance metrics.
        
        Args:
            current_performance: Current performance metrics
            
        Returns:
            Optimized page size
        """
        # Simple heuristic: if latency is high, reduce page size
        # if memory usage is high, increase page size
        current_latency = current_performance.get('latency_ms', 0)
        current_memory = current_performance.get('memory_gb', 0)
        
        if current_latency > self.config.target_latency_ms:
            # Reduce page size to improve latency
            new_page_size = max(4, self.page_size // 2)
        elif current_memory > self.config.target_memory_gb:
            # Increase page size to reduce memory overhead
            new_page_size = min(64, self.page_size * 2)
        else:
            new_page_size = self.page_size
        
        return new_page_size
