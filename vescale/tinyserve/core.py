"""
Core TinyServe implementation for efficient LLM inference serving.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .kv_retriever import QueryAwareKVRetriever
from .scheduler import ModularScheduler
from .attention import SparseAttentionExecutor
from .plugins import PluginManager
from .utils import TinyServeConfig


@dataclass
class TinyServeRequest:
    """Represents a single inference request."""
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    request_id: Optional[str] = None


@dataclass
class TinyServeResponse:
    """Represents the response from TinyServe."""
    generated_text: str
    tokens: List[int]
    latency_ms: float
    memory_usage_gb: float
    kv_cache_hit_rate: float
    request_id: Optional[str] = None


class TinyServe:
    """
    Main TinyServe class implementing query-aware cache selection for efficient LLM serving.
    
    Based on the paper: "TinyServe: Query-Aware Cache Selection for Efficient LLM Serving"
    """
    
    def __init__(self, config: TinyServeConfig):
        """
        Initialize TinyServe with configuration.
        
        Args:
            config: TinyServe configuration parameters
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize core components
        self.kv_retriever = QueryAwareKVRetriever(config)
        self.scheduler = ModularScheduler(config)
        self.attention_executor = SparseAttentionExecutor(config)
        self.plugin_manager = PluginManager(config)
        
        # KV cache management
        self.kv_cache = {}
        self.page_metadata = {}
        self.session_manager = {}
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'avg_latency_ms': 0.0,
            'avg_memory_gb': 0.0,
            'kv_hit_rate': 0.0
        }
    
    def load_model(self, model_path: str, model_type: str = "auto"):
        """
        Load a tiny LLM model for serving.
        
        Args:
            model_path: Path to model checkpoint or HuggingFace model name
            model_type: Type of model (e.g., "tinylama", "gpt2", "opt")
        """
        # Load model based on type
        if model_type == "tinylama":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif model_type == "gpt2":
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        else:
            # Auto-detect
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize KV cache structure
        self._init_kv_cache()
    
    def _init_kv_cache(self):
        """Initialize the KV cache with page-based structure."""
        hidden_size = self.model.config.hidden_size
        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        
        # Calculate page size based on config
        page_size = self.config.page_size
        
        # Initialize page metadata storage
        self.page_metadata = {
            'keys': torch.zeros((num_layers, 0, num_heads, hidden_size // num_heads), 
                              device=self.device, dtype=torch.float16),
            'values': torch.zeros((num_layers, 0, num_heads, hidden_size // num_heads), 
                                device=self.device, dtype=torch.float16),
            'page_bounds': [],  # Store min/max bounds for each page
            'page_tokens': []   # Store token indices for each page
        }
    
    def serve(self, request: TinyServeRequest) -> TinyServeResponse:
        """
        Main serving function implementing query-aware cache selection.
        
        Args:
            request: Inference request
            
        Returns:
            Generated response with performance metrics
        """
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        # Tokenize input
        input_ids = self.tokenizer.encode(request.prompt, return_tensors="pt").to(self.device)
        
        # Prefill stage - process all prompt tokens
        kv_cache = self._prefill_stage(input_ids)
        
        # Decode stage - generate tokens one by one with query-aware selection
        generated_tokens = self._decode_stage(input_ids, kv_cache, request.max_tokens)
        
        # Decode final text
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        end_time.record()
        torch.cuda.synchronize()
        latency_ms = start_time.elapsed_time(end_time)
        
        # Calculate memory usage and KV hit rate
        memory_usage = self._get_memory_usage()
        kv_hit_rate = self._calculate_kv_hit_rate()
        
        # Update statistics
        self._update_stats(latency_ms, memory_usage, kv_hit_rate, len(generated_tokens))
        
        return TinyServeResponse(
            generated_text=generated_text,
            tokens=generated_tokens.tolist(),
            latency_ms=latency_ms,
            memory_usage_gb=memory_usage,
            kv_cache_hit_rate=kv_hit_rate,
            request_id=request.request_id
        )
    
    def _prefill_stage(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Prefill stage: process all prompt tokens and store in KV cache."""
        # Forward pass through model to get KV cache
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            kv_cache = outputs.past_key_values
        
        # Store KV cache with page-based organization
        self._store_kv_cache(kv_cache, input_ids.shape[1])
        
        return kv_cache
    
    def _decode_stage(self, input_ids: torch.Tensor, kv_cache: Dict, max_tokens: int) -> torch.Tensor:
        """Decode stage: generate tokens with query-aware KV selection."""
        generated_tokens = []
        current_input = input_ids[:, -1:]  # Start with last token
        
        for _ in range(max_tokens):
            # Generate query vector for current token
            with torch.no_grad():
                query_output = self.model(current_input, use_cache=False)
                query_vector = query_output.logits[:, -1, :]
            
            # Query-aware KV page selection
            selected_pages = self.kv_retriever.select_relevant_pages(
                query_vector, self.page_metadata
            )
            
            # Execute sparse attention over selected pages
            attention_output = self.attention_executor.execute_sparse_attention(
                query_vector, selected_pages, self.page_metadata
            )
            
            # Generate next token
            next_token = self._sample_next_token(attention_output, request.temperature, request.top_p)
            generated_tokens.append(next_token.item())
            
            # Update input for next iteration
            current_input = torch.cat([current_input, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Check for early stopping via plugins
            if self.plugin_manager.should_stop_early(generated_tokens, attention_output):
                break
        
        return torch.tensor(generated_tokens, device=self.device)
    
    def _store_kv_cache(self, kv_cache: Dict, num_tokens: int):
        """Store KV cache with page-based organization and metadata."""
        # Implementation for storing KV cache in pages
        # This would include the bounding-box metadata calculation
        pass
    
    def _sample_next_token(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        """Sample next token using temperature and top-p sampling."""
        if temperature > 0:
            logits = logits / temperature
        
        # Apply top-p sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def _calculate_kv_hit_rate(self) -> float:
        """Calculate KV cache hit rate."""
        # Implementation for calculating cache hit rate
        return 0.95  # Placeholder
    
    def _update_stats(self, latency_ms: float, memory_gb: float, kv_hit_rate: float, num_tokens: int):
        """Update performance statistics."""
        self.stats['total_requests'] += 1
        self.stats['total_tokens'] += num_tokens
        
        # Update running averages
        self.stats['avg_latency_ms'] = (
            (self.stats['avg_latency_ms'] * (self.stats['total_requests'] - 1) + latency_ms) / 
            self.stats['total_requests']
        )
        self.stats['avg_memory_gb'] = (
            (self.stats['avg_memory_gb'] * (self.stats['total_requests'] - 1) + memory_gb) / 
            self.stats['total_requests']
        )
        self.stats['kv_hit_rate'] = kv_hit_rate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear KV cache and reset metadata."""
        self.kv_cache.clear()
        self.page_metadata.clear()
        self.session_manager.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
