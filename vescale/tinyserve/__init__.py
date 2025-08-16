"""
TinyServe: Query-Aware Cache Selection for Efficient LLM Serving

A lightweight and extensible runtime system for deploying tiny LLMs with support for:
- Structured KV sparsity
- Plugin-based token selection  
- Hardware-efficient attention kernels
- Query-aware page selection mechanism
"""

from .core import TinyServe
from .kv_retriever import QueryAwareKVRetriever
from .scheduler import ModularScheduler
from .attention import SparseAttentionExecutor
from .plugins import PluginManager
from .utils import TinyServeConfig

__version__ = "0.1.0"
__all__ = [
    "TinyServe",
    "QueryAwareKVRetriever", 
    "ModularScheduler",
    "SparseAttentionExecutor",
    "PluginManager",
    "TinyServeConfig"
]
