"""
Basic tests for TinyServe functionality.
"""

import unittest
import torch
from unittest.mock import Mock, patch

# Import TinyServe components
from vescale.tinyserve import (
    TinyServeConfig,
    TinyServeRequest,
    TinyServeResponse,
    QueryAwareKVRetriever,
    SparseAttentionExecutor,
    PluginManager
)


class TestTinyServeConfig(unittest.TestCase):
    """Test TinyServe configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = TinyServeConfig()
        
        self.assertEqual(config.page_size, 16)
        self.assertEqual(config.selection_ratio, 0.3)
        self.assertEqual(config.num_attention_heads, 12)
        self.assertEqual(config.head_dim, 64)
        self.assertTrue(config.use_fused_kernel)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid page size
        with self.assertRaises(ValueError):
            TinyServeConfig(page_size=0)
        
        # Test invalid selection ratio
        with self.assertRaises(ValueError):
            TinyServeConfig(selection_ratio=1.5)
        
        # Test invalid entropy threshold
        with self.assertRaises(ValueError):
            TinyServeConfig(entropy_threshold=-0.1)
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = TinyServeConfig(
            page_size=32,
            selection_ratio=0.2,
            target_latency_ms=100.0
        )
        
        config_dict = config.to_dict()
        self.assertEqual(config_dict['page_size'], 32)
        self.assertEqual(config_dict['selection_ratio'], 0.2)
        self.assertEqual(config_dict['target_latency_ms'], 100.0)
        
        # Test reconstruction
        new_config = TinyServeConfig.from_dict(config_dict)
        self.assertEqual(new_config.page_size, 32)
        self.assertEqual(new_config.selection_ratio, 0.2)


class TestTinyServeRequest(unittest.TestCase):
    """Test TinyServe request."""
    
    def test_request_creation(self):
        """Test request creation."""
        request = TinyServeRequest(
            prompt="Test prompt",
            max_tokens=100,
            temperature=0.8,
            top_p=0.9,
            request_id="test-123"
        )
        
        self.assertEqual(request.prompt, "Test prompt")
        self.assertEqual(request.max_tokens, 100)
        self.assertEqual(request.temperature, 0.8)
        self.assertEqual(request.top_p, 0.9)
        self.assertEqual(request.request_id, "test-123")
    
    def test_request_defaults(self):
        """Test request default values."""
        request = TinyServeRequest(prompt="Test")
        
        self.assertEqual(request.max_tokens, 512)
        self.assertEqual(request.temperature, 0.7)
        self.assertEqual(request.top_p, 0.9)
        self.assertIsNone(request.request_id)


class TestTinyServeResponse(unittest.TestCase):
    """Test TinyServe response."""
    
    def test_response_creation(self):
        """Test response creation."""
        response = TinyServeResponse(
            generated_text="Generated text",
            tokens=[1, 2, 3, 4],
            latency_ms=50.0,
            memory_usage_gb=2.5,
            kv_cache_hit_rate=0.95,
            request_id="test-123"
        )
        
        self.assertEqual(response.generated_text, "Generated text")
        self.assertEqual(response.tokens, [1, 2, 3, 4])
        self.assertEqual(response.latency_ms, 50.0)
        self.assertEqual(response.memory_usage_gb, 2.5)
        self.assertEqual(response.kv_cache_hit_rate, 0.95)
        self.assertEqual(response.request_id, "test-123")


class TestQueryAwareKVRetriever(unittest.TestCase):
    """Test QueryAwareKVRetriever."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TinyServeConfig()
        self.retriever = QueryAwareKVRetriever(self.config)
    
    def test_initialization(self):
        """Test retriever initialization."""
        self.assertEqual(self.retriever.page_size, 16)
        self.assertEqual(self.retriever.selection_ratio, 0.3)
    
    def test_select_relevant_pages_empty(self):
        """Test page selection with empty metadata."""
        query = torch.randn(1, 768)
        metadata = {'page_bounds': []}
        
        selected = self.retriever.select_relevant_pages(query, metadata)
        self.assertEqual(selected, [])
    
    def test_page_statistics_empty(self):
        """Test page statistics with empty metadata."""
        metadata = {'page_bounds': []}
        stats = self.retriever.get_page_statistics(metadata)
        
        self.assertEqual(stats['num_pages'], 0)
        self.assertEqual(stats['total_tokens'], 0)
        self.assertEqual(stats['memory_usage_gb'], 0.0)


class TestSparseAttentionExecutor(unittest.TestCase):
    """Test SparseAttentionExecutor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TinyServeConfig()
        self.executor = SparseAttentionExecutor(self.config)
    
    def test_initialization(self):
        """Test executor initialization."""
        self.assertEqual(self.executor.num_heads, 12)
        self.assertEqual(self.executor.head_dim, 64)
        self.assertTrue(self.executor.use_fused_kernel)
    
    def test_execute_sparse_attention_empty(self):
        """Test sparse attention with empty pages."""
        query = torch.randn(1, 768)
        selected_pages = []
        metadata = {}
        
        output = self.executor.execute_sparse_attention(query, selected_pages, metadata)
        self.assertTrue(torch.allclose(output, torch.zeros_like(query)))
    
    def test_attention_stats(self):
        """Test attention statistics."""
        stats = self.executor.get_attention_stats()
        
        self.assertEqual(stats['total_attention_calls'], 0)
        self.assertEqual(stats['avg_attention_time_ms'], 0.0)
        self.assertEqual(stats['total_sparse_operations'], 0)


class TestPluginManager(unittest.TestCase):
    """Test PluginManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TinyServeConfig()
        self.plugin_manager = PluginManager(self.config)
    
    def test_initialization(self):
        """Test plugin manager initialization."""
        self.assertIn('entropy_early_exit', self.plugin_manager.plugins)
        self.assertIn('token_pruning', self.plugin_manager.plugins)
        self.assertIn('approximate_attention', self.plugin_manager.plugins)
        self.assertIn('cache_optimization', self.plugin_manager.plugins)
    
    def test_plugin_registration(self):
        """Test plugin registration."""
        def test_plugin(context):
            return context
        
        self.plugin_manager.register_plugin('test_plugin', test_plugin)
        self.assertIn('test_plugin', self.plugin_manager.plugins)
    
    def test_plugin_enable_disable(self):
        """Test plugin enable/disable."""
        self.plugin_manager.disable_plugin('entropy_early_exit')
        self.assertNotIn('entropy_early_exit', self.plugin_manager.enabled_plugins)
        
        self.plugin_manager.enable_plugin('entropy_early_exit')
        self.assertIn('entropy_early_exit', self.plugin_manager.enabled_plugins)
    
    def test_plugin_stats(self):
        """Test plugin statistics."""
        stats = self.plugin_manager.get_plugin_stats()
        
        self.assertEqual(stats['total_plugin_calls'], 0)
        self.assertEqual(stats['plugin_success_count'], 0)
        self.assertEqual(stats['early_exit_count'], 0)
        self.assertEqual(stats['pruning_count'], 0)


class TestTinyServeIntegration(unittest.TestCase):
    """Test TinyServe integration."""
    
    @patch('vescale.tinyserve.core.AutoModelForCausalLM')
    @patch('vescale.tinyserve.core.AutoTokenizer')
    def test_tinyserve_initialization(self, mock_tokenizer, mock_model):
        """Test TinyServe initialization."""
        from vescale.tinyserve import TinyServe
        
        config = TinyServeConfig()
        tinyserve = TinyServe(config)
        
        self.assertIsNotNone(tinyserve.kv_retriever)
        self.assertIsNotNone(tinyserve.scheduler)
        self.assertIsNotNone(tinyserve.attention_executor)
        self.assertIsNotNone(tinyserve.plugin_manager)
    
    def test_config_optimization(self):
        """Test configuration optimization."""
        config = TinyServeConfig()
        
        performance_metrics = {
            'latency_ms': 75.0,
            'memory_gb': 6.0
        }
        
        optimized = config.get_optimized_config(performance_metrics)
        
        # Should optimize based on performance
        self.assertIsInstance(optimized, TinyServeConfig)
        self.assertNotEqual(config.page_size, optimized.page_size)


if __name__ == '__main__':
    unittest.main()
