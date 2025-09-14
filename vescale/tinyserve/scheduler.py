"""
Modular Scheduling Pipeline for handling incoming queries and routing through configurable plugins.
"""

import torch
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from queue import Queue, Empty
import threading


@dataclass
class ScheduledRequest:
    """Represents a scheduled inference request."""
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    priority: int = 0
    timestamp: float = 0.0
    session_id: Optional[str] = None


class ModularScheduler:
    """
    Modular scheduling pipeline that handles incoming queries and routes them through configurable plugins.
    
    Based on the paper's description of the modular scheduling pipeline.
    """
    
    def __init__(self, config):
        """
        Initialize the modular scheduler.
        
        Args:
            config: Configuration containing scheduling parameters
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Request queues with different priorities
        self.high_priority_queue = Queue()
        self.normal_priority_queue = Queue()
        self.low_priority_queue = Queue()
        
        # Plugin registry
        self.plugins = {}
        self.plugin_order = []
        
        # Session management
        self.sessions = {}
        self.session_timeout = getattr(config, 'session_timeout', 300.0)  # 5 minutes
        
        # Performance tracking
        self.scheduler_stats = {
            'total_requests': 0,
            'processed_requests': 0,
            'avg_queue_time_ms': 0.0,
            'avg_processing_time_ms': 0.0,
            'active_sessions': 0
        }
        
        # Start background processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_requests, daemon=True)
        self.processing_thread.start()
    
    def register_plugin(self, name: str, plugin_func: Callable, priority: int = 0):
        """
        Register a plugin function for request processing.
        
        Args:
            name: Plugin name
            plugin_func: Plugin function to execute
            priority: Execution priority (lower = higher priority)
        """
        self.plugins[name] = {
            'function': plugin_func,
            'priority': priority,
            'enabled': True
        }
        
        # Update plugin execution order
        self.plugin_order = sorted(self.plugins.keys(), 
                                 key=lambda x: self.plugins[x]['priority'])
    
    def submit_request(self, request: ScheduledRequest) -> str:
        """
        Submit a request for processing.
        
        Args:
            request: Request to schedule
            
        Returns:
            Request ID
        """
        request.timestamp = time.time()
        
        # Route to appropriate queue based on priority
        if request.priority == 0:  # High priority
            self.high_priority_queue.put(request)
        elif request.priority == 1:  # Normal priority
            self.normal_priority_queue.put(request)
        else:  # Low priority
            self.low_priority_queue.put(request)
        
        self.scheduler_stats['total_requests'] += 1
        
        return request.request_id
    
    def _process_requests(self):
        """Background thread for processing requests."""
        while self.running:
            try:
                # Process high priority requests first
                request = self._get_next_request()
                if request:
                    self._execute_request(request)
                else:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    
            except Exception as e:
                print(f"Error in request processing: {e}")
                time.sleep(0.1)
    
    def _get_next_request(self) -> Optional[ScheduledRequest]:
        """Get the next request to process based on priority."""
        # Try high priority queue first
        try:
            return self.high_priority_queue.get_nowait()
        except Empty:
            pass
        
        # Try normal priority queue
        try:
            return self.normal_priority_queue.get_nowait()
        except Empty:
            pass
        
        # Try low priority queue
        try:
            return self.low_priority_queue.get_nowait()
        except Empty:
            pass
        
        return None
    
    def _execute_request(self, request: ScheduledRequest):
        """Execute a request through the plugin pipeline."""
        start_time = time.time()
        
        try:
            # Execute plugins in order
            result = request
            for plugin_name in self.plugin_order:
                plugin = self.plugins[plugin_name]
                if plugin['enabled']:
                    try:
                        result = plugin['function'](result)
                        if result is None:
                            # Plugin requested to stop processing
                            break
                    except Exception as e:
                        print(f"Plugin {plugin_name} error: {e}")
                        continue
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self._update_stats(processing_time)
            
            # Clean up session if needed
            if request.session_id:
                self._update_session(request.session_id, result)
                
        except Exception as e:
            print(f"Error executing request {request.request_id}: {e}")
    
    def _update_session(self, session_id: str, result: Any):
        """Update session information."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'created_at': time.time(),
                'last_activity': time.time(),
                'request_count': 0,
                'total_tokens': 0
            }
        
        session = self.sessions[session_id]
        session['last_activity'] = time.time()
        session['request_count'] += 1
        
        # Clean up expired sessions
        self._cleanup_expired_sessions()
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session['last_activity'] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
    
    def _update_stats(self, processing_time_ms: float):
        """Update scheduler statistics."""
        self.scheduler_stats['processed_requests'] += 1
        
        # Update running averages
        current_avg = self.scheduler_stats['avg_processing_time_ms']
        total_processed = self.scheduler_stats['processed_requests']
        
        self.scheduler_stats['avg_processing_time_ms'] = (
            (current_avg * (total_processed - 1) + processing_time_ms) / total_processed
        )
        
        # Update active sessions count
        self.scheduler_stats['active_sessions'] = len(self.sessions)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            'high_priority_queue_size': self.high_priority_queue.qsize(),
            'normal_priority_queue_size': self.normal_priority_queue.qsize(),
            'low_priority_queue_size': self.low_priority_queue.qsize(),
            'total_queued': (self.high_priority_queue.qsize() + 
                           self.normal_priority_queue.qsize() + 
                           self.low_priority_queue.qsize())
        }
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get current scheduler statistics."""
        return self.scheduler_stats.copy()
    
    def enable_plugin(self, plugin_name: str):
        """Enable a specific plugin."""
        if plugin_name in self.plugins:
            self.plugins[plugin_name]['enabled'] = True
    
    def disable_plugin(self, plugin_name: str):
        """Disable a specific plugin."""
        if plugin_name in self.plugins:
            self.plugins[plugin_name]['enabled'] = False
    
    def get_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all plugins."""
        return {
            name: {
                'enabled': plugin['enabled'],
                'priority': plugin['priority']
            }
            for name, plugin in self.plugins.items()
        }
    
    def stop(self):
        """Stop the scheduler and background processing."""
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
    
    def clear_queues(self):
        """Clear all request queues."""
        while not self.high_priority_queue.empty():
            self.high_priority_queue.get()
        while not self.normal_priority_queue.empty():
            self.normal_priority_queue.get()
        while not self.low_priority_queue.empty():
            self.low_priority_queue.get()
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session."""
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self.sessions.keys())
