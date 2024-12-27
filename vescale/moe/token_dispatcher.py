from typing import Dict, List, Optional, Tuple
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from vescale import DeviceMesh

class TokenDispatcher(ABC):

    @abstractmethod
    def __init__(self, exp_config=None, env_config=None):
        pass

    @abstractmethod
    def assign_task(self, layer_id, token_id, expert_id, hidden_state, token_weight):
        pass

    @abstractmethod
    def set_experts_alloc(self, experts_alloc):
        pass

    @abstractmethod
    def collect_performance(self, perf, iter=-1):
        pass

    @abstractmethod
    def dispatch_token(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class BasicTokenDispatcher(TokenDispatcher):

    def __init__(self, exp_config=None, env_config=None):
        self.experts_alloc: Optional[List[Optional[DeviceMesh]]] = None
        self.expert_id: Optional[torch.Tensor] = None
        self.num_replicate: Optional[torch.Tensor] = None

    def assign_task(self, layer_id, token_id, expert_id, hidden_state, token_weight):
        self.expert_id = expert_id

    def set_experts_alloc(self, experts_alloc_info: Dict) -> None:
        self.experts_alloc = experts_alloc_info["experts_alloc"]
        self.num_replicate = experts_alloc_info["dp_size"]

    def collect_performance(self, perf, iter=-1):
        pass

    def dispatch_token(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        num_replicate = self.num_replicate[self.expert_id]
        replicate_id = torch.randint_like(num_replicate, 65535) % num_replicate
        return self.expert_id, replicate_id
