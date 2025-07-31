from typing import Type, Any, Generator
from abc import ABC, abstractmethod
import torch
from torch import nn
from .mlp import MLPOutput, MLPBase, MLP


class Solver(nn.Module, ABC):
    def __init__(
        self,
        # specifying the model architecture
        in_dim: int = 4,
        hid_dim: int = 200, 
        out_dim: int = 3,
        depth: int = 6,
        activation: Type[nn.Module] = nn.Tanh,
        model_type: Type[MLPBase] = MLP,
        # configuring the environment
        ground_pos: float = 0.0,    # the ground will be at `y = ground_pos`
        gravity: float = 9.8,       # positive sign for downward
        up_index: int = 1,          # 0:x, 1:y, 2:z -> suppose `y` is the vertical axis`
    ):
        super().__init__()
        self.model = model_type(
            in_dim=in_dim, 
            hid_dim=hid_dim,
            out_dim=out_dim,
            depth=depth,
            activation=activation,
        )
        self._ground_pos = ground_pos
        self._gravity = gravity
        self._up_index = up_index
        
        self._config: dict[str, Any] = dict(
            in_dim=in_dim,
            hid_dim=hid_dim, 
            out_dim=out_dim,
            depth=depth,
            activation=activation,
            model_type=model_type,
            ground_pos=ground_pos, 
            gravity=gravity,
            up_index=up_index,
        )
        
    @property
    def ground_pos(self) -> float:
        return self._ground_pos
        
    @property
    def gravity(self) -> float:
        return self._gravity
        
    @property
    def up_index(self) -> int:
        return self._up_index
    
    @property
    def config(self) -> dict[str, Any]:
        return self._config
    
    def register_physical_property(self, name: str, value: float, optimized: bool):
        self._config[name] = value
        self._config[f'optim_{name}'] = optimized

    @abstractmethod
    def property_parameters(self) -> Generator[nn.Parameter, None, None]:
        ''' Returns the material properties parameters only. '''
        pass
    
    def network_parameters(self) -> Generator[nn.Parameter, None, None]:
        yield from self.model.parameters()

    def forward(self, x: torch.Tensor) -> MLPOutput:
        return self.model.forward(x)

    @abstractmethod
    def compute_loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pass