from abc import ABC, abstractmethod
from typing import Generator, Optional, Literal
import torch
from torch import nn


class MLPBase(nn.Module, ABC):
    def __init__(
        self,
        in_dim: int, hid_dim: int, out_dim: int, depth: int,
        activation=nn.Tanh,
    ):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.depth = depth
        self.activation_type = activation
        
    @property
    def input_shape(self) -> Literal['flat', 'spatio-temporal']:
        return 'flat'
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def _make_stem_block(self, hid_dim: Optional[int]=None):
        if hid_dim is None:
            hid_dim = self.hid_dim
        return [
            nn.Linear(self.in_dim, hid_dim),
            self.activation_type(),
        ]
    
    def _make_hidden_blocks(self, depth: int, hid_dim: Optional[int]=None):
        if hid_dim is None:
            hid_dim = self.hid_dim
        return sum([
            [nn.Linear(hid_dim, hid_dim), self.activation_type()]
            for _ in range(depth)
        ], list())
    
    def _make_linear_head(self, hid_dim: Optional[int]=None):
        if hid_dim is None:
            hid_dim = self.hid_dim
        return nn.Linear(hid_dim, self.out_dim)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def arguments(self) -> Generator:
        yield ('in_dim', self.in_dim)
        yield ('hid_dim', self.hid_dim)
        yield ('out_dim', self.out_dim)
        yield ('depth', self.depth)
    
    @property
    def num_parameters(self):
        return sum(map(torch.Tensor.numel, self.parameters()))
    
    def __str__(self) -> str:
        arg_str = ', '.join([
            f'{name}={repr(val)}'
            for name, val in self.arguments()
        ])
        return f'{type(self).__name__}({arg_str})'


class MLP(MLPBase):
    def __init__(
        self,
        in_dim: int, hid_dim: int, out_dim: int, depth: int,
        activation=nn.Tanh,
    ):
        nn.Module.__init__(self)
        MLPBase.__init__(self, in_dim, hid_dim, out_dim, depth, activation)
        
        self.layers = nn.Sequential(
            *self._make_stem_block(),
            *self._make_hidden_blocks(self.depth),
            self._make_linear_head(),
        )
        self._initialize_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers.forward(x)


class GlobalMLP(MLPBase):
    def __init__(
        self,
        in_dim: int, hid_dim: int, out_dim: int, depth: int,
        activation=nn.Tanh,
    ):
        nn.Module.__init__(self)
        MLPBase.__init__(self, in_dim, hid_dim, out_dim, depth, activation)

        self.layer_input = nn.Sequential(*self._make_stem_block())
        self.layer_hidden_0 = nn.Sequential(*self._make_hidden_blocks(self.depth - 2))
        self.layer_hidden_1 = nn.Sequential(*self._make_hidden_blocks(1))
        self.layer_hidden_2 = nn.Sequential(*self._make_hidden_blocks(1))
        self.layer_output = self._make_linear_head()
        
        self._initialize_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_i = self.layer_input.forward(x)
        x_h0 = self.layer_hidden_0.forward(x_i)
        x_h1 = self.layer_hidden_1.forward(x_h0) + x_h0
        x_h2 = self.layer_hidden_2.forward(x_h1)
        x_o = self.layer_output.forward(x_h2)
        return x_o
        

if __name__ == '__main__':
    sample_x = torch.randn(10, 4)
    target_shape = (10, 3)

    for mlp_type in (MLP, GlobalMLP):
        model = mlp_type(4, 200, 3, 6)
        assert model.forward(sample_x).shape == target_shape
        print(mlp_type, 'passed')
