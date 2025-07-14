from typing import Generator
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int, hid_dim: int, out_dim: int, depth: int,
        activation=nn.Tanh,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.depth = depth
        
        self.layers = nn.Sequential(*[
            nn.Linear(in_dim, hid_dim), activation(),
            *sum([
                [nn.Linear(hid_dim, hid_dim), activation()]
                for _ in range(depth)
            ], list()),
            nn.Linear(hid_dim, out_dim)
        ])
        self._initialize_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers.forward(x)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
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
    
    def __str__(self) -> str:
        arg_str = ', '.join([
            f'{name}={repr(val)}'
            for name, val in self.arguments()
        ])
        return f'{type(self).__name__}({arg_str})'


if __name__ == '__main__':
    model = MLP(4, 200, 3, 6)
    print(model)
    print(model.layers)
